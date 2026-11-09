"""Baselines on a fixed training set.

    python -m mfb.train --data data/processed --b0                      # VLM alone + gap spread (room to adapt)
    python -m mfb.train --data data/processed --mode all                # B1: all training geometries
    python -m mfb.train --data data/processed --mode random --n-geoms 8 16 32 64 128 --seeds 0 1 2   # B2
    python -m mfb.train --data data/processed --mode sobol  --n-geoms 8 16 32 64 128                  # B3

Results are appended to results/baselines.jsonl, one JSON line per run, with RANS CPU-hours spent.
"""
import argparse, json, pathlib, random, time
import numpy as np
from .data import read_index, load_graphs, normaliser
from .core import Trainer, vlm_only, DEFAULT_CFG


def pick_geoms(train_geoms, rank, mode, n, seed):
    if mode == "all":
        return list(train_geoms)
    if mode == "random":
        g = list(train_geoms); random.Random(seed).shuffle(g); return g[:n]
    if mode == "sobol":   # the paper's nested design: first n geometries in design order
        return sorted(train_geoms, key=lambda x: rank[x])[:n]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--out", default="results/baselines.jsonl")
    ap.add_argument("--b0", action="store_true")
    ap.add_argument("--mode", choices=["all", "random", "sobol"], default="all")
    ap.add_argument("--n-geoms", type=int, nargs="*", default=[0]); ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"]); ap.add_argument("--hidden", type=int, default=DEFAULT_CFG["hidden"])
    ap.add_argument("--aoas", type=float, nargs="*", default=None,
                    help="use only these angles, e.g. 11 13 15 17 19 (the paper's Sec. 4 setting)")
    a = ap.parse_args()

    rows = read_index(a.data)
    if a.aoas:
        rows = [r for r in rows if r["aoa_deg"] in set(a.aoas)]
    train_rows = [r for r in rows if r["split"] == "train"]; test_rows = [r for r in rows if r["split"] == "test"]
    test = load_graphs(a.data, test_rows)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    if a.b0:
        res = vlm_only(test)
        all_train = load_graphs(a.data, train_rows)
        gap = np.sort(vlm_only(all_train)["per_snapshot_mse"])[::-1]
        top20 = gap[: max(1, len(gap) // 5)].sum() / gap.sum()
        out = dict(baseline="B0_vlm_only", aoas=a.aoas, n_test=len(test), **{k: v for k, v in res.items() if k != "per_snapshot_mse"},
                   share_of_gap_in_worst_20pct=float(top20))
        print(json.dumps(out)); open(a.out, "a").write(json.dumps(out) + "\n"); return

    rank = {r["geom"]: r["sobol_rank"] for r in train_rows}
    cost = {}
    for r in train_rows: cost[r["geom"]] = cost.get(r["geom"], 0) + r["cost_cpuh"]
    by_geom = {}
    for r in train_rows: by_geom.setdefault(r["geom"], []).append(r)

    cfg = dict(DEFAULT_CFG, epochs=a.epochs, hidden=a.hidden)
    for n in a.n_geoms:
        for seed in a.seeds:
            geoms = pick_geoms(by_geom, rank, a.mode, n, seed)
            sel = [r for g in geoms for r in by_geom[g]]
            tr = load_graphs(a.data, sel)
            t0 = time.time()
            T = Trainer(tr[0].x.shape[1], normaliser(tr), cfg); T.fit(tr, cfg["epochs"], seed)
            out = dict(baseline={"all": "B1", "random": "B2", "sobol": "B3"}[a.mode], mode=a.mode, n_geoms=len(geoms),
                       n_snapshots=len(sel), n_test=len(test), aoas=a.aoas, epochs=cfg["epochs"], seed=seed, rans_cpuh=sum(cost[g] for g in geoms),
                       train_seconds=round(time.time() - t0, 1), **T.evaluate(test))
            print(json.dumps(out)); open(a.out, "a").write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
