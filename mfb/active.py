"""Adaptive loop (retrospective): every RANS result exists already; 'buying' a geometry reveals its stored
results and charges its CPU-hours. Strategies are compared at equal CPU-hours on the same test wings.

    python -m mfb.active --data data/processed --strategy uncertainty_diverse --budget-geoms 64 --seeds 0 1 2
    strategies: random | sobol | uncertainty | uncertainty_diverse

Buys whole geometries (all 9 angles), the way CFD is actually run. Logs test error after every round.
"""
import argparse, json, pathlib, random
import numpy as np
from .data import read_index, load_graphs, normaliser
from .core import Trainer, DEFAULT_CFG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed"); ap.add_argument("--out", default="results/active.jsonl")
    ap.add_argument("--strategy", default="uncertainty_diverse",
                    choices=["random", "sobol", "uncertainty", "uncertainty_diverse"])
    ap.add_argument("--init-geoms", type=int, default=8); ap.add_argument("--batch-geoms", type=int, default=4)
    ap.add_argument("--budget-geoms", type=int, default=64, help="stop after this many geometries (CPU-hours logged too)")
    ap.add_argument("--min-dist", type=float, default=0.15, help="diversity rule: min distance in scaled design space")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0]); ap.add_argument("--epochs", type=int, default=DEFAULT_CFG["epochs"])
    ap.add_argument("--finetune-epochs", type=int, default=DEFAULT_CFG["finetune_epochs"])
    a = ap.parse_args()

    rows = read_index(a.data)
    train_rows = [r for r in rows if r["split"] == "train"]
    test = load_graphs(a.data, [r for r in rows if r["split"] == "test"])
    by_geom = {}
    for r in train_rows: by_geom.setdefault(r["geom"], []).append(r)
    graphs = {g: load_graphs(a.data, rs) for g, rs in by_geom.items()}   # VLM inputs visible for all; y only used once bought
    cost = {g: sum(r["cost_cpuh"] for r in rs) for g, rs in by_geom.items()}
    rank = {g: rs[0]["sobol_rank"] for g, rs in by_geom.items()}
    dkeys = [k for k in train_rows[0] if k.startswith("d")]
    D = {g: np.array([float(rs[0][k]) for k in dkeys]) for g, rs in by_geom.items()}
    lo = np.min(list(D.values()), 0); hi = np.max(list(D.values()), 0)
    D = {g: (v - lo) / np.where(hi > lo, hi - lo, 1) for g, v in D.items()}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(DEFAULT_CFG, epochs=a.epochs, finetune_epochs=a.finetune_epochs)

    for seed in a.seeds:
        order = sorted(by_geom, key=lambda g: rank[g])
        bought = order[: a.init_geoms]   # same start for every strategy: first Sobol geometries
        rnd = random.Random(seed); T = None; rnd_round = 0
        while True:
            tr = [gr for g in bought for gr in graphs[g]]
            if T is None:
                T = Trainer(tr[0].x.shape[1], normaliser(tr), cfg); T.fit(tr, cfg["epochs"], seed)
            else:   # warm start: fine-tune on everything bought so far, no restart
                T.fit(tr, cfg["finetune_epochs"], seed + rnd_round)
            log = dict(strategy=a.strategy, seed=seed, round=rnd_round, n_geoms=len(bought),
                       rans_cpuh=sum(cost[g] for g in bought), **T.evaluate(test))
            print(json.dumps(log)); open(a.out, "a").write(json.dumps(log) + "\n")
            pool = [g for g in by_geom if g not in bought]
            if len(bought) >= a.budget_geoms or not pool:
                break
            k = min(a.batch_geoms, a.budget_geoms - len(bought), len(pool))
            if a.strategy == "random":
                new = rnd.sample(pool, k)
            elif a.strategy == "sobol":
                new = [g for g in order if g in pool][:k]
            else:
                score = {g: sum(T.uncertainty(gr) for gr in graphs[g]) / cost[g] for g in pool}
                ranked = sorted(pool, key=score.get, reverse=True)
                if a.strategy == "uncertainty":
                    new = ranked[:k]
                else:   # skip candidates too close in design space to one already picked this round
                    new = []
                    for g in ranked:
                        if all(np.linalg.norm(D[g] - D[h]) >= a.min_dist for h in new):
                            new.append(g)
                        if len(new) == k:
                            break
            bought += new; rnd_round += 1


if __name__ == "__main__":
    main()
