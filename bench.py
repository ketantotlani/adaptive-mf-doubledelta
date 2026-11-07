"""How long would the baselines and the adaptive loop take on this machine?

    python bench.py                       # uses the processed test wings already on the drive

Times real training steps of the baseline GNN on the processed snapshots (any split: timing only, no results),
then scales up to the full dataset. Training-set size and loop settings are the planned ones; change below.
Results are printed and appended to results/bench.jsonl.
"""
import argparse, json, pathlib, platform, time
import torch
from mfb.data import read_index, load_graphs, normaliser
from mfb.core import Trainer, DEFAULT_CFG

FULL_TRAIN = 256 * 9          # 2,304 training snapshots
LOOP = dict(strategies=4, seeds=3, rounds=10, finetune_epochs=DEFAULT_CFG["finetune_epochs"])


def sync(dev):
    if dev == "cuda": torch.cuda.synchronize()
    elif dev == "mps": torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/Volumes/CRUCIAL_X6/double-delta/processed")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--device", default=None)
    ap.add_argument("--hidden", type=int, nargs="*", default=[64, 128])
    a = ap.parse_args()

    rows = read_index(a.data); G = load_graphs(a.data, rows)
    print(f"{len(G)} snapshots loaded, {G[0].num_nodes} panels each")
    out = []
    for hidden in a.hidden:
        cfg = dict(DEFAULT_CFG, hidden=hidden)
        T = Trainer(G[0].x.shape[1], normaliser(G), cfg, device=a.device)
        T.fit(G[:32], 1)                                  # warm-up (kernel compilation etc.)
        sync(T.device); t0 = time.time()
        T.fit(G, a.epochs)
        sync(T.device); per_graph_epoch = (time.time() - t0) / (a.epochs * len(G))

        b1_min = per_graph_epoch * FULL_TRAIN * cfg["epochs"] / 60
        # adaptive loop: each round fine-tunes on the set bought so far (on average ~half the training set)
        loop_h = (per_graph_epoch * (FULL_TRAIN / 2) * LOOP["finetune_epochs"] * LOOP["rounds"]
                  * LOOP["strategies"] * LOOP["seeds"]) / 3600
        b2b3_h = per_graph_epoch * sum(n * 9 for n in (8, 16, 32, 64, 128)) * cfg["epochs"] * (3 + 1) / 3600
        r = dict(machine=platform.platform(), device=T.device, hidden=hidden,
                 params=sum(p.numel() for p in T.model.parameters()),
                 ms_per_snapshot_epoch=round(per_graph_epoch * 1000, 2),
                 B1_full_200ep_min=round(b1_min, 1), B2_B3_sweeps_h=round(b2b3_h, 2),
                 adaptive_loop_all_strategies_h=round(loop_h, 2))
        out.append(r); print(json.dumps(r))

    pathlib.Path("results").mkdir(exist_ok=True)
    with open("results/bench.jsonl", "a") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    r = out[0]
    print(f"\nOn this machine ({r['device']}), hidden={r['hidden']}:")
    print(f"  B1, all 256 wings, 200 epochs : ~{r['B1_full_200ep_min']} min")
    print(f"  B2 + B3 size sweeps          : ~{r['B2_B3_sweeps_h']} h")
    print(f"  adaptive loop, 4 strategies x 3 seeds x 10 rounds : ~{r['adaptive_loop_all_strategies_h']} h")


if __name__ == "__main__":
    main()
