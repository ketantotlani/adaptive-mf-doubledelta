"""Fake dataset with the real shapes (30x32 lattice, 9 angles, 6 design variables) for smoke tests ONLY.
The 'VLM' and 'RANS' fields are made-up functions; nothing here says anything about real aerodynamics.

    python -m mfb.synthetic --out data/synthetic --geoms 40 --test 8
"""
import argparse, pathlib
import numpy as np
from .data import knn_edges, write_index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic"); ap.add_argument("--geoms", type=int, default=40)
    ap.add_argument("--test", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out); (out / "snap").mkdir(parents=True, exist_ok=True)
    nc, ns = 30, 32; rows = []
    for g in range(a.geoms):
        d = rng.uniform(0, 1, 6)
        span = 1.0 + 0.3 * d[5]; sweep = 0.5 + 0.5 * d[1]
        u, v = np.meshgrid(np.linspace(0, 1, nc), np.linspace(0, 1, ns), indexing="ij")
        xs = (u * (1 - 0.6 * v) + sweep * v).ravel(); ys = (span * v).ravel(); zs = (0.02 * d[0] * np.sin(np.pi * u)).ravel()
        pos = np.stack([xs, ys, zs], 1).astype(np.float32)
        area = np.full(len(pos), (1 - 0.3) * span / (nc * ns), np.float32)
        ei, ea = knn_edges(pos)
        for aoa in range(11, 20):
            a_r = np.radians(aoa)
            vlm = 2 * np.pi * a_r * (1 - u.ravel()) ** 1.5 * (1 + 0.2 * d[2])
            nonlin = (a_r ** 2) * (4 + 6 * d[3]) * np.exp(-((v.ravel() - 0.6 - 0.2 * d[4]) ** 2) / 0.02) * (aoa > 14 + 3 * d[3])
            rans = vlm + nonlin + 0.02 * rng.standard_normal(len(pos))
            x = np.concatenate([pos, area[:, None] * 1e3, np.repeat(d[None], len(pos), 0),
                                np.full((len(pos), 1), np.sin(a_r)), np.full((len(pos), 1), np.cos(a_r)), vlm[:, None]], 1)
            sid = f"g{g:03d}_a{aoa:02d}"
            np.savez(out / "snap" / f"{sid}.npz", x=x.astype(np.float32), y=rans.astype(np.float32), pos=pos, area=area,
                     edge_index=ei, edge_attr=ea)
            rows.append(dict(sid=sid, geom=f"g{g:03d}", aoa_deg=aoa, split="test" if g >= a.geoms - a.test else "train",
                             sobol_rank=g, cost_cpuh=127.0, **{f"d{i+1}": round(float(d[i]), 4) for i in range(6)}))
    write_index(out, rows); print(f"wrote {len(rows)} synthetic snapshots to {out}")


if __name__ == "__main__":
    main()
