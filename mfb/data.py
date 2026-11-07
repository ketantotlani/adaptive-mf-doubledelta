"""Processed-data format shared by every script.

data/processed/
  index.csv        one row per snapshot:
                   sid, geom, aoa_deg, split (train|test), sobol_rank, cost_cpuh, d1..d6 (design variables)
  snap/<sid>.npz   x (N,F) node features   -- must include the VLM pressure as one column
                   y (N,)  target: RANS pressure mapped onto the same N lattice panels
                   pos (N,3) panel centres, area (N,) panel areas
                   edge_index (2,E), edge_attr (E,4) = (dx, dy, dz, dist)

preprocess.py writes this from the raw files; synthetic.py writes a fake copy for smoke tests.
"""
import csv, pathlib
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_geometric.data import Data


def knn_edges(pos, k=8):
    """Symmetric k-nearest-neighbour graph over panel centres, with relative position + distance."""
    tree = cKDTree(pos)
    _, nb = tree.query(pos, k=k + 1)
    src = np.repeat(np.arange(len(pos)), k)
    dst = nb[:, 1:].reshape(-1)
    ei = np.unique(np.concatenate([np.stack([src, dst]), np.stack([dst, src])], 1), axis=1)
    rel = pos[ei[1]] - pos[ei[0]]
    attr = np.concatenate([rel, np.linalg.norm(rel, axis=1, keepdims=True)], 1)
    return ei.astype(np.int64), attr.astype(np.float32)


def read_index(root):
    with open(pathlib.Path(root) / "index.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["aoa_deg"] = float(r["aoa_deg"]); r["cost_cpuh"] = float(r["cost_cpuh"])
        r["sobol_rank"] = int(r["sobol_rank"])
    return rows


def write_index(root, rows):
    root = pathlib.Path(root); root.mkdir(parents=True, exist_ok=True)
    with open(root / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def load_graphs(root, rows):
    """Load every snapshot listed in rows as a PyG Data object (kept in memory: 2,448 x 960 nodes is small)."""
    root = pathlib.Path(root); out = []
    for r in rows:
        z = np.load(root / "snap" / f"{r['sid']}.npz")
        out.append(Data(x=torch.tensor(z["x"]), y=torch.tensor(z["y"]),
                        pos=torch.tensor(z["pos"]), area=torch.tensor(z["area"]),
                        edge_index=torch.tensor(z["edge_index"]), edge_attr=torch.tensor(z["edge_attr"]),
                        aoa=torch.tensor([r["aoa_deg"]]), sid=r["sid"], geom=r["geom"],
                        # real reference area / moment centre / length (absent in synthetic data -> proxies)
                        sref=float(z["sref"]) if "sref" in z else None, xref=float(z["xref"]) if "xref" in z else None,
                        lref=float(z["lref"]) if "lref" in z else None))
    return out


def normaliser(graphs):
    """Per-feature mean/std from training graphs only (never from test)."""
    x = torch.cat([g.x for g in graphs]); y = torch.cat([g.y for g in graphs])
    return dict(xm=x.mean(0), xs=x.std(0).clamp_min(1e-6), ym=y.mean(), ys=y.std().clamp_min(1e-6))
