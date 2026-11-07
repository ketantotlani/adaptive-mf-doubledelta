"""Training, prediction and metrics shared by train.py (baselines) and active.py (adaptive loop)."""
import copy, math
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from .model import PanelGNN


def integrate(dcp, area, pos, aoa_deg, sref=None, xref=None, lref=None):
    """CL and CMy from panel loading, same formula for truth, VLM and prediction:
    CL = sum(dCp*A)*cos(a)/S,  CMy = -sum(dCp*A*(x-xref))/(S*L).
    Real data passes the wing's reference area, moment centre and length; synthetic data falls back to proxies."""
    x = pos[:, 0]
    S = area.sum() if sref is None else sref
    xref = x.mean() if xref is None else xref
    c = (x.max() - x.min()).clamp_min(1e-6) if lref is None else lref
    cl = (dcp * area).sum() / S * math.cos(math.radians(aoa_deg))
    cm = -(dcp * area * (x - xref)).sum() / (S * c)
    return cl, cm


class Trainer:
    def __init__(self, in_dim, norm, cfg, device=None):
        if device is None:   # use a GPU (CUDA) or Apple-silicon GPU (MPS) when present
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.cfg, self.norm, self.device = cfg, {k: v.to(device) for k, v in norm.items()}, device
        self.model = PanelGNN(in_dim, cfg["hidden"], cfg["layers"], k_heads=cfg["k_heads"]).to(device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    def _prep(self, b):
        n = self.norm
        return (b.x.to(self.device) - n["xm"]) / n["xs"], b.edge_index.to(self.device), b.edge_attr.to(self.device)

    def fit(self, graphs, epochs, seed=0):
        """Each head sees its own bootstrap weighting of the training graphs (Poisson(1) per graph)."""
        g = torch.Generator().manual_seed(seed)
        K = self.cfg["k_heads"]
        w = torch.poisson(torch.ones(len(graphs), K), generator=g)
        data = []
        for i, gr in enumerate(graphs):
            d = copy.copy(gr); d.hw = w[i].repeat(gr.num_nodes, 1); data.append(d)
        loader = DataLoader(data, batch_size=self.cfg["batch"], shuffle=True, generator=g)
        self.model.train()
        for _ in range(epochs):
            for b in loader:
                y = ((b.y.to(self.device) - self.norm["ym"]) / self.norm["ys"])[:, None]
                pred = self.model(*self._prep(b))
                loss = ((pred - y) ** 2 * b.hw.to(self.device)).sum() / b.hw.sum().clamp_min(1)
                self.opt.zero_grad(); loss.backward(); self.opt.step()

    @torch.no_grad()
    def predict(self, graph):
        """Returns per-head predictions (N, K) in physical units."""
        self.model.eval()
        p = self.model(*self._prep(graph))
        return (p * self.norm["ys"] + self.norm["ym"]).cpu()

    @torch.no_grad()
    def evaluate(self, graphs):
        mse, cl_err, cm_err = [], [], []
        for g in graphs:
            p = self.predict(g).mean(1)
            mse.append(((p - g.y) ** 2).mean().item())
            clp, cmp_ = integrate(p, g.area, g.pos, g.aoa.item(), g.sref, g.xref, g.lref); clt, cmt = integrate(g.y, g.area, g.pos, g.aoa.item(), g.sref, g.xref, g.lref)
            cl_err.append(abs(clp - clt).item()); cm_err.append(abs(cmp_ - cmt).item())
        return dict(mse_cp=float(np.mean(mse)), cl_abs_err=float(np.mean(cl_err)), cm_abs_err=float(np.mean(cm_err)))

    @torch.no_grad()
    def uncertainty(self, graph, lam=1.0):
        """Spread of the K heads on the integrated quantities (what the loop scores)."""
        P = self.predict(graph)
        cls, cms = zip(*(integrate(P[:, k], graph.area, graph.pos, graph.aoa.item(), graph.sref, graph.xref, graph.lref) for k in range(P.shape[1])))
        return float(torch.stack(cls).var() + lam * torch.stack(cms).var())


def vlm_only(graphs, vlm_col=-1):
    """Baseline B0: use the VLM pressure column directly as the prediction (no learning)."""
    mse, cl_err, cm_err = [], [], []
    for g in graphs:
        p = g.x[:, vlm_col]
        mse.append(((p - g.y) ** 2).mean().item())
        clp, cmp_ = integrate(p, g.area, g.pos, g.aoa.item(), g.sref, g.xref, g.lref); clt, cmt = integrate(g.y, g.area, g.pos, g.aoa.item(), g.sref, g.xref, g.lref)
        cl_err.append(abs(clp - clt).item()); cm_err.append(abs(cmp_ - cmt).item())
    return dict(mse_cp=float(np.mean(mse)), cl_abs_err=float(np.mean(cl_err)), cm_abs_err=float(np.mean(cm_err)),
                per_snapshot_mse=mse)


DEFAULT_CFG = dict(hidden=64, layers=4, k_heads=5, lr=1e-3, wd=1e-5, batch=16, epochs=200, finetune_epochs=40)
