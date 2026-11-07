"""GNN baseline in the spirit of MF-VortexNet (Shen & Alonso): graph attention over the panel graph,
input = geometry + flow condition + VLM pressure per panel, output = RANS pressure per panel.
NOT the paper's exact architecture (theirs is a graph-attention U-Net); a simpler encoder-processor-decoder
with K output heads, so the same model gives a prediction (head mean) and an uncertainty (head spread).
"""
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


class PanelGNN(nn.Module):
    def __init__(self, in_dim, hidden=64, layers=4, att_heads=4, edge_dim=4, k_heads=5, dropout=0.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(GATv2Conv(hidden, hidden // att_heads, heads=att_heads, edge_dim=edge_dim)
                                   for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        # K small heads on one shared trunk: their disagreement is the uncertainty signal
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                                   for _ in range(k_heads))

    def forward(self, x, edge_index, edge_attr):
        h = self.enc(x)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.drop(torch.nn.functional.silu(conv(h, edge_index, edge_attr))))
        return torch.cat([hd(h) for hd in self.heads], dim=1)   # (N, K)
