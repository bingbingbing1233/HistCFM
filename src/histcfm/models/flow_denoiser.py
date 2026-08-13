"""Endpoint-expression denoiser used by the HistCFM flow wrapper.

Each block updates an endpoint expression estimate. The iterative direction is
formed by the calling framework from endpoint minus current state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBlock(nn.Module):
    def __init__(self, feat_dim, n_genes, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(feat_dim, hidden_dim)
        self.k_proj = nn.Linear(feat_dim, hidden_dim)
        self.v_proj = nn.Linear(feat_dim, hidden_dim)
        self.c_proj = nn.Linear(3, hidden_dim)
        self.y_proj = nn.Linear(n_genes, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm_outz = nn.LayerNorm(feat_dim)
        self.out_z = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        self.out_y = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, Z, coords, Y_t, t, knn_graph):
        N = Z.shape[0]
        Q = self.q_proj(Z)
        K = self.k_proj(Z)
        V = self.v_proj(Z)
        t_feat = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=Z.device)
        # ensure shape [N, 1]
        if t_feat.ndim == 0:
            t_feat = t_feat.view(1, 1).expand(N, 1)
        elif t_feat.ndim == 1:
            if t_feat.numel() == 1:
                t_feat = t_feat.view(1, 1).expand(N, 1)
            else:
                t_feat = t_feat.view(-1, 1)
        else:
            if t_feat.shape[1] != 1:
                t_feat = t_feat.view(-1, 1)
        c_node = torch.cat([coords, t_feat], dim=-1)
        C = self.c_proj(c_node)
        Yh = self.y_proj(Y_t)
        Keys = K + C + Yh
        Qb = Q.unsqueeze(0)
        Kb = Keys.unsqueeze(0)
        Vb = V.unsqueeze(0)
        Qb = torch.nan_to_num(Qb)
        Kb = torch.nan_to_num(Kb)
        Vb = torch.nan_to_num(Vb)
        attn_out, _ = self.attn(Qb, Kb, Vb)
        attn_out = attn_out.squeeze(0)
        agg = torch.cat([attn_out, C], dim=-1)
        Z_new = self.out_z(torch.nan_to_num(agg)) + Z
        Z_new = torch.tanh(self.norm_outz(torch.nan_to_num(Z_new)))
        Y_new = Y_t + self.out_y(torch.nan_to_num(Z_new))
        Y_new = torch.nan_to_num(Y_new)
        Y_new = torch.clamp(Y_new, 0.0, 100.0)
        return Z_new, Y_new


class CellFlowDenoiser(nn.Module):
    def __init__(self, n_genes, feat_dim=256, hidden_dim=512, n_layers=4, k_neighbors=16):
        super().__init__()
        self.n_layers = n_layers
        self.k = k_neighbors
        self.in_proj = nn.Linear(feat_dim, feat_dim)
        self.norm_in = nn.LayerNorm(feat_dim)
        self.blocks = nn.ModuleList([
            SpatialBlock(feat_dim, n_genes, hidden_dim) for _ in range(n_layers)
        ])

    def forward(self, xnucleus, coords, y_t, t, knn_graph):
        Z = self.in_proj(xnucleus)
        Z = torch.tanh(self.norm_in(torch.nan_to_num(Z)))
        Y = y_t
        for blk in self.blocks:
            Z, Y = blk(Z, coords, Y, t, knn_graph)
        return torch.nan_to_num(Y)
