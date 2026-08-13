"""UNI-to-hint fusion blocks used by the audited HistCFM framework.

The audited cell-level framework uses the fused representation for its hint
branch; these blocks do not route UNI features directly into the flow input.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class FusionGate(nn.Module):
    def __init__(self, h_dim, u_dim, out_dim, hidden_dim=256, dropout=0.0, bias_init=-2.0):
        super().__init__()
        self.proj_u = nn.Linear(u_dim, out_dim)
        self.h_norm = nn.LayerNorm(h_dim)
        self.up_norm = nn.LayerNorm(out_dim)
        self.g_mlp = MLP(h_dim + out_dim, hidden_dim, out_dim, dropout=dropout)
        with torch.no_grad():
            self.g_mlp.fc2.bias.fill_(bias_init)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h, u):
        up = self.proj_u(u)
        h = self.h_norm(h)
        up = self.up_norm(up)
        g = self.sigmoid(self.g_mlp(torch.cat([h, up], dim=-1)))
        z = h + g * up
        return z, g, up


class FusionFiLM(nn.Module):
    def __init__(self, h_dim, u_dim, out_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.gamma = MLP(u_dim, hidden_dim, out_dim, dropout=dropout)
        self.beta = MLP(u_dim, hidden_dim, out_dim, dropout=dropout)
        self.h_norm = nn.LayerNorm(h_dim)

    def forward(self, h, u):
        h = self.h_norm(h)
        g = self.gamma(u)
        b = self.beta(u)
        z = g * h + b
        return z, g, b


class FusionAttention(nn.Module):
    def __init__(self, h_dim, u_dim, out_dim, hidden_dim=256, dropout=0.0, num_heads=8):
        super().__init__()
        self.q_proj = nn.Linear(h_dim, out_dim)
        self.k_proj = nn.Linear(u_dim, out_dim)
        self.v_proj = nn.Linear(u_dim, out_dim)
        self.attn = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, h, u):
        q = self.q_proj(h).unsqueeze(0)
        k = self.k_proj(u).unsqueeze(0)
        v = self.v_proj(u).unsqueeze(0)
        o, _ = self.attn(q, k, v)
        o = o.squeeze(0)
        o = self.norm1(o + self.q_proj(h))
        z = self.norm2(self.ffn(o) + o)
        return z
