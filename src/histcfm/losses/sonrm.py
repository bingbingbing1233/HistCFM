"""Spatially ordered neighborhood representation margin loss."""

import torch
import torch.nn.functional as F

from ..graph import build_knn_graph


def _neighbors_hops(knn_idx, start_idx, hops):
    visited = set([int(start_idx)])
    current = set([int(start_idx)])
    result = []
    for _ in range(hops):
        nxt = set()
        for i in current:
            row = knn_idx[i]
            for j in row.tolist():
                if j not in visited:
                    nxt.add(int(j))
        visited |= nxt
        result.append(list(nxt))
        current = nxt
    return result


class SONRMLoss(torch.nn.Module):
    def __init__(self, k=16, hops=3, m12=0.1, m23=0.1):
        super().__init__()
        self.k = int(k)
        self.hops = int(hops)
        self.m12 = float(m12)
        self.m23 = float(m23)

    def forward(self, z, coords):
        if z is None or coords is None or z.shape[0] == 0:
            return torch.tensor(
                0.0,
                device=z.device if isinstance(z, torch.Tensor) else torch.device("cpu"),
            )
        knn_idx = build_knn_graph(coords, k=self.k)
        N = z.shape[0]
        H = min(self.hops, 3)
        s_sums = [torch.zeros((), device=z.device) for _ in range(H)]
        cnt = [0] * H
        for i in range(N):
            hops_lists = _neighbors_hops(knn_idx, i, H)
            for h, lst in enumerate(hops_lists):
                if len(lst) == 0:
                    continue
                sims = F.cosine_similarity(z[i].unsqueeze(0), z[lst], dim=1)
                s_sums[h] = s_sums[h] + sims.mean()
                cnt[h] += 1
        s_vals = []
        for h in range(H):
            if cnt[h] > 0:
                s_vals.append(s_sums[h] / cnt[h])
            else:
                s_vals.append(torch.zeros((), device=z.device))
        s1 = s_vals[0] if H > 0 else torch.zeros((), device=z.device)
        s2 = s_vals[1] if H > 1 else torch.zeros((), device=z.device)
        s3 = s_vals[2] if H > 2 else torch.zeros((), device=z.device)
        t1 = s1 - s2
        t2 = s2 - s3
        l1 = F.relu(torch.tensor(self.m12, device=z.device) - t1)
        l2 = F.relu(torch.tensor(self.m23, device=z.device) - t2)
        return l1 + l2
