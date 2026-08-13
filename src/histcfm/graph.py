"""K-nearest-neighbor graph construction used by HistCFM."""

import torch


def build_knn_graph(coords, k=16):
    N = coords.shape[0]
    if N == 0:
        return torch.empty((0, 0), dtype=torch.long, device=coords.device)
    dists = torch.cdist(coords, coords)
    dists.fill_diagonal_(float("inf"))
    k_eff = max(1, min(k, max(N - 1, 1)))
    knn_idx = torch.topk(-dists, k=k_eff, dim=1).indices
    return knn_idx
