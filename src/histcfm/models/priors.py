"""Prior distributions used by the HistCFM expression flow."""

import torch
import torch.nn as nn


class ZeroPrior(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.n_genes = n_genes
        self.register_buffer("_anchor", torch.tensor(0.0))

    def sample(self, n_cells, device=None):
        dev = device if device is not None else self._anchor.device
        return torch.zeros((n_cells, self.n_genes), device=dev)


class GaussianPrior(nn.Module):
    def __init__(self, n_genes, mu=None, std=None):
        super().__init__()
        self.n_genes = n_genes
        self.register_buffer("mu", torch.zeros(n_genes) if mu is None else mu)
        self.register_buffer("std", torch.ones(n_genes) if std is None else std)

    def sample(self, n_cells, device=None):
        dev = device if device is not None else self.mu.device
        mean = self.mu.unsqueeze(0).expand(n_cells, -1)
        std = self.std.unsqueeze(0).expand(n_cells, -1)
        return torch.randn(mean.shape, device=dev) * std + mean


class ZINBPrior(nn.Module):
    def __init__(self, n_genes, mu, theta, pi):
        super().__init__()
        self.n_genes = n_genes
        self.register_buffer("mu", mu)
        self.register_buffer("theta", theta)
        self.register_buffer("pi", pi)

    def sample(self, n_cells, device=None):
        dev = device if device is not None else self.mu.device
        probs_zero = self.pi.unsqueeze(0).expand(n_cells, -1).to(dev)
        is_zero = torch.bernoulli(probs_zero)
        mean = self.mu.unsqueeze(0).expand(n_cells, -1).to(dev)
        theta = self.theta.unsqueeze(0).expand(n_cells, -1).to(dev)
        var = mean + mean ** 2 / (theta + 1e-8)
        std = torch.sqrt(var.clamp_min(1e-8))
        nb_samples = torch.randn(mean.shape, device=dev) * std + mean
        nb_samples = torch.relu(nb_samples)
        return nb_samples * (1 - is_zero)
