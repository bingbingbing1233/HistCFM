"""Expression flow wrapper using endpoint prediction.

The denoiser predicts endpoint expression. The calling framework constructs the
iterative direction from the difference between that endpoint and the current
state; training does not directly regress a velocity in this module.
"""

import torch
import torch.nn as nn

from .flow_denoiser import CellFlowDenoiser
from .priors import GaussianPrior, ZINBPrior, ZeroPrior


class FlowExpressionModel(nn.Module):
    def __init__(
        self,
        n_genes,
        feat_dim=256,
        hidden_dim=512,
        n_layers=4,
        k_neighbors=16,
        use_zinb=True,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.denoiser = CellFlowDenoiser(
            n_genes=n_genes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            k_neighbors=k_neighbors,
        )
        self.use_zinb = use_zinb
        self.register_buffer("prior_mu", torch.zeros(n_genes))
        self.register_buffer("prior_theta", torch.ones(n_genes))
        self.register_buffer("prior_pi", torch.zeros(n_genes))
        if use_zinb:
            self.prior = ZINBPrior(n_genes, self.prior_mu, self.prior_theta, self.prior_pi)
        else:
            self.prior = GaussianPrior(n_genes)

    def load_zinb_params(self, mu, theta, pi):
        self.prior_mu.copy_(mu)
        self.prior_theta.copy_(theta)
        self.prior_pi.copy_(pi)
        self.prior = ZINBPrior(self.n_genes, self.prior_mu, self.prior_theta, self.prior_pi)

    def sample_prior(self, n_cells):
        dev = self.prior_mu.device if hasattr(self, "prior_mu") else None
        return self.prior.sample(n_cells, device=dev)

    def forward_step(self, xnucleus, coords, y_t, t, knn_graph):
        return self.denoiser(xnucleus, coords, y_t, t, knn_graph)
