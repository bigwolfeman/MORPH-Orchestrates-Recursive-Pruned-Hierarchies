"""
SIGReg: Sketched Isotropic Gaussian Regularization.

Enforces isotropic Gaussian distribution on embedding vectors via the Epps-Pulley
characteristic function test. Prevents representational collapse without requiring
EMA teacher networks.

From LeJEPA (2025): ~20 lines, single hyperparameter (lambda), proven across 60+ architectures.

Applied to neural memory output to prevent the oscillation-collapse failure mode
where memory MLP outputs degenerate into a low-rank subspace during inner/outer
loop timescale mismatch.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Epps-Pulley characteristic function test for anti-collapse regularization.

    Computes a statistic measuring deviation of the input distribution from
    an isotropic Gaussian. Returns a scalar loss — add to total loss with
    weight lambda (default 0.02).

    Args:
        knots: Number of quadrature points on [0, 3] (default 17).
        n_slices: Number of random projection slices (default 256).
    """

    def __init__(self, knots: int = 17, n_slices: int = 256):
        super().__init__()
        self.n_slices = n_slices

        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute SIGReg statistic.

        Args:
            proj: Embedding tensor [..., N, D] where N is batch/token dim
                  and D is embedding dim. Operates on last two dims.

        Returns:
            Scalar loss measuring deviation from isotropic Gaussian.
        """
        # Random projection directions (regenerated each call for stochasticity)
        A = torch.randn(proj.size(-1), self.n_slices, device=proj.device, dtype=proj.dtype)
        A = A.div_(A.norm(p=2, dim=0))

        # Project embeddings onto random directions: [..., N, n_slices]
        projected = proj @ A

        # Characteristic function test at quadrature points
        # x_t: [..., N, n_slices, knots]
        x_t = projected.unsqueeze(-1) * self.t

        # Empirical CF vs Gaussian CF
        # Mean over sample dim (-3 is the N dim after unsqueeze)
        cos_mean = x_t.cos().mean(-3)
        sin_mean = x_t.sin().mean(-3)
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # Weighted integral, scaled by sample size
        statistic = (err @ self.weights) * proj.size(-2)

        return statistic.mean()
