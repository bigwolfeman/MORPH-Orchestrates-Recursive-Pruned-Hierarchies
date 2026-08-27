"""SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv 2511.08544).

Why MORPH has this: the TUL slot states are measurably COLLAPSED. The 50 valid
slot states of a row have an effective rank of 1.7-4.8 in a 1024-dimensional
space with a mean pairwise cosine of +0.39 to +0.71, at every checkpoint
including the healthy ones (lab/experiments/failures/2026-08-24-tul-takeover-cure.md).
They are built from one shared ``E_slot`` plus a span bag-mean, and a mean over
many token embeddings concentrates, so they are near-parallel by construction.

SIGReg enforces ``E[Z] = 0`` and ``Cov(Z) = I`` on a set of embeddings without
moment matching, prototypes, stop-gradients or a teacher: it tests each of M
random 1-D projections against a standard Gaussian with the Epps-Pulley
characteristic-function statistic, which the paper's Theorem 4 shows has bounded
loss, gradient and curvature for ANY input distribution. Cost is O(N) in the
number of embeddings.

The implementation below is Algorithm 1 of the paper transcribed, with the
single-GPU simplification that the DDP ``all_reduce(ecf, op="AVG")`` is dropped
(one device ⇒ the mean is already global) and ``N = z.size(0)``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sigreg_epps_pulley"]


def sigreg_epps_pulley(z: Tensor, step: int | None = None, *, num_slices: int = 256,
                       n_knots: int = 17, t_max: float = 5.0) -> Tensor:
    """SIGReg loss for a set of embeddings ``z`` of shape ``[N, d]``.

    Args:
        z:          the embeddings to push toward an isotropic standard Gaussian.
            NOT standardised first — standardising would make the loss vacuous;
            the whole point is that it PULLS the distribution to N(0, I).
        step:       seeds the direction draw. The paper seeds by global step so
            that DDP ranks agree; on one device that sync is moot, so ``None``
            (the default in the model path) draws from the global RNG, which
            varies the directions every step — the coverage property that
            matters — and keeps the run reproducible through the run seed.
        num_slices: M, the number of random directions (paper default 256).
        n_knots:    trapezoid knots for the CF integral. The paper ablates this
            (their Figure 20) and finds 17 sufficient.
        t_max:      integration half-range; the paper uses t ∈ [-5, 5].

    Returns:
        Scalar loss, the MEAN over directions. Definition 2 uses the average
        rather than the maximum of Theorem 2 deliberately, "to avoid sparse
        gradient over the directions".
    """
    if z.dim() != 2:
        raise ValueError(f"sigreg expects [N, d], got {tuple(z.shape)}")
    n, d = z.shape
    if n < 2:
        # A characteristic function of one point carries no distributional
        # information; returning 0 keeps a degenerate batch from injecting noise.
        return z.new_zeros(())

    if step is None:
        a = torch.randn(d, num_slices, device=z.device, dtype=torch.float32)
    else:
        g = torch.Generator(device=z.device)
        g.manual_seed(int(step))
        a = torch.randn(d, num_slices, generator=g, device=z.device, dtype=torch.float32)
    a = a / a.norm(p=2, dim=0, keepdim=True)

    t = torch.linspace(-t_max, t_max, n_knots, device=z.device, dtype=torch.float32)
    exp_f = torch.exp(-0.5 * t.square())              # CF of N(0,1) == Gauss window

    proj = (z.float() @ a).unsqueeze(2) * t           # [N, M, T]
    ecf = torch.exp(1j * proj).mean(0)                # [M, T], complex
    err = (ecf - exp_f).abs().square() * exp_f        # [M, T], weighted L2
    per_slice = torch.trapz(err, t, dim=1) * n        # [M]
    return per_slice.mean()
