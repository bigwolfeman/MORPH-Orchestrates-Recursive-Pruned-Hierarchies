"""LXTUL-E's enumerated loop code (``tul.code_enum_k > 1``).

The slot loop runs K rollouts per row (rollout-major batch expansion, the GK machinery in
``MORPHTransformer._forward_tul``). Rollout ``k`` differs from the others in ONE thing: a
fixed-size code added at the end of EVERY pass of the slot loop,

    h  <-  f(h) + r * rms(f(h)).detach() * u_k,

``r = tul.code_enum_ratio``, ``rms`` per slot over the Hyper-Connection streams and the
channels. Placement and reasons: ``MORPHTransformer._tul_core`` (search "tul.code_enum").

THE CODES. ``u_k`` for ``k = 0 .. K-1`` are the vertices of a regular simplex in a LEARNED
``K-1``-dimensional subspace of the model width ``C``:

    u = sqrt(C) * S @ Q^T,      S [K, K-1] fixed,   Q [C, K-1] orthonormal columns,

``S`` the Helmert simplex (rows unit norm, columns summing to zero) and ``Q`` the
Gram-Schmidt orthonormalisation of the learned ``basis [K-1, C]``. So, EXACTLY and for every
value of the parameter:

* every ``u_k`` has unit RMS (``|u_k|^2 = C |S_k|^2 = C``) — the choice has a fixed size;
* ``sum_k u_k = 0`` — the codes are mean-free, so the rollout average adds nothing;
* every pair is equally far apart (``|u_k - u_l|^2 = 2 C K / (K-1)``).

What is learned is the orientation of the simplex, nothing else: no scale (Lean
``iid_matched_strict``: a learned size dies), no merging of two codes. The note asked for
"a learned unit direction, mean-free across k"; per-code normalisation of free vectors
cannot give both exactly (dividing by different norms breaks the zero sum), and a jointly
normalised table lets one code shrink while another grows. The simplex gives both.

The basis carries no weight decay (``optimizer._NO_DECAY_KEYWORDS``): the map from it to
``u`` is scale-invariant, so decay would only shrink its norm and raise its effective step.
It is not a ``nn.Linear``, so the ternary QAT never touches it. RNG-neutral: the basis is
drawn from a private generator, so an arm with the key on keeps its ruler's base weights.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

_SEED_BASIS = 0xE4C0DE


def helmert_simplex(k: int) -> Tensor:
    """``[k, k-1]`` float64: the vertices of a regular simplex centred at 0, unit norm.

    Column ``j`` (``j = 1 .. k-1``) is the ``j``-th Helmert contrast ``(1, .., 1, -j, 0, ..)
    / sqrt(j (j+1))``: the columns are orthonormal and each sums to zero, so a row is the
    coordinates of ``e_k - 1/k`` in that basis, of norm ``sqrt((k-1)/k)``; rescaled to 1."""
    if k < 2:
        raise ValueError(f"a simplex needs k >= 2 vertices, got {k}")
    H = torch.zeros(k, k - 1, dtype=torch.float64)
    for j in range(1, k):
        H[:j, j - 1] = 1.0
        H[j, j - 1] = -float(j)
        H[:, j - 1] /= math.sqrt(j * (j + 1))
    return H * math.sqrt(k / (k - 1))


def gram_schmidt_rows(a: Tensor) -> Tensor:
    """``[n, C]`` -> ``[n, C]`` with orthonormal rows spanning the same flag (classical
    Gram-Schmidt, run twice for fp32 accuracy; differentiable; n is K-1 = 3 here)."""
    rows: list[Tensor] = []
    for i in range(a.shape[0]):
        v = a[i]
        for _ in range(2):
            for q in rows:
                v = v - (v @ q) * q
        rows.append(v / v.norm().clamp_min(1e-12))
    return torch.stack(rows, dim=0)


class TULCodeEnum(nn.Module):
    """K enumerated codes and the per-pass term that re-adds them (module doc)."""

    def __init__(self, d_model: int, k: int, ratio: float):
        super().__init__()
        if k < 2:
            raise ValueError(f"TULCodeEnum needs k >= 2 (k = 1 builds nothing), got {k}")
        if not ratio > 0.0:
            raise ValueError(f"code ratio must be > 0, got {ratio}")
        self.k = int(k)
        self.ratio = float(ratio)
        self.d_model = int(d_model)
        g = torch.Generator().manual_seed(_SEED_BASIS)
        self.basis = nn.Parameter(torch.randn(self.k - 1, self.d_model, generator=g))
        self.register_buffer("simplex", helmert_simplex(self.k).float(), persistent=False)

    def directions(self) -> Tensor:
        """``[K, C]`` fp32: the codes ``u_k`` (unit RMS, sum zero, equidistant)."""
        with torch.autocast(device_type=self.basis.device.type, enabled=False):
            q = gram_schmidt_rows(self.basis.float())                     # [K-1, C]
            return (self.simplex.float() @ q) * math.sqrt(self.d_model)

    def term(self, h: Tensor, valid: Tensor, n_rollouts: int) -> Tensor:
        """``[B, S, C]`` the code term for the carrier ``h`` ``[B, S, (n,) C]`` whose rows
        are ``n_rollouts`` rollout-major copies of a base batch: row ``b`` carries code
        ``b // (B / n_rollouts)``. ``r * rms(h).detach() * u_k`` per slot, zero on pads
        (``valid [B, S]``). ``n_rollouts`` must be ``K``: a loop run on the base batch
        would give every row code 0, a silent one-code model."""
        if n_rollouts != self.k:
            raise RuntimeError(
                f"tul.code_enum_k={self.k}: the slot loop must run on the {self.k}-fold "
                f"rollout batch (iw_rollouts={self.k}), got {n_rollouts}. Call the model's "
                f"forward, or pass the expanded batch and iw_rollouts to `_tul_core`.")
        B = h.shape[0]
        if B % self.k:
            raise RuntimeError(f"batch {B} is not a multiple of code_enum_k {self.k}")
        rms = h.detach().float().flatten(2).pow(2).mean(-1).sqrt()          # [B, S]
        u = self.directions().repeat_interleave(B // self.k, dim=0)        # [B, C]
        t = (self.ratio * rms * valid.float()).unsqueeze(-1) * u.unsqueeze(1)
        return t.to(h.dtype)
