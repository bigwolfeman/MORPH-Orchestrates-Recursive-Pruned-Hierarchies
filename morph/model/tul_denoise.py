"""``tul.loop_denoise`` — each pass of the slot loop gets a job: a noise level.

WHY THIS FILE EXISTS (the measured reason, not a hope). Fourteen strict slot-loop arms
read token ``K3−K6`` inside ``[−0.0002, +0.002]`` and pass 1 does 89–95 % of the work
(`.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`).
Every target tried so far is a deterministic function of the past, and a deterministic
target has a ONE-STEP optimum: nothing in the objective asks pass 2 to do what pass 1
could not. LCM's denoising steps earn (their MI rises with steps, Figure 10) because each
step RECEIVES a noise level and has the clean target at that level as its own trained
target. This module is the arithmetic of putting that structure inside the slot loop:

* the straight-line path ``z_t = (1 − t)·z_0 + t·x_0`` (:func:`loop_denoise_interp`) —
  the same CFM convention :func:`morph.model.tul_code.cfm_pair` already draws, with
  ``x_0`` the FROZEN reference encoder's rms-normed code of the next span;
* the per-slot grid of levels over that slot's REALISED Poisson depth
  (:func:`loop_denoise_levels`);
* the map from the M code cells back into the loop's one-vector carrier
  (:class:`TULLoopDenoiseIn`), the counterpart of
  :class:`morph.model.tul_code.TULCodeProj` on the way out.

WHAT IS NOT HERE. The loop wiring (which state enters pass ``i``, teacher forcing at
train against the DDIM-style rollout at eval, the per-pass loss) lives at the one seam it
belongs to, ``MORPHTransformer._tul_core``. This file holds the arithmetic and nothing
that reads ``self``.

THE SCALE, named because it is a real difference from LCM. LCM's forward process is
variance-PRESERVING (``x^t = α_t x^0 + σ_t ε`` with ``α² + σ² = 1``); the straight line
is not. With unit-RMS ``x_0`` and ``z_0 ~ N(0, I)`` the entry's per-component RMS is
``sqrt((1−t)² + t²·r)`` with ``r ≤ 1`` the RMS of the cells' mix — 1.0 at ``t = 0``,
about 0.71 at ``t = 0.5``, back to ``r`` at ``t → 1``. The core's blocks are pre-norm, so
that range moves the residual-to-block ratio by about 1.4x across the passes and nothing
else. It is NOT renormalised here: the path has to be the one the interpolation defines,
or ``(1 − t)·z_0 + t·x_0`` stops being the straight line the target sits on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["LOOP_DENOISE_GRIDS", "loop_denoise_levels", "loop_denoise_interp",
           "TULLoopDenoiseIn"]

#: The grids ``tul.loop_denoise_grid`` accepts. One for now, by design: a second grid is a
#: second arm, and an unknown name RAISES at construction rather than falling back.
LOOP_DENOISE_GRIDS = ("linear",)


def loop_denoise_levels(depths: Tensor, t_index: int, grid: str = "linear") -> Tensor:
    """The noise level pass ``t_index`` (0-based) receives, per slot: ``[B, S]`` fp32.

    ``depths`` ``[B, S]`` is the slot's REALISED depth ``T_s`` (the Poisson draw, or the
    forced table of a depth sweep), so the grid is PER SLOT — a slot that runs three
    passes walks three levels of its own line, not the first three of a global six.

    ``"linear"``: ``t_i = (i − 1) / T_s`` in the note's 1-based indexing, i.e.
    ``t_index / T_s`` here. Pass 1 therefore enters at ``t = 0`` — PURE NOISE, the only
    level at which the entry carries nothing of the target — and the last pass enters at
    ``(T_s − 1) / T_s``, never at 1.0 (an entry at 1.0 would BE the target and the pass
    would have nothing to do).

    A slot with ``T_s = 0`` (a pad; the depth draw floors at 1, so this is defensive)
    reads 0.0 rather than dividing by zero.
    """
    if grid not in LOOP_DENOISE_GRIDS:
        raise ValueError(
            f"loop_denoise_levels: unknown grid {grid!r}, known {LOOP_DENOISE_GRIDS}")
    if t_index < 0:
        raise ValueError(f"loop_denoise_levels: t_index must be >= 0, got {t_index}")
    d = depths.float().clamp(min=1.0)
    return torch.full_like(d, float(t_index)) / d


def loop_denoise_interp(z0: Tensor, x0: Tensor, t: Tensor) -> Tensor:
    """``z_t = (1 − t)·z_0 + t·x_0`` — the straight-line path, in fp32.

    ``z0`` / ``x0`` ``[B, S, M, C]``, ``t`` ``[B, S]`` (one level per SLOT, shared by that
    slot's M cells: one code per span, one level per span, the :func:`cfm_pair` rule).
    Returns fp32; the caller casts to the carrier's dtype.
    """
    if z0.shape != x0.shape:
        raise ValueError(
            f"loop_denoise_interp: z0 {tuple(z0.shape)} must match x0 {tuple(x0.shape)}")
    if t.shape != z0.shape[:2]:
        raise ValueError(
            f"loop_denoise_interp: t {tuple(t.shape)} must be [B, S] = "
            f"{tuple(z0.shape[:2])}")
    tt = t.float().view(*t.shape, 1, 1)
    return (1.0 - tt) * z0.float() + tt * x0.float()


class TULLoopDenoiseIn(nn.Module):
    """The M code cells -> ONE loop-carrier state ``[B, S, M, C] -> [B, S, C]``.

    ``h = Σ_i W_in[i] · z_i``, ``W_in[i]`` initialised to ``I / M`` and no bias, so at
    init this IS the mean of the slot's M cells — the exact counterpart of
    :class:`~morph.model.tul_code.TULCodeProj`, whose ``W_code`` is identity at init and
    whose forward is ``rmsnorm(W_code[i] · r)``. Round-tripping a state through the two at
    init returns the state's direction with its RMS normalised, which is what makes the
    entry and the readout the same geometry.

    NO RNG DRAW (the ``TULCodeProj`` precedent): a denoise model's other weights are
    byte-identical to its ruler's. Excluded from ternary QAT for the same reason
    ``TULCodeProj`` is — it is the channel between the loop and a dense frozen code.

    A slot with ``valid`` false leaves EXACTLY 0, so a pad enters the loop at the zero the
    gather would have given it.
    """

    def __init__(self, d_model: int, m_cells: int):
        super().__init__()
        if m_cells < 1:
            raise ValueError(f"TULLoopDenoiseIn needs m_cells >= 1, got {m_cells}")
        self.m = int(m_cells)
        self.W_in = nn.Parameter(
            torch.eye(d_model).unsqueeze(0).repeat(self.m, 1, 1) / float(self.m))
        self._ternary_exclude = True

    def forward(self, z: Tensor, valid: Tensor) -> Tensor:
        """``z`` ``[B, S, M, C]`` -> ``[B, S, C]``; ``valid`` ``[B, S]`` bool."""
        if z.dim() != 4 or z.shape[2] != self.m:
            raise ValueError(
                f"TULLoopDenoiseIn: z must be [B, S, {self.m}, C], got {tuple(z.shape)}")
        B, S, M, C = z.shape
        w = self.W_in.to(z.dtype)
        out = torch.einsum("bsmc,mdc->bsd", z, w)
        return out * valid.view(B, S, 1).to(out.dtype)
