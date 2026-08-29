"""TUL-FM Phase 1 — the flow-matching span planner.

Arc: ``.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md``.
Gates: ``docs/tul-fm-probing.md`` §5 (P1).
Pre-registration: ``lab/experiments/planned/2026-08-28-tulfm-p1.md``.

WHAT P1 IS. A small NEW trainable planner sits on top of a FROZEN MORPH-A3 backbone.
The backbone contributes only features: ``Hpre = input_norm(prelude(input_ids))``, the
exact tensor the A3 coda consumes (A3 is ``n_core == 0``, so ``_core_region`` reduces to
``input_norm``). The planner learns to denoise the POOLED representation of the NEXT
span from noise, conditioned on the frozen states over the context that precedes it.
It answers one question and no other: **can this objective write span-(i+1) content into
a latent at all?** The gate is a retrieval probe (``lab/tulfm/retrieval_probe.py``), not
cross-entropy.

WHAT IS REUSED, VERBATIM, FROM THE AUDITED DIFFUSIONBLOCKS MODULE
(``morph/model/diffusion_blocks.py``, audit ``docs/diffusionblocks-reference-audit.md``):

* ``DBSchedule`` — the truncated log-normal restricted to a range and RENORMALISED in
  CDF space, plus ``inference_sigmas`` (equi-probability, strictly descending).
* ``EDMPrecond`` — ``c_skip / c_out / c_in / c_noise`` and the EDM loss weight
  ``w(σ) = (σ² + σ_d²)/(σ·σ_d)²``.
* ``SigmaConditioning`` / ``AdaLNGate`` — σ → AdaLN-Zero modulation.
* ``euler_step`` — the authors'-code sign (``dt = σ_next − σ < 0``), NOT the paper's
  rendered Eq (3)-(5).

The ladder shape (``z ~ N(0, (1 + σ_max²)·I)`` at init, descending σ, one extra read at
σ_min so the lowest-noise regime gets to speak) is lifted from the audited sampler
``morph/inference/db_generate.py`` on branch ``feat/db-objective-l2``.

WHAT IS DELIBERATELY *NOT* REUSED. ``SliceScaler``. Per-component-std target scaling is
the known scar: it pushed σ* to 3.3 and put 77-98 % of training into trivial
autoencoding. Targets here are **unit L2 norm** (``‖y‖₂ = 1``), per the DB paper's own
App. C and per the P1 spec. See ``sigma_data`` below for the tension this creates and
how it is instrumented rather than hidden.

DEVIATIONS FROM THE P1 PSEUDO-CODE (each one, with its reason)
--------------------------------------------------------------
1. **Positional information in the cross-attention.** The pseudo-code's planner had no
   positions, which makes the cross-attention a bag-of-words reader: it could not tell
   "the end of my own span" from "some token 400 positions back". Fixed sinusoidal
   position codes are added to the projected context, and the SAME code at ``e_i`` (plus
   a learned slot-index embedding) is added to the slot query. No new failure mode: the
   codes are a fixed buffer, they carry no gradient, and they do not widen the mask.
2. **One 4-layer stack, applied once per ``denoise`` call.** The pseudo-code reads
   "ONE weight-shared σ-conditioned block (DB §5.5 Huginn style)" together with
   "blocks: N=4 pre-norm transformer layers". Those are reconciled the only way that is
   consistent with §5.5: the *whole 4-layer stack* is the one denoiser (B = 1), shared
   across every σ and every Euler step; it is not four separate per-band denoisers.
3. **``out`` is zero-initialised** (DiT AdaLN-Zero discipline, already used by
   ``AdaLNGate`` and MORPH's ``ChannelInject``). At init ``F_θ ≡ 0`` so
   ``D̂ = c_skip·z`` exactly. This makes the untrained-planner control in the probe a
   *defined* floor (a scaled copy of noise ranks at chance) instead of an accident of
   initialisation.
4. **The generated plan is the final read at σ_min**, i.e. ``D̂``, not the post-step
   ``z``. This follows the audited sampler (``db_generate.py``: "the loop above stops one
   short, so without it the lowest-noise block never gets to speak"). ``final_read=False``
   returns the pseudo-code's ``z`` and is exercised by the test suite.
5. **The MLP is SwiGLU**, MORPH's house style, rather than an unspecified "MLP". The
   hidden bank is ``2 * d_ff`` wide, which is why the shipped ``d_ff`` is 1408 (2.75x
   ``d_p``, the same ratio ``base.yaml`` uses) rather than 2048: at 2048 the planner
   weighs 25.97 M, outside the declared 15-25 M band.
6. **The context features are also the target features.** ``Hpre`` is used for BOTH the
   conditioning and the pooled targets, as the spec requires. That choice is what makes
   the causality argument checkable: a target lives strictly at positions > ``e_i``, and
   the conditioning mask admits strictly positions ≤ ``e_i``.

σ_data AND THE UNIT-NORM TENSION (read before changing either)
--------------------------------------------------------------
Unit L2 norm over ``d = 1024`` means a per-component std of ``1/√1024 = 0.031``, while
``sigma_data`` is 0.5 — the value our own DB oracle sweep picked and the value the EDM
preconditioning constants were read off with. ``SliceScaler``'s docstring calls those two
settings mutually inconsistent, and it is right *as arithmetic*. P1 keeps them anyway,
because the scar we are avoiding is the other one: making ``σ_data`` "true" by inflating
the target is exactly the move that shoved σ* to 3.3 and drowned training in autoencoding.

MEASURED CONSEQUENCE, stated so nobody rediscovers it in a loss curve. Write the EDM
objective in its preconditioned form, ``loss = ‖F_θ − F_target‖²`` with
``F_target = (y − c_skip·z)/c_out``. Then:

* ``σ ≪ σ_data``: ``F_target → −ε``, so the floor is ``‖ε‖² = d = 1024`` and NOTHING in
  that band is learnable — it is pure noise by construction.
* ``σ ≫ σ_data``: ``F_target → y/σ_data``, so the floor is ``‖y‖²/σ_data² = 4``.

Standard EDM has ``d`` in BOTH limits, because it assumes the target's per-component std
IS ``σ_data``. Ours does not, so the untrainable low-σ band carries ~256x the loss
magnitude of the band we actually care about. The total loss is therefore a bad summary
statistic here — this is the same shape of error as "EDM weighting applied to
cross-entropy", which the DB campaign already paid for. Two consequences are built in
rather than argued about: :func:`fm_loss` logs a NULL FLOOR per band (the loss a
``F_θ ≡ 0`` denoiser gets on the same σ and the same ε) so every band curve is read as
``loss / null``, and ``sigma_data`` is a config key. If the high-σ bands refuse to move,
``sigma.sigma_data`` ≈ ``1/√d`` ≈ 0.031 is the first thing to sweep — it is the setting
that makes the two limits balance without inflating the target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from morph.model.attention import RMSNorm
from morph.model.diffusion_blocks import (
    AdaLNGate,
    DBConfig,
    DBSchedule,
    EDMPrecond,
    SigmaConditioning,
    euler_step,
)
from morph.model.tul_layout import BoundaryRule

__all__ = [
    "FMPlannerConfig",
    "SpanGeometry",
    "FMPlanner",
    "TargetWhitener",
    "segment_rows",
    "pool_targets",
    "make_targets",
    "fit_whitener",
    "build_masks",
    "build_schedule",
    "band_edges",
    "band_edges_for",
    "band_of_sigma",
    "analytic_null_floor",
    "fm_loss",
    "generate_plans",
    "effective_rank",
    "mean_pairwise_cos",
]


# ── span geometry ────────────────────────────────────────────────────────────

@dataclass
class SpanGeometry:
    """Fixed-shape per-row slot geometry for one batch.

    Slot ``i`` sits at the END of span ``i`` (position ``slot_end[b, i] == e_i``, the last
    token the slot may condition on). Its TARGET is span ``i+1``, the inclusive token
    range ``[tgt_start, tgt_end]``. A slot is VALID only when span ``i+1`` exists and is
    fully closed inside the row — the trailing span after the last boundary is left open
    by ``BoundaryRule.cut`` and is therefore never a target.

    Every field is ``[B, S]`` with ``S == max_slots``; padding entries are 0 and masked
    by ``valid``.
    """

    slot_end: Tensor      # [B, S] long — e_i
    tgt_start: Tensor     # [B, S] long — s_{i+1}
    tgt_end: Tensor       # [B, S] long — e_{i+1}, inclusive
    valid: Tensor         # [B, S] bool
    n_spans_total: int    # spans found in the batch (closed spans only)
    n_slots_valid: int    # slots that survived every rule
    n_dropped_budget: int # valid slots discarded because they exceeded max_slots
    seq_len: int

    @property
    def dropped_fraction(self) -> float:
        """Fraction of closed spans that do NOT become a training slot.

        A span is not a slot when it is the last closed span of its row (no span i+1) or
        when it fell off the ``max_slots`` budget. Reported every run — a silently high
        number means the objective is training on far less data than the token count
        suggests.
        """
        if self.n_spans_total == 0:
            return 1.0
        return 1.0 - (self.n_slots_valid / self.n_spans_total)

    def to(self, device) -> "SpanGeometry":
        return SpanGeometry(
            slot_end=self.slot_end.to(device), tgt_start=self.tgt_start.to(device),
            tgt_end=self.tgt_end.to(device), valid=self.valid.to(device),
            n_spans_total=self.n_spans_total, n_slots_valid=self.n_slots_valid,
            n_dropped_budget=self.n_dropped_budget, seq_len=self.seq_len,
        )


def segment_rows(input_ids: Tensor, rule: BoundaryRule, max_slots: int) -> SpanGeometry:
    """Cut every row into spans with THE boundary rule and build the slot geometry.

    ``rule`` is ``morph.model.tul_layout.BoundaryRule`` — the same resumable state machine
    the TUL loader and the TUL generator use (``.;!?`` suffixes + newline/dash substrings,
    no comma; ``min_span`` suppression; ``span_cap`` forcing; EOS unconditional). There is
    exactly one boundary rule in this repository and this is it.

    ``cut`` returns the positions ``p_0 … p_{m-1}`` of span-final tokens. Spans are
    ``[0, p_0]``, ``[p_0+1, p_1]``, … and the tail after ``p_{m-1}`` is OPEN. So slot ``i``
    (ending at ``p_i``) has a closed target span ``[p_i + 1, p_{i+1}]`` for
    ``i = 0 … m-2`` — ``m-1`` valid slots from ``m`` closed spans.
    """
    if max_slots < 1:
        raise ValueError(f"max_slots must be >= 1, got {max_slots}")
    B, L = input_ids.shape
    ids_np = input_ids.detach().to("cpu").numpy()

    slot_end = np.zeros((B, max_slots), dtype=np.int64)
    tgt_start = np.zeros((B, max_slots), dtype=np.int64)
    tgt_end = np.zeros((B, max_slots), dtype=np.int64)
    valid = np.zeros((B, max_slots), dtype=bool)

    n_spans_total = 0
    n_slots_valid = 0
    n_dropped_budget = 0
    for b in range(B):
        pos, _ = rule.cut(ids_np[b])
        m = int(pos.shape[0])
        n_spans_total += m
        n_avail = max(m - 1, 0)
        n_keep = min(n_avail, max_slots)
        n_dropped_budget += n_avail - n_keep
        n_slots_valid += n_keep
        if n_keep == 0:
            continue
        slot_end[b, :n_keep] = pos[:n_keep]
        tgt_start[b, :n_keep] = pos[:n_keep] + 1
        tgt_end[b, :n_keep] = pos[1:n_keep + 1]
        valid[b, :n_keep] = True

    dev = input_ids.device
    return SpanGeometry(
        slot_end=torch.from_numpy(slot_end).to(dev),
        tgt_start=torch.from_numpy(tgt_start).to(dev),
        tgt_end=torch.from_numpy(tgt_end).to(dev),
        valid=torch.from_numpy(valid).to(dev),
        n_spans_total=n_spans_total, n_slots_valid=n_slots_valid,
        n_dropped_budget=n_dropped_budget, seq_len=int(L),
    )


def pool_targets(h: Tensor, geom: SpanGeometry, eps: float = 1e-8) -> Tensor:
    """``[B, S, d]`` UNIT-NORM pooled representation of each slot's NEXT span.

    ``y_i = mean(h[b, s_{i+1} : e_{i+1}+1])`` then ``y_i ← y_i / ‖y_i‖₂``.

    Unit L2 norm, NOT per-component-std scaling. ``SliceScaler`` is the scar: normalising
    each embedding slice to per-component std ``σ_data`` inflated the target and pushed
    the useful σ range up to 3.3, which put 77-98 % of training into the autoencoding
    regime where the denoiser only has to copy ``z``. Rows for invalid slots come back
    EXACTLY zero.
    """
    B, L, d = h.shape
    idx = torch.arange(L, device=h.device)
    span = ((idx[None, None, :] >= geom.tgt_start[..., None])
            & (idx[None, None, :] <= geom.tgt_end[..., None])
            & geom.valid[..., None])                      # [B, S, L]
    cnt = span.sum(-1, keepdim=True).clamp_min(1).float()
    y = torch.bmm(span.float(), h.float()) / cnt          # [B, S, d]
    y = y / y.norm(dim=-1, keepdim=True).clamp_min(eps)
    return y * geom.valid[..., None].float()


# ── target whitening (P1c) ───────────────────────────────────────────────────

class TargetWhitener:
    """Mean-centre, project onto the top-``rank`` PCA basis, standardise per component.

    WHY THIS EXISTS. P1/P1b measured the frozen pooled targets at effective rank ~25 in
    1024 dimensions with mean pairwise cosine ~0.61 — a thin, off-centre, anisotropic
    pencil. The band-0 subspace diagnostic
    (``lab/experiments/results/2026-08-28-tulfm-p1/band0_subspace_diag.py``) then showed
    where the capacity went: rel 1.018 along the ~25 LIVE directions (nothing learned)
    against 0.717 in the dead complement. The planner was spending itself modelling the
    999 directions the targets barely occupy. Rescaling ``sigma_data`` cannot fix that —
    it is one scalar against a geometry.

    After whitening, every retained component has zero mean and unit variance over the
    calibration set. Two consequences the arms are built on:

    * ``sigma_data = 1`` becomes EXACTLY true per component for the EDM arm, so the two
      EDM limits balance: the low-σ floor is ``d = rank`` and the high-σ floor is
      ``E‖y‖²/σ_d² = rank``. Same number. The imbalance P1 documented disappears.
    * ``source_std = 1`` is the matched CFM source scale for the same reason.

    Not an ``nn.Module``: it carries no parameters, is never trained, and must survive a
    checkpoint round-trip as plain tensors so the probe rebuilds exactly what training
    used.
    """

    def __init__(self, mu: Tensor, basis: Tensor, scales: Tensor,
                 calib_slots: int = 0, explained_var: float = 0.0):
        if basis.ndim != 2:
            raise ValueError(f"basis must be [rank, d], got {tuple(basis.shape)}")
        if scales.shape != basis.shape[:1] or mu.shape != basis.shape[1:]:
            raise ValueError(
                f"shape mismatch: mu {tuple(mu.shape)}, basis {tuple(basis.shape)}, "
                f"scales {tuple(scales.shape)}")
        self.mu = mu.float()
        self.basis = basis.float()
        self.scales = scales.float()
        self.calib_slots = int(calib_slots)
        self.explained_var = float(explained_var)

    @property
    def rank(self) -> int:
        return int(self.basis.shape[0])

    @property
    def d_in(self) -> int:
        return int(self.basis.shape[1])

    def to(self, device) -> "TargetWhitener":
        return TargetWhitener(self.mu.to(device), self.basis.to(device),
                              self.scales.to(device), self.calib_slots,
                              self.explained_var)

    def apply(self, y: Tensor, valid: Tensor | None = None) -> Tensor:
        """``[..., d] -> [..., rank]``. Invalid slots come back EXACTLY zero.

        The re-mask is not decoration: whitening subtracts ``mu``, so a padding row that
        was zero would come out as ``-mu·Bᵀ/scales`` and would then be scored by the
        probe and weighted by the loss as if it were a real slot.
        """
        if y.shape[-1] != self.d_in:
            raise ValueError(f"whitener expects last dim {self.d_in}, got {y.shape[-1]}")
        w = ((y.float() - self.mu) @ self.basis.T) / self.scales
        if valid is not None:
            w = w * valid[..., None].float()
        return w

    def invert(self, w: Tensor) -> Tensor:
        """``[..., rank] -> [..., d]`` — the projection of the original onto the basis."""
        if w.shape[-1] != self.rank:
            raise ValueError(f"whitener inverse expects last dim {self.rank}, "
                             f"got {w.shape[-1]}")
        return (w.float() * self.scales) @ self.basis + self.mu

    def state_dict(self) -> dict:
        return {"mu": self.mu.cpu(), "basis": self.basis.cpu(),
                "scales": self.scales.cpu(), "calib_slots": self.calib_slots,
                "explained_var": self.explained_var}

    @staticmethod
    def from_state_dict(sd: dict) -> "TargetWhitener":
        return TargetWhitener(sd["mu"], sd["basis"], sd["scales"],
                              int(sd.get("calib_slots", 0)),
                              float(sd.get("explained_var", 0.0)))

    def manifest(self) -> dict:
        sc = self.scales
        return {
            "whiten/rank": self.rank,
            "whiten/d_in": self.d_in,
            "whiten/calib_slots": self.calib_slots,
            "whiten/explained_var": self.explained_var,
            "whiten/scale_max": float(sc.max()),
            "whiten/scale_min": float(sc.min()),
            "whiten/condition": float(sc.max() / sc.min().clamp_min(1e-30)),
            "whiten/mu_norm": float(self.mu.norm()),
        }


def fit_whitener(rows: Tensor, rank: int) -> TargetWhitener:
    """Fit the whitener on ``[N, d]`` pooled (unit-L2) targets from the frozen backbone.

    PCA by SVD of the centred matrix; ``scales`` are the per-component standard
    deviations, so the whitened calibration set has per-component std exactly 1.

    Raises rather than flooring a degenerate component. A retained direction whose
    variance is ~0 gets divided by ~0, which turns numerical noise into a unit-variance
    "feature" and quietly poisons both the loss and the cosine ranking. If this fires,
    lower ``target.whiten.rank`` — do not add an epsilon.
    """
    if rows.ndim != 2:
        raise ValueError(f"rows must be [N, d], got {tuple(rows.shape)}")
    n, d = rows.shape
    rank = int(rank)
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if rank > min(n - 1, d):
        raise ValueError(
            f"rank {rank} exceeds min(N-1, d) = {min(n - 1, d)} for a calibration set of "
            f"{n} slots in {d} dims — raise target.whiten.calib_batches or lower rank")

    x = rows.double()
    mu = x.mean(0)
    xc = x - mu
    _u, sv, vh = torch.linalg.svd(xc, full_matrices=False)
    var = (sv * sv) / max(n - 1, 1)
    basis = vh[:rank]
    scales = var[:rank].clamp_min(0.0).sqrt()
    explained = float(var[:rank].sum() / var.sum().clamp_min(1e-30))
    cond = float(scales.max() / scales.min().clamp_min(1e-30))
    if not torch.isfinite(scales).all() or cond > 1e6:
        raise ValueError(
            f"whitener is degenerate at rank {rank}: scale condition number {cond:.3e} "
            f"(max {float(scales.max()):.3e}, min {float(scales.min()):.3e}). The "
            f"retained directions include ones the targets do not occupy; lower "
            f"target.whiten.rank.")
    return TargetWhitener(mu.float(), basis.float(), scales.float(),
                          calib_slots=n, explained_var=explained)


def make_targets(h: Tensor, geom: SpanGeometry,
                 whitener: TargetWhitener | None = None) -> tuple[Tensor, Tensor]:
    """``(y_train, y_raw)`` — the targets the objective sees, and the raw pooled ones.

    ONE construction site, shared by the trainer and the probe, because "which space is
    this scored in" is exactly the kind of thing that drifts between two call sites and
    then invalidates a cross-arm comparison.

    ``y_raw`` is always the unit-L2 pooled next-span representation, and the collapse
    guards (:func:`effective_rank`, :func:`mean_pairwise_cos`) are always read off it —
    whitened targets have effective rank ~= ``rank`` and pairwise cosine ~= 0 BY
    CONSTRUCTION, so reading the guard in the whitened space would report health that
    the whitener manufactured.
    """
    y_raw = pool_targets(h, geom)
    if whitener is None:
        return y_raw, y_raw
    return whitener.apply(y_raw, geom.valid), y_raw


def build_masks(geom: SpanGeometry) -> tuple[Tensor, Tensor]:
    """``(self_mask [B, S, S], cross_mask [B, S, L])``, True = MAY attend.

    Returned in the order the layer consumes them (self-attention, then cross-attention)
    so a call site cannot silently swap two same-rank bool tensors.

    * cross: slot ``i`` sees context positions ``j <= e_i`` and NOTHING else. This is the
      anti-leak line. Its target lives at positions ``> e_i`` by construction, so an
      off-by-one here would hand the denoiser its own answer — the DB ``clean_noisy_mask``
      failure mode, one level up.
    * self: causal in SLOT order over VALID slots, so slot ``i`` sees slots ``j <= i``
      (whose own ends satisfy ``e_j <= e_i``, keeping the whole path within ``<= e_i``).

    Padding slots keep a self-loop so ``softmax`` is defined; their outputs are discarded
    by ``geom.valid`` at every consumer. Padding ``slot_end`` is 0, so their cross row
    admits position 0 only — again defined, again discarded.
    """
    B, S = geom.valid.shape
    L = geom.seq_len
    dev = geom.valid.device
    idx = torch.arange(L, device=dev)
    cross = idx[None, None, :] <= geom.slot_end[..., None]              # [B, S, L]

    j = torch.arange(S, device=dev)
    causal = j[None, :, None] >= j[None, None, :]                        # [1, S, S]
    self_mask = causal & geom.valid[:, None, :]                          # keys must be valid
    self_mask = self_mask | torch.eye(S, dtype=torch.bool, device=dev)[None]
    return self_mask.expand(B, S, S), cross


# ── σ schedule helpers ───────────────────────────────────────────────────────

def build_schedule(p_mean: float = -1.2, p_std: float = 1.2,
                   sigma_data: float = 0.5) -> DBSchedule:
    """The B = 1 (Huginn / DB §5.5) schedule over the whole ``[σ_min, σ_max]`` range.

    ``mode='b1'`` makes ``n_blocks == 1``, so ``block_range(0)`` is the FULL range and
    ``sample_sigma`` draws the truncated log-normal restricted to it and renormalised in
    CDF space — exactly the P1 spec's "LogNormal(-1.2, 1.2) clipped to [0.002, 80]", with
    the audited renormalisation instead of a naive clamp (a clamp piles probability mass
    onto the two endpoints; renormalisation does not).

    ``overlap_gamma`` is 0: γ only matters when neighbouring blocks must overlap, and
    B = 1 has no neighbour. The other ``DBConfig`` fields (``conditioning``, ``loss_kind``,
    ``visit``) never reach the schedule; they exist for the DB model path, which P1 does
    not use.
    """
    cfg = DBConfig(mode="b1", overlap_gamma=0.0, sigma_data=float(sigma_data),
                   p_mean=float(p_mean), p_std=float(p_std), loss_kind="l2")
    return DBSchedule(cfg)


def band_edges(schedule: DBSchedule, n_bands: int) -> Tensor:
    """``[n_bands + 1]`` ASCENDING equi-probability σ boundaries.

    ``inference_sigmas(n+1)`` walks CDF space in ``n`` equal steps across the truncated
    log-normal and returns the result descending; flipping it gives the band edges with no
    second implementation of the quantile map (and no reach into a private method).
    """
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}")
    return schedule.inference_sigmas(n_bands + 1).flip(0).contiguous()


def band_of_sigma(sigma: Tensor, edges: Tensor) -> Tensor:
    """Band index in ``[0, n_bands-1]``; band 0 is the LOWEST edge.

    Works for either conditioning variable — σ for the EDM objective, ``t`` for CFM —
    because it is nothing but a bucketize against ascending edges. The name is kept from
    P1 so the existing gate tests keep pointing at the same function.
    """
    n_bands = edges.numel() - 1
    inner = edges[1:-1].to(sigma.device, sigma.dtype)
    return torch.bucketize(sigma, inner, right=True).clamp_(0, n_bands - 1).long()


def band_edges_for(objective: str, schedule: DBSchedule, n_bands: int) -> Tensor:
    """``[n_bands + 1]`` ascending band edges in the ACTIVE objective's own variable.

    * ``edm`` — equi-probability σ quantiles (:func:`band_edges`). Equal probability mass
      per band is what makes the per-band curves comparable to each other.
    * ``cfm`` — ``t`` is drawn ``U(0, 1)``, so equal probability mass IS equal width:
      ``[0, 1/n, …, 1]``. The two objectives therefore both report bands that each hold
      1/n of the training draws, which is what makes their ``rel`` curves readable side
      by side.
    """
    if objective == "cfm":
        return torch.linspace(0.0, 1.0, n_bands + 1)
    return band_edges(schedule, n_bands)


def analytic_null_floor(cfg: "FMPlannerConfig", schedule: DBSchedule, e_y_sq: float,
                        n_mc: int = 200_000, seed: int = 0) -> float:
    """Expected loss of the ZERO network, in closed form. Used by ``loss_scale: auto``.

    This is a derived constant of (objective, σ schedule, target scale, target dim), not
    a tuned number, and it is logged so a run is reproducible from its config alone.

    **CFM.** The zero net predicts ``v̂ = 0`` against ``v = y − x₀`` with
    ``x₀ ~ N(0, s²I)`` independent of ``y``, so

        ``E‖v‖² = E‖y‖² + d·s²``   — constant in ``t``.

    **EDM.** Write the objective in its preconditioned form,
    ``loss = ‖F_θ − F_target‖²`` with ``F_target = (y − c_skip·z)/c_out``. Substituting
    ``z = y + σ·ε`` and the EDM coefficients gives
    ``F_target = (σ·y − σ_d²·ε) / (σ_d·√(σ² + σ_d²))``, hence, taking expectations over
    ``ε`` (zero-mean, independent of ``y``, ``E‖ε‖² = d``):

        ``E‖F_target‖² = (σ²·E‖y‖² + σ_d⁴·d) / (σ_d²·(σ² + σ_d²))``.

    The outer expectation over σ has no elementary closed form for a truncated
    log-normal, so it is a fixed-seed Monte-Carlo mean over ``n_mc`` draws from the SAME
    audited sampler training uses. ``n_mc = 200k`` puts the standard error well under
    1 %, and the seed is pinned, so the number is deterministic.

    ``e_y_sq`` is ``E‖y‖²`` in the space the planner actually trains in: ``1.0`` for raw
    unit-L2 targets (exact, gated by a test) and ``rank`` for whitened targets (exact by
    construction — every kept component has unit variance over the calibration set).
    """
    d = int(cfg.d_target)
    if cfg.objective == "cfm":
        return float(e_y_sq + d * cfg.source_std ** 2)
    sd = float(cfg.sigma_data)
    g = torch.Generator().manual_seed(int(seed))
    s2 = schedule.sample_sigma(0, int(n_mc), torch.device("cpu"),
                               generator=g).double().pow(2)
    return float(((s2 * e_y_sq + sd ** 4 * d) / (sd * sd * (s2 + sd * sd))).mean())


# ── the planner ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FMPlannerConfig:
    """Construction-time planner settings. No runtime feature flags reach the forward."""

    # ── objective ────────────────────────────────────────────────────────
    # "edm"  — the DiffusionBlocks-style preconditioned denoiser of P1/P1b.
    # "cfm"  — TRUE conditional flow matching: x_t = (1-t)·x0 + t·y, regress the
    #          velocity y − x0. No preconditioning, no c_skip/c_out/c_in — identity.
    # The two share EVERY other part: the same body, the same masks, the same targets,
    # the same probe. That is deliberate — it is what makes them one-variable arms.
    objective: str = "edm"

    d_ctx: int = 1024          # frozen backbone d_model — the CONTEXT feature width
    # Width of the TARGET space the planner writes into. 0 -> d_ctx (raw pooled states).
    # With whitening on, this is the retained PCA rank, and z_in / out follow it.
    d_target: int = 0
    d_p: int = 512             # planner width
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048
    cond_dim: int = 256
    # Sinusoidal frequencies feeding the sigma MLP. 0 -> min(128, 2*cond_dim), the
    # largest value the audited SigmaConditioning accepts for this cond_dim.
    sigma_n_freq: int = 0
    max_slots: int = 256
    max_ctx_len: int = 4096
    sigma_data: float = 0.5    # edm only
    # ── cfm only ─────────────────────────────────────────────────────────
    # Std of the CFM source distribution x0 ~ N(0, source_std²·I), in TARGET space.
    # LOAD-BEARING, not cosmetic: DeepWeightFlow App. H found the source variance decides
    # whether a low-capacity velocity net learns anything, because it sets the scale of
    # v = y − x0 that the net must reproduce. Match it to the target's per-component std:
    # 1/sqrt(d) for raw unit-L2 targets (0.031 at d=1024), 1.0 for whitened targets.
    source_std: float = 1.0
    # Multiplier on t before it enters the audited SigmaConditioning basis. 1.0 is the
    # literal spec ("t in place of c_noise"). Worth knowing what that basis does with it:
    # its top frequency is 1 rad/unit, so raw t in (0,1) covers 1/6 of a cycle and the
    # 256-dim basis degenerates to roughly {t, t²} — smooth and usable, but low order.
    # Raising this to ~16 gives a genuine Fourier basis over the unit interval. It is a
    # sweep knob, left at the spec value so the shipped arms differ by one thing each.
    t_embed_scale: float = 1.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.objective not in ("edm", "cfm"):
            raise ValueError(f"objective must be 'edm' or 'cfm', got {self.objective!r}")
        if int(self.d_target) < 0:
            raise ValueError(f"d_target must be >= 0, got {self.d_target}")
        if int(self.d_target) == 0:
            object.__setattr__(self, "d_target", int(self.d_ctx))
        if self.d_p % self.n_heads != 0:
            raise ValueError(f"d_p {self.d_p} not divisible by n_heads {self.n_heads}")
        for name in ("d_ctx", "d_target", "d_p", "n_layers", "n_heads", "d_ff",
                     "cond_dim", "max_slots", "max_ctx_len"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if self.sigma_data <= 0.0:
            raise ValueError(f"sigma_data must be > 0, got {self.sigma_data}")
        if self.source_std <= 0.0:
            raise ValueError(f"source_std must be > 0, got {self.source_std}")
        if self.t_embed_scale <= 0.0:
            raise ValueError(f"t_embed_scale must be > 0, got {self.t_embed_scale}")


def _sinusoid(n_pos: int, dim: int) -> Tensor:
    """Standard fixed sinusoidal position table ``[n_pos, dim]`` (no parameters)."""
    if dim % 2 != 0:
        raise ValueError(f"sinusoid dim must be even, got {dim}")
    pos = torch.arange(n_pos, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32)
                     * (-math.log(10000.0) / dim))[None, :]
    out = torch.zeros(n_pos, dim)
    out[:, 0::2] = torch.sin(pos * freq)
    out[:, 1::2] = torch.cos(pos * freq)
    return out


class _PlannerLayer(nn.Module):
    """Pre-norm: masked slot self-attention → masked cross-attention → MLP.

    Every sub-block is σ-modulated by a zero-init :class:`AdaLNGate`, so at step 0 the
    layer is the un-conditioned pre-norm transformer layer exactly.
    """

    def __init__(self, cfg: FMPlannerConfig):
        super().__init__()
        d, h = cfg.d_p, cfg.n_heads
        self.n_heads = h
        self.d_head = d // h
        self.dropout = float(cfg.dropout)

        self.norm_self = RMSNorm(d)
        self.ada_self = AdaLNGate(cfg.cond_dim, d)
        self.q_self = nn.Linear(d, d, bias=False)
        self.k_self = nn.Linear(d, d, bias=False)
        self.v_self = nn.Linear(d, d, bias=False)
        self.o_self = nn.Linear(d, d, bias=False)

        self.norm_cross = RMSNorm(d)
        self.ada_cross = AdaLNGate(cfg.cond_dim, d)
        self.q_cross = nn.Linear(d, d, bias=False)
        self.k_cross = nn.Linear(d, d, bias=False)
        self.v_cross = nn.Linear(d, d, bias=False)
        self.o_cross = nn.Linear(d, d, bias=False)

        self.norm_mlp = RMSNorm(d)
        self.ada_mlp = AdaLNGate(cfg.cond_dim, d)
        self.mlp_up = nn.Linear(d, 2 * cfg.d_ff, bias=False)   # SwiGLU: gate + value
        self.mlp_down = nn.Linear(cfg.d_ff, d, bias=False)

    def _split(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

    def _merge(self, x: Tensor) -> Tensor:
        B, H, N, dh = x.shape
        return x.transpose(1, 2).reshape(B, N, H * dh)

    def forward(self, h: Tensor, ctx: Tensor, cond: Tensor,
                self_mask: Tensor, cross_mask: Tensor) -> Tensor:
        p = self.dropout if self.training else 0.0

        a = self.ada_self(self.norm_self(h), cond)
        att = F.scaled_dot_product_attention(
            self._split(self.q_self(a)), self._split(self.k_self(a)),
            self._split(self.v_self(a)), attn_mask=self_mask[:, None], dropout_p=p)
        h = h + self.o_self(self._merge(att))

        c = self.ada_cross(self.norm_cross(h), cond)
        att = F.scaled_dot_product_attention(
            self._split(self.q_cross(c)), self._split(self.k_cross(ctx)),
            self._split(self.v_cross(ctx)), attn_mask=cross_mask[:, None], dropout_p=p)
        h = h + self.o_cross(self._merge(att))

        m = self.ada_mlp(self.norm_mlp(h), cond)
        gate, val = self.mlp_up(m).chunk(2, dim=-1)
        return h + self.mlp_down(F.silu(gate) * val)


class FMPlanner(nn.Module):
    """The P1 planner body, in target space ``R^{d_target}``. Two heads, one stack.

    * :meth:`denoise` — EDM-preconditioned, conditioned on ``c_noise(σ)``. Arm ``edm``.
    * :meth:`velocity` — raw velocity, conditioned on ``t``. Arm ``cfm``.

    ONE conditioned stack, shared across every σ (or every ``t``) and every Euler step
    (DB §5.5, the Huginn setting). Training NEVER iterates it: one draw, one pass, one
    loss. Only inference iterates. That asymmetry is the entire point of the arc — it is
    what removes BPTT through an iterated map, and with it the takeover disease.
    """

    def __init__(self, cfg: FMPlannerConfig):
        super().__init__()
        self.cfg = cfg
        self.precond = EDMPrecond(cfg.sigma_data)

        self.ctx_proj = nn.Linear(cfg.d_ctx, cfg.d_p)
        self.ctx_norm = RMSNorm(cfg.d_p)
        self.z_in = nn.Linear(cfg.d_target, cfg.d_p)
        self.slot_idx_embed = nn.Embedding(cfg.max_slots, cfg.d_p)
        nn.init.normal_(self.slot_idx_embed.weight, std=0.02)

        n_freq = int(cfg.sigma_n_freq) or min(128, 2 * cfg.cond_dim)
        self.sigma_cond = SigmaConditioning(cond_dim=cfg.cond_dim, n_freq=n_freq)
        self.layers = nn.ModuleList([_PlannerLayer(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_p)
        self.out = nn.Linear(cfg.d_p, cfg.d_target)
        # DiT AdaLN-Zero discipline: F_theta ≡ 0 at init. For edm that means
        # D̂ = c_skip·z exactly; for cfm it means v̂ = 0 exactly. Either way the untrained
        # control in the probe is a DEFINED floor rather than a lucky draw, and the
        # measured null floor equals the analytic one (gated by a test).
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        self.register_buffer("pos_table", _sinusoid(cfg.max_ctx_len, cfg.d_p),
                             persistent=False)

    # -- conditioning ------------------------------------------------------
    def encode_ctx(self, h_ctx: Tensor) -> Tensor:
        """``[B, L, d_ctx]`` frozen states → ``[B, L, d_p]`` keys/values.

        ``detach()`` here is belt-and-braces: the backbone is already frozen and the
        features already arrive under ``no_grad``. It costs nothing and it makes "no
        gradient reaches the backbone" a property of THIS function rather than of the
        caller's discipline.
        """
        L = h_ctx.shape[1]
        if L > self.pos_table.shape[0]:
            raise ValueError(
                f"context length {L} exceeds max_ctx_len {self.pos_table.shape[0]}")
        x = self.ctx_proj(h_ctx.detach().to(self.ctx_proj.weight.dtype))
        x = x + self.pos_table[:L].to(x.dtype)[None]
        return self.ctx_norm(x)

    def _body(self, z_scaled: Tensor, c_noise: Tensor, ctx: Tensor,
              geom: SpanGeometry, self_mask: Tensor, cross_mask: Tensor) -> Tensor:
        B, S, _ = z_scaled.shape
        cond = self.sigma_cond(c_noise.reshape(-1)).reshape(B, S, self.cfg.cond_dim)

        h = self.z_in(z_scaled)
        h = h + self.slot_idx_embed.weight[:S].to(h.dtype)[None]
        h = h + self.pos_table[geom.slot_end.clamp_max(self.pos_table.shape[0] - 1)].to(h.dtype)

        for layer in self.layers:
            h = layer(h, ctx, cond, self_mask, cross_mask)
        return self.out(self.final_norm(h))

    def denoise(self, z: Tensor, sigma: Tensor, ctx: Tensor, geom: SpanGeometry,
                self_mask: Tensor, cross_mask: Tensor) -> Tensor:
        """EDM-preconditioned denoiser. ``z``/return ``[B, S, d_ctx]``; ``sigma`` ``[B, S]``.

        ``D̂ = c_skip(σ)·z + c_out(σ)·F_θ(c_in(σ)·z, c_noise(σ), ctx)`` — the audited
        coefficients, unchanged.
        """
        if sigma.shape != z.shape[:2]:
            raise ValueError(f"sigma {tuple(sigma.shape)} must be [B, S] for z "
                             f"{tuple(z.shape)}")
        c_skip, c_out, c_in, c_noise = self.precond.coeffs(sigma)
        f = self._body((z.float() * c_in[..., None]).to(z.dtype), c_noise, ctx, geom,
                       self_mask, cross_mask)
        return (c_skip[..., None] * z.float() + c_out[..., None] * f.float()).to(z.dtype)

    def velocity(self, x_t: Tensor, t: Tensor, ctx: Tensor, geom: SpanGeometry,
                 self_mask: Tensor, cross_mask: Tensor) -> Tensor:
        """CFM velocity head. ``x_t``/return ``[B, S, d_target]``; ``t`` ``[B, S]`` in (0,1).

        NO preconditioning. The spec is explicit: the conditioning input is ``t`` in place
        of ``c_noise`` and the in/out maps are the identity, so the network output IS the
        velocity estimate. ``c_skip``/``c_out``/``c_in`` do not appear anywhere on this
        path — grep for them and you will find them only in :meth:`denoise`.
        """
        if t.shape != x_t.shape[:2]:
            raise ValueError(f"t {tuple(t.shape)} must be [B, S] for x_t "
                             f"{tuple(x_t.shape)}")
        return self._body(x_t, t.float() * self.cfg.t_embed_scale, ctx, geom,
                          self_mask, cross_mask)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── training objective ───────────────────────────────────────────────────────

def fm_loss(planner: FMPlanner, h_ctx: Tensor, geom: SpanGeometry,
            schedule: DBSchedule, *, generator: torch.Generator | None = None,
            edges: Tensor | None = None,
            y: Tensor | None = None,
            loss_scale: float = 1.0) -> tuple[Tensor, dict]:
    """One P1 training step's loss. NO loop, NO BPTT — that is the entire point.

    Dispatches on ``planner.cfg.objective``; both branches share the batch, the geometry,
    the masks, the targets and the reporting, so an arm differs by the objective alone.

    * ``edm`` — ``σ ~ p(σ)`` per slot independently, ``ε ~ N(0, I)``, ``z = y + σ·ε``,
      one denoiser pass, ``loss = mean_valid[ w(σ)·‖D̂ − y‖² ]`` with the audited EDM
      weight. Byte-identical to P1/P1b when ``loss_scale == 1``.
    * ``cfm`` — ``x₀ ~ N(0, s²I)``, ``t ~ U(0,1)`` per slot independently,
      ``x_t = (1−t)·x₀ + t·y``, one velocity pass,
      ``loss = mean_valid[ ‖v̂ − (y − x₀)‖² ]``. No σ weighting: CFM's target is already
      scale-uniform in ``t``, so there is nothing for a ``w`` to equalise.

    ``loss_scale`` divides the returned loss (see :func:`analytic_null_floor`). The
    STATS stay in unscaled units so a band curve, a ``rel``, and a ``null`` mean the same
    number they meant in P1 and P1b.

    Returns ``(loss, stats)``. ``stats`` carries the per-band means. For ``edm`` the high-σ
    bands are the real prediction regime and the low-σ bands are autoencoding; for ``cfm``
    high ``t`` is near the data and low ``t`` near the source. Read ``rel = loss / null``,
    never the raw band loss.
    """
    if y is None:
        y, _ = make_targets(h_ctx, geom)
    if planner.cfg.objective == "cfm":
        return _cfm_loss(planner, h_ctx, geom, generator=generator, edges=edges, y=y,
                         loss_scale=loss_scale)
    return _edm_loss(planner, h_ctx, geom, schedule, generator=generator, edges=edges,
                     y=y, loss_scale=loss_scale)


def _finish(loss_raw: Tensor, per_slot: Tensor, per_slot_null: Tensor,
            cond: Tensor, geom: SpanGeometry, y: Tensor, edges: Tensor | None,
            loss_scale: float, extra: dict) -> tuple[Tensor, dict]:
    """Shared reduction, null floor and per-band reporting for both objectives.

    The null floor is the EMPIRICAL loss of a zero network on the SAME draw — the same
    σ (or t) and the same noise — not the analytic expectation. Two reasons: it makes
    ``rel`` exactly 1.0 at initialisation (the output head is zero-init, so the planner
    IS the zero network), which is a precise testable statement; and it is the same
    estimator both objectives use, so their ``rel`` curves are directly comparable. The
    analytic expectation is logged separately by the trainer and is gated against this
    one by a test.
    """
    vmask = geom.valid.float()
    n_valid = vmask.sum().clamp_min(1.0)
    null_loss = float(((per_slot_null * vmask).sum() / n_valid).item())
    loss = loss_raw / float(loss_scale)

    stats: dict = {
        "n_valid": float(n_valid.item()),
        "y_norm_mean": float(((y.norm(dim=-1) * vmask).sum() / n_valid).item()),
        "y_sq_mean": float(((y.pow(2).sum(-1) * vmask).sum() / n_valid).item()),
        "null_loss": null_loss,
        "loss_unscaled": float(loss_raw.detach().item()),
        "loss_scale": float(loss_scale),
        "rel_loss": float(loss_raw.detach().item()) / max(null_loss, 1e-12),
        **extra,
    }
    if edges is not None:
        band = band_of_sigma(cond, edges)
        n_bands = edges.numel() - 1
        for b in range(n_bands):
            sel = (band == b) & geom.valid
            n = sel.sum()
            if n > 0:
                lb = float(per_slot[sel].mean().item())
                nb = float(per_slot_null[sel].mean().item())
                stats[f"band{b}/loss"] = lb
                stats[f"band{b}/null"] = nb
                stats[f"band{b}/rel"] = lb / max(nb, 1e-12)
            stats[f"band{b}/n"] = float(n.item())
    return loss, stats


def _edm_loss(planner: FMPlanner, h_ctx: Tensor, geom: SpanGeometry,
              schedule: DBSchedule, *, generator, edges, y, loss_scale):
    B, S, _d = y.shape
    dev = y.device
    self_mask, cross_mask = build_masks(geom)

    sigma = schedule.sample_sigma(0, B * S, dev, generator=generator).reshape(B, S)
    eps = torch.randn(y.shape, device=dev, dtype=torch.float32, generator=generator)
    z = y + sigma[..., None] * eps

    ctx = planner.encode_ctx(h_ctx)
    d_hat = planner.denoise(z.to(ctx.dtype), sigma, ctx, geom, self_mask, cross_mask)

    sq = (d_hat.float() - y).pow(2).sum(-1)                 # [B, S] — ‖D̂ − y‖²
    w = planner.precond.weight(sigma)                       # [B, S]
    per_slot = w * sq
    vmask = geom.valid.float()
    loss_raw = (per_slot * vmask).sum() / vmask.sum().clamp_min(1.0)

    with torch.no_grad():
        c_skip_n, _, _, _ = planner.precond.coeffs(sigma)
        per_slot_null = w * (c_skip_n[..., None] * z - y).pow(2).sum(-1)

    return _finish(loss_raw, per_slot, per_slot_null, sigma, geom, y, edges, loss_scale,
                   {"sigma_mean": float(((sigma * vmask).sum()
                                         / vmask.sum().clamp_min(1.0)).item()),
                    "sq_mean": float(((sq * vmask).sum()
                                      / vmask.sum().clamp_min(1.0)).item())})


def _cfm_loss(planner: FMPlanner, h_ctx: Tensor, geom: SpanGeometry, *,
              generator, edges, y, loss_scale):
    """Conditional flow matching, straight-line (rectified) probability path.

    ``x_t = (1−t)·x₀ + t·y`` has ``dx/dt = y − x₀`` exactly, so the regression target is
    a CONSTANT in ``t`` for a given ``(x₀, y)`` pair. That is why inference can integrate
    it with a handful of forward-Euler steps and why the recovery test below is exact
    rather than approximate.
    """
    B, S, _d = y.shape
    dev = y.device
    self_mask, cross_mask = build_masks(geom)

    x0 = torch.randn(y.shape, device=dev, dtype=torch.float32,
                     generator=generator) * float(planner.cfg.source_std)
    t = torch.rand((B, S), device=dev, dtype=torch.float32, generator=generator)
    x_t = (1.0 - t[..., None]) * x0 + t[..., None] * y
    v_target = y - x0

    ctx = planner.encode_ctx(h_ctx)
    v_hat = planner.velocity(x_t.to(ctx.dtype), t, ctx, geom, self_mask, cross_mask)

    per_slot = (v_hat.float() - v_target).pow(2).sum(-1)    # [B, S] — ‖v̂ − v‖²
    vmask = geom.valid.float()
    loss_raw = (per_slot * vmask).sum() / vmask.sum().clamp_min(1.0)

    with torch.no_grad():
        per_slot_null = v_target.pow(2).sum(-1)             # v̂ ≡ 0

    return _finish(loss_raw, per_slot, per_slot_null, t, geom, y, edges, loss_scale,
                   {"t_mean": float(((t * vmask).sum()
                                     / vmask.sum().clamp_min(1.0)).item()),
                    "sq_mean": float(((per_slot * vmask).sum()
                                      / vmask.sum().clamp_min(1.0)).item())})


# ── inference: the Euler ladder ──────────────────────────────────────────────

@torch.no_grad()
def generate_plans(planner: FMPlanner, h_ctx: Tensor, geom: SpanGeometry,
                   schedule: DBSchedule, n_steps: int = 6,
                   generator: torch.Generator | None = None,
                   final_read: bool = True) -> Tensor:
    """``[B, S, d_target]`` generated plans — the ONLY thing the retrieval probe scores.

    Dispatches on the objective. ``T = n_steps`` is a FIXED inference constant in P1;
    there is no loop-depth variation here and none is to be added (Wolfe veto, arc note).

    * ``edm`` — the audited ladder (``morph/inference/db_generate.py``):
      ``z ~ N(0, (1 + σ_max²)·I)``, ``n_steps − 1`` Euler steps down a strictly DESCENDING
      equi-probability σ ladder, then one final read at σ_min.
    * ``cfm`` — ``x(0) ~ N(0, s²·I)``, then ``n_steps`` forward-Euler steps of size
      ``1/n_steps`` on ``dx/dt = v̂(x, t)`` from ``t = 0`` to ``t = 1``. The plan is
      ``x(1)``. ``final_read`` does not apply and is ignored on this path — the last step
      already lands at ``t = 1``.

    The probe is UNCHANGED across both. It scores cosine between whatever this returns and
    the targets in the same space, which is exactly why it is the cross-arm gate.
    """
    if n_steps < 2:
        raise ValueError(f"n_steps must be >= 2, got {n_steps}")
    dev = h_ctx.device
    B, S = geom.valid.shape
    d = planner.cfg.d_target
    self_mask, cross_mask = build_masks(geom)
    ctx = planner.encode_ctx(h_ctx)

    if planner.cfg.objective == "cfm":
        x = torch.randn((B, S, d), device=dev, dtype=torch.float32,
                        generator=generator) * float(planner.cfg.source_std)
        dt = 1.0 / float(n_steps)
        for k in range(n_steps):
            t = torch.full((B, S), k * dt, device=dev, dtype=torch.float32)
            v = planner.velocity(x.to(ctx.dtype), t, ctx, geom, self_mask, cross_mask)
            x = x + dt * v.float()
        return x * geom.valid[..., None].float()

    sigmas = schedule.inference_sigmas(n_steps).to(dev)
    if not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("inference_sigmas must be strictly descending (euler_step reads "
                         "its sign from next_sigma - sigma)")

    z = torch.randn((B, S, d), device=dev, dtype=torch.float32, generator=generator) \
        * float(torch.sqrt(1.0 + sigmas[0] ** 2))

    for i in range(n_steps - 1):
        s = sigmas[i].expand(B, S)
        ns = sigmas[i + 1].expand(B, S)
        d_hat = planner.denoise(z.to(ctx.dtype), s, ctx, geom, self_mask, cross_mask)
        z = euler_step(z.reshape(-1, d), d_hat.float().reshape(-1, d),
                       s.reshape(-1), ns.reshape(-1)).reshape(B, S, d)

    if final_read:
        s = sigmas[-1].expand(B, S)
        z = planner.denoise(z.to(ctx.dtype), s, ctx, geom, self_mask, cross_mask).float()
    return z * geom.valid[..., None].float()


# ── target-health diagnostic ─────────────────────────────────────────────────

def effective_rank(y: Tensor, valid: Tensor) -> float:
    """Participation ratio of the CENTERED target covariance spectrum: ``(Σλ)² / Σλ²``.

    How many directions the targets actually vary along. ``d`` for an isotropic cloud,
    ``0`` for a set that has collapsed to a single point (the centered covariance is then
    identically zero).

    Read it TOGETHER with :func:`mean_pairwise_cos`, never alone. Centering makes this
    number blind to a tight cluster around one dominant direction: 200 copies of one
    vector plus 1 % isotropic jitter still varies along every axis, so the participation
    ratio reads high while the targets are, for any practical purpose, the same vector.
    The cosine number is what catches that case.
    """
    rows = y[valid].float()
    if rows.shape[0] < 2:
        return 0.0
    rows = rows - rows.mean(0, keepdim=True)
    cov = rows.T @ rows / max(rows.shape[0] - 1, 1)
    lam = torch.linalg.eigvalsh(cov.double()).clamp_min(0.0)
    s1, s2 = lam.sum(), (lam * lam).sum()
    if float(s2) <= 0.0:
        return 0.0
    return float((s1 * s1 / s2).item())


def mean_pairwise_cos(y: Tensor, valid: Tensor) -> float:
    """Mean cosine between DISTINCT target pairs — the second half of the collapse guard.

    The TUL campaign already measures slot states this way (``base.yaml``: "mean pairwise
    cosine of +0.39 to +0.71 … near-parallel by construction"), so P1 reports the same
    quantity on the targets and the two are directly comparable. 0 for an isotropic
    cloud, 1 for a collapsed one.
    """
    rows = y[valid].float()
    n = rows.shape[0]
    if n < 2:
        return 0.0
    rows = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = rows @ rows.T
    off = sim.sum() - sim.diagonal().sum()
    return float((off / (n * (n - 1))).item())
