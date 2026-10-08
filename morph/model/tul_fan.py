"""LXTUL — the FAN: K latent streams per span through the ONE shared core.

``tul.fan_k`` is the empty cell of the 2x2 the 2026-09-18 latent-exploration survey
draws (``docs/references/looping-depth/latent-exploration/2026-09-18-latent-exploration-survey.md``,
"Proposal for the empty cell"): K streams present at TRAINING time, an explicit
diversity term, a learned selector, and an oracle-over-stream instrument. Every other
cell of that 2x2 is already measured on this tree and every one of them is flat.

WHAT IS REUSED AND WHAT IS NEW. The K streams ARE the Thought Register's M cells
(``tul.slot_cells``, 2026-09-13): the same per-cell learned trigger
(:class:`~morph.model.tul.TULSlotRegister`), the same ``slot_cell_relation`` inside the
loop, the same cell-level layout in ``_tul_core``. ``tul.fan_k`` ALIASES ``slot_cells``
in ``morph/training/tul_setup.py`` so none of that is duplicated. What is new is
everything the register did NOT have and the survey says it needed:

  * ``tul.fan_repel_lambda`` / ``tul.fan_repel_passes`` — a cosine repulsion on the
    EARLY passes. PLR's Theorem 4.4 (arXiv 2601.03153) says mean pairwise stream
    distance obeys ``D(T) = L^(2T) D(0)`` under an L-Lipschitz shared map, so a
    contractive core collapses the streams exponentially in depth: repelling at pass 6
    fights a factor ``L^12`` and repelling at pass 1 fights ``L^2``.
    ``tul.row_contrast_lambda`` is NOT this term and must not be reused for it — it
    separates a ROW's slots from each other and at ``slot_cells > 1`` it reads the
    cells' MEAN, so it cannot see the within-slot axis at all.
  * ``tul.fan_mix`` — the exit selector. PLR's ablation makes the gate its LARGEST
    contributor (Recall@20 0.0873 -> 0.0785 without it), larger than the repulsion.
  * the oracle-over-stream instrument (``MORPHTransformer._tul_fan_oracle``), which is
    PLR Figure 4 / Parallel-TTS coverage@N in the units MORPH is scored in.

THE WRITE IS THE RULER'S, NOT THE REGISTER'S, and that is the one deliberate departure
from `slot-register-m4`. The register writes cell i into prefix cell i, 1:1, so a
register arm differs from the strict ruler by the cells AND by a four-times-wider
readout — and the 2026-09-13 reading was that its 0.022 CE win was the WIDTH
(``trajectory-prefix-is-width-plus-pad-artefact``). The fan collapses the K streams to
ONE state HERE, before every reader, and that one state goes through the ordinary
single-source ``TULSlots.prefix_project``. So at ``prefix_k: 4`` a fan arm's coda sees
exactly what the ``prefix_k: 4`` strict ruler's coda sees, and the width confound is
closed by construction rather than by a control run.

PRECISION. :class:`TULFanMix` at ``fan_mix="softmax"`` owns ONE ``d_model -> 1`` linear.
It carries ``_ternary_exclude = True`` for the reason the Hyper-Connection coefficient
projection does (``ternary_qat._categorize``): it is a tiny control path whose output is
a softmax over K, and snapping 1024 weights to {-1, 0, +1} would decide the mixture by
quantisation noise. It is zero-init, so the gate starts EXACTLY uniform and a
``softmax`` arm's step-0 forward equals the ``mean`` control's. The linear draws from a
PRIVATE generator with the global stream snapshotted and restored (the ``W_sent`` /
``TULSlotRegister`` precedent), so a fan model's base weights are byte-identical to its
ruler's.

Record: ``lab/experiments/planned/2026-09-19-lxtul-fan4.md``
Note: ``.agents/notes/proposed/architecture/2026-09-19-lxtul-fan-streams.md``
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from morph.model.host_shadow import (exact_col_mean, exact_col_std, exact_mean_1d,
                                     masked_rows, valid_first)

__all__ = ["TULFanMix", "select_streams", "select_winners", "select_gate_loss", "fan_stream_stats", "fan_stream_cos", "fan_repel_term",
           "FanReservoir", "ridge_map", "epi_score", "fan_epi_term", "fan_vol_term", "plan_streams",
           "logdet_i_wwt", "ridge_map_fast", "fan_epi_term_fast", "fan_vol_term_fast"]


def _cell_readout(cells: Tensor) -> Tensor:
    """``[B, S, M, *carrier, C]`` -> ``[B, S, M, C]``.

    The Hyper-Connection carrier's stream axis is reduced by its MEAN, which is the
    reduction ``MORPHTransformer._readout``, ``TULSlots.unpack`` and the mux head already
    use — one convention for "the state of this cell", so the fan's instruments and the
    fan's gate cannot read a different object from every other reader in the tree.
    """
    while cells.dim() > 4:
        cells = cells.mean(dim=-2)
    return cells


def fan_stream_stats(cells: Tensor) -> tuple[Tensor, Tensor]:
    """``[N, M, C]`` valid slots -> ``(effective_rank, mean_pairwise_cos)``, each ``[N]``.

    Vectorised on the DEVICE in float64, no eigendecomposition, because the first version
    of this reading moved the cells to the CPU and ran one ``eigvalsh`` per (row, slot)
    and idled the GPU 3.7 minutes per val (``slot-register-m4``, 2026-09-13; memory note
    ``eval-probes-must-not-stall-the-gpu``).

    The ``[M, C]`` cells of a slot have a ``[C, C]`` covariance whose NONZERO eigenvalues
    are those of the ``[M, M]`` Gram ``X Xᵀ / (M-1)``, so the participation ratio
    ``(Σλ)² / Σλ²`` is ``tr(G)² / ‖G‖_F²``. The rank is read on CENTERED cells (it is a
    covariance statistic) and the cosine on the RAW ones (a shared offset is exactly what
    it must be able to see) — the same two definitions
    ``MORPHTransformer.slot_eval_probe`` reports as ``val/slot_cell_eff_rank`` and
    ``val/slot_cell_pairwise_cos``, which is why both callers go through this function.
    """
    if cells.dim() != 3:
        raise ValueError(f"fan_stream_stats wants [N, M, C], got {tuple(cells.shape)}")
    n, m, _ = cells.shape
    if m < 2:
        raise ValueError(f"fan_stream_stats needs M >= 2 streams, got {m}")
    x = cells.double()
    xc = x - x.mean(dim=1, keepdim=True)
    g = xc @ xc.transpose(1, 2) / max(m - 1, 1)                     # [N, M, M]
    tr = g.diagonal(dim1=1, dim2=2).sum(-1)
    fro2 = (g * g).sum((1, 2))
    er = torch.where(fro2 > 0, tr * tr / fro2.clamp_min(1e-300), torch.zeros_like(tr))
    gn = F.normalize(x, dim=-1)
    gn = gn @ gn.transpose(1, 2)                                    # [N, M, M]
    cos = (gn.sum((1, 2)) - gn.diagonal(dim1=1, dim2=2).sum(-1)) / float(m * (m - 1))
    return er, cos


def _exact_valid_mean(v: Tensor, n_valid: Tensor) -> Tensor:
    """The gathered ``v[:n_valid].mean()`` bit for bit (``host_shadow.exact_mean_1d``), 0
    when no row is valid (the gathered readers' early return), for ``valid_first`` rows."""
    m = exact_mean_1d(v.float(), n_valid)
    return torch.where(n_valid > 0, m, torch.zeros_like(m)).to(v.dtype)


def fan_stream_cos(state: Tensor, valid: Tensor, m_cells: int,
                   fixed: bool = False) -> Tensor:
    """THE REPULSION TERM: mean pairwise cosine among a slot's K streams, one scalar.

    ``state`` is the compact CELL axis ``[B, S*M, *carrier, C]`` — exactly what
    ``_tul_core`` carries and what every entry of its per-pass trajectory holds — and
    ``valid`` is the PER-SLOT ``[B, S]`` mask. Pad slots are dropped; a batch with no
    valid slot returns an exact 0 on the input's graph, so the caller never branches on
    the count.

    NOTHING IS DETACHED. The term's whole job is to train the streams apart, so every
    stream of every valid slot carries gradient. It reads 1.0 when a slot's streams are
    identical, 0.0 when they are orthogonal, and -1/(M-1) at the simplex floor.

    ``fixed`` (``model.graph_safe``): the valid slots' rows first in the gathered order,
    the rest zeroed, and the gathered mean itself (``host_shadow.valid_first`` /
    ``exact_mean_1d``): bit-identical to ``fixed=False``, no host read, fixed shapes.
    """
    b, sm = state.shape[0], state.shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(
            f"fan_stream_cos: compact axis {sm} != S*M = {s}*{m_cells}")
    z = _cell_readout(state.reshape(b, s, m_cells, *state.shape[2:]))   # [B, S, M, C]
    if fixed:
        sel, n_valid = valid_first(z, valid)                           # [B*S, M, C]
    else:
        sel, n_valid = masked_rows(z, valid), None                     # [N, M, C]
        if sel.shape[0] == 0:
            return z.sum() * 0.0
    n = F.normalize(sel.float(), dim=-1)
    g = n @ n.transpose(1, 2)
    m = float(m_cells)
    cos = (g.sum((1, 2)) - g.diagonal(dim1=1, dim2=2).sum(-1)) / (m * (m - 1))
    return cos.mean() if n_valid is None else _exact_valid_mean(cos, n_valid)


def plan_streams(traj: Tensor, m_cells: int, h: int) -> tuple[Tensor, int]:
    """``tul.fan_history_streams`` — slice a ``_tul_core`` trajectory entry ``[B, S*M,
    *carrier, C]`` to its PLAN streams alone: ``[B, S*(M-h), *carrier, C]``, dropping the
    first ``h`` (HISTORY) cells of every slot. Returns ``(traj_plan, m_plan)`` with
    ``m_plan = M - h``.

    ONE helper because the repulsion terms (:func:`fan_repel_term`, :func:`fan_epi_term`,
    :func:`fan_vol_term`) and their instrument-only calls must slice the SAME way, or a
    charged term and its reported cosine would silently read different cells. The slot
    axis is untouched: this only removes cells from the STREAM axis within each slot, so
    a caller with a per-slot ``valid`` mask (``[B, S]``) passes it through unchanged.
    """
    if h < 1:
        raise ValueError(f"plan_streams: h must be >= 1, got {h}")
    if h >= m_cells:
        raise ValueError(f"plan_streams: h={h} must be < m_cells={m_cells}")
    b, sm = traj.shape[0], traj.shape[1]
    if sm % m_cells != 0:
        raise ValueError(
            f"plan_streams: compact axis {sm} not divisible by m_cells={m_cells}")
    s = sm // m_cells
    m_plan = m_cells - h
    traj_plan = traj.reshape(b, s, m_cells, *traj.shape[2:])[:, :, h:].reshape(
        b, s * m_plan, *traj.shape[2:])
    return traj_plan, m_plan


def fan_repel_term(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int,
                   stats: dict[str, float] | None = None,
                   instruments: bool = True, fixed: bool = False) -> Tensor | None:
    """``tul.fan_repel_lambda``'s raw term, plus the per-pass cosine of EVERY pass.

    ``traj`` is ``_tul_core``'s per-pass trajectory: ``traj[0]`` is the SEED state (the
    streams as the trigger made them, before any pass) and ``traj[t]`` the state after
    pass ``t``. The PENALISED passes are ``1 .. n_passes`` — the early ones, per PLR
    Theorem 4.4 — and they carry gradient. Every other pass, the seed included, is read
    under ``no_grad`` and reported as ``fan_stream_cos_t{t}`` so the collapse SHAPE is
    visible and not only the part the loss touches (the standing
    ``depth-summing-instruments-hide-pass-trades`` rule).

    Returns the MEAN of the penalised passes' cosines, so ``fan_repel_lambda`` keeps its
    meaning whatever the batch's depth draw was, or ``None`` when the batch is shallower
    than one pass.

    ``instruments=False`` skips the no_grad passes (their ``stats`` keys are then absent):
    the trainer sets it on the steps it does not log (``_train_instruments``), and the
    penalised passes, the only ones the loss reads, run either way.

    ``fixed``: :func:`fan_stream_cos`'s ``model.graph_safe`` form.
    """
    if not traj:
        return None
    if n_passes < 1:
        raise ValueError(f"fan_repel_passes must be >= 1, got {n_passes}")
    live: list[Tensor] = []
    for t in range(len(traj)):
        if not instruments and not (1 <= t <= n_passes):
            continue
        if 1 <= t <= n_passes:
            c = fan_stream_cos(traj[t], valid, m_cells, fixed=fixed)
            live.append(c)
        else:
            with torch.no_grad():
                c = fan_stream_cos(traj[t], valid, m_cells, fixed=fixed)
        if stats is not None:
            # Detached 0-dim TENSOR, not `float(...)` — perf: `float()` on a CUDA tensor
            # is a `cudaStreamSynchronize` mid-step, and this fires once per PASS on
            # every training step. The caller (`_forward_tul`) lifts every `stats` entry
            # into the loss `groups` dict as a tensor already; train.py's logger is the
            # one place that calls `float()` on these, on the steps it logs.
            stats[f"stream_cos_t{t}"] = c.detach()
    if not live:
        return None
    if stats is not None:
        stats["repel_terms"] = float(len(live))       # a Python int count, no GPU sync
    return torch.stack(live).mean()


class TULFanMix(nn.Module):
    """``tul.fan_mix`` — how a slot's K streams become the ONE state every reader sees.

    ``"mean"`` is the CONTROL and owns no parameter: the plain average, which is the
    Thought Register's own read (``h_slots = cells.mean(dim=2)``) for the MUX, the span
    decoder, SIGReg and the energy.

    ``"softmax"`` is the arm: ONE shared ``d_model -> 1`` linear scores each stream's
    exit state, a softmax over the K scores gives the per-slot mixture weights, and the
    mixed state is their convex combination. The linear is ZERO-init, so at step 0 every
    logit is 0, the softmax is exactly uniform, and the arm's forward equals the mean
    control's to the last bit — the same "start at the ruler" rule ``TULSlotRegister``'s
    zero-init ``W_o`` follows.

    ``forward`` returns ``(mixed, weights)``: ``mixed`` ``[B, S, *carrier, C]`` and
    ``weights`` ``[B, S, M]``. The caller reports :meth:`entropy` of those weights as
    ``fan/mix_entropy`` — ``ln(K)`` is a gate that has not chosen, 0 a gate that always
    picks one stream.

    ``"all"`` (2026-09-20) owns no parameter either and returns the MEAN with uniform
    weights, exactly as ``"mean"`` does — but it is not a mixture arm: the caller writes
    every stream into its own prefix cell (``prefix_project(cells=...)``) and the mean is
    only what the auxiliary readers of ``h_slots`` (the span decoder, SIGReg, the state
    probe) see. The coda reads all K. Responsibility is the winner-takes-all term the
    caller charges (``MORPHTransformer._tul_fan_all``).
    """

    def __init__(self, d_model: int, k: int, mode: str = "mean"):
        super().__init__()
        if k < 2:
            raise ValueError(f"TULFanMix needs k >= 2 streams, got {k}")
        if mode not in ("mean", "softmax", "select", "all"):
            raise ValueError(
                f"tul.fan_mix must be 'mean', 'softmax', 'select' or 'all', got {mode!r}")
        self.k = int(k)
        self.mode = str(mode)
        self.gate: nn.Linear | None = None
        if mode in ("softmax", "select"):
            # RNG-NEUTRAL CONSTRUCTION (the TULSlotRegister precedent): nn.Linear runs a
            # kaiming draw on the GLOBAL stream before the weight is overwritten, so the
            # stream is snapshotted and restored and no module built after this one is
            # shifted. The weight is then ZEROED, so the gate is uniform at step 0.
            _rng0 = torch.random.get_rng_state()
            self.gate = nn.Linear(d_model, 1, bias=False)
            with torch.no_grad():
                self.gate.weight.zero_()
            # A d_model -> 1 control path, for the reason `_categorize` excludes the
            # Hyper-Connection coefficient projection: ternarising 1024 weights to
            # {-1, 0, +1} would let quantisation noise decide the mixture.
            self.gate._ternary_exclude = True
            torch.random.set_rng_state(_rng0)

    def logits(self, cells: Tensor) -> Tensor:
        """``[B, S, M]`` gate scores (fp32). Only a ``softmax`` / ``select`` mix has them."""
        if self.gate is None:
            raise RuntimeError("TULFanMix at fan_mix='mean' has no gate")
        return self.gate(_cell_readout(cells)).squeeze(-1).float()

    def forward(self, cells: Tensor, choice: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``cells`` ``[B, S, M, *carrier, C]`` -> ``(state, weights)``.

        ``mean`` / ``softmax``: ``state`` is the convex mixture, ``weights`` ``[B, S, M]``.
        ``select``: ``state`` is ONE stream per slot, gathered HARD (its gradient reaches
        that stream alone); ``choice`` ``[B, S]`` int64 names it, or, when ``None``, the
        gate's argmax (the eval write). ``weights`` is still the gate's softmax so
        ``fan/mix_entropy`` keeps its meaning (a gate that has not chosen reads ``ln K``).
        """
        if cells.shape[2] != self.k:
            raise ValueError(
                f"TULFanMix built for k={self.k} got {cells.shape[2]} streams "
                f"({tuple(cells.shape)})")
        b, s, m = cells.shape[:3]
        if self.gate is None:
            w = cells.new_full((b, s, m), 1.0 / float(m))
        else:
            w = torch.softmax(self.logits(cells), dim=-1).to(cells.dtype)
        if self.mode == "select":
            if choice is None:
                choice = w.argmax(dim=-1)
            return select_streams(cells, choice), w
        if choice is not None:
            raise ValueError(f"TULFanMix at fan_mix={self.mode!r} takes no choice")
        shape = (b, s, m, *([1] * (cells.dim() - 4)), 1)
        return (cells * w.view(shape)).sum(dim=2), w

    @staticmethod
    def entropy(weights: Tensor, valid: Tensor, fixed: bool = False) -> Tensor:
        """Mean entropy in NATS of the mixture weights over VALID slots (max ``ln K``).

        ``fixed`` (``model.graph_safe``): the valid slots' rows first, the gathered mean
        itself (``host_shadow.valid_first`` / ``exact_mean_1d``); bit-identical."""
        if fixed:
            w, n_valid = valid_first(weights, valid)
            w = w.float().clamp_min(1e-12)
            return _exact_valid_mean(-(w * w.log()).sum(-1), n_valid)
        w = masked_rows(weights, valid).float().clamp_min(1e-12)
        if w.shape[0] == 0:
            return weights.sum() * 0.0
        return (-(w * w.log()).sum(-1)).mean()


def select_streams(cells: Tensor, choice: Tensor) -> Tensor:
    """``cells`` ``[B, S, M, *carrier, C]``, ``choice`` ``[B, S]`` -> ``[B, S, *carrier, C]``.

    A hard gather: the returned state IS stream ``choice[b, s]`` of slot ``(b, s)`` and its
    gradient flows to that stream alone. ``choice`` is validated against ``M`` so a stale
    index from a different K cannot silently read the wrong stream.
    """
    b, s, m = cells.shape[:3]
    if choice.shape != (b, s):
        raise ValueError(f"select_streams: choice {tuple(choice.shape)} for cells {tuple(cells.shape)}")
    if choice.numel() and (int(choice.min()) < 0 or int(choice.max()) >= m):
        raise ValueError(f"select_streams: choice out of range for M={m}")
    idx = choice.view(b, s, 1, *([1] * (cells.dim() - 3))).expand(b, s, 1, *cells.shape[3:])
    return cells.gather(2, idx).squeeze(2)


def select_winners(span_ce: Tensor, valid: Tensor, eps: float,
                   generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """The per-slot WINNER of ``fan_mix: "select"`` from the K single-stream span CEs.

    ``span_ce`` ``[B, S, K]`` is the summed CE of the span AFTER slot ``s`` when stream
    ``k`` alone is written into every slot (the oracle instrument's own table, built at
    train time under ``no_grad``). Returns ``(choice, forced)``: ``choice`` ``[B, S]`` is the
    argmin, except that with probability ``eps`` a VALID slot is handed a uniformly random
    stream instead (``forced`` marks those) — the winner-takes-all guard: a stream that
    never wins would otherwise never be read, and never being read is why it never wins.
    Invalid slots get 0 (they are never written; the value is inert).
    """
    if span_ce.dim() != 3:
        raise ValueError(f"select_winners wants [B, S, K], got {tuple(span_ce.shape)}")
    b, s, k = span_ce.shape
    choice = span_ce.argmin(dim=-1)
    forced = torch.zeros(b, s, dtype=torch.bool, device=span_ce.device)
    if eps > 0.0:
        u = torch.rand(b, s, device=span_ce.device, generator=generator)
        forced = (u < eps) & valid
        rnd = torch.randint(0, k, (b, s), device=span_ce.device, generator=generator)
        choice = torch.where(forced, rnd, choice)
    choice = torch.where(valid, choice, torch.zeros_like(choice))
    return choice, forced


def select_gate_loss(logits: Tensor, choice: Tensor, valid: Tensor) -> Tensor:
    """Mean CE of the gate's ``[B, S, K]`` logits against the DETACHED winner over VALID
    slots — the gate learns to predict which stream the coda will read best, so that at
    eval its argmax stands in for the K-pass oracle. Exactly 0 gradient into the streams:
    the logits are the gate's own linear on the cell readout, and the loss is on them."""
    if not bool(valid.any()):
        return logits.sum() * 0.0
    lg = logits[valid].float()
    return F.cross_entropy(lg, choice[valid].detach(), reduction="mean")


@torch.no_grad()
def fan_stream_rank(state: Tensor, valid: Tensor, m_cells: int) -> float:
    """Mean effective rank of a slot's K streams, over the VALID slots of the batch.

    ``state`` is one entry of ``_tul_core``'s trajectory, the compact CELL axis
    ``[B, S*M, *carrier, C]``. Bounded above by ``K``; the Thought Register read 1.24 of 4
    at 5000 steps, which is the number this arm exists to move.
    """
    b, sm = state.shape[0], state.shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan_stream_rank: compact axis {sm} != S*M = {s}*{m_cells}")
    z = _cell_readout(state.reshape(b, s, m_cells, *state.shape[2:]))
    sel = z[valid]
    if sel.shape[0] == 0:
        return 0.0
    er, _ = fan_stream_stats(sel)
    return float(er.mean())


# ── tul.fan_repel_mode: "epi" — the epiplexity diversity term ─────────────────────────
# WHY A SECOND TERM. The cosine repulsion has a degenerate optimum and fan4 found it: the
# mean pairwise cosine of K unit vectors is bounded below by -1/(K-1), and a rank-1 split
# of two copies against two negated copies on ONE fixed direction reaches that floor with
# no content (stream probe on slot-spandec-strict-fan4 @ 5000: rank 1.05 of 4, the same
# direction in 96 % of slots, streams 0,1 = +u and 2,3 = -u in 99 % of slots;
# lab/experiments/results/2026-09-19-lxtul-fan4/fan_geom_fan4_5000_d6.json). The
# epiplexity score of EpiJEPA (github.com/the-puzzler/epijepa, from Zhang & Levin,
# "Intelligence from Learnable Novelty", arXiv 2607.18433) is blind to exactly that shape:
#   (1) it centers over the batch, so a constant offset scores nothing;
#   (2) it is a log-det, so magnitude spread over MORE directions scores more;
#   (3) its readout is a ridge map from a FROZEN random function of the INPUT, so only
#       variation that is a linear function of the input counts — not noise, not constants.
# WHAT IS SCORED. Not the streams themselves: their between-stream DEVIATIONS
# d_i = z_i - mean_j z_j, one [N, C] matrix per stream over the batch's valid slots, so
# the term asks that the streams DIFFER in an input-dependent, spread-out way. The
# deviations are divided by the slot's mean stream norm (live, not detached) so the score
# is scale-free and cannot be raised by growing the streams: a looped map with a gain
# constraint must not be handed a term that pays for norm.
# THE INPUT the reservoir sees is the slot's SEED, mean over the K streams of traj[0]
# (the state the trigger made before any pass), DETACHED: the ridge map is computed under
# no_grad exactly as EpiJEPA computes it, so the gradient reaches the streams only.
# THE SCALE. EpiJEPA divides by an S0 measured at init; here the streams are identical
# at step 0 (W_o zero-init), S0 would be 0, so the score is reported and charged in BITS
# PER RESERVOIR FEATURE (S / F), bounded by log2(1 + eta * s^2) where s is the largest
# readout singular value, a few bits at most. `fan_repel_lambda` weights it as it weights
# the cosine. `fan_epi_t{t}` is reported for every pass, the seed included.


class FanReservoir(nn.Module):
    """Frozen random two-layer MLP ``C -> hidden -> F`` with ELU, the epiplexity reservoir.

    Buffers, not parameters: the optimizer never sees it, the ternary walker never wraps
    it (``_ternary_exclude`` as a second guard), and it lives in the checkpoint so a resume
    scores against the same random function. Drawn from a PRIVATE generator with the global
    RNG snapshotted and restored, so a model with the reservoir has the same base weights
    as one without (the ``TULFanMix`` / ``TULSlotRegister`` rule).
    """

    def __init__(self, d_model: int, n_features: int, seed: int = 0, hidden: int = 256):
        super().__init__()
        if n_features < 2:
            raise ValueError(f"tul.fan_epi_features must be >= 2, got {n_features}")
        self._ternary_exclude = True
        self.n_features = int(n_features)
        g = torch.Generator().manual_seed(int(seed))
        w1 = torch.randn(hidden, d_model, generator=g) / math.sqrt(d_model)
        w2 = torch.randn(n_features, hidden, generator=g) / math.sqrt(hidden)
        self.register_buffer("w1", w1)
        self.register_buffer("w2", w2)

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """``[N, C]`` -> ``[N, F]``; LayerNorm on the input so the scale of the seed is
        not the reservoir's business (EpiJEPA's ChannelNorm)."""
        h = F.layer_norm(x.float(), (x.shape[-1],))
        h = F.elu(F.linear(h, self.w1))
        return F.linear(h, self.w2)


def ridge_map(h: Tensor, lam: float, n_valid: Tensor | None = None) -> Tensor:
    """``(H^T H + lam I)^-1 H^T`` on standardised features via QR — ``[F, N]``, double.

    ``h`` is ``[N, F]``: columns are centered, divided by their batch std and by
    ``sqrt(F)``, EpiJEPA's ``ridge_map`` line for line.

    ``n_valid`` (``model.graph_safe``; ``h``'s rows from ``host_shadow.valid_first``, the
    first ``n_valid`` live): the gathered map bit for bit at the full row count. The mean
    and the std are the eager reductions over the live rows (``exact_col_mean`` /
    ``exact_col_std``); the QR input is ``[H_live; sqrt(lam) I; 0]``, whose Householder
    factorisation gives the gathered ``R`` and ``Q`` rows bit for bit (the zero rows trail);
    the solve's right-hand side is ``Q``'s live rows then zero columns. The map's columns
    past ``n_valid`` are exactly 0, so a pad row can never reach a score.
    """
    n, f = h.shape
    if n_valid is not None:
        hs = ((h - exact_col_mean(h, n_valid)) / exact_col_std(h, n_valid).clamp_min(1e-6)
              / math.sqrt(f)).double()
        eye = math.sqrt(lam) * torch.eye(f, dtype=hs.dtype, device=hs.device)
        # Row r of the QR input: live row r, then the ridge rows, then zeros.
        r_idx = torch.arange(n + f, device=hs.device)
        src = torch.where(r_idx < n_valid, r_idx,
                          torch.where(r_idx < n_valid + f, n + (r_idx - n_valid).clamp(max=f - 1),
                                      torch.full_like(r_idx, n + f)))
        aug = torch.cat([hs, eye, hs.new_zeros(1, f)]).index_select(0, src)
        q, r = torch.linalg.qr(aug, mode="reduced")
        live = (r_idx[:n] < n_valid).unsqueeze(1)
        return torch.linalg.solve_triangular(
            r, torch.where(live, q[:n], torch.zeros_like(q[:n])).T, upper=True)
    h = ((h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(f)).double()
    aug = torch.cat([h, math.sqrt(lam) * torch.eye(f, dtype=h.dtype, device=h.device)])
    q, r = torch.linalg.qr(aug, mode="reduced")
    return torch.linalg.solve_triangular(r, q[:n].T, upper=True)


def epi_score(z: Tensor, a: Tensor, eta: float, n_valid: Tensor | None = None) -> Tensor:
    """``0.5 * log2 det(I + eta * W^T W)`` with ``W = a @ (z - mean(z))``, ``[F, C]``.

    Sylvester: ``det(I_C + eta W^T W) = det(I_F + eta W W^T)``, so the ``[F, F]`` side is
    taken and a 1024-wide state costs an F x F slogdet, not a C x C one. Differentiable in ``z``; ``a`` is constant.

    ``n_valid`` (``model.graph_safe``; ``z``'s rows from ``host_shadow.valid_first``): the
    gathered score bit for bit. The mean is the eager one over the live rows
    (``exact_col_mean``); ``a``'s columns past ``n_valid`` are exactly 0 (:func:`ridge_map`),
    so the pad rows add exact zeros to ``W`` and receive zero gradient.
    """
    if n_valid is not None:
        # Two casts, as the eager line: each carries its own gradient back to fp32 before
        # the two are added (one shared cast would add them in fp64 and round once).
        w = a @ (z.double() - exact_col_mean(z.double(), n_valid))
        eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
        return 0.5 * torch.linalg.slogdet(eye + float(eta) * (w @ w.T))[1] / math.log(2)
    w = a @ (z.double() - z.double().mean(0))
    eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)
    return 0.5 * torch.linalg.slogdet(eye + float(eta) * (w @ w.T))[1] / math.log(2)


def _fan_epi_pass(state: Tensor, valid: Tensor, m_cells: int, a: Tensor,
                  eta: float, fixed: bool = False) -> Tensor:
    """Mean over the K streams of the epiplexity of their normalised deviations, in bits
    per reservoir feature, on one trajectory entry.

    ``fixed`` (``model.graph_safe``): the valid slots' rows first (``valid_first``), ``a``
    from :func:`ridge_map`'s exact form: bit-identical to the gathered score."""
    b, sm = state.shape[0], state.shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan_epi_term: compact axis {sm} != S*M = {s}*{m_cells}")
    z = _cell_readout(state.reshape(b, s, m_cells, *state.shape[2:]))   # [B, S, M, C]
    if fixed:
        sel, n_live = valid_first(z, valid)                            # [B*S, M, C]
        sel, rw = sel.float(), n_live.clamp(min=2)
    else:
        sel, rw = masked_rows(z, valid).float(), None                  # [N, M, C]
        n = sel.shape[0]
        if n < 2:
            return z.sum() * 0.0
    dev = sel - sel.mean(dim=1, keepdim=True)                          # [N, M, C]
    scale = sel.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    dev = dev / scale                                                  # scale-free
    f = float(a.shape[0])
    scores = [epi_score(dev[:, i], a, eta, rw) / f for i in range(m_cells)]
    e = torch.stack(scores).mean().to(state.dtype)
    return e if rw is None else torch.where(n_live >= 2, e, torch.zeros_like(e))


def fan_epi_term(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int,
                 reservoir: FanReservoir, ridge: float, eta: float,
                 stats: dict[str, float] | None = None,
                 instruments: bool = True, fixed: bool = False) -> Tensor | None:
    """``tul.fan_repel_mode: "epi"``'s raw term: MINUS the mean epiplexity (bits per
    reservoir feature) of the streams' deviations over passes ``1 .. n_passes``.

    Same contract as :func:`fan_repel_term`: ``traj[0]`` is the seed, passes ``1..n``
    carry gradient, every pass is reported (``fan_epi_t{t}``), ``None`` when the batch is
    shallower than one pass. Minimising the term maximises the score, so
    ``fan_repel_lambda`` keeps its sign and its meaning. ``instruments``: as in
    :func:`fan_repel_term` (the ridge readout ``a`` is fitted either way). ``fixed``:
    the ``model.graph_safe`` form (:func:`_fan_epi_pass`); ``a`` is then always fitted.
    """
    if not traj:
        return None
    if n_passes < 1:
        raise ValueError(f"fan_repel_passes must be >= 1, got {n_passes}")
    b, sm = traj[0].shape[0], traj[0].shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan_epi_term: compact axis {sm} != S*M = {s}*{m_cells}")
    with torch.no_grad():
        seed = _cell_readout(traj[0].reshape(b, s, m_cells, *traj[0].shape[2:]))
        if fixed:
            # Fewer than two valid slots: the gathered path scores 0 (`a = None`); here
            # the map is fitted on two rows (finite) and `_fan_epi_pass` returns 0.
            seed, sw = valid_first(seed.mean(dim=2), valid)            # [B*S, C]
            a = ridge_map(reservoir(seed), ridge, sw.clamp(min=2))     # [F, B*S]
        else:
            seed = masked_rows(seed.mean(dim=2), valid)                # [N, C], detached
            if seed.shape[0] < 2:
                a = None
            else:
                a = ridge_map(reservoir(seed), ridge)                  # [F, N]
    live: list[Tensor] = []
    for t in range(len(traj)):
        if not instruments and not (1 <= t <= n_passes):
            continue
        if a is None:
            e = traj[t].sum() * 0.0
        elif 1 <= t <= n_passes:
            e = _fan_epi_pass(traj[t], valid, m_cells, a, eta, fixed=fixed)
        else:
            with torch.no_grad():
                e = _fan_epi_pass(traj[t], valid, m_cells, a, eta, fixed=fixed)
        if 1 <= t <= n_passes:
            live.append(e)
        if stats is not None:
            # Detached 0-dim TENSOR, not `float(...)` — see the same note in
            # `fan_repel_term` (perf: no host sync on the training path).
            stats[f"epi_t{t}"] = e.detach()
    if not live:
        return None
    if stats is not None:
        stats["repel_terms"] = float(len(live))       # a Python int count, no GPU sync
    return -torch.stack(live).mean()


# ── tul.fan_repel_mode: "vol" and "epivol" — the WITHIN-slot volume ─────────────────────
# WHY A THIRD TERM. The epi term scores each stream's deviation ACROSS slots, so it never
# looks inside a slot. fan4-epi (2026-09-20, step 1000) paid it off with
# d_i(n) = c_i * v(n): one input-dependent direction per slot and fixed per-stream
# scalars. Every stream's deviation is a learnable function of the seed (epi_t1 1.67 bits
# per feature and rising) while the four streams of a slot stay on ONE line
# (fan/stream_rank_t1 1.001 of 4, below fan4's 1.05). The volume term is the within-slot
# reading the epi term lacks: with D̂ the K normalised deviations of a slot, `G = D̂ D̂ᵀ`
# [K, K] has rank <= K-1 (deviations sum to zero) and `0.5 log2 det(I + eta G)` is
# largest when the K-1 nonzero eigenvalues are equal and largest — orthogonal, equal
# deviations — and near zero for a line. Scale-free through the same normalisation.
# `vol` alone can be met by FIXED orthogonal axes (content-free, the cosine's failure one
# rank up); `epivol` sums the two so the spread must also be input-dependent: a constant
# deviation scores 0 on epi, and the epi part can only gain by making the axes vary with
# the seed. Reported per pass as `fan_vol_t{t}` in bits per (K-1).


def _fan_vol_pass(state: Tensor, valid: Tensor, m_cells: int, eta: float,
                  fixed: bool = False) -> Tensor:
    """Mean over valid slots of the within-slot volume of the normalised deviations.

    ``fixed`` (``model.graph_safe``): the valid slots' rows FIRST in the gathered order,
    the rest zeroed (a zero slot has ``G = 0`` and ``log det I = 0``), and the gathered
    mean itself (``host_shadow.exact_mean_1d``). Every per-row op gives each row the
    gathered bits, so the term and its gradient are bit-identical to ``fixed=False``."""
    b, sm = state.shape[0], state.shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan_vol_term: compact axis {sm} != S*M = {s}*{m_cells}")
    z = _cell_readout(state.reshape(b, s, m_cells, *state.shape[2:]))   # [B, S, M, C]
    if fixed:
        sel, n_valid = valid_first(z, valid)                           # [B*S, M, C]
        sel = sel.float()
    else:
        sel, n_valid = masked_rows(z, valid).float(), None             # [N, M, C]
        if sel.shape[0] == 0:
            return z.sum() * 0.0
    dev = sel - sel.mean(dim=1, keepdim=True)
    scale = sel.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    dev = dev / scale
    # Autocast OFF for the Gram: under bf16 autocast the matmul would come back in bf16 and
    # `slogdet` refuses low-precision inputs (the first Spark smoke of `epivol`, 2026-09-20).
    # The epi path is safe because its readout is double, which autocast leaves alone.
    with torch.autocast(device_type=dev.device.type, enabled=False):
        d32 = dev.float()
        g = d32 @ d32.transpose(1, 2)                                  # [N, M, M] fp32
        eye = torch.eye(m_cells, dtype=g.dtype, device=g.device)
        ld = torch.linalg.slogdet(eye + float(eta) * g)[1]             # [N]
    v = 0.5 * ld / math.log(2) / float(m_cells - 1)                   # [N]
    if n_valid is None:
        return v.mean().to(state.dtype)
    return _exact_valid_mean(v, n_valid).to(state.dtype)


def fan_vol_term(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int, eta: float,
                 stats: dict[str, float] | None = None,
                 instruments: bool = True, fixed: bool = False) -> Tensor | None:
    """``"vol"``'s raw term: MINUS the mean within-slot volume over passes ``1..n_passes``.

    Same contract as :func:`fan_repel_term`: passes ``1..n`` carry gradient, every pass is
    reported (``fan_vol_t{t}``), ``None`` when the batch is shallower than one pass.
    ``instruments``: as in :func:`fan_repel_term`. ``fixed``: the ``model.graph_safe``
    form (:func:`_fan_vol_pass`).
    """
    if not traj:
        return None
    if n_passes < 1:
        raise ValueError(f"fan_repel_passes must be >= 1, got {n_passes}")
    live: list[Tensor] = []
    for t in range(len(traj)):
        if not instruments and not (1 <= t <= n_passes):
            continue
        if 1 <= t <= n_passes:
            v = _fan_vol_pass(traj[t], valid, m_cells, eta, fixed=fixed)
            live.append(v)
        else:
            with torch.no_grad():
                v = _fan_vol_pass(traj[t], valid, m_cells, eta, fixed=fixed)
        if stats is not None:
            # Detached 0-dim TENSOR, not `float(...)` — see the same note in
            # `fan_repel_term` (perf: no host sync on the training path).
            stats[f"vol_t{t}"] = v.detach()
    if not live:
        return None
    if stats is not None:
        stats["repel_terms"] = float(len(live))       # a Python int count, no GPU sync
    return -torch.stack(live).mean()


# ── model.fan_div_fast (2026-10-08): the epi and vol terms in fp32, every pass batched ──
# WHY. On the FAST step the two terms cost 7.2 ms for ~0.5 GFLOP (perf/flame/LEDGER.md):
# the epi readout ran in fp64 (the 5090's fp64 rate is 1/64 of fp32), one [F, R] x [R, C]
# product, one Gram and one LU per (pass, stream), and the ridge map as a Householder QR.
# Here every (pass, stream) is ONE batched product and ONE batched Cholesky, in fp32.
# WHY fp32 IS ENOUGH. Every determinant here is of an SPD matrix ``I + eta W Wᵀ`` (its
# eigenvalues are >= 1), and the ridge system ``Hᵀ H + lam I`` on standardised features has
# condition <= 1 + N / (F lam) (~3 at the FAST shape: 384 rows, 64 features, lam 3). The
# fp64 reference test (tests/test_tul_fan_div_fast.py) bounds the term's and every
# gradient's error against the fp64 path, next to the change the bf16 carrier's own
# rounding makes in the fp64 term.
# WHAT IS THE SAME. The functions: the ridge map is (Hᵀ H + lam I)^-1 Hᵀ (what the QR form
# computes), the scores and their per-pass means are the same formulas, rows past the valid
# count are exact zeros as in the ``fixed`` forms, and a batch with fewer than two valid
# slots scores an exact 0. NOT the same: the bits (another precision and another order).


class _LogdetIWWt(torch.autograd.Function):
    """``log det(I + eta W Wᵀ)`` for ``W`` ``[..., K, C]`` fp32, one value per leading index.

    Forward: the ``[K, K]`` matrix and its Cholesky factor ``L`` (``cholesky_ex``: no host
    sync); ``log det = 2 sum log diag L``. Backward: ``d/dW = 2 eta (I + eta W Wᵀ)^-1 W``,
    two batched triangular solves against the saved factor (no inverse is formed)."""

    @staticmethod
    def forward(ctx, w: Tensor, eta: float) -> Tensor:
        k = torch.matmul(w, w.transpose(-1, -2)) * eta
        k.diagonal(dim1=-2, dim2=-1).add_(1.0)
        lo, _ = torch.linalg.cholesky_ex(k)
        ctx.save_for_backward(w, lo)
        ctx.eta = float(eta)
        return 2.0 * lo.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    @staticmethod
    def backward(ctx, g: Tensor):
        w, lo = ctx.saved_tensors
        return (2.0 * ctx.eta) * g[..., None, None] * _chol_solve(lo, w), None


def _chol_solve(lo: Tensor, b: Tensor) -> Tensor:
    """``(L Lᵀ)^-1 B`` by two batched triangular solves (``L`` lower)."""
    y = torch.linalg.solve_triangular(lo, b, upper=False)
    return torch.linalg.solve_triangular(lo.transpose(-1, -2), y, upper=True)


def logdet_i_wwt(w: Tensor, eta: float) -> Tensor:
    """``log det(I + eta W Wᵀ)`` over ``W``'s last two dims, fp32, autocast off."""
    with torch.autocast(device_type=w.device.type, enabled=False):
        return _LogdetIWWt.apply(w.float().contiguous(), float(eta))


def ridge_map_fast(h: Tensor, lam: float, n_fit: Tensor) -> Tensor:
    """:func:`ridge_map`'s ``n_valid`` form in fp32 by the normal equations: ``h`` ``[R, F]``
    with its first ``n_fit`` rows live (a 0-dim device count), the same column centring and
    batch-std scaling over the live rows, then ``(Hᵀ H + lam I)^-1 Hᵀ`` by Cholesky. ``[F, R]``;
    the columns past ``n_fit`` are exactly 0. No graph (the caller holds ``no_grad``)."""
    r, f = h.shape
    with torch.autocast(device_type=h.device.type, enabled=False):
        hf = h.float()
        live = (torch.arange(r, device=h.device) < n_fit).unsqueeze(1)
        n = n_fit.to(torch.float32)
        mean = torch.where(live, hf, torch.zeros_like(hf)).sum(0) / n
        cen = torch.where(live, hf - mean, torch.zeros_like(hf))
        std = (cen.square().sum(0) / n).sqrt()
        hs = cen / std.clamp_min(1e-6) / math.sqrt(f)
        a = hs.t() @ hs
        a.diagonal().add_(float(lam))
        lo, _ = torch.linalg.cholesky_ex(a)
        return _chol_solve(lo, hs.t().contiguous())


def _pass_rows(traj: list[Tensor], ts: list[int], valid: Tensor,
               m_cells: int) -> tuple[Tensor, Tensor]:
    """``(dev [R, P, M, C] fp32, n)``: passes ``ts`` of the trajectory, every slot's K cells
    as their scale-free deviations from the slot mean (the epi and vol terms' common
    input), the valid slots' rows first (``valid_first``'s order), the rest exact 0."""
    b, sm = traj[0].shape[0], traj[0].shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan diversity terms: compact axis {sm} != S*M = {s}*{m_cells}")
    z = torch.stack([_cell_readout(traj[t].reshape(b, s, m_cells, *traj[t].shape[2:]))
                     for t in ts], dim=2)                               # [B, S, P, M, C]
    sel, n = valid_first(z, valid)                                     # [R, P, M, C]
    sel = sel.float()
    dev = sel - sel.mean(dim=2, keepdim=True)
    scale = sel.norm(dim=-1).mean(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-6)
    return dev / scale, n


def _epi_scores_fast(traj: list[Tensor], ts: list[int], valid: Tensor, m_cells: int,
                     a: Tensor, eta: float) -> Tensor:
    """``[P]``: :func:`_fan_epi_pass` of each pass in ``ts``, one batched computation."""
    with torch.autocast(device_type=a.device.type, enabled=False):
        dev, n = _pass_rows(traj, ts, valid, m_cells)                  # [R, P, M, C]
        r, p, m, c = dev.shape
        # No column centring of `dev` (epi_score subtracts the live rows' mean): `a`'s
        # rows are combinations of the standardised features' columns, which are centred
        # over the same live rows, so `a @ 1 = 0` and the mean drops out in exact
        # arithmetic (the uncentred form passes tests/test_tul_fan_div_fast.py's fp64
        # bounds; so did the centred one).
        w = (a @ dev.reshape(r, p * m * c)).view(-1, p, m, c).permute(1, 2, 0, 3)
        f = float(a.shape[0])
        e = (0.5 / math.log(2) / f) * logdet_i_wwt(w, eta).mean(dim=-1)   # [P]
        return torch.where(n >= 2, e, torch.zeros_like(e))


def _vol_scores_fast(traj: list[Tensor], ts: list[int], valid: Tensor, m_cells: int,
                     eta: float) -> Tensor:
    """``[P]``: :func:`_fan_vol_pass` of each pass in ``ts``, one batched computation."""
    with torch.autocast(device_type=traj[0].device.type, enabled=False):
        dev, n = _pass_rows(traj, ts, valid, m_cells)                  # [R, P, M, C]
        v = (0.5 / math.log(2) / float(m_cells - 1)) * logdet_i_wwt(dev, eta)   # [R, P]
        live = (torch.arange(v.shape[0], device=v.device) < n).unsqueeze(1)
        return torch.where(live, v, torch.zeros_like(v)).sum(0) / n.clamp(min=1).to(v.dtype)


def _batched_term(score_fn, traj: list[Tensor], n_passes: int, key: str,
                  stats: dict | None, instruments: bool) -> Tensor | None:
    """The :func:`fan_epi_term` / :func:`fan_vol_term` contract over a batched scorer:
    passes ``1..n_passes`` in ONE call with grad, every other pass (instruments) in ONE
    call under ``no_grad``; ``stats[f"{key}_t{t}"]`` for every scored pass."""
    if not traj:
        return None
    if n_passes < 1:
        raise ValueError(f"fan_repel_passes must be >= 1, got {n_passes}")
    pen = [t for t in range(len(traj)) if 1 <= t <= n_passes]
    other = [t for t in range(len(traj)) if not 1 <= t <= n_passes] if instruments else []
    out: dict[int, Tensor] = {}
    live = score_fn(pen) if pen else None
    if pen:
        out.update({t: live[i] for i, t in enumerate(pen)})
    if other:
        with torch.no_grad():
            ins = score_fn(other)
        out.update({t: ins[i] for i, t in enumerate(other)})
    if stats is not None:
        for t in sorted(out):
            stats[f"{key}_t{t}"] = out[t].detach().to(traj[t].dtype)
    if live is None:
        return None
    if stats is not None:
        stats["repel_terms"] = float(len(pen))
    return -live.mean().to(traj[0].dtype)


def fan_epi_term_fast(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int,
                      reservoir: FanReservoir, ridge: float, eta: float,
                      stats: dict[str, float] | None = None,
                      instruments: bool = True) -> Tensor | None:
    """:func:`fan_epi_term` under ``model.fan_div_fast``: the same term and readings, fp32,
    batched (block comment above). Fixed shapes and no host sync, with or without
    ``model.graph_safe``."""
    if not traj:
        return None
    b, sm = traj[0].shape[0], traj[0].shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(f"fan_epi_term: compact axis {sm} != S*M = {s}*{m_cells}")
    with torch.no_grad():
        seed = _cell_readout(traj[0].reshape(b, s, m_cells, *traj[0].shape[2:]))
        seed, sw = valid_first(seed.mean(dim=2), valid)                # [B*S, C]
        n_fit = sw.clamp(min=2)
        a = ridge_map_fast(reservoir(seed), ridge, n_fit)              # [F, B*S] fp32
    return _batched_term(
        lambda ts: _epi_scores_fast(traj, ts, valid, m_cells, a, eta),
        traj, n_passes, "epi", stats, instruments)


def fan_vol_term_fast(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int,
                      eta: float, stats: dict[str, float] | None = None,
                      instruments: bool = True) -> Tensor | None:
    """:func:`fan_vol_term` under ``model.fan_div_fast``: the same term and readings, every
    pass in one batched Cholesky (block comment above)."""
    return _batched_term(lambda ts: _vol_scores_fast(traj, ts, valid, m_cells, eta),
                         traj, n_passes, "vol", stats, instruments)
