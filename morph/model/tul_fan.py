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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["TULFanMix", "fan_stream_stats", "fan_stream_cos", "fan_repel_term"]


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


def fan_stream_cos(state: Tensor, valid: Tensor, m_cells: int) -> Tensor:
    """THE REPULSION TERM: mean pairwise cosine among a slot's K streams, one scalar.

    ``state`` is the compact CELL axis ``[B, S*M, *carrier, C]`` — exactly what
    ``_tul_core`` carries and what every entry of its per-pass trajectory holds — and
    ``valid`` is the PER-SLOT ``[B, S]`` mask. Pad slots are dropped; a batch with no
    valid slot returns an exact 0 on the input's graph, so the caller never branches on
    the count.

    NOTHING IS DETACHED. The term's whole job is to train the streams apart, so every
    stream of every valid slot carries gradient. It reads 1.0 when a slot's streams are
    identical, 0.0 when they are orthogonal, and -1/(M-1) at the simplex floor.
    """
    b, sm = state.shape[0], state.shape[1]
    s = valid.shape[1]
    if sm != s * m_cells:
        raise ValueError(
            f"fan_stream_cos: compact axis {sm} != S*M = {s}*{m_cells}")
    z = _cell_readout(state.reshape(b, s, m_cells, *state.shape[2:]))   # [B, S, M, C]
    sel = z[valid]                                                     # [N, M, C]
    if sel.shape[0] == 0:
        return z.sum() * 0.0
    n = F.normalize(sel.float(), dim=-1)
    g = n @ n.transpose(1, 2)
    m = float(m_cells)
    return ((g.sum((1, 2)) - g.diagonal(dim1=1, dim2=2).sum(-1)) / (m * (m - 1))).mean()


def fan_repel_term(traj: list[Tensor], valid: Tensor, m_cells: int, n_passes: int,
                   stats: dict[str, float] | None = None) -> Tensor | None:
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
    """
    if not traj:
        return None
    if n_passes < 1:
        raise ValueError(f"fan_repel_passes must be >= 1, got {n_passes}")
    live: list[Tensor] = []
    for t in range(len(traj)):
        if 1 <= t <= n_passes:
            c = fan_stream_cos(traj[t], valid, m_cells)
            live.append(c)
        else:
            with torch.no_grad():
                c = fan_stream_cos(traj[t], valid, m_cells)
        if stats is not None:
            stats[f"stream_cos_t{t}"] = float(c.detach())
    if not live:
        return None
    if stats is not None:
        stats["repel_terms"] = float(len(live))
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
    """

    def __init__(self, d_model: int, k: int, mode: str = "mean"):
        super().__init__()
        if k < 2:
            raise ValueError(f"TULFanMix needs k >= 2 streams, got {k}")
        if mode not in ("mean", "softmax"):
            raise ValueError(f"tul.fan_mix must be 'mean' or 'softmax', got {mode!r}")
        self.k = int(k)
        self.mode = str(mode)
        self.gate: nn.Linear | None = None
        if mode == "softmax":
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

    def forward(self, cells: Tensor) -> tuple[Tensor, Tensor]:
        """``cells`` ``[B, S, M, *carrier, C]`` -> ``(mixed, weights)``."""
        if cells.shape[2] != self.k:
            raise ValueError(
                f"TULFanMix built for k={self.k} got {cells.shape[2]} streams "
                f"({tuple(cells.shape)})")
        b, s, m = cells.shape[:3]
        if self.gate is None:
            w = cells.new_full((b, s, m), 1.0 / float(m))
        else:
            logits = self.gate(_cell_readout(cells)).squeeze(-1)        # [B, S, M]
            w = torch.softmax(logits.float(), dim=-1).to(cells.dtype)
        shape = (b, s, m, *([1] * (cells.dim() - 4)), 1)
        return (cells * w.view(shape)).sum(dim=2), w

    @staticmethod
    def entropy(weights: Tensor, valid: Tensor) -> Tensor:
        """Mean entropy in NATS of the mixture weights over VALID slots (max ``ln K``)."""
        w = weights[valid].float().clamp_min(1e-12)
        if w.shape[0] == 0:
            return weights.sum() * 0.0
        return (-(w * w.log()).sum(-1)).mean()


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
