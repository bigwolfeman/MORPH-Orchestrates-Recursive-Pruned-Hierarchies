"""TUL model pieces — slot parameters and the pure tensor plumbing around them.

Spec: ``docs/tul-spec.md`` §3.2 (slot input embedding), §3.4 (coda, token-state
dropout), §5 (losses), §7.2 (metrics); the shipped arm is the PAID loop of
``docs/tul-paid-loop-recipe.md`` — slots are ordinary positions of ONE sequence and
the core loops over every position. The forward that uses these lives in
:mod:`morph.model.transformer`; this module holds the parameters and the tensor
plumbing so ``transformer.py`` stays readable and every piece is unit-testable.

Nothing here branches on a runtime flag: :class:`TULConfig` is resolved at
construction (spec §8), and ``slot_layout=None`` never reaches this module at all.

2026-09-03: the slot-only loop (arms A0/A1/A3), the span-length gate, the MUX head,
SIGReg, the DB1 one-pass step, the GRT recurrence gate, the compaction-window arm,
arm A4, the TG restriction and the ``e_slot`` / ``content`` / ``bound`` seed modes
were removed after the 20k pair (`lab/experiments/failures/2026-09-02-warmup-20k-pair.md`);
git history before that date has them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .tul_layout import SlotLayout

__all__ = ["TULConfig", "TULSlots", "bag_mean", "boundary_token_index",
           "gather_positions", "gather_valid", "scatter_positions"]

SLOT_SEED_MODES = ("bag_mean", "boundary")


@dataclass
class TULConfig:
    """Construction-time TUL settings (spec §8). Mirrors the Hydra ``tul:`` block.

    Only the keys that change PARAMETERS or SHAPES live here; the schedule key
    (``activate_at``) and the segmentation keys (``min_span``, ``span_cap``,
    ``boundary_chars``) belong to the loader and never reach the model.
    """

    prefix_k: int = 2                    # coda positions per slot [W] (§3.1)
    slot_id: int = 4                     # "<fim_pad>"; its LM-head logit is −inf (§3.1)
    token_state_dropout: float = 0.15    # Bowman word dropout on the coda input (§3.4)
    # ── §5 double-label weights ──────────────────────────────────────────────
    # The layout puts the slot between a span's last token and the next span's first
    # token, so that first token is predicted twice: from ``t_last`` (plain LM) and from
    # the slot's emitting position. The shipped recipe trains ONLY the token position
    # (emit 0.0 / plast 1.0, measured in the write-side and pair campaigns); the emit
    # position stays a METRIC (``ce_emit``) with no gradient.
    emit_weight: float = 0.0             # training weight of the slot's emit position
    plast_weight: float = 1.0            # training weight of the t_last token position
    # ── slot seed (the slot's input embedding) ───────────────────────────────
    #   "boundary" : E_slot + W_sent . embed(t_last), t_last the LAST token of the
    #                span — a seed-level approximation of Thought Gestalt's
    #                boundary tap (arXiv 2512.25026). Builds one bias-free
    #                ``nn.Linear(d, d)`` (``TULSlots.W_sent``). The shipped mode.
    #   "bag_mean" : E_slot + mean_j embed(t_j) over the span (spec §3.2, the
    #                original A0/A1 seed). Builds nothing extra.
    # Bigram / value-embed signals for the slot are the plain span bag-mean in BOTH
    # modes (the TST ``ve_bagged`` path with a data-dependent bag map).
    slot_seed: str = "boundary"

    def __post_init__(self) -> None:
        if self.prefix_k < 1:
            raise ValueError(f"tul.prefix_k must be ≥ 1, got {self.prefix_k}")
        # 1.0 is legal and meaningful: it is Bowman 2015's INPUTLESS decoder control
        # (every token state replaced by E_mask), the extreme end of the §3.4 arm sweep.
        if not 0.0 <= self.token_state_dropout <= 1.0:
            raise ValueError(
                f"tul.token_state_dropout must be in [0,1], got {self.token_state_dropout}")
        if self.emit_weight < 0.0 or self.plast_weight < 0.0:
            raise ValueError("tul.emit_weight / tul.plast_weight must be >= 0")
        if self.slot_seed not in SLOT_SEED_MODES:
            raise ValueError(
                f"tul.slot_seed must be one of {SLOT_SEED_MODES}, got {self.slot_seed!r}")


# ── pure tensor plumbing ─────────────────────────────────────────────────────

def bag_mean(signal: Tensor, bag_id: Tensor, token_sel: Tensor, n_bags: int) -> Tensor:
    """Mean of ``signal`` over the TOKEN positions of each span (spec §3.2).

    This is the TST ``ve_bagged`` operation with a data-dependent bag map instead of
    a fixed stride: the slot's input is the mean of its span's token embeddings
    (Dynamic Token Pooling: mean-pool beats take-last; BLT Eq. 5 uses the mean as the
    pooling query init). Gradient flows to the embedding table through the scatter.

    Args:
        signal:    ``[B, L, C]`` per-position signal (token embedding, bigram, value-embed).
        bag_id:    ``[B, L]`` int64 bag index; ``n_bags`` is the dump bin.
        token_sel: ``[B, L]`` float 1.0 at token positions, 0.0 at slot positions —
                   slot positions must not pollute their own bag.
        n_bags:    number of real bags (``max_slots``).

    Returns:
        ``[B, n_bags + 1, C]``; row ``n_bags`` is the dump bin and is exactly 0, so a
        gather at the dump bin contributes nothing (tail pads get ``E_slot`` alone).
    """
    B, L, C = signal.shape
    n_out = n_bags + 1
    sel = token_sel.unsqueeze(-1).to(signal.dtype)
    # A one-hot [B, n_out, L] bag map times the signal. The obvious index_add_ form uses
    # float atomics, so its summation ORDER varies run to run and the result is not
    # bit-reproducible forward OR backward (measured: 20/20 repeats differ, 30.7 % of
    # backward elements, max 3.9e-3 in bf16). A GEMM has a fixed reduction order, agrees
    # with index_add_ to bf16 epsilon, and is ~10 % FASTER here. See
    # .agents/notes/proposed/process/2026-08-23-divergence-root-cause-plan.md task 0.1.
    # scatter_ WRITES (one bag per position) rather than accumulating, so it is exact.
    oh = signal.new_zeros(B, n_out, L)
    oh.scatter_(1, bag_id.unsqueeze(1), 1.0)
    oh = oh * sel.squeeze(-1).unsqueeze(1)
    cnt = oh.sum(dim=2, keepdim=True)
    out = torch.bmm(oh, signal) / cnt.clamp(min=1.0)
    # The dump bin aggregates trailing tokens that have no slot; zero it so the gather
    # at slot positions of tail pads reads 0 rather than a stray span mean.
    out = torch.cat([out[:, :n_bags], out.new_zeros(B, 1, C)], dim=1)
    return out


def boundary_token_index(bag_id: Tensor, token_sel: Tensor, n_bags: int) -> Tensor:
    """Position of the LAST token of each bag — the "boundary token" (``slot_seed="boundary"``).

    Companion to :func:`bag_mean`: same inputs, but instead of averaging the span's
    token signal it locates the single position that terminates it. Vectorized with
    ``scatter_reduce_(reduce="amax")`` over the position index — no Python loop over
    slots (a per-row Python loop over up to ``max_slots`` spans would be the actual
    hot-path cost here; this is one kernel launch regardless of span count).

    Args:
        bag_id:    ``[B, L]`` int64 — see :func:`bag_mean`.
        token_sel: ``[B, L]`` bool/float, 1/True at token positions — see :func:`bag_mean`.
        n_bags:    number of real bags (``max_slots``).

    Returns:
        ``[B, n_bags + 1]`` int64. Row ``s`` is the largest token position ``p`` with
        ``bag_id[p] == s``, or ``-1`` when bag ``s`` owns no token position — a real
        slot index the row never reached, OR the dump bin. The ``-1`` sentinel is the
        pad-slot / dump-bin invariant callers must check before gathering.
    """
    B, L = bag_id.shape
    n_out = n_bags + 1
    pos = torch.arange(L, device=bag_id.device).unsqueeze(0).expand(B, L)
    sel = token_sel.to(torch.bool)
    cand = torch.where(sel, pos, pos.new_full((), -1))
    out = bag_id.new_full((B, n_out), -1)
    out.scatter_reduce_(1, bag_id, cand, reduce="amax", include_self=True)
    # The dump bin (index n_bags) aggregates TOKEN positions past the row's last
    # boundary (bag_mean's tail-pad case) — force it to -1 so the gather at a
    # tail-pad SLOT position (bag_mean's documented invariant: tail pads get
    # E_slot alone) never picks up a stray "boundary" from those leftover tokens.
    out[:, n_bags] = -1
    return out


def gather_positions(x: Tensor, index: Tensor) -> Tensor:
    """Gather along the sequence axis. ``x``: ``[B, L, …]``, ``index``: ``[B, N]`` → ``[B, N, …]``.

    ``index`` may address row ``L`` — the caller is expected to have appended a zero
    dump row, which is how a variable-length compaction keeps a static shape.
    """
    idx = index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])
    return torch.gather(x, 1, idx)


def gather_valid(x: Tensor, index: Tensor, valid: Tensor) -> Tensor:
    """Gather ``[B, N]`` positions, zeroing the rows whose ``valid`` is False.

    Equivalent to appending a zero dump row and pointing invalid entries at it, but
    WITHOUT materialising that copy — the carrier is ``[B, L, n, C]`` fp32 after
    ``input_norm`` (335 MB at the 1024×16 arm shape), so the pad copy is the single
    largest avoidable allocation on the TUL path.
    """
    safe = torch.where(valid, index, torch.zeros_like(index))
    out = gather_positions(x, safe)
    return out * valid.reshape(*valid.shape, *([1] * (x.dim() - 2))).to(out.dtype)


def scatter_positions(x: Tensor, index: Tensor, values: Tensor) -> Tensor:
    """Out-of-place scatter along the sequence axis with a dump row.

    ``x``: ``[B, L, …]``, ``index``: ``[B, N]`` (entries in ``[0, L]``; ``L`` = discard),
    ``values``: ``[B, N, …]``. Returns ``[B, L, …]``. One extra row makes invalid slots
    free of a per-row mask and keeps the shape static.

    The scatter is IN-PLACE on the freshly concatenated buffer: ``cat``'s backward needs
    only to slice the incoming gradient, never its own output, so mutating it is
    autograd-safe and saves a second full-carrier copy.
    """
    B, L = x.shape[0], x.shape[1]
    pad = torch.cat([x, x.new_zeros(B, 1, *x.shape[2:])], dim=1)
    idx = index.reshape(*index.shape, *([1] * (x.dim() - 2))).expand(*index.shape, *x.shape[2:])
    # Cast at the boundary, as every other injection site in the model does: under
    # autocast RMSNorm returns fp32 (its fp32 weight promotes the product) while the
    # prefix projection comes out of a bf16 matmul, and scatter demands one dtype.
    pad.scatter_(1, idx, values.to(pad.dtype))
    return pad[:, :L]


# ── parameters ───────────────────────────────────────────────────────────────

class TULSlots(nn.Module):
    """The TUL parameter groups (spec §3.1/§3.2/§3.4/§5).

    * ``E_slot`` ``[d]`` — the slot token's own embedding, added to the slot's seed.
      Initialised to the MEAN of the embedding table at the activation step, following
      Block Transformer §3.7's uptraining recipe ("init block embedding = mean of token
      embeddings"), which recovers near-full performance from a vanilla checkpoint with
      ~10 % of the tokens. See :meth:`init_at_activation`.
    * ``E_mask`` ``[d]`` — the learned vector that replaces a dropped token state in the
      coda (Bowman word dropout / He 2019 §3.1 / Optimus "tax the cheap channel"). Init 0.
    * ``W_sent`` ``[d, d]``, bias-free — ONLY built when ``tul.slot_seed == "boundary"``.
      Projects the span's boundary token embedding into the slot input. An unused
      Linear still draws weight decay and perturbs the optimizer state, so the
      ``bag_mean`` mode builds nothing. Init ``std=0.02``, matching the rest of the
      model's Linear/Embedding inits, from a PRIVATE generator (see below).
    * ``W_prefix`` ``[prefix_k, d, d]`` — ONLY built with ``with_prefix=True``, which the
      model sets when an FM planner is configured: the planner's plans are projected
      into the slot's ``prefix_k`` coda positions through it (Block Transformer App.
      F.2 / Fig 3f). The paid loop never reads it — the core runs over the slot
      positions themselves — so a plain TUL model does not pay ``prefix_k · d²``
      checkpointed parameters for a dead identity. Init identity.

    Constructed only when TUL is configured, and LAST in ``MORPHTransformer.__init__``
    so a non-TUL model is byte-identical to the baseline (the ``attach_retention``
    convention). Every init here is RNG-NEUTRAL: ``E_slot`` / ``E_mask`` / ``W_prefix``
    are deterministic (zero draws) and ``W_sent`` takes its one real draw from a
    PRIVATE fixed generator, so the global RNG stream is untouched and a TUL model's
    base weights match a baseline built with the same seed.
    """

    def __init__(self, d_model: int, tul: TULConfig, with_prefix: bool = False):
        super().__init__()
        self.tul = tul
        self.E_slot = nn.Parameter(torch.zeros(d_model))
        self.E_mask = nn.Parameter(torch.zeros(d_model))
        self.W_prefix: nn.Parameter | None = None
        if with_prefix:
            eye = torch.eye(d_model).unsqueeze(0).repeat(tul.prefix_k, 1, 1)
            self.W_prefix = nn.Parameter(eye)
        self.W_sent: nn.Linear | None = None
        if tul.slot_seed == "boundary":
            self.W_sent = nn.Linear(d_model, d_model, bias=False)
            # RNG-NEUTRAL init from a FIXED generator. W_sent is the only TUL parameter
            # with no meaningful zero/identity init (unlike E_slot there is no
            # activation-step re-init to rescue a zero start), so it needs a real draw —
            # but taking that draw from the GLOBAL stream would shift every parameter
            # built AFTER TULSlots (`_SCSEInit` draws a Linear init whenever
            # `core_init_scale > 0`). A private generator keeps the base weights
            # byte-identical to a bag_mean build under EVERY config.
            g = torch.Generator(device="cpu").manual_seed(0x5E17)
            with torch.no_grad():
                self.W_sent.weight.copy_(
                    torch.empty(self.W_sent.weight.shape, device="cpu").normal_(
                        mean=0.0, std=0.02, generator=g))

    @torch.no_grad()
    def init_at_activation(self, lm_weight: Tensor) -> None:
        """Set ``E_slot`` to the mean of the (live, trained) embedding table — spec §5.

        Called at the activation step, not at construction: the point of the Block
        Transformer init is that the new position starts as the average TRAINED token,
        which a randomly-initialised table cannot provide. Idempotent-unsafe by design —
        the training loop calls it exactly once and records it in the checkpoint.
        """
        self.E_slot.copy_(lm_weight.mean(dim=0).to(self.E_slot.dtype))

    # -- forward helpers ---------------------------------------------------
    def slot_input(self, signal: Tensor, layout: SlotLayout, add_e_slot: bool) -> Tensor:
        """Replace slot positions of ``signal`` ``[B, L, C]`` with the slot's input.

        ``add_e_slot`` is True for the token embedding and False for the bigram /
        value-embed signals. ``tul.slot_seed`` (construction-time) changes ONLY the
        ``add_e_slot=True`` path — bigram / value-embed signals stay the plain bag-mean
        of the span in both modes, so a caller with ``add_e_slot=False`` always falls
        through to the bag-mean below:

            "boundary" (``add_e_slot=True``): ``E_slot + W_sent . embed(t_last)``,
                        ``t_last`` the span's LAST token position. A slot with no span
                        (a tail-pad position, bag_id at the dump bin) gets ``E_slot``
                        alone — see :func:`boundary_token_index`'s dump-bin handling.
            "bag_mean": ``E_slot + mean_j embed(t_j)`` over the span (spec §3.2).
        """
        token_sel = (~layout.slot_mask).to(signal.dtype)
        e_slot = self.E_slot.to(signal.dtype)

        if add_e_slot and self.tul.slot_seed == "boundary":
            assert self.W_sent is not None    # built iff slot_seed == "boundary" (__init__)
            b_idx = boundary_token_index(layout.bag_id, token_sel, layout.max_slots)
            b_idx_at_pos = torch.gather(b_idx, 1, layout.bag_id)              # [B, L]
            valid = (b_idx_at_pos >= 0).unsqueeze(-1)                         # False: no span
            safe_idx = b_idx_at_pos.clamp(min=0)
            boundary_sig = torch.gather(
                signal, 1, safe_idx.unsqueeze(-1).expand(*safe_idx.shape, signal.shape[-1]))
            proj = self.W_sent(boundary_sig.to(signal.dtype))
            at_pos = torch.where(valid, proj, torch.zeros_like(proj)) + e_slot
            return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

        bags = bag_mean(signal, layout.bag_id, token_sel, layout.max_slots)
        at_pos = torch.gather(
            bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, signal.shape[-1]))
        if add_e_slot:
            at_pos = at_pos + e_slot
        return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

    def prefix_project(self, h_slots: Tensor, layout: SlotLayout, l_total: int) -> Tensor:
        """``[B, S, …, C]`` plan states → ``[B, S·prefix_k]`` values and their positions.

        The FM planner's write path (:meth:`MORPHTransformer._forward_tul`, the
        ``fm_planner`` branch). Returns ``(values, index)`` ready for
        :func:`scatter_positions`: value ``k`` of slot ``s`` is ``h_s W_k`` and lands at
        ``slot_index[s] + k``. Invalid slots address the dump row. Spec §3.1: the first
        ``prefix_k − 1`` positions carry the plan with NO label, the last one predicts
        the first token of the next span.
        """
        if self.W_prefix is None:
            raise RuntimeError(
                "TULSlots.prefix_project needs W_prefix, which is built only for a model "
                "with an FM planner (TULSlots(with_prefix=True)); the paid loop has no "
                "plan to project.")
        K = self.tul.prefix_k
        B, S = layout.slot_index.shape
        C = h_slots.shape[-1]
        mid = h_slots.shape[2:-1]                      # () plain carrier, (n,) HC carrier
        w = self.W_prefix.to(h_slots.dtype)
        # [B,S,M,C] ⊗ [K,C,C] → [B,S,K,M,C] by broadcast matmul (batch dims (B,S,1)×(1,1,K)).
        hm = h_slots.reshape(B, S, -1, C)
        proj = torch.matmul(hm.unsqueeze(2), w.view(1, 1, K, C, C))
        values = proj.reshape(B, S * K, *mid, C)       # slot-major: index s·K + k
        offs = torch.arange(K, device=layout.slot_index.device)
        pos = layout.slot_index.unsqueeze(-1) + offs                      # [B, S, K]
        pos = torch.where(layout.slot_valid.unsqueeze(-1), pos, l_total)
        return values, pos.reshape(B, S * K)

    def apply_token_dropout(self, x: Tensor, layout: SlotLayout, training: bool
                            ) -> tuple[Tensor, Tensor | None]:
        """Replace a fraction ``p`` of TOKEN coda inputs with ``E_mask`` (spec §3.4).

        Returns ``(x, keep)`` where ``keep`` is ``[B, L, 1]`` (1.0 kept, 0.0 dropped) or
        None when nothing was dropped.

        RESOLVED SPEC AMBIGUITY: the drop also zeroes the CODA's x0 / bigram injection at
        the dropped positions. §3.4 only says "the coda input is replaced", but x0 is
        ``proj(embed(t))`` and the bigram term is a hash of ``(t, t−1)`` — both are injected
        into every coda layer, so leaving them would hand the token's own identity straight
        back and make Bowman's word dropout a no-op after the injection scales train up.
        The stated purpose ("the position must then be decoded from the plan slots and its
        neighbours through attention") requires the token to be genuinely absent.
        """
        p = self.tul.token_state_dropout
        if not training or p <= 0.0:
            return x, None
        drop = (torch.rand(layout.slot_mask.shape, device=x.device) < p) & (~layout.slot_mask)
        keep = (~drop).to(x.dtype).unsqueeze(-1)                       # [B, L, 1]
        mask_vec = self.E_mask.to(x.dtype)
        if x.dim() == 4:                       # HC carrier [B, L, n, C]
            x = torch.where(drop[:, :, None, None], mask_vec, x)
        else:
            x = torch.where(drop[:, :, None], mask_vec, x)
        return x, keep
