"""TUL model pieces — slot parameters and the gather/scatter that the core loops on.

Spec: ``docs/tul-spec.md`` §3.2 (slot input embedding), §3.3 (core on slots only),
§3.4 (coda, token-state dropout, prefix projections), §5 (losses), §7.2 (metrics).
The forward that uses these lives in :mod:`morph.model.transformer`; this module
holds the parameters and the pure tensor plumbing so ``transformer.py`` stays
readable and every piece is unit-testable on its own.

Nothing here branches on a runtime flag: :class:`TULConfig` is resolved at
construction (spec §8), and ``slot_layout=None`` never reaches this module at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .tul_layout import SlotLayout

__all__ = ["TULConfig", "TULGate", "TULGateConfig", "TULSlots", "bag_mean",
           "compact_index", "cw2_retain_mask", "gather_positions", "scatter_positions",
           "window_drop_mask"]


@dataclass
class TULGateConfig:
    """Construction-time settings of the span-length gate (docs/tul-gate-spec.md).

    Present ⇒ :class:`TULGate` is built and the layout must carry ``span_len`` /
    ``len_supervised``. Absent (``TULConfig.gate is None``) ⇒ nothing is built, no
    parameter exists, and the arm is arm A1 (spec §9 invariant 1).

    Args:
        k_max:        the regression DENOMINATOR: the head predicts ``span_len / k_max``
                      and decodes ``k = round(g · k_max)``. It is deliberately allowed to
                      exceed ``span_cap``, and the arms set it to ``1.25 × span_cap``.
                      **Why:** measured on OpenWebText at ``span_cap`` 32, **24.5 %** of
                      labels are a span of exactly 32, so with ``k_max = span_cap`` a
                      quarter of the training signal sits on the target ``g = 1.0`` —
                      an asymptote a sigmoid reaches only in the limit, with a gradient
                      that vanishes as it approaches. Headroom moves the largest target
                      to 0.8 (logit 1.39) and collapses the q10…q90 logit spread the
                      audit must cover from 10.90 to 3.33.
        k_decode_max: the largest ``k`` the model may ask for, ``= span_cap``. Without it
                      the head could pick a budget above ``span_cap``, index a budget row
                      no training example ever reaches, and silently condition the coda
                      on a zero vector.
        lam:          ``gate_lambda``, the weight of the length term in the total loss.
                      1.0 is the predecessor's ``lambda_g`` (``coconut/tul/config.py:81``,
                      the setting under which its gate reached p50 9 against gold p50 9).
                      0.0 ⇒ the term is not added and the arm is bit-identical to A1.
        budget_cond:  §5 — add ``budget_embed(span_len)`` to the slot state before the
                      coda. False ⇒ the head's output changes nothing downstream, which
                      is exactly the predecessor's configuration and why its length
                      decision was never a trade-off.
        huber_beta:   the Huber knee. 1.0 = the predecessor's ``delta`` default.
        train_zeros: the ``k = 0`` ("keep thinking") half of the encoding: supervise ``g``
                      toward 0 on every iteration before a slot's last. **Default False,
                      and the reason is arithmetic, not taste.** The Poisson depth is
                      independent of the input, so no head can know which iteration is the
                      last one; the Bayes-optimal output at iteration ``t`` is then the
                      HAZARD times the mean target, not the length. At ``mean_depth`` 6
                      that is ``0.29 × 0.45 = 0.13`` at ``t = 5`` — the iteration a
                      fixed-depth generation reads — i.e. ``k = 5`` against a true span of
                      19. Measured on the 5090 at step 40 and step 120: ``k = 5.00`` and
                      ``5.68`` against gold ``18.98`` / ``19.58``, matching the predicted
                      table row for row. With the zeros off, the length is regressed at
                      every iteration and the prediction is unbiased at any depth. The
                      stop decision belongs in the separate head of §12, not multiplexed
                      onto the same scalar.
        drives_depth: §7 — the gate chooses the loop depth AT GENERATION/EVAL (arm
                      ``TUL-halt``). Never affects training: §4 teacher-forces the depth,
                      so ``TUL-gate`` and ``TUL-halt`` are ONE training run scored twice.
    """

    k_max: int = 32
    k_decode_max: int = 0                # 0 → k_max
    train_zeros: bool = False            # see the docstring; the measurement says False
    lam: float = 0.0
    budget_cond: bool = True
    huber_beta: float = 1.0
    drives_depth: bool = False
    # Specified in §12 and NOT built. A silently-ignored key is worse than a missing one.
    scheduled_sampling: float = 0.0
    stop_head: bool = False
    ponder_lambda: float = 0.0

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError(f"tul.gate_k_max must be ≥ 1, got {self.k_max}")
        if self.k_decode_max == 0:
            self.k_decode_max = self.k_max
        if not 1 <= self.k_decode_max <= self.k_max:
            raise ValueError(
                f"tul.gate_k_decode_max must be in [1, gate_k_max={self.k_max}], "
                f"got {self.k_decode_max}")
        if self.lam < 0.0:
            raise ValueError(f"tul.gate_lambda must be ≥ 0, got {self.lam}")
        if self.huber_beta <= 0.0:
            raise ValueError(f"tul.gate_huber_beta must be > 0, got {self.huber_beta}")
        for name, key in (("scheduled_sampling", "gate_scheduled_sampling"),
                          ("ponder_lambda", "gate_ponder_lambda")):
            if float(getattr(self, name)) != 0.0:
                raise NotImplementedError(
                    f"tul.{key}={getattr(self, name)} — specified in "
                    f"docs/tul-gate-spec.md §12 and NOT implemented. Set it to 0.0.")
        if self.stop_head:
            raise NotImplementedError(
                "tul.gate_stop_head — the split stop/length encoding of "
                "docs/tul-gate-spec.md §7 is specified and NOT implemented. Leave it false.")


@dataclass
class TULConfig:
    """Construction-time TUL settings (spec §8). Mirrors the Hydra ``tul:`` block.

    Only the keys that change PARAMETERS or SHAPES live here; the schedule keys
    (``activate_at``) and the segmentation keys (``min_span``, ``span_cap``,
    ``fixed_stride``, ``boundary_chars``) belong to the loader and never reach the
    model. ``coda_sees_slots`` and ``tokens_through_core`` are construction-time
    by spec §8 — they change masks and gathers, not an ``if`` in the hot loop.
    """

    prefix_k: int = 2                    # coda positions per slot [W] (§3.1)
    slot_id: int = 4                     # "<fim_pad>"; its LM-head logit is −inf (§3.1)
    token_state_dropout: float = 0.15    # Bowman word dropout on the coda input (§3.4)
    slot_mean_depth: int = 0             # 0 → cfg.mean_depth
    slot_max_depth: int = 0              # 0 → cfg.max_depth
    coda_sees_slots: bool = True         # A4 sets False (§7.1)
    tokens_through_core: bool = False    # A2 sets True (§7.1)
    stp_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    set_lambda: float = 0.0              # arm (§3.5) — asserted 0 until implemented
    carry: bool = False                  # arm (§3.5)
    xattn: bool = False                  # arm (§3.5)
    bcast: bool = False                  # arm (§3.5)
    gate: "TULGateConfig | None" = None  # docs/tul-gate-spec.md; None = arm A1 (nothing built)
    coda_token_cut: int = 0              # arm CW (docs/tul-compaction-window-spec.md) — drop
                                          # TOKEN positions with row index < C from the coda's
                                          # sequence; every slot stays regardless of its index.
                                          # 0 = off, bit-identical (no new tensors, no new ops).

    def __post_init__(self) -> None:
        if self.prefix_k < 1:
            raise ValueError(f"tul.prefix_k must be ≥ 1, got {self.prefix_k}")
        # 1.0 is legal and meaningful: it is Bowman 2015's INPUTLESS decoder control
        # (every token state replaced by E_mask), the extreme end of the §3.4 arm sweep.
        if not 0.0 <= self.token_state_dropout <= 1.0:
            raise ValueError(
                f"tul.token_state_dropout must be in [0,1], got {self.token_state_dropout}")
        if self.coda_token_cut < 0:
            raise ValueError(
                f"tul.coda_token_cut must be ≥ 0, got {self.coda_token_cut}")
        # Spec §3.5 lists these as arms that are NOT in v1. A config key that is silently
        # ignored is worse than a missing one — fail loudly instead (no-theater).
        for name in ("stp_lambda", "set_lambda"):
            if float(getattr(self, name)) != 0.0:
                raise NotImplementedError(
                    f"tul.{name}={getattr(self, name)} — the {name.split('_')[0]} arm "
                    f"(spec §3.5/§5) is specified but NOT implemented in v1. Set it to 0.0; "
                    f"do not run the arm until the loss term exists."
                )
        for name in ("carry", "xattn", "bcast"):
            if bool(getattr(self, name)):
                raise NotImplementedError(
                    f"tul.{name}=true — arm '{name}' (spec §3.5) is specified but NOT "
                    f"implemented in v1. Leave it false."
                )


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


def compact_index(slot_mask: Tensor) -> Tensor:
    """``[B, L]`` gather map that moves TOKEN positions to the front, slots to a dump row.

    Spec §7.2 / arm A4: dropping slots from the coda is done by a gather, "exact, and it
    needs no per-position attention mask the fused kernels may not have". Token order is
    preserved (stable sort), so the compacted sequence is the plain token stream; the
    tail addresses row ``L`` and is discarded by :func:`gather_positions`' dump row.

    Despite the name, the argument is generic: any ``[B, L]`` bool tensor that is True at
    the positions to push to the dump row works (arm CW reuses this with a token-cut mask
    instead of the slot mask, docs/tul-compaction-window-spec.md §"the change").
    """
    B, L = slot_mask.shape
    order = torch.argsort(slot_mask.to(torch.uint8), dim=1, stable=True)
    n_tok = (~slot_mask).sum(dim=1, keepdim=True)
    pos = torch.arange(L, device=slot_mask.device).unsqueeze(0)
    return torch.where(pos < n_tok, order, torch.full_like(order, L))


def window_drop_mask(slot_mask: Tensor, cut: int) -> Tensor:
    """``[B, L]`` bool, True at TOKEN positions with row index ``< cut`` (arm CW's mirror
    of :func:`compact_index`'s slot drop — docs/tul-compaction-window-spec.md).

    Every slot position is False here regardless of its row index — "KEEP every slot
    position, at every index" is the spec's line, and this is a GLOBAL cut (the same
    ``cut`` for every row), not a per-row window: "every query in the coda sees the same
    reduced sequence" is deliberate (spec §"the change" — a per-query window needs a mask,
    which spec §7.2 already rules out for the fused kernels).
    """
    B, L = slot_mask.shape
    pos = torch.arange(L, device=slot_mask.device).unsqueeze(0).expand(B, L)
    return (pos < cut) & (~slot_mask)


def cw2_retain_mask(candidates: Tensor, budget: Tensor, seed: int) -> Tensor:
    """``[B, L]`` bool: a per-row SEEDED random ``budget[row]``-sized subset of ``candidates``.

    Arm CW2's control (docs/tul-compaction-window-spec.md): drop every slot and every
    early token EXCEPT an equal-KV-budget random subset. Vectorised with the same
    argsort + row-count-threshold trick as :func:`compact_index` — draw one uniform score
    per position, push non-candidates off the front with a score that always sorts last,
    then keep the ``budget[row]`` lowest-scored candidates in each row. Reproducible: the
    RNG state used is seeded here and nowhere else, so the same ``seed`` always retains
    the same positions.

    Args:
        candidates: ``[B, L]`` bool — the pool a position may be drawn from (e.g. token
                    positions with row index ``< C``). Never a slot position.
        budget:     ``[B]`` int — how many of each row's candidates to retain. Every
                    candidate sorts strictly before every non-candidate (score ``< 1``
                    vs. exactly ``2.0``), so a candidate's RANK never exceeds its row's
                    candidate count regardless of ``budget`` — a ``budget`` that exceeds
                    the pool is a no-op (retains the whole pool), not an error, with no
                    separate clamp needed: the final ``candidates &`` intersection is
                    what actually enforces it.
        seed:       int, logged by the caller so the eval screen is reproducible.
    """
    B, L = candidates.shape
    gen = torch.Generator(device=candidates.device)
    gen.manual_seed(int(seed))
    scores = torch.rand(B, L, generator=gen, device=candidates.device)
    scores = torch.where(candidates, scores, torch.full_like(scores, 2.0))  # non-candidates sort last
    order = torch.argsort(scores, dim=1, stable=True)
    rank = torch.empty_like(order)
    ar = torch.arange(L, device=candidates.device).unsqueeze(0).expand(B, L)
    rank.scatter_(1, order, ar)
    return candidates & (rank < budget.unsqueeze(1))


# ── parameters ───────────────────────────────────────────────────────────────

class TULSlots(nn.Module):
    """The three TUL parameter groups (spec §3.1/§3.2/§3.4/§5).

    * ``E_slot`` ``[d]`` — the slot token's own embedding, added to the span bag-mean.
      Initialised to the MEAN of the embedding table at the activation step, following
      Block Transformer §3.7's uptraining recipe ("init block embedding = mean of token
      embeddings"), which recovers near-full performance from a vanilla checkpoint with
      ~10 % of the tokens. See :meth:`init_at_activation`.
    * ``E_mask`` ``[d]`` — the learned vector that replaces a dropped token state in the
      coda (Bowman word dropout / He 2019 §3.1 / Optimus "tax the cheap channel"). Init 0.
    * ``W_prefix`` ``[prefix_k, d, d]`` — one looped state ``h_i`` projected into the
      slot's ``prefix_k`` coda positions (spec §3.1; Block Transformer App. F.2 / Fig 3f
      picks prefix length 2 over 1). Init identity, so at the activation step both coda
      positions see ``h_i`` unchanged and the extra position costs nothing.

    Constructed only when TUL is configured, and LAST in ``MORPHTransformer.__init__``
    so a non-TUL model is byte-identical to the baseline (the ``attach_retention``
    convention). All three inits are deterministic — zero RNG draws — so even the TUL
    model's base weights match a baseline built with the same seed.
    """

    def __init__(self, d_model: int, tul: TULConfig):
        super().__init__()
        self.tul = tul
        self.E_slot = nn.Parameter(torch.zeros(d_model))
        self.E_mask = nn.Parameter(torch.zeros(d_model))
        eye = torch.eye(d_model).unsqueeze(0).repeat(tul.prefix_k, 1, 1)
        self.W_prefix = nn.Parameter(eye)

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
        """Replace slot positions of ``signal`` ``[B, L, C]`` with the span bag-mean.

        ``add_e_slot`` is True for the token embedding (the slot's input embedding is
        ``E_slot + mean_j embed(t_j)``, spec §3.2) and False for the bigram / value-embed
        signals, whose slot value is the plain bag-mean of the span ("bigram/value-embed
        signals for the slot are the bag-mean, exactly the TST ``ve_bagged`` path").
        """
        token_sel = (~layout.slot_mask).to(signal.dtype)
        bags = bag_mean(signal, layout.bag_id, token_sel, layout.max_slots)
        at_pos = torch.gather(
            bags, 1, layout.bag_id.unsqueeze(-1).expand(*layout.bag_id.shape, signal.shape[-1]))
        if add_e_slot:
            at_pos = at_pos + self.E_slot.to(signal.dtype)
        return torch.where(layout.slot_mask.unsqueeze(-1), at_pos, signal)

    def prefix_project(self, h_slots: Tensor, layout: SlotLayout, l_total: int) -> Tensor:
        """``[B, S, …, C]`` looped states → ``[B, S·prefix_k]`` values and their positions.

        Returns ``(values, index)`` ready for :func:`scatter_positions`: value ``k`` of
        slot ``s`` is ``h_s W_k`` and lands at ``slot_index[s] + k``. Invalid slots address
        the dump row. Spec §3.1: the first ``prefix_k − 1`` positions carry the plan with
        NO label, the last one predicts the first token of the next span.
        """
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


class TULGate(nn.Module):
    """The span-length gate: one scalar read off each slot's core state, and the
    budget embedding that tells the coda how many tokens the plan covers.

    docs/tul-gate-spec.md §4 (forward), §5 (why the coda must be told), §6 (loss),
    §9 (invariants), §10 (instruments).

    **Zero RNG draws at construction.** ``nn.Linear`` and ``nn.Embedding`` both draw from
    the global generator in ``reset_parameters``; a model that drew them would advance the
    RNG stream and change every later Poisson depth and dropout mask, so
    ``gate_lambda = 0`` would NOT be bit-identical to arm A1 (§9 invariant 1). The head is
    therefore a rank-1 linear written out as two zero parameters, and the budget table is a
    plain zero ``Parameter`` addressed with ``F.embedding`` — mathematically identical to
    ``nn.Linear(d, 1)`` and ``nn.Embedding(k_max+1, d)``, and deterministic. Constructed
    LAST (after :class:`TULSlots`), the same convention that keeps the TUL parameters from
    perturbing the baseline.

    **What gradient reaches the core.** The readout runs on the core's output OUTSIDE the
    checkpoint / ``no_grad`` block, so it shapes the core state exactly on the iterations
    inside the truncated-BPTT window and is a pure readout on the frozen ones — the SAME
    window the token loss uses. Every iteration still supervises the head itself (``w``,
    ``b``, ``norm.scale``), so no slot's label is silently dead (a depth ≤ ``n_nograd``
    slot would otherwise contribute nothing; that is ~28 % of slots at mean_depth 6).
    """

    def __init__(self, d_model: int, gate: TULGateConfig):
        super().__init__()
        self.gate = gate
        self.norm = RMSNorm(d_model)                       # scale init = ones, no RNG
        self.w = nn.Parameter(torch.zeros(d_model))        # ≡ nn.Linear(d,1).weight
        self.b = nn.Parameter(torch.zeros(1))              # ≡ nn.Linear(d,1).bias
        # Index 0 is the pad slot's budget and stays at zero-init unless a real span of
        # length 0 exists, which the packer forbids (span_len is clamped to ≥ 1).
        self.budget = nn.Parameter(torch.zeros(gate.k_decode_max + 1, d_model))

    # -- readout -----------------------------------------------------------
    def readout(self, h: Tensor) -> Tensor:
        """``[B, S, (n,) C]`` slot state → ``[B, S]`` gate output ``g ∈ (0, 1)``.

        The Hyper-Connection streams are collapsed by the MEAN, the same reduction the
        LM head uses (:meth:`MORPHTransformer._readout`), so the gate reads the same
        representation the rest of the model reads out. The norm is what makes the scalar
        scale-free: the core state is pre-``final_norm`` and its magnitude drifts over
        training, which would otherwise move ``g`` with no change in the length it means.
        """
        z = h.mean(dim=2) if h.dim() == 4 else h
        z = self.norm(z.float())
        return torch.sigmoid((z * self.w).sum(-1) + self.b)

    def choose_k(self, g: Tensor) -> Tensor:
        """``g`` → the integer budget ``round(g · k_max)``, clamped to ``[0, k_decode_max]``.

        ``k = 0`` means "keep thinking" (§1). Callers that need a length — the generator —
        clamp the low end to 1 themselves (§8); this does not, so the halting policy can
        see the zero.
        """
        return (g * self.gate.k_max).round().clamp_(0, self.gate.k_decode_max).long()

    # -- budget conditioning (§5) ------------------------------------------
    def budget_term(self, span_len: Tensor) -> Tensor:
        """``[B, S]`` int64 lengths → ``[B, S, C]`` additive term for the slot state."""
        return F.embedding(span_len.clamp(0, self.gate.k_decode_max), self.budget)

    def apply_budget(self, h_slots: Tensor, span_len: Tensor) -> Tensor:
        """Add the budget embedding to the looped slot states before the coda (§4).

        Broadcast over the Hyper-Connection stream axis, exactly as
        :meth:`MORPHTransformer._apply_injection` broadcasts every other additive signal.
        Zero-initialised, so at step 0 this is an exact no-op and the arm starts as A1.
        """
        if not self.gate.budget_cond:
            return h_slots
        term = self.budget_term(span_len).to(h_slots.dtype)
        if h_slots.dim() == 4:
            term = term.unsqueeze(2)
        return h_slots + term

    # -- bias seating (§10) ------------------------------------------------
    @torch.no_grad()
    def seat_bias(self, span_len: Tensor, valid: Tensor) -> float:
        """Set ``b`` to ``logit(mean span_len / k_max)`` — the corpus base rate (§10).

        The predecessor's gate had to TRAVEL to the base rate and never got there (bias
        −2.00000 → −2.00071 against a required 1.88). Starting there costs one batch of
        arithmetic and removes the failure mode. ``w`` is zero at init, so immediately
        after this call the gate emits the base rate for every slot — the correct
        constant predictor, and the floor that ``gate_separation`` is measured against.

        Returns the seated bias, for the log line and the wandb config.
        """
        sel = valid & (span_len > 0)
        n = sel.sum()
        if n == 0:
            raise ValueError("seat_gate_bias: the batch has no valid slot to seat from")
        mean_len = (span_len * sel).sum().float() / n.float()
        q = (mean_len / self.gate.k_max).clamp(1e-4, 1 - 1e-4)
        self.b.fill_(float(torch.log(q / (1 - q))))
        return float(self.b.item())

    # -- loss + instruments (§6, §10) --------------------------------------
    def loss(self, g_traj: Tensor, depths: Tensor, layout: SlotLayout,
             want_metrics: bool = True) -> dict:
        """``g_traj`` ``[B, S, T]`` + the realised depths → the §6 term and §10 numbers.

        Default target (``train_zeros=False``): ``span_len / k_max`` on EVERY iteration a
        slot is still looping. The head then predicts the length of the span it plans,
        unbiased at whatever depth generation happens to read it.

        ``train_zeros=True`` restores §6's original two-part target — zeros before the
        slot's last iteration, the length on it. Kept as a switch because it is the
        predecessor's shape, but it is not the default: the depth is a Poisson draw the
        head cannot observe, so that target's optimum is the HAZARD, and the length is
        multiplied away (see :class:`TULGateConfig`). Rows whose length came from OUR
        truncation RNG are excluded from the length term either way (§6, §9).
        """
        if layout.span_len is None:
            raise RuntimeError(
                "the TUL gate is built but the layout carries no span_len; the loader was "
                "built without a TulGateSpec (docs/tul-gate-spec.md §3.3).")
        B, S, T = g_traj.shape
        k_max = self.gate.k_max
        t_idx = torch.arange(T, device=g_traj.device).view(1, 1, T)
        last = (depths - 1).unsqueeze(-1)                        # [B, S, 1]
        at_final = t_idx == last
        before = t_idx < last
        valid = layout.slot_valid.unsqueeze(-1)
        sup = layout.len_supervised.unsqueeze(-1) & valid

        alive = t_idx <= last                                    # still looping at t
        tgt_len = (layout.span_len.float() / k_max).unsqueeze(-1)
        if self.gate.train_zeros:
            target = torch.where(at_final, tgt_len, torch.zeros((), device=g_traj.device))
            mask = (sup & at_final) | (valid & before)
        else:
            target = tgt_len.expand_as(g_traj)
            mask = sup & alive
        per = F.smooth_l1_loss(g_traj, target, reduction="none", beta=self.gate.huber_beta)
        denom = mask.sum().clamp(min=1)
        out = {"loss_gate": (per * mask).sum() / denom, "n_gate": denom.float()}
        if not want_metrics:
            return out

        # §10: a gate can sit at a tiny loss and still be dead. These are the numbers
        # that tell the difference, and every one of them exists because the predecessor
        # lost a ladder without it.
        fin_m = (valid & at_final).float()
        bef_m = (valid & before).float()
        g_fin = (g_traj * fin_m).sum() / fin_m.sum().clamp(min=1)
        g_bef = (g_traj * bef_m).sum() / bef_m.sum().clamp(min=1)
        out["gate_g_final"] = g_fin
        out["gate_g_before"] = g_bef
        out["gate_separation"] = g_fin - g_bef            # a dead gate reads ~0 here
        # per-iteration mean g over the slots still looping. Under train_zeros it IS the
        # hazard curve (§7's table is the prediction, this the observation); with the
        # zeros off it should be FLAT at E[span_len]/k_max, and a slope means the slot
        # state's readable length drifts with depth.
        al_v = alive & valid
        out["gate_hazard"] = ((g_traj * al_v).sum(dim=(0, 1))
                              / al_v.sum(dim=(0, 1)).clamp(min=1))        # [T]
        # chosen k vs gold, over the supervised final-iteration slots. Masked with NaN
        # and reduced with the nan-aware ops rather than boolean-indexed: `x[bool_mask]`
        # has a data-dependent output shape, which forces a device→host sync EVERY step.
        pick = sup & (at_final if self.gate.train_zeros else alive)
        nan = torch.tensor(float("nan"), device=g_traj.device)
        k_hat = torch.where(pick, self.choose_k(g_traj).float(), nan)
        gold = torch.where(pick, layout.span_len.unsqueeze(-1).expand_as(g_traj).float(), nan)
        out["gate_k_mean"] = k_hat.nanmean()
        out["gate_k_p50"] = k_hat.nanmedian()
        out["gate_gold_mean"] = gold.nanmean()
        out["gate_gold_p50"] = gold.nanmedian()
        out["gate_k_abs_err"] = (k_hat - gold).abs().nanmean()
        out["gate_k_zero_frac"] = torch.where(pick, (k_hat == 0).float(), nan).nanmean()
        # THE dead-gate number once the zeros are off: a constant predictor scores corr 0
        # however low its loss is, which is precisely the state the predecessor shipped.
        # gate_separation cannot serve that role here — with one target at every iteration
        # it is ~0 BY DESIGN, not by failure.
        kf = torch.where(pick, k_hat, nan)
        gf = torch.where(pick, gold, nan)
        km, gm = kf.nanmean(), gf.nanmean()
        cov = ((kf - km) * (gf - gm)).nanmean()
        sd = (kf - km).pow(2).nanmean().sqrt() * (gf - gm).pow(2).nanmean().sqrt()
        out["gate_k_corr"] = cov / sd.clamp(min=1e-6)
        # …and the SKILL: how much the gate beats the best CONSTANT predictor by, in
        # tokens. The median minimises mean-absolute-error, so `mae_const` is the floor
        # any constant can reach on this batch. Without it `gate_k_abs_err` is unreadable
        # — 8.62 tokens sounds bad and 9.03 is what predicting one number forever gets
        # you, so the whole claim lives in the 0.41 between them. Positive = real.
        out["gate_k_mae_const"] = (gf - gf.nanmedian()).abs().nanmean()
        out["gate_k_skill"] = out["gate_k_mae_const"] - out["gate_k_abs_err"]
        out["gate_bias"] = self.b.detach().squeeze()
        out["gate_w_norm"] = self.w.detach().norm()
        return out
