"""The span decoder: grade a slot's thought by decoding the WHOLE next span from it.

Why this module exists
----------------------
The MUX local loss (``transformer._tul_mux_loss``) reads a slot's exit state ``z``
through the tied head ONCE and scores it against a geometric superposition of the next
span's tokens (``tul.mux_span_targets``, ``rho`` 0.9). That target is **order-free**: it
is a single categorical distribution, so the best ``z`` is the span's weighted unigram
marginal and nothing more. Wolfe, 2026-09-11: "M-next asks for the next span's first
token through the tied head; this is probably 99 % of our problem. We need the whole
span to decode from z, and we are essentially getting like 2 tokens out of it."

The cross-span budget measured the same day
(``lab/experiments/failures/2026-09-11-arc-span-budget.md``) says the upcoming span needs
**0.40 nats** from across its boundary — a flat **0.31** at every offset eight or more
tokens in, plus a 0.96-nat spike at the span's first position. A marginal cannot carry
the flat part. A teacher-forced decoder can be asked to.

What it is
----------
For each slot ``s`` the decoder runs one CAUSAL sequence of length ``J``::

    input   [ z_s , e(t_0) , e(t_1) , ... , e(t_{J-2}) ]
    target  [ t_0 , t_1    , t_2    , ... , t_{J-1}    ]

``t_j`` is the ``j``-th token of span ``s+1`` (the span the slot's plan is decoded into —
``mux_target="next"``'s span, and the same relation :func:`morph.model.tul.mux_span_targets`
supervises). Position ``j`` sees ``z`` and every earlier token of the span, so the loss at
``j`` is the conditional ``-log p(t_j | z, t_{<j})`` and the gradient reaches ``z`` from
EVERY token of the span, not once per span.

There IS a token path. ``.agents/specs/tul-spec.md`` and the root ``CLAUDE.md`` are emphatic:
never decode a span from one vector plus an offset with no token path (Huginn 2026-08-16,
MegaByte T7, Bowman T2, Hourglass T6). The teacher-forced token prefix is that path, and
it is why this is a decoder and not a ``W_bcast``-style offset read.

Contracts
---------
* **Off is nothing.** ``tul.spandec: false`` builds no parameter, draws no RNG and adds no
  term; ``slot_layout=None`` never reaches this module at all.
* **RNG-neutral.** Every real draw comes from a PRIVATE generator (the ``TULSlots.W_sent``
  precedent), so a ``spandec`` arm's base weights are byte-identical to its ruler's.
* **Never quantised, never pruned.** Every ``nn.Linear`` here carries
  ``_ternary_exclude = True`` (honoured by ``ternary_qat._categorize``) and none of them is
  a ``MortarLinear``/``CMSBlockLinear``, so the CMS prune, the MORTAR carve, the ReMoE
  router and the deploy packer walk past by construction. The decoder is a TRAINING-ONLY
  scorer: it is not in the deployed forward, and grading ``z`` through a ternarised reader
  would mix the reader's precision into the target the loop is judged on.
* **The tied head is the OUTPUT head**, read exactly the way the MUX head reads it —
  through ``tul.mux_detach_head``. ``embed.lm_weight()`` IS the input embedding table, so
  an undetached auxiliary head trains the embeddings that every token representation and
  every span target are made of; arm v1a diverged at step 2800 with the detach off
  (``lab/experiments/failures/2026-08-25-mux-head-arm-v1a.md``).
* **The INPUT embeddings are ALWAYS detached**, whatever that knob says, and
  :attr:`SpanDecoder.tok_in` is the learnable map in front of them. The MUX has no
  input-side read of the table, so there is no precedent to inherit, and an undetached one
  would let the decoder reshape the table the slot's own seed is a bag-mean OF.

Two knobs change WHAT the decoder reads and WHAT it decodes (2026-09-13)
------------------------------------------------------------------------
* ``tul.spandec_reads_cells`` — WHAT IT READS. On a register model (``tul.slot_cells: M``)
  the decoder is handed the M cells as a memory and cross-attends to them per layer
  (:class:`_SpanDecCross`), instead of grading only their mean. The mean stays as the
  ``z`` conditioning, the cross output projection is zero-init, and the knob is refused at
  ``M == 1``, so the change is exactly one mechanism.
* ``tul.spandec_target_offset`` — WHAT IT DECODES. Slot ``s`` decodes span ``s + k`` and
  ONLY that span. This is not ``spandec_horizon``: horizon 3 decodes spans s+1, s+2 AND
  s+3 as one concatenated run, so the next span is still in the target; offset 3 removes
  the next span from the target entirely. The 2026-08-14 JEPA screen said the informative
  target is two to three spans downstream, and no arc arm has ever asked for a downstream
  span without also asking for the next one.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .tul_layout import SlotLayout

__all__ = ["PackedCtx", "PackedSpans", "SpanDecoder", "horizon_span_slots", "next_span_slots",
           "own_span_slots", "chunk_segment_mask", "pack_spans", "span_slots"]


class SpanFold(NamedTuple):
    """The compiled forms ``model.glue_fold`` swaps into the decoder (morph/model/glue_fold.py):
    ``rms_cast(x, w, eps)`` is ``RMSNorm(x).to(x.dtype)``, ``rms`` the norm alone, ``silu_mul``
    the SwiGLU product. ``None`` on every decoder that was not folded: the eager ops run."""
    rms_cast: object
    rms: object
    silu_mul: object


def _norm_cast(fold: "SpanFold | None", norm: RMSNorm, x: Tensor) -> Tensor:
    """``norm(x).to(x.dtype)``, through the compiled form when the decoder is folded."""
    if fold is None:
        return norm(x).to(x.dtype)
    return fold.rms_cast(x, norm.weight, norm.eps)


# Private init streams. Two constants, never the global RNG: building the decoder must not
# shift a single weight of the model it is bolted onto (the `TULSlots.W_sent` rule).
_SEED_BLOCKS = 0x5DEC
_SEED_PROJ = 0x5DEC1
_SEED_CROSS = 0x5DEC2


def span_slots(input_ids: Tensor, layout: SlotLayout, max_tokens: int, shift: int = 1
               ) -> tuple[Tensor, Tensor]:
    """``(ids [B, S, J] int64, valid [B, S, J] bool)`` — one span's tokens, per slot.

    ``shift`` picks WHICH span supervises slot ``s`` — ``s + shift`` — using the same
    relations :func:`morph.model.tul.mux_span_targets` uses. ``shift >= 2`` is
    :func:`horizon_span_slots`'s DOWNSTREAM target (``tul.spandec_horizon``): the same
    rule read further ahead, with the same "the span must be complete AND the graded slot
    must exist" validity.

    * ``shift=1`` (:func:`next_span_slots`, ``mux_target="next"``) — slot ``s`` is graded
      on span ``s + 1``, the span its plan is decoded into. A span supervises its
      preceding slot only when BOTH slots exist: slot ``s`` (whose plan is graded) and
      slot ``s + 1`` (whose presence proves the span complete).
    * ``shift=0`` (:func:`own_span_slots`, ``mux_target="own"``) — slot ``s`` is graded on
      span ``s``, the span it terminates. Every one of those tokens sits BEFORE the slot,
      so a generator can compute this target with no lookahead; that causality is why the
      own span is what the gradient-conditioning energies read
      (``morph/model/tul_egrad.py``).

    Written as a per-slot gather because a decoder needs the span's tokens IN ORDER and
    the MUX only needs their weights;
    ``tests/test_tul_spandec.py::test_targets_agree_with_mux_span_targets`` pins the
    ``shift=1`` case to ``mux_span_targets`` token for token and
    ``tests/test_tul_egrad.py`` does the same for ``shift=0``.

    The trailing unterminated text, the dump bin, the span with no slot to grade and every
    pad slot carry ``valid = False`` everywhere, which the caller turns into
    ``ignore_index``.

    Tokens at offset ``>= max_tokens`` inside a span are DROPPED (not folded into the last
    slot): the packer caps a span at ``tul.span_cap`` and ``max_tokens`` defaults to that
    cap, so the drop is empty on the shipped rule and a smaller cap is an explicit choice.
    """
    if shift < 0:
        raise ValueError(f"span_slots shift must be >= 0 (0 = own span, 1 = next span, "
                         f"h = h spans downstream), got {shift}")
    B, L = input_ids.shape
    S = layout.slot_index.shape[1]
    if shift >= S:
        # Slot ``s`` is graded on span ``s + shift``, so a shift at or past the row's slot
        # budget leaves EVERY slot unsupervised and the term is identically zero. Raise
        # rather than train on an empty label set (``tul.spandec_target_offset``).
        raise ValueError(
            f"span_slots shift {shift} is at or past the row's slot budget "
            f"max_slots={S}: slot s is graded on span s+{shift}, so no slot in the row "
            f"would have a target and the term would be identically zero.")
    J = int(max_tokens)
    dev = input_ids.device
    k = layout.bag_id                                            # [B, L] span of a token
    kc = k.clamp(0, S - 1)
    span_done = torch.gather(layout.slot_valid, 1, kc)           # slot k exists
    pos_valid = (~layout.slot_mask) & (k >= shift) & (k < S) & span_done
    if shift >= 1:
        # The graded slot is k-shift, so it must exist too. At shift >= 2 this is the
        # "all H next spans exist" rule of `horizon_span_slots`, applied one span at a
        # time: a slot is supervised at block h only when span s+h is there and complete.
        pos_valid = pos_valid & torch.gather(layout.slot_valid, 1,
                                             (kc - shift).clamp(min=0))
    # Span 0 starts at position 0; span k > 0 starts after slot k-1's prefix cells. The
    # `where` is exact for every shift — at shift >= 1 the k == 0 arm is unreachable
    # (pos_valid already needs k >= shift) and its value is dumped.
    start = torch.where(
        kc >= 1,
        torch.gather(layout.slot_index, 1, (kc - 1).clamp(min=0)) + layout.prefix_k,
        torch.zeros_like(kc))
    # offset inside the span
    j = (torch.arange(L, device=dev).unsqueeze(0) - start).clamp(min=0)
    keep = pos_valid & (j < J)
    # Scatter into [B, S*J]. (slot, offset) is unique per position, so this is a plain
    # `scatter_`, not an atomic add — no nondeterminism and no gradient (ids are int64).
    tgt = (kc - shift).clamp(min=0)                              # the slot being graded
    flat = (tgt * J + j.clamp(max=J - 1))
    flat = torch.where(keep, flat, torch.full_like(flat, S * J))  # dump column
    ids = input_ids.new_zeros(B, S * J + 1)
    ids.scatter_(1, flat, input_ids)
    ok = torch.zeros(B, S * J + 1, dtype=torch.bool, device=dev)
    ok.scatter_(1, flat, keep)
    return ids[:, :S * J].reshape(B, S, J), ok[:, :S * J].reshape(B, S, J)


def next_span_slots(input_ids: Tensor, layout: SlotLayout, max_tokens: int
                    ) -> tuple[Tensor, Tensor]:
    """Span ``s + 1``'s tokens, per slot. :func:`span_slots` at ``shift=1``."""
    return span_slots(input_ids, layout, max_tokens, shift=1)


def own_span_slots(input_ids: Tensor, layout: SlotLayout, max_tokens: int
                   ) -> tuple[Tensor, Tensor]:
    """Span ``s``'s own tokens, per slot. :func:`span_slots` at ``shift=0``."""
    return span_slots(input_ids, layout, max_tokens, shift=0)


def horizon_span_slots(input_ids: Tensor, layout: SlotLayout, per_span_tokens: int,
                       horizon: int, start: int = 1) -> tuple[Tensor, Tensor]:
    """``(ids, valid)`` ``[B, S, horizon * per_span_tokens]`` — spans ``s+start .. s+start+H-1``.

    ``tul.spandec_horizon``. Block ``h`` (offset ``(h-1) * per_span_tokens``) holds span
    ``s + start + h - 1``'s tokens, left-aligned inside its block and invalid after the
    span ends, so the shape is fixed and the decoder reads one concatenated causal
    sequence. Slot cells never appear: :func:`span_slots` selects token positions only.

    A slot is supervised at block ``h`` ONLY when that span exists AND is complete (its
    own terminating slot is present) — `span_slots`' rule at the block's shift, so a slot
    near the end of a row is supervised on the blocks it has and masked on the rest rather
    than dropped. ``horizon = 1, start = 1`` returns exactly :func:`next_span_slots`,
    tensor for tensor, which is what keeps the default bit-identical.

    ``start`` is ``tul.spandec_target_offset`` and it is a DIFFERENT knob from ``horizon``.
    Horizon WIDENS the target — H = 3 decodes spans s+1, s+2 and s+3, so the next span is
    still in it. Start MOVES the target — start = 3 at H = 1 decodes ONLY span s+3, and the
    next span gets no direct z target at all. The 2026-08-14 JEPA screen said the
    informative target sits two to three spans downstream, and nothing in the arc has ever
    asked the decoder for a span WITHOUT also asking it for the next one.

    The gaps between a short span and the next block carry `valid = False`, so their
    labels are ``ignore_index`` and their input embeddings are zeroed by
    :meth:`SpanDecoder.decode` — the same treatment a short span already gets inside one
    block.
    """
    if horizon < 1:
        raise ValueError(f"tul.spandec_horizon must be >= 1, got {horizon}")
    if start < 1:
        raise ValueError(
            f"tul.spandec_target_offset must be >= 1 (1 = the next span), got {start}")
    if horizon == 1 and start == 1:
        return next_span_slots(input_ids, layout, per_span_tokens)
    parts = [span_slots(input_ids, layout, per_span_tokens, shift=h)
             for h in range(start, start + horizon)]
    if horizon == 1:
        return parts[0]
    return (torch.cat([p[0] for p in parts], dim=2),
            torch.cat([p[1] for p in parts], dim=2))


class PackedSpans(NamedTuple):
    """:func:`pack_spans`' result: every tensor ``[B, cap]``, ``over`` a 0-dim bool."""
    slot: Tensor      # int64 slot id of the position (0 at an unused tail position)
    j: Tensor         # int64 offset inside the slot's span (0 at the tail)
    ids: Tensor       # int64 token the position PREDICTS (its label), 0 at the tail
    prev: Tensor      # int64 token the position READS (span token j-1), 0 at j == 0
    ok: Tensor        # bool: a real span token (False = unused tail)
    over: Tensor      # bool 0-dim: some row holds more than ``cap`` span tokens


def pack_spans(ids: Tensor, valid: Tensor, cap: int) -> PackedSpans:
    """Pack ``ids`` / ``valid`` ``[B, S, J]`` into ``[B, cap]``, per row, in slot order.

    Slot ``s`` contributes its ``n_s = valid[b, s].sum()`` real positions (a span's valid
    offsets are a prefix, see :func:`span_slots`); position ``(s, j)`` lands at
    ``sum_{s' < s} n_s' + j``. The destination is an exclusive cumsum on the device and the
    move is a ``scatter_`` into ``cap + 1`` columns whose last one is a dump: no boolean
    indexing and no host read, so the shapes are fixed and a CUDA graph can capture it.

    ``over`` is True when a row has more than ``cap`` real positions. The positions past
    ``cap`` are dropped into the dump column, so the caller MUST turn ``over`` into a loud
    failure (the model turns the loss into NaN, the ``row_cap`` precedent): a silent drop
    would train on fewer tokens than the loss reports.
    """
    B, S, J = ids.shape
    dev = ids.device
    n = valid.sum(-1)                                              # [B, S]
    end = n.cumsum(1)
    dest = (end - n).unsqueeze(-1) + torch.arange(J, device=dev)   # [B, S, J]
    flat = torch.where(valid & (dest < cap), dest, torch.full_like(dest, cap)).reshape(B, S * J)

    def put(src: Tensor) -> Tensor:
        out = src.new_zeros(B, cap + 1)
        out.scatter_(1, flat, src.reshape(B, S * J))
        return out[:, :cap]

    prev = torch.cat([torch.zeros_like(ids[..., :1]), ids[..., :-1]], dim=-1)
    return PackedSpans(
        slot=put(torch.arange(S, device=dev).view(1, S, 1).expand(B, S, J)),
        j=put(torch.arange(J, device=dev).view(1, 1, J).expand(B, S, J)),
        ids=put(ids), prev=put(prev), ok=put(valid), over=(end[:, -1] > cap).any())


KEEP_RES = 1024          # `tul.spandec_slot_keep` is used as round(keep * 1024) / 1024
_M31 = 0x7FFFFFFF


def keep_q_of(keep: float) -> int:
    """``tul.spandec_slot_keep`` as an integer numerator over :data:`KEEP_RES`: the host cap
    and the device budget use the SAME integer arithmetic, so the cap is a proven bound."""
    return int(round(float(keep) * KEEP_RES))


def keep_scaled(n, keep_q: int):
    """``round(n * keep_q / KEEP_RES)`` in integers (ties up); ``n`` an int or an int tensor."""
    return (n * keep_q + KEEP_RES // 2) // KEEP_RES


def _mix31(x: Tensor) -> Tensor:
    """A 31-bit integer hash (lowbias32 on 31 bits), int64 in and out. Every product stays
    under 2^62, so nothing wraps and CPU and CUDA agree bit for bit."""
    x = x & _M31
    x = x ^ (x >> 15)
    x = (x * 739982445) & _M31
    x = x ^ (x >> 12)
    x = (x * 696884537) & _M31
    return x ^ (x >> 15)


def slot_keep_mask(valid: Tensor, keep_q: int, seed: int, ctr: Tensor) -> Tensor:
    """The slots the packed span decoder trains on this step, ``[B, S]`` bool
    (``tul.spandec_slot_keep``).

    ``valid`` ``[B, S, J]`` is the span decoder's label mask. Per row the budget is
    ``keep_scaled(valid tokens of the row, keep_q)``. The row's slots are put in a random order
    (an integer hash of ``seed``, the 0-dim int64 device counter ``ctr``, row and slot) and the
    longest prefix of that order whose spans fit the budget is kept, each kept slot whole. A
    row's kept tokens are therefore at most its budget, which ``keep_scaled(cap, keep_q)``
    bounds from the row cap ``cap`` alone: the packed capacity shrinks by ``keep``. Pure device
    ops on fixed shapes (no host read, no RNG state), so a CUDA graph that records it replays
    a new subset each step as ``ctr`` advances.
    """
    B, S, _J = valid.shape
    dev = valid.device
    n = valid.sum(-1).to(torch.int64)                                  # [B, S] tokens per slot
    budget = keep_scaled(n.sum(-1), keep_q)                            # [B]
    cell = (torch.arange(B, device=dev).view(B, 1) * S + torch.arange(S, device=dev).view(1, S))
    score = _mix31(_mix31(cell + int(seed)) ^ _mix31(ctr.to(torch.int64) + 0x2545F491))
    idx = torch.arange(S, device=dev)
    # first[b, i, j] = slot j comes at or before slot i in the row's random order (ties by index)
    first = (score.unsqueeze(1) < score.unsqueeze(2)) | (
        (score.unsqueeze(1) == score.unsqueeze(2)) & (idx.view(1, 1, S) <= idx.view(1, S, 1)))
    cum = (first * n.unsqueeze(1)).sum(-1)                             # [B, S] tokens up to and incl. slot i
    return (n > 0) & (cum <= budget.unsqueeze(-1))


def chunk_segment_mask(seg: Tensor, chunk: int) -> Tensor:
    """``[B * nc, 1, chunk, 2 * chunk]`` bool for :meth:`PackedCtx.attend`, ``seg`` ``[B, P]``
    segment ids with ``P = nc * chunk``.

    Query ``i`` of chunk ``n`` sits at global position ``n * chunk + i``; its keys are chunk
    ``n - 1`` then chunk ``n`` (key ``k`` at ``n * chunk - chunk + k``). It may read a key
    iff the key is at or before it and carries the same segment id. A segment is at most
    ``chunk`` positions long (a slot decodes at most ``J`` offsets), so the two chunks hold
    every key a position may read and the window loses nothing. Chunk 0 has no previous
    chunk: its keys get id ``-1``, which no position carries. Every row keeps its own
    diagonal, so no row is fully masked (SDPA would return NaN)."""
    B, P = seg.shape
    nc = P // chunk
    cur = seg.view(B, nc, chunk)
    prev = F.pad(cur, (0, 0, 1, 0), value=-1)[:, :-1]
    key = torch.cat([prev, cur], dim=2)                                 # [B, nc, 2c]
    same = cur.unsqueeze(3) == key.unsqueeze(2)                         # [B, nc, c, 2c]
    i = torch.arange(chunk, device=seg.device).view(chunk, 1)
    k = torch.arange(2 * chunk, device=seg.device).view(1, 2 * chunk)
    return (same & (k <= i + chunk)).reshape(B * nc, 1, chunk, 2 * chunk)


class PackedCtx(NamedTuple):
    """What the blocks need to run on a packed layout (:meth:`SpanDecoder.decode_packed`)."""
    slot_oh: Tensor   # [B, P, S] one-hot of the position's slot, in the compute dtype (0 at the tail)
    mask: Tensor      # :func:`chunk_segment_mask`
    chunk: int

    def attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Block-diagonal causal attention over the packed row, ``q`` / ``k`` / ``v``
        ``[B, H, P, dh]`` -> ``[B, H, P, dh]``. Each query chunk reads its own and the
        previous chunk (``2 * chunk`` keys), not all ``P``: a segment never exceeds one
        chunk, so this is exactly the full segment-causal attention at ``chunk / P`` of its
        cost."""
        B, H, P, dh = q.shape
        c = self.chunk
        nc = P // c

        def chunks(t: Tensor) -> Tensor:
            return t.reshape(B, H, nc, c, dh)

        kc, vc = chunks(k), chunks(v)
        k2 = torch.cat([F.pad(kc, (0, 0, 0, 0, 1, 0))[:, :, :-1], kc], dim=3)
        v2 = torch.cat([F.pad(vc, (0, 0, 0, 0, 1, 0))[:, :, :-1], vc], dim=3)
        to_batch = lambda t: t.transpose(1, 2).reshape(B * nc, H, t.shape[3], dh)  # noqa: E731
        a = F.scaled_dot_product_attention(to_batch(chunks(q)), to_batch(k2), to_batch(v2),
                                           attn_mask=self.mask)
        return a.reshape(B, nc, H, c, dh).transpose(1, 2).reshape(B, H, P, dh)


class _SpanDecCross(nn.Module):
    """``tul.spandec_reads_cells`` — cross-attention onto a slot's M REGISTER CELLS.

    WHY IT EXISTS. On a register model (``tul.slot_cells: M > 1``) the span decoder today
    grades ``mean_i cell_i``: one target per span, read off the weakest possible reader of
    an M-cell thought. The register note names this as the arm's most likely reason to
    read a flat K-curve, and names the fix as its own follow-up — "a span decoder that
    cross-attends to the M cells as a memory"
    (``.agents/notes/proposed/architecture/2026-09-13-the-thought-register.md``,
    Alternatives). This is that follow-up.

    WHAT IT IS. Queries are the decoder's own token states ``[N, J, C]``; keys and values
    are the slot's ``M`` cells ``[N, M, C]``, the SAME states the coda's prefix write
    projects through ``W_prefix[i]`` — the stack's output where ``tul.cond_layers > 0``,
    because :meth:`MORPHTransformer._tul_cond_apply` runs before the cells are formed. So
    the decoder and the coda read the same object, which is the only version of this a
    comparison between the two readers can be made on.

    No mask. Every key belongs to the ONE slot whose thought this row decodes, so there is
    nothing causal to enforce among the cells; a pad slot's rows are all ``ignore_index``
    in the label, so its attention output is multiplied out of the loss downstream.

    ``proj`` is ZERO-INIT, so the block's output at step 0 is EXACTLY the mean-only
    decoder's and an arm with this knob on starts from its ruler. The other weights draw
    from a SECOND private generator, and the whole construction snapshots and restores the
    GLOBAL RNG state (the ``TULSlotRegister`` lesson: ``nn.Linear`` kaiming-draws on the
    global stream before the weight is overwritten), so turning the knob on moves not one
    weight of the decoder it is bolted into, nor of the model around it.
    """

    _fold: "SpanFold | None" = None      # SpanDecoder.fold_glue (model.glue_fold)

    def __init__(self, d_model: int, n_heads: int, gen: torch.Generator):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"span decoder d_model {d_model} not divisible by "
                             f"n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        for m in (self.q, self.kv):
            _init_linear(m, gen)
        with torch.no_grad():
            self.proj.weight.zero_()
        for m in (self.q, self.kv, self.proj):
            m._ternary_exclude = True

    def forward(self, x: Tensor, mem: Tensor, slot_oh: Tensor | None = None) -> Tensor:
        """``x [N, J, C]``, ``mem [N, M, C]`` -> ``[N, J, C]`` residual term.

        ``slot_oh`` ``[B, P, S]`` (``model.spandec_packed``) switches to the PACKED layout:
        ``x`` is ``[B, P, C]``, ``mem`` is ``[B, S, M, C]`` and position ``p`` attends to the
        M cells of its own slot, the same attention the per-slot path computes for it.
        """
        if slot_oh is not None:
            return self._forward_packed(x, mem, slot_oh)
        N, J, C = x.shape
        M = mem.shape[1]
        q = self.q(_norm_cast(self._fold, self.norm_q, x))
        k, v = self.kv(_norm_cast(self._fold, self.norm_kv, mem)).chunk(2, dim=-1)
        q = q.view(N, J, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(N, M, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(N, M, self.n_heads, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v)
        return self.proj(a.transpose(1, 2).reshape(N, J, C))

    def _forward_packed(self, x: Tensor, mem: Tensor, slot_oh: Tensor) -> Tensor:
        """The packed twin. K and V are projected once per SLOT (``[B, S, M, C]``), the
        scores against every slot are one small einsum, and the one-hot ``slot_oh`` keeps the
        position's own slot (a product with 0 or 1, so exact). No gather of K / V to the
        positions: that gather, and the scatter-add behind it, cost more than the einsum.
        The softmax over the M cells is fp32, like SDPA's. ``proj`` keeps its zero init."""
        B, P, C = x.shape
        S, M = mem.shape[1], mem.shape[2]
        H, dh = self.n_heads, self.d_head
        q = self.q(_norm_cast(self._fold, self.norm_q, x)).view(B, P, H, dh)
        k, v = self.kv(_norm_cast(self._fold, self.norm_kv, mem)).chunk(2, dim=-1)
        k, v = k.reshape(B, S, M, H, dh), v.reshape(B, S, M, H, dh)
        oh = slot_oh.view(B, P, S, 1, 1)
        w = (torch.einsum("bphd,bsmhd->bpshm", q, k) * oh).sum(2)         # [B, P, H, M]
        w = (w.float() * (dh ** -0.5)).softmax(-1).to(v.dtype)
        a = torch.einsum("bpshm,bsmhd->bphd", w.unsqueeze(2) * oh, v)
        return self.proj(a.reshape(B, P, C))


class _SpanDecBlock(nn.Module):
    """Pre-norm causal self-attention + SwiGLU, single stream, plain ``nn.Linear``.

    Deliberately NOT a :class:`MORPHBlock`: the decoder runs on its own ``[B·S, J, C]``
    axis with no Hyper-Connection carrier, no CCA prologue, no injections and no
    retention, and a MORPHBlock would have to be fed a 4-D carrier and a layer index it
    has no place in. This is the ``ParcaeCoreBlock`` precedent — a small, dense, honest
    block whose only job is to be a reader.

    ``gen_cross`` (``tul.spandec_reads_cells``) adds a :class:`_SpanDecCross` between the
    self-attention and the MLP. ``None`` — every model before that knob — builds nothing,
    draws nothing and traces the graph it always traced.
    """

    _fold: "SpanFold | None" = None      # SpanDecoder.fold_glue (model.glue_fold)

    def __init__(self, d_model: int, n_heads: int, d_ff: int, gen: torch.Generator,
                 gen_cross: torch.Generator | None = None):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"span decoder d_model {d_model} not divisible by "
                             f"n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        for m in (self.qkv, self.proj, self.gate_up, self.down):
            _init_linear(m, gen)
            m._ternary_exclude = True
        # RNG-NEUTRAL. `nn.Linear` kaiming-draws on the GLOBAL stream before the weight is
        # overwritten, so building the cross module would shift every module constructed
        # after this decoder. Snapshot and restore, exactly as `TULSlotRegister.__init__`
        # does after sabotage D7 caught the same leak there. The cross weights come from
        # their OWN generator, so `gen`'s stream is where an OFF build left it too.
        self.cross: _SpanDecCross | None = None
        if gen_cross is not None:
            _rng0 = torch.random.get_rng_state()
            self.cross = _SpanDecCross(d_model, n_heads, gen_cross)
            torch.random.set_rng_state(_rng0)

    def forward(self, x: Tensor, mem: Tensor | None = None,
                packed: PackedCtx | None = None) -> Tensor:
        """``packed`` is the PACKED layout's context (``model.spandec_packed``): the
        self-attention is causal inside a slot's segment and blind across segments
        (:meth:`PackedCtx.attend`, in place of ``is_causal``) and the cross-attention reads
        each position's own slot's cells. ``None`` is the per-slot path, op for op what it
        was before it."""
        B, T, C = x.shape
        h = _norm_cast(self._fold, self.norm1, x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shp = (B, T, self.n_heads, self.d_head)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        if packed is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = packed.attend(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        # A Python-level constant decided at construction (the tree's no-runtime-flag
        # rule): a decoder without the knob traces no branch here at all.
        if self.cross is not None:
            x = x + (self.cross(x, mem) if packed is None
                     else self.cross(x, mem, packed.slot_oh))
        h = _norm_cast(self._fold, self.norm2, x)
        g, u = self.gate_up(h).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u if self._fold is None else self._fold.silu_mul(g, u))


def _init_linear(m: nn.Linear, gen: torch.Generator) -> None:
    """std=0.02 from a PRIVATE generator — the rest of the model's stream is untouched."""
    with torch.no_grad():
        m.weight.copy_(torch.empty(m.weight.shape, device="cpu").normal_(
            mean=0.0, std=0.02, generator=gen))


class SpanDecoder(nn.Module):
    """``(z, next-span ids) -> [B, S, J, C]`` readout states, one per decoded token.

    ``forward`` returns the STATES; the caller applies the tied head and the chunked CE
    (``transformer._tul_spandec_loss``), because the head belongs to the model and the
    chunked kernel is what keeps ``[B, S, J, V]`` (2.4 GB fp32 at the panel shape) from
    ever existing.
    """

    _fold: "SpanFold | None" = None      # fold_glue (model.glue_fold)

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 max_tokens: int, seed_offset: int = 0, horizon: int = 1,
                 pass_positions: int = 0, reads_cells: bool = False,
                 target_offset: int = 1):
        """``seed_offset`` shifts BOTH private init streams.

        ``horizon`` (``tul.spandec_horizon``) decodes spans ``s+1 .. s+H`` as ONE causal
        sequence of ``H * max_tokens`` positions, so ``max_tokens`` stays the PER-SPAN cap
        and the decoded axis scales with H. ``horizon=1`` builds exactly the tensors it
        built before this parameter existed (the position table is zero-init, so a longer
        one draws no RNG either).

        A model can hold two of these at once — the ``tul.spandec`` TARGET decoder and the
        ``grad_pass_energy='recon'`` ENERGY decoder — and at offset 0 they would start from
        byte-identical weights. The offset is a construction-time constant, so it still
        draws nothing from the global RNG and an arm's other weights are unmoved.

        ``reads_cells`` (``tul.spandec_reads_cells``) adds a per-layer cross-attention onto
        the slot's ``M`` register cells. The conditioning on ``z`` stays the cells' MEAN,
        so at ``tul.slot_cells: 1`` or with the knob off the decoder is bit-identical.

        ``target_offset`` (``tul.spandec_target_offset``) is WHICH span the decoder
        decodes: ``1`` is the next span (every arm before the knob), ``k`` is span
        ``s + k`` and ONLY that span. It is recorded here rather than read from the config
        at loss time so that one object answers "what does this decoder decode".
        """
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"tul.spandec_layers must be >= 1, got {n_layers}")
        if max_tokens < 2:
            raise ValueError(f"tul.spandec_max_tokens must be >= 2, got {max_tokens}")
        if horizon < 1:
            raise ValueError(f"tul.spandec_horizon must be >= 1, got {horizon}")
        if target_offset < 1:
            raise ValueError(
                f"tul.spandec_target_offset must be >= 1 (1 = the next span), got "
                f"{target_offset}")
        self.per_span_tokens = int(max_tokens)
        self.horizon = int(horizon)
        self.reads_cells = bool(reads_cells)
        self.target_offset = int(target_offset)
        self.max_tokens = int(max_tokens) * int(horizon)
        gp = torch.Generator(device="cpu").manual_seed(_SEED_PROJ + int(seed_offset))
        # z and the token embeddings enter through their own bias-free maps. The token map
        # exists so the decoder can re-scale and re-orient the DETACHED tied table without
        # a private [V, d] embedding (37.7 M parameters at V=49169, d=768) and without
        # writing into the table the main CE owns.
        self.z_in = nn.Linear(d_model, d_model, bias=False)
        self.tok_in = nn.Linear(d_model, d_model, bias=False)
        for m in (self.z_in, self.tok_in):
            _init_linear(m, gp)
            m._ternary_exclude = True
        # Learned absolute position inside the span, ZERO-init: at step 0 the decoder is
        # position-blind and learns the offsets from the data. Deterministic, no draw.
        self.pos = nn.Parameter(torch.zeros(self.max_tokens, d_model))
        # A SECOND position table, for `tul.spandec_per_pass`, and it is not a convenience.
        # The per-pass target runs the same blocks over a DIFFERENT block geometry:
        # `spandec_pass_horizon_max` blocks of `spandec_pass_tokens` tokens, against the
        # exit target's one block of `max_tokens`. Sharing `self.pos` would make row 8 mean
        # "span s+2, token 0" to one term and "span s+1, token 8" to the other — one
        # parameter asked to be two different offsets at once. Zero-init, so it draws no
        # RNG and adds nothing at step 0; `None` (the default) builds nothing at all.
        self.pass_positions = int(pass_positions)
        if self.pass_positions > 0:
            self.pos_pass = nn.Parameter(torch.zeros(self.pass_positions, d_model))
        else:
            self.register_parameter("pos_pass", None)
        gb = torch.Generator(device="cpu").manual_seed(_SEED_BLOCKS + int(seed_offset))
        # A THIRD private stream, used only by the cross-attention. It is separate so the
        # self-attention and MLP weights of a `reads_cells` decoder are byte-identical to
        # a mean-only decoder's: the knob adds tensors, it does not move any.
        gx = (torch.Generator(device="cpu").manual_seed(_SEED_CROSS + int(seed_offset))
              if self.reads_cells else None)
        self.blocks = nn.ModuleList(
            [_SpanDecBlock(d_model, n_heads, d_ff, gb, gen_cross=gx)
             for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def fold_glue(self, fold: SpanFold) -> None:
        """``model.glue_fold``: run the decoder's RMS norms and SwiGLU products through the
        compiled forms in ``fold`` (same functions, fused). Called once at model build."""
        self._fold = fold
        for blk in self.blocks:
            blk._fold = fold
            if blk.cross is not None:
                blk.cross._fold = fold

    def _out_norm(self, x: Tensor) -> Tensor:
        """``out_norm(x)`` (fp32: the fp32 weight promotes), compiled when folded."""
        if self._fold is None:
            return self.out_norm(x)
        return self._fold.rms(x, self.out_norm.weight, self.out_norm.eps)

    def _check_mem(self, mem: Tensor | None, B: int, S: int, C: int, dtype) -> Tensor | None:
        """Validate the register cells ``mem`` and return them as ``[B, S, M, C]`` in
        ``dtype`` (``None`` stays ``None``)."""
        if self.reads_cells and mem is None:
            raise ValueError(
                "SpanDecoder was built with reads_cells=True (tul.spandec_reads_cells) "
                "and got no `mem`: the per-layer cross-attention has nothing to read. The "
                "caller must hand it the slot's M register cells.")
        if mem is not None and not self.reads_cells:
            raise ValueError(
                "SpanDecoder.decode got `mem` but was built with reads_cells=False: no "
                "block can read it, so the argument would be silently ignored.")
        if mem is not None:
            if mem.dim() != 4 or mem.shape[0] != B or mem.shape[1] != S \
                    or mem.shape[-1] != C:
                raise ValueError(
                    f"SpanDecoder.decode mem {tuple(mem.shape)} must be "
                    f"[B, S, M, C] = [{B}, {S}, M, {C}]")
            return mem.to(dtype)
        return None

    def decode(self, z: Tensor, ids: Tensor, valid: Tensor, emb: Tensor,
               pos: Tensor | None = None, mem: Tensor | None = None) -> Tensor:
        """``z [B, S, C]``, ``ids``/``valid`` ``[B, S, J]``, ``emb [V, C]`` -> ``[B, S, J, C]``.

        ``emb`` is the tied table the caller has already detached (or not — the caller
        obeys ``tul.mux_detach_head`` for BOTH ends of this decoder; see the module
        docstring). The returned states are pre-head: position ``j`` of slot ``s`` is the
        state that predicts the span's token ``j``.

        ``pos`` overrides the position table (:attr:`pos_pass` for the per-pass target,
        whose block geometry differs — see the constructor). ``None`` uses :attr:`pos`, so
        every existing caller is unchanged.

        ``mem`` ``[B, S, M, C]`` is the register's M cells (``tul.spandec_reads_cells``),
        already through the model's ``_readout``, cell by cell. Required when the decoder
        was built with ``reads_cells`` and refused when it was not: a memory the blocks
        cannot read is a silently ignored argument, and a missing one would make the arm
        its own ruler without saying so.

        Named ``decode`` rather than ``forward`` on purpose: the tied table is not this
        module's parameter, so ``__call__`` would hide a required argument that belongs to
        the model.
        """
        B, S, J = ids.shape
        C = z.shape[-1]
        dtype = z.dtype
        mem = self._check_mem(mem, B, S, C, dtype)
        if mem is not None:
            mem = mem.reshape(B * S, mem.shape[2], C)
        table = self.pos if pos is None else pos
        limit = int(table.shape[0])
        if J > limit:
            raise ValueError(
                f"SpanDecoder.decode got J={J} positions but was built for "
                f"max_tokens={limit}: the position table has no row past that.")
        # Input token at decoder position j is the span's token j-1; position 0 carries z.
        # An invalid (past-the-end) token contributes exactly zero — it is only ever read
        # by later positions, which are invalid too, because a span's valid offsets are a
        # prefix and the attention is causal.
        e = F.embedding(ids, emb.to(dtype))                            # [B, S, J, C]
        e = torch.where(valid.unsqueeze(-1), e, torch.zeros_like(e))
        x = torch.cat([self.z_in(z).unsqueeze(2), self.tok_in(e[:, :, :-1])], dim=2)
        # A caller may decode a PREFIX of the span (the oracle-z teacher at
        # `tul.oracle_z_max_tokens` < `max_tokens`): take the first J rows of the table,
        # never a view of the whole table onto J columns (a reshape error at J 8 vs 32
        # killed the first strict-oracle smoke, 2026-09-12).
        x = x + table[:J].to(dtype).view(1, 1, J, C)
        x = x.reshape(B * S, J, C)
        for blk in self.blocks:
            x = blk(x, mem)
        return self._out_norm(x).reshape(B, S, J, -1)

    def decode_packed(self, z: Tensor, ids: Tensor, valid: Tensor, emb: Tensor, cap: int,
                      mem: Tensor | None = None, chunk: int = 0
                      ) -> tuple[Tensor, Tensor, Tensor]:
        """:meth:`decode` on a PACKED layout (``model.spandec_packed``).

        :meth:`decode` runs the blocks on ``B * S * J`` rows although a slot's span fills
        only its first ``n_s`` positions (past the span's end and at pad slots the row is
        dead weight). Here each row's real positions are concatenated in slot order into
        ``P`` positions (:func:`pack_spans`; ``cap`` rounded up to a multiple of ``chunk``)
        and the blocks run on ``B * P`` rows.
        ``cap`` is a bound on ``sum_s n_s`` proven from shapes by the caller
        (``MORPHTransformer._spandec_packed_cap``): a row's labels are token positions,
        and a row with ``n`` valid slots holds ``L - K n`` of them, each slot at most ``J``.

        Per position nothing changes: position ``(s, 0)`` reads ``z_in(z_s)``, ``(s, j)``
        reads ``tok_in(emb(t_{j-1}))``, both plus ``pos[j]``. Self-attention is causal
        inside a slot's segment and blind across segments, so a position sees exactly the
        positions :meth:`decode` shows it. Cross-attention reads the position's own slot's
        cells. The unused tail gets a zero input, its own segment and label ``-100``.
        ``chunk`` (default ``J``) is the self-attention window: a segment is at most ``J``
        positions, so a query needs only its own and the previous ``chunk`` positions
        (:meth:`PackedCtx.attend`); it must be at least ``J``.

        Returns ``(states [B, P, C], labels [B, P], over)``: the pre-head states, the
        label per position (``-100`` at the tail) and the 0-dim bool of
        :class:`PackedSpans`. Only the next-span target of one block is defined here.
        """
        B, S, J = ids.shape
        C = z.shape[-1]
        dtype = z.dtype
        if self.horizon != 1 or self.target_offset != 1:
            raise ValueError(
                f"SpanDecoder.decode_packed supports horizon 1 / target_offset 1 only (got "
                f"{self.horizon} / {self.target_offset}).")
        if J > self.pos.shape[0]:
            raise ValueError(
                f"SpanDecoder.decode_packed got J={J} positions but was built for "
                f"max_tokens={self.pos.shape[0]}: the position table has no row past that.")
        chunk = int(chunk) or J
        if chunk < J:
            raise ValueError(f"SpanDecoder.decode_packed chunk {chunk} < J {J}: a segment "
                             f"would not fit in the attention window.")
        P = -(-int(cap) // chunk) * chunk
        mem = self._check_mem(mem, B, S, C, dtype)
        pk = pack_spans(ids, valid, P)
        # One-hot of the position's slot (all zero at the unused tail): the z read and the
        # cross-attention select with it, a product with 0 or 1, instead of a gather whose
        # backward is a scatter-add.
        oh = F.one_hot(pk.slot, S).to(dtype) * pk.ok.unsqueeze(-1).to(dtype)
        z_p = torch.bmm(oh, self.z_in(z))
        tok = self.tok_in(F.embedding(pk.prev, emb.to(dtype)))
        x = torch.where((pk.j == 0).unsqueeze(-1), z_p, tok)
        x = x + F.embedding(pk.j, self.pos.to(dtype))
        x = torch.where(pk.ok.unsqueeze(-1), x, torch.zeros_like(x))
        # Block-diagonal causal relation. The tail shares segment S, so every row of the
        # mask keeps at least its own diagonal (no all-masked row, no NaN).
        ctx = PackedCtx(oh, chunk_segment_mask(
            torch.where(pk.ok, pk.slot, torch.full_like(pk.slot, S)), chunk), chunk)
        for blk in self.blocks:
            x = blk(x, mem, packed=ctx)
        lab = torch.where(pk.ok, pk.ids, torch.full_like(pk.ids, -100))
        return self._out_norm(x), lab, pk.over
