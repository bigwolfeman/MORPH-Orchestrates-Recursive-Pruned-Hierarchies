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

There IS a token path. ``docs/tul-spec.md`` and the root ``CLAUDE.md`` are emphatic:
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
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm
from .tul_layout import SlotLayout

__all__ = ["SpanDecoder", "horizon_span_slots", "next_span_slots",
           "own_span_slots", "span_slots"]

# Private init streams. Two constants, never the global RNG: building the decoder must not
# shift a single weight of the model it is bolted onto (the `TULSlots.W_sent` rule).
_SEED_BLOCKS = 0x5DEC
_SEED_PROJ = 0x5DEC1


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
                       horizon: int) -> tuple[Tensor, Tensor]:
    """``(ids, valid)`` ``[B, S, horizon * per_span_tokens]`` — spans ``s+1 .. s+H``.

    ``tul.spandec_horizon``. Block ``h`` (offset ``(h-1) * per_span_tokens``) holds span
    ``s + h``'s tokens, left-aligned inside its block and invalid after the span ends, so
    the shape is fixed and the decoder reads one concatenated causal sequence. Slot cells
    never appear: :func:`span_slots` selects token positions only.

    A slot is supervised at block ``h`` ONLY when span ``s + h`` exists AND is complete
    (its own terminating slot is present) — `span_slots`' rule at ``shift = h``, so a slot
    near the end of a row is supervised on the blocks it has and masked on the rest rather
    than dropped. ``horizon = 1`` returns exactly :func:`next_span_slots`, tensor for
    tensor, which is what keeps the default bit-identical.

    The gaps between a short span and the next block carry `valid = False`, so their
    labels are ``ignore_index`` and their input embeddings are zeroed by
    :meth:`SpanDecoder.decode` — the same treatment a short span already gets inside one
    block.
    """
    if horizon < 1:
        raise ValueError(f"tul.spandec_horizon must be >= 1, got {horizon}")
    if horizon == 1:
        return next_span_slots(input_ids, layout, per_span_tokens)
    parts = [span_slots(input_ids, layout, per_span_tokens, shift=h)
             for h in range(1, horizon + 1)]
    return (torch.cat([p[0] for p in parts], dim=2),
            torch.cat([p[1] for p in parts], dim=2))


class _SpanDecBlock(nn.Module):
    """Pre-norm causal self-attention + SwiGLU, single stream, plain ``nn.Linear``.

    Deliberately NOT a :class:`MORPHBlock`: the decoder runs on its own ``[B·S, J, C]``
    axis with no Hyper-Connection carrier, no CCA prologue, no injections and no
    retention, and a MORPHBlock would have to be fed a 4-D carrier and a layer index it
    has no place in. This is the ``ParcaeCoreBlock`` precedent — a small, dense, honest
    block whose only job is to be a reader.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, gen: torch.Generator):
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

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        h = self.norm1(x).to(x.dtype)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        shp = (B, T, self.n_heads, self.d_head)
        q = q.view(shp).transpose(1, 2)
        k = k.view(shp).transpose(1, 2)
        v = v.view(shp).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, C))
        h = self.norm2(x).to(x.dtype)
        g, u = self.gate_up(h).chunk(2, dim=-1)
        return x + self.down(F.silu(g) * u)


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

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
                 max_tokens: int, seed_offset: int = 0, horizon: int = 1):
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
        """
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"tul.spandec_layers must be >= 1, got {n_layers}")
        if max_tokens < 2:
            raise ValueError(f"tul.spandec_max_tokens must be >= 2, got {max_tokens}")
        if horizon < 1:
            raise ValueError(f"tul.spandec_horizon must be >= 1, got {horizon}")
        self.per_span_tokens = int(max_tokens)
        self.horizon = int(horizon)
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
        gb = torch.Generator(device="cpu").manual_seed(_SEED_BLOCKS + int(seed_offset))
        self.blocks = nn.ModuleList(
            [_SpanDecBlock(d_model, n_heads, d_ff, gb) for _ in range(n_layers)])
        self.out_norm = RMSNorm(d_model)

    def decode(self, z: Tensor, ids: Tensor, valid: Tensor, emb: Tensor) -> Tensor:
        """``z [B, S, C]``, ``ids``/``valid`` ``[B, S, J]``, ``emb [V, C]`` -> ``[B, S, J, C]``.

        ``emb`` is the tied table the caller has already detached (or not — the caller
        obeys ``tul.mux_detach_head`` for BOTH ends of this decoder; see the module
        docstring). The returned states are pre-head: position ``j`` of slot ``s`` is the
        state that predicts the span's token ``j``.

        Named ``decode`` rather than ``forward`` on purpose: the tied table is not this
        module's parameter, so ``__call__`` would hide a required argument that belongs to
        the model.
        """
        B, S, J = ids.shape
        C = z.shape[-1]
        dtype = z.dtype
        # Input token at decoder position j is the span's token j-1; position 0 carries z.
        # An invalid (past-the-end) token contributes exactly zero — it is only ever read
        # by later positions, which are invalid too, because a span's valid offsets are a
        # prefix and the attention is causal.
        e = F.embedding(ids, emb.to(dtype))                            # [B, S, J, C]
        e = torch.where(valid.unsqueeze(-1), e, torch.zeros_like(e))
        x = torch.cat([self.z_in(z).unsqueeze(2), self.tok_in(e[:, :, :-1])], dim=2)
        x = x + self.pos.to(dtype).view(1, 1, J, C)
        x = x.reshape(B * S, J, C)
        for blk in self.blocks:
            x = blk(x)
        return self.out_norm(x).reshape(B, S, J, -1)
