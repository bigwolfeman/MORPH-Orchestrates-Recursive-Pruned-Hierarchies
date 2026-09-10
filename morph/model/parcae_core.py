"""A plain Parcae-style looped core — the one-factor swap for ``model.core_impl``.

Why this file exists
--------------------
The slot loop (TUL, ``tul.tokens_through_core: false``) reads FLAT on every arm measured
so far: passes 2-6 of a 268 M-parameter core are worth 0.0004-0.0018 nats to the coda
(`lab/experiments/results/2026-09-10-slot-geometry-audit/README.md`). Two explanations
survive that audit and cannot be separated by any instrument already in the tree:

* **H-core** — the fault is MORPH's core AT THE SLOT SHAPE. The looped block stack there
  is ternary-STE weights, CCA-compressed attention with a CSA/HCA alternation whose HCA
  compressed branch is EXACTLY DEAD at a 64-cell budget (finding F1), XSA self-exclusion
  (query 0 attends nothing), a 4-stream Cayley hyper-connection carrier and a diagonal
  injection on 256 of 1024 dims.
* **H-mech** — the fault is the TUL mechanism or its target, and any core would read flat.

This module is the instrument that separates them: the SAME slot layout, prelude, coda,
packer, MUX loss and losses, with the looped core replaced by a plain pre-norm
transformer block — dense causal softmax attention, a single-stream additive residual,
a SwiGLU MLP, and dense (never ternarised, never pruned, never carved) weights.

What a ``ParcaeCoreBlock`` is, precisely
---------------------------------------
``h <- h + attn(RMSNorm(h))`` then ``h <- h + mlp(RMSNorm(h))``. Nothing else.

* **Attention**: one dense causal softmax over the whole compact sequence, `n_heads`
  heads of `d_head`, rotary positions from the tree's own :class:`CoPEEmbedding` (so the
  positional encoding is NOT a second changed factor), no compression, no block
  selection, no gate mixture, no self-exclusion (query 0 attends itself, unlike XSA), no
  QK-Norm, no residual-attention carry, no conv, no indexer.
* **MLP**: the tree's plain dense :class:`_SwiGLU` (``nn.Linear``), NOT ``_SwiGLUMortar``.
  Being an ``nn.Linear`` rather than a ``MortarLinear``/``CMSBlockLinear`` is what makes
  the Parcae core invisible to the prune / carve / ReMoE / deploy-packer walks BY
  CONSTRUCTION rather than by an exclusion list somebody has to maintain — the same
  argument :class:`morph.model.mhc.PassLoRA` records for its own parameters.
* **Residual**: a plain add. No hyper-connections, so the block takes and returns a
  single-stream ``[B, S, C]`` carrier. The stream collapse/expand happens ONCE per pass
  in ``MORPHTransformer._apply_core_step`` (stream mean in, broadcast out — the same
  reduction ``_readout`` and ``TULSlots.unpack`` already use), so the prelude and the
  coda keep their 4-stream Cayley carrier untouched and are bit-identical to the ruler's.
* **Precision**: every ``nn.Linear`` here carries ``_ternary_exclude = True``, which
  ``morph.model.ternary_qat._categorize`` reads and returns ``None`` for. The Parcae core
  is therefore bf16/fp32 under EVERY ternary scope, while prelude, coda and embeddings
  ternarise exactly as the ruler's do.

What it deliberately keeps from MORPH, and why
----------------------------------------------
* The **loop entry** is whatever ``model.core_state_init`` says — the knob, not a new one.
* The **diagonal carry** (``self.injection``, ``DiagonalInjection``) and the per-core-layer
  x0/bigram term are applied by ``_apply_core_step``, OUTSIDE this block, exactly as they
  are for a MORPH core. Parcae's own recurrence is `h_t = A h_{t-1} + B e + R(h_{t-1}, e)`
  (arXiv 2604.12946 §3, App. C) — `A`/`B` is that injection and `R` is this block stack —
  so the faithful Parcae form is ``injection_channels: all`` with ``injection_B: true``,
  set in the arm's config, not hardcoded here.
* The **head geometry** defaults to ``d_model // (compression * n_heads)``, the tree's own
  ``d_head``, so the swap is roughly parameter-matched against the MORPH core it replaces
  and a depth gain cannot be read as raw extra capacity. ``model.parcae_core_d_head``
  overrides it (``d_model // n_heads`` is the textbook choice).
* ``pass_idx`` and :class:`morph.model.mhc.PassLoRA` work here identically to
  :class:`morph.model.mhc.MORPHBlock`, so the per-pass-LoRA arm composes with this core.

What it refuses rather than silently ignoring
---------------------------------------------
Non-empty ``attn_kwargs`` (the TG masks), a retention (GLA) carry, and the HC carrier
engine's ``next_inject_term`` all RAISE. This block cannot honour any of them, and a
silently dropped mask is exactly the class of defect finding F1 was.
"""

from __future__ import annotations

import math

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import CoPEEmbedding, RMSNorm

__all__ = ["ParcaeDenseAttention", "ParcaeCoreBlock"]


class ParcaeDenseAttention(nn.Module):
    """Dense causal multi-head softmax attention with rotary positions.

    ``forward(x)`` takes and returns ``[B, S, C]``. Q, K and V all use ``n_heads`` heads
    (no grouped-query sharing): the block is meant to be the plainest thing that can
    stand where MORPH's CCA+CSA/HCA+XSA stack stood.
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int,
                 max_seq_len: int, context_len: int, rope_base: float = 10000.0,
                 residual_blocks: int = 1):
        super().__init__()
        if d_head % 2:
            raise ValueError(f"ParcaeDenseAttention needs an even d_head, got {d_head}")
        self.n_heads = int(n_heads)
        self.d_head = int(d_head)
        inner = self.n_heads * self.d_head
        self.q_proj = nn.Linear(d_model, inner, bias=False)
        self.k_proj = nn.Linear(d_model, inner, bias=False)
        self.v_proj = nn.Linear(d_model, inner, bias=False)
        self.o_proj = nn.Linear(inner, d_model, bias=False)
        # Rotary: the tree's own CoPE (parameter-free, buffers are non-persistent so this
        # module adds no checkpoint key). Frequencies whose wavelength exceeds context_len
        # are tapered; at the slot budget every wavelength is well inside it, so this is
        # plain RoPE there.
        self.rope = CoPEEmbedding(self.d_head, max_seq_len, base=rope_base,
                                  context_len=context_len)
        # Scaled init (Parcae App. Q "scaled init"): in-projections at 1/sqrt(d_model),
        # the out-projection additionally divided by sqrt(2 * n_blocks) so the residual
        # stream's variance does not grow with the number of applied blocks.
        std = d_model ** -0.5
        for lin in (self.q_proj, self.k_proj, self.v_proj):
            nn.init.normal_(lin.weight, std=std)
        nn.init.normal_(self.o_proj.weight, std=std / math.sqrt(2.0 * max(1, residual_blocks)))

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        H, D = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)      # [B,H,S,D]
        k = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, S, H, D).transpose(1, 2)
        q, k = self.rope(q, k)
        # is_causal=True: query i attends keys 0..i INCLUSIVE. MORPH's XSA excludes the
        # self token, which leaves query 0 with an all -inf softmax row on every shape
        # (measured, both S = 64 and S = 1152). This block does not do that.
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(o.transpose(1, 2).reshape(B, S, H * D))

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, d_head={self.d_head}"


class ParcaeCoreBlock(nn.Module):
    """Pre-norm attention + SwiGLU MLP with a PLAIN additive residual, single stream.

    Forward signature is byte-for-byte :meth:`morph.model.mhc.MORPHBlock.forward`'s, so
    ``MORPHTransformer._apply_core_step`` calls it with no branch of its own beyond the
    stream collapse. The arguments this block cannot honour raise instead of being
    dropped; see the module docstring.
    """

    def __init__(self, d_model: int, d_ff: int, n_heads: int, d_head: int,
                 max_seq_len: int, context_len: int, dropout: float = 0.0,
                 rope_base: float = 10000.0, residual_blocks: int = 1):
        super().__init__()
        from .transformer import _SwiGLU        # local: transformer imports this module

        self.norm_attn = RMSNorm(d_model)
        self.attention = ParcaeDenseAttention(
            d_model, n_heads, d_head, max_seq_len, context_len,
            rope_base=rope_base, residual_blocks=residual_blocks)
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = _SwiGLU(d_model, d_ff)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        std = d_model ** -0.5
        nn.init.normal_(self.mlp.gate_up.weight, std=std)
        nn.init.normal_(self.mlp.down.weight,
                        std=(d_ff ** -0.5) / math.sqrt(2.0 * max(1, residual_blocks)))

        # Per-pass low-rank deltas, attached post-construction exactly as MORPHBlock does
        # (PassLoRA draws from a private generator, so attaching it leaves every other
        # weight byte-identical). None -> both call sites below are Python-level no-ops.
        self.pass_lora = None

        # THE PRECISION EXCLUSION. `ternary_qat._categorize` returns None for any module
        # carrying this attribute, under every scope. Set here, on the leaves, so the
        # exclusion travels with the module instead of living in a path list.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m._ternary_exclude = True

    def attach_pass_lora(self, lora: nn.Module) -> None:
        """Give this block its per-pass low-rank deltas (Bae et al. 2024).

        Same contract as :meth:`morph.model.mhc.MORPHBlock.attach_pass_lora`: NEW
        parameters, so the optimizer must be built after this.
        """
        self.pass_lora = lora

    def forward(
        self,
        h: Tensor,
        attn_kwargs: dict | None = None,
        mlp_kwargs: dict | None = None,
        next_inject_term: Tensor | None = None,
        ret_state: Tensor | None = None,
        ret_capture: dict | None = None,
        ret_reset_mask: Tensor | None = None,
        pass_idx: int = 0,
    ) -> Tensor:
        if attn_kwargs:
            raise NotImplementedError(
                "ParcaeCoreBlock received attention kwargs "
                f"{sorted(attn_kwargs)} — it is a plain dense causal block and honours "
                "none of them (TG masks included). Raising rather than dropping them: a "
                "silently ignored attention restriction is the defect class finding F1 "
                "of the 2026-09-10 slot-geometry audit was.")
        if next_inject_term is not None:
            raise NotImplementedError(
                "ParcaeCoreBlock has no HC carrier engine, so there is no POST write to "
                "fold `next_inject_term` into.")
        if ret_state is not None or ret_capture is not None or ret_reset_mask is not None:
            raise NotImplementedError(
                "ParcaeCoreBlock carries no GLA retention branch; a core-section "
                "retention carry is refused at build (MORPHTransformer.__init__).")
        if h.dim() != 3:
            raise ValueError(
                f"ParcaeCoreBlock expects a single-stream [B, S, C] carrier, got shape "
                f"{tuple(h.shape)}. The stream collapse belongs to _apply_core_step.")
        lora = self.pass_lora            # Python-level constant per module instance

        xa = self.norm_attn(h)
        a = self.attention(xa)
        if lora is not None and lora.has_attn:
            a = a + lora.delta("attn", xa.to(a.dtype), pass_idx)
        h = h + self.drop(a)

        xm = self.norm_mlp(h)
        y = self.mlp(xm, **(mlp_kwargs or {}))
        if lora is not None and lora.has_mlp:
            y = y + lora.delta("mlp", xm.to(y.dtype), pass_idx)
        return h + self.drop(y)


def parcae_core_d_head(d_model: int, n_heads: int, compression: int,
                       override: int | None) -> int:
    """The Parcae core's head width.

    ``None`` -> ``d_model // (compression * n_heads)``, which is exactly the ``d_head``
    ``MORPHAttention`` computes, so the swapped core is roughly parameter-matched against
    the MORPH core it replaces. A capacity difference would be a confound on the one
    question this arm asks.
    """
    if override is not None:
        d = int(override)
    else:
        d = d_model // (max(1, int(compression)) * int(n_heads))
    if d < 2 or d % 2:
        raise ValueError(
            f"parcae core d_head resolved to {d} (d_model={d_model}, n_heads={n_heads}, "
            f"compression={compression}, override={override}); it must be even and >= 2. "
            "Set model.parcae_core_d_head explicitly.")
    return d
