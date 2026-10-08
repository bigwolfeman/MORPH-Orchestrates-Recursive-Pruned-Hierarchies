"""FlexAttention for Parcae blocks, and the LXTUL strict-geometry masks.

Parcae's `CausalSelfAttention` ignores its `mask` argument and always runs causal
flash attention. The strict slot loop needs span-local masks, so `FlexSelfAttention`
keeps every Parcae projection, gate, RoPE and QK-norm and replaces only the kernel call:
a `BlockMask` passed as `mask` runs `flex_attention`; `mask=None` keeps Parcae's causal
path. `use_flex()` swaps the class of an existing block's attention in place, so the
parameters, their names and their init are Parcae's own.

Masks (lxtul/SPEC.md section 2), per row `b`, positions in row order:
  prelude  kv <= q  and  bag[q] == bag[kv]                      (nothing crosses a span)
  core     kv // M <= q // M                                    (compact cell axis)
  coda     slot q: kv == q
           token q: kv <= q and (bag[q] == bag[kv] or (slot[kv] and bag[kv] < bag[q]))
`open=True` (the open-geometry control, not MORPH's design): a TOKEN query also reads every
earlier TOKEN, in the prelude and in the coda; cell queries keep the strict rule, so the
loop's input is unchanged and only the token side gains the plain model's context.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

from parcae_lm.modules.mixer import CausalSelfAttention, norm
from parcae_lm.modules.utils import apply_rotary_emb_complex_like

_flex = torch.compile(flex_attention, dynamic=False)
_mask = torch.compile(create_block_mask)


class FlexSelfAttention(CausalSelfAttention):
    def forward(self, x: Tensor, freqs_cis: Tensor, mask=None, **kwargs) -> Tensor:
        if not isinstance(mask, BlockMask):
            if mask is not None:
                raise TypeError(f"FlexSelfAttention takes a BlockMask or None, got {type(mask)}")
            return super().forward(x, freqs_cis, None, **kwargs)
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        ve = kwargs.get("ve")
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        if self.config.clip_qkv is not None or self.config.qk_bias:
            raise NotImplementedError("clip_qkv / qk_bias are not ported to the flex path")
        if self.config.rope_settings.use_rope:
            q, k = apply_rotary_emb_complex_like(q, k, freqs_cis=freqs_cis)
        if self.config.qk_norm:
            q, k = norm(q), norm(k)
        # one dtype for the kernel: v is fp32 where a value embedding was added
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else q.dtype
        q, k, v = q.to(dt), k.to(dt), v.to(dt)
        y = _flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=mask,
                  enable_gqa=self.n_rep > 1)
        return self.c_proj(y.transpose(1, 2).reshape(B, T, C))


def use_flex(blocks) -> None:
    """Swap every block's attention to FlexSelfAttention in place (same parameters)."""
    for block in blocks:
        if type(block.attn) is not CausalSelfAttention:
            raise TypeError(f"expected Parcae CausalSelfAttention, got {type(block.attn)}")
        block.attn.__class__ = FlexSelfAttention


def prelude_mask(bag: Tensor, is_slot: Tensor, open: bool = False) -> BlockMask:
    B, L = bag.shape

    def mod(b, h, q, kv):
        strict = (kv <= q) & (bag[b, q] == bag[b, kv])
        if not open:
            return strict
        return strict | ((kv <= q) & ~is_slot[b, q] & ~is_slot[b, kv])
    return _mask(mod, B, None, L, L, device=bag.device)


def coda_mask(bag: Tensor, is_slot: Tensor, open: bool = False) -> BlockMask:
    B, L = bag.shape

    def mod(b, h, q, kv):
        tok = (bag[b, q] == bag[b, kv]) | (is_slot[b, kv] & (bag[b, kv] < bag[b, q]))
        if open:
            tok = tok | ~is_slot[b, kv]
        return torch.where(is_slot[b, q], kv == q, (kv <= q) & tok)
    return _mask(mod, B, None, L, L, device=bag.device)


def core_mask(n_cells: int, cells_per_slot: int, device) -> BlockMask:
    M = cells_per_slot

    def mod(b, h, q, kv):
        return (kv // M) <= (q // M)
    return create_block_mask(mod, None, None, n_cells, n_cells, device=device)
