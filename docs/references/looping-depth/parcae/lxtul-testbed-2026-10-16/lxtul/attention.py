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

from lxtul.slot_compact import put_rows

_flex = torch.compile(flex_attention, dynamic=False)
_mask = torch.compile(create_block_mask)


class FlexSelfAttention(CausalSelfAttention):
    def project_qkv(self, x: Tensor, freqs_cis: Tensor, ve=None):
        """q, k, v [B, T, heads, head_dim] as the kernel reads them: value-embedding gate, RoPE, QK-norm,
        one dtype. Shared by `forward` and the compacted loop (`compact_layer`)."""
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
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
        return q.to(dt), k.to(dt), v.to(dt)

    def forward(self, x: Tensor, freqs_cis: Tensor, mask=None, **kwargs) -> Tensor:
        if not isinstance(mask, BlockMask):
            if mask is not None:
                raise TypeError(f"FlexSelfAttention takes a BlockMask or None, got {type(mask)}")
            return super().forward(x, freqs_cis, None, **kwargs)
        B, T, C = x.size()
        q, k, v = self.project_qkv(x, freqs_cis, kwargs.get("ve"))
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


def block_memory_mask(G: int, L: int, g: int, device) -> BlockMask:
    """Coda mask of Block-Parcae with a loop memory (`spiral.block_memory`). The sequence is [G memory
    positions ; L token positions] (memory m_j = the loop state of chunk j; chunks of g tokens, G = L // g):
      token i (chunk c = i // g) reads the tokens j <= i of its OWN chunk and the memory m_j of the strictly
          EARLIER chunks (j <= c - 1): m_j depends on the tokens of chunks <= j, all before chunk c;
      memory m_j reads m_k, k <= j (causal among the memory), and no token.
    The mask is the same for every row and head."""
    T = G + L

    def mod(b, h, q, kv):
        q_tok, kv_tok = q >= G, kv >= G
        tt = (kv - G <= q - G) & ((kv - G) // g == (q - G) // g)
        tm = kv < (q - G) // g                              # memory index < chunk of the token
        mm = kv <= q
        return torch.where(q_tok, torch.where(kv_tok, tt, tm), (~kv_tok) & mm)
    return _mask(mod, None, None, T, T, device=device)


def core_mask(n_cells: int, cells_per_slot: int, device) -> BlockMask:
    M = cells_per_slot

    def mod(b, h, q, kv):
        return (kv // M) <= (q // M)
    return create_block_mask(mod, None, None, n_cells, n_cells, device=device)


def slot_pack_mask(cell_row: Tensor, cell_slot: Tensor) -> BlockMask:
    """Core mask for the packed loop (`compact_slots`): the valid slots of the whole batch as ONE
    sequence of N cells. A cell reads the cells of its own row whose slot is <= its own, which is
    `core_mask` per row (`kv // M <= q // M`). `cell_row`, `cell_slot`: [N] long."""
    N = cell_row.shape[0]

    def mod(b, h, q, kv):
        return (cell_row[q] == cell_row[kv]) & (cell_slot[kv] <= cell_slot[q])
    return _mask(mod, 1, None, N, N, device=cell_row.device)


def span_pack_mask(doc: Tensor) -> BlockMask:
    """Mask of the packed span decoder (`spandec_packed`): per row, a position reads the earlier
    positions of its own span (`doc` [B, T]: the span's slot index, an out-of-range value at unused
    positions). It is the per-slot causal mask of the unpacked decoder."""
    B, T = doc.shape

    def mod(b, h, q, kv):
        return (doc[b, q] == doc[b, kv]) & (kv <= q)
    return _mask(mod, B, None, T, T, device=doc.device)


def gather_core_mask(idx: Tensor, n_kv: int, cells_per_slot: int) -> BlockMask:
    """Core mask of a compacted pass: query row r of batch row b is the cell at position idx[b, r]
    of the dense cell axis, and reads the cells whose slot is <= its own (`core_mask`'s relation at the
    ORIGINAL position). idx [B, n] long; keys are all n_kv cells."""
    B, n = idx.shape
    M = cells_per_slot

    def mod(b, h, q, kv):
        return (kv // M) <= (idx[b, q] // M)
    return _mask(mod, B, None, n, n_kv, device=idx.device)


def compact_layer(block, x: Tensor, freqs: Tensor, mask: BlockMask, ve, k_cache, v_cache,
                  idx, active: Tensor):
    """One core layer of a compacted pass (`slot_compact`): `TransformerPreNormBlock.forward` with the
    keys and values of the cells NOT computed in this pass read from a cache.

      x       [B, n, C] the cells computed (n = every cell in "full" mode, C_t * M in "gather" mode)
      idx     [B, n] their positions on the dense axis, or None ("full": n = N, idx = position)
      active  [B, N] the cells whose pass this is (depth > t)
      k_cache, v_cache  [B, N, kv_heads, head_dim] this layer's keys and values as the last pass
              left them, or None at pass 0 (every cell is active then)
    Returns (x_out, K, V): the layer output on the n cells, and the caches the next pass reads: this
    pass's keys and values at the active cells, the old ones elsewhere. The cached tensors are the
    projections of the attention input (norm_1(x)), so a frozen cell serves the keys and values its
    last active pass computed; caching them rather than the input only skips the re-projection."""
    attn = block.attn
    q, k, v = attn.project_qkv(block.norm_1(x), freqs, ve)
    if idx is None:
        if k_cache is None:
            K, V = k, v
        else:
            a = active[..., None, None]
            K, V = torch.where(a, k, k_cache), torch.where(a, v, v_cache)
    else:
        if k_cache is None:          # pass 0: every cell is active, the zeros are overwritten
            shape = (k.shape[0], active.shape[1], *k.shape[2:])
            k_cache, v_cache = k.new_zeros(shape), v.new_zeros(shape)
        K, V = put_rows(k, k_cache, idx, active), put_rows(v, v_cache, idx, active)
    y = _flex(q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2), block_mask=mask,
              enable_gqa=attn.n_rep > 1)
    B, T, C = x.shape
    x = x + attn.c_proj(y.transpose(1, 2).reshape(B, T, C))
    x = x + block.mlp(block.norm_2(x))
    return x, K, V
