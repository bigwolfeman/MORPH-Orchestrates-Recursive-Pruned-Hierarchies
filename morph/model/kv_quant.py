"""KV-cache quantization for MORPH — INFERENCE-ONLY, post-training (no QAT).

The training forward never touches the KV cache (`kv_cache.py`); it recomputes the full
sequence in bf16. So quantizing the cached K/V state is *post-training quantization of stored
activations at inference* — NOT activation-QAT. The model is trained in bf16 and deployed with
a low-bit cache; the loss landscape is never shaped around the quantization. This is the
"KV-quant generally works without QAT" regime (cf. KIVI / KVQuant).

What gets quantized (the cached K/V-like tensors, see kv_cache.AttnSiteCache):
  - window keys/values  `win_k`/`win_v`  [B,H,L,D]   — bounded (window-1), small absolute win
  - compressed blocks   `C_comp`         [B,n,D]      — GROWS O(T/m); CSA m=4 is the long-ctx sink
  - indexer keys        `K_I`            [B,n,d_I]    — GROWS O(T/m); quantizing it perturbs the
                                                        CSA top-k SELECTION (a real, measured effect)
NOT quantized: `x_recent`/`comp_x` (conv history — bounded, conv is precision-sensitive) and the
CSA `top_idx` routing indices (integers, recomputed at query time — never stored quantized).

Mechanism: symmetric absmax, per-row over the last (feature) dim, optionally grouped. Quant is
applied at the STORE site in kv_cache.py (quantize-then-store), so the cached buffer already holds
the dequantized-from-int value and the verified read math is unchanged. The value a real packed
cache would produce on read is bit-identical to this fake-quant round-trip — proven by the
pack/unpack round-trip test (`quant_dequant(x) == unpack(*pack(x))`). The packed footprint
(`packed_nbytes`) is the deployable memory number.
"""
from __future__ import annotations

import torch
from torch import Tensor

# Symmetric signed range max for each bit-width (we use [-qmax, qmax], no negative extreme).
_QMAX: dict[int, int] = {8: 127, 6: 31, 4: 7}
VALID_BITS = (4, 6, 8)


def _resolve_group(group_size: int, feat: int) -> int:
    """Effective group size along the feature dim.

    group_size<=0, >=feat, or not dividing feat → per-row (one scale per feature vector).
    Otherwise group along the feature dim in chunks of group_size.
    """
    if group_size <= 0 or group_size >= feat or feat % group_size != 0:
        return feat
    return group_size


def quant_dequant(x: Tensor, bits: int, group_size: int = 0) -> Tensor:
    """Symmetric absmax fake-quant round-trip along the last dim (per-row, optionally grouped).

    Returns a tensor of x's dtype whose values are exactly representable as int{bits}×scale.
    This is the value a real packed int{bits} cache yields on read (see pack/unpack).
    """
    if bits not in _QMAX:
        raise ValueError(f"kv_quant bits must be in {VALID_BITS}, got {bits}")
    qmax = _QMAX[bits]
    feat = x.shape[-1]
    g = _resolve_group(group_size, feat)
    lead = x.shape[:-1]
    xg = x.reshape(*lead, feat // g, g)
    s = xg.detach().abs().amax(dim=-1, keepdim=True).div(qmax).clamp(min=1e-8)
    q = (xg / s).round().clamp(-qmax, qmax)
    # Dequantize with the fp16-rounded scale — the deployed packed cache stores fp16 scales
    # (see `pack`), so this round-trip is BIT-IDENTICAL to unpack(pack(x)). Quantization uses
    # the full-precision s (matches pack's q), dequant uses the fp16 scale (matches unpack).
    s16 = s.to(torch.float16).to(torch.float32)
    out = (q * s16).reshape(*lead, feat)
    return out.to(x.dtype)


def pack(x: Tensor, bits: int, group_size: int = 0) -> tuple[Tensor, Tensor, tuple]:
    """Real packed representation: (codes, scales, meta).

    int8/int6 → codes is int8 [.. , feat] (int6 stored in int8, 0.25B wasted unless re-packed).
    int4      → codes is uint8 [.. , feat//2], two nibbles per byte (low nibble = even index).
    scales    → fp16, shape [.. , feat//g, 1].
    meta      → (bits, group_size, feat) for unpack.
    The dequantized value `unpack(pack(x))` is bit-identical to `quant_dequant(x)`.
    """
    if bits not in _QMAX:
        raise ValueError(f"kv_quant bits must be in {VALID_BITS}, got {bits}")
    qmax = _QMAX[bits]
    feat = x.shape[-1]
    g = _resolve_group(group_size, feat)
    lead = x.shape[:-1]
    xg = x.reshape(*lead, feat // g, g)
    s = xg.detach().abs().amax(dim=-1, keepdim=True).div(qmax).clamp(min=1e-8)
    q = (xg / s).round().clamp(-qmax, qmax).reshape(*lead, feat)
    scales = s.to(torch.float16)
    if bits == 4:
        # shift to unsigned nibble [0,14] (centered at 8 → stored value q+8 ∈ [1,15]); pack pairs.
        u = (q + 8).clamp(0, 15).to(torch.uint8)
        u = u.reshape(*lead, feat // 2, 2)
        codes = (u[..., 0] | (u[..., 1] << 4)).to(torch.uint8)   # [.., feat//2]
    else:
        codes = q.to(torch.int8)
    return codes, scales, (bits, group_size, feat)


def unpack(codes: Tensor, scales: Tensor, meta: tuple) -> Tensor:
    """Inverse of `pack` → fp32/bf16 values (bit-identical to quant_dequant)."""
    bits, group_size, feat = meta
    g = _resolve_group(group_size, feat)
    lead = codes.shape[:-1]
    if bits == 4:
        c = codes.to(torch.int16)
        lo = (c & 0xF).to(torch.float32) - 8.0
        hi = ((c >> 4) & 0xF).to(torch.float32) - 8.0
        q = torch.stack([lo, hi], dim=-1).reshape(*lead, feat)
    else:
        q = codes.to(torch.float32)
    qg = q.reshape(*lead, feat // g, g)
    out = (qg * scales.to(torch.float32)).reshape(*lead, feat)
    return out


def packed_nbytes(numel: int, feat: int, bits: int, group_size: int = 0,
                  scale_bytes: int = 2) -> int:
    """Deployable byte footprint of the packed representation for `numel` elements.

    codes: numel·bits/8 (int4 = 0.5 B/elem). scales: one fp16 per group:
    (numel/feat) rows × (feat/g) groups × scale_bytes.
    """
    g = _resolve_group(group_size, feat)
    code_bytes = numel * bits / 8.0
    n_rows = numel // feat
    n_scales = n_rows * (feat // g)
    return int(round(code_bytes + n_scales * scale_bytes))
