"""The tied LM head read as int codes x row scale: 88 MB per decode step instead of 201 MB.

MORPH's head is ``embed.lm_weight()`` = ``[euclidean table | Lorentz log-map]``
(``morph/model/embeddings.py``). With ``model.embed_quant`` on, the euclidean table is the
``IntNRowSTE`` forward of its fp32 shadow (``morph/model/embed_quant.py``): per row
``s = max|w| / hi`` and ``q = round(w / s).clamp(lo, hi) * s``. So its 768 columns are
int6 (or int8) codes times one fp32 scale per row. ``head_codes_pack`` recomputes the
codes and scales from the shadow with the module's own ops and keeps them ONLY if
``fp32(code) * s`` equals the head table bit for bit (the STE's forward is
``w + (q - w)``, which can differ from ``q`` by an ulp; measured on the LXTUL-pointer and
plain-panel step-5000 checkpoints: 0 of 37,761,792 entries differ). The Lorentz block stays
fp32. The kernel forms each weight as ``fp32(code) * s`` (one rounding, the STE's own
product), so the logits are the fp32 table's logits up to the order of the fp32 sums.

Bytes per step at the model's widths: 49169 x 768 int8 (37.8 MB) + 49169 x 256 fp32
(50.3 MB) + scales (0.2 MB) = 88.3 MB, against 201.4 MB for the fp32 table (5090 HBM
1.79 TB/s: 49 us against 112 us of pure weight read).

Schedule (``_head_kernel``): one CTA per BV = 8 vocab rows. The CTA dequantizes its
[8, C] weight tile ONCE into registers and then forms each query row as one broadcast
multiply and one reduction; the query rows (at most 16 in the decoder) come from L1/L2.
Measured in-graph with L2 flushed before every call (RTX 5090, the model's widths,
/home/wolfe/morph-scratch/megadec/head/mb_head2.py, mb_head3.py): 53-60 / 61-82 /
51-86 / 124-150 us at 1 / 4 / 8 / 16 rows, against the fp32 table's cuBLAS SGEMM
124-147 / 255-286 us at 1-8 / 16 rows and the engine's ``bf16_gemv`` 133-144 us at 1 row.
Rejected schedules, same bench: rows looped OUTSIDE the k loop (the tile re-read per row
from L2: 58 / 70 / 92 / 219 us), ``tl.dot(input_precision="ieee")`` with rows padded to 16
(164-290 us at every row count), and a 3D [rows, vocab, k] product (61-327 us).
"""
from __future__ import annotations

import torch
import torch.nn.utils.parametrize as parametrize
import triton
import triton.language as tl
from torch import Tensor, nn

from morph.model.embed_quant import IntNRowSTE

__all__ = ["head_codes_pack", "head_codes_logits"]

_BV = 8                 # vocab rows per CTA


def head_codes_pack(euc: nn.Module, w_head: Tensor) -> tuple[Tensor, Tensor, Tensor] | None:
    """``(codes [V, CE] int8, scale [V] fp32, w_lor [V, C - CE] fp32)`` of the head table
    ``w_head`` [V, C] (``lm_weight()``, fp32), whose first CE columns are the
    ``IntNRowSTE`` forward of ``euc.weight``. None if ``euc`` carries no single
    ``IntNRowSTE`` parametrization, or if any entry of ``fp32(code) * scale`` differs from
    ``w_head[:, :CE]`` (then only the fp32 table computes the head's function)."""
    if w_head.dtype != torch.float32 or not parametrize.is_parametrized(euc, "weight"):
        return None
    plist = euc.parametrizations.weight
    if len(plist) != 1 or not isinstance(plist[0], IntNRowSTE):
        return None
    ste = plist[0]
    with torch.no_grad():
        w = plist.original.detach().float()
        V, CE = w.shape
        if w_head.shape[0] != V or w_head.shape[1] <= CE:
            return None
        # IntNRowSTE.forward's ops, in its order (same device: same IEEE results)
        s = w.abs().amax(dim=-1, keepdim=True).div(ste.hi).clamp(min=1e-8)
        code = (w / s).round().clamp(ste.lo, ste.hi)
        if not torch.equal(code * s, w_head[:, :CE].to(w.device)):
            return None
        codes = code.to(torch.int8).contiguous()
        if not torch.equal(codes.float() * s, w_head[:, :CE].to(w.device)):
            return None                              # int8 cannot hold the code range
        return (codes, s.view(V).contiguous(), w_head[:, CE:].contiguous())


@triton.jit
def _head_kernel(X, CODE, SC, WL, OUT, R, sx, sout, V: tl.constexpr, CE: tl.constexpr,
                 C: tl.constexpr, BC: tl.constexpr, BV: tl.constexpr):
    """OUT[r, v] = sum_k W[v, k] X[r, k], W[v, k] = fp32(CODE[v, k]) * SC[v] (k < CE) or
    WL[v, k - CE] (CE <= k < C), for the CTA's BV vocab rows and every query row r < R."""
    t = tl.program_id(0)
    v = t * BV + tl.arange(0, BV)
    mv = v < V
    k = tl.arange(0, BC)
    CL: tl.constexpr = C - CE
    s = tl.load(SC + v, mask=mv, other=0.0)
    ke = k < CE
    kl = (k >= CE) & (k < C)
    we = tl.load(CODE + v[:, None] * CE + k[None, :], mask=mv[:, None] & ke[None, :], other=0)
    wl = tl.load(WL + v[:, None] * CL + (k - CE)[None, :], mask=mv[:, None] & kl[None, :],
                 other=0.0)
    w = tl.where(ke[None, :], we.to(tl.float32) * s[:, None], wl)   # 0.0 past C
    for r in range(R):
        x = tl.load(X + r * sx + k, mask=k < C, other=0.0)
        tl.store(OUT + r * sout + v, tl.sum(w * x[None, :], 1), mask=mv)


def head_codes_logits(x: Tensor, codes: Tensor, scale: Tensor, w_lor: Tensor,
                      out: Tensor) -> Tensor:
    """``out`` [R, V] fp32 = ``x`` [R, C] fp32 @ W.T for the packed head (``head_codes_pack``).
    Rows of ``x`` and ``out`` must be contiguous (any row stride). Returns ``out``."""
    R, C = x.shape
    V, CE = codes.shape
    assert codes.dtype == torch.int8 and codes.is_contiguous()
    assert scale.shape == (V,) and scale.dtype == torch.float32 and scale.is_contiguous()
    assert w_lor.shape == (V, C - CE) and w_lor.dtype == torch.float32 and w_lor.is_contiguous()
    assert x.dtype == torch.float32 and x.stride(1) == 1 and out.shape == (R, V)
    assert out.dtype == torch.float32 and out.stride(1) == 1
    BC = triton.next_power_of_2(C)
    assert BC <= 1024, "the [BV, C] register tile is sized for d_model <= 1024"
    _head_kernel[(triton.cdiv(V, _BV),)](x, codes, scale, w_lor, out, R, x.stride(0),
                                         out.stride(0), V=V, CE=CE, C=C, BC=BC, BV=_BV,
                                         num_warps=4)
    return out
