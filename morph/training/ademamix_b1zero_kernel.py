"""Fused Triton kernel for AdEMAMixB1Zero (β1=0 AdEMAMix, arXiv:2409.03137).

One Triton program per BLOCK (256 elements) does, in-place, per param tensor:
    dequant(m2, sqrt(ν))  →  EMA update  →  param update  →  requant(m2, sqrt(ν))

QUANT SCHEME: linear-int8 symmetric blockwise (per-256-block absmax).
    dequant: val = code * (absmax / 127)
    requant: new_absmax = max(|val|) over the 256-block
             code = round(val / (new_absmax/127 + tiny)) clamped to [-127, 127]   (int8)

ν is non-negative and used under a sqrt in the denominator, so the fused path stores
sqrt(ν), not ν, in its int8 state. This keeps the same one byte per element while
reducing the effective small-ν zero threshold from O(amax/127) to O(amax/127)^2.
Positive sqrt(ν) values are also floored to code 1 during requantization so a live
element's second moment cannot become exactly zero just because another element in
the same 256-block sets a much larger absmax.

Compute is fp32; params are loaded/stored as their native dtype (bf16).
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 256  # quant blocksize == Triton program tile (one program per block)


@triton.jit
def _ademamix_b1zero_fused_kernel(
    p_ptr,            # param (bf16/fp16/fp32), in-place
    g_ptr,            # gradient (fp32)
    m2_code_ptr,      # int8 code for slow-EMA m2, in-place
    m2_amax_ptr,      # fp32 per-block absmax for m2, in-place
    nu_code_ptr,      # int8 code for sqrt(second-moment nu), in-place
    nu_amax_ptr,      # fp32 per-block absmax for sqrt(nu), in-place
    lr, beta2, beta3, alpha, eps, bc2, wd,   # fp32 scalars
    is_init,          # 1 → state is zero (skip dequant), 0 → dequant existing code
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # ── dequant m2, nu (or zero on init) ──
    if is_init == 0:
        m2_amax = tl.load(m2_amax_ptr + pid).to(tl.float32)
        nu_sqrt_amax = tl.load(nu_amax_ptr + pid).to(tl.float32)
        m2_code = tl.load(m2_code_ptr + offs, mask=mask, other=0).to(tl.float32)
        nu_code = tl.load(nu_code_ptr + offs, mask=mask, other=0).to(tl.float32)
        m2 = m2_code * (m2_amax / 127.0)
        nu_sqrt = nu_code * (nu_sqrt_amax / 127.0)
        nu = nu_sqrt * nu_sqrt
    else:
        m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        nu = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ── EMA updates (fp32) ──
    m2 = beta3 * m2 + (1.0 - beta3) * g
    nu = beta2 * nu + (1.0 - beta2) * g * g

    # ── param update: update = (g + alpha*m2)/denom ; p = p*(1-lr*wd) - lr*update ──
    # eps INSIDE the sqrt (Adam-eps-inside convention). CRITICAL for linear-int8 nu:
    # small nu in a 256-block whose absmax is set by a larger element quantizes to int8 0
    # → dequants to EXACTLY 0. eps-outside gave denom=eps=1e-8 → update=g/1e-8≈2e4 explosion
    # on LIVE params (measured: 360/360 steps, root cause of the prune divergence). eps-inside
    # floors the underflowed denom to sqrt(eps)=1e-4 → update=g/1e-4≈2 (bounded). The sqrt-ν
    # state below preserves small-but-real second moments so eps is a safety floor, not the
    # primary representation for ordinary live small-ν lanes.
    denom = tl.sqrt(nu / bc2 + eps)
    update = (g + alpha * m2) / denom
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = p * (1.0 - lr * wd) - lr * update
    tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)

    # ── requant m2, sqrt(nu) (symmetric int8, per-block absmax) ──
    # masked lanes contribute 0 to the max (they hold dequant'd/updated junk otherwise,
    # but `other=0` on load + EMA keeps them at (1-b)*0 = 0 only if g masked → g=0 there,
    # so m2/nu on masked lanes = 0; still, mask the reduction to be safe).
    m2_abs = tl.where(mask, tl.abs(m2), 0.0)
    nu_sqrt = tl.sqrt(nu)
    nu_abs = tl.where(mask, nu_sqrt, 0.0)
    new_m2_amax = tl.max(m2_abs, axis=0)
    new_nu_sqrt_amax = tl.max(nu_abs, axis=0)

    # scale = absmax/127 ; guard zero-block (scale 0 → all codes 0, absmax stored 0)
    tiny = 1e-20
    m2_scale = new_m2_amax / 127.0
    nu_scale = new_nu_sqrt_amax / 127.0
    m2_q = libdevice.round(m2 / (m2_scale + tiny))
    nu_q = libdevice.round(nu_sqrt / (nu_scale + tiny))
    m2_q = tl.minimum(tl.maximum(m2_q, -127.0), 127.0)
    nu_q = tl.minimum(tl.maximum(nu_q, 0.0), 127.0)
    # Keep true zero blocks at 0, and let _mask_dead_state still zero pruned positions.
    nu_q = tl.where(
        (mask) & (nu_sqrt > 0.0) & (new_nu_sqrt_amax > 0.0) & (nu_q == 0.0),
        1.0,
        nu_q,
    )

    tl.store(m2_amax_ptr + pid, new_m2_amax)
    tl.store(nu_amax_ptr + pid, new_nu_sqrt_amax)
    tl.store(m2_code_ptr + offs, m2_q.to(tl.int8), mask=mask)
    tl.store(nu_code_ptr + offs, nu_q.to(tl.int8), mask=mask)


def fused_ademamix_b1zero_step(
    p, g, m2_code, m2_amax, nu_code, nu_amax,
    lr, beta2, beta3, alpha, eps, bc2, wd, is_init,
):
    """Launch the fused kernel for one param tensor (all flat, contiguous)."""
    n = p.numel()
    grid = (triton.cdiv(n, BLOCK),)
    _ademamix_b1zero_fused_kernel[grid](
        p, g, m2_code, m2_amax, nu_code, nu_amax,
        float(lr), float(beta2), float(beta3), float(alpha),
        float(eps), float(bc2), float(wd),
        1 if is_init else 0,
        n,
        BLOCK_SIZE=BLOCK,
    )
