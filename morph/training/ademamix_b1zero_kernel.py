"""Fused Triton kernel for AdEMAMixB1Zero (β1=0 AdEMAMix, arXiv:2409.03137).

One Triton program per BLOCK (256 elements) does, in-place, per param tensor:
    dequant(m2, nu)  →  EMA update  →  param update (decoupled wd)  →  requant(m2, nu)

QUANT SCHEME: linear-int8 symmetric blockwise (per-256-block absmax).
    dequant: val = code * (absmax / 127)
    requant: new_absmax = max(|val|) over the 256-block
             code = round(val / (new_absmax/127 + tiny)) clamped to [-127, 127]   (int8)

ν is non-negative; symmetric int8 wastes the sign bit on ν but the fidelity gate
(cos > 0.999, rel < 5e-3) is met, so we keep one code path for both buffers.

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
    nu_code_ptr,      # int8 code for second-moment nu, in-place
    nu_amax_ptr,      # fp32 per-block absmax for nu, in-place
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
        nu_amax = tl.load(nu_amax_ptr + pid).to(tl.float32)
        m2_code = tl.load(m2_code_ptr + offs, mask=mask, other=0).to(tl.float32)
        nu_code = tl.load(nu_code_ptr + offs, mask=mask, other=0).to(tl.float32)
        m2 = m2_code * (m2_amax / 127.0)
        nu = nu_code * (nu_amax / 127.0)
    else:
        m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        nu = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ── EMA updates (fp32) ──
    m2 = beta3 * m2 + (1.0 - beta3) * g
    nu = beta2 * nu + (1.0 - beta2) * g * g

    # ── param update: update = (g + alpha*m2)/denom ; p = p*(1-lr*wd) - lr*update ──
    denom = tl.sqrt(nu / bc2) + eps
    update = (g + alpha * m2) / denom
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = p * (1.0 - lr * wd) - lr * update
    tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)

    # ── requant m2, nu (symmetric int8, per-block absmax) ──
    # masked lanes contribute 0 to the max (they hold dequant'd/updated junk otherwise,
    # but `other=0` on load + EMA keeps them at (1-b)*0 = 0 only if g masked → g=0 there,
    # so m2/nu on masked lanes = 0; still, mask the reduction to be safe).
    m2_abs = tl.where(mask, tl.abs(m2), 0.0)
    nu_abs = tl.where(mask, tl.abs(nu), 0.0)
    new_m2_amax = tl.max(m2_abs, axis=0)
    new_nu_amax = tl.max(nu_abs, axis=0)

    # scale = absmax/127 ; guard zero-block (scale 0 → all codes 0, absmax stored 0)
    tiny = 1e-20
    m2_scale = new_m2_amax / 127.0
    nu_scale = new_nu_amax / 127.0
    m2_q = libdevice.round(m2 / (m2_scale + tiny))
    nu_q = libdevice.round(nu / (nu_scale + tiny))
    m2_q = tl.minimum(tl.maximum(m2_q, -127.0), 127.0)
    nu_q = tl.minimum(tl.maximum(nu_q, -127.0), 127.0)

    tl.store(m2_amax_ptr + pid, new_m2_amax)
    tl.store(nu_amax_ptr + pid, new_nu_amax)
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
