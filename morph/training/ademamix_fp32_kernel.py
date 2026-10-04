"""One-pass Triton kernel for AdEMAMixB1Zero's fp32-state params (the ``_foreach`` path).

WHY (2026-10-04 training-speed pass). The no-decay group keeps 32-bit optimizer state
(``optimizer.py``: the embedding tables, 110.6 M params on the lead slot-loop arm), and
every sub-4096 tensor of the 8-bit group does too. ``AdEMAMixB1Zero.step`` updates those
with about 25 ``torch._foreach_*`` calls, each a full read-modify-write pass over every
element: 20.6 ms of GPU time per step in the profile, against 2.3 ms for the fused 8-bit
kernel over 189 M params. This kernel does the same update in ONE pass (read p, g, m2, nu;
write p, g, m2, nu).

BIT-IDENTICAL to the ``_foreach`` sequence, by construction and by test
(``tests/test_ademamix_fp32_kernel.py``). Each line below performs the SAME IEEE fp32
operation ATen's foreach kernel performs, in the same order. Measured on the 5090
(``ignore``-free probe: /home/wolfe/morph-scratch/perf/opt/aten_sem.py, 2026-10-04):

* ``_foreach_add_(a, b, alpha=s)``     = ``fma(f32(s), b, a)`` (one rounding)
* ``_foreach_addcmul_(a, b, c, value=s)`` = ``fma(b * c, f32(s), a)``
* ``_foreach_mul_(a, s)``              = ``a * f32(s)``
* ``_foreach_div(a, s)`` (scalar)      = ``a * f32(1 / s)`` (ATen multiplies by the
  reciprocal; ``1 / s`` is taken in double on the host)
* ``_foreach_div_(a, b)`` (list)       = IEEE ``a / b`` (``div_rn``)
* ``_foreach_sqrt_``                   = IEEE ``sqrt`` (``sqrt_rn``)
* ``_foreach_add_(a, s)`` (scalar)     = ``a + f32(s)``

Floating-point contraction is OFF for this kernel (``enable_fp_fusion=False``), so every
``*`` and ``+`` rounds on its own and the only fused multiply-adds are the explicit
``tl.fma`` calls that mirror ATen's. ``clamp`` is ``maximum`` / ``minimum``: the same on
every finite input; a NaN input is where they could differ (ATen propagates a NaN through
clamp), and a NaN in an fp32 optimizer state is already a dead run.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

BLOCK = 1024


@triton.jit
def _sign(x):
    one = tl.full(x.shape, 1.0, tl.float32)
    zero = tl.zeros(x.shape, tl.float32)
    return tl.where(x > 0.0, one, tl.where(x < 0.0, -one, zero))


@triton.jit
def _ademamix_fp32_kernel(
    p_ptr, g_ptr, m2_ptr, nu_ptr, n,
    b3, one_m_b3, b2, one_m_b2, inv_bc2, eps,
    inv_kappa, one_m_floor, snr_floor, g_coef,
    alpha, cap_c, clip_c, wd, neg_lr,
    EPS_INSIDE: tl.constexpr, HAS_GATE: tl.constexpr, HAS_GCOEF: tl.constexpr,
    HAS_CAP: tl.constexpr, HAS_CLIP: tl.constexpr, HAS_WD: tl.constexpr,
    WRITE_G: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    p = tl.load(p_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    m2 = tl.load(m2_ptr + offs, mask=mask, other=0.0)
    nu = tl.load(nu_ptr + offs, mask=mask, other=0.0)

    m2 = m2 * b3                                   # _foreach_mul_(m2s, beta3_t)
    m2 = tl.fma(one_m_b3, g, m2)                   # _foreach_add_(m2s, gs, alpha=1-b3)
    nu = nu * b2                                   # _foreach_mul_(nus, beta2)
    nu = tl.fma(g * g, one_m_b2, nu)               # _foreach_addcmul_(nus, gs, gs, 1-b2)
    den = nu * inv_bc2                             # _foreach_div(nus, bc2)
    if EPS_INSIDE:
        den = den + eps
        den = tl.sqrt_rn(den)
    else:
        den = tl.sqrt_rn(den)
        den = den + eps
    if HAS_GATE:
        gate = tl.abs(m2)
        gate = tl.div_rn(gate, den)                # _foreach_div_(gate, denoms)
        gate = gate * inv_kappa                    # _foreach_div_(gate, kappa)
        gate = tl.maximum(gate, 0.0)
        gate = tl.minimum(gate, 1.0)
        gate = gate * one_m_floor
        gate = gate + snr_floor
        if HAS_GCOEF:
            gate = gate * g_coef
        g = g * gate
    elif HAS_GCOEF:
        g = g * g_coef
    upd = m2 * alpha                               # _foreach_mul(m2s, alpha_t)
    if HAS_CAP:
        caps = tl.abs(g) * cap_c
        au = tl.abs(upd)
        over = au - caps
        over = tl.maximum(over, 0.0)
        au = au - over
        upd = _sign(upd) * au
    upd = upd + g                                  # _foreach_add_(upd, gs)
    upd = tl.div_rn(upd, den)                      # _foreach_div_(upd, denoms)
    if HAS_CLIP:
        upd = tl.maximum(upd, -clip_c)
        upd = tl.minimum(upd, clip_c)
    if HAS_WD:
        upd = tl.fma(wd, p, upd)                   # _foreach_add_(upd, p, alpha=wd)
    p = tl.fma(neg_lr, upd, p)                     # _foreach_add_(params, upd, alpha=-lr)

    tl.store(p_ptr + offs, p, mask=mask)
    tl.store(m2_ptr + offs, m2, mask=mask)
    tl.store(nu_ptr + offs, nu, mask=mask)
    if WRITE_G:
        # The _foreach path gates `gs` IN PLACE, and for an fp32 grad `p.grad.float()` IS
        # the grad, so the gated g lands in p.grad. Kept, so anything that reads the grad
        # after the step reads what it read before.
        tl.store(g_ptr + offs, g, mask=mask)


def ademamix_fp32_step(p, g, m2, nu, *, beta3, beta2, bc2, eps, eps_inside, g_snr_gate_kappa,
                       g_snr_gate_floor, g_coef, alpha, stale_push_cap_coord, update_clip,
                       wd, lr, write_g) -> None:
    """Update ``p``, ``m2``, ``nu`` (and ``g`` when ``write_g``) in place, exactly as the
    ``_foreach`` block of ``AdEMAMixB1Zero.step`` does for one fp32 tensor. All four are
    contiguous fp32 CUDA tensors of the same numel. Python scalars are passed as the fp32
    values ATen would use (``1 - beta3`` etc. are formed in double first, as there)."""
    for t in (p, g, m2, nu):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.numel() != p.numel():
            raise ValueError("ademamix_fp32_step wants contiguous fp32 tensors of one numel")
    n = p.numel()
    if n == 0:
        return
    has_gate = g_snr_gate_kappa > 0.0
    grid = (triton.cdiv(n, BLOCK),)
    _ademamix_fp32_kernel[grid](
        p, g, m2, nu, n,
        float(beta3), float(1.0 - beta3), float(beta2), float(1.0 - beta2),
        float(1.0 / bc2), float(eps),
        float(1.0 / g_snr_gate_kappa) if has_gate else 1.0,
        float(1.0 - g_snr_gate_floor), float(g_snr_gate_floor), float(g_coef),
        float(alpha), float(stale_push_cap_coord), float(update_clip), float(wd), float(-lr),
        EPS_INSIDE=bool(eps_inside), HAS_GATE=has_gate, HAS_GCOEF=(g_coef != 1.0),
        HAS_CAP=stale_push_cap_coord > 0.0, HAS_CLIP=update_clip > 0.0, HAS_WD=wd != 0.0,
        WRITE_G=bool(write_g), BLOCK_SIZE=BLOCK, enable_fp_fusion=False,
    )
