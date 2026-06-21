"""Fused Triton kernel for AdEMAMixB1Zero (β1=0 AdEMAMix, arXiv:2409.03137).

One Triton program per BLOCK (256 elements) does, in-place, per param tensor:
    dequant(m2, nu)  →  EMA update  →  [CURE]  →  param update  →  requant(m2, nu)

Stability knobs (fused port): the coord-cap knobs, all elementwise / in-register, so the
fused path (≈3.7ms opt.step) supports the validated β1=0 deploy config without dropping to the
de-fused _foreach path (≈65ms). Ported, gated by constexpr flags (off → bit-identical to the
baseline kernel):
    EPS_INSIDE   — False = √(ν/bc2)+ε (eps-OUTSIDE, the cure's convergence-preserving denom);
                   True  = √(ν/bc2+ε) (legacy safety floor).
    HAS_SNR_GATE — soft per-coord SNR gate on the raw-g numerator: snr=|m2|/denom,
                   gate=floor+(1-floor)·clamp(snr/κ,0,1) [·g_coef]; gg = g·gate.
    HAS_COORD_CAP— per-coord stale-push cap: |α·m2| ≤ c·|gg| (clamp, sign preserved).
    HAS_UPD_CLIP — per-coord update clamp: update ∈ [-clip, clip] (post-/denom).
Order MATCHES the de-fused step exactly (gate g → cap α·m2 by c·|gated g| → +gated g → /denom →
clip → p·(1-lr·wd) - lr·update). Diagnostic counters (track_diag) are de-fused-only.

QUANT SCHEME (two modes, picked by DYNAMIC_QMAP constexpr):

  LINEAR  (DYNAMIC_QMAP=False, the original fused path): linear-int8 symmetric blockwise.
    dequant: val = code * (absmax / 127)         requant: code = round(val/(absmax/127)) ∈ int8.
    ν stored as sqrt(ν); positive sqrt(ν) floored to code 1 (NU_FLOOR) so a live lane's 2nd
    moment can't go exactly zero from another lane's larger absmax. UNIFORM quant of the
    heavy-tailed optimizer state crushes small coords; the sqrt-ν floor biases denom UP on
    small-ν coords → systematic under-stepping → a measured ~0.10-nat loss tax vs de-fused.

  DYNAMIC (DYNAMIC_QMAP=True): bnb's OWN non-linear dynamic qmap, so the
    fused path is the SAME quantizer as the validated de-fused reference (no systematic bias;
    residual = sub-1e-4 rounding noise). m2 → signed map (uint8), ν stored DIRECTLY (no sqrt,
    no floor — the dynamic map represents small ν faithfully) → unsigned map (uint8).
    dequant: val = code_map[code] * absmax     (code_map ∈ [-1,1] signed / [0,1] unsigned).
    requant: normalized = val/absmax; nearest entry in the 256-wide ascending map via an
    8-step vectorized binary search (≈10 LUT gathers; map stays in L2, ~1KB). absmax = block
    max(|val|). Same 2.03 B/param footprint as linear int8 (uint8 code + fp32 per-block absmax).

Compute fp32; params loaded/stored native dtype (bf16).
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
    m2_code_ptr,      # int8 (linear) / uint8 (dynamic) code for slow-EMA m2, in-place
    m2_amax_ptr,      # fp32 per-block absmax for m2, in-place
    nu_code_ptr,      # int8 (linear, sqrt-ν) / uint8 (dynamic, ν) code, in-place
    nu_amax_ptr,      # fp32 per-block absmax, in-place
    code_signed_ptr,  # 256 fp32 ascending dynamic map (signed)   — dynamic mode only
    code_unsigned_ptr,# 256 fp32 ascending dynamic map (unsigned) — dynamic mode only
    lr, beta2, beta3, alpha, eps, bc2, wd,   # fp32 scalars
    g_coef, snr_kappa, snr_floor, coord_cap, upd_clip,  # fp32 cure scalars
    is_init,          # 1 → state is zero (skip dequant), 0 → dequant existing code
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    EPS_INSIDE: tl.constexpr,     # True → √(ν/bc2+ε); False → √(ν/bc2)+ε (convergence-preserving)
    HAS_SNR_GATE: tl.constexpr,
    HAS_COORD_CAP: tl.constexpr,
    HAS_UPD_CLIP: tl.constexpr,
    HAS_GCOEF: tl.constexpr,
    DYNAMIC_QMAP: tl.constexpr,   # True → bnb dynamic non-linear map; False → linear int8
    NU_FLOOR: tl.constexpr,       # linear-only: floor positive sqrt(ν) to code 1
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
        m2_code = tl.load(m2_code_ptr + offs, mask=mask, other=0)
        nu_code = tl.load(nu_code_ptr + offs, mask=mask, other=0)
        if DYNAMIC_QMAP:
            # uint8 code indexes the ascending dynamic map; value already in [-1,1]/[0,1].
            m2_idx = m2_code.to(tl.int32)
            nu_idx = nu_code.to(tl.int32)
            m2 = tl.load(code_signed_ptr + m2_idx, mask=mask, other=0.0) * m2_amax
            nu = tl.load(code_unsigned_ptr + nu_idx, mask=mask, other=0.0) * nu_amax
        else:
            m2 = m2_code.to(tl.float32) * (m2_amax / 127.0)
            nu_sqrt = nu_code.to(tl.float32) * (nu_amax / 127.0)
            nu = nu_sqrt * nu_sqrt
    else:
        m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        nu = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # ── EMA updates (fp32) ──
    m2 = beta3 * m2 + (1.0 - beta3) * g
    nu = beta2 * nu + (1.0 - beta2) * g * g

    # ── denom (eps-inside legacy floor, or eps-outside true-Adam) ──
    if EPS_INSIDE:
        denom = tl.sqrt(nu / bc2 + eps)
    else:
        denom = tl.sqrt(nu / bc2) + eps

    # ── SNR gate on the raw-g numerator term (gg = gated g) ──
    if HAS_SNR_GATE:
        snr = tl.abs(m2) / denom / snr_kappa
        gate = tl.minimum(tl.maximum(snr, 0.0), 1.0)
        gate = snr_floor + (1.0 - snr_floor) * gate
        if HAS_GCOEF:
            gate = gate * g_coef
        gg = g * gate
    else:
        if HAS_GCOEF:
            gg = g * g_coef
        else:
            gg = g

    # ── per-coord stale-push cap on α·m2, relative to |gated g| (sign preserved) ──
    am = alpha * m2
    if HAS_COORD_CAP:
        cg = coord_cap * tl.abs(gg)
        am = tl.minimum(tl.maximum(am, -cg), cg)

    update = (am + gg) / denom

    # ── per-coord update clamp (post-/denom, pre-weight-decay; wd not clipped) ──
    if HAS_UPD_CLIP:
        update = tl.minimum(tl.maximum(update, -upd_clip), upd_clip)

    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    p = p * (1.0 - lr * wd) - lr * update
    tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)

    # ── requant ──
    if DYNAMIC_QMAP:
        # m2 → signed map ; nu → unsigned map. absmax = block max(|val|).
        m2_abs = tl.where(mask, tl.abs(m2), 0.0)
        nu_abs = tl.where(mask, tl.abs(nu), 0.0)   # nu ≥ 0 already
        new_m2_amax = tl.max(m2_abs, axis=0)
        new_nu_amax = tl.max(nu_abs, axis=0)
        tiny = 1e-20
        m2_norm = m2 / (new_m2_amax + tiny)        # ∈ [-1,1]
        nu_norm = nu / (new_nu_amax + tiny)        # ∈ [0,1]
        m2_q = _nearest_code(m2_norm, code_signed_ptr)
        nu_q = _nearest_code(nu_norm, code_unsigned_ptr)
        tl.store(m2_amax_ptr + pid, new_m2_amax)
        tl.store(nu_amax_ptr + pid, new_nu_amax)
        tl.store(m2_code_ptr + offs, m2_q.to(tl.uint8), mask=mask)
        tl.store(nu_code_ptr + offs, nu_q.to(tl.uint8), mask=mask)
    else:
        # linear-int8 symmetric, per-block absmax; ν stored as sqrt(ν).
        m2_abs = tl.where(mask, tl.abs(m2), 0.0)
        nu_sqrt = tl.sqrt(nu)
        nu_abs = tl.where(mask, nu_sqrt, 0.0)
        new_m2_amax = tl.max(m2_abs, axis=0)
        new_nu_sqrt_amax = tl.max(nu_abs, axis=0)
        tiny = 1e-20
        m2_scale = new_m2_amax / 127.0
        nu_scale = new_nu_sqrt_amax / 127.0
        m2_q = libdevice.round(m2 / (m2_scale + tiny))
        nu_q = libdevice.round(nu_sqrt / (nu_scale + tiny))
        m2_q = tl.minimum(tl.maximum(m2_q, -127.0), 127.0)
        nu_q = tl.minimum(tl.maximum(nu_q, 0.0), 127.0)
        if NU_FLOOR:
            nu_q = tl.where(
                (mask) & (nu_sqrt > 0.0) & (new_nu_sqrt_amax > 0.0) & (nu_q == 0.0),
                1.0,
                nu_q,
            )
        tl.store(m2_amax_ptr + pid, new_m2_amax)
        tl.store(nu_amax_ptr + pid, new_nu_sqrt_amax)
        tl.store(m2_code_ptr + offs, m2_q.to(tl.int8), mask=mask)
        tl.store(nu_code_ptr + offs, nu_q.to(tl.int8), mask=mask)


@triton.jit
def _nearest_code(v, map_ptr):
    """Nearest index in a 256-wide ASCENDING map to each element of v (vectorized).

    8-step binary search for the lower bound, then pick whichever of {lo-1, lo} is closer.
    Matches bnb's nearest-neighbour quantize_blockwise statistically (the dynamic map family),
    so it carries no systematic bias vs the de-fused reference — only sub-code rounding noise.
    """
    lo = tl.zeros_like(v).to(tl.int32)
    hi = lo + 255
    for _ in tl.static_range(8):
        mid = (lo + hi) // 2
        midval = tl.load(map_ptr + mid)
        cond = midval < v
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(cond, hi, mid)
    # lo = first index with map[lo] >= v. Compare with lo-1 for nearest.
    loc = tl.minimum(tl.maximum(lo, 0), 255)
    locm1 = tl.maximum(loc - 1, 0)
    d_lo = tl.abs(tl.load(map_ptr + loc) - v)
    d_m1 = tl.abs(tl.load(map_ptr + locm1) - v)
    code = tl.where(d_m1 <= d_lo, locm1, loc)
    return code


def fused_ademamix_b1zero_step(
    p, g, m2_code, m2_amax, nu_code, nu_amax,
    lr, beta2, beta3, alpha, eps, bc2, wd, is_init,
    *,
    eps_inside: bool = True,
    g_coef: float = 1.0,
    snr_kappa: float = 0.0,
    snr_floor: float = 0.1,
    coord_cap: float = 0.0,
    upd_clip: float = 0.0,
    dynamic_qmap: bool = False,
    nu_floor: bool = True,
    code_signed=None,
    code_unsigned=None,
):
    """Launch the fused kernel for one param tensor (all flat, contiguous).

    Stability knobs default to OFF (snr_kappa/coord_cap/upd_clip=0, eps_inside=True) → bit-identical
    to the baseline version. dynamic_qmap=False keeps the original linear-int8 path (nu_floor=True
    preserves the legacy code-1 floor). dynamic_qmap=True requires code_signed/code_unsigned
    (256-wide ascending fp32 maps on the param's device) and matches the de-fused quantizer.
    """
    n = p.numel()
    grid = (triton.cdiv(n, BLOCK),)
    if dynamic_qmap:
        if code_signed is None or code_unsigned is None:
            raise ValueError("dynamic_qmap=True requires code_signed and code_unsigned maps.")
        cs, cu = code_signed, code_unsigned
    else:
        # Pass m2_amax as a harmless non-null placeholder; the kernel never reads it when
        # DYNAMIC_QMAP is False (the branch is constexpr-eliminated).
        cs = cu = m2_amax
    _ademamix_b1zero_fused_kernel[grid](
        p, g, m2_code, m2_amax, nu_code, nu_amax,
        cs, cu,
        float(lr), float(beta2), float(beta3), float(alpha),
        float(eps), float(bc2), float(wd),
        float(g_coef), float(snr_kappa), float(snr_floor),
        float(coord_cap), float(upd_clip),
        1 if is_init else 0,
        n,
        BLOCK_SIZE=BLOCK,
        EPS_INSIDE=bool(eps_inside),
        HAS_SNR_GATE=(float(snr_kappa) > 0.0),
        HAS_COORD_CAP=(float(coord_cap) > 0.0),
        HAS_UPD_CLIP=(float(upd_clip) > 0.0),
        HAS_GCOEF=(float(g_coef) != 1.0),
        DYNAMIC_QMAP=bool(dynamic_qmap),
        NU_FLOOR=bool(nu_floor),
    )
