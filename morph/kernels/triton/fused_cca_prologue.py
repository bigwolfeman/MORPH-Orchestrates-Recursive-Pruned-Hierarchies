"""Fused CCA attention prologue — Triton forward AND backward (sm_120 / Blackwell).

Collapses the fusable region of ``_CCABase._cca_project`` (everything AFTER the
down-projections and causal convolutions) into a small number of Triton launches.
The eager path is ~10 launches and launch-bound; this is 3 (Q, K, V).

Fused region, in order (byte-faithful to ``ignore/p1_prologue_harness.py``):
  1. QK-mean coupling:
        qk_mean_q[h] = 0.5 * (q_lat[h] + k_lat[h // n_rep])
        qk_mean_k[kv] = mean over the n_rep heads in group kv of qk_mean_q
                      = 0.5 * (mean_{g in kv} q_lat[g] + k_lat[kv])
  2. q = q_conv[h] + qk_mean_q[h]          (per query head)
     k = k_conv[kv] + qk_mean_k[kv]        (per kv head)
  3. RMSNorm over D (fp32) with learned q_norm.weight / k_norm.weight, eps=1e-6
  4. k *= exp(temp[kv])                    (per kv head)
  5. CoPE-RoPE (rotate_half), with the LAST n_skip_rope positions skipping rope
  6. k = repeat_interleave(k, n_rep) over heads -> [B, H, S, D]
  7. v = cat(v_curr, v_prev, dim=-1).reshape(B,S,Hkv,D).transpose -> repeat_interleave

Output: q, k, v each [B, H, S, D] (H query heads, GQA-expanded), dtype = q_lat.dtype.

Precision (fixed 2026-10-10, branch fix/cca-prologue-precision): the forward computes in
fp32 from the load to the store and rounds ONCE, to the output dtype, at the store. The
first version rounded every intermediate to bf16 (qk-mean, the pre-norm sum, the
normalized value, norm * weight, exp(temp), cos/sin, both RoPE products and their sum) to
copy a bf16-parameter reference. That made the kernel wrong in both dtypes it meets:
  * fp32 inputs (the fp32 decoders): q / k relative error vs fp64 3.8e-3 / 4.6e-3, where
    the eager reference gets 6e-8. A bf16 quantization of an fp32 model.
  * bf16 inputs under training autocast: the norm weights and temp are fp32 parameters, so
    the eager path computes norm * weight, temp and RoPE in fp32 (bf16 * fp32 promotes);
    the kernel was less exact than that path (q 3.9e-3 vs 2.6e-3), not a copy of it.
The backward already recomputed the forward in fp32 with no rounding, so it now
differentiates the function the forward computes. tests/test_cca_prologue_precision.py
holds the bounds.

Design for sm_120 (RTX 5090 / Blackwell):
  * num_stages=1, num_warps=8 (consumer Blackwell has NO TMA pipeline).
  * input-dtype loads and output-dtype stores (bf16 under autocast); fp32 for all
    arithmetic and ALL reductions (RMSNorm mean, qk-mean, GQA grad-accumulation).
  * D=32 head dim fits in registers; the whole per-row computation is register-resident.
  * One program = one (batch, seq-pos) row processing one head (Q) or one kv-head
    group (K / V). Reductions over D and over the n_rep group are in-register.
  * Branchless: n_skip_rope handling and group reductions are static / masked, no
    data-dependent host branching in the hot path.

Author: TileProver (Claude Code, Opus 4.8)
Date:   2026-05-31
"""

from __future__ import annotations

from ._pdl import pdl_opts

import math
from typing import Optional

import torch
from ._eager_flag import kernel_fence
from .kv_flat import _sum_partials
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


# ===========================================================================
# Triton kernels
# ===========================================================================

if TRITON_AVAILABLE:

    # -----------------------------------------------------------------------
    # Q forward: one program = one (b, s, head). D in registers.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_q_fwd_kernel(
        q_lat_ptr, k_lat_ptr, q_conv_ptr,      # [B, S, H*D] / [B, S, Hkv*D] / [B, S, H*D]
        wq_ptr,                                 # q_norm.weight [D]
        cos_ptr, sin_ptr,                       # [S, D]
        out_q_ptr,                              # [B, H, S, D]
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr,
        n_skip_rope,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)            # over B * S * H
        h = pid % H
        bs = pid // H
        s = bs % S
        b = bs // S
        kv = h // N_REP

        d = tl.arange(0, BLOCK_D)
        dmask = d < D

        # q_lat[b, s, h*D + d], k_lat[b, s, kv*D + d], q_conv[b, s, h*D + d]
        ql = tl.load(q_lat_ptr + (b * S + s) * (H * D) + h * D + d, mask=dmask, other=0.0)
        kl = tl.load(k_lat_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0)
        qc = tl.load(q_conv_ptr + (b * S + s) * (H * D) + h * D + d, mask=dmask, other=0.0)

        # fp32 from the load to the store; ONE rounding, to the output dtype, at the store
        # (see "Precision" in the module docstring). The backward recomputes this same fp32
        # function, so the gradient is the gradient of what the forward computed.
        a = qc.to(tl.float32) + 0.5 * (ql.to(tl.float32) + kl.to(tl.float32))

        # RMSNorm over D
        ms = tl.sum(a * a, axis=0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w = tl.load(wq_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        y = a * inv * w

        # RoPE (rotate_half). The last n_skip_rope positions skip it.
        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        half = D // 2
        # rotate_half(y)[i] = -y[i+half] if i<half else y[i-half]
        is_lo = d < half
        partner = tl.where(is_lo, d + half, d - half)
        y_part = tl.sum(tl.where(d[None, :] == partner[:, None], y[None, :], 0.0), axis=1)
        rot = tl.where(is_lo, -y_part, y_part)
        out = tl.where(do_rope, y * cos + rot * sin, y)

        out_off = ((b * H + h) * S + s) * D + d
        tl.store(out_q_ptr + out_off, out.to(out_q_ptr.dtype.element_ty), mask=dmask)

    # -----------------------------------------------------------------------
    # K forward: one program = one (b, s, kv). Reduces n_rep q_lat heads.
    # Writes the GQA-broadcast result to all n_rep output heads.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_k_fwd_kernel(
        q_lat_ptr, k_lat_ptr, k_conv_ptr,
        wk_ptr, temp_ptr,
        cos_ptr, sin_ptr,
        out_k_ptr,                                  # [B, H, S, D]
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr,
        n_skip_rope,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)            # over B * S * Hkv
        kv = pid % Hkv
        bs = pid // Hkv
        s = bs % S
        b = bs // S

        d = tl.arange(0, BLOCK_D)
        dmask = d < D

        kl = tl.load(k_lat_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0)
        kc = tl.load(k_conv_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0)

        # fp32 from the load to the store (as _cca_q_fwd_kernel):
        #   qk_mean_k[kv] = mean_g 0.5*(q_lat[g] + k_lat[kv]) = 0.5*(mean_g q_lat[g] + k_lat[kv])
        # summed in _cca_k_bwd_kernel's order, so the backward recomputes this exact value.
        qmean = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            qmean += tl.load(q_lat_ptr + (b * S + s) * (H * D) + h * D + d,
                             mask=dmask, other=0.0).to(tl.float32)
        qmean = qmean / N_REP
        a = kc.to(tl.float32) + 0.5 * (qmean + kl.to(tl.float32))

        ms = tl.sum(a * a, axis=0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w = tl.load(wk_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        etemp = tl.exp(tl.load(temp_ptr + kv).to(tl.float32))
        y = a * inv * w * etemp

        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        half = D // 2
        is_lo = d < half
        partner = tl.where(is_lo, d + half, d - half)
        y_part = tl.sum(tl.where(d[None, :] == partner[:, None], y[None, :], 0.0), axis=1)
        rot = tl.where(is_lo, -y_part, y_part)
        out = tl.where(do_rope, y * cos + rot * sin, y)

        outc = out.to(out_k_ptr.dtype.element_ty)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            tl.store(out_k_ptr + ((b * H + h) * S + s) * D + d, outc, mask=dmask)

    # -----------------------------------------------------------------------
    # V forward: gather cat(v_curr, v_prev) -> [B,H,S,D] with GQA broadcast.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_v_fwd_kernel(
        v_curr_ptr, v_prev_ptr,                     # [B, S, Hkv*(D//2)] each
        out_v_ptr,                                  # [B, H, S, D]
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, VHALF: tl.constexpr,   # VHALF = Hkv * (D//2)
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)            # over B * S * Hkv
        kv = pid % Hkv
        bs = pid // Hkv
        s = bs % S
        b = bs // S

        d = tl.arange(0, BLOCK_D)
        dmask = d < D

        # cat index into the 128-dim vector for (kv, d): idx = kv*D + d
        cat_idx = kv * D + d
        from_curr = cat_idx < VHALF
        curr_off = (b * S + s) * VHALF + cat_idx
        prev_off = (b * S + s) * VHALF + (cat_idx - VHALF)
        vc = tl.load(v_curr_ptr + curr_off, mask=dmask & from_curr, other=0.0)
        vp = tl.load(v_prev_ptr + prev_off, mask=dmask & (~from_curr), other=0.0)
        val = tl.where(from_curr, vc, vp)

        valc = val.to(out_v_ptr.dtype.element_ty)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            tl.store(out_v_ptr + ((b * H + h) * S + s) * D + d, valc, mask=dmask)

    # -----------------------------------------------------------------------
    # Q backward: grads wrt q_lat, k_lat (partial), q_conv, q_norm.weight.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_q_bwd_kernel(
        grad_q_ptr,                                 # [B, H, S, D]
        q_lat_ptr, k_lat_ptr, q_conv_ptr,
        wq_ptr, cos_ptr, sin_ptr,
        d_qlat_ptr, d_klat_q_ptr, d_qconv_ptr,      # outputs (klat partial from Q path)
        d_wq_ptr,                                   # [B*S*H, D] partial, reduced on host
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr,
        n_skip_rope,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        bs = pid // H
        s = bs % S
        b = bs // S
        kv = h // N_REP

        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        half = D // 2
        is_lo = d < half
        partner = tl.where(is_lo, d + half, d - half)

        # recompute forward up to y (normalized*weight)
        ql = tl.load(q_lat_ptr + (b * S + s) * (H * D) + h * D + d, mask=dmask, other=0.0).to(tl.float32)
        kl = tl.load(k_lat_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0).to(tl.float32)
        qc = tl.load(q_conv_ptr + (b * S + s) * (H * D) + h * D + d, mask=dmask, other=0.0).to(tl.float32)
        a = qc + 0.5 * (ql + kl)
        ms = tl.sum(a * a, axis=0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w = tl.load(wq_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        n = a * inv                                # normalized (no weight)
        y = n * w

        go = tl.load(grad_q_ptr + ((b * H + h) * S + s) * D + d, mask=dmask, other=0.0).to(tl.float32)

        # --- backward through RoPE ---
        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        # out = y*cos + rot(y)*sin ; rot is linear orthogonal-ish:
        # dy = go*cos + rot^T(go*sin). rot^T(g)[i] = g[i+half] if i<half else -g[i-half]
        gsin = go * sin
        gsin_part = tl.sum(tl.where(d[None, :] == partner[:, None], gsin[None, :], 0.0), axis=1)
        rotT = tl.where(is_lo, gsin_part, -gsin_part)
        dy_rope = go * cos + rotT
        dy = tl.where(do_rope, dy_rope, go)

        # --- backward through y = n * w ---
        dn = dy * w
        dw = dy * n                                # per-element weight grad (reduced on host)

        # --- backward through RMSNorm: n = a * inv, inv = (mean(a^2)+eps)^-1/2 ---
        # da = inv * (dn - (sum(dn*a)/D) * a * inv^2)
        sum_dn_a = tl.sum(dn * a, axis=0)
        da = inv * dn - (sum_dn_a / D) * (inv * inv * inv) * a

        # --- backward through a = qc + 0.5*(ql + kl) ---
        d_qconv = da
        d_ql = 0.5 * da
        d_kl = 0.5 * da

        off_qH = (b * S + s) * (H * D) + h * D + d
        off_kKv = (b * S + s) * (Hkv * D) + kv * D + d
        tl.store(d_qlat_ptr + off_qH, d_ql, mask=dmask)
        tl.store(d_qconv_ptr + off_qH, d_qconv, mask=dmask)
        # k_lat grad from the Q path: each (b,s,kv) is touched by n_rep q-heads.
        # write per (b,s,h) into a [B,S,H,D] buffer; host scatter-sums over the group.
        tl.store(d_klat_q_ptr + off_qH, d_kl, mask=dmask)
        tl.store(d_wq_ptr + pid * D + d, dw, mask=dmask)

    # -----------------------------------------------------------------------
    # K backward: grads wrt q_lat (partial), k_lat (partial), k_conv,
    # k_norm.weight, temp. Sums grad_k over the n_rep GQA output heads.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_k_bwd_kernel(
        grad_k_ptr,                                 # [B, H, S, D]
        q_lat_ptr, k_lat_ptr, k_conv_ptr,
        wk_ptr, temp_ptr, cos_ptr, sin_ptr,
        d_qlat_k_ptr, d_klat_ptr, d_kconv_ptr,      # [B,S,Hkv,D] buffers
        d_wk_ptr, d_temp_ptr,                       # [B*S*Hkv, D] and [B*S*Hkv]
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr,
        n_skip_rope,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)            # over B * S * Hkv
        kv = pid % Hkv
        bs = pid // Hkv
        s = bs % S
        b = bs // S

        d = tl.arange(0, BLOCK_D)
        dmask = d < D
        half = D // 2
        is_lo = d < half
        partner = tl.where(is_lo, d + half, d - half)

        kl = tl.load(k_lat_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0).to(tl.float32)
        kc = tl.load(k_conv_ptr + (b * S + s) * (Hkv * D) + kv * D + d, mask=dmask, other=0.0).to(tl.float32)
        qmean = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            ql = tl.load(q_lat_ptr + (b * S + s) * (H * D) + h * D + d, mask=dmask, other=0.0).to(tl.float32)
            qmean += ql
        qmean = qmean / N_REP

        a = kc + 0.5 * (qmean + kl)
        ms = tl.sum(a * a, axis=0) / D
        inv = 1.0 / tl.sqrt(ms + EPS)
        w = tl.load(wk_ptr + d, mask=dmask, other=0.0).to(tl.float32)
        n = a * inv
        yw = n * w                                  # before temp
        temp = tl.load(temp_ptr + kv).to(tl.float32)
        etemp = tl.exp(temp)
        y = yw * etemp                              # after temp, before rope

        # grad_k summed over the n_rep GQA output heads
        go = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            go += tl.load(grad_k_ptr + ((b * H + h) * S + s) * D + d, mask=dmask, other=0.0).to(tl.float32)

        # backward RoPE
        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s * D + d, mask=dmask, other=0.0).to(tl.float32)
        gsin = go * sin
        gsin_part = tl.sum(tl.where(d[None, :] == partner[:, None], gsin[None, :], 0.0), axis=1)
        rotT = tl.where(is_lo, gsin_part, -gsin_part)
        dy_rope = go * cos + rotT
        dy = tl.where(do_rope, dy_rope, go)         # grad wrt y (after temp)

        # backward temp: y = yw * etemp
        d_yw = dy * etemp
        # dtemp = sum(dy * yw * etemp) = sum(dy * y)
        d_temp_local = tl.sum(dy * y, axis=0)

        # backward y = n * w
        dn = d_yw * w
        dw = d_yw * n

        # backward RMSNorm
        sum_dn_a = tl.sum(dn * a, axis=0)
        da = inv * dn - (sum_dn_a / D) * (inv * inv * inv) * a

        # backward a = kc + 0.5*(qmean + kl); qmean = mean_g q_lat[g]
        d_kconv = da
        d_kl = 0.5 * da
        d_qmean = 0.5 * da
        d_ql_each = d_qmean / N_REP                 # each q-head in group gets this

        off_kKv = (b * S + s) * (Hkv * D) + kv * D + d
        tl.store(d_kconv_ptr + off_kKv, d_kconv, mask=dmask)
        tl.store(d_klat_ptr + off_kKv, d_kl, mask=dmask)
        tl.store(d_wk_ptr + pid * D + d, dw, mask=dmask)
        tl.store(d_temp_ptr + pid, d_temp_local)
        # q_lat grad from K path: same value for every q-head in the group
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            tl.store(d_qlat_k_ptr + (b * S + s) * (H * D) + h * D + d, d_ql_each, mask=dmask)

    # -----------------------------------------------------------------------
    # V backward: scatter grad_v (summed over n_rep heads) back to v_curr/v_prev.
    # -----------------------------------------------------------------------
    @triton.jit
    def _cca_v_bwd_kernel(
        grad_v_ptr,                                 # [B, H, S, D]
        d_vcurr_ptr, d_vprev_ptr,                   # [B, S, Hkv*(D//2)]
        B, S,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, VHALF: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        kv = pid % Hkv
        bs = pid // Hkv
        s = bs % S
        b = bs // S

        d = tl.arange(0, BLOCK_D)
        dmask = d < D

        gv = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for r in tl.static_range(N_REP):
            h = kv * N_REP + r
            gv += tl.load(grad_v_ptr + ((b * H + h) * S + s) * D + d, mask=dmask, other=0.0).to(tl.float32)

        cat_idx = kv * D + d
        from_curr = cat_idx < VHALF
        curr_off = (b * S + s) * VHALF + cat_idx
        prev_off = (b * S + s) * VHALF + (cat_idx - VHALF)
        tl.store(d_vcurr_ptr + curr_off, gv, mask=dmask & from_curr)
        tl.store(d_vprev_ptr + prev_off, gv, mask=dmask & (~from_curr))


# ===========================================================================
# Tile config (sm_120)
# ===========================================================================

def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


_LAUNCH = dict(num_stages=1, num_warps=8)


# ===========================================================================
# Python wrappers
# ===========================================================================

def _cca_prologue_forward(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
                          wq, wk, temp, cos, sin,
                          H, Hkv, D, n_rep, n_skip_rope, eps):
    B, S, _ = q_lat.shape
    VHALF = v_curr.shape[-1]
    BLOCK_D = _next_pow2(D)
    dev = q_lat.device
    odt = q_lat.dtype

    out_q = torch.empty(B, H, S, D, device=dev, dtype=odt)
    out_k = torch.empty(B, H, S, D, device=dev, dtype=odt)
    out_v = torch.empty(B, H, S, D, device=dev, dtype=odt)

    _cca_q_fwd_kernel[(B * S * H,)](
        q_lat, k_lat, q_conv, wq, cos, sin, out_q,
        B, S, H, Hkv, D, n_rep, eps, n_skip_rope, BLOCK_D=BLOCK_D, **_LAUNCH,
    )
    _cca_k_fwd_kernel[(B * S * Hkv,)](
        q_lat, k_lat, k_conv, wk, temp, cos, sin, out_k,
        B, S, H, Hkv, D, n_rep, eps, n_skip_rope, BLOCK_D=BLOCK_D, **_LAUNCH,
    )
    _cca_v_fwd_kernel[(B * S * Hkv,)](
        v_curr, v_prev, out_v,
        B, S, H, Hkv, D, n_rep, VHALF, BLOCK_D=BLOCK_D, **_LAUNCH,
    )
    return out_q, out_k, out_v


def _cca_prologue_backward(grad_q, grad_k, grad_v,
                           q_lat, k_lat, q_conv, k_conv,
                           wq, wk, temp, cos, sin,
                           H, Hkv, D, n_rep, n_skip_rope, eps):
    B, S, _ = q_lat.shape
    BLOCK_D = _next_pow2(D)
    dev = q_lat.device

    grad_q = grad_q.contiguous()
    grad_k = grad_k.contiguous()
    grad_v = grad_v.contiguous()

    f32 = torch.float32
    # Q-path outputs
    d_qlat = torch.empty(B, S, H * D, device=dev, dtype=f32)
    d_qconv = torch.empty(B, S, H * D, device=dev, dtype=f32)
    d_klat_q = torch.empty(B, S, H * D, device=dev, dtype=f32)   # per q-head, grouped on host
    d_wq = torch.empty(B * S * H, D, device=dev, dtype=f32)
    # K-path outputs
    d_qlat_k = torch.empty(B, S, H * D, device=dev, dtype=f32)   # per q-head (broadcast)
    d_klat = torch.empty(B, S, Hkv * D, device=dev, dtype=f32)
    d_kconv = torch.empty(B, S, Hkv * D, device=dev, dtype=f32)
    d_wk = torch.empty(B * S * Hkv, D, device=dev, dtype=f32)
    d_temp = torch.empty(B * S * Hkv, device=dev, dtype=f32)

    _cca_q_bwd_kernel[(B * S * H,)](
        grad_q, q_lat, k_lat, q_conv, wq, cos, sin,
        d_qlat, d_klat_q, d_qconv, d_wq,
        B, S, H, Hkv, D, n_rep, eps, n_skip_rope, BLOCK_D=BLOCK_D, **_LAUNCH,
    )
    _cca_k_bwd_kernel[(B * S * Hkv,)](
        grad_k, q_lat, k_lat, k_conv, wk, temp, cos, sin,
        d_qlat_k, d_klat, d_kconv, d_wk, d_temp,
        B, S, H, Hkv, D, n_rep, eps, n_skip_rope, BLOCK_D=BLOCK_D, **_LAUNCH,
    )

    # --- combine grads on host (fp32 accumulation) ---
    # q_lat total grad = Q-path (per head) + K-path (per head, from qmean)
    d_q_lat = d_qlat + d_qlat_k                                  # [B,S,H*D]

    # k_lat total grad = K-path (per kv) + Q-path (sum the n_rep q-heads in group)
    d_klat_q_grp = d_klat_q.reshape(B, S, Hkv, n_rep, D).sum(dim=3).reshape(B, S, Hkv * D)
    d_k_lat = d_klat + d_klat_q_grp                             # [B,S,Hkv*D]

    d_q_conv = d_qconv
    d_k_conv = d_kconv

    d_wq_total = d_wq.reshape(B * S * H, D).sum(dim=0)
    d_wk_total = d_wk.reshape(B * S * Hkv, D).sum(dim=0)
    d_temp_total = d_temp.reshape(B, S, Hkv).sum(dim=(0, 1))    # [Hkv]

    # V backward
    vhalf = v_half_holder[0]
    d_vcurr = torch.empty(B, S, vhalf, device=dev, dtype=f32)
    d_vprev = torch.empty(B, S, vhalf, device=dev, dtype=f32)
    _cca_v_bwd_kernel[(B * S * Hkv,)](
        grad_v, d_vcurr, d_vprev,
        B, S, H, Hkv, D, n_rep, vhalf, BLOCK_D=BLOCK_D, **_LAUNCH,
    )

    return (d_q_lat, d_k_lat, d_q_conv, d_k_conv, d_vcurr, d_vprev,
            d_wq_total, d_wk_total, d_temp_total)


# module-level holder so backward knows VHALF (set in autograd.forward)
v_half_holder = [0]


class _FusedCCAPrologue(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
                wq, wk, temp, cos, sin,
                H, Hkv, D, n_rep, n_skip_rope, eps):
        q_lat = q_lat.contiguous(); k_lat = k_lat.contiguous()
        q_conv = q_conv.contiguous(); k_conv = k_conv.contiguous()
        v_curr = v_curr.contiguous(); v_prev = v_prev.contiguous()
        cos = cos.contiguous(); sin = sin.contiguous()

        v_half_holder[0] = v_curr.shape[-1]

        out_q, out_k, out_v = _cca_prologue_forward(
            q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
            wq, wk, temp, cos, sin, H, Hkv, D, n_rep, n_skip_rope, eps,
        )
        ctx.save_for_backward(q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin)
        ctx.dims = (H, Hkv, D, n_rep, n_skip_rope, eps)
        ctx.vhalf = v_curr.shape[-1]
        return out_q, out_k, out_v

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin = ctx.saved_tensors
        H, Hkv, D, n_rep, n_skip_rope, eps = ctx.dims
        v_half_holder[0] = ctx.vhalf
        (d_q_lat, d_k_lat, d_q_conv, d_k_conv, d_vcurr, d_vprev,
         d_wq, d_wk, d_temp) = _cca_prologue_backward(
            grad_q, grad_k, grad_v,
            q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin,
            H, Hkv, D, n_rep, n_skip_rope, eps,
        )
        # cast grads back to input dtypes
        def cast(g, ref):
            return g.to(ref.dtype) if g is not None else None
        return (cast(d_q_lat, q_lat), cast(d_k_lat, k_lat),
                cast(d_q_conv, q_conv), cast(d_k_conv, k_conv),
                cast(d_vcurr, q_lat), cast(d_vprev, q_lat),
                cast(d_wq, wq), cast(d_wk, wk), cast(d_temp, temp),
                None, None, None, None, None, None, None, None)


# ===========================================================================
# model.cca_prologue_tiled — the same prologue, row-tiled: ONE forward and ONE backward
# launch, BR (b, s) rows x every head per program.
#
# Why (blockfuse census 2026-10-08, FAST2 graph replay): the per-(row, head) kernels above
# run one 64-element program per (b, s, h) with 8 warps; at the prelude shape (7680 rows) the
# Q forward alone took 294 us for ~27 MB of traffic, and the whole prologue was 25 ms of a
# 354 ms step, launch geometry rather than bandwidth. Here a program holds a [BR, D] tile per
# head, the RoPE partner comes from a second load at the rotated column (instead of an
# O(D^2) masked sum), the backward combines the Q-path and K-path grads in registers and
# writes every input grad once in its final dtype (no host-side adds, group sums or casts),
# and the inputs are read through their strides (the fused GEMM's slices and the conv's
# transposed output need no `.contiguous()` copy).
#
# Arithmetic: op for op the kernels above (fp32 throughout, one rounding at the store; the
# backward is fp32 from a recomputed forward). The differences are reduction
# orders only: the RMSNorm sums over D (a [BR, D] tile reduces in a different order than
# a [D] vector), and the norm-weight / temp grads, summed per program and then over
# programs instead of over (row, head). Not bit-identical; tests/test_cca_prologue_tiled.py
# scores both paths against an fp64 reference.
# ===========================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _rows_ptrs(ptr, b, s, sb, ss, sc, col):
        """[BR, D] pointer tile: rows (b, s) [BR], columns col [D], strides in elements."""
        return ptr + (b * sb + s * ss)[:, None] + (col * sc)[None, :]

    @triton.jit
    def _rope_f32(y, y_p, cos, sin, is_lo, do_rope):
        """_cca_q_fwd_kernel's RoPE on a tile, fp32: y_p is y at the rotated column."""
        rot = tl.where(is_lo[None, :], -y_p, y_p)
        return tl.where(do_rope[:, None], y * cos + rot * sin, y)

    @triton.jit
    def _cca_rows_fwd_kernel(
        ql_ptr, kl_ptr, qc_ptr, kc_ptr, vc_ptr, vp_ptr,
        wq_ptr, wk_ptr, temp_ptr, cos_ptr, sin_ptr,
        oq_ptr, ok_ptr, ov_ptr,                      # [B, H, S, D] contiguous
        n_rows, S, n_skip_rope,
        s_ql_b, s_ql_s, s_ql_c, s_kl_b, s_kl_s, s_kl_c,
        s_qc_b, s_qc_s, s_qc_c, s_kc_b, s_kc_s, s_kc_c,
        s_vc_b, s_vc_s, s_vc_c, s_vp_b, s_vp_s, s_vp_c,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr, VHALF: tl.constexpr,
        BR: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
        pid = tl.program_id(0)
        r = pid * BR + tl.arange(0, BR)
        rmask = r < n_rows
        b = r // S
        s = r % S
        d = tl.arange(0, D)
        half = D // 2
        is_lo = d < half
        dp = tl.where(is_lo, d + half, d - half)          # rotate_half partner column
        m2 = rmask[:, None]
        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        wq = tl.load(wq_ptr + d).to(tl.float32)
        wq_p = tl.load(wq_ptr + dp).to(tl.float32)
        wk = tl.load(wk_ptr + d).to(tl.float32)
        wk_p = tl.load(wk_ptr + dp).to(tl.float32)
        o_row = (b * H * S + s) * D                        # + h*S*D + d

        # fp32 from the load to the store, one rounding at the store (module docstring,
        # "Precision"); each value op for op as the per-(row, head) kernels compute it.
        for kv in tl.static_range(Hkv):
            klf = tl.load(_rows_ptrs(kl_ptr, b, s, s_kl_b, s_kl_s, s_kl_c, kv * D + d),
                          mask=m2, other=0.0).to(tl.float32)
            klf_p = tl.load(_rows_ptrs(kl_ptr, b, s, s_kl_b, s_kl_s, s_kl_c, kv * D + dp),
                            mask=m2, other=0.0).to(tl.float32)
            qmean = tl.zeros((BR, D), dtype=tl.float32)
            qmean_p = tl.zeros((BR, D), dtype=tl.float32)
            for rr in tl.static_range(N_REP):
                h = kv * N_REP + rr
                ql = tl.load(_rows_ptrs(ql_ptr, b, s, s_ql_b, s_ql_s, s_ql_c, h * D + d),
                             mask=m2, other=0.0).to(tl.float32)
                ql_p = tl.load(_rows_ptrs(ql_ptr, b, s, s_ql_b, s_ql_s, s_ql_c, h * D + dp),
                               mask=m2, other=0.0).to(tl.float32)
                qc = tl.load(_rows_ptrs(qc_ptr, b, s, s_qc_b, s_qc_s, s_qc_c, h * D + d),
                             mask=m2, other=0.0).to(tl.float32)
                qc_p = tl.load(_rows_ptrs(qc_ptr, b, s, s_qc_b, s_qc_s, s_qc_c, h * D + dp),
                               mask=m2, other=0.0).to(tl.float32)
                # Q: _cca_q_fwd_kernel, per element
                a = qc + 0.5 * (ql + klf)
                a_p = qc_p + 0.5 * (ql_p + klf_p)
                inv = 1.0 / tl.sqrt(tl.sum(a * a, axis=1) / D + EPS)
                y = a * inv[:, None] * wq[None, :]
                y_p = a_p * inv[:, None] * wq_p[None, :]
                out = _rope_f32(y, y_p, cos, sin, is_lo, do_rope)
                tl.store(oq_ptr + o_row[:, None] + h * S * D + d[None, :],
                         out.to(oq_ptr.dtype.element_ty), mask=m2)
                # K: the group mean of q_lat (_cca_k_fwd_kernel)
                qmean += ql
                qmean_p += ql_p

            kc = tl.load(_rows_ptrs(kc_ptr, b, s, s_kc_b, s_kc_s, s_kc_c, kv * D + d),
                         mask=m2, other=0.0).to(tl.float32)
            kc_p = tl.load(_rows_ptrs(kc_ptr, b, s, s_kc_b, s_kc_s, s_kc_c, kv * D + dp),
                           mask=m2, other=0.0).to(tl.float32)
            a = kc + 0.5 * (qmean / N_REP + klf)
            a_p = kc_p + 0.5 * (qmean_p / N_REP + klf_p)
            inv = 1.0 / tl.sqrt(tl.sum(a * a, axis=1) / D + EPS)
            etemp = tl.exp(tl.load(temp_ptr + kv).to(tl.float32))
            y = a * inv[:, None] * wk[None, :] * etemp
            y_p = a_p * inv[:, None] * wk_p[None, :] * etemp
            outk = _rope_f32(y, y_p, cos, sin, is_lo, do_rope).to(ok_ptr.dtype.element_ty)
            # V: cat(v_curr, v_prev) column kv*D + d, GQA-broadcast like K
            ci = kv * D + d
            from_c = ci < VHALF
            vcv = tl.load(_rows_ptrs(vc_ptr, b, s, s_vc_b, s_vc_s, s_vc_c, ci),
                          mask=m2 & from_c[None, :], other=0.0)
            vpv = tl.load(_rows_ptrs(vp_ptr, b, s, s_vp_b, s_vp_s, s_vp_c, ci - VHALF),
                          mask=m2 & (~from_c)[None, :], other=0.0)
            outv = tl.where(from_c[None, :], vcv, vpv).to(ov_ptr.dtype.element_ty)
            for rr in tl.static_range(N_REP):
                h = kv * N_REP + rr
                optr = o_row[:, None] + h * S * D + d[None, :]
                tl.store(ok_ptr + optr, outk, mask=m2)
                tl.store(ov_ptr + optr, outv, mask=m2)

    @triton.jit
    def _rope_vjp(go, go_p, cos, sin, sin_p, is_lo, do_rope):
        """Backward of out = y*cos + rotate_half(y)*sin on a tile (fp32); go_p, sin_p at the
        rotated column. Equals _cca_q_bwd_kernel's masked-sum form exactly."""
        gsin_p = go_p * sin_p
        rot_t = tl.where(is_lo[None, :], gsin_p, -gsin_p)
        return tl.where(do_rope[:, None], go * cos + rot_t, go)

    @triton.jit
    def _cca_rows_bwd_kernel(
        gq_ptr, gk_ptr, gv_ptr,                      # [B, H, S, D] contiguous
        ql_ptr, kl_ptr, qc_ptr, kc_ptr,
        wq_ptr, wk_ptr, temp_ptr, cos_ptr, sin_ptr,
        dql_ptr, dkl_ptr, dqc_ptr, dkc_ptr, dvc_ptr, dvp_ptr,
        dwq_part_ptr, dwk_part_ptr, dtemp_part_ptr,  # [n_prog, D], [n_prog, D], [n_prog, Hkv]
        n_rows, S, n_skip_rope,
        s_ql_b, s_ql_s, s_ql_c, s_kl_b, s_kl_s, s_kl_c,
        s_qc_b, s_qc_s, s_qc_c, s_kc_b, s_kc_s, s_kc_c,
        s_dql_b, s_dql_s, s_dql_c, s_dkl_b, s_dkl_s, s_dkl_c,
        s_dqc_b, s_dqc_s, s_dqc_c, s_dkc_b, s_dkc_s, s_dkc_c,
        s_dvc_b, s_dvc_s, s_dvc_c, s_dvp_b, s_dvp_s, s_dvp_c,
        H: tl.constexpr, Hkv: tl.constexpr, D: tl.constexpr,
        N_REP: tl.constexpr, EPS: tl.constexpr, VHALF: tl.constexpr,
        BR: tl.constexpr, KV_N: tl.constexpr,
        PDL: tl.constexpr = False,
    ):
        if PDL:
            tl.extra.cuda.gdc_wait()
        pid = tl.program_id(0)
        r = pid * BR + tl.arange(0, BR)
        rmask = r < n_rows
        b = r // S
        s = r % S
        d = tl.arange(0, D)
        half = D // 2
        is_lo = d < half
        dp = tl.where(is_lo, d + half, d - half)
        m2 = rmask[:, None]
        do_rope = s < (S - n_skip_rope)
        cos = tl.load(cos_ptr + s[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s[:, None] * D + d[None, :], mask=m2, other=0.0).to(tl.float32)
        sin_p = tl.load(sin_ptr + s[:, None] * D + dp[None, :], mask=m2, other=0.0).to(tl.float32)
        wq = tl.load(wq_ptr + d).to(tl.float32)
        wk = tl.load(wk_ptr + d).to(tl.float32)
        o_row = (b * H * S + s) * D
        dwq = tl.zeros((D,), dtype=tl.float32)
        dwk = tl.zeros((D,), dtype=tl.float32)
        odt = dql_ptr.dtype.element_ty

        # KV_N == Hkv: one program walks every kv group (program_id(1) == 0, grid y == 1).
        # KV_N == 1 (model.kv_flat_kernels): grid y == Hkv, one program per (row block, kv group).
        for kvi in tl.static_range(KV_N):
            kv = tl.program_id(1) + kvi
            kl = tl.load(_rows_ptrs(kl_ptr, b, s, s_kl_b, s_kl_s, s_kl_c, kv * D + d),
                         mask=m2, other=0.0).to(tl.float32)
            # ---- K path (_cca_k_bwd_kernel) ----
            kc = tl.load(_rows_ptrs(kc_ptr, b, s, s_kc_b, s_kc_s, s_kc_c, kv * D + d),
                         mask=m2, other=0.0).to(tl.float32)
            qmean = tl.zeros((BR, D), dtype=tl.float32)
            go = tl.zeros((BR, D), dtype=tl.float32)
            go_p = tl.zeros((BR, D), dtype=tl.float32)
            gv = tl.zeros((BR, D), dtype=tl.float32)
            for rr in tl.static_range(N_REP):
                h = kv * N_REP + rr
                qmean += tl.load(_rows_ptrs(ql_ptr, b, s, s_ql_b, s_ql_s, s_ql_c, h * D + d),
                                 mask=m2, other=0.0).to(tl.float32)
                gptr = o_row[:, None] + h * S * D
                go += tl.load(gk_ptr + gptr + d[None, :], mask=m2, other=0.0).to(tl.float32)
                go_p += tl.load(gk_ptr + gptr + dp[None, :], mask=m2, other=0.0).to(tl.float32)
                gv += tl.load(gv_ptr + gptr + d[None, :], mask=m2, other=0.0).to(tl.float32)
            qmean = qmean / N_REP
            a = kc + 0.5 * (qmean + kl)
            inv = 1.0 / tl.sqrt(tl.sum(a * a, axis=1) / D + EPS)
            n = a * inv[:, None]
            etemp = tl.exp(tl.load(temp_ptr + kv).to(tl.float32))
            y = n * wk[None, :] * etemp
            dy = _rope_vjp(go, go_p, cos, sin, sin_p, is_lo, do_rope)
            d_yw = dy * etemp
            tl.store(dtemp_part_ptr + pid * Hkv + kv, tl.sum(tl.sum(dy * y, axis=1), axis=0))
            dn = d_yw * wk[None, :]
            dwk += tl.sum(d_yw * n, axis=0)
            sum_dn_a = tl.sum(dn * a, axis=1)
            da_k = inv[:, None] * dn - (sum_dn_a / D)[:, None] * (inv * inv * inv)[:, None] * a
            d_ql_each = (0.5 * da_k) / N_REP
            tl.store(_rows_ptrs(dkc_ptr, b, s, s_dkc_b, s_dkc_s, s_dkc_c, kv * D + d),
                     da_k.to(odt), mask=m2)
            # ---- V: the GQA group sum of grad_v into the cat's halves ----
            ci = kv * D + d
            from_c = ci < VHALF
            tl.store(_rows_ptrs(dvc_ptr, b, s, s_dvc_b, s_dvc_s, s_dvc_c, ci), gv.to(odt),
                     mask=m2 & from_c[None, :])
            tl.store(_rows_ptrs(dvp_ptr, b, s, s_dvp_b, s_dvp_s, s_dvp_c, ci - VHALF),
                     gv.to(odt), mask=m2 & (~from_c)[None, :])
            # ---- Q path per head (_cca_q_bwd_kernel), combined with the K path ----
            d_kl_q = tl.zeros((BR, D), dtype=tl.float32)
            for rr in tl.static_range(N_REP):
                h = kv * N_REP + rr
                ql = tl.load(_rows_ptrs(ql_ptr, b, s, s_ql_b, s_ql_s, s_ql_c, h * D + d),
                             mask=m2, other=0.0).to(tl.float32)
                qc = tl.load(_rows_ptrs(qc_ptr, b, s, s_qc_b, s_qc_s, s_qc_c, h * D + d),
                             mask=m2, other=0.0).to(tl.float32)
                a = qc + 0.5 * (ql + kl)
                inv = 1.0 / tl.sqrt(tl.sum(a * a, axis=1) / D + EPS)
                n = a * inv[:, None]
                gptr = o_row[:, None] + h * S * D
                gq = tl.load(gq_ptr + gptr + d[None, :], mask=m2, other=0.0).to(tl.float32)
                gq_p = tl.load(gq_ptr + gptr + dp[None, :], mask=m2, other=0.0).to(tl.float32)
                dy = _rope_vjp(gq, gq_p, cos, sin, sin_p, is_lo, do_rope)
                dn = dy * wq[None, :]
                dwq += tl.sum(dy * n, axis=0)
                sum_dn_a = tl.sum(dn * a, axis=1)
                da = inv[:, None] * dn - (sum_dn_a / D)[:, None] * (inv * inv * inv)[:, None] * a
                tl.store(_rows_ptrs(dqc_ptr, b, s, s_dqc_b, s_dqc_s, s_dqc_c, h * D + d),
                         da.to(odt), mask=m2)
                tl.store(_rows_ptrs(dql_ptr, b, s, s_dql_b, s_dql_s, s_dql_c, h * D + d),
                         (0.5 * da + d_ql_each).to(odt), mask=m2)
                d_kl_q += 0.5 * da
            tl.store(_rows_ptrs(dkl_ptr, b, s, s_dkl_b, s_dkl_s, s_dkl_c, kv * D + d),
                     (0.5 * da_k + d_kl_q).to(odt), mask=m2)

        slot = pid * tl.num_programs(1) + tl.program_id(1)
        tl.store(dwq_part_ptr + slot * D + d, dwq)
        tl.store(dwk_part_ptr + slot * D + d, dwk)


_ROWS_BR = 16
_ROWS_LAUNCH = dict(num_stages=1, num_warps=4)
# model.kv_flat_kernels: fewer rows per program (more programs at the core pass's 1,536 rows),
# partials summed by one fixed-order kernel. Tile from lab/perf/bench_kv_flat.py.
_ROWS_BR_FLAT = 16
# below this many row blocks (170 SMs) the kv groups also split across programs (grid y)
_ROWS_KV_SPLIT_BELOW = 192
_ROWS_LAUNCH_FLAT = dict(num_stages=1, num_warps=2)


def _strides3(t: Tensor) -> tuple[int, int, int]:
    return t.stride(0), t.stride(1), t.stride(2)


def _rows_fwd_impl(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
                   H, Hkv, D, n_rep, n_skip_rope, eps):
    """The row-tiled prologue forward: ``(out_q, out_k, out_v)``. ``_FusedCCAPrologueRows`` and
    the custom op (``block_ops``, model.block_custom_ops) both call it."""
    B, S, _ = q_lat.shape
    n_rows = B * S
    dev, odt = q_lat.device, q_lat.dtype
    out_q = torch.empty(B, H, S, D, device=dev, dtype=odt)
    out_k = torch.empty(B, H, S, D, device=dev, dtype=odt)
    out_v = torch.empty(B, H, S, D, device=dev, dtype=odt)
    vhalf = v_curr.shape[-1]
    _cca_rows_fwd_kernel[(triton.cdiv(n_rows, _ROWS_BR),)](
        q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos, sin,
        out_q, out_k, out_v, n_rows, S, n_skip_rope,
        *_strides3(q_lat), *_strides3(k_lat), *_strides3(q_conv), *_strides3(k_conv),
        *_strides3(v_curr), *_strides3(v_prev),
        H=H, Hkv=Hkv, D=D, N_REP=n_rep, EPS=eps, VHALF=vhalf, BR=_ROWS_BR,
        **_ROWS_LAUNCH, **pdl_opts(),
    )
    return out_q, out_k, out_v


def _rows_bwd_impl(grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin,
                   H, Hkv, D, n_rep, n_skip_rope, eps, vhalf, flat=False):
    """The row-tiled prologue backward: the nine input grads (q_lat, k_lat, q_conv, k_conv,
    v_curr, v_prev, wq, wk, temp). ``flat`` (model.kv_flat_kernels): the re-tiled launch and
    one fixed-order kernel for the three partial sums."""
    B, S, _ = q_lat.shape
    n_rows = B * S
    br, launch = (_ROWS_BR_FLAT, _ROWS_LAUNCH_FLAT) if flat else (_ROWS_BR, _ROWS_LAUNCH)
    n_prog = triton.cdiv(n_rows, br)
    dev = q_lat.device
    grad_q = grad_q.contiguous()
    grad_k = grad_k.contiguous()
    grad_v = grad_v.contiguous()
    # Every grad lands in its input's dtype (q_lat's for v, as _FusedCCAPrologue does).
    d_ql = torch.empty_like(q_lat)
    d_kl = torch.empty_like(k_lat)
    d_qc = torch.empty_like(q_conv)
    d_kc = torch.empty_like(k_conv)
    d_vc = torch.empty(B, S, vhalf, device=dev, dtype=q_lat.dtype)
    d_vp = torch.empty(B, S, vhalf, device=dev, dtype=q_lat.dtype)
    f32 = torch.float32
    ny = Hkv if (flat and n_prog < _ROWS_KV_SPLIT_BELOW) else 1
    dwq_part = torch.empty(n_prog * ny, D, device=dev, dtype=f32)
    dwk_part = torch.empty(n_prog * ny, D, device=dev, dtype=f32)
    dtemp_part = torch.empty(n_prog, Hkv, device=dev, dtype=f32)
    _cca_rows_bwd_kernel[(n_prog, ny)](
        grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv,
        wq, wk, temp, cos, sin, d_ql, d_kl, d_qc, d_kc, d_vc, d_vp,
        dwq_part, dwk_part, dtemp_part, n_rows, S, n_skip_rope,
        *_strides3(q_lat), *_strides3(k_lat), *_strides3(q_conv), *_strides3(k_conv),
        *_strides3(d_ql), *_strides3(d_kl), *_strides3(d_qc), *_strides3(d_kc),
        *_strides3(d_vc), *_strides3(d_vp),
        H=H, Hkv=Hkv, D=D, N_REP=n_rep, EPS=eps, VHALF=vhalf, BR=br, KV_N=Hkv // ny,
        **launch, **pdl_opts())
    if flat:
        d_wq, d_wk, d_temp = torch.empty_like(wq), torch.empty_like(wk), torch.empty_like(temp)
        _sum_partials([(dwq_part, d_wq, 1), (dwk_part, d_wk, 1), (dtemp_part, d_temp, 1)])
        return d_ql, d_kl, d_qc, d_kc, d_vc, d_vp, d_wq, d_wk, d_temp
    return (d_ql, d_kl, d_qc, d_kc, d_vc, d_vp,
            dwq_part.sum(0).to(wq.dtype), dwk_part.sum(0).to(wk.dtype),
            dtemp_part.sum(0).to(temp.dtype))


class _FusedCCAPrologueRows(torch.autograd.Function):
    """``_FusedCCAPrologue`` on the row-tiled kernels (model.cca_prologue_tiled). Same
    inputs, outputs and gradient contract; inputs are read through their strides and each
    input grad is allocated with its input's layout (``empty_like``: a transposed conv
    output gets a transposed grad, so the conv backward's ``.contiguous()`` is free)."""

    @staticmethod
    def forward(ctx, q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
                wq, wk, temp, cos, sin, H, Hkv, D, n_rep, n_skip_rope, eps):
        out = _rows_fwd_impl(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp, cos,
                             sin, H, Hkv, D, n_rep, n_skip_rope, eps)
        ctx.save_for_backward(q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin)
        ctx.dims = (H, Hkv, D, n_rep, n_skip_rope, eps, v_curr.shape[-1])
        ctx.vdt = (v_curr.dtype, tuple(v_curr.shape), v_prev.dtype, tuple(v_prev.shape))
        return out

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin = ctx.saved_tensors
        H, Hkv, D, n_rep, n_skip_rope, eps, vhalf = ctx.dims
        grads = _rows_bwd_impl(grad_q, grad_k, grad_v, q_lat, k_lat, q_conv, k_conv, wq, wk,
                               temp, cos, sin, H, Hkv, D, n_rep, n_skip_rope, eps, vhalf)
        return (*grads, None, None, None, None, None, None, None, None)


# ===========================================================================
# Public API
# ===========================================================================

def fused_cca_prologue(
    q_lat: Tensor, k_lat: Tensor, q_conv: Tensor, k_conv: Tensor,
    v_curr: Tensor, v_prev: Tensor,
    q_norm_weight: Tensor, k_norm_weight: Tensor, temp: Tensor,
    cos: Tensor, sin: Tensor,
    n_heads: int, n_kv_heads: int, d_head: int,
    n_skip_rope: int = 0, eps: float = 1e-6, tiled: bool = False, custom_op: bool = False,
    flat: bool = False,
):
    """Fused CCA attention prologue (forward + backward in Triton).

    Args:
        q_lat:  [B, S, n_heads*d_head]      pre-conv Q latent (for qk-mean)
        k_lat:  [B, S, n_kv_heads*d_head]   pre-conv K latent (for qk-mean)
        q_conv: [B, S, n_heads*d_head]      post-conv Q
        k_conv: [B, S, n_kv_heads*d_head]   post-conv K
        v_curr: [B, S, n_kv_heads*(d_head//2)]
        v_prev: [B, S, n_kv_heads*(d_head//2)]
        q_norm_weight / k_norm_weight: [d_head]
        temp:   [n_kv_heads]
        cos / sin: rope caches, broadcastable to [S, d_head] (any leading singleton dims OK)
        n_skip_rope: number of trailing positions that skip RoPE.
        tiled:  model.cca_prologue_tiled — run the row-tiled kernels
                (``_FusedCCAPrologueRows``: one launch each way, strided inputs). Not
                bit-identical to the per-(row, head) kernels (reduction orders only).
        custom_op: model.block_custom_ops — the tiled prologue as a ``torch.library`` op
                (block_ops.py), not behind the dynamo fence. Needs ``tiled``.
        flat:   model.kv_flat_kernels — the backward on the re-tiled launch (kv_flat.py).
                Needs ``custom_op``; raises otherwise.

    Returns:
        q, k, v each [B, n_heads, S, d_head], dtype = q_lat.dtype.
    """
    H, Hkv, D = n_heads, n_kv_heads, d_head
    n_rep = H // Hkv
    cos2 = cos.reshape(-1, D)[: q_lat.shape[1]].contiguous()
    sin2 = sin.reshape(-1, D)[: q_lat.shape[1]].contiguous()

    from morph.kernels.triton._eager_flag import force_eager
    if force_eager() or not TRITON_AVAILABLE or not q_lat.is_cuda:
        return cca_prologue_reference(
            q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
            q_norm_weight, k_norm_weight, temp, cos, sin,
            H, Hkv, D, n_skip_rope, eps,
        )  # traceable

    if flat and not (tiled and custom_op):
        raise ValueError("model.kv_flat_kernels needs model.cca_prologue_tiled and "
                         "model.block_custom_ops (the re-tiled backward runs inside the op)")
    if tiled:
        # model.cca_prologue_tiled: D must be a power of two (the tile's column range).
        assert D & (D - 1) == 0, f"cca_prologue_tiled needs a power-of-two d_head, got {D}"
        if custom_op:
            from morph.kernels.triton.block_ops import cca_prologue_rows_op
            return cca_prologue_rows_op(
                q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
                q_norm_weight, k_norm_weight, temp, cos2, sin2,
                H, Hkv, D, n_rep, n_skip_rope, eps, flat,
            )
        return _cca_prologue_rows_dispatch(
            q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
            q_norm_weight, k_norm_weight, temp, cos2, sin2,
            H, Hkv, D, n_rep, n_skip_rope, eps,
        )
    if custom_op:
        raise ValueError("model.block_custom_ops needs model.cca_prologue_tiled=true (only the "
                         "row-tiled prologue is an op)")
    return _cca_prologue_dispatch(
        q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
        q_norm_weight, k_norm_weight, temp, cos2, sin2,
        H, Hkv, D, n_rep, n_skip_rope, eps,
    )


@kernel_fence  # Dynamo fence: kernel is opaque (autograd.Function)
def _cca_prologue_dispatch(*args):
    return _FusedCCAPrologue.apply(*args)


@kernel_fence  # Dynamo fence: kernel is opaque (autograd.Function)
def _cca_prologue_rows_dispatch(*args):
    return _FusedCCAPrologueRows.apply(*args)


# ===========================================================================
# Pure-PyTorch reference (the spec; adapted from p1_prologue_harness)
# ===========================================================================

def _rmsnorm(x, weight, eps):
    norm = x.float().pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
    return (x.float() * norm).to(x.dtype) * weight


def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def cca_prologue_reference(
    q_lat, k_lat, q_conv, k_conv, v_curr, v_prev,
    q_norm_weight, k_norm_weight, temp, cos, sin,
    n_heads, n_kv_heads, d_head, n_skip_rope=0, eps=1e-6,
):
    """Pure-PyTorch reference, byte-faithful to ``_CCABase._cca_project``."""
    B, S, _ = q_lat.shape
    H, Hkv, D = n_heads, n_kv_heads, d_head
    n_rep = H // Hkv

    q_pre = q_lat.reshape(B, S, H, D)
    k_pre = k_lat.reshape(B, S, Hkv, D)
    k_pre_exp = k_pre.repeat_interleave(n_rep, dim=2)
    qk_mean_q = (q_pre + k_pre_exp) * 0.5
    qk_mean_k = qk_mean_q.reshape(B, S, Hkv, n_rep, D).mean(dim=3)

    q = (q_conv.reshape(B, S, H, D) + qk_mean_q).transpose(1, 2)
    k = (k_conv.reshape(B, S, Hkv, D) + qk_mean_k).transpose(1, 2)

    q = _rmsnorm(q, q_norm_weight, eps)
    k = _rmsnorm(k, k_norm_weight, eps)
    k = k * torch.exp(temp).view(1, Hkv, 1, 1)

    cos_full = cos.reshape(-1, D).to(q.dtype)
    sin_full = sin.reshape(-1, D).to(q.dtype)

    def rope(t):
        # mirror CoPEEmbedding.forward: cos/sin sliced to the *input* seq length
        Sl = t.shape[2]
        cb = cos_full[:Sl].view(1, 1, Sl, D)
        sb = sin_full[:Sl].view(1, 1, Sl, D)
        return t * cb + _rotate_half(t) * sb

    if n_skip_rope > 0:
        q = torch.cat([rope(q[:, :, :-n_skip_rope]), q[:, :, -n_skip_rope:]], dim=2)
        k = torch.cat([rope(k[:, :, :-n_skip_rope]), k[:, :, -n_skip_rope:]], dim=2)
    else:
        q = rope(q)
        k = rope(k)

    k = k.repeat_interleave(n_rep, dim=1)

    v = torch.cat([v_curr, v_prev], dim=-1)
    v = v.reshape(B, S, Hkv, D).transpose(1, 2)
    v = v.repeat_interleave(n_rep, dim=1)
    return q, k, v


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    print("=" * 100)
    print("fused_cca_prologue — forward + backward correctness test")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    d_model, H, Hkv, comp = 1024, 8, 4, 2   # base.yaml widths
    D = d_model // (comp * H)         # 64
    pdt = torch.float32               # params stay fp32 under training autocast
    n_rep = H // Hkv
    VHALF = Hkv * (D // 2)
    eps = 1e-6
    base = 10000.0
    max_seq = 8192
    # build a rope cos/sin cache like CoPEEmbedding (no taper needed for parity since
    # the kernel just consumes whatever cache is passed; use plain rope here)
    inv_freq = 1.0 / (base ** (torch.arange(0, D, 2).float() / D))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    COS = emb.cos().to(dev)
    SIN = emb.sin().to(dev)

    INPUT_NAMES = ["q_lat", "k_lat", "q_conv", "k_conv", "v_curr", "v_prev",
                   "q_norm_w", "k_norm_w", "temp"]

    def make_inputs(B, S):
        ins = {
            "q_lat":  torch.randn(B, S, H * D, device=dev, dtype=dt),
            "k_lat":  torch.randn(B, S, Hkv * D, device=dev, dtype=dt),
            "q_conv": torch.randn(B, S, H * D, device=dev, dtype=dt),
            "k_conv": torch.randn(B, S, Hkv * D, device=dev, dtype=dt),
            "v_curr": torch.randn(B, S, VHALF, device=dev, dtype=dt),
            "v_prev": torch.randn(B, S, VHALF, device=dev, dtype=dt),
            "q_norm_w": torch.randn(D, device=dev, dtype=pdt).mul_(0.1).add_(1.0),
            "k_norm_w": torch.randn(D, device=dev, dtype=pdt).mul_(0.1).add_(1.0),
            "temp":   torch.randn(Hkv, device=dev, dtype=pdt).mul_(0.3),
        }
        return ins

    def clone_req(ins):
        return {k: v.detach().clone().requires_grad_(True) for k, v in ins.items()}

    def _rel(x, ref):
        x, ref = x.double(), ref.double()
        return ((x - ref).norm() / ref.norm().clamp_min(1e-300)).item()

    def run_case(B, S, n_skip, floor_slack=1.05, grad_slack=1.02):
        """Gates against float64 truth (cca_prologue_reference on fp64 copies of the same
        inputs), NOT against the bf16 reference: that reference rounds every bf16 op and is
        the less exact path (module docstring, "Precision").
          * forward: q / k / v relative L2 error <= floor_slack x the one-rounding floor
            (the fp64 truth rounded once to bf16);
          * backward: every input grad's relative L2 error <= grad_slack x the bf16
            reference's own error.
        The same tests/test_cca_prologue_precision.py holds for the training layouts."""
        ins = make_inputs(B, S)
        a = clone_req(ins)                                               # kernel
        r = clone_req(ins)                                               # bf16 reference
        t = {k: v.detach().double().requires_grad_(True) for k, v in ins.items()}   # truth
        outs = {
            "kernel": fused_cca_prologue(*(a[n] for n in INPUT_NAMES), COS, SIN,
                                         H, Hkv, D, n_skip, eps),
            "bf16ref": cca_prologue_reference(*(r[n] for n in INPUT_NAMES), COS, SIN,
                                              H, Hkv, D, n_skip, eps),
            "fp64": cca_prologue_reference(*(t[n] for n in INPUT_NAMES), COS.double(),
                                           SIN.double(), H, Hkv, D, n_skip, eps),
        }
        # One upstream grad, bf16-representable, so all three paths see the same values.
        gos = [torch.randn(o.shape, device=dev, dtype=dt) for o in outs["fp64"]]
        for name in outs:
            torch.autograd.backward(list(outs[name]),
                                    [g.to(o.dtype) for g, o in zip(gos, outs[name])])

        fwd_ok, cells = True, []
        for i, on in enumerate("qkv"):
            truth = outs["fp64"][i]
            floor = _rel(truth.to(dt), truth)
            ek, er = _rel(outs["kernel"][i], truth), _rel(outs["bf16ref"][i], truth)
            fwd_ok &= ek <= floor_slack * floor
            cells.append(f"{on} {ek:.2e}/{er:.2e}/{floor:.2e}")
        bwd_ok, gcells = True, []
        for n in INPUT_NAMES:
            ek, er = _rel(a[n].grad, t[n].grad), _rel(r[n].grad, t[n].grad)
            bwd_ok &= ek <= grad_slack * er + 1e-7
            gcells.append(f"{n}={ek:.1e}/{er:.1e}")
        status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
        print(f"  [{status}] B={B} S={S:<5} n_skip={n_skip}  fwd rel err kernel/bf16ref/floor: "
              + "  ".join(cells))
        print(f"           grad rel err kernel/bf16ref: {'  '.join(gcells)}")
        return fwd_ok and bwd_ok

    print("\n[Correctness — forward + backward across dim sweep]")
    all_ok = True
    for S in (512, 1024, 2048, 4096):
        all_ok &= run_case(2, S, 0)
    # n_skip_rope parity (kept for parity even though prod uses 0)
    all_ok &= run_case(2, 512, 8)
    all_ok &= run_case(2, 1024, 16)

    print("\n[Speed — fused Triton vs eager reference, B=2, S=2048]")
    B, S = 2, 2048
    ins = make_inputs(B, S)
    a = clone_req(ins); r = clone_req(ins)

    def fused_fb():
        qf, kf, vf = fused_cca_prologue(
            a["q_lat"], a["k_lat"], a["q_conv"], a["k_conv"], a["v_curr"], a["v_prev"],
            a["q_norm_w"], a["k_norm_w"], a["temp"], COS, SIN, H, Hkv, D, 0, eps)
        (qf.sum() + kf.sum() + vf.sum()).backward()
        a["q_lat"].grad = None; a["k_lat"].grad = None; a["q_conv"].grad = None
        a["k_conv"].grad = None; a["v_curr"].grad = None; a["v_prev"].grad = None
        a["q_norm_w"].grad = None; a["k_norm_w"].grad = None; a["temp"].grad = None

    def eager_fb():
        qr, kr, vr = cca_prologue_reference(
            r["q_lat"], r["k_lat"], r["q_conv"], r["k_conv"], r["v_curr"], r["v_prev"],
            r["q_norm_w"], r["k_norm_w"], r["temp"], COS, SIN, H, Hkv, D, 0, eps)
        (qr.sum() + kr.sum() + vr.sum()).backward()
        r["q_lat"].grad = None; r["k_lat"].grad = None; r["q_conv"].grad = None
        r["k_conv"].grad = None; r["v_curr"].grad = None; r["v_prev"].grad = None
        r["q_norm_w"].grad = None; r["k_norm_w"].grad = None; r["temp"].grad = None

    def bench(fn, n=50, warmup=10):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    t_fused = bench(fused_fb)
    t_eager = bench(eager_fb)
    print(f"  Eager reference fwd+bwd:  {t_eager:.3f} ms")
    print(f"  Fused Triton  fwd+bwd:    {t_fused:.3f} ms")
    print(f"  Speedup:                  {t_eager / t_fused:.2f}x")

    # forward-only speed (production hot path is fwd; bwd recompute heavier)
    def fused_fwd():
        with torch.no_grad():
            fused_cca_prologue(
                a["q_lat"], a["k_lat"], a["q_conv"], a["k_conv"], a["v_curr"], a["v_prev"],
                a["q_norm_w"], a["k_norm_w"], a["temp"], COS, SIN, H, Hkv, D, 0, eps)

    def eager_fwd():
        with torch.no_grad():
            cca_prologue_reference(
                r["q_lat"], r["k_lat"], r["q_conv"], r["k_conv"], r["v_curr"], r["v_prev"],
                r["q_norm_w"], r["k_norm_w"], r["temp"], COS, SIN, H, Hkv, D, 0, eps)

    t_ff = bench(fused_fwd)
    t_ef = bench(eager_fwd)
    print(f"\n  [fwd-only] eager: {t_ef:.3f} ms  fused: {t_ff:.3f} ms  speedup: {t_ef / t_ff:.2f}x")

    # =====================================================================
    # End-to-end: the REAL _CCABase._cca_project under bf16 autocast (what ships). Its
    # prologue inputs are captured at the fused_cca_prologue call and scored in float64.
    # =====================================================================
    print("\n[End-to-end: real _CCABase._cca_project, bf16 autocast]")
    import morph.model.attention as _attn
    from morph.model.attention import _CCABase

    e2e_ok = True
    captured = {}
    _real_prologue = _attn.fused_cca_prologue

    def _capture(*args, **kw):
        captured["args"], captured["kw"] = args, kw
        return _real_prologue(*args, **kw)

    _attn.fused_cca_prologue = _capture
    try:
        for S in (512, 1024, 2048):
            cca = _CCABase(d_model=d_model, n_heads=H, n_kv_heads=Hkv, compression=comp,
                           max_seq_len=8192, context_len=4096, window_size=128,
                           init_alpha=0.1, conv_kernel=4).to(dev)
            with torch.no_grad():
                cca.temp.normal_(0, 0.3)
                cca.q_norm.weight.normal_(1.0, 0.1)
                cca.k_norm.weight.normal_(1.0, 0.1)
            x = torch.randn(2, S, d_model, device=dev, dtype=dt)
            with torch.no_grad(), torch.autocast("cuda", dtype=dt):
                got = cca._cca_project(x, 0)
            args, kw = captured["args"], captured["kw"]
            a64 = [v.double() if torch.is_tensor(v) else v for v in args]
            with torch.no_grad():
                truth = cca_prologue_reference(*a64[:14], kw.get("n_skip_rope", 0),
                                               kw.get("eps", 1e-6))
            cells, ok = [], True
            for on, o, tr in zip("qkv", got, truth):
                floor = _rel(tr.to(o.dtype), tr)
                e = _rel(o, tr)
                ok &= e <= 1.05 * floor
                cells.append(f"{on} {e:.2e} (floor {floor:.2e}, {str(o.dtype)[6:]})")
            e2e_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] S={S:<5} " + "  ".join(cells))
    finally:
        _attn.fused_cca_prologue = _real_prologue

    all_ok &= e2e_ok

    print("\n" + "=" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 100)
    assert all_ok, "fused_cca_prologue self-test FAILED — see rows marked FAIL above"
