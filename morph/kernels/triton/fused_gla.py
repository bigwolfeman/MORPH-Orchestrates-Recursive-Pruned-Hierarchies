"""Fused chunked Gated Linear Attention (GLA) — forward AND backward (sm_120 / RTX 5090).

Replaces the eager chunked training path in ``morph/model/gla.py``
(``GatedLinearAttention._chunked``) with a single fused Triton kernel per direction.
Numerically faithful to the sequential oracle ``_recurrent`` (the math both eager
paths implement), gated in ``ignore/verify_fused_gla.py``.

MATH (per head; dk = dv = dh, per-key-channel forget gate)
----------------------------------------------------------
Recurrence (the oracle):
    S_t = diag(alpha_t) · S_{t-1} + k_t^T v_t        alpha_t = exp(log_alpha_t) ∈ R^{dh}
    o_t = q_t · S_t

Chunked form actually computed (chunk length C, local indices 0..L-1):
    b      = clamp(cumsum_t(log_alpha), min=-30)      # cumulative log-gate, fp32
    B_t    = exp(b_t)                                  # prod alpha (per channel)
    qb_t   = q_t ⊙ B_t                                 # decayed query
    kb_j   = k_j ⊙ exp(-b_j)                           # k / B_j   (the -30 floor keeps exp(-b) finite)
    o_inter= qb @ S_in                                 # carried-state contribution
    P[t,j] = qb_t · kb_j   (j<=t)                       # intra-chunk scores (causal)
    o_intra= P @ v
    o      = o_inter + o_intra
    B_L    = exp(clamp(Σ_t log_alpha, -30))            # full-chunk gate product
    ke_j   = k_j ⊙ (B_L / B_j) = kb_j ⊙ B_L
    S_out  = diag(B_L) S_in + ke^T @ v

The chunked output is chunk-length-independent (proven by the eager parity gate),
so the kernel's C need not match the eager module's chunk.

NUMERICAL STABILITY (critical): exp(-b_j) overflows fp32 for aggressive decay; the
-30 floor on the cumulative log-gate (exactly the eager ``b.clamp(min=-30)``) bounds
exp(-b) ≤ e^30 ≈ 1e13, safe in fp32. All accumulation (state, scores) is fp32.

BACKWARD (analytic, chunk-parallel)
-----------------------------------
Derived directly from the chunked algebra above (no per-token scan). The forward
saves the chunk-start state for each chunk; the backward kernel walks chunks in
REVERSE, recomputing the chunk intermediates, and carries the state-adjoint matrix
dS across chunk boundaries (init = grad on final_state; final value = grad on S0).
Within a chunk, every grad (dq, dk, dv, dlog_alpha, dS_in) is a handful of [C,dh] /
[dh,dh] matmuls. The log-gate grad uses a reverse cumulative-sum of the per-(t,k)
db contributions; the -30 clamp contributes a 0/1 mask. Validated to autograd-through-
``_recurrent`` (cosine ≥ 0.999, small rel-err) in ignore/verify_fused_gla.py.

Hardware (RTX 5090 / sm_120):
  * num_stages=1, num_warps=8 (consumer Blackwell, no TMA / no Hopper tl.dot features).
  * bf16 (or fp32) in/out; fp32 math throughout. tl.dot(input_precision="ieee") → true
    fp32 (no TF32 truncation) so the kernel holds the eager fp32 oracle to ~1e-3.
  * Grid = (B*H,). State recurrence is sequential per (b,h); the chunk loop lives
    IN-kernel, collapsing the eager Python per-chunk launch storm into one launch.

Author: TileProver (Claude Code, Opus 4.8)  Date: 2026-06-06
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


_LAUNCH = dict(num_stages=1, num_warps=8)
_CLAMP = -30.0


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


if TRITON_AVAILABLE:

    @triton.jit
    def _gla_fwd_kernel(
        q_ptr, k_ptr, v_ptr, la_ptr,        # [B, S, H, DH]
        s0_ptr,                              # [B, H, DH, DH] or unused
        o_ptr,                               # [B, S, H, DH]   out
        sfinal_ptr,                          # [B, H, DH, DH]  out (fp32)
        states_ptr,                          # [B, H, NCH, DH, DH] out (fp32) chunk-start states
        H, S, n_chunks,
        DH: tl.constexpr, C: tl.constexpr, BD: tl.constexpr,
        HAS_S0: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H

        rows = tl.arange(0, C)
        kk = tl.arange(0, BD)
        cmask = kk < DH                      # channel-valid mask [BD]

        # token-major base for q/k/v/la at (b, *, h, *)
        tok_stride = H * DH                  # stride between consecutive tokens
        base = b * S * tok_stride + h * DH

        # state [BD, BD] fp32, rows = key channel k, cols = value channel v
        kkc = kk[:, None]
        vvc = kk[None, :]
        full_mask = (kkc < DH) & (vvc < DH)
        if HAS_S0:
            s0_base = (b * H + h) * DH * DH
            state = tl.load(s0_ptr + s0_base + kkc * DH + vvc,
                            mask=full_mask, other=0.0).to(tl.float32)
        else:
            state = tl.zeros((BD, BD), dtype=tl.float32)

        st_bh_base = (b * H + h) * n_chunks * DH * DH
        causal = rows[:, None] >= rows[None, :]   # [C,C] j<=t

        c = 0
        while c < n_chunks:
            s0 = c * C
            tok = s0 + rows
            rmask = tok < S
            ld_mask = rmask[:, None] & cmask[None, :]
            offs = base + tok[:, None] * tok_stride + kk[None, :]

            q_t = tl.load(q_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            la_t = tl.load(la_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)

            # save chunk-start state
            st_off = states_ptr + st_bh_base + c * DH * DH + kkc * DH + vvc
            tl.store(st_off, state, mask=full_mask)

            b_cum = tl.cumsum(la_t, axis=0)
            b_cl = tl.maximum(b_cum, -30.0)
            B_ = tl.exp(b_cl)
            negB = tl.exp(-b_cl)
            qb = q_t * B_
            kb = k_t * negB

            o_inter = tl.dot(qb, state, input_precision="ieee")
            P = tl.dot(qb, tl.trans(kb), input_precision="ieee")
            P = tl.where(causal, P, 0.0)
            o_intra = tl.dot(P, v_t, input_precision="ieee")
            o = o_inter + o_intra
            tl.store(o_ptr + offs, o.to(o_ptr.dtype.element_ty), mask=ld_mask)

            # state update to chunk end
            sum_la = tl.sum(la_t, axis=0)                 # [BD]
            BL = tl.exp(tl.maximum(sum_la, -30.0))       # full-chunk product
            decay_end = BL[None, :] * negB                # B_L / B_j  [C,BD]
            ke = k_t * decay_end
            KV = tl.dot(tl.trans(ke), v_t, input_precision="ieee")   # [BD,BD]
            state = BL[:, None] * state + KV
            c += 1

        sf_base = (b * H + h) * DH * DH
        tl.store(sfinal_ptr + sf_base + kkc * DH + vvc, state, mask=full_mask)

    @triton.jit
    def _gla_bwd_kernel(
        q_ptr, k_ptr, v_ptr, la_ptr,        # [B, S, H, DH]
        states_ptr,                          # [B, H, NCH, DH, DH] fp32
        do_ptr,                              # [B, S, H, DH]  grad on o
        dsf_ptr,                             # [B, H, DH, DH]  grad on final_state (fp32)
        dq_ptr, dk_ptr, dv_ptr, dla_ptr,    # [B, S, H, DH]  out
        ds0_ptr,                             # [B, H, DH, DH]  out (fp32) grad on initial_state
        H, S, n_chunks,
        DH: tl.constexpr, C: tl.constexpr, BD: tl.constexpr,
    ):
        bh = tl.program_id(0)
        b = bh // H
        h = bh % H

        rows = tl.arange(0, C)
        kk = tl.arange(0, BD)
        cmask = kk < DH
        kkc = kk[:, None]
        vvc = kk[None, :]
        full_mask = (kkc < DH) & (vvc < DH)

        tok_stride = H * DH
        base = b * S * tok_stride + h * DH
        st_bh_base = (b * H + h) * n_chunks * DH * DH
        causal = rows[:, None] >= rows[None, :]
        last_row = (rows == (C - 1))[:, None]            # [C,1]

        # dS carried backward; init = grad on final_state
        dsf_base = (b * H + h) * DH * DH
        dS = tl.load(dsf_ptr + dsf_base + kkc * DH + vvc,
                     mask=full_mask, other=0.0).to(tl.float32)

        cc = 0
        while cc < n_chunks:
            c = n_chunks - 1 - cc
            s0 = c * C
            tok = s0 + rows
            rmask = tok < S
            ld_mask = rmask[:, None] & cmask[None, :]
            offs = base + tok[:, None] * tok_stride + kk[None, :]

            q_t = tl.load(q_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            k_t = tl.load(k_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            v_t = tl.load(v_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            la_t = tl.load(la_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)
            dO = tl.load(do_ptr + offs, mask=ld_mask, other=0.0).to(tl.float32)

            st_off = states_ptr + st_bh_base + c * DH * DH + kkc * DH + vvc
            S_in = tl.load(st_off, mask=full_mask, other=0.0).to(tl.float32)

            # --- recompute forward intermediates ---
            b_cum = tl.cumsum(la_t, axis=0)
            b_cl = tl.maximum(b_cum, -30.0)
            B_ = tl.exp(b_cl)
            negB = tl.exp(-b_cl)
            qb = q_t * B_
            kb = k_t * negB
            sum_la = tl.sum(la_t, axis=0)
            BL = tl.exp(tl.maximum(sum_la, -30.0))       # [BD]
            decay_end = BL[None, :] * negB
            ke = k_t * decay_end
            P = tl.dot(qb, tl.trans(kb), input_precision="ieee")
            P = tl.where(causal, P, 0.0)

            # --- backward through o = o_inter + o_intra ---
            # o_inter = qb @ S_in
            dqb = tl.dot(dO, tl.trans(S_in), input_precision="ieee")          # [C,BD]
            dS_in = tl.dot(tl.trans(qb), dO, input_precision="ieee")          # [BD,BD]
            # o_intra = P @ v
            dP = tl.dot(dO, tl.trans(v_t), input_precision="ieee")            # [C,C]
            dP = tl.where(causal, dP, 0.0)
            dv = tl.dot(tl.trans(P), dO, input_precision="ieee")              # [C,BD]
            # P = qb @ kb^T
            dqb += tl.dot(dP, kb, input_precision="ieee")
            dkb = tl.dot(tl.trans(dP), qb, input_precision="ieee")

            # --- backward through state update S_out = diag(BL) S_in + ke^T @ v ---
            dS_in += BL[:, None] * dS                                         # diag term
            dBL_diag = tl.sum(dS * S_in, axis=1)                              # [BD]
            dke = tl.dot(v_t, tl.trans(dS), input_precision="ieee")           # [C,BD]
            dv += tl.dot(ke, dS, input_precision="ieee")                      # [C,BD]

            # --- convert decayed grads back to q, k, log_alpha ---
            dq = dqb * B_
            dk = dkb * negB + dke * decay_end

            db = dqb * qb - dkb * kb - dke * ke                               # per-(t,k)
            colterm = tl.sum(dke * ke, axis=0) + dBL_diag * BL                # [BD], lands at row C-1
            db += tl.where(last_row, colterm[None, :], 0.0)

            # clamp VJP: grad flows only where cumsum was above the floor
            mask_clamp = b_cum >= -30.0
            dbm = db * mask_clamp
            # dlog_alpha_j = Σ_{t>=j} dbm_t  (reverse cumulative sum)
            cs = tl.cumsum(dbm, axis=0)
            total = tl.sum(dbm, axis=0)
            dla = total[None, :] - cs + dbm

            tl.store(dq_ptr + offs, dq.to(dq_ptr.dtype.element_ty), mask=ld_mask)
            tl.store(dk_ptr + offs, dk.to(dk_ptr.dtype.element_ty), mask=ld_mask)
            tl.store(dv_ptr + offs, dv.to(dv_ptr.dtype.element_ty), mask=ld_mask)
            tl.store(dla_ptr + offs, dla.to(dla_ptr.dtype.element_ty), mask=ld_mask)

            dS = dS_in
            cc += 1

        # dS now = grad wrt state-before-chunk-0 = grad wrt initial_state
        tl.store(ds0_ptr + dsf_base + kkc * DH + vvc, dS, mask=full_mask)


# ===========================================================================
# autograd.Function
# ===========================================================================

_DEFAULT_CHUNK = 32


class FusedGLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, log_alpha, initial_state, chunk):
        assert q.is_cuda, "FusedGLA requires CUDA tensors"
        assert q.dim() == 4, "expected [B, S, H, DH]"
        B, S, H, DH = q.shape
        C = chunk
        BD = _next_pow2(DH)
        n_chunks = (S + C - 1) // C

        q, k, v, log_alpha = (t.contiguous() for t in (q, k, v, log_alpha))
        has_s0 = initial_state is not None
        if has_s0:
            s0 = initial_state.contiguous().to(torch.float32)
        else:
            s0 = q.new_empty(0)

        o = torch.empty_like(q)
        sfinal = q.new_empty((B, H, DH, DH), dtype=torch.float32)
        states = q.new_empty((B, H, n_chunks, DH, DH), dtype=torch.float32)

        grid = (B * H,)
        _gla_fwd_kernel[grid](
            q, k, v, log_alpha, s0, o, sfinal, states,
            H, S, n_chunks,
            DH=DH, C=C, BD=BD, HAS_S0=has_s0, **_LAUNCH,
        )

        ctx.save_for_backward(q, k, v, log_alpha, states)
        ctx.has_s0 = has_s0
        ctx.dims = (B, S, H, DH, C, BD, n_chunks)
        return o, sfinal

    @staticmethod
    def backward(ctx, do, dsfinal):
        q, k, v, log_alpha, states = ctx.saved_tensors
        B, S, H, DH, C, BD, n_chunks = ctx.dims

        do = do.contiguous()
        dsf = dsfinal.contiguous().to(torch.float32)

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dla = torch.empty_like(log_alpha)
        ds0 = q.new_empty((B, H, DH, DH), dtype=torch.float32)

        grid = (B * H,)
        _gla_bwd_kernel[grid](
            q, k, v, log_alpha, states, do, dsf,
            dq, dk, dv, dla, ds0,
            H, S, n_chunks,
            DH=DH, C=C, BD=BD, **_LAUNCH,
        )

        d_initial = ds0 if ctx.has_s0 else None
        # grads for: q, k, v, log_alpha, initial_state, chunk
        return dq, dk, dv, dla, d_initial, None


def fused_gla(q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor,
              initial_state: Tensor | None = None, chunk: int = _DEFAULT_CHUNK):
    """Fused chunked GLA. Inputs [B,S,H,dh] (bf16/fp32); log_alpha ≤ 0.

    Returns (o [B,S,H,dh] in input dtype, final_state [B,H,dh,dh] fp32).
    Matches ``GatedLinearAttention._recurrent`` up to fp accumulation.
    """
    return FusedGLA.apply(q, k, v, log_alpha, initial_state, chunk)
