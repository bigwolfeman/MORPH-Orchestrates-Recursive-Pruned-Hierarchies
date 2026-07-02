"""Fused sliding-window causal attention Triton kernel — forward AND backward.

Computes causal sliding-window attention without materialising the full S×S mask.
Designed for sm_120 (RTX 5090 / Blackwell): num_stages=1, num_warps=8.

Forward:  Online softmax with tile-skipping over window bounds.
Backward: Triton dQ/dK/dV kernels — same tile-skipping, no SDPA fallback.

Memory savings vs materialised-mask baseline:
  S=2048 : 16 MB  →  O(S·w/BLOCK_K) tile visits
  S=8192 : 256 MB →  O(S·w/BLOCK_K) tile visits

Grid: (B*H, num_q_tiles) for forward + dQ
      (B*H, num_k_tiles) for dK/dV

Author: Claude Code
Date:   2026-05-14
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if TRITON_AVAILABLE:

    # ── Window mask helper (shared by forward and backward) ──────────────
    # Inlined in each kernel since Triton JIT doesn't support function calls
    # across jit boundaries. The mask logic is:
    #   visible = (causal & window) | mem_col | mem_row
    #   if exclude_self: visible &= (k_pos != q_pos)

    # =====================================================================
    # Forward kernel
    # =====================================================================

    @triton.jit
    def _fused_window_attn_kernel(
        Q_ptr, K_ptr, V_ptr, Out_ptr, LSE_ptr,
        stride_qb, stride_qh, stride_qs, stride_qd,
        KT_lo_ptr, KT_hi_ptr,
        S, D, scale, window_size, n_skip_rope,
        exclude_self: tl.constexpr,
        BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """Forward: one CTA = one (batch·head, q-tile) pair.
        Outputs both attention result and LSE for backward."""

        bh_id = tl.program_id(0)
        qt_id = tl.program_id(1)

        q_start = qt_id * BLOCK_Q
        q_offs = q_start + tl.arange(0, BLOCK_Q)
        q_mask = q_offs < S
        d_offs = tl.arange(0, BLOCK_D)

        base = bh_id * stride_qh

        Q_ptrs = Q_ptr + base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        q = tl.load(Q_ptrs, mask=q_mask[:, None] & (d_offs[None, :] < D), other=0.0)
        q = q.to(tl.float32) * scale

        m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), dtype=tl.float32)

        kt_lo = tl.load(KT_lo_ptr + qt_id)
        kt_hi = tl.load(KT_hi_ptr + qt_id)

        for kt in range(kt_lo, kt_hi + 1):
            k_start = kt * BLOCK_K
            k_offs = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offs < S

            K_ptrs = K_ptr + base + k_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
            k = tl.load(K_ptrs, mask=k_mask[:, None] & (d_offs[None, :] < D), other=0.0)

            scores = tl.dot(q, tl.trans(k.to(tl.float32)), allow_tf32=True)

            q_pos = q_offs[:, None]
            k_pos = k_offs[None, :]
            is_causal = k_pos <= q_pos
            is_window = k_pos >= (q_pos - window_size + 1)
            is_mem_col = k_pos >= S - n_skip_rope
            is_mem_row = q_pos >= S - n_skip_rope
            vmask = (is_causal & is_window) | is_mem_col | is_mem_row
            if exclude_self:
                vmask = vmask & (k_pos != q_pos)
            vmask = vmask & q_mask[:, None] & k_mask[None, :]
            scores = tl.where(vmask, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            alpha = tl.where(m_new == float("-inf"), 1.0, tl.exp(m_i - m_new))
            p = tl.where(m_new[:, None] == float("-inf"),
                         tl.zeros_like(scores), tl.exp(scores - m_new[:, None]))

            V_ptrs = V_ptr + base + k_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
            v = tl.load(V_ptrs, mask=k_mask[:, None] & (d_offs[None, :] < D), other=0.0)

            l_new = alpha * l_i + tl.sum(p, axis=1)
            acc = alpha[:, None] * acc + tl.dot(p, v.to(tl.float32), allow_tf32=True)
            m_i = m_new
            l_i = l_new

        safe_l = tl.where(l_i == 0.0, 1.0, l_i)
        out = (acc / safe_l[:, None]).to(Q_ptr.dtype.element_ty)

        Out_ptrs = Out_ptr + base + q_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
        tl.store(Out_ptrs, out, mask=q_mask[:, None] & (d_offs[None, :] < D))

        lse_i = m_i + tl.log(safe_l)
        LSE_ptrs = LSE_ptr + bh_id * S + q_offs
        tl.store(LSE_ptrs, lse_i, mask=q_mask)

    # =====================================================================
    # Backward: preprocess D_i = rowsum(O_i * dO_i)
    # =====================================================================

    @triton.jit
    def _fused_window_bwd_preprocess(
        O_ptr, dO_ptr, D_ptr,
        S, head_dim: tl.constexpr,
        stride_bh, stride_s, stride_d,
        BLOCK_M: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, head_dim)
        mask = offs_m[:, None] < S

        base = pid_bh * stride_bh
        o = tl.load(base + O_ptr + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                     mask=mask, other=0.0).to(tl.float32)
        do = tl.load(base + dO_ptr + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                      mask=mask, other=0.0).to(tl.float32)
        d = tl.sum(o * do, axis=1)
        tl.store(D_ptr + pid_bh * S + offs_m, d, mask=offs_m < S)

    # =====================================================================
    # Backward: dK, dV — grid over KV tiles, iterate over Q tiles
    # =====================================================================

    @triton.jit
    def _fused_window_bwd_dkdv(
        Q_ptr, K_ptr, V_ptr, dO_ptr, dK_ptr, dV_ptr, LSE_ptr, D_ptr,
        QT_lo_ptr, QT_hi_ptr,
        S, head_dim: tl.constexpr, sm_scale,
        window_size, n_skip_rope,
        exclude_self: tl.constexpr,
        stride_bh, stride_s, stride_d,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Compute dK, dV for one KV-tile. Iterates over Q-tiles that attend to it."""
        pid_n = tl.program_id(0)
        pid_bh = tl.program_id(1)

        base = pid_bh * stride_bh

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, head_dim)
        k_mask = offs_n[:, None] < S

        k = tl.load(K_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                     mask=k_mask, other=0.0)
        v = tl.load(V_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                     mask=k_mask, other=0.0)

        dk = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)

        qt_lo = tl.load(QT_lo_ptr + pid_n)
        qt_hi = tl.load(QT_hi_ptr + pid_n)

        for qt in range(qt_lo, qt_hi + 1):
            offs_m = qt * BLOCK_M + tl.arange(0, BLOCK_M)
            q_valid = offs_m[:, None] < S

            q = tl.load(Q_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                         mask=q_valid, other=0.0)
            do = tl.load(dO_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                          mask=q_valid, other=0.0)
            lse = tl.load(LSE_ptr + pid_bh * S + offs_m, mask=offs_m < S, other=0.0)
            Di = tl.load(D_ptr + pid_bh * S + offs_m, mask=offs_m < S, other=0.0)

            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), allow_tf32=True) * sm_scale

            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            is_causal = k_pos <= q_pos
            is_window = k_pos >= (q_pos - window_size + 1)
            is_mem_col = k_pos >= S - n_skip_rope
            is_mem_row = q_pos >= S - n_skip_rope
            vmask = (is_causal & is_window) | is_mem_col | is_mem_row
            if exclude_self:
                vmask = vmask & (k_pos != q_pos)
            valid = vmask & (offs_m[:, None] < S) & (offs_n[None, :] < S)
            s = tl.where(valid, s, float("-inf"))

            p = tl.exp(s - lse[:, None])
            p = tl.where(valid, p, 0.0)

            dv += tl.dot(tl.trans(p.to(do.dtype)), do.to(do.dtype), allow_tf32=True)

            dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), allow_tf32=True)
            ds = p * (dp - Di[:, None])
            ds = tl.where(valid, ds, 0.0)

            dk += tl.dot(tl.trans((ds * sm_scale).to(q.dtype)), q.to(q.dtype), allow_tf32=True)

        tl.store(dK_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                 dk.to(k.dtype), mask=k_mask)
        tl.store(dV_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                 dv.to(v.dtype), mask=k_mask)

    # =====================================================================
    # Backward: dQ — grid over Q tiles, iterate over KV tiles
    # =====================================================================

    @triton.jit
    def _fused_window_bwd_dq(
        Q_ptr, K_ptr, V_ptr, dO_ptr, dQ_ptr, LSE_ptr, D_ptr,
        KT_lo_ptr, KT_hi_ptr,
        S, head_dim: tl.constexpr, sm_scale,
        window_size, n_skip_rope,
        exclude_self: tl.constexpr,
        stride_bh, stride_s, stride_d,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Compute dQ for one Q-tile. Iterates over KV-tiles it attends to."""
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        base = pid_bh * stride_bh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, head_dim)
        q_valid = offs_m[:, None] < S

        q = tl.load(Q_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                     mask=q_valid, other=0.0)
        do = tl.load(dO_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                      mask=q_valid, other=0.0)
        lse = tl.load(LSE_ptr + pid_bh * S + offs_m, mask=offs_m < S, other=0.0)
        Di = tl.load(D_ptr + pid_bh * S + offs_m, mask=offs_m < S, other=0.0)

        dq = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

        kt_lo = tl.load(KT_lo_ptr + pid_m)
        kt_hi = tl.load(KT_hi_ptr + pid_m)

        for kt in range(kt_lo, kt_hi + 1):
            offs_n = kt * BLOCK_N + tl.arange(0, BLOCK_N)

            k = tl.load(K_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                         mask=offs_n[:, None] < S, other=0.0)
            v = tl.load(V_ptr + base + offs_n[:, None] * stride_s + offs_d[None, :] * stride_d,
                         mask=offs_n[:, None] < S, other=0.0)

            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)), allow_tf32=True) * sm_scale

            q_pos = offs_m[:, None]
            k_pos = offs_n[None, :]
            is_causal = k_pos <= q_pos
            is_window = k_pos >= (q_pos - window_size + 1)
            is_mem_col = k_pos >= S - n_skip_rope
            is_mem_row = q_pos >= S - n_skip_rope
            vmask = (is_causal & is_window) | is_mem_col | is_mem_row
            if exclude_self:
                vmask = vmask & (k_pos != q_pos)
            valid = vmask & (offs_m[:, None] < S) & (offs_n[None, :] < S)
            s = tl.where(valid, s, float("-inf"))

            p = tl.exp(s - lse[:, None])
            p = tl.where(valid, p, 0.0)

            dp = tl.dot(do.to(tl.float32), tl.trans(v.to(tl.float32)), allow_tf32=True)
            ds = p * (dp - Di[:, None])
            ds = tl.where(valid, ds, 0.0)

            dq += tl.dot((ds * sm_scale).to(k.dtype), k.to(k.dtype), allow_tf32=True)

        tl.store(dQ_ptr + base + offs_m[:, None] * stride_s + offs_d[None, :] * stride_d,
                 dq.to(q.dtype), mask=q_valid)


# ---------------------------------------------------------------------------
# Tile-size constants (tuned for sm_120 / Blackwell)
# ---------------------------------------------------------------------------

_BLOCK_Q = 64
_BLOCK_K = 64


# ---------------------------------------------------------------------------
# Tile-bound computation (shared by forward and backward)
# ---------------------------------------------------------------------------

def _compute_kt_bounds(S, window_size, n_skip_rope, num_q_tiles, num_k_tiles, device):
    """Compute per-Q-tile KV-tile bounds [kt_lo, kt_hi] for forward and dQ backward.
    Skip-RoPE tokens are appended as a suffix: positions [S-n_skip_rope, S)."""
    qt_idx = torch.arange(num_q_tiles, device=device, dtype=torch.int32)
    q_starts = qt_idx * _BLOCK_Q
    q_ends = torch.clamp(q_starts + _BLOCK_Q - 1, max=S - 1)

    window_lefts = torch.clamp(q_starts - window_size + 1, min=0)
    kt_lo = (window_lefts // _BLOCK_K).to(torch.int32)
    kt_hi = (q_ends // _BLOCK_K).to(torch.int32)

    if n_skip_rope > 0:
        # Suffix keys at end → all q-tiles must reach the last k-tile.
        kt_hi[:] = num_k_tiles - 1
        # Suffix q-tiles see all k-tiles → kt_lo = 0.
        has_memory_query = q_starts >= S - n_skip_rope
        zero_start = torch.zeros(num_q_tiles, device=device, dtype=torch.int32)
        kt_lo = torch.where(has_memory_query, zero_start, kt_lo)

    kt_lo = torch.clamp(kt_lo, min=0, max=num_k_tiles - 1)
    kt_hi = torch.clamp(kt_hi, min=0, max=num_k_tiles - 1)
    return kt_lo, kt_hi


def _compute_qt_bounds(S, window_size, n_skip_rope, num_q_tiles, num_k_tiles, device):
    """Compute per-KV-tile Q-tile bounds [qt_lo, qt_hi] for dK/dV backward.
    Skip-RoPE tokens are appended as a suffix: positions [S-n_skip_rope, S)."""
    kt_idx = torch.arange(num_k_tiles, device=device, dtype=torch.int32)
    k_starts = kt_idx * _BLOCK_K
    k_ends = torch.clamp(k_starts + _BLOCK_K - 1, max=S - 1)

    qt_lo = (k_starts // _BLOCK_Q).to(torch.int32)
    qt_hi = ((k_ends + window_size - 1) // _BLOCK_Q).to(torch.int32)
    qt_hi = torch.clamp(qt_hi, max=num_q_tiles - 1)

    if n_skip_rope > 0:
        # MAC q-tiles (at end) attend to ALL k-tiles → all k-tiles need qt_hi at end
        qt_hi[:] = num_q_tiles - 1
        # MAC k-tiles (at end) are attended by ALL q-tiles → qt_lo = 0 for those
        has_memory_key = k_starts >= S - n_skip_rope
        zero_start = torch.zeros(num_k_tiles, device=device, dtype=torch.int32)
        qt_lo = torch.where(has_memory_key, zero_start, qt_lo)

    qt_lo = torch.clamp(qt_lo, min=0, max=num_q_tiles - 1)
    qt_hi = torch.clamp(qt_hi, min=0, max=num_q_tiles - 1)
    return qt_lo, qt_hi


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _fused_window_forward(q, k, v, window_size, n_skip_rope, exclude_self, scale):
    """Raw Triton forward — returns (output, lse)."""
    B, H, S, D = q.shape
    BLOCK_D = D

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    out = torch.empty_like(q)
    lse = torch.empty(B * H, S, dtype=torch.float32, device=q.device)

    num_q_tiles = triton.cdiv(S, _BLOCK_Q)
    num_k_tiles = triton.cdiv(S, _BLOCK_K)
    kt_lo, kt_hi = _compute_kt_bounds(S, window_size, n_skip_rope, num_q_tiles, num_k_tiles, q.device)

    stride_b = H * S * D
    stride_h = S * D
    stride_s = D
    stride_d = 1

    grid = (B * H, num_q_tiles)
    _fused_window_attn_kernel[grid](
        q, k, v, out, lse,
        stride_b, stride_h, stride_s, stride_d,
        kt_lo, kt_hi,
        S, D, scale,
        window_size, n_skip_rope, exclude_self,
        BLOCK_Q=_BLOCK_Q, BLOCK_K=_BLOCK_K, BLOCK_D=BLOCK_D,
        num_stages=1, num_warps=8,
    )
    return out, lse


def _fused_window_backward(q, k, v, o, do, lse, window_size, n_skip_rope, exclude_self, scale):
    """Triton backward — returns (dq, dk, dv). No mask materialization."""
    B, H, S, D = q.shape
    BH = B * H
    sm_scale = scale

    q_flat = q.reshape(BH, S, D).contiguous()
    k_flat = k.reshape(BH, S, D).contiguous()
    v_flat = v.reshape(BH, S, D).contiguous()
    o_flat = o.reshape(BH, S, D).contiguous()
    do_flat = do.reshape(BH, S, D).contiguous()

    stride_bh = S * D
    stride_s = D
    stride_d = 1

    dq = torch.zeros_like(q_flat)
    dk = torch.zeros_like(k_flat)
    dv = torch.zeros_like(v_flat)

    D_vec = (o_flat.float() * do_flat.float()).sum(dim=-1)

    num_q_tiles = triton.cdiv(S, _BLOCK_Q)
    num_k_tiles = triton.cdiv(S, _BLOCK_K)
    kt_lo, kt_hi = _compute_kt_bounds(S, window_size, n_skip_rope, num_q_tiles, num_k_tiles, q.device)
    qt_lo, qt_hi = _compute_qt_bounds(S, window_size, n_skip_rope, num_q_tiles, num_k_tiles, q.device)

    launch_kw = dict(num_stages=1, num_warps=8)

    grid_kv = (num_k_tiles, BH)
    _fused_window_bwd_dkdv[grid_kv](
        q_flat, k_flat, v_flat, do_flat, dk, dv, lse, D_vec,
        qt_lo, qt_hi,
        S, D, sm_scale,
        window_size, n_skip_rope, exclude_self,
        stride_bh, stride_s, stride_d,
        BLOCK_M=_BLOCK_Q, BLOCK_N=_BLOCK_K,
        **launch_kw,
    )

    grid_q = (num_q_tiles, BH)
    _fused_window_bwd_dq[grid_q](
        q_flat, k_flat, v_flat, do_flat, dq, lse, D_vec,
        kt_lo, kt_hi,
        S, D, sm_scale,
        window_size, n_skip_rope, exclude_self,
        stride_bh, stride_s, stride_d,
        BLOCK_M=_BLOCK_Q, BLOCK_N=_BLOCK_K,
        **launch_kw,
    )

    return dq.reshape(B, H, S, D), dk.reshape(B, H, S, D), dv.reshape(B, H, S, D)


def _build_window_mask(S, window_size, n_skip_rope, exclude_self, device, dtype):
    """Build the S×S mask (kept for reference implementation only)."""
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)
    win_mask = (col <= row) & (col >= row - window_size + 1)
    if n_skip_rope > 0:
        win_mask[:, -n_skip_rope:] = True
        win_mask[-n_skip_rope:, :] = True
    if exclude_self:
        win_mask = win_mask & (col != row)
    return torch.where(win_mask, 0.0, float("-inf")).to(dtype).unsqueeze(0).unsqueeze(0)


class _FusedWindowAttnFunction(torch.autograd.Function):
    """Fused forward + backward, both in Triton. No SDPA fallback."""

    @staticmethod
    def forward(ctx, q, k, v, window_size, n_skip_rope, exclude_self, scale):
        out, lse = _fused_window_forward(q, k, v, window_size, n_skip_rope, exclude_self, scale)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.window_size = window_size
        ctx.n_skip_rope = n_skip_rope
        ctx.exclude_self = exclude_self
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, k, v, o, lse = ctx.saved_tensors
        dq, dk, dv = _fused_window_backward(
            q, k, v, o, grad_output.contiguous(), lse,
            ctx.window_size, ctx.n_skip_rope, ctx.exclude_self, ctx.scale,
        )
        return dq, dk, dv, None, None, None, None


def fused_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    n_skip_rope: int = 0,
    exclude_self: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Fused causal sliding-window attention with autograd support.

    Forward AND backward both use Triton kernels — no S×S mask, no SDPA fallback.

    Args:
        q: [B, H, S, D]
        k: [B, H, S, D]
        v: [B, H, S, D]
        window_size: Positions each query attends to (window covers [q-w+1, q]).
        n_skip_rope: Memory token count (always visible, bidirectional).
        exclude_self: If True, queries cannot attend to own position.
        scale: Attention scale. Defaults to D**-0.5.

    Returns:
        Tensor [B, H, S, D], same dtype as q.
    """
    B, H, S, D = q.shape
    if scale is None:
        scale = D ** -0.5

    if not TRITON_AVAILABLE or D not in (32, 64, 128):
        return fused_window_attention_reference(
            q, k, v, window_size, n_skip_rope, exclude_self, scale
        )

    return _FusedWindowAttnFunction.apply(
        q, k, v, window_size, n_skip_rope, exclude_self, scale
    )


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch — for correctness testing)
# ---------------------------------------------------------------------------

def fused_window_attention_reference(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
    n_skip_rope: int = 0,
    exclude_self: bool = False,
    scale: Optional[float] = None,
) -> Tensor:
    """Pure-PyTorch reference.  Materialises the S×S mask — correct but slow."""
    B, H, S, D = q.shape
    if scale is None:
        scale = D ** -0.5

    device = q.device
    row = torch.arange(S, device=device).unsqueeze(1)
    col = torch.arange(S, device=device).unsqueeze(0)

    win_mask = (col <= row) & (col >= row - window_size + 1)

    if n_skip_rope > 0:
        win_mask[:, -n_skip_rope:] = True
        win_mask[-n_skip_rope:, :] = True

    if exclude_self:
        win_mask = win_mask & (col != row)

    win_bias = torch.where(win_mask, 0.0, float("-inf")).to(q.dtype)
    win_bias = win_bias.unsqueeze(0).unsqueeze(0)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=win_bias, scale=scale)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    torch.manual_seed(42)
    device = "cuda"
    dtype  = torch.bfloat16

    def run_test(tag, B, H, S, D, window, n_skip, excl_self, tol=2e-2):
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        ref = fused_window_attention_reference(q_ref, k_ref, v_ref, window, n_skip, excl_self)
        ker = fused_window_attention(q, k, v, window, n_skip, excl_self)

        fwd_err = (ref.float() - ker.float()).abs().max().item()

        grad_out = torch.randn_like(ker)
        ref.backward(grad_out)
        ker.backward(grad_out)

        dq_err = (q.grad.float() - q_ref.grad.float()).abs().max().item()
        dk_err = (k.grad.float() - k_ref.grad.float()).abs().max().item()
        dv_err = (v.grad.float() - v_ref.grad.float()).abs().max().item()

        dq_cos = F.cosine_similarity(q.grad.reshape(-1).float(), q_ref.grad.reshape(-1).float(), dim=0).item()
        dk_cos = F.cosine_similarity(k.grad.reshape(-1).float(), k_ref.grad.reshape(-1).float(), dim=0).item()

        fwd_ok = fwd_err < tol
        bwd_ok = dq_cos > 0.995 and dk_cos > 0.995
        status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
        print(f"  [{status}] {tag:<45} fwd={fwd_err:.2e}  dQ={dq_err:.2e}(cos={dq_cos:.4f})  dK={dk_err:.2e}(cos={dk_cos:.4f})  dV={dv_err:.2e}")
        return fwd_err, dq_err

    print("=" * 100)
    print("fused_window_attention — forward + backward correctness test")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    print("\n[Correctness — forward + backward]")
    run_test("S=512,  w=128, n_skip=0",          1, 12,  512, 64, 128,  0, False)
    run_test("S=512,  w=128, n_skip=8",          1, 12,  512, 64, 128,  8, False)
    run_test("S=512,  w=128, excl_self",         1, 12,  512, 64, 128,  0, True)
    run_test("S=1024, w=256, n_skip=16",         2,  8, 1024, 64, 256, 16, False)
    run_test("S=2048, w=256, n_skip=0",          1, 12, 2048, 64, 256,  0, False)
    run_test("S=2048, w=256, n_skip=8",          1, 12, 2048, 64, 256,  8, False)
    run_test("S=2048, w=512, n_skip=8",          1, 12, 2048, 64, 512,  8, False)

    print("\n[Speed — FWD+BWD, B=1, H=12, S=2048, D=64, w=256]")
    B, H, S, D, w = 1, 12, 2048, 64, 256
    q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(B, H, S, D, device=device, dtype=dtype)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)

    WARMUP, REPS = 10, 50

    def bench_fwdbwd(fn, n=REPS, warmup=WARMUP):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    win_bias = _build_window_mask(S, w, 0, False, device, dtype)

    def fb_sdpa():
        o = F.scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=win_bias)
        o.backward(grad)

    def fb_triton():
        o = fused_window_attention(q, k, v, w)
        o.backward(grad)

    t_sdpa = bench_fwdbwd(fb_sdpa)
    t_triton = bench_fwdbwd(fb_triton)

    print(f"  SDPA + S×S mask fwd+bwd:  {t_sdpa:.3f} ms")
    print(f"  Triton fused fwd+bwd:     {t_triton:.3f} ms")
    print(f"  Speedup:                  {t_sdpa / t_triton:.2f}×")
    print()
