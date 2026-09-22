"""Fused SEGMENT-RESET CCA causal convolutions — Triton forward AND backward.

The segmented twin of :mod:`morph.kernels.triton.fused_cca_conv`. It computes
``morph.model.attention.segment_causal_conv_reference`` — the two stacked causal
convs (depthwise, then head-grouped) with every tap that crosses a SEGMENT
boundary zeroed at BOTH stages:

    out1[b, c, t] = sum_j  w_dw[c, 0, K-1-j] * x[b, c, t-j]    * m[b, t, j]
    out2[b, c, t] = sum_j  sum_i w_gp[c, i, K-1-j] * out1[b, g*Cg+i, t-j] * m[b, t, j]
    m[b, t, j]    = (t - j >= 0) and (seg[b, t-j] == seg[b, t])

with g = c // Cg, Cg = w_gp.shape[1], and the SAME tap order nn.Conv1d uses on a
left-padded input (weight index ``jw = K-1-j``, i.e. ``src = t - p + jw``, p = K-1).

Why this exists
---------------
``segment_causal_conv`` is reached whenever attention gets ``tg_seg`` — which the
LXTUL-R register-reach path passes at EVERY core layer of EVERY loop pass
(``transformer.py``: ``_kw0["tg_seg"] = _seg_slot``). The eager form builds the K-1
boundary masks, then runs K shifted pointwise ``F.conv1d`` per stage. Measured on a
5090 with ``torch.profiler`` at the arm's own shapes: **147 CUDA kernel launches** for
one forward on [6, 512, 256] (99 on the [6, 256, 256] k stream) and **236** for
forward+backward, on tensors far too small to hide that. A profile of
``tul_slot_spandec_strict_fan4_all_reach1`` (971 ms/step, GPU 86 % busy at 8.8
TFLOPS — latency bound) attributed ~175 ms/step (~18 %) to this prologue:
cudnn_convolution 46 ms, convolution_backward 61 ms, dgrad2d_grouped_direct 37 ms,
nchwToNhwc/nhwcToNchw 30 ms. The unsegmented path costs ~25 ms/step because it is
this module's sibling — one kernel per stage.

Relation to ``fused_cca_conv``
------------------------------
Same grid, same tiles, same two-pass weight-gradient reduction. The ONLY
difference is that every load of the convolution's INPUT (forward and both
backward stages) carries one extra mask term, the segment equality:

    forward / dW :  seg[b, src] == seg[b, t]     src = t - p + jw
    dX           :  seg[b, gsrc] == seg[b, T]    gsrc = T + p - jw

Both read as "the tap and the output position are in the same segment", which is
what the reference multiplies in as ``same[j]``. The mask depends on (t, jw) only —
never on the channel — so on the grouped stage it applies to one operand tile of
the ``tl.dot`` and the contraction is untouched.

Design notes (sm_120 / RTX 5090)
--------------------------------
  * Two stages, two kernels — NOT one fused kernel. Fusing them would need a
    (K-1)-wide halo of stage-1 output per time tile and a stage-1 recompute in the
    backward; the sibling file measured that merging even the two grouped BACKWARD
    kernels was ~2x slower from register/dot pressure. Two kernels already take the
    launch count from 147 to 2 (forward) and from 236 to 12 (forward+backward),
    measured, which is where the win is on a latency-bound step.
  * Weight gradients reduce in TWO PASSES: each program writes an fp32 partial to a
    per-(b, time-tile) slab, the host sums dim 0. No atomics — the result is
    bit-reproducible run to run (MORPH has a deterministic mode that would otherwise
    lose it) and there is no contention on the C*K / C*Cg*K hot addresses.
  * num_stages=1, num_warps=8: consumer Blackwell has no TMA pipeline.
  * fp32 accumulation everywhere; the output is cast back to the input dtype on
    store. ``tl.dot`` runs on tensor cores for bf16/fp16 and takes
    ``input_precision="ieee"`` for fp32 inputs (TF32 would cost ~1e-3 relative).
  * The segment ids are read through an explicit batch stride, so the expanded
    ``[B, S]`` view the call sites build (``torch.arange(...).expand(B, ...)``,
    stride 0 on the batch axis) needs no materialisation.

Rounding vs the reference: the reference rounds each tap's result to the input
dtype and sums the K taps in that dtype; this kernel accumulates all K taps in
fp32. Under bf16 that is a ~1e-2 relative difference, exactly as the unsegmented
``fused_cca_conv`` already differs from ``causal_conv_reference``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ._eager_flag import force_eager, kernel_fence

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - triton is a hard dep of the GPU path
    TRITON_AVAILABLE = False


_LAUNCH = dict(num_stages=1, num_warps=8)

# tl.dot needs every tile dim >= 16, so the grouped stage needs Cg >= 16 and a
# time tile >= 16. Anything narrower falls back to the eager reference.
_MIN_DOT = 16

# Shared-memory budget for the grouped stage's two live [CG, BLOCK_T] tiles. A
# Blackwell block may request ~99 KiB; 64 KiB keeps one block per SM comfortably and
# is what CG=64 / BLOCK_T=128 / fp32 already costs (measured by AOT compile).
_SMEM_BUDGET = 64 * 1024


# ===========================================================================
# Triton kernels
# ===========================================================================

if TRITON_AVAILABLE:

    @triton.jit
    def _seg_dw_fwd_kernel(
        x_ptr,           # [B, C, S]
        w_ptr,           # [C, K]
        seg_ptr,         # [*, S] int, batch stride passed explicitly
        y_ptr,           # [B, C, S]
        B, C, S, seg_sb,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ttiles = (S + BLOCK_T - 1) // BLOCK_T
        t_tile = pid % n_ttiles
        bc = pid // n_ttiles
        c = bc % C
        b = bc // C

        p = K - 1
        t = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < S

        row = (b * C + c) * S
        srow = b * seg_sb
        seg_t = tl.load(seg_ptr + srow + t, mask=tmask, other=-1)

        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for jw in tl.static_range(K):
            src = t - p + jw
            smask = tmask & (src >= 0) & (src < S)
            seg_s = tl.load(seg_ptr + srow + src, mask=smask, other=-1)
            smask = smask & (seg_s == seg_t)
            xv = tl.load(x_ptr + row + src, mask=smask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + jw).to(tl.float32)
            acc += wv * xv
        tl.store(y_ptr + row + t, acc.to(y_ptr.dtype.element_ty), mask=tmask)

    @triton.jit
    def _seg_gp_fwd_kernel(
        x_ptr,           # [B, C, S]  (depthwise output)
        w_ptr,           # [C, Cg, K]
        seg_ptr,         # [*, S]
        y_ptr,           # [B, C, S]
        B, C, S, seg_sb,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
        PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ttiles = (S + BLOCK_T - 1) // BLOCK_T
        t_tile = pid % n_ttiles
        bg = pid // n_ttiles
        g = bg % (C // CG)
        b = bg // (C // CG)
        ch0 = g * CG

        p = K - 1
        co = tl.arange(0, CG)
        ci = tl.arange(0, CG)
        t = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < S
        srow = b * seg_sb
        seg_t = tl.load(seg_ptr + srow + t, mask=tmask, other=-1)

        acc = tl.zeros((CG, BLOCK_T), dtype=tl.float32)
        for jw in tl.static_range(K):
            w_off = (ch0 + co)[:, None] * (CG * K) + ci[None, :] * K + jw
            Wj = tl.load(w_ptr + w_off)                                  # [CG, CG]
            src = t - p + jw
            colmask = tmask & (src >= 0) & (src < S)
            seg_s = tl.load(seg_ptr + srow + src, mask=colmask, other=-1)
            colmask = colmask & (seg_s == seg_t)
            x_off = (b * C + ch0 + ci)[:, None] * S + src[None, :]
            Xj = tl.load(x_ptr + x_off, mask=colmask[None, :], other=0.0)  # [CG, BT]
            if PREC == "ieee":
                acc = tl.dot(Wj, Xj, acc, input_precision="ieee")
            else:
                acc = tl.dot(Wj, Xj, acc)
        out_off = (b * C + ch0 + co)[:, None] * S + t[None, :]
        tl.store(y_ptr + out_off, acc.to(y_ptr.dtype.element_ty), mask=tmask[None, :])

    @triton.jit
    def _seg_dw_bwd_kernel(
        go_ptr,          # [B, C, S]  grad wrt depthwise output
        w_ptr,           # [C, K]
        x_ptr,           # [B, C, S]  depthwise INPUT
        seg_ptr,         # [*, S]
        dx_ptr,          # [B, C, S]
        dw_ptr,          # [B*n_tt, C, K] fp32 partials
        B, C, S, seg_sb,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ttiles = (S + BLOCK_T - 1) // BLOCK_T
        t_tile = pid % n_ttiles
        bc = pid // n_ttiles
        c = bc % C
        b = bc // C
        p = K - 1
        row = (b * C + c) * S
        srow = b * seg_sb
        slab = b * n_ttiles + t_tile

        t = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < S
        seg_t = tl.load(seg_ptr + srow + t, mask=tmask, other=-1)

        gv_cur = tl.load(go_ptr + row + t, mask=tmask, other=0.0).to(tl.float32)
        slab_base = slab * (C * K) + c * K

        dx_acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for jw in tl.static_range(K):
            # dX[T] += w[c,jw] * go[c, T+p-jw] * [seg[T+p-jw] == seg[T]]
            gsrc = t + p - jw
            gmask = tmask & (gsrc >= 0) & (gsrc < S)
            seg_g = tl.load(seg_ptr + srow + gsrc, mask=gmask, other=-1)
            gmask = gmask & (seg_g == seg_t)
            gv_sh = tl.load(go_ptr + row + gsrc, mask=gmask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + jw).to(tl.float32)
            dx_acc += wv * gv_sh
            # dW[c,jw] += sum_t go[c,t] * x[c, t-p+jw] * [seg[t-p+jw] == seg[t]]
            src = t - p + jw
            smask = tmask & (src >= 0) & (src < S)
            seg_s = tl.load(seg_ptr + srow + src, mask=smask, other=-1)
            smask = smask & (seg_s == seg_t)
            xv = tl.load(x_ptr + row + src, mask=smask, other=0.0).to(tl.float32)
            tl.store(dw_ptr + slab_base + jw, tl.sum(gv_cur * xv, axis=0))
        tl.store(dx_ptr + row + t, dx_acc.to(dx_ptr.dtype.element_ty), mask=tmask)

    @triton.jit
    def _seg_gp_bwd_dx_kernel(
        go_ptr,          # [B, C, S]  grad wrt grouped output
        w_ptr,           # [C, Cg, K]
        seg_ptr,         # [*, S]
        dx_ptr,          # [B, C, S]
        B, C, S, seg_sb,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
        PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ttiles = (S + BLOCK_T - 1) // BLOCK_T
        t_tile = pid % n_ttiles
        bg = pid // n_ttiles
        g = bg % (C // CG)
        b = bg // (C // CG)
        ch0 = g * CG
        p = K - 1
        ci = tl.arange(0, CG)
        cl = tl.arange(0, CG)
        T = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = T < S
        srow = b * seg_sb
        seg_T = tl.load(seg_ptr + srow + T, mask=tmask, other=-1)

        acc = tl.zeros((CG, BLOCK_T), dtype=tl.float32)
        for jw in tl.static_range(K):
            w_off = (ch0 + cl)[None, :] * (CG * K) + ci[:, None] * K + jw
            WjT = tl.load(w_ptr + w_off)                                  # [CG, CG]
            gsrc = T + p - jw
            colmask = tmask & (gsrc >= 0) & (gsrc < S)
            seg_g = tl.load(seg_ptr + srow + gsrc, mask=colmask, other=-1)
            colmask = colmask & (seg_g == seg_T)
            g_off = (b * C + ch0 + cl)[:, None] * S + gsrc[None, :]
            Gj = tl.load(go_ptr + g_off, mask=colmask[None, :], other=0.0)  # [CG, BT]
            if PREC == "ieee":
                acc = tl.dot(WjT, Gj, acc, input_precision="ieee")
            else:
                acc = tl.dot(WjT, Gj, acc)
        dx_off = (b * C + ch0 + ci)[:, None] * S + T[None, :]
        tl.store(dx_ptr + dx_off, acc.to(dx_ptr.dtype.element_ty), mask=tmask[None, :])

    @triton.jit
    def _seg_gp_bwd_dw_kernel(
        go_ptr,          # [B, C, S]
        x_ptr,           # [B, C, S]  grouped INPUT (= depthwise output)
        seg_ptr,         # [*, S]
        dw_ptr,          # [B*n_tt, C, Cg, K] fp32 partials
        B, C, S, seg_sb,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
        PREC: tl.constexpr,
    ):
        pid = tl.program_id(0)
        n_ttiles = (S + BLOCK_T - 1) // BLOCK_T
        t_tile = pid % n_ttiles
        bg = pid // n_ttiles
        g = bg % (C // CG)
        b = bg // (C // CG)
        ch0 = g * CG
        p = K - 1
        slab = b * n_ttiles + t_tile

        co = tl.arange(0, CG)
        ci = tl.arange(0, CG)
        T = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = T < S
        srow = b * seg_sb
        seg_T = tl.load(seg_ptr + srow + T, mask=tmask, other=-1)

        go_off = (b * C + ch0 + co)[:, None] * S + T[None, :]
        Go = tl.load(go_ptr + go_off, mask=tmask[None, :], other=0.0)      # [CG, BT]

        slab_base = slab * (C * CG * K)
        for jw in tl.static_range(K):
            src = T - p + jw
            colmask = tmask & (src >= 0) & (src < S)
            seg_s = tl.load(seg_ptr + srow + src, mask=colmask, other=-1)
            colmask = colmask & (seg_s == seg_T)
            x_off = (b * C + ch0 + ci)[:, None] * S + src[None, :]
            Xj = tl.load(x_ptr + x_off, mask=colmask[None, :], other=0.0)  # [CG, BT]
            if PREC == "ieee":
                dWj = tl.dot(Go, tl.trans(Xj), tl.zeros((CG, CG), dtype=tl.float32),
                             input_precision="ieee")
            else:
                dWj = tl.dot(Go, tl.trans(Xj), tl.zeros((CG, CG), dtype=tl.float32))
            w_off = slab_base + (ch0 + co)[:, None] * (CG * K) + ci[None, :] * K + jw
            tl.store(dw_ptr + w_off, dWj)


# ===========================================================================
# Tile config / launch helpers
# ===========================================================================

def _block_t(S: int) -> int:
    """One warp-friendly time tile, never below the tl.dot minimum of 16."""
    if S >= 128:
        return 128
    return max(_MIN_DOT, 1 << max(S - 1, 0).bit_length())


def _gp_tile_cap(CG: int, itemsize: int) -> int:
    """Largest BLOCK_T the grouped kernels can ask for inside ``_SMEM_BUDGET``.

    Measured by AOT-compiling every kernel for sm_120 (scratch script in the build
    report), the grouped stage's shared-memory request is:

        forward / dX :  (CG*CG + CG*BLOCK_T) * itemsize   ([CG,CG] weight tile + one operand)
        dW           :  <= 2*CG*BLOCK_T * itemsize        (Go and Xj, fp32 needs both)

    e.g. CG=64, BLOCK_T=128: 24,576 B at bf16 and 49,152 / 65,536 B at fp32, all
    measured. The number that bites is the [CG,CG] weight tile: at CG=128 and fp32 it
    is 64 KiB on its own, so NO tile fits and the shapes are refused (the cap comes
    back below _MIN_DOT) rather than launched into the ~99 KiB per-block ceiling.
    """
    per_row = CG * itemsize
    return min(_SMEM_BUDGET // per_row - CG, _SMEM_BUDGET // (2 * per_row))


def _block_t_gp(S: int, CG: int, itemsize: int) -> int:
    """The grouped stage's time tile: the depthwise tile, shrunk to fit shared memory."""
    bt = _block_t(S)
    cap = _gp_tile_cap(CG, itemsize)
    while bt > _MIN_DOT and bt > cap:
        bt //= 2
    return bt


def _prec(dtype: torch.dtype) -> str:
    """fp32 operands must not silently drop to TF32 inside tl.dot."""
    return "ieee" if dtype == torch.float32 else "tf32"


def _seg_strides(seg: Tensor, B: int) -> tuple[Tensor, int]:
    """Return (seg tensor the kernel can index, batch stride in elements).

    The call sites hand in ``torch.arange(n).repeat_interleave(m).expand(B, L)``
    (batch stride 0) and ``tg_segment_ids(layout)`` (contiguous). Both are indexed
    through the stride with no copy. A ``[1, S]`` row broadcasts like the reference's
    mask does.
    """
    if seg.dim() != 2:
        raise ValueError(f"seg must be [B, S], got {tuple(seg.shape)}")
    if seg.shape[0] not in (1, B):
        raise ValueError(f"seg batch {seg.shape[0]} matches neither 1 nor B={B}")
    if seg.stride(1) != 1:
        seg = seg.contiguous()
    sb = 0 if seg.shape[0] == 1 else seg.stride(0)
    return seg, sb


def _dw_forward(x, w_dw, seg, sb, K):
    B, C, S = x.shape
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    y = torch.empty_like(x)
    _seg_dw_fwd_kernel[(B * C * n_tt,)](x, w_dw, seg, y, B, C, S, sb,
                                        K=K, BLOCK_T=BT, **_LAUNCH)
    return y


def _gp_forward(x, w_gp, seg, sb, CG, K):
    B, C, S = x.shape
    G = C // CG
    BT = _block_t_gp(S, CG, x.element_size())
    y = torch.empty_like(x)
    _seg_gp_fwd_kernel[(B * G * ((S + BT - 1) // BT),)](
        x, w_gp, seg, y, B, C, S, sb,
        CG=CG, K=K, BLOCK_T=BT, PREC=_prec(x.dtype), **_LAUNCH)
    return y


def _dw_backward(go, x_in, w_dw, seg, sb, K):
    B, C, S = go.shape
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    dx = torch.empty_like(go)
    dw_part = torch.empty(B * n_tt, C, K, device=go.device, dtype=torch.float32)
    _seg_dw_bwd_kernel[(B * C * n_tt,)](go, w_dw, x_in, seg, dx, dw_part,
                                        B, C, S, sb, K=K, BLOCK_T=BT, **_LAUNCH)
    return dx, dw_part.sum(dim=0)                        # [C, K]


def _gp_backward(go, x_in, w_gp, seg, sb, CG, K, need_dx: bool, need_dw: bool):
    B, C, S = go.shape
    G = C // CG
    BT = _block_t_gp(S, CG, go.element_size())
    n_tt = (S + BT - 1) // BT
    grid = (B * G * n_tt,)
    prec = _prec(go.dtype)
    dx = None
    dw = None
    if need_dx:
        dx = torch.empty_like(go)
        _seg_gp_bwd_dx_kernel[grid](go, w_gp, seg, dx, B, C, S, sb,
                                    CG=CG, K=K, BLOCK_T=BT, PREC=prec, **_LAUNCH)
    if need_dw:
        dw_part = torch.empty(B * n_tt, C, CG, K, device=go.device, dtype=torch.float32)
        _seg_gp_bwd_dw_kernel[grid](go, x_in, seg, dw_part, B, C, S, sb,
                                    CG=CG, K=K, BLOCK_T=BT, PREC=prec, **_LAUNCH)
        dw = dw_part.sum(dim=0)                          # [C, Cg, K]
    return dx, dw


# ===========================================================================
# autograd.Function
# ===========================================================================

class _FusedSegCausalConv(torch.autograd.Function):
    """Both stages of the segment-reset causal conv, forward and backward.

    Gradient contract, matching what the eager reference's autograd produces:
    ``dx`` in ``x``'s dtype, ``dw_dw`` in ``w_dw``'s dtype and shape ``[C, 1, K]``,
    ``dw_gp`` in ``w_gp``'s dtype. An input with ``requires_grad=False`` gets ``None``.

    Which kernels that skips: the grouped dW kernel runs only for ``w_gp``, and the
    whole depthwise backward is skipped when neither ``x`` nor ``w_dw`` wants a
    gradient. The depthwise kernel computes dX and the dW partials TOGETHER (it loads
    the incoming gradient once for both), so asking for only one of them still runs
    both — splitting it would cost a second pass over the gradient.
    """

    @staticmethod
    def forward(ctx, x, w_dw, w_gp, seg, CG, K):
        x = x.contiguous()
        w_dw2 = w_dw.reshape(w_dw.shape[0], K).contiguous()   # [C, K]
        w_gp = w_gp.contiguous()                              # [C, Cg, K]
        seg, sb = _seg_strides(seg, x.shape[0])

        y_dw = _dw_forward(x, w_dw2, seg, sb, K)
        y_gp = _gp_forward(y_dw, w_gp, seg, sb, CG, K)

        ctx.save_for_backward(x, y_dw, w_dw2, w_gp, seg)
        ctx.CG, ctx.K, ctx.sb = CG, K, sb
        ctx.w_dw_shape = tuple(w_dw.shape)
        return y_gp

    @staticmethod
    def backward(ctx, grad_out):
        x, y_dw, w_dw2, w_gp, seg = ctx.saved_tensors
        CG, K, sb = ctx.CG, ctx.K, ctx.sb
        need_x, need_dw, need_gp = ctx.needs_input_grad[:3]
        if grad_out.dtype != w_gp.dtype:
            # x, w_dw and w_gp share one dtype (seg_conv_shapes_supported enforces it)
            # and the output carries it, so the incoming grad carries it too. tl.dot
            # would otherwise be handed two different element types.
            raise TypeError(
                f"fused segment conv: grad dtype {grad_out.dtype} != weight dtype "
                f"{w_gp.dtype}")
        grad_out = grad_out.contiguous()

        # The grouped stage's d_input feeds the depthwise stage, so it is needed
        # whenever ANY of x / w_dw wants a gradient, not only x.
        need_ydw = need_x or need_dw
        d_ydw, dw_gp = _gp_backward(grad_out, y_dw, w_gp, seg, sb, CG, K,
                                    need_dx=need_ydw, need_dw=need_gp)

        dx = None
        dw_dw = None
        if need_ydw:
            dx_full, dw_dw_full = _dw_backward(d_ydw, x, w_dw2, seg, sb, K)
            if need_x:
                dx = dx_full.to(x.dtype)
            if need_dw:
                dw_dw = dw_dw_full.reshape(ctx.w_dw_shape).to(w_dw2.dtype)
        if dw_gp is not None:
            dw_gp = dw_gp.to(w_gp.dtype)
        return dx, dw_dw, dw_gp, None, None, None


# ===========================================================================
# Public API
# ===========================================================================

def seg_conv_shapes_supported(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor) -> bool:
    """Can the Triton path run these shapes/dtypes at all? (device-agnostic)

    The grouped stage is a ``tl.dot`` over a ``[Cg, Cg]`` weight tile, so Cg must be
    a power of two of at least 16 and must divide C. ``tl.dot`` also needs both
    operands in one dtype, which is what the call sites already do (they cast the
    weights to the conv input's dtype). Everything else falls back to the eager
    reference — which is the only implementation for those shapes, not a second one.
    """
    if x_BCS.dim() != 3 or w_dw.dim() != 3 or w_gp.dim() != 3:
        return False
    C = x_BCS.shape[1]
    K = w_dw.shape[-1]
    Cg = w_gp.shape[1]
    if w_dw.shape[0] != C or w_gp.shape[0] != C or w_gp.shape[-1] != K:
        return False
    if w_dw.shape[1] != 1:
        return False
    if K < 1 or Cg < _MIN_DOT or (Cg & (Cg - 1)) != 0 or C % Cg != 0:
        return False
    if x_BCS.dtype != w_dw.dtype or x_BCS.dtype != w_gp.dtype:
        return False
    if x_BCS.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return False
    # The grouped stage must have a legal time tile inside the shared-memory budget.
    # A head too wide for it (d_head 128 at fp32) stays on the eager reference: the
    # fix there is a split group contraction, not a launch that may not fit.
    if _gp_tile_cap(Cg, x_BCS.element_size()) < _MIN_DOT:
        return False
    return True


def seg_conv_available(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor) -> bool:
    """Shape support PLUS the runtime conditions: Triton, CUDA, kernels not forced off."""
    if not TRITON_AVAILABLE or force_eager() or not x_BCS.is_cuda:
        return False
    return seg_conv_shapes_supported(x_BCS, w_dw, w_gp)


@kernel_fence  # Dynamo fence: kernel is opaque (autograd.Function)
def fused_segment_causal_conv(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor,
                              seg: Tensor) -> Tensor:
    """Fused segment-reset causal conv pair on ``[B, C, S]``.

    Args:
        x_BCS: [B, C, S] channels-first input (the call site's ``q_lat.transpose(1, 2)``).
        w_dw:  [C, 1, K] depthwise weight (nn.Conv1d groups=C).
        w_gp:  [C, C//G, K] grouped weight (nn.Conv1d groups=G, G inferred from the shape).
        seg:   [B, S] or [1, S] integer segment ids. A tap is kept only when its
               position carries the same id as the output position.

    Returns:
        [B, C, S] in ``x_BCS``'s dtype — ``segment_causal_conv_reference``'s function,
        with the K taps accumulated in fp32 instead of the input dtype.

    Raises if the shapes are outside the kernel's support; callers use
    :func:`seg_conv_available` to pick the eager reference instead.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("fused_segment_causal_conv needs Triton")
    if not seg_conv_shapes_supported(x_BCS, w_dw, w_gp):
        raise ValueError(
            f"unsupported shapes for the fused segment conv: x {tuple(x_BCS.shape)} "
            f"{x_BCS.dtype}, w_dw {tuple(w_dw.shape)}, w_gp {tuple(w_gp.shape)}")
    return _FusedSegCausalConv.apply(x_BCS, w_dw, w_gp, seg, w_gp.shape[1],
                                     w_dw.shape[-1])
