"""Fused CCA causal convolutions — Triton forward AND backward (sm_120 / Blackwell).

Replaces the pair of stacked causal Conv1d calls in ``_CCABase._causal_conv``:

    p = kernel - 1                       # causal left-pad
    x = conv_dw(F.pad(x, (p, 0)))        # depthwise: Conv1d(C, C, k, groups=C)
    x = conv_gp(F.pad(x, (p, 0)))        # grouped:   Conv1d(C, C, k, groups=G)

on a ``[B, C, S]`` tensor (C = latent channel dim). No bias, causal (left-pad
by p=k-1, output length == input length S).

Why this exists
---------------
On consumer Blackwell (sm_120, RTX 5090) cuDNN's GROUPED conv1d weight-gradient
backward is poorly optimized: a torch.profiler run shows
``convolution_backward`` (88us x 1200) + ``wgrad2d_grouped_direct_kernel``
(135us x 600) dominating after the matmuls (~10% of the step). This kernel
computes d_input, d_w_dw, d_w_gp directly in Triton, avoiding cuDNN's slow
grouped wgrad path entirely.

Layout convention
------------------
Operates on ``[B, C, S]`` (channels-first) to match the call site exactly
(``q_lat.transpose(1, 2)`` is already ``[B, C, S]``). Weights match nn.Conv1d:
  * depthwise w_dw: ``[C, 1, k]``     (groups=C)
  * grouped   w_gp: ``[C, C//G, k]``  (groups=G)

Causal correlation semantics (matches nn.Conv1d on a left-padded input)
-----------------------------------------------------------------------
For an output position t in [0, S):
    y[c, t] = sum_{j=0..k-1} w[c, o, j] * xin[in_ch, t - p + j]
where p = k - 1, and the read at (t - p + j) is masked to [0, S) (causal zero
pad on the left). For the depthwise conv in_ch == c, o == 0. For the grouped
conv the inner channel ci runs over the group of width Cg = C//G and
in_ch = g*Cg + ci, o = ci.

Backward of a causal correlation y[c,t] = sum_j w[c,o,j]*xin[in_ch, t-p+j]:
    d_w[c,o,j]      = sum_t  go[c,t] * xin[in_ch, t - p + j]   (masked t-p+j)
    d_xin[in_ch,T]  = sum_{routes c,o,j with t-p+j==T} go[c, T + p - j] * w[c,o,j]

Design for sm_120
-----------------
  * num_stages=1, num_warps=8 (consumer Blackwell has NO TMA pipeline).
  * bf16 in/out, fp32 accumulation for every reduction.
  * One forward program = one (b, channel/group, time-tile). The k-window and
    (for grouped) the Cg=32-wide group contraction are register/SRAM-resident.
  * Branchless: kernel size and modes are tl.constexpr; causal boundary is a
    masked load (other=0.0), never a host branch.

Author: TileProver (Claude Code, Opus 4.8)
Date:   2026-05-31
Branch: 006-looped-block-ell
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    TRITON_AVAILABLE = False


_LAUNCH = dict(num_stages=1, num_warps=8)


# ===========================================================================
# Triton kernels
# ===========================================================================

if TRITON_AVAILABLE:

    # -----------------------------------------------------------------------
    # Depthwise forward. One program = one (b, c, time-tile).
    # y[c, t] = sum_j w_dw[c, j] * x[c, t - p + j]   (masked, p = K-1)
    # -----------------------------------------------------------------------
    @triton.jit
    def _dw_fwd_kernel(
        x_ptr,           # [B, C, S]
        w_ptr,           # [C, K]   (depthwise: middle dim 1 squeezed by caller)
        y_ptr,           # [B, C, S]
        B, C, S,
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
        t = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)      # output positions
        tmask = t < S

        row = (b * C + c) * S
        acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(K):
            src = t - p + j                                # input position
            smask = tmask & (src >= 0) & (src < S)
            xv = tl.load(x_ptr + row + src, mask=smask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + j).to(tl.float32)
            acc += wv * xv
        tl.store(y_ptr + row + t, acc.to(y_ptr.dtype.element_ty), mask=tmask)

    # -----------------------------------------------------------------------
    # Grouped forward (tensor-core). One program = one (b, group g, time-tile).
    # Computes the whole CG-wide group output block [CG, BLOCK_T] via K matmuls:
    #   Y[co, t] = sum_j (Wj @ Xj)[co, t]
    #   Wj[co, ci] = w[g*Cg+co, ci, j]   ([CG, CG])
    #   Xj[ci, t]  = xin[g*Cg+ci, t - p + j]  ([CG, BLOCK_T], causal-masked)
    # -----------------------------------------------------------------------
    @triton.jit
    def _gp_fwd_kernel(
        x_ptr,           # [B, C, S]   (depthwise output)
        w_ptr,           # [C, Cg, K]
        y_ptr,           # [B, C, S]
        B, C, S,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
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

        acc = tl.zeros((CG, BLOCK_T), dtype=tl.float32)
        for j in tl.static_range(K):
            # weight tile Wj[co, ci] = w[(ch0+co)*Cg*K + ci*K + j]
            w_off = (ch0 + co)[:, None] * (CG * K) + ci[None, :] * K + j
            Wj = tl.load(w_ptr + w_off).to(tl.float32)                 # [CG, CG]
            # input tile Xj[ci, t] = xin[ch0+ci, t-p+j]
            src = t - p + j
            xmask = (src >= 0) & (src < S) & tmask[None, :]
            x_off = (b * C + ch0 + ci)[:, None] * S + src[None, :]
            Xj = tl.load(x_ptr + x_off, mask=xmask, other=0.0).to(tl.float32)  # [CG, BT]
            acc += tl.dot(Wj, Xj, allow_tf32=False)
        # store [CG, BLOCK_T] block
        out_off = (b * C + ch0 + co)[:, None] * S + t[None, :]
        tl.store(y_ptr + out_off, acc.to(y_ptr.dtype.element_ty),
                 mask=tmask[None, :])

    # -----------------------------------------------------------------------
    # Depthwise backward — FUSED d_input + d_weight. One program = (b, c, t-tile).
    # Loads go[c,*] once and computes both:
    #   dX[c, T] = sum_j w_dw[c,j] * go[c, T + p - j]        (masked T+p-j in [0,S))
    #   dW[c, j] = sum_t go[c,t]   * x[c, t - p + j]         (masked, partial slab)
    # dW -> per-(b,t_tile) partial slab [B*n_tt, C, K] (no atomics); host reduces.
    # -----------------------------------------------------------------------
    @triton.jit
    def _dw_bwd_kernel(
        go_ptr,          # [B, C, S]   grad wrt depthwise output
        w_ptr,           # [C, K]
        x_ptr,           # [B, C, S]   depthwise INPUT
        dx_ptr,          # [B, C, S]
        dw_ptr,          # [B*n_tt, C, K]  partials
        B, C, S,
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
        slab = b * n_ttiles + t_tile

        t = t_tile * BLOCK_T + tl.arange(0, BLOCK_T)
        tmask = t < S

        # dW path: go at the tile positions (also reused conceptually below)
        gv_cur = tl.load(go_ptr + row + t, mask=tmask, other=0.0).to(tl.float32)
        slab_base = slab * (C * K) + c * K

        dx_acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
        for j in tl.static_range(K):
            # dX: dX[T] += w[c,j] * go[c, T+p-j]
            gsrc = t + p - j
            gmask = tmask & (gsrc >= 0) & (gsrc < S)
            gv_sh = tl.load(go_ptr + row + gsrc, mask=gmask, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + c * K + j).to(tl.float32)
            dx_acc += wv * gv_sh
            # dW: dW[c,j] += sum_t go[c,t] * x[c, t-p+j]
            src = t - p + j
            smask = tmask & (src >= 0) & (src < S)
            xv = tl.load(x_ptr + row + src, mask=smask, other=0.0).to(tl.float32)
            tl.store(dw_ptr + slab_base + j, tl.sum(gv_cur * xv, axis=0))
        tl.store(dx_ptr + row + t, dx_acc.to(dx_ptr.dtype.element_ty), mask=tmask)

    # -----------------------------------------------------------------------
    # Grouped backward — d_input (tensor-core). One program = (b, group, t-tile).
    # NOTE: kept SEPARATE from the dW kernel on purpose — fusing the two [CG,CG]
    # matmuls per j into one program doubles register/dot pressure and measured
    # ~2x SLOWER on the (compute-bound) Q stream. Separate launches win here.
    #   dX[ci, T] = sum_cl sum_j w[g*Cg+cl, ci, j] * go[g*Cg+cl, T+p-j]
    #   per j: WjT[ci,cl] = w[g*Cg+cl, ci, j]  ([CG,CG]) ; Gj[cl,T]=go[..,T+p-j]
    #          dX += WjT @ Gj
    # -----------------------------------------------------------------------
    @triton.jit
    def _gp_bwd_dx_kernel(
        go_ptr,          # [B, C, S]   grad wrt grouped output
        w_ptr,           # [C, Cg, K]
        dx_ptr,          # [B, C, S]
        B, C, S,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
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

        acc = tl.zeros((CG, BLOCK_T), dtype=tl.float32)
        for j in tl.static_range(K):
            w_off = (ch0 + cl)[None, :] * (CG * K) + ci[:, None] * K + j
            WjT = tl.load(w_ptr + w_off).to(tl.float32)                           # [CG,CG]
            gsrc = T + p - j
            gmask = (gsrc >= 0) & (gsrc < S) & tmask[None, :]
            g_off = (b * C + ch0 + cl)[:, None] * S + gsrc[None, :]
            Gj = tl.load(go_ptr + g_off, mask=gmask, other=0.0).to(tl.float32)    # [CG,BT]
            acc += tl.dot(WjT, Gj, allow_tf32=False)
        dx_off = (b * C + ch0 + ci)[:, None] * S + T[None, :]
        tl.store(dx_ptr + dx_off, acc.to(dx_ptr.dtype.element_ty),
                 mask=tmask[None, :])

    # -----------------------------------------------------------------------
    # Grouped backward — d_weight (tensor-core). One program = (b, group, t-tile).
    # dW[co,ci,j] = sum_t go[ch0+co,t] * x[ch0+ci, t-p+j]  ;  dWj = Go @ Xj^T.
    # Per-(b,t_tile) partial slab [B*n_tt, C, Cg, K] (no atomics); host reduces.
    # -----------------------------------------------------------------------
    @triton.jit
    def _gp_bwd_dw_kernel(
        go_ptr,          # [B, C, S]
        x_ptr,           # [B, C, S]   grouped INPUT (= depthwise output)
        dw_ptr,          # [B*n_tt, C, Cg, K]  partials
        B, C, S,
        CG: tl.constexpr,
        K: tl.constexpr,
        BLOCK_T: tl.constexpr,
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

        go_off = (b * C + ch0 + co)[:, None] * S + T[None, :]
        Go = tl.load(go_ptr + go_off, mask=tmask[None, :], other=0.0).to(tl.float32)

        slab_base = slab * (C * CG * K)
        for j in tl.static_range(K):
            src = T - p + j
            xmask = (src >= 0) & (src < S) & tmask[None, :]
            x_off = (b * C + ch0 + ci)[:, None] * S + src[None, :]
            Xj = tl.load(x_ptr + x_off, mask=xmask, other=0.0).to(tl.float32)     # [CG,BT]
            dWj = tl.dot(Go, tl.trans(Xj), allow_tf32=False)                      # [CG,CG]
            w_off = slab_base + (ch0 + co)[:, None] * (CG * K) + ci[None, :] * K + j
            tl.store(dw_ptr + w_off, dWj)


# ===========================================================================
# Tile config
# ===========================================================================

def _block_t(S: int) -> int:
    # one warp-friendly tile; keep small enough that the static K-window unroll
    # stays register-resident. 128 covers S in {512..4096} with few tiles.
    return 128 if S >= 128 else 1 << (max(S - 1, 0)).bit_length()


# ===========================================================================
# Python wrappers
# ===========================================================================

def _dw_forward(x, w_dw, K):
    B, C, S = x.shape
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    y = torch.empty_like(x)
    _dw_fwd_kernel[(B * C * n_tt,)](x, w_dw, y, B, C, S, K=K, BLOCK_T=BT, **_LAUNCH)
    return y


def _gp_forward(x, w_gp, CG, K):
    B, C, S = x.shape
    G = C // CG
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    y = torch.empty_like(x)
    _gp_fwd_kernel[(B * G * n_tt,)](x, w_gp, y, B, C, S, CG=CG, K=K, BLOCK_T=BT, **_LAUNCH)
    return y


def _dw_backward(go, x_in, w_dw, K):
    B, C, S = go.shape
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    dx = torch.empty_like(go)
    dw_part = torch.empty(B * n_tt, C, K, device=go.device, dtype=torch.float32)
    _dw_bwd_kernel[(B * C * n_tt,)](go, w_dw, x_in, dx, dw_part,
                                    B, C, S, K=K, BLOCK_T=BT, **_LAUNCH)
    dw = dw_part.sum(dim=0)                              # [C, K]
    return dx, dw


def _gp_backward(go, x_in, w_gp, CG, K):
    B, C, S = go.shape
    G = C // CG
    BT = _block_t(S)
    n_tt = (S + BT - 1) // BT
    dx = torch.empty_like(go)
    dw_part = torch.empty(B * n_tt, C, CG, K, device=go.device, dtype=torch.float32)
    _gp_bwd_dx_kernel[(B * G * n_tt,)](go, w_gp, dx, B, C, S, CG=CG, K=K, BLOCK_T=BT, **_LAUNCH)
    _gp_bwd_dw_kernel[(B * G * n_tt,)](go, x_in, dw_part, B, C, S, CG=CG, K=K, BLOCK_T=BT, **_LAUNCH)
    dw = dw_part.sum(dim=0)                              # [C, Cg, K]
    return dx, dw


# ===========================================================================
# autograd.Function
# ===========================================================================

class _FusedCCAConv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w_dw, w_gp, CG, K):
        x = x.contiguous()
        w_dw2 = w_dw.reshape(w_dw.shape[0], K).contiguous()   # [C, K]
        w_gp = w_gp.contiguous()                              # [C, Cg, K]

        y_dw = _dw_forward(x, w_dw2, K)
        y_gp = _gp_forward(y_dw, w_gp, CG, K)

        ctx.save_for_backward(x, y_dw, w_dw2, w_gp)
        ctx.CG = CG
        ctx.K = K
        ctx.w_dw_shape = tuple(w_dw.shape)
        return y_gp

    @staticmethod
    def backward(ctx, grad_out):
        x, y_dw, w_dw2, w_gp = ctx.saved_tensors
        CG, K = ctx.CG, ctx.K
        grad_out = grad_out.contiguous()

        # grouped: d_input (-> grad of y_dw), d_weight_gp
        d_ydw, dw_gp = _gp_backward(grad_out, y_dw, w_gp, CG, K)
        # depthwise: d_input (-> grad of x), d_weight_dw
        dx, dw_dw = _dw_backward(d_ydw, x, w_dw2, K)

        dw_dw = dw_dw.reshape(ctx.w_dw_shape).to(w_dw2.dtype)   # back to [C,1,K]
        dw_gp = dw_gp.to(w_gp.dtype)
        dx = dx.to(x.dtype)
        return dx, dw_dw, dw_gp, None, None


# ===========================================================================
# Public API
# ===========================================================================

def fused_cca_conv(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor,
                   groups: int, kernel: int) -> Tensor:
    """Fused stacked causal convs (depthwise then head-grouped) on [B, C, S].

    Args:
        x_BCS:  [B, C, S] input (channels-first, matches the call site).
        w_dw:   depthwise weight [C, 1, kernel]   (nn.Conv1d groups=C).
        w_gp:   grouped weight   [C, C//groups, kernel] (nn.Conv1d groups=groups).
        groups: number of head-groups for the grouped conv (G).
        kernel: kernel size k (causal left-pad p = k-1).

    Returns:
        [B, C, S], same dtype as x_BCS. Equivalent to ``_CCABase._causal_conv``.
    """
    C = x_BCS.shape[1]
    assert C % groups == 0, f"C={C} not divisible by groups={groups}"
    CG = C // groups

    if not TRITON_AVAILABLE or not x_BCS.is_cuda:
        return causal_conv_reference(x_BCS, w_dw, w_gp, kernel)

    return _FusedCCAConv.apply(x_BCS, w_dw, w_gp, CG, kernel)


# ===========================================================================
# Pure-PyTorch reference (EXACTLY matches _CCABase._causal_conv)
# ===========================================================================

def causal_conv_reference(x_BCS: Tensor, w_dw: Tensor, w_gp: Tensor,
                          kernel: int) -> Tensor:
    """Two F.pad + F.conv1d, byte-identical to ``_CCABase._causal_conv``.

    w_dw: [C, 1, k] (groups=C);  w_gp: [C, C//G, k] (groups=G inferred from shape).
    """
    p = kernel - 1
    C = x_BCS.shape[1]
    Cg = w_gp.shape[1]
    G = C // Cg
    y = F.conv1d(F.pad(x_BCS, (p, 0)), w_dw, bias=None, groups=C)
    y = F.conv1d(F.pad(y, (p, 0)), w_gp, bias=None, groups=G)
    return y


# ===========================================================================
# Self-test
# ===========================================================================

if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    dev = torch.device("cuda")
    dt = torch.bfloat16
    print("=" * 100)
    print("fused_cca_conv — forward + backward correctness test")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print("=" * 100)

    # production CCA dims:
    #   d_model=768, n_heads=12, n_kv_heads=4, compression=2 -> d_head=32
    #   latent_q_dim = 12*32 = 384, G=12, Cg=32
    #   latent_k_dim = 4*32  = 128, G=4,  Cg=32
    K = 4

    def build_convs(C, G):
        """nn.Conv1d pair with the same causal semantics; returns module pair."""
        dw = torch.nn.Conv1d(C, C, K, groups=C, bias=False).to(dev, dt)
        gp = torch.nn.Conv1d(C, C, K, groups=G, bias=False).to(dev, dt)
        return dw, gp

    def eager_path(x, dw, gp):
        p = K - 1
        y = dw(F.pad(x, (p, 0)))
        y = gp(F.pad(y, (p, 0)))
        return y

    def run_case(B, S, C, G, fwd_max_tol=2e-2, gcos_thresh=0.995):
        dw, gp = build_convs(C, G)
        w_dw = dw.weight.detach().clone()      # [C,1,K]
        w_gp = gp.weight.detach().clone()      # [C,C/G,K]

        x = torch.randn(B, C, S, device=dev, dtype=dt)

        # --- fused ---
        xa = x.detach().clone().requires_grad_(True)
        wda = w_dw.detach().clone().requires_grad_(True)
        wga = w_gp.detach().clone().requires_grad_(True)
        yf = fused_cca_conv(xa, wda, wga, G, K)

        # --- eager cuDNN (shared weights) ---
        xr = x.detach().clone().requires_grad_(True)
        dwr, gpr = build_convs(C, G)
        with torch.no_grad():
            dwr.weight.copy_(w_dw)
            gpr.weight.copy_(w_gp)
        dwr.weight.requires_grad_(True)
        gpr.weight.requires_grad_(True)
        yr = eager_path(xr, dwr, gpr)

        # forward parity
        emax = (yf.float() - yr.float()).abs().max().item()
        emean = (yf.float() - yr.float()).abs().mean().item()
        fcos = F.cosine_similarity(yf.reshape(-1).float(), yr.reshape(-1).float(), 0).item()

        # backward parity (same upstream grad)
        go = torch.randn_like(yf)
        (yf.float() * go.float()).sum().backward()
        (yr.float() * go.float()).sum().backward()

        def gcos(a, b):
            return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), 0).item()

        c_dx = gcos(xa.grad, xr.grad)
        c_wdw = gcos(wda.grad, dwr.weight.grad)
        c_wgp = gcos(wga.grad, gpr.weight.grad)
        worst = min(c_dx, c_wdw, c_wgp)

        fwd_ok = (emax < fwd_max_tol) and (fcos > 0.999)
        bwd_ok = worst > gcos_thresh
        status = "PASS" if (fwd_ok and bwd_ok) else "FAIL"
        print(f"  [{status}] B={B} S={S:<5} C={C:<4} G={G:<3}  "
              f"fwd_max={emax:.2e} fwd_mean={emean:.2e} fwd_cos={fcos:.6f}")
        print(f"           grad cos: d_input={c_dx:.4f}  d_w_dw={c_wdw:.4f}  d_w_gp={c_wgp:.4f}")
        return fwd_ok and bwd_ok

    print("\n[Correctness — forward + backward across dim sweep]")
    all_ok = True
    for S in (512, 1024, 2048, 4096):
        # Q stream: C=384, G=12, Cg=32
        all_ok &= run_case(2, S, 384, 12)
        # K stream: C=128, G=4, Cg=32
        all_ok &= run_case(2, S, 128, 4)

    # --- speed: fwd AND fwd+bwd, fused vs eager cuDNN pair ---
    def bench(fn, n=300, warmup=60):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e3

    # ---- isolated GROUPED-conv backward: the documented profiler bottleneck ----
    # (wgrad2d_grouped_direct_kernel + convolution_backward). cuDNN's grouped
    # wgrad is the slow path on sm_120; this is the apples-to-apples target.
    print("\n[Speed — isolated GROUPED conv backward (dx+dw), the profiler target, B=2 S=2048]")
    for (C, G, name) in [(384, 12, "Q stream"), (128, 4, "K stream")]:
        B, S = 2, 2048
        CG = C // G
        x = torch.randn(B, C, S, device=dev, dtype=dt)
        w_gp = torch.randn(C, CG, K, device=dev, dtype=dt).mul_(0.1)
        go = torch.randn(B, C, S, device=dev, dtype=dt)
        xr = x.detach().clone().requires_grad_(True)
        gp = torch.nn.Conv1d(C, C, K, groups=G, bias=False).to(dev, dt)
        with torch.no_grad():
            gp.weight.copy_(w_gp)
        p = K - 1

        def cudnn_gp_bwd():
            xr.grad = None
            gp.weight.grad = None
            gp(F.pad(xr, (p, 0))).backward(go)

        def fused_gp_bwd():
            _gp_backward(go, x, w_gp, CG, K)

        tc = bench(cudnn_gp_bwd)
        tf = bench(fused_gp_bwd)
        print(f"  {name} (C={C}, G={G}):  cuDNN {tc:.4f} ms  fused {tf:.4f} ms  "
              f"speedup {tc/tf:.2f}x")

    print("\n[Speed — full stacked-pair (dw+gp) fused Triton vs eager cuDNN, B=2, S=2048]")
    for (C, G, name) in [(384, 12, "Q stream"), (128, 4, "K stream")]:
        B, S = 2, 2048
        dw, gp = build_convs(C, G)
        w_dw = dw.weight.detach().clone()
        w_gp = gp.weight.detach().clone()
        x = torch.randn(B, C, S, device=dev, dtype=dt)

        xa = x.detach().clone().requires_grad_(True)
        wda = w_dw.detach().clone().requires_grad_(True)
        wga = w_gp.detach().clone().requires_grad_(True)
        xr = x.detach().clone().requires_grad_(True)
        dwr, gpr = build_convs(C, G)
        with torch.no_grad():
            dwr.weight.copy_(w_dw); gpr.weight.copy_(w_gp)

        def fused_fwd():
            with torch.no_grad():
                fused_cca_conv(xa, wda, wga, G, K)

        def eager_fwd():
            with torch.no_grad():
                eager_path(xr, dwr, gpr)

        def fused_fb():
            xa.grad = wda.grad = wga.grad = None
            y = fused_cca_conv(xa, wda, wga, G, K)
            y.sum().backward()

        def eager_fb():
            xr.grad = None
            dwr.weight.grad = None; gpr.weight.grad = None
            y = eager_path(xr, dwr, gpr)
            y.sum().backward()

        t_ff = bench(fused_fwd); t_ef = bench(eager_fwd)
        t_fb = bench(fused_fb);  t_eb = bench(eager_fb)
        print(f"  {name} (C={C}, G={G}):")
        print(f"    fwd      eager {t_ef:.3f} ms  fused {t_ff:.3f} ms  speedup {t_ef/t_ff:.2f}x")
        print(f"    fwd+bwd  eager {t_eb:.3f} ms  fused {t_fb:.3f} ms  speedup {t_eb/t_fb:.2f}x")

    print("\n" + "=" * 100)
    print("ALL PASS" if all_ok else "SOME FAILED")
    print("=" * 100)
    assert all_ok, "fused_cca_conv self-test FAILED — see rows marked FAIL above"
