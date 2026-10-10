"""``lx_attn``: the strict-geometry attention of one LXTUL decode site, in one launch.

The LXTUL fused decoder (``morph/inference/lx_fused_decode.py``, design note
``.agents/notes/proposed/feature/2026-10-10-lxtul-fused-decode.md``, section "Kernel specs
for the kernel agent", K1) runs this kernel at every attention site of the token step and
of the slot step, after ``fused_decode_step.decode_front`` has written the site's query,
its K/V cache entry, the raw gate hidden and the residual row. It replaces the reference's
``tul_generate_cached._branches`` (window = SDPA over a 0 / -inf bias, slot branch = fp32
einsum + softmax with the per-head sink column) and ``attention._CCABase._gate_combine_up``
up to, not including, ``W_up``.

Per (query row b', head h), with ``row = meta[b', 0]`` and
``s_k = (q[b', h] . K[row, h, k]) * scale``:

* slot branch: a softmax over ``{s_k : lo1 <= k < hi1}`` plus the logit ``sink[h]``, whose
  value is the zero vector. The sink enters at the final merge as one more term of logit
  ``sink[h]``, weight 1 and value 0, so an empty range gives ``oc = 0 / 1 = 0`` exactly.
* window branch: the keys of range 1 with ``0 < |d1| < window``, ``d1 = qidx1 - k``
  (``HAS_KPOS`` False) or ``qpos - kpos[row, k]`` (True), and the keys of range 2 with
  ``0 < |qidx2 - k| < window``. Range 2 feeds the window branch only, so its loop runs over
  ``[max(lo2, qidx2 - window + 1), min(hi2, qidx2 + window))``: every key cut by that trim
  has ``|d| >= window`` and is outside the window set (Z3 proof, outside the tree:
  ``morph-scratch/lxkernel/proofs/lx_attn/bounds.py``). An empty set gives ``ow = 0``
  (every running max stays ``-inf``; the shift is then 0, so every weight is
  ``exp(-inf) = 0``, no ``inf - inf`` is formed, and the 0 denominator is replaced by 1).
* ``g0 = sigmoid(w_gate2[2h] . silu(gateh[b']))``, ``g1`` the same with row ``2h + 1``
  (the ``GH > 0`` fold of ``_decode_attn_kernel``, same padded reduction);
  ``out[b', h*D:(h+1)*D] = g0 * oc + g1 * ow + alpha[h] * xres[b', h*D:(h+1)*D]``.

Range 1 is read ONCE: its K and V tiles feed both online softmaxes. Each softmax runs as
BLOCK_N independent streams (lane r of a tile sees keys n0 + r), so the key loop has no
reduction across the key axis; the streams merge once after the loops (``_lx_merge``).
Measured on the 5090 against the first version (one running max per tile, two cross-warp
reductions per branch per tile): loop cell at 388 keys, B' = 8, 13.5 -> 7.8 us.

Preconditions (the decoder's meta tables meet them; the kernel does not check them on the
device, a host check would break graph capture): ``0 <= row < R``,
``0 <= lo1 <= hi1 <= N``, ``0 <= lo2 <= hi2 <= N``, and ranges 1 and 2 disjoint (a key in
both would enter the window branch twice). ``qidx1``, ``qidx2``, ``qpos`` and ``kpos`` are
any int32 values with ``|value| < 2**30`` (no overflow in the distance).

Numerics: fp32 FMA throughout, no ``tl.dot`` (Triton runs an fp32 ``tl.dot`` at TF32 by
default). The score is ``sum(K * q) * scale``, the reference's operand order; the
deviation from the reference is the softmax's reduction order (BLOCK_N online streams
and one merge, vs SDPA's and einsum's). Graph-capturable: every range and index is read from ``meta`` on the
device, the launch grid is ``B' * H`` and every loop bound is a device value, so a replay
with a new ``meta`` runs the new ranges.

Launch: one CTA per (row, head); the tile size and warp count come from ``_launch_cfg``
(measured on the RTX 5090, numbers there).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _lx_stream(m, den, acc, s, v, valid):
    """One online-softmax step of BLOCK_N independent streams: lane r of the tile is its
    own stream (``m``, ``den`` [BLOCK_N], ``acc`` [BLOCK_N, D]), so the loop body has no
    reduction across the key axis (the cross-warp max / sum and their barriers happen once,
    in ``_lx_merge``). ``m`` may be -inf (the stream has seen no valid key): the shift is
    then 0, so a masked score gives exp(-inf) = 0 and the correction exp(-inf - 0) = 0."""
    s = tl.where(valid, s, float("-inf"))
    m_new = tl.maximum(m, s)
    m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
    corr = tl.exp(m - m_use)
    p = tl.exp(s - m_use)
    den = den * corr + p
    acc = acc * corr[:, None] + p[:, None] * v
    return m_new, den, acc


@triton.jit
def _lx_merge(m, den, acc, m0, l0):
    """Merge the BLOCK_N streams and one extra term of logit ``m0``, weight ``l0`` and
    value 0 (the slot branch's sink: m0 = sink, l0 = 1; the window branch: m0 = -inf,
    l0 = 0). Returns (denominator, numerator [D]). Same -inf guard as ``_lx_stream``."""
    mx = tl.maximum(tl.max(m, 0), m0)
    mx = tl.where(mx == float("-inf"), 0.0, mx)
    w = tl.exp(m - mx)
    tot = tl.sum(den * w, 0) + l0 * tl.exp(m0 - mx)
    num = tl.sum(acc * w[:, None], 0)
    return tot, num


@triton.jit
def _lx_attn_kernel(
    Q, K, V, META, KPOS, GATEH, WG2, XRES, SINK, ALPHA, OUT,
    scale,
    H: tl.constexpr, D: tl.constexpr, WIN: tl.constexpr,
    GH: tl.constexpr, BG: tl.constexpr, BLOCK_N: tl.constexpr,
    HAS_KPOS: tl.constexpr,
    sq_b, sq_h,                 # Q [B', H, D]
    sk_r, sk_h, sk_n,           # K [R, H, N, D]
    sv_r, sv_h, sv_n,           # V [R, H, N, D]
    sm_b,                       # META [B', 8]
    skp_r,                      # KPOS [R, N]
    sg_b,                       # GATEH [B', GH]
    sx_b,                       # XRES [B', H*D]
    so_b,                       # OUT [B', H*D]
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    offs_d = tl.arange(0, D)
    offs_bn = tl.arange(0, BLOCK_N)

    mrow = META + b * sm_b
    row = tl.load(mrow + 0).to(tl.int64)
    lo1 = tl.load(mrow + 1)
    hi1 = tl.load(mrow + 2)
    lo2 = tl.load(mrow + 3)
    hi2 = tl.load(mrow + 4)
    qidx1 = tl.load(mrow + 5)
    qidx2 = tl.load(mrow + 6)
    qpos = tl.load(mrow + 7)

    q = tl.load(Q + b * sq_b + h * sq_h + offs_d)                        # [D] fp32
    kb = K + row * sk_r + h * sk_h
    vb = V + row * sv_r + h * sv_h

    # BLOCK_N streams per branch, nothing seen yet (the sink enters at the merge)
    m_s = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    l_s = tl.zeros((BLOCK_N,), tl.float32)
    acc_s = tl.zeros((BLOCK_N, D), tl.float32)
    m_w = tl.full((BLOCK_N,), float("-inf"), tl.float32)
    l_w = tl.zeros((BLOCK_N,), tl.float32)
    acc_w = tl.zeros((BLOCK_N, D), tl.float32)

    # ── range 1: slot columns, both branches from one K/V read ──
    for n0 in range(lo1, hi1, BLOCK_N):
        offs_n = n0 + offs_bn
        mk = offs_n < hi1
        kt = tl.load(kb + offs_n[:, None] * sk_n + offs_d[None, :], mask=mk[:, None], other=0.0)
        vt = tl.load(vb + offs_n[:, None] * sv_n + offs_d[None, :], mask=mk[:, None], other=0.0)
        s = tl.sum(kt * q[None, :], 1) * scale                           # [BLOCK_N]
        m_s, l_s, acc_s = _lx_stream(m_s, l_s, acc_s, s, vt, mk)
        if HAS_KPOS:
            kp = tl.load(KPOS + row * skp_r + offs_n, mask=mk, other=0)
            d1 = qpos - kp
        else:
            d1 = qidx1 - offs_n
        ad = tl.abs(d1)
        m_w, l_w, acc_w = _lx_stream(m_w, l_w, acc_w, s, vt, mk & (ad > 0) & (ad < WIN))

    # ── range 2: token columns, window branch only, trimmed to |qidx2 - k| < WIN ──
    lo2t = tl.maximum(lo2, qidx2 - WIN + 1)
    hi2t = tl.minimum(hi2, qidx2 + WIN)
    for n0 in range(lo2t, hi2t, BLOCK_N):
        offs_n = n0 + offs_bn
        mk = offs_n < hi2t
        kt = tl.load(kb + offs_n[:, None] * sk_n + offs_d[None, :], mask=mk[:, None], other=0.0)
        vt = tl.load(vb + offs_n[:, None] * sv_n + offs_d[None, :], mask=mk[:, None], other=0.0)
        s = tl.sum(kt * q[None, :], 1) * scale
        d2 = qidx2 - offs_n
        m_w, l_w, acc_w = _lx_stream(m_w, l_w, acc_w, s, vt, mk & (d2 != 0))

    sink = tl.load(SINK + h)
    den_s, num_s = _lx_merge(m_s, l_s, acc_s, sink, 1.0)
    den_w, num_w = _lx_merge(m_w, l_w, acc_w, float("-inf"), 0.0)
    oc = num_s / den_s                                                   # den_s >= 1
    ow = num_w / tl.where(den_w > 0.0, den_w, 1.0)

    # ── gate2 fold (decode_attn's GH > 0 path) + residual alpha ──
    offs_g = tl.arange(0, BG)
    mg = offs_g < GH
    ghid = tl.load(GATEH + b * sg_b + offs_g, mask=mg, other=0.0)
    ghid = ghid * tl.sigmoid(ghid)                                       # silu
    w0 = tl.load(WG2 + (h * 2 + 0) * GH + offs_g, mask=mg, other=0.0).to(tl.float32)
    w1 = tl.load(WG2 + (h * 2 + 1) * GH + offs_g, mask=mg, other=0.0).to(tl.float32)
    g0 = tl.sigmoid(tl.sum(ghid * w0, 0))
    g1 = tl.sigmoid(tl.sum(ghid * w1, 0))
    xres = tl.load(XRES + b * sx_b + h * D + offs_d)
    alpha = tl.load(ALPHA + h)
    out = g0 * oc + g1 * ow + alpha * xres
    tl.store(OUT + b * so_b + h * D + offs_d, out)


def _launch_cfg(N: int, n_ctas: int) -> tuple[int, int]:
    """(BLOCK_N, num_warps) from what the host knows: the cache length N and the grid size
    B' * H. Sweep of BLOCK_N {16..128} x warps {1..8} x stages {1, 2}, CUDA-graph replay
    on the RTX 5090 at production widths (H 8, D 64, GH 256, window 256), us per call:

    | site (keys read)              | B' | (16, 1) | (32, 4) | best of sweep     |
    |-------------------------------|----|---------|---------|-------------------|
    | prelude token (31)            |  8 |   2.56  |   2.57  | 2.56  (16, 1)     |
    | prelude token (31)            | 64 |   3.16  |   4.24  | 3.14  (16, 1, st2)|
    | prelude cell (32 + 4)         |  8 |   3.29  |   2.71  | 2.71  (32, 4)     |
    | loop cell (88)                |  8 |   6.53  |   3.42  | 3.42  (32, 4)     |
    | loop cell (388)               |  8 |  23.12  |   7.78  | 6.79  (64, 8)     |
    | coda token (84 + 31)          |  8 |   7.72  |   3.92  | 3.92  (32, 4)     |
    | coda token (388 + 31)         |  8 |  24.41  |   8.50  | 7.27  (64, 8)     |
    | coda cell (1)                 |  8 |   2.01  |   2.33  | 2.00  (16, 1)     |

    (32, 4) is within 1.13x of each case's best on average over the 24 cases (worst 1.48x,
    the coda cell at B' = 64). The one host-visible case where it loses clearly is a short
    cache (the prelude, N = P + M) at a large grid: (16, 1) there. The fixture width (D 32)
    uses the same rule; it was not tuned."""
    if N <= 64 and n_ctas >= 512:
        return 16, 1
    return 32, 4


def lx_attn(q: Tensor, k_cache: Tensor, v_cache: Tensor, meta: Tensor,
            kpos: Tensor | None, gateh: Tensor, w_gate2: Tensor, xres: Tensor,
            sink: Tensor, alpha: Tensor, window: int, scale: float) -> Tensor:
    """Strict-geometry LXTUL site attention + gate + residual alpha (module docstring).

    q [B', H, D] fp32 (last dim contiguous); k_cache / v_cache [R, H, N, D] fp32 (last dim
    contiguous); meta [B', 8] int32 = row, lo1, hi1, lo2, hi2, qidx1, qidx2, qpos;
    kpos [R, N] int32 or None; gateh [B', GH] fp32 (gate[0](x), before SiLU); w_gate2
    [2H, GH] fp32 or bf16 (contiguous); xres [B', H*D] fp32; sink [H] fp32; alpha [H] fp32;
    window (cca.window_size); scale = D ** -0.5. Returns [B', H*D] fp32, W_up-ready.
    Shape checks only: nothing here reads a device value, so the call is graph-capturable."""
    Bq, H, D = q.shape
    R, Hk, N, Dk = k_cache.shape
    GH = gateh.shape[1]
    assert (Hk, Dk) == (H, D) and v_cache.shape == k_cache.shape, (
        f"k/v cache {tuple(k_cache.shape)}/{tuple(v_cache.shape)} vs q heads {H} x {D}")
    assert D >= 16 and (D & (D - 1)) == 0, f"d_head {D} must be a power of two >= 16"
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.float32
    assert q.stride(2) == 1 and k_cache.stride(3) == 1 and v_cache.stride(3) == 1
    assert meta.shape == (Bq, 8) and meta.dtype == torch.int32 and meta.stride(1) == 1
    assert gateh.shape == (Bq, GH) and gateh.stride(1) == 1 and gateh.dtype == torch.float32
    assert w_gate2.shape == (2 * H, GH) and w_gate2.is_contiguous()
    assert w_gate2.dtype in (torch.float32, torch.bfloat16)
    assert xres.shape == (Bq, H * D) and xres.stride(1) == 1 and xres.dtype == torch.float32
    assert sink.numel() == H and alpha.numel() == H
    assert sink.is_contiguous() and alpha.is_contiguous()
    assert sink.dtype == alpha.dtype == torch.float32
    assert window >= 1
    has_kpos = kpos is not None
    if has_kpos:
        assert kpos.shape == (R, N) and kpos.dtype == torch.int32 and kpos.stride(1) == 1
    out = torch.empty(Bq, H * D, device=q.device, dtype=torch.float32)
    if Bq == 0:
        return out
    block_n, warps = _launch_cfg(N, Bq * H)
    _lx_attn_kernel[(Bq * H,)](
        q, k_cache, v_cache, meta, kpos if has_kpos else meta, gateh, w_gate2, xres,
        sink, alpha, out, float(scale),
        H=H, D=D, WIN=int(window), GH=GH, BG=triton.next_power_of_2(max(GH, 1)),
        BLOCK_N=block_n, HAS_KPOS=has_kpos,
        sq_b=q.stride(0), sq_h=q.stride(1),
        sk_r=k_cache.stride(0), sk_h=k_cache.stride(1), sk_n=k_cache.stride(2),
        sv_r=v_cache.stride(0), sv_h=v_cache.stride(1), sv_n=v_cache.stride(2),
        sm_b=meta.stride(0),
        skp_r=kpos.stride(0) if has_kpos else 0,
        sg_b=gateh.stride(0), sx_b=xres.stride(0), so_b=out.stride(0),
        num_stages=1, num_warps=warps,
    )
    return out
