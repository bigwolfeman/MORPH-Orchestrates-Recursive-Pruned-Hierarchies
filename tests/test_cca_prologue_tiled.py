"""model.cca_prologue_tiled: the row-tiled CCA prologue against an fp64 reference.

The tiled kernels (``_FusedCCAPrologueRows``) and the per-(row, head) kernels
(``_FusedCCAPrologue``) compute the same function with different reduction orders, so they
are not bit-identical. The contract tested here (blockfuse, 2026-10-08): on the same bf16
inputs, the tiled path's error against ``cca_prologue_reference`` run in float64 is no worse
than the per-(row, head) path's error, for the three outputs and for EVERY input gradient,
at the production shapes and layouts (q_lat / k_lat / v_curr as slices of one fused GEMM
output, q_conv / k_conv as transposed conv outputs) and at edge shapes (a row count that is
not a multiple of the tile, trailing positions that skip RoPE, contiguous inputs). Also: each
input gradient comes back in its input's memory layout.

CUDA only (Triton kernels).
"""

import pytest
import torch

from morph.kernels.triton.fused_cca_prologue import cca_prologue_reference, fused_cca_prologue

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H, HKV, D = 8, 4, 64              # the winner's d_model 1024, 8 heads, 4 kv heads, compression 2
VH = HKV * D // 2
NAMES = ["q_lat", "k_lat", "q_conv", "k_conv", "v_curr", "v_prev", "wq", "wk", "temp"]
# The tiled error may exceed the per-(row, head) error by this relative margin plus this
# absolute floor (the two paths round a handful of elements differently; measured ratios
# 0.98-1.01). A real defect moves the error by orders of magnitude (see the sabotage
# record in the note .agents/notes/proposed/architecture/2026-10-08-block-glue-fusion.md).
REL_SLACK, ABS_FLOOR = 0.02, 1e-7


def _inputs(B, S, strided, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    n = H * D + HKV * D + 2 * VH + 256
    y = torch.randn(B, S, n, device=dev, dtype=bf, generator=g)
    q_lat = y[..., :H * D]
    k_lat = y[..., H * D:H * D + HKV * D]
    v_curr = y[..., H * D + HKV * D:H * D + HKV * D + VH]
    v_prev = torch.randn(B, S, VH, device=dev, dtype=bf, generator=g)
    q_conv = torch.randn(B, H * D, S, device=dev, dtype=bf, generator=g).transpose(1, 2)
    k_conv = torch.randn(B, HKV * D, S, device=dev, dtype=bf, generator=g).transpose(1, 2)
    if not strided:
        q_lat, k_lat, v_curr, q_conv, k_conv = (
            t.contiguous() for t in (q_lat, k_lat, v_curr, q_conv, k_conv))
    wq = 1 + 0.1 * torch.randn(D, device=dev, generator=g)
    wk = 1 + 0.1 * torch.randn(D, device=dev, generator=g)
    temp = 0.3 * torch.randn(HKV, device=dev, generator=g)
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=dev).float() / D))
    f = torch.outer(torch.arange(S, device=dev).float(), inv)
    e = torch.cat([f, f], -1)
    gos = [torch.randn(B, H, S, D, device=dev, generator=g) for _ in range(3)]
    return [q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp], e.cos(), e.sin(), gos


def _run(ins, cos, sin, gos, n_skip, tiled):
    xs = [t.detach().requires_grad_(True) for t in ins]        # detach keeps the strides
    outs = fused_cca_prologue(*xs[:6], *xs[6:9], cos, sin, H, HKV, D, n_skip, 1e-6,
                              tiled=tiled)
    torch.autograd.backward(list(outs), [g.to(o.dtype) for g, o in zip(gos, outs)])
    return [o.detach() for o in outs], [x.grad for x in xs]


def _run64(ins, cos, sin, gos, n_skip):
    xs = [t.detach().double().requires_grad_(True) for t in ins]
    outs = cca_prologue_reference(*xs[:6], *xs[6:9], cos.double(), sin.double(), H, HKV, D,
                                  n_skip, 1e-6)
    torch.autograd.backward(list(outs), [g.double() for g in gos])
    return [o.detach() for o in outs], [x.grad for x in xs]


def _err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


CASES = [
    (6, 1280, True, 0),     # prelude / coda block, production layouts
    (6, 256, True, 0),      # core pass
    (3, 37, True, 5),       # rows not a multiple of the tile; trailing positions skip RoPE
    (2, 64, False, 0),      # contiguous inputs
]


@pytest.mark.parametrize("B,S,strided,n_skip", CASES)
def test_tiled_error_no_worse_than_per_row_kernels(B, S, strided, n_skip):
    ins, cos, sin, gos = _inputs(B, S, strided)
    o_ref, g_ref = _run64(ins, cos, sin, gos, n_skip)
    o_old, g_old = _run(ins, cos, sin, gos, n_skip, tiled=False)
    o_new, g_new = _run(ins, cos, sin, gos, n_skip, tiled=True)
    rows = []
    for name, a, b, r in zip(["q", "k", "v"] + ["grad_" + n for n in NAMES],
                             o_old + g_old, o_new + g_new, o_ref + g_ref):
        e_old, e_new = _err(a, r), _err(b, r)
        rows.append((name, e_old, e_new))
        assert e_new <= e_old * (1 + REL_SLACK) + ABS_FLOOR, (
            f"{name}: tiled error {e_new:.3e} > per-row error {e_old:.3e}")
    print("\n".join(f"  {n:12s} old {a:.3e} new {b:.3e}" for n, a, b in rows))


def test_grads_keep_input_layouts():
    ins, cos, sin, gos = _inputs(2, 96, True)
    _, g_new = _run(ins, cos, sin, gos, 0, tiled=True)
    for name, x, g in zip(NAMES[:4], ins[:4], g_new[:4]):
        # q_lat / k_lat are slices (not dense): their grads are contiguous; q_conv / k_conv
        # are dense transposed views: their grads keep the transposed strides.
        want = x.stride() if x.is_contiguous() or name.endswith("conv") else \
            torch.empty(x.shape).stride()
        assert g.stride() == want, (name, g.stride(), want)
