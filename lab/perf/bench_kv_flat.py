"""Microbench + parity for model.kv_flat_kernels (morph/kernels/triton/kv_flat.py).

Times the in-tree backward and the re-tiled one at the core-pass shapes, each replayed from a
CUDA graph (launch gaps excluded, L2 warm like the model's pass), and scores both against an
fp64 reference. Run under the GPU lock:
  PYTHONPATH=. python lab/perf/bench_kv_flat.py [--which conv|rows|all]
"""
from __future__ import annotations

import argparse

import torch

from morph.kernels.triton import fused_cca_conv as cv
from morph.kernels.triton import fused_cca_prologue as pr
from morph.kernels.triton import kv_flat

DEV = "cuda"
BF = torch.bfloat16


def graph_time(fn, reps=200, warm=5) -> float:
    """Median-of-5 us per call of ``fn`` replayed from one CUDA graph of ``reps`` calls."""
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    ts = []
    for _ in range(5):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / reps)
    return sorted(ts)[2]


def conv_ref64(go, x, w_dw, w_gp, CG, K):
    """fp64 autograd of the two stacked causal convs: (dx, dw_dw [C,K], dw_gp [C,CG,K])."""
    C = x.shape[1]
    xd = x.double().requires_grad_(True)
    wd = w_dw.double().reshape(C, 1, K).requires_grad_(True)
    wg = w_gp.double().requires_grad_(True)
    y = cv.causal_conv_reference(xd, wd, wg, K)
    y.backward(go.double())
    return xd.grad, wd.grad.reshape(C, K), wg.grad


def rel_err(a, ref):
    return ((a.double() - ref).norm() / ref.norm()).item()


def bench_conv(B=6, C=768, S=256, CG=64, K=4, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, C, S, device=DEV, dtype=BF, generator=g)
    go = torch.randn(B, C, S, device=DEV, dtype=BF, generator=g)
    w_dw = (torch.randn(C, K, device=DEV, generator=g) * 0.3).to(BF)
    w_gp = (torch.randn(C, CG, K, device=DEV, generator=g) * 0.1).to(BF)
    y_dw = cv._dw_forward(x, w_dw, K)
    old = lambda: cv._cca_conv_bwd_impl(go, x, y_dw, w_dw, w_gp, CG, K)
    new = lambda: kv_flat.cca_conv_bwd_flat(go, x, y_dw, w_dw, w_gp, CG, K)
    t_old, t_new = graph_time(old), graph_time(new)
    ref = conv_ref64(go, x, w_dw, w_gp, CG, K)
    o, n = old(), new()
    names = ("dx", "dw_dw", "dw_gp")
    print(f"conv bwd B={B} C={C} S={S} CG={CG}: old {t_old:.1f} us  new {t_new:.1f} us  "
          f"x{t_old / t_new:.2f}")
    for nm, a, b, r in zip(names, o, n, ref):
        print(f"   {nm:6s} rel err vs fp64: old {rel_err(a, r):.3e}  new {rel_err(b, r):.3e}")
    return t_old, t_new


def rows_inputs(B, S, H=8, Hkv=4, D=64, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *sh: torch.randn(*sh, device=DEV, dtype=BF, generator=g)
    n_rep = H // Hkv
    # the model's layouts: q_lat / k_lat slices of the in-projection output, the conv outputs
    # transposed views of [B, C, S]
    proj = r(B, S, 1280)
    q_lat, k_lat = proj[..., :H * D], proj[..., H * D:H * D + Hkv * D]
    q_conv = r(B, H * D + Hkv * D, S)[:, :H * D].transpose(1, 2)
    k_conv = r(B, H * D + Hkv * D, S)[:, H * D:].transpose(1, 2)
    gq, gk, gv = r(B, H, S, D), r(B, H, S, D), r(B, H, S, D)
    wq = (1 + 0.1 * torch.randn(D, device=DEV, generator=g))
    wk = (1 + 0.1 * torch.randn(D, device=DEV, generator=g))
    temp = 0.1 * torch.randn(Hkv, device=DEV, generator=g)
    cos = torch.randn(S, D, device=DEV, generator=g)
    sin = torch.randn(S, D, device=DEV, generator=g)
    return (gq, gk, gv, q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin,
            H, Hkv, D, n_rep, 0, 1e-6, Hkv * D // 2)


def rows_ref64(args):
    """fp64 autograd of cca_prologue_reference: the nine input grads in _rows_bwd_impl order."""
    (gq, gk, gv, q_lat, k_lat, q_conv, k_conv, wq, wk, temp, cos, sin, H, Hkv, D, n_rep, nsr,
     eps, vhalf) = args
    B, S, _ = q_lat.shape
    d = lambda t: t.double().detach().requires_grad_(True)
    ql, kl, qc, kc, wq_, wk_, tp = map(d, (q_lat, k_lat, q_conv, k_conv, wq, wk, temp))
    vc = d(torch.randn(B, S, vhalf, device=DEV))
    vp = d(torch.randn(B, S, vhalf, device=DEV))
    q, k, v = pr.cca_prologue_reference(ql, kl, qc, kc, vc, vp, wq_, wk_, tp, cos.double(),
                                        sin.double(), H, Hkv, D, nsr, eps)
    torch.autograd.backward([q, k, v], [gq.double(), gk.double(), gv.double()])
    return [t.grad for t in (ql, kl, qc, kc, vc, vp, wq_, wk_, tp)]


def bench_rows(B=6, S=256):
    args = rows_inputs(B, S)
    old = lambda: pr._rows_bwd_impl(*args, flat=False)
    new = lambda: pr._rows_bwd_impl(*args, flat=True)
    t_old, t_new = graph_time(old), graph_time(new)
    print(f"rows bwd B={B} S={S}: old {t_old:.1f} us  new {t_new:.1f} us  x{t_old / t_new:.2f}")
    try:
        ref = rows_ref64(args)
    except Exception as e:  # reference signature drift must not hide the timing
        raise RuntimeError(f"fp64 reference failed: {e}") from e
    names = ("d_ql", "d_kl", "d_qc", "d_kc", "d_vc", "d_vp", "d_wq", "d_wk", "d_temp")
    for nm, a, b, r in zip(names, old(), new(), ref):
        print(f"   {nm:7s} rel err vs fp64: old {rel_err(a, r):.3e}  new {rel_err(b, r):.3e}")
    return t_old, t_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    if a.which in ("conv", "all"):
        bench_conv()
    if a.which in ("rows", "all"):
        bench_rows(6, 256)
        bench_rows(6, 1280)


if __name__ == "__main__":
    main()
