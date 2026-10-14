"""Micro-benchmark of the model.hc_region_fused kernels (Cayley HC, n=4) on the 5090.

Runs the entry and exit autograd Functions (forward and backward) at the shapes the
fast recipe uses, collects each Triton kernel's GPU time with the profiler, and reports
achieved bandwidth against the unique bytes the kernel must move.

  rows 7680  prelude / coda   (B=6 x 1280 positions)
  rows 1536 .. 192             core passes (slot_compact shrinks the live rows per pass)

Run under the GPU lock, from the worktree root:
  flock /home/wolfe/morph-scratch/gpu.lock env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTHONPATH=. python lab/perf/bench_hc_kernels.py [--ref BASELINE_ROOT]

Bytes are the unique bytes a kernel must move; rows whose data fits the 96 MB L2 (T <= 1536)
read above the DRAM peak because the carrier is warm. --ref takes a tree root whose
morph/kernels/triton/fused_hyper_connection.py is the baseline: it adds interleaved before/after
timings (min over 5 rounds, so clock drift does not favour one side) and prints, for every
output and grad, "IDENT" (bit-identical) or the max abs error over max |ref|.
"""
import argparse
import collections
import importlib.util
import sys

import torch
from torch.profiler import ProfilerActivity, profile

C = 1024
N = 4
PEAK_GBS = 1790.0
ROWS = [7680, 1536, 768, 384, 192]


def load_module(root: str, name: str):
    path = f"{root}/morph/kernels/triton/fused_hyper_connection.py"
    import morph.kernels.triton  # noqa: F401  (parent package, for the relative imports)
    name = f"morph.kernels.triton.fused_hyper_connection_{name}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_inputs(T, dtype, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    h = (torch.randn(1, T, N, C, device=dev, generator=g) * 2).to(dtype)
    w = torch.randn(48, N * C, device=dev, generator=g) * (3.0 / (N * C) ** 0.5)
    b = torch.randn(48, device=dev, generator=g) * 0.5
    y = torch.randn(1, T, C, device=dev, generator=g).to(torch.bfloat16)
    term = torch.randn(1, T, C, device=dev, generator=g).to(torch.bfloat16)
    gout = torch.randn(1, T, N, C, device=dev, generator=g).to(dtype)
    gx = torch.randn(1, T, C, device=dev, generator=g).to(dtype)
    return h, w, b, y, term, gout, gx


def one_step(K, h0, w0, b0, y0, term0, gout, gx, fused=False):
    """entry fwd -> exit fwd -> exit bwd -> entry bwd. Returns outputs and grads."""
    h = h0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    b = b0.clone().requires_grad_(True)
    y = y0.clone().requires_grad_(True)
    term = term0.clone().requires_grad_(True)
    xbar, hres, hpr, link = K._HCRegionEntry.apply(
        h, w, b, 1.0, 1.0, 1e-6, *((True,) if fused else ()))
    out = K._HCRegionExit.apply(hres, hpr, link, y, term)
    # one backward over both outputs: autograd hands gout to the entry as link's grad
    torch.autograd.backward([out, xbar], [gout, gx])
    return [xbar, hres, hpr, out, h.grad, w.grad, b.grad, y.grad, term.grad]


def kernel_bytes(name, T, e):
    """Unique bytes one call of the named kernel must move (carrier element size e)."""
    h = T * N * C * e
    c_bf = T * C * 2
    if name.startswith("_hcr_fwd_proj"):
        return h + N * T * 64 * 4 + N * T * 4
    if name.startswith("_hcr_fwd_map"):
        return h + c_bf * 0 + T * C * e + N * T * 64 * 4 + T * (16 + 4 + 48 + 1) * 4
    if name.startswith("_hc_post_fwd"):
        return 2 * h + c_bf * 2
    if name.startswith("_hcr_exit_bwd"):
        return h + c_bf * 3
    if name.startswith("_hcr_bwd_reduce"):
        return 2 * h + T * C * e
    if name.startswith("_hcr_bwd_carrier"):
        return 3 * h + T * C * e
    return None


def profile_kernels(K, T, dtype, iters=20, fused=False):
    args = make_inputs(T, dtype)
    for _ in range(5):
        one_step(K, *args, fused=fused)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            one_step(K, *args, fused=fused)
        torch.cuda.synchronize()
    agg = collections.defaultdict(float)
    cnt = collections.Counter()
    for e in p.events():
        if e.device_type.name == "CUDA" and e.name.startswith("_hc"):
            agg[e.name] += e.device_time
            cnt[e.name] += 1
    return {k: (v / cnt[k], cnt[k] // iters) for k, v in agg.items()}


def grids(K, T):
    """Programs launched by each kernel at T rows (mirrors _HCRegionEntry / _HCRegionExit)."""
    import triton
    tu = dict(K._hcr_tune(T)) if hasattr(K, "_hcr_tune") else dict(K._HCR_TUNE)
    tu.setdefault("red_launch_bt", tu["red_bt"])
    sms = torch.cuda.get_device_properties(0).multi_processor_count

    def split(bt, bc, cap):
        s = 1
        while triton.cdiv(T, bt) * s < 2 * sms and C % (2 * s * min(bc, C)) == 0 and s < cap:
            s *= 2
        return s

    ns = split(tu["red_bt"], tu["red_bc"], tu["red_max_split"])
    return {
        "fwd_proj": triton.cdiv(T, tu["fsp_bt"]) * N,
        "fwd_map": triton.cdiv(T, tu["fmap_bt"]) * split(tu["fmap_bt"], tu["fmap_bc"], tu["fmap_max_split"]),
        "post_fwd": T,
        "exit_bwd": T,
        "bwd_reduce": triton.cdiv(T, tu["red_launch_bt"]) * ns,
        "bwd_carrier": triton.cdiv(T, 16) * split(16, tu["car_bc"], tu["car_max_split"]),
    }


def errs(a, b):
    a = a.double()
    b = b.double()
    d = (a - b).abs()
    return d.max().item(), (d.max() / b.abs().max().clamp_min(1e-30)).item()


def ab_times(K, R, T, dtype, rounds=5):
    """Per-kernel us for the new (K) and reference (R) module, interleaved; min over rounds.
    Interleaving keeps GPU clock drift from favouring one side."""
    best_new, best_ref = {}, {}
    for _ in range(rounds):
        for mod, best in ((R, best_ref), (K, best_new)):
            for name, (us, _) in profile_kernels(mod, T, dtype).items():
                best[name] = min(best.get(name, 1e30), us)
    return best_new, best_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--ref", default=None, help="baseline tree root: adds before/after columns and the error check")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--rows", type=int, nargs="*", default=ROWS)
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--bwd-fused", action="store_true", help="model.hc_bwd_fused: the tensor-core carrier-grad kernel")
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)
    e = torch.tensor([], dtype=dtype).element_size()
    K = load_module(a.root, "hc_new")
    R = load_module(a.ref, "hc_ref") if a.ref else None
    print(f"carrier dtype {a.dtype}, C={C}, n={N}, peak {PEAK_GBS} GB/s, SMs "
          f"{torch.cuda.get_device_properties(0).multi_processor_count}")
    for T in a.rows:
        if R:
            new, old = ab_times(K, R, T, dtype)
        else:
            new = {k: v[0] for k, v in profile_kernels(K, T, dtype, fused=a.bwd_fused).items()}
            old = {}
        print(f"\nrows {T}  grids {grids(K, T)}" + (f"  (before grids {grids(R, T)})" if R else ""))
        tn = to = 0.0
        for name in sorted(new):
            by = kernel_bytes(name, T, e)
            gbs = lambda us: by / (us * 1e-6) / 1e9 if by else float("nan")
            row = f"  {name:26s} after {new[name]:7.1f} us {gbs(new[name]):6.0f} GB/s ({100 * gbs(new[name]) / PEAK_GBS:4.0f}%)"
            if name in old:
                row += f" | before {old[name]:7.1f} us {gbs(old[name]):6.0f} GB/s"
                to += old[name]
            tn += new[name]
            print(row)
        print(f"  total per HC sublayer step: after {tn:.1f} us" + (f", before {to:.1f} us" if R else ""))
    if R and not a.no_check:
        names = ["xbar", "hres", "hpr", "out", "g_h", "g_w", "g_b", "g_y", "g_term"]
        print("\nnew vs reference (max abs err, max abs err / max|ref|)")
        for T in sorted(set(a.rows + [37, 5, 1024, 2001, 2048, 4096, 5000])):
            args = make_inputs(T, dtype)
            nw = one_step(K, *args)
            od = one_step(R, *args)
            line = []
            for nm, x, y in zip(names, nw, od):
                ma, mr = errs(x, y)
                line.append(f"{nm}:{'IDENT' if torch.equal(x, y) else f'{mr:.1e}'}")
            print(f"  T={T:5d} " + " ".join(line))


if __name__ == "__main__":
    main()
