#!/usr/bin/env python
"""Cheap config sweep: vary BO / num_warps / num_stages on the under-peak GEMVs to
see if achieved cold-L2 bandwidth moves WITHOUT a new kernel. If a config bump
raises GB/s, that's the real cheap lever (vs a megakernel). Same shapes as
gemv_bw_30b.py."""
import statistics, torch, triton
from morph.kernels.triton.fused_decode_step import (
    _i8row_gemv_kernel, _ternary_gemv_kernel,
    _mortar_gemv_kernel, _mortar_combine_kernel, _MORTAR_CB,
)

dev = torch.device("cuda")
PEAK = 1790e9
NSM = 170
D, D_FF, BLK = 8192, 21824, 128
_FLUSH = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)


def cold_time(launch, iters=120, warmup=20):
    for _ in range(warmup):
        _FLUSH.zero_(); launch()
    torch.cuda.synchronize()
    out = []
    for _ in range(3):
        per = []
        for _ in range(iters):
            _FLUSH.zero_(); torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); launch(); e.record(); torch.cuda.synchronize()
            per.append(s.elapsed_time(e) * 1e3)
        out.append(statistics.median(per))
    return statistics.median(out)


def line(name, bytes_read, us, grid):
    bw = bytes_read / (us * 1e-6)
    print(f"  {name:<34} {us:7.2f}us {bw/1e9:7.1f} GB/s {100*bw/PEAK:5.1f}% "
          f"grid={grid} ({grid/NSM:.1f}w)")


# ── int8_gemv o_proj [8192,8192]: sweep BO (rows/CTA) and num_warps ──────────
print("── int8_gemv o_proj [8192,8192] (was BO=16,BI=128,warps=4,stages=4 → 54%) ──")
O, I = D, D
x = torch.randn(1, I, device=dev, dtype=torch.float32)
codes = torch.randint(-127, 127, (O, I), device=dev, dtype=torch.int8)
sc = torch.randn(O, device=dev, dtype=torch.float32)
out = torch.empty(1, O, device=dev, dtype=torch.float32)
rd = O * I
for BO in (8, 16, 32, 64):
    for BI in (128, 256):
        for nw in (2, 4, 8):
            for ns in (2, 4):
                if O % BO or I % BI:
                    continue
                grid = O // BO
                def launch(BO=BO, BI=BI, nw=nw, ns=ns, grid=grid):
                    _i8row_gemv_kernel[(grid, 1)](
                        x, codes, sc, out, I=I, O=O, BI=BI, BO=BO, PACK4=False,
                        sx_b=x.stride(0), so_b=out.stride(0),
                        num_stages=ns, num_warps=nw)
                try:
                    us = cold_time(launch)
                except Exception as ex:
                    continue
                line(f"BO={BO} BI={BI} w={nw} s={ns}", rd, us, grid)
print()

# ── int8_gemv k_proj [512,8192]: only 32 CTAs — try smaller BO + K-split ─────
print("── int8_gemv k_proj [512,8192] (was 6% — 32 CTAs, 0.2 waves) ──")
O, I = 512, D
x = torch.randn(1, I, device=dev, dtype=torch.float32)
codes = torch.randint(-127, 127, (O, I), device=dev, dtype=torch.int8)
sc = torch.randn(O, device=dev, dtype=torch.float32)
out = torch.empty(1, O, device=dev, dtype=torch.float32)
rd = O * I
for BO in (1, 2, 4, 8, 16):
    for BI in (128, 256, 512):
        for nw in (4, 8):
            if O % BO or I % BI:
                continue
            grid = O // BO
            def launch(BO=BO, BI=BI, nw=nw, grid=grid):
                _i8row_gemv_kernel[(grid, 1)](
                    x, codes, sc, out, I=I, O=O, BI=BI, BO=BO, PACK4=False,
                    sx_b=x.stride(0), so_b=out.stride(0),
                    num_stages=4, num_warps=nw)
            try:
                us = cold_time(launch)
            except Exception:
                continue
            line(f"BO={BO} BI={BI} w={nw}", rd, us, grid)
print()

# ── ternary_gemv down [8192,21824]: sweep BO (only 6 waves at BO=8) ──────────
print("── ternary_gemv down [8192,21824] (was BO=8 → 6 waves, 33%) ──")
O, I = D, D_FF
I4 = I // 4
x = torch.randn(1, I, device=dev, dtype=torch.float32)
codes = torch.randint(0, 255, (O, I4), device=dev, dtype=torch.uint8)
scale = torch.ones(1, device=dev, dtype=torch.float32)
out = torch.empty(1, 1, O, device=dev, dtype=torch.float32)
rd = O * I4
BI4 = min(triton.next_power_of_2(I4), 512)
for BO in (2, 4, 8, 16):
    for nw in (2, 4, 8):
        if O % BO:
            continue
        grid = triton.cdiv(O, BO)
        def launch(BO=BO, nw=nw, grid=grid):
            _ternary_gemv_kernel[(grid, 1)](
                x, codes, scale, x, out, I=I, O=O, BI4=BI4, BO=BO,
                BC=triton.next_power_of_2(I), X_MODE=0, eps=1e-6,
                num_stages=1, num_warps=nw)
        try:
            us = cold_time(launch)
        except Exception:
            continue
        line(f"BO={BO} w={nw}", rd, us, grid)
print("SWEEP_DONE")
