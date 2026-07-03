#!/usr/bin/env python
"""Mortar GEMV sweep + correctness check for the int8 BI=256 win.
Mortar is ALU/unpack-bound (per its own header), so we sweep num_warps/num_stages
and BO; achieved-BW is a lower bound on its efficiency (it does real unpack work)."""
import statistics, torch, triton
from morph.kernels.triton.fused_decode_step import (
    _mortar_gemv_kernel, _MORTAR_CB, int8_gemv, _i8row_gemv_kernel,
)

dev = torch.device("cuda")
PEAK = 1790e9; NSM = 170
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


def build_mortar_pack(O, I, density=0.25, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    R, C = O // BLK, I // BLK
    col_idx, offs = [], [0]
    for r in range(R):
        k = int(round(density * C * (0.5 + torch.rand(1, generator=g).item())))
        k = max(1, min(C, k))
        cols = torch.randperm(C, generator=g)[:k].sort().values
        col_idx.extend(cols.tolist()); offs.append(offs[-1] + k)
    nnz = offs[-1]
    strips = torch.randint(0, 255, (nnz, BLK, BLK // 4), device=dev, dtype=torch.uint8)
    col_idx_t = torch.tensor(col_idx, dtype=torch.int32, device=dev)
    offsets = torch.tensor(offs, dtype=torch.int32, device=dev)
    CB = _MORTAR_CB
    rws, j0s, slots, cnts = [], [], [], []
    nspl_max = 0
    for r in range(R):
        lo, hi = offs[r], offs[r + 1]
        nch = -(-(hi - lo) // CB); nspl_max = max(nspl_max, nch); cnts.append(nch)
        for s in range(nch):
            rws.append(r); j0s.append(lo + s * CB); slots.append(s)
    ent = torch.stack([torch.tensor(rws, dtype=torch.int32),
                       torch.tensor(j0s, dtype=torch.int32),
                       torch.tensor(slots, dtype=torch.int32)]).to(dev).contiguous()
    return (strips, col_idx_t, offsets, ent, nspl_max,
            torch.tensor(cnts, dtype=torch.int32, device=dev), nnz)


print("── mortar_gemv gate_up [43648,8192] (shipped BO=32 w=4 s=3 → 29%) ──")
O, I = 43648, D
strips, col_idx, offsets, ent, nspl_max, cnts, nnz = build_mortar_pack(O, I, seed=1)
x = torch.randn(1, I, device=dev, dtype=torch.float32)
rd = nnz * BLK * (BLK // 4)
E = ent.shape[1]
for BO in (16, 32, 64):
    for nw in (2, 4, 8):
        for ns in (2, 3, 4):
            if BLK % BO:
                continue
            grid = E * (BLK // BO)
            part = torch.empty(nspl_max, 1, O, device=dev, dtype=torch.float32)
            def launch(BO=BO, nw=nw, ns=ns, grid=grid, part=part):
                _mortar_gemv_kernel[(grid, 1)](
                    x, strips, col_idx, offsets, ent[0], ent[1], ent[2], part,
                    offsets, offsets, BLK=BLK, BO=BO, CB=_MORTAR_CB, OTOT=O, NB=1,
                    SWIGLU=False, FF=0, HAS_RACT=False, HAS_CACT=False,
                    sx_b=x.stride(0), ra_b=0, ca_b=0, num_stages=ns, num_warps=nw)
            try:
                us = cold_time(launch)
            except Exception:
                continue
            bw = rd / (us * 1e-6)
            print(f"  BO={BO} w={nw} s={ns}  {us:7.2f}us {bw/1e9:7.1f} GB/s "
                  f"{100*bw/PEAK:5.1f}% grid={grid} ({grid/NSM:.1f}w)")
print()

# ── correctness: int8_gemv BI=256 vs BI=128 must be bit-equal (same reduction) ─
print("── int8_gemv correctness: BI=256 (win) vs default BI=128 ──")
O, I = D, D
x = torch.randn(1, I, device=dev, dtype=torch.float32)
codes = torch.randint(-127, 127, (O, I), device=dev, dtype=torch.int8)
sc = torch.randn(O, device=dev, dtype=torch.float32)
o128 = torch.empty(1, O, device=dev, dtype=torch.float32)
o256 = torch.empty(1, O, device=dev, dtype=torch.float32)
_i8row_gemv_kernel[(O // 16, 1)](x, codes, sc, o128, I=I, O=O, BI=128, BO=16,
    PACK4=False, sx_b=x.stride(0), so_b=o128.stride(0), num_stages=4, num_warps=4)
_i8row_gemv_kernel[(O // 32, 1)](x, codes, sc, o256, I=I, O=O, BI=256, BO=32,
    PACK4=False, sx_b=x.stride(0), so_b=o256.stride(0), num_stages=4, num_warps=8)
torch.cuda.synchronize()
ref = (codes.float() * x).sum(1) * sc   # fp32 reference
d128 = (o128.squeeze() - ref).abs().max().item()
d256 = (o256.squeeze() - ref).abs().max().item()
dcross = (o256.squeeze() - o128.squeeze()).abs().max().item()
rel = dcross / (ref.abs().max().item() + 1e-9)
print(f"  max|BI128 - ref_fp32| = {d128:.3e}")
print(f"  max|BI256 - ref_fp32| = {d256:.3e}")
print(f"  max|BI256 - BI128|    = {dcross:.3e}  (rel {rel:.2e})")
print("  NOTE: BI changes the K-tile width → reduction grouping differs → fp")
print("        non-associativity gives a small delta; quantized stack is tol-gated.")
print("MORTAR_SWEEP_DONE")
