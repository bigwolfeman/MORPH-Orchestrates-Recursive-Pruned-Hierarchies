#!/usr/bin/env python
"""30B-shape achieved-bandwidth probe for the dominant decode GEMVs.

Roofline measurement (ncu absent): achieved GB/s = weight_bytes_read / kernel_time,
reported as % of the 5090 HBM peak (1.79 TB/s). Shapes are the EXACT scale30b.yaml
per-call shapes (d_model=8192, n_kv=8, d_head=64, d_ff=21824, mortar density 0.25,
vocab 49152). Packed tensors are constructed in the kernels' native byte layout —
the kernel reads identical bytes whether codes come from a real carve or a synthetic
fill, so achieved BW is faithful (this is a memory-bound probe, values are random).

Each kernel ALONE, B=1, with grid/occupancy reported. We also sweep cheap config
knobs (BO / num_warps / num_stages) to see if achieved BW moves without a new kernel.
"""
import sys, math, statistics
import torch, triton
from morph.kernels.triton.fused_decode_step import (
    int8_gemv, ternary_gemv, mortar_gemv, _mortar_gemv_kernel, _mortar_combine_kernel,
    _MORTAR_CB,
)

dev = torch.device("cuda")
PEAK = 1790e9  # 1.79 TB/s HBM, RTX 5090
NSM = 170      # 5090 SM count (GB202, 170 SMs enabled on 5090)

# ── 30B per-call shapes ──────────────────────────────────────────────────────
D = 8192
N_KV = 8
D_HEAD = 64
LK = N_KV * D_HEAD          # 512  latent_k width
D_FF = 21824                # 8/3*8192 round64
VOCAB = 49152
DENSITY = 0.25
BLK = 128


# L2 flush buffer: 5090 L2 ~96MB → 256MB scratch guarantees full eviction.
_FLUSH = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)


def _flush_l2():
    _FLUSH.zero_()


def time_kernel(fn, iters=120, warmup=20, cold=True):
    """cold=True: flush L2 before EACH timed call so the weight stream is read
    from HBM (the real decode condition — no weight survives in 96MB L2 across the
    17.5GB/token stream). We time fn() ALONE via per-call events and subtract
    nothing (the flush is outside the timed region). Hot (cold=False) lets the
    weight sit in L2 across iters → over-reports BW (cache hits, not HBM)."""
    for _ in range(warmup):
        if cold:
            _flush_l2()
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(5):
        per = []
        for _ in range(iters):
            if cold:
                _flush_l2()
                torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            fn()
            e.record(); torch.cuda.synchronize()
            per.append(s.elapsed_time(e) * 1e3)  # us
        times.append(statistics.median(per))
    return statistics.median(times), (max(times) - min(times))


def report(name, bytes_read, us, jitter_us, grid_ctas, extra=""):
    bw = bytes_read / (us * 1e-6)
    waves = grid_ctas / NSM
    print(f"{name:<40} {us:7.2f}us (±{jitter_us:4.2f}) "
          f"{bw/1e9:7.1f} GB/s {100*bw/PEAK:5.1f}% peak | "
          f"grid={grid_ctas:6d} CTAs ={waves:5.1f} waves {extra}")
    return bw


print(f"# 30B GEMV achieved-bandwidth probe | peak {PEAK/1e12:.2f} TB/s, {NSM} SMs")
print(f"# d_model={D} d_ff={D_FF} latent_k={LK} vocab={VOCAB} density={DENSITY}\n")


# ── 1. int8_gemv  (attention projections, Int8RowLinear deploy) ──────────────
# The front_gemm handles q/k/v together, but int8_gemv is the simplest int8 attn
# proj. Representative attn proj: out-proj [D, D] = [8192, 8192] int8.
print("── int8_gemv (attn projection, int8 codes) ──")
for (O, I, tag) in [(D, D, "o_proj [8192,8192]"),
                    (LK, D, "k_proj [512,8192]"),
                    (2 * D, D, "qkv-ish [16384,8192]")]:
    x = torch.randn(1, I, device=dev, dtype=torch.float32)
    codes = torch.randint(-127, 127, (O, I), device=dev, dtype=torch.int8)
    sc = torch.randn(O, device=dev, dtype=torch.float32)
    rd = O * I  # int8 = 1 byte/weight (dominant traffic)
    BO, BI = 16, 128
    grid = (O // BO)  # B=1
    us, jit = time_kernel(lambda: int8_gemv(x, codes, sc))
    report(f"int8_gemv {tag}", rd, us, jit, grid)
print()


# ── 2. ternary_gemv (dense-arm MLP, 2-bit) ───────────────────────────────────
# gate_up [2*d_ff, d_model] = [43648, 8192] packed 2-bit (1/4 byte/weight).
print("── ternary_gemv (dense MLP arm, 2-bit packed) ──")
for (O, I, tag) in [(2 * D_FF, D, "gate_up [43648,8192]"),
                    (D, D_FF, "down [8192,21824]")]:
    x = torch.randn(1, I, device=dev, dtype=torch.float32)
    I4 = I // 4
    codes = torch.randint(0, 255, (O, I4), device=dev, dtype=torch.uint8)
    scale = torch.ones(1, device=dev, dtype=torch.float32)
    rd = O * I4  # 2-bit = I/4 bytes per row
    BO = 8
    grid = O // BO
    us, jit = time_kernel(lambda: ternary_gemv(x, codes, scale))
    report(f"ternary_gemv {tag}", rd, us, jit, grid)
print()


# ── 3. mortar_gemv (carved BCSR MLP, 2-bit, density 0.25) ────────────────────
# This is THE dominant MLP path. Build a realistic ragged BCSR at density 0.25.
print("── mortar_gemv (carved BCSR MLP, density 0.25) ──")


def build_mortar_pack(O, I, density=DENSITY, ragged=True, seed=0):
    """Synthesize the engine's strip-packed mortar format at real shapes.
    BCSR: R=O/BLK block-rows, C=I/BLK block-cols, ~density kept blocks, ragged
    per-row (matches the real carve: some rows keep more blocks than others)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    R, C = O // BLK, I // BLK
    rows_kept = []
    col_idx = []
    offs = [0]
    for r in range(R):
        if ragged:
            # ragged: vary per-row count around density*C, like the real carve
            k = int(round(density * C * (0.5 + (torch.rand(1, generator=g).item()))))
        else:
            k = int(round(density * C))
        k = max(1, min(C, k))
        cols = torch.randperm(C, generator=g)[:k].sort().values
        col_idx.extend(cols.tolist())
        offs.append(offs[-1] + k)
    nnz = offs[-1]
    strips = torch.randint(0, 255, (nnz, BLK, BLK // 4), device=dev, dtype=torch.uint8)
    col_idx_t = torch.tensor(col_idx, dtype=torch.int32, device=dev)
    offsets = torch.tensor(offs, dtype=torch.int32, device=dev)
    scale = torch.ones(1, device=dev, dtype=torch.float32)
    # work list (one entry per <=CB consecutive blocks of a row)
    CB = _MORTAR_CB
    rws, j0s, slots, cnts = [], [], [], []
    nspl_max = 0
    for r in range(R):
        lo, hi = offs[r], offs[r + 1]
        nch = -(-(hi - lo) // CB)
        nspl_max = max(nspl_max, nch)
        cnts.append(nch)
        for s in range(nch):
            rws.append(r); j0s.append(lo + s * CB); slots.append(s)
    ent = torch.stack([torch.tensor(rws, dtype=torch.int32),
                       torch.tensor(j0s, dtype=torch.int32),
                       torch.tensor(slots, dtype=torch.int32)]).to(dev).contiguous()
    cnts_t = torch.tensor(cnts, dtype=torch.int32, device=dev)
    pack = (strips, col_idx_t, offsets, scale, O, ent, nspl_max, cnts_t)
    nnz_bytes = nnz * BLK * (BLK // 4)  # uint8 codes actually read
    return pack, nnz_bytes, ent.shape[1]


# gate_up: [2*d_ff, d_model] = [43648, 8192]; must be /128
O_gu = (2 * D_FF // BLK) * BLK   # 43648 = 341*128 exact
I_gu = D
pack_gu, bytes_gu, E_gu = build_mortar_pack(O_gu, I_gu, seed=1)
x_gu = torch.randn(1, I_gu, device=dev, dtype=torch.float32)
BO = 32
grid_gu = E_gu * (BLK // BO)
us, jit = time_kernel(lambda: mortar_gemv(x_gu, pack_gu, swiglu_out=True))
report(f"mortar_gemv gate_up [{O_gu},{I_gu}]", bytes_gu, us, jit, grid_gu,
       extra=f"nnz_MB={bytes_gu/1e6:.1f}")

# down: [d_model, d_ff] = [8192, 21824] swiglu_x input
O_dn = D
I_dn = D_FF
pack_dn, bytes_dn, E_dn = build_mortar_pack(O_dn, I_dn, seed=2)
x_dn = torch.randn(1, 2 * I_dn, device=dev, dtype=torch.float32)  # packed gate_up
grid_dn = E_dn * (BLK // BO)
us, jit = time_kernel(lambda: mortar_gemv(x_dn, pack_dn, swiglu_x=True))
report(f"mortar_gemv down    [{O_dn},{I_dn}]", bytes_dn, us, jit, grid_dn,
       extra=f"nnz_MB={bytes_dn/1e6:.1f}")
print()

print("GEMV_BW_30B_DONE")
