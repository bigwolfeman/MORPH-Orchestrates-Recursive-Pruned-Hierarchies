#!/usr/bin/env python
"""Inter-kernel gap probe at 30B shapes. Build a representative per-core-block GEMV
chain (attn int8 projections + mortar MLP gate_up/down), run it back-to-back inside
ONE CUDA graph (the real engine path), and compare:
  - sum of individual cold kernel times (GPU-busy lower bound)
  - graph-replay wall time of the whole chain
If wall ≈ sum(busy), the inter-kernel gap is ~0 → a megakernel cannot recover bus
idle (the prior near-zero-gap claim holds). If wall >> sum(busy), there are bubbles.

We also run WITHOUT a graph (per-kernel python launches) to expose any launch gap a
megakernel WOULD remove — but the engine already uses a graph, so the graph number
is the deployed condition."""
import statistics, torch, triton
from morph.kernels.triton.fused_decode_step import int8_gemv, mortar_gemv, _MORTAR_CB

dev = torch.device("cuda")
D, D_FF, BLK = 8192, 21824, 128


def build_mortar_pack(O, I, density=0.25, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    R, C = O // BLK, I // BLK
    col_idx, offs = [], [0]
    for r in range(R):
        k = max(1, min(C, int(round(density * C * (0.5 + torch.rand(1, generator=g).item())))))
        cols = torch.randperm(C, generator=g)[:k].sort().values
        col_idx.extend(cols.tolist()); offs.append(offs[-1] + k)
    nnz = offs[-1]
    strips = torch.randint(0, 255, (nnz, BLK, BLK // 4), device=dev, dtype=torch.uint8)
    CB = _MORTAR_CB
    rws, j0s, slots, cnts = [], [], [], []; nspl_max = 0
    for r in range(R):
        lo, hi = offs[r], offs[r + 1]
        nch = -(-(hi - lo) // CB); nspl_max = max(nspl_max, nch); cnts.append(nch)
        for s in range(nch):
            rws.append(r); j0s.append(lo + s * CB); slots.append(s)
    ent = torch.stack([torch.tensor(rws, dtype=torch.int32),
                       torch.tensor(j0s, dtype=torch.int32),
                       torch.tensor(slots, dtype=torch.int32)]).to(dev).contiguous()
    return (strips, torch.tensor(col_idx, dtype=torch.int32, device=dev),
            torch.tensor(offs, dtype=torch.int32, device=dev),
            torch.ones(1, device=dev), O, ent, nspl_max,
            torch.tensor(cnts, dtype=torch.int32, device=dev))


# one core block's dominant GEMVs: qkv (int8), o_proj (int8), mlp gate_up + down (mortar)
qkv_w = torch.randint(-127, 127, (2 * D, D), device=dev, dtype=torch.int8)
qkv_sc = torch.randn(2 * D, device=dev, dtype=torch.float32)
o_w = torch.randint(-127, 127, (D, D), device=dev, dtype=torch.int8)
o_sc = torch.randn(D, device=dev, dtype=torch.float32)
gu_pack = build_mortar_pack(2 * D_FF, D, seed=1)
dn_pack = build_mortar_pack(D, D_FF, seed=2)

x = torch.randn(1, D, device=dev, dtype=torch.float32)


def one_block(x):
    qkv = int8_gemv(x, qkv_w, qkv_sc)                  # [1, 16384]
    h = x                                              # (attn math elided; BW proxy)
    o = int8_gemv(h, o_w, o_sc)                        # [1, 8192]
    gu = mortar_gemv(o, gu_pack, swiglu_out=True)      # [1, 21824]
    gu2 = torch.cat([gu, gu], dim=-1)                  # pack to [1, 2*F] for down swiglu_x
    dn = mortar_gemv(gu2, dn_pack, swiglu_x=True)      # [1, 8192]
    return dn


N_BLOCKS = 35 * 4  # n_core * mean_depth = full core-loop GEMV count

# ── individual cold-kernel busy time (sum) ──────────────────────────────────
_FLUSH = torch.empty(256 * 1024 * 1024 // 4, device=dev, dtype=torch.float32)


def cold_one(fn, iters=80):
    for _ in range(15):
        _FLUSH.zero_(); fn()
    torch.cuda.synchronize()
    per = []
    for _ in range(iters):
        _FLUSH.zero_(); torch.cuda.synchronize()
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        per.append(s.elapsed_time(e) * 1e3)
    return statistics.median(per)


busy = cold_one(lambda: one_block(x))
print(f"one_block cold busy (median, L2-flushed each call): {busy:.2f} us")
print(f"projected full core-loop ({N_BLOCKS} blocks): {busy * N_BLOCKS / 1000:.2f} ms")

# ── back-to-back WITHOUT graph: python-launch chain of N_BLOCKS, no flush ────
# (weights cycle; with distinct buffers we'd exceed L2, but here we reuse → upper
#  bound on launch efficiency. The GAP we expose = python launch + dependency.)
def chain_eager(n):
    xx = x
    for _ in range(n):
        xx = one_block(xx.view(1, -1)[:, :D].contiguous())
    return xx


for _ in range(5):
    chain_eager(4)
torch.cuda.synchronize()
s = torch.cuda.Event(True); e = torch.cuda.Event(True)
s.record(); chain_eager(N_BLOCKS); e.record(); torch.cuda.synchronize()
eager_ms = s.elapsed_time(e)
print(f"\neager chain {N_BLOCKS} blocks (HOT weights, python launch): {eager_ms:.2f} ms")
print(f"  per-block: {eager_ms*1000/N_BLOCKS:.2f} us  (HOT → << cold busy; isolates launch+dep gap)")

# ── back-to-back in ONE CUDA graph (the engine condition) ───────────────────
g = torch.cuda.CUDAGraph()
xx = x.clone()
# warmup on a side stream (required before capture)
sidestream = torch.cuda.Stream()
with torch.cuda.stream(sidestream):
    for _ in range(3):
        tmp = xx
        for _ in range(4):
            tmp = one_block(tmp.view(1, -1)[:, :D].contiguous())
torch.cuda.current_stream().wait_stream(sidestream)
try:
    with torch.cuda.graph(g):
        tmp = xx
        for _ in range(N_BLOCKS):
            tmp = one_block(tmp.view(1, -1)[:, :D].contiguous())
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(20):
        g.replay()
    e.record(); torch.cuda.synchronize()
    graph_ms = s.elapsed_time(e) / 20
    print(f"\nCUDA-graph replay {N_BLOCKS} blocks (HOT weights): {graph_ms:.2f} ms")
    print(f"  per-block: {graph_ms*1000/N_BLOCKS:.2f} us")
    print(f"  graph vs eager: {eager_ms/graph_ms:.2f}x  "
          f"(launch-gap removed by graph = {(eager_ms-graph_ms):.2f} ms of {eager_ms:.2f})")
except Exception as ex:
    print(f"\n[graph capture failed: {ex}]")
print("GAP_PROBE_DONE")
