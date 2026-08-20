# Agent Note: Perf Pass Focus2 Bcsr

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Focus2-BCSR-Findings.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Focus 2 — BCSR carved-path speed (measured 2026-07-03)

Artifacts: `ignore/perf/stk_microbench.py` (faithful deploy-shape bench + bitwise
gate), `ignore/perf/sdd_split.py` (split-tile prototype + parity), traces
`ignore/perf/trace_{baseline,baseline2,split}.txt`.
Landed change: `morph/sparse/stk/backend/triton_kernels.py` (output-tiled SDD).

## Headline

The stk BCSR kernels are **NOT mistuned on stages/warps** — that premise is
FALSE (measured, register-bound root cause below). The one real class-A win is a
new **output-tiled SDD** kernel that fixes a severe CTA-occupancy starvation:
**2.17× on the hottest kernel (dW backward), bitwise-identical, landed by default**.
In-model routed step: **735.6 → 726.4 ms (−9.2 ms/step)**, bwd GPU 385.2 → 375.4 ms.

## Deliverable 1 — kernel SOL / occupancy

ncu is HARD-BLOCKED here: `ERR_NVGPUCTRPERM` (no GPU perf-counter permission, no
passwordless sudo). I did NOT fabricate SOL numbers. Substituted with Triton
compiled-kernel metadata (n_regs/spills/smem) + the CUDA occupancy model — which
gives the occupancy story ncu would have, without counters. 5090 = 170 SMs,
1536 thr/SM, 65536 regs/SM, 102400 B smem/SM.

Deploy shape: carved linear in=768 out=4096, block 128, density 0.25, nnz=48
blocks. M = 4·4096 = 16384. dtype bf16, fp32 accumulate.

Deploy carved path uses only TWO kernels (confirmed — `custom_ops._dds_backward`
calls `stk_dds`+`stk_sdd_at_b`; **`_dsd_kernel` is DEAD in deploy**, legacy
autograd path only):
- `_dds_kernel`: forward (x@Wᵀ) AND dx-backward (dy@W).
- `_sdd_kernel`: dW-backward (dyᵀ@x on topology).

| kernel | stock µs/call | regs | smem | CTAs/SM (reg-cap) | grid CTAs | waves |
|---|---|---|---|---|---|---|
| dds fwd  | 141.9 | 212 | 49152 | 2 | 4096 | ~12 (fine) |
| dds dx   | 180.6 | 212 | 49152 | 2 |  768 | ~2 (fine) |
| **sdd dW** | **445.7** | 217 | 49152 | 2 | **48** | **0.28 (STARVED)** |

Root cause: BOTH kernels are **register-bound at 2 CTAs/SM** (212–217 regs/thread
→ 65536/(212·128) = 2). Occupancy caps at 17% (w4) / 33% (w8) regardless of
stages. **sdd launches only 48 CTAs** — the register-limited machine can hold
2·170 = 340 concurrent, so sdd fills <15% of even that, and 72% of SMs get zero
work. sdd is not compute- or config-bound; it is **CTA-count-starved**.

## Deliverable 2 — the 61 ms/step topk, re-attributed

Microbench at real config (base.yaml: CSA compression m=2 ⇒ n_blocks = S/m =
2048, top_k=128; router n_clusters=16 ⇒ n_products=16=n_tile_groups ⇒ DIRECT
branch, routing.py:193 topk NEVER fires; only routing.py:207 activation_k topk):

1. **CSA indexer topk** `[4,4096,2048].topk(128)` fp32 = **741 µs/call**. ~82
   calls/step in routed (6–8 attn blocks × ~10–13 loop-iter forward passes incl.
   ckpt recompute) ⇒ 82·0.741 ≈ **61 ms/step. THIS is the topk.**
2. **Router activation_k topk** `[16384,16].topk(≈2–8)` = **25 µs/call**. Even at
   113 calls/step ⇒ ~2.8 ms/step. **Negligible.**

Conclusion: the 61 ms topk is the **CSA attention indexer, present in dense AND
routed** — it is NOT a routing regression. The router's post-carve cost is its
fp32 query_proj GEMM + LN + aux (Focus-1: ~650 MB fp32, +60–80 ms), NOT topk.
The CSA topk is the single biggest topk lever but is shared with dense and is
tie-break-sensitive → **class B** (see menu).

## Deliverable 3 — class-A stages/warps sweep: NULL result

Bitwise gate (`stk_microbench.py bitwise`): all of {w2s1,w4s1,w4s2,w4s4,w8s1,
w8s2,w8s4,w16s1} produce `torch.equal=True` on all three kernels (K-loop order
unchanged, as predicted). So the sweep is legitimately class A — but it buys
nothing:

| kernel | stock w4s4 | best config | speedup |
|---|---|---|---|
| dds fwd | 141.9 µs | w4s4 (stock) | 1.00× |
| dds dx  | 180.6 µs | w8s2 177.9 | 1.02× (noise) |
| sdd     | 445.7 µs | w8s4 434.1 | 1.03× (noise) |

The house "SM120 ⇒ stages=1/warps=8" prior does NOT transfer — it is actively
WORSE for dds_fwd (w8s1 = 177.6 vs 141.9). Reason: occupancy is REGISTER-bound,
not smem/pipeline-bound, so freeing smem (s1) or adding warps can't raise the
2-CTA/SM ceiling. **No stages/warps change landed — none is warranted.**

## Deliverable 3 (the actual win) — output-tiled SDD, LANDED

Since sdd is CTA-starved, the fix is MORE CTAs. Splitting each 128×128 output
block into SPLIT_M×SPLIT_N sub-tiles (each an independent CTA) leaves every
output element's K-reduction **byte-identical** (same BLOCK_K, same tl.dot, same
order) — only the output PARTITION changes. This is class A (no reduction-tree
change) and empirically bitwise.

`_sdd_split_factors(nnz, 128, 170)` adaptively targets ~2–4 CTAs/SM: nnz=48
(gate_up) → (4,2) = 384 CTAs; nnz=24 (down) → (4,4) = 384 CTAs. Sub-tiles stay
≥32 (tensor-core min). Falls back to stock when nnz already fills the machine.

Evidence:
- **Isolated kernel**: `torch.equal=True`, max|Δ|=0 across all split factors
  (m2n1,m2n2,m4n2,m4n4,m2n4). Best m4n2 = 197 µs vs stock 427 µs = **2.17×**.
- **Through the real custom op** `stk_sdd_at_b`: `torch.equal=True`, max|Δ|=0;
  409 → 189 µs = **2.17×**.
- **In-model routed** (resume step_36000, ademamix_b1zero, seed 1234, split
  OFF vs ON): step 36040 bwd GPU 385.2 → 375.4 ms (−9.8 ms), step
  735.6 → 726.4 ms (**−9.2 ms/step**). Consistent with 24 mortar dW calls ×
  ~220 µs saved. Both mortar shapes (gate_up nnz48, down nnz24) exercised
  end-to-end without error.
- **Bit-exact gate PASS** (two independent lines):
  1. Direct: torch.equal / max|Δ|=0 above — byte-identical output. Definitive.
  2. Loss-trace vs B-vs-B′ noise floor (same-config split-OFF run twice):
     floor mean_rel 3.68e-4 / max 2.03e-3 (diverges step 36002 via pre-existing
     embedding/cuBLAS atomics); change mean_rel 3.20–4.23e-4 / max 2.62–4.65e-3,
     SAME step, SAME mechanism. The split kernel uses NO atomics → adds zero
     nondeterminism; divergence is entirely the pre-existing pipeline chaos
     present identically in both arms.

Landed: `MORPH_SDD_SPLIT` env (default ON) killswitch; split path is the default.

## Deliverable 4 — class-B candidates (measure only, Wolfe opt-in)

- **BLOCK_K on dds**: BK=64/128 are SLOWER (188 µs vs 138 µs at BK=32) —
  smem/register pressure grows, no reduction-tree win. (Surprisingly bitwise at
  these shapes, but pointless — do NOT pursue.)
- **bf16 router** (from Focus-1): halves ~650 MB fp32 intermediates AND speeds
  the fp32 query_proj GEMM (the real post-carve cost). Biggest single lever;
  changes routing numerics.
- **CSA indexer topk** (741 µs/call, ~61 ms/step, dense+routed): a
  threshold/bucket top-128-of-2048 could cut this, but changes torch.topk
  tie-break selection → numerics + train/decode-consistency risk. High value,
  needs a careful parity + KV-cache-consistency study.

## Deliverable 5 — success metric

"carved+routed ms/step ≤ dense": **NOT met by class-A alone, and it can't be** —
the dominant post-carve regression is the ReMoE router (fp32 GEMM + topk-free
gate, +60–80 ms), which is class B (bf16). The kernel side is now
fully-characterized and optimal:
- sdd dW: fixed (2.17×, −9.2 ms/step, bitwise, landed).
- dds fwd/dx: already optimal at stock (register-bound; stages/warps/BLOCK_K all
  confirmed null-or-worse).
The remaining gap is a router (class-B) problem, not a BCSR-kernel problem.
