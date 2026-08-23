# Agent Note: Perf Pass Phase1

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Phase1-Findings.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Phase 1: Measured Findings (2026-07-03)

Harness: `MORPH_PERF_REGIONS=1` (CUDA-event region timing in train.py, default off) +
`MORPH_PROF_WINDOW=a:b:prefix` (in-process kineto — nsys is UNUSABLE here: its injected
threads hold the Triton launcher pipe's write end across the warmup gcc fork → the
fork-deadlock, reproduced twice, main thread parked in anon_pipe_read 40 min).
Workload: base.yaml + `training.optimizer=ademamix_b1zero` (deploy truth), 5090.
Traces: `ignore/perf/r1_dense.*`, `ignore/perf/r4_routed.*` (10-step kineto windows).

## Region timing (wall ms/step, steady state)

| region | R1 dense (bag0) | R1 dense (bag6) | R4 carved+routed |
|---|---|---|---|
| data (CPU-blocking) | 35 | **130–163** | 35 |
| fwd | 260 (GPU 325) | ~286 | 275 (GPU 338) |
| bwd | 430 | 414 | 485–520 |
| prune bookkeep | **47 (GPU 6)** | 47 | 0.6 |
| clip | 2.7 | 3.7 | 2.7 |
| opt (b1zero fused) | 9.5 (GPU 44) | — | 48 (GPU 45) |
| **step total** | **~790 (1.27 sps)** | ~947 | **~845 (1.18 sps)** |

## Structural diagnosis: launch-count-bound

Per step (R1): **~22,000 kernel launches** (17.3k cudaLaunchKernel + 4.7k Triton
cuLaunchKernel*), ~250 ms/step of CPU launch work, **241 ms/step CPU blocked on
"Command Buffer Full"**, and **30 cudaStreamSynchronize/step ≈ 100 ms CPU sync-wait**
(24 = apply_prune_mask's `bool(mask.all)`, +CE n_valid, +loop control, +scaler).
GPU time split: ~48% is memory-bound micro-kernel soup (copy_ 90ms — 5k calls/step;
mul 48; add/add_ 78; misc elementwise ~150; `aten::to` casts 3.8k calls/step = 64ms).
aten::mm = 2,368 calls/step averaging 78 µs (small-GEMM storm: CCA projections ×
loop iterations × backward). Fused-kernel GPU shares: HC 77ms, GLA 40ms (bwd
3.7ms/call!), CCA prologue 32ms, window+CSA ~28ms, CE cutlass ~40ms.

## Post-carve regression (the report, now quantified)

**Speed:** R4 is NET SLOWER than dense (+55ms/step). Decomposition:
- dense-MLP mm removed: −62 ms; BCSR replacements (_dds 21 + sdd 18 + bwd glue 9): +48
 → **the carve itself nets only ≈ −14 ms** at d768 (Gate G1's 3.09× was MLP-isolated;
 MLPs are a small share of in-model step time).
- ReMoE router: +60–80 ms — aten::topk 26→61 ms/step (113 calls/step: 2 topks × every
 MLP call INCLUDING the 24 checkpoint-recompute calls), fp32 query GEMM [N,768]×[768,768]
 per call, LayerNorm, gate normalize, aux var, +2k launches.
- carved+routed MLPs run EAGER (documented compile guard-thrash decision) where dense
 MLPs were compiled → extra elementwise/copy launches.

**Memory:** peak alloc 15.0–17.4 → 16.2–19.3 GB; reserved 18.55 → 20.06 GB.
Weights got 4× smaller; activations grew. Suspects (unverified, snapshot next):
router activation retention in prelude/coda (outside checkpointing: q_proj input fp32
50MB + LN + subkey scores per call, ×6), gates path intermediates ([B,T,d_ff] gated
hidden retained per un-checkpointed MLP), `.to(bf16)` graph copies in _forward_mortar,
router param optimizer state. Attribution tool ready: MORPH_MEM_SNAPSHOT_STEP.

Also noted: 15–17 GB for a 276M model at mb4/seq4k is dominated by eager-attention
intermediate retention in the 6 non-checkpointed prelude/coda blocks + HC 4× carrier.
This memory is WHY ckpt_grad_iters (the free exact speed knob) has no headroom —
memory work UNLOCKS speed work.

## Optimizer detail

b1zero fused = 45 ms/step GPU. The no-decay group (optim_bits=32) takes the fp32
_foreach fallback — and the `embed` keyword catches value_embed_tables + bigram +
hybrid (~150M params, >half the model) → fp32 m2/ν state (~1.2 GB) + ~30 ms of
foreach ops. Also `_fused_step` does `g.reshape(-1).float.contiguous` per param
per step (full-model bf16→fp32 grad copy). (Optimizer is settled; these are
implementation-cost notes, not a recommendation to change the algorithm.)

## Ranked Phase-2 plan (bit-exact class A first; ms × regime-weight)

1. **Sync removal batch** (~50–100 ms/step, ALL regimes): apply_prune_mask CPU-cached
 all-alive flag (kills 24 syncs/step); CE n_valid on-device; cache _find_cms_layers;
 stop constructing TopologyScorer per layer per step; skip accumulate_scores GPU work
 before it's needed? (NO — scores needed from step 0 for EMA; but the taylor-mode
 `self.weight` access re-runs the STE → reuse realized ternary from fwd? needs care).
2. **Data prefetch thread + pinned staging** (35 ms bag0 / 130–160 ms bag6 = the whole
 TST phase): bit-identical stream (Prefetcher pattern from data_placement).
3. **Post-carve memory root-cause** (MORPH_MEM_SNAPSHOT_STEP on r4) → reclaim the
 +2 GB, then re-evaluate **ckpt_grad_iters** headroom (config-documented exact;
 RNG-consumption caveat must be harness-verified) — the biggest single fwd/bwd lever
 available without touching numerics.
4. **Router cost (regime 4, 70% of run)**: bit-exact reductions only — hoist the topk
 threshold trick? cache invariant `.to(proj_dtype)` casts; batch the 2 topks; the
 fp32→bf16 router is class B (numerics). Big structural option (needs operator/ablation:
 route once per loop-iteration-group or share gates across iterations — CLA-adjacent,
 parked).
5. **stk BCSR kernels**: stock upstream config (stages 4 / warps 4, one autotune
 config) vs MORPH house style (stages 1 / warps 8 measured better on SM120).
 num_stages/num_warps sweep is bit-exact; ncu first.
6. **Async checkpoint save** (amortized ~1–2 ms/step; UX win at 2.5k cadence).
7. **GLA bwd kernel** (3.7 ms/call × 6 = 22 ms/step): ncu; schedule-only tuning class A.
8. **CUDA-graph the optimizer step** (~45 ms GPU + launch storm; fixed shapes every
 step): ambitious, bit-exact by construction if capture-safe. Investigate after 1–3.

Class B (surface with evidence, opt-in): GradScaler removal under bf16 (scale is
2^k → likely bit-exact absent overflow; harness will decide), Liger-style fused CE,
router in bf16, inductor fusion coverage of eager glue (elementwise-only fusion is
often bit-exact but reductions aren't — scope-by-scope).
