# Agent Note: Perf Pass Phase3 Plan

Status: proposed

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Phase3-Plan.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Phase 3 Plan (agreed 2026-07-03, pre-compaction)

Two focuses, in this order. Constraints unchanged: bit-exact hard requirement
(gate = unit bitwise + exact kernel census + B-vs-B′ noise-floor traces), ckpt
interop soft. context: the BCSR/megablocks kernels DID show real improvements
previously (Gate G1, 3.09× MLP-isolated) — the in-model regression is an integration
problem to fix properly, not a reason to distrust the backend.

## Focus 1 — Post-carve memory (+1.5–2GB; blocks everything else)

Peak 15.0–17.4GB dense → 16.2–19.3GB routed; reserved 18.55 → 20.06GB. Memory is
why ckpt_grad_iters (documented-exact speed knob) has no headroom.

1. **Allocator snapshots**: MORPH_MEM_PROBE=1 + MORPH_MEM_SNAPSHOT_STEP on
 (a) dense step ~25 fresh run, (b) routed step ~36020 via
 `training.resume=checkpoints/morph/tst_stp_on_50k/step_36000.pt
 training.resume_fresh_optimizer=true`. Diff allocation stacks → attribute BOTH
 the +2GB routed delta AND the 15GB dense base (huge for 276M @ mb4/seq4k —
 suspect eager-attention intermediate retention in the 6 non-checkpointed
 prelude/coda blocks + HC 4× carrier).
 CAVEAT from train.py comment: allocator history hooks can NULL the fused HC
 autograd kernels — snapshot mode may need kernels eagerness care; peak-only
 first, snapshot second.
2. **Suspects to confirm/kill** (from Phase 0/1 reading):
 - Prelude/coda routers live OUTSIDE checkpointing → each call retains q_proj
 input fp32 [16384,768] ≈ 50MB + LayerNorm saves + subkey scores (×6).
 - Routed-MLP gated-hidden intermediates ([B,T,d_ff] retained per
 un-checkpointed MLP; equal-cluster reshape path already avoids the worst).
 - `_forward_mortar` `.to(bf16)` graph copies of padded x2 per call.
 - stk dds custom-op saved tensors (backward saves inputs — duplicates?).
 - Router param fp32 + optimizer state; rebuild-at-carve fragmentation
 (reserved delta > alloc delta).
3. **Fix and verify**: peak_alloc routed ≤ dense at same shapes; gate each fix
 (census + traces). Then **re-measure ckpt_grad_iters headroom** (e.g. 4→2
 checkpointed grad iters ⇒ 2 core-iteration recomputes saved ≈ 2×~35ms AND
 ~24 router+attn recompute calls dropped). MUST verify the "exact, never
 changes gradients" claim under dropout: checkpoint(use_reentrant=False)
 preserves RNG per docs, but changing WHICH iters checkpoint may shift global
 RNG consumption — census+trace gate decides; if RNG shifts, it's a
 config-level knob (still allowed per-run), not a silent default change.

## Focus 2 — BCSR carved-path speed, fixed properly

In-model: carve nets −14ms (removed 62ms dense mm, added 48ms BCSR) and router
adds +60–80ms ⇒ net +55ms/step. G1's 3.09× was real; recover it in-model.

1. **ncu the stk kernels** at real shapes ([16384,768]×[768,4096] @ 0.25 BCSR,
 fwd dds + bwd dsd/sdd — sdd_at_b measured 294µs×60/step): SOL, occupancy,
 smem/LSU pressure. Stock upstream config (BLOCK 128×128×32, stages=4,
 warps=4, ONE autotune config) vs house SM120 style (stages=1, warps=8, "no
 TMA pipeline"). stages/warps sweep = bit-exact (K-loop order unchanged).
 BLOCK_K change = class B (reduction tree) — measure, surface with parity
 evidence + tile-prover schedule proofs, opts in.
2. **Re-attribute the 61ms/step aten::topk** BEFORE touching the router: split
 CSA-indexer topk ([B,S,n_blocks]→top128, big) vs router topk ([N,16]→top8,
 tiny ×~76 calls). At n_clusters=16 with n_sub_keys=4 the product-space path
 is the DIRECT branch (16==16, no extra topk). Fix what's actually hot.
3. **Router bit-exact trims**: hoist the per-call `x_flat.to(proj_dtype)` fp32
 upcast where loop-invariant; cache iter_embed rows; the no_grad-iteration
 router calls (~6×(T−4)/step) compute gates discarded by truncated BPTT —
 verify they're needed for the forward value only (they are — gates shape the
 frozen iters' outputs) so no free skip there; recompute reduction comes from
 ckpt_grad_iters (Focus 1). Class B (opt-in): bf16 router. Ablation (parked):
 share gates across loop iterations / route per-iteration-group (CLA-adjacent).
4. **Carved-MLP compile revisit** ONLY if ncu shows elementwise glue dominating
 around the sparse GEMM (prior measurement: carved-compiled −6% vs eager due
 to grad_mode guard thrashing; MORPH_COMPILE_CARVED gate exists).
5. **Success metric**: carved+routed ≤ dense ms/step at d768, no memory regression.
 Cloud lens: at d2048 the MLP share grows → BCSR win grows, router share
 shrinks; kernel tuning transfers (RTX Pro 6000 = GB202/SM120 like the 5090).

## Standing context for resumed sessions

- Bench regimes: fresh run = dense (prune_start 3000 never hit at short steps);
 routed = resume tst_stp_on_50k/step_36000.pt + resume_fresh_optimizer=true.
- Always `training.optimizer=ademamix_b1zero` (deploy truth).
- Harness: MORPH_PERF_REGIONS / MORPH_PROF_WINDOW=a:b:prefix (kineto; nsys
 fork-deadlocks) / MORPH_EXACT_TRACE / MORPH_MEM_PROBE(+SNAPSHOT_STEP).
- Runs are now seeded (3a5076d) — matched-RNG comparisons valid; pre-3a5076d
 numbers have ±40ms Poisson-depth noise across runs.
- Deferred queue after these two focuses: async ckpt save, launch-count strategy
 (CUDA-graph the fixed-shape optimizer step; command-buffer wall = 241ms/step),
 class-B menu (GradScaler removal, Liger CE, bf16 router), CLA re-ablation.
