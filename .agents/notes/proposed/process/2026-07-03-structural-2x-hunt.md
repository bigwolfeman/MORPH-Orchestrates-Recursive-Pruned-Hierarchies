# Agent Note: Structural 2X Hunt

Status: proposed

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Structural-2x-Hunt.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Structural 2× Hunt (launch-count-bound) — 2026-07-03

Analysis + worktree-prototyping only (no GPU; single 5090/UPS → orchestrator serializes
GPU gates). Trust Phase 0/1/3 + Focus-1 measured numbers; this note builds on them.

Worktree: `.claude/worktrees/structural-2x-hunt` (branch `perf/structural-2x-hunt`,
NOT committed). Prototype patch: `morph/model/transformer.py` (injection-term hoist).

## The wall, re-measured from the captured traces (÷10 = per step)

Parsed `ignore/perf/r4_routed.cpu.txt` (routed, 70% of run) and `r1_dense.cpu.txt`:

| launch source | R4 routed /step | R1 dense /step |
|---|---|---|
| cudaLaunchKernel | **22,833** | 17,285 |
| cuLaunchKernel (Triton) | 2,404 | 3,067 |
| cuLaunchKernelEx | 1,996 | 1,681 |
| **total launches** | **~27,200** | ~22,000 |
| Command Buffer Full (CPU-blocked) | 1,625 events / **225 ms** | 1,591 / ~225 ms |
| cudaStreamSynchronize | **6** /step (82 ms wait) | **30** /step |
| aten::_to_copy (dtype casts) | 5,091 | 3,824 |
| aten::copy_ | 6,446 | 5,042 |
| aten::mul | 3,589 | 2,760 |
| aten::mm | 2,518 | 2,368 |
| aten::add / add_ | 1,821 / 1,458 | 1,449 / 1,178 |
| aten::sum | 1,264 | 998 |
| aten::empty_strided | 8,305 | 6,298 |
| aten::topk | 113 | 36 |

**Confirmations / corrections to Phase 1:**
- The "30 syncs/step" is the DENSE regime (prune-mask storm, already fixed at 3a5076d).
 **Routed r4 has only 6 syncs/step** — but each is ~13 ms because it drains a backed-up
 command buffer. So in the 70%-of-run regime, sync *removal* is NOT the lever; **launch
 *count* is** (it's what fills the command buffer → the 225 ms stall). This sharpens the
 mandate: cut launches, not syncs, for the dominant regime.
- Routing adds ~5,500 cudaLaunchKernel/step over dense (17.3k→22.8k) and +1,300 casts/step.
- `aten::_to_copy` = 5,091/step (~19% of all launches) is the single biggest *removable*
 category — dtype casts in the injection glue, HC, and the router fp32 upcast.

## Central structural verdict

**No single lever is a clean 2×.** The 2× is the SUM of stacked launch cuts + relieving
the 225 ms command-buffer stall. The mandate's marquee idea — CUDA-graph the core loop —
is **blocked locally** for a concrete, consistent reason (below), so the achievable local
path is a *stack* of launch reductions; the big graph wins are **cloud-target** (dual RTX
Pro 6000, 96 GB).

### Why the core-loop CUDA graph is LOCAL-BLOCKED (mandate item 1, definitive)

Three independent blockers, two fatal locally:
1. **Dynamic shape, two axes.** Per-iteration core shape is `[n_active, S, n, C]`.
 `n_active` shrinks *within* a step (active-set) AND changes *across* steps (Poisson
 depth resample at `transformer.py:689`). CUDA graphs need static shape + static
 addresses. Fix = pad every iteration to fixed `[B,S,n,C]` (process frozen samples too,
 discard them). That IS bit-exact (no cross-sample mixing; frozen suffix carried by the
 existing `cat`), it just re-pays the ~10–15 % FLOPs active-set shrinking saved — fine
 when launch-bound. **Not the fatal one.**
2. **Checkpoint ⇄ graph, FATAL locally.** A graphed core iteration retains its activations
 in the graph's static pool → you cannot checkpoint a replay. Un-checkpointing the core
 is exactly what Focus-1 measured: **+7 GB per eager iter → OOM at 24.66 GB** on the
 5090 (`transformer.py:281` comment names it). The ~10.5 GB routed headroom evaporates
 after <2 eager iters. **This is the same wall that blocks `ckpt_grad_iters`.**
3. **Dropout RNG under capture** (dropout=0.1 on): graph replay advances philox
 deterministically but the stream differs from eager unless the graph-safe generator is
 used → a bit-exactness gate item, not a blocker.

**Verdict:** core-loop graph = **cloud-only** (96 GB fits the un-checkpointed +7 GB/iter),
fixed-B padded, bit-exact pending the RNG-under-capture noise-floor gate. Big launch win
*there*; do NOT pursue locally. This is consistent with Focus-1's `ckpt_grad_iters`
finding — same memory wall, same cloud unlock.

### What a CUDA graph CAN capture locally

The **static-shape regions that bracket the variable-depth core**: embed → prelude(3
blocks) → [eager core loop] → coda(3 blocks) → lm_mixer → final_norm → fused CE. Prelude
and coda run at full batch `[B,S,n,C]` every step, un-checkpointed already (Focus-1), fixed
shape. Two graphs (pre-core, post-core) with the eager core between = ~100–150 block
launches + embed + CE captured, static I/O buffers, graph-safe dropout. Class A pending RNG
gate. Medium-high effort (static buffer plumbing). This is the *right* CUDA-graph target
locally, not the core.

### The optimizer step (mandate item 1c) — best-confidence graph target

`ademamix_b1zero._fused_step` has a **per-param Python loop** (`ademamix_b1zero.py:310`):
per fused param it does `g.reshape(-1).float.contiguous` (1 copy/cast kernel) + the
Triton step kernel (1 launch). Post-carve the param set is static → ~200–400 fused params
× 2 = **~400–800 launches/step** + Python-loop CPU overhead, all fixed-shape. Prime graph
target. Blockers to resolve (all tractable):
- `train.py` uses `optimizer.zero_grad(set_to_none=True)` (lines 1346/1634/1710) → grads
 get **fresh addresses each step** → graph replay reads stale grad pointers. **Must switch
 to `set_to_none=False`** (static `.grad` buffers) to graph the optimizer. Small change,
 bit-exact (zeroed grads either way).
- `_sched` scalars (`alpha_t, beta3_t, bc2, lr`) are Python floats baked into the capture
 → FROZEN at capture step, but they change during α/β3 warmup + LR schedule. Fix = pass
 as 0-dim GPU tensors the graph reads from memory (update values before each replay).
 `torch._foreach_*` don't take tensor-alpha cleanly → this is the real engineering cost.
 Mitigation: capture only AFTER warmup completes, where `alpha_t/beta3_t` are constant and
 `bc2`→~1 asymptotes (still needs lr-as-tensor for the LR schedule).
- `_mask_dead_state` has data-dependent control flow (`bool(dead.any)` early-return) →
 not capturable. Post-carve, confirm params lack `_dead_mask` (compacted) so it's a
 skippable no-op during capture — **GPU-verify this before capture.**

## Ranked queue (bit-exact-gated)

| # | win | mechanism | est. launch/ms Δ (per step) | class | patch status | GPU-validation |
|---|---|---|---|---|---|---|
| 1 | **Injection-term hoist** | per-core-layer additive term is loop-invariant; build 6 once instead of ~145×/step; drop ids/bg/x0 gathers | **~-696 launches** (~2.6%) + 3 gathers; removes a slice of the 5k _to_copy + 172 cat/step | **A*** (loss bitwise-exact CPU; grad Δ=1.6e-8 fp32 reassoc, ≪ 6e-4 noise floor, < bf16 eps) | **PROTOTYPED + CPU-verified** `transformer.py` | resume r4 step_36000, `MORPH_EXACT_TRACE`+seed=1234, ckpt=-1; assert loss-trace ≤ 6e-4 by step 11 vs `ckpt_base.trace`; region-time ms/step |
| 2 | **Optimizer CUDA graph** | capture static post-carve `_fused_step`; kill per-param Python-loop launches | ~-400 to -800 launches + Python/CB relief; GPU ~45 ms unchanged | A (same kernels/order/addr) | SPEC (needs set_to_none=False + tensor-scalar plumbing) | set_to_none=False alone first (bit-exact, measure ms); then graph, loss-trace noise-floor gate |
| 3 | **Attention small-GEMM batching** | `aten::mm` 2,518/step @48µs = ~121 ms/step small per-proj/per-head GEMMs across ~24 core-execs; group into bmm/grouped-GEMM | ~-2,000 launches + wall-time (better GEMM SOL) — the single biggest mm lever | A if same math/accumulation; B if reduction tree changes | SPEC (attention.py; high effort) | ncu the CCA/CSA proj GEMMs at real shapes; parity trace; tile-prover for schedule if BLOCK changes |
| 4 | **Static-region CUDA graphs** | graph embed+prelude and coda+head+CE (fixed shape); eager core between | ~-200 to -400 launches + CB stall relief on those regions | A pending dropout-RNG-under-capture gate | SPEC (static I/O buffers) | capture on warm step; loss-trace noise-floor gate (RNG!) |
| 5 | **Router elementwise-tail fusion** | fuse relu/threshold-topk/sum/div-normalize/aux-var per router call (113 calls/step); drop invariant casts | ~-300 to -500 launches (r4 only) | A for the fusion; **B** for bf16 router (Focus-1: numerics + ~0.65 GB) | SPEC (routing.py Triton) | ncu router forward; A-fusion parity trace; bf16 → opt-in |
| 6 | **RoPE cos/sin cast cache** | `attention.py:151/429` casts fixed cos/sin buffers to bf16 every call (~120 casts/step) | ~-120 launches | A (cache per dtype) | SPEC (attention.py) | parity trace |

`*` Class-A caveat for #1: loss is **bitwise** identical; the fp32 param-grad reassociation
is 1.6e-8 (a mathematically-identical sum accumulated in a different order — same class as
the already-landed x0-hoist). In the bf16 training carrier this is below bf16 epsilon. Gate
on GPU with the loss-trace noise-floor test before landing; it will pass ≤ 6e-4 by step 11.

## Prototype #1 — injection-term hoist (DONE, verified)

**File:** `.claude/worktrees/structural-2x-hunt/morph/model/transformer.py`

**Mechanism.** `_build_injection_term(np_+i, x0_core_terms[i], input_ids, bigram_emb,
dtype)` for core layer `i` depends on nothing iteration-varying: `x0_core_terms` is already
hoisted (loop-invariant), `input_ids`/`bigram_emb`/`dtype` are constant across the loop. The
old code rebuilt this term inside `_apply_core_step` **every iteration** (~145 rebuilds/step
incl. checkpoint recompute; each ~5 cast/mul/cat kernels). Now: precompute the 6 distinct
terms once (`inj_core_terms`), sort into active-set order (`inj_s`), pass the active slice as
a **checkpoint input** so the backward recompute reuses it (no rebuild in backward either).
Also removes the `ids_s`/`bg_s`/`x0_s` per-step gathers — the core no longer needs them.

**Bit-exactness argument.** Same inputs ⇒ same term value. The single shared term added into
each iteration's carrier accumulates the identical sum-over-iterations gradient to
proj/value-embed/bigram-λ as the per-iteration rebuild (autograd sums the multi-use grad) —
exactly the argument the existing x0-hoist already relies on.

**Verification (CPU, eager reference path, d=384 toy, no GPU):**
- Eval path (uniform depth, no active-set/dropout): **loss diff = 0.000e+00 (bitwise)**.
- Train path (Poisson depth, active-set shrinking, checkpoint recompute, backward):
 **loss diff = 0.000e+00 (bitwise)**; over 40 grad tensors (bigram, x0_injects, value_embed,
 core weights, lm_mixer) **max |Δgrad| = 1.583e-08** (fp32 reassociation, ≪ noise floor).
- Scripts: `scratchpad/parity_inj.py`, `scratchpad/parity_train.py`.

**GPU gate before landing:** resume `checkpoints/morph/tst_stp_on_50k/step_36000.pt`
`training.resume_fresh_optimizer=true training.optimizer=ademamix_b1zero +training.seed=1234`,
`MORPH_EXACT_TRACE=<path>`, ckpt_grad_iters=-1; assert the loss trace stays ≤ 6e-4 by step 11
vs `ignore/perf/ckpt_base.trace`; measure ms/step with `MORPH_PERF_REGIONS=1`.

## Notes / honest edges (unverified)

- The ~696-launch estimate for #1 is derived from trace kernel counts (`_FusedHCPreMap`
 151/step ⇒ 145 core-block execs ⇒ 145 rebuilds) × ~5 kernels/rebuild. The exact kernel/
 rebuild count and the ms impact are **GPU-validation required** — I did not run the GPU.
- #2/#3/#4/#5 are **specs, not implemented.** Writing CUDA-graph capture or a fused Triton
 router tail blind (no GPU to validate) would be theater; they carry exact validation
 recipes instead. #3 (attention mm batching) is likely the biggest single remaining lever
 by both launch count and wall time — recommend it as the next deep GPU-gated work item.
- Local 2× realistically = #1 + #2(set_to_none first) + #3 stacked, plus relieving the
 225 ms command-buffer stall as launches drop. The clean 2×-in-one-shot (whole-step graph)
 is a **cloud** deliverable (memory), which dovetails with Focus-1's cloud `ckpt_grad_iters`
 unlock — both gated by the same +7 GB/iter local wall.
