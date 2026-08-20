# Agent Note: Optgraph Static Graphs

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/OptGraph-StaticGraphs.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# CUDA-graph launch cuts: optimizer step + static regions (2026-07-03/04)

Fable subagent session. Two class-A (bit-exact) captures, gated on the local 5090
against `checkpoints/morph/tst_stp_on_50k/step_36000.pt` (resume_fresh_optimizer,
ademamix_b1zero fused, seed 1234, steps 36045). NOT committed; flags default OFF.

## Gate baselines (this session)

- `ignore/perf/optgraph_base.trace` / `optgraph_base2.trace`: two identical flag-OFF
  runs → the REAL 45-step same-code noise floor. **The quoted 5.9e-4 floor is a
  step-11 number; atomics chaos-amplify it to 9.81e-3 by step 43.** Any 45-step gate
  must compare against the envelope, not the step-11 scalar.
- Census window: kineto steps 36020:36022 (2 steps), `optgraph_off.cpu.txt`.

## Task 1 — MORPH_OPT_CUDA_GRAPH (optimizer step graph)

**Code:** `morph/training/ademamix_b1zero_kernel.py` (LOAD_SCHED constexpr: the 7
step-varying scalars lr/β2/β3_t/α_t/ε/bc2/λ read from a persistent device fp32[7]
instead of baked launch args — bc2/α_t/β3_t change EVERY step), `ademamix_b1zero.py`
(`_graphed_fused_step`: warm 3 → capture group-0 fused path → replay; per-step
data_ptr signature check → auto invalidate/recapture; dead-mask via masked_fill_
with precapture masks; hybrid `zero_grad`), `train.py` (graph_invalidate on prune
events). Kill: `MORPH_OPT_CUDA_GRAPH` / `set_opt_cuda_graph()`.

**Gates:**
- Unit (`ignore/perf/gpu_probe_optgraph.py`): 12 steps, varying lr, dead-mask flip +
  invalidate/recapture → **params AND optimizer state torch.equal** (bitwise PASS).
- Loss trace: worst 7.31e-3 @36027 vs floor 9.81e-3; by-step-11 3.7e-4. **Bit-exact.**
- Census (2-step window, OFF→ON): cuLaunchKernelEx −456 (the 228 fused kernels →
  1 cudaGraphLaunch/step); aten::add_ +456 (AccumulateGrad tax on kept grad buffers);
  **TOTAL launch APIs +6 (net zero)**; **Command-Buffer-Full 3541→3244 (−148/step, −8.4%)**.
- Timing: opt GPU-region ~45.3→42-45ms (≤2ms, window-dependent); wall flat (±14 noise).

**Structural finding (honest):** an optimizer CUDA graph CANNOT net-drop launch
count here — baseline `set_to_none=True` grad handling is zero-kernel (steal path),
and stable-address requirements convert it into per-param accumulate adds. First
version (blanket set_to_none=False) was +1,030 launches/step — 4× worse than the
cut; the hybrid zero_grad (only the 228 captured params keep buffers, batched
foreach zero) brings it to net-zero. The real win is the launch PATTERN: the
end-of-step 228-launch burst becomes one graph launch → CBF −8.4%.

## Task 2 — MORPH_STATIC_GRAPHS (front/back region graphs)

**Code:** `morph/model/transformer.py` (`_front_region`/`_front_tail`/`_back_region`
extraction — flag-OFF is pure code motion; `_StaticRegion` wrapper; `build_static_graphs`
via torch.cuda.make_graphed_callables; `_drain_region_aux`; dispatch in
`_forward_single`; `static_graphs_invalidate`), `train.py` (one-time build hook at
start+3; invalidation at phase-boundary rebuild + curriculum set_context),
`ademamix_b1zero.py` (`_grad_via_graph_static` steal-path tag honored in zero_grad).
CE stays EAGER (fused_ce n_valid `.item()` host sync = capture blocker; its
python-float division is last-bit load-bearing). Kill: `MORPH_STATIC_GRAPHS` /
`set_static_graphs()`.

**Gates:**
- Unit (`ignore/perf/gpu_probe_static_graphs.py`, real 0.6M MORPHTransformer, dropout
  0.1, routers on every MLP, retention, checkpointed Poisson core, autocast+scaler+
  fused AdEMAMix, A-vs-A' floor exactly 0): **10 training steps bitwise (losses,
  all 292 grads, params) in BOTH arms — static-only AND static+opt-graph.**
- RNG semantics (`ignore/perf/gpu_probe_rng_graph.py`): graphed dropout regions
  advance the default generator EXACTLY like eager, interleaved with eager RNG
  consumers — 8 steps bitwise.
- Full model: capture works with fused Triton CSA/HCA, compiled-MLP wrappers,
  routers, retention. Loss trace worst 5.07e-3 vs floor 9.81e-3 (by-step-11 8.4e-4
  < 1.09e-3 floor-at-11). **Bit-exact.**
- Census (OFF→ON): **TOTAL launch APIs −3,322/step (−13.6%)**; **Command-Buffer-Full
  −370/step (−21%)**; aten::copy_ −718, ::_to_copy −505, ::mm −195 per step;
  regions = 4 cudaGraphLaunch/step (2 fwd + 2 bwd).
- Timing: bwd GPU −25/−8ms, fwd GPU −7/−10ms across the two windows — CAVEAT: the
  ON run also uses expandable_segments (allocator change), so census is the clean
  attribution, timing is indicative.
- **Memory (the local wall):** the graphs' private mempool reserves **9.27GB**
  permanently (region activations become exclusive to the graphs, no longer
  time-shared with the core loop's transient peak). Default allocator → OOM at
  step 36007 (25.1GB used). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  → fits, peak reserved 24.4GB. Cloud 96GB: trivial.

## Combined stack (both flags) — local memory verdict

Both captures compose cleanly (unit probe arm C: bitwise) and both captured fine in
the real run, but the stack **OOMs at mb4/seq4k even with expandable_segments**
(static pool 8.7GB + opt pool + eager peak = 25.1GB; static-only had ~200MB margin;
died at step 36012 in a fused-HC backward alloc). The 12 steps it ran sit inside the
atomics envelope (worst 1.96e-3 vs the 2.08e-3 the first same-code floor pair showed
at the same depth). **The full stack at deploy shape is a 96GB-cloud configuration
locally; static-only or opt-only fit on the 5090.**

## Pitfalls found (each probed, each cost a failed capture or a wrong bit)

1. **Stale autograd graphs kill capture.** Any previous-step graph alive at build →
   params' AccumulateGrad nodes cached with default-stream metadata → capture-stream
   backward syncs with the uncapturable legacy stream (cudaErrorStreamCaptureImplicit/
   Invalidated). train.py had THREE graph-holding locals: loss, out, AND routing_aux
   (reaches the region routers' accumulators) + _sp. All cleared + gc before build.
2. **A failed capture poisons the CUDA generator** ("Offset increment outside graph
   capture" on the next eager RNG op). No silent try/except fallback is possible —
   build failure must abort the run.
3. **Backward must NOT be captured under ambient autocast.** Eager training calls
   .backward() outside the autocast block; capturing autograd.grad inside autocast
   re-dispatches autocast-eligible backward ops to bf16 → 232/292 grads off ~1e-3
   with a bitwise-identical forward. Fix: `_StaticRegion.forward` enters autocast
   itself; make_graphed_callables is called with NO ambient autocast.
4. **Every build-time tensor must be RNG-free** (arange ramps, not randint/randn on
   the CUDA generator): one device randint shifted every later poisson/dropout draw
   (2.6e-2 loss divergence).
5. **The router aux-loss stash protocol is python-side** and does not re-run on
   replay: aux would silently vanish from the loss, and stale stashed tensors are a
   capture killer (see 1). Regions return each router's aux as an explicit graph
   output; dispatch re-stashes per-module. Summing per-region was a measured
   fp-reassociation (1.9e-3 on a 0-floor probe) — per-tensor outputs preserve the
   exact eager addition order.
6. **make_graphed_callables backward returns VIEWS of static buffers** → with
   set_to_none=False + accumulate, `buffer.add_(buffer)` doubles gradients. Regions'
   params are tagged `_grad_via_graph_static`; the optimizer keeps steal-path
   zeroing for them (their grad data_ptrs are stable for free — the opt graph's
   signature check stays green).
7. `bool(dead.any())` + bool-index `[dead]=v` (nonzero) host-sync during capture —
   precompute masks, masked_fill_ inside the graph.

## Not verified / caveats

- _ga>1 (grad accumulation) is UNSUPPORTED with static graphs (2nd micro-backward
  would overwrite the bwd static grad buffers); the build hook refuses. Curriculum
  stage-ups and phase boundaries invalidate to permanent-eager (no auto-rebuild yet).
- Timing deltas for Task 2 are confounded with expandable_segments; a clean A/B of
  allocator-vs-graphs timing was not run. Census/CBF are the attribution.
- Eval/generation paths fall back to eager by the self.training gate — exercised in
  the gate run's post-loop only; no dedicated eval-parity probe.
- Long-horizon behavior (>45 steps), checkpoint save/RESUME while graphs are live
  (save ran fine at 36045; resume-after was not re-tested), wandb-enabled runs.
- The 5.9e-4 "floor" in older notes is step-11-only; this session's 45-step floor
  pair (9.8e-3) should be the reference envelope going forward.

## Files

Flags: `MORPH_OPT_CUDA_GRAPH`, `MORPH_STATIC_GRAPHS` (+ expandable_segments locally).
Probes: `ignore/perf/gpu_probe_optgraph.py`, `gpu_probe_rng_graph.py`,
`gpu_probe_static_graphs.py`, `onestep_grad_diff.py`. Traces/census/logs:
`ignore/perf/{optgraph_base,optgraph_base2,optgraph_on,optgraph_on2,sgraph_on,sgraph_on2,stack_on}.*`.
vlt thread: `morph-optgraph`.
