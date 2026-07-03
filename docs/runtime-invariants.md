# Runtime Invariants

MORPH is a looped / recurrent model trained with truncated BPTT. Several
process-global and import-time choices look like ordinary global state, but they
exist to keep **autograd, `torch.compile`, CUDA graphs, and the recurrent core**
coherent. Treat them as invariants, not cleanup targets.

This document is the public map. Implementation comments in
`morph/training/train.py` and `morph/model/transformer.py` remain the detailed
source of truth.

## 1. Process-global kernel mode

| Switch | Location | Default |
| --- | --- | --- |
| `MORPH_FORCE_EAGER` / `set_force_eager` | `morph/kernels/triton/_eager_flag.py` | off (`0`) |
| `MORPH_HC_FORCE_EAGER` / `set_hc_force_eager` | same | off |
| `model.use_kernels` | Hydra / `MORPHConfig` | `true` |

At `MORPHTransformer` construction, `use_kernels=False` calls
`set_force_eager(True)` so every fused Triton entry point falls back to its
pure-PyTorch reference. That flag is **process-global**: the last model built in
the process wins. Do not build two models with different `use_kernels` values in
one process and expect both to stay correct.

**Why it exists:** same architecture and weights, kernel-ON vs kernel-OFF A/B,
without threading a flag through every call site. Reference paths stay Dynamo-
traceable; fused autograd Functions are fenced with `@torch.compiler.disable`.

**Do not:** flip `set_force_eager` mid-training under a live compile stance, or
assume import-time `DISABLE_FUSED_KERNELS` is the primary switch (runtime path is
`force_eager()`).

## 2. BPTT, checkpointing, and compile

| Invariant | Where |
| --- | --- |
| `torch._functorch.config.donated_buffer = False` | `train.py` import-time |
| Compile **MLP submodules only** (core MLPs `dynamic=True`) | `train.py` |
| Warmup every active-set size (incl. `n_active==1`) **before** wandb / dataloader threads | `warmup_compile_all_shapes` |
| After warmup: `torch.compiler.set_stance("eager_on_recompile")` | `train.py` |
| Generation: `@torch.compiler.set_stance("force_eager")` | `run_generation_test` |
| Truncated BPTT depth / selective activation checkpointing | `model.bptt_depth`, `model.ckpt_grad_iters` |

**Why:** the looped core uses non-reentrant gradient checkpointing. Donated-buffer
reuse aliases inputs under compile. Mid-loop Inductor/Triton recompiles fork
workers while wandb/httpx threads hold glibc arena locks → intermittent hangs.
Warmup in a thread-free window plus `eager_on_recompile` is the mitigation, not
optional polish.

`MORPH_COMPILE_CARVED` (default off at d=768) is an opt-in recompile window at
carve/route boundaries; measured net-negative locally, kept for cloud-scale
revisit.

## 3. Optional CUDA graphs (default off)

| Switch | Role |
| --- | --- |
| `MORPH_STATIC_GRAPHS` | Capture fixed-shape embed/prelude and coda/head regions |
| `MORPH_OPT_CUDA_GRAPH` | Capture fused AdEMAMix step |

The variable-depth core loop stays eager. Fused CE stays eager (host `.item()` is
illegal during capture). Failed capture must abort — never silently fall back
with a poisoned CUDA RNG. See comments on `MORPHTransformer.build_static_graphs`.

Graphs require a large memory footprint, but are faster.

## 4. Phase schedule (prune → carve → route → TST)

Canonical local recipe: `morph/configs/base.yaml`.

| Phase | Keys | Notes |
| --- | --- | --- |
| CMS prune | `prune_start`, `prune_interval`, `target_density` | Density must reach target **before** prune step |
| MORTAR carve | `compact_step` | Freezes topology into BCSR; rebuild optimizer |
| ReMoE route | `routing.route_start` | Requires compact (unless carve explicitly disabled) |
| TST superposition | `tst_bag_size`, `tst_ratio` | Training-only; eval/gen always `bag_size=0` |

`PruningSchedule.step` must run **after** `loss.backward()` and **before**
`optimizer.zero_grad()` so saliency sees live grads.

Routing aux uses `aux_detach_input: true` so load-balance grads do not extend
BPTT depth into OOM.

## 5. Diagnostic env knobs (default off)

These are observation-only when unset: `MORPH_EXACT_TRACE`, `MORPH_MEM_PROBE`,
`MORPH_DIAG_*`, `MORPH_PERF_REGIONS`, `MORPH_PROF_WINDOW`, `MORPH_NSYS_WINDOW`,
`MORPH_DEBUG_STEP`, `MORPH_FAULT_TIMEOUT`, `MORPH_DIV_PPL`.

`MORPH_EXACT_TRACE=<path>` appends per-step loss hex for bit-identical A/B gates.
Use only on gate runs (adds a host sync per step).

## 6. What not to “fix”

- Removing process-global `force_eager` without a per-module replacement that
  preserves reference A/B and Dynamo fences.
- Enabling `compile_mode=reduce-overhead` (CUDA graphs + eval OOM history).
- Setting `fullgraph=True` on the looped core.
- Calling `carve()` while density is still ~1.0 (produces a “sparse” model with
  K/C=1.0).
- Silent fallbacks when a kernel, dataset path, or checkpoint topology fails.

Public contract tests under `tests/test_lifecycle_*.py` cover a minimal subset.
Longer campaign logs and gate scripts live under gitignored `ignore/`.
