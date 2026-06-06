# MORPH — Remaining Kernel Work (compaction handoff, 2026-06-06)

Build-ready handoff for the fused carrier-engine ("megakernel"). Read with
`spec_v2_carrier_engine.md` (the design) and `memory/project_morph_carrier_fusion.md` (the
verified backward math). Repo: `00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`, branch
`perf/cuda-megakernel-campaign`, HEAD `ad8d2b2`. Run: `PYTHONPATH=$PWD /home/wolfe/.venv/bin/python`.

## STATUS SNAPSHOT (where we are right now)

- **Injection-fold (Component 1): BUILT, VERIFIED, then REVERTED.** Bit-exact (kernel fp32 grad
  cos 1.000000; full-model fp32 parity Δloss=0 vs git-stashed baseline). But net-negative:
  **+0.76% wall, +2.9% peak mem (+0.42 GiB), +564 launches** → reverted (`morph/` is clean at
  ad8d2b2). The verified glue-backward math is preserved in memory + the doc — REUSE it in the
  engine. Lesson encoded as the two iron gates below.
- **RUNNING:** n=2-vs-n=4 ppl A/B on the full deploy quant stack, 7k steps
  (`ignore/run_n2_quant_ab.sh`; logs `ignore/n2_quant_7k.log`, `n4_quant_7k.log`; wandb
  `morph/n2_quant_7k`, `n4_quant_7k`). n=2 healthy at step ~1200 (val ppl 326 @ 1000, descending).
  **This locks the carrier `n`.** n=4 arm = the matched baseline + deploy baseline.

## THE REFRAME THAT DRIVES EVERYTHING (measured)

MORPH is **bandwidth-bound, not compute-bound.** Whole-step aten-op profile (B4·S4096, bf16):
`aten::mm` = 133 ms = **20% of busy**; the other **~80%** is memory-bound carrier traffic
(`copy_` 91, `mul`+`add`+`sum` ~160, HC kernels 88 [elementwise + 4×4 FMA, NOT tensor cores],
attention ~50). On the quant stack it's even more lopsided (~12% GEMM). **Per-op glue folds are
sub-1%** (proven: inject-fold +0.76%) because the glue is hundreds of scattered ops. The engine's
thesis: fuse the whole IN/OUT carrier pass + cross-layer residency to clear the relocation
overhead that sank the per-op fold. UNPROVEN until built + measured against the re-baseline.

## TWO IRON GATES (every kernel change must pass — these encode tonight's failure)

1. **peak memory ≤ baseline.** The fold's +0.42 GiB came from saving `term` for backward.
   Engine MUST NOT inflate retained activations: use "previous-POST writes h_inj" (post writes
   the already-injected carrier → saves h_inj like baseline, no extra term) OR recompute in bwd.
2. **launches ≤ baseline.** The fold's +564 came from the proj-split's small GEMV/cast/reduce.
   Fuse INTO existing kernels (no new small kernels); cache loop-invariants (proj_w_sum,
   version-keyed like the FP8 weight cache).

## REMAINING KERNEL WORK (ordered, each gated)

### Step 0 — Lock carrier + re-baseline  [BLOCKING, in progress]
- Read the A/B: if n=2 val ppl ≈ n=4 at 7k → **lock n=2**; else n=4 (+ pursue fp8/int8 carrier
  precision instead, ppl-gated). The kernel is built n-GENERIC regardless.
- Re-baseline on the LOCKED config (n + full quant stack) with `ignore/profile_step.py` +
  `attribute_elementwise.py`. ALL engine deltas are measured vs THIS, not the bf16 numbers.

### Step 1 — Make the HC kernels n-GENERIC  [✅ BUILT + VERIFIED 2026-06-06]
- DONE. Added a constexpr-N **[N,N] register-tile** reformulation of the fused mapping in
  `morph/kernels/triton/fused_hyper_connection.py`: `_tile_mm`, `_hc_premap_fwd_kernel_g`,
  `_hc_premap_bwd_kernel_g`, autograd `_FusedHCPreMapGeneric`. The mapping math (rms,
  softmax×2, 3-iter Cayley, analytic VJP) is now matrix-tile ops (`tl.trans`, broadcast
  matmul, axis softmax) instead of `a00..a33` scalars — serves any power-of-2 N. The proven
  4×4 **scalar** kernel is KEPT as the tuned n=4 production path (zero regression risk to the
  live deploy run); the generic kernel handles n=2 (and any pow2). `hc_pre_map` dispatch:
  pow2 N & iters==3 → (N==4 ? scalar : generic), else eager. **Model needed ZERO edits** —
  `HyperConnectionResidual.forward` already binds `hc_pre_map` unconditionally (no n==4 guard);
  the POST kernels were already n-generic. So flipping the kernel dispatch is all it took.
- GATES (all PASS):
  - `ignore/verify_hc_ngeneric.py`: (A) generic@N=4 vs **scalar kernel** (regression oracle) —
    fwd cos ≥ .9999999, ALL grad cos = 1.000000 (two-oracle proof); (B) generic@N=2 vs eager
    ref — fwd .999996, grad 1.0, kernel-vs-truth ≤ ref-vs-truth; (C) dispatch routes n=2→generic;
    (D) tamper fails-when-broken.
  - `ignore/parity_model_n2_fused.py`: FULL 262M model fwd+bwd, n=2, HC-fused vs HC-eager
    (`set_hc_force_eager`, all other kernels held on). **fp32 bug-detector: rel_loss 1.2e-6,
    gnorm 2.3e-5, worst_sig_cos 0.999996 → NO composition bug.** bf16: worst grads =
    `core.*.mrr_mlp/attn.proj.weight` cos~0.991 (looped-core HC proj weights accumulate bf16
    reassoc over ~6-8 core applications/step) = same floor regime as the scalar n=4 kernel.
- REMAINING for this step (honest): (i) torch.compile path NOT exercised (eager runs only;
  shared-structure with the proven scalar autograd.Function → low risk); (ii) **perf NOT
  measured** (last-chopper: full-step re-baseline only after n locks; module-level the fused
  n=2 path replaces eager's ~65-launch mapping storm with 1 kernel + 1 GEMV so a win is
  near-certain, but unmeasured); (iii) n=2 deploy only happens if the A/B locks n=2. UNCOMMITTED.

### Step 2 — Carrier-engine (the holistic fusion)  [the main prize]
- Extend the premap (IN) + post (OUT) kernels to ABSORB the inter-GEMM glue: inject (via
  "previous-POST writes h_inj"), residual add, bf16↔fp32 casts — so the carrier is touched
  ONCE per side, not once-per-op. Reuse the verified inject-grad (grad_term = grad_h.sum(streams),
  grad_w += tile) from `memory/project_morph_carrier_fusion.md`.
- Hand-written fused backward: (inject-grad) ∘ (existing analytic premap/post bwd, already in
  repo) ∘ (residual/cast). RECOMPUTE cheap glue in bwd (FlashAttention-style), don't store.
  Compose with truncated-BPTT (last `bptt_depth` iters only), checkpoint (use_reentrant=False),
  active-set shrinking (all already in `_core_step`).
- Gate the FULL chain: bit-exact fwd → model parity (`parity_model_fold.py`, Δloss=0 vs
  git-stash) → **peak mem ≤ baseline** → **launches ≤ baseline** → re-profile (the 80% dropped)
  → 7k ppl parity.

### Step 3 — Cross-layer residency  [stretch]
- L2-persistence (host-side `cudaAccessPolicyWindow` + `cudaDeviceSetLimit`) pins the carrier;
  cc 8.0+ → all deploy archs, numerically a no-op. Measure on/off.
- CUDA C++ POST(i)+inject(i+1)+PRE(i+1) register-resident kernel = the only thing Triton can't
  do. Build ONLY if it provably beats the Triton+L2 baseline in a head-to-head A/B. Toolchain
  de-risked: `ignore/cuda_toolchain_test.py` (g++-13 ccbin, sm_120 OK).

### Step 4 — Validate + ship
- 15k run on the deploy config (Hydra, wandb FULL config). Must reproduce the locked-config
  baseline curve. Cleanup leaked wandb runs + `ignore/` temp scripts before commit.

## PORTABILITY / MULTI-GPU (bake in from line one)
- ONE portable Triton path (auto-ports sm_90/100/120, auto-tunes per arch). n-generic,
  quant-aware (coexist with ternary backbone / int6 embed / 8bit AdamW).
- Multi-GPU: `c10::cuda::getCurrentCUDAStream()`, `CUDAGuard(tensor.device())`, `.contiguous()`,
  prefer `torch.library.custom_op` + `register_autograd` + `register_fake`.

## REUSABLE TOOLING (all built tonight, all in `ignore/`)
| file | purpose |
|---|---|
| `verify_inject_fold.py` | kernel bit-exact gate (fp32 grad cos 1.0 pattern; tamper fail-when-broken) |
| `parity_model_fold.py` | full-model fp32/bf16 parity via `save`/`save_fp32` + git-stash + `compare` |
| `profile_step.py` | whole-step busy/idle + kernel-duration histogram + top kernels |
| `attribute_elementwise.py` | aten-OP breakdown (maps kernels→source: mm/copy_/mul/add/sum) |
| `profile_hc_idle.py` | model fwd+bwd busy/idle/launches/peak; `SKIP_INJECT_ADD` upper-bound probe |
| `run_n2_quant_ab.sh` | the n2/n4 quant A/B driver (MORPH_DEBUG_STEP retry pattern) |
| `cuda_toolchain_test.py` | sm_120 CUDA C++ toolchain proof (for Step 3) |

## KEY NUMBERS / CONFIGS (resume context)
- Baseline (bf16, B4·S4096, pre-fold): wall ~656 ms, busy ~651 ms (99% busy), 22264 launches,
  peak 14.63 GiB. inject-add ceiling (SKIP probe) = 31 ms (4.6%). aten::mm 133, copy_ 91.
- Deploy quant stack flags: `training.ternary=true training.ternary_scope=backbone
  training.adam8bit=true training.embed_quant=int6`.
- 7k A/B common: `training.steps=7000 training.prune_start=999999 training.compact_step=9999999
  training.eval_every=500 training.lr=1e-4 training.min_lr=1e-4 model.use_kernels=true
  training.batch_size=4` + `model.hc_streams={2,4}`.
- Startup timing-race (OPEN, masked): launch with `MORPH_DEBUG_STEP=1` + retry-on-no-step-200
  (run_*.sh pattern). Startup-only; relaunch recovers.

## CARRIER DATAFLOW (the surgery surface)
- `transformer.py`: `_core_step` (diagonal `self.injection` once → per-layer `_build_injection_term`
  → `_apply_injection` → `layer()`), active-set shrinking (`sort_depths`/`perm`,
  `active_counts`, `h_s = cat([h_new, h_s[n_active:]])`), prelude/coda loops, `x.mean(dim=2)` readout.
- `hyper_connections.py`: `HyperConnectionResidual.forward` (fused path: `hc_pre_map` → sublayer →
  `hc_post`; eager path for n≠4/sinkhorn).
- `mhc.py`: `MORPHBlock.forward` (mrr_attn then mrr_mlp), `MultiRateResidual`/`StandardResidual`,
  `ChannelInject`.
- `fused_hyper_connection.py`: `_hc_premap_fwd/bwd_kernel`, `_hc_post_fwd/bwd_kernel`,
  `_FusedHCPreMap`, `_FusedHCPost`, public `hc_pre_map`/`hc_post` + references. **N==4 hardcoded.**

## RISKS / UNVERIFIED
- n=2 ppl verdict pending (~1.5h out). fp8/int8 carrier precision = ppl risk (gated).
- Engine must MEASURABLY beat sum-of-per-op-folds (per-op = sub-1%; thesis unproven till built).
- Cross-layer residency win unproven (L2 may capture most of it).
- No multi-GPU run yet. Startup race still open (masked).
