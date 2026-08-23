# Agent Note: Throughput budget and cloud plan

Status: proposed

## Problem

The deploy-scale run is not tractable at today's throughput. Wolfe's 2026-08-21 target:
train on ~8 rented RTX 5090s, order $400 budget, and reach as many tokens as possible on
the cloud model (d2048, 4:8:4, ~850M physical params, ~3.5B effective params per token
before pruning, ~1B after the MORTAR prune to 0.25 density). The earlier perf pass
(2026-07-03, [structural-2x-hunt](2026-07-03-structural-2x-hunt.md)) was boxed to bit-exact
launch cuts and produced ~5%. This note records what is measured, where the remaining
uplift is, what the cloud costs, and what has to be developed on a cheap dev box first.

## Measured baseline (2026-08-21, 5090, `tul_short.yaml` A0: d1024, bs14, seq1024)

| Quantity | Value | Source |
|---|---|---|
| Step rate | 1.235 sps, **17.7k tok/s**, 284.0M params | wandb `morph-tul/tul-a0-acap1`, step 20k |
| Forward FLOPs per token | 1793 MFLOP; core loop 5.69 × 254 = **1444 (80%)**; prelude+coda 347; attention score/value 1.8% | `morph/training/flops.py` on the built model |
| Executed FLOPs per token | ~5.84 GFLOP = fwd + bwd + **4-iteration checkpoint recompute (1016 MFLOP, 17%)** + LM head (not counted by `flops.py`) | same |
| Realized compute | **103 TFLOPS** | 5.84 GFLOP × 17.7k tok/s |
| 5090 cuBLAS bf16 ceiling | **219–223 TFLOPS** at 16k×4k×4k and at the gate_up shape 14336×1024×5632; 150 at a skinny N=512 projection | measured on the local card |
| MFU | **~47% executed, ~40% useful** | |
| Core-loop region | ~57% MFU (A0 step minus A3 step) | A3 = `n_core 0` arm, 49.2k tok/s |
| Non-core region | **~30% MFU**: 0.29 s/step, of which ~75 ms is the 8 layers' GEMMs; ~165 ms is embed, fused CE, HC expand/reduce, optimizer, launches | A3 arm |

Conclusion: there is no 4× in kernels. Kernel-perfect is 2.1× theoretical; ~1.3× is the
practical kernel headroom. The cost is FLOPs per token: the loop and the recompute.

## Proposal

### A. Levers that do not change the loop budget (Wolfe: loop budget is out of scope for now)

| # | Lever | Expected | Class | Status |
|---|---|---|---|---|
| 1 | `ckpt_grad_iters=0` (no backward recompute of the 4 grad iterations) | **×1.21 measured** at equal batch | exact within the noise floor (control-run gated) | measured 2026-08-21, see A2; VRAM-gated: +0.61 GB per retained row-iteration |
| 2 | Non-core region cut: profile the A3 arm, not A0. Targets: fused CE (a 201 MB fp32 `grad_w` per call), hybrid-embedding + Lorentz `lm_weight()` rebuild, int6 embed STE, HC expand/reduce, the 284M-param optimizer step | ×1.1–1.15 on the full step | mostly exact | not profiled |
| 3 | TUL on (`tul_a1`) | **×1.6 measured** (28.1k tok/s, slightly better CE, [the arms result](../../../../docs/experiments/results/2026-08-18-tul-arms-first-comparison.md)) | validated on the 5090 arms; not validated with prune/carve/route | measured |
| 4 | FP8 GEMMs on the ternary backbone. Ternary {−1,0,+1}×γ is exact in e4m3, so only activations are newly quantized. 5090 FP8 tensor rate is 2× bf16 | **~×1.07** (was ≤×1.3 — withdrawn, see A4) | B (numerics); `base.yaml` says ternary and FP8 are mutually exclusive per layer — a code constraint from the d768 era, not a math one | not built |
| 6 | Dead saliency scoring: `accumulate_scores()` ran on all 14 MortarLinears every step even when `prune_start` can never fire (the dense TUL arms). Measured **50 ms wall / step** of a 950 ms step (MORPH_PERF_REGIONS `prune` region, A0 batch 14, 2026-08-21). `PruningSchedule.scoring_live` now skips it when `prune_start >= total_steps` | ×1.05 on dense arms; 0 on the deploy recipe (which prunes) | exact (touches only the EMA buffers, never params) | built; `tests/test_lifecycle_phase_transition.py::test_scoring_is_skipped_when_prune_can_never_fire` |
| 5 | Lorentz/int6/QAT fusion and the `_to_copy` cast storm (5,091/step at d768) | part of #2 | exact | see structural-2x-hunt |

Stacked without the loop budget: **×2.2** (1+2+3), **×2.8** with 4. Numbers 1, 2, 4 are
estimates; 3 is measured.

### A2. Lever 1 measured (2026-08-21, local 5090, `tul_short`, 150 steps, seed 0, `ignore/perf/ckptgate_*`)

| batch | `ckpt_grad_iters` | step ms | tok/s | peak VRAM |
|---|---|---|---|---|
| 4 | −1 | 333 | 12,617 | 8.32 GB |
| 4 | **0** | **274.5** | **14,991** | **18.07 GB** |
| 8 | −1 | 528 | 15,211 | 13.28 GB |
| 8 | 3 (last iter only) | 531 | 15,390 | 15.95 GB |
| 14 | −1 | 950 | 15,300 | 20.58 GB |
| 14 | 0 or 2 | OOM in compile warm-up (~25.8 GB process) | | |

- **×1.21 at equal batch** when all 4 grad iterations are retained — matches the −17% FLOP estimate.
- Cost: **0.61 GB per retained row-iteration** at seq 1024 (the 4-stream HC carrier). Retaining all 4
  at batch 14 needs ~+22 GB; no 32 GB card fits it. At batch 4 the gain (×1.21) barely covers the
  batch-size penalty (the batch-4 baseline is 17% slower per token than batch 14).
- Retaining only the LAST iteration buys nothing: the last iter has the smallest active set, so it is
  cheapest to retain and cheapest to recompute. The config comment calls this the "efficient
  frontier"; it is backwards. Un-checkpointing must start from the FIRST grad iteration to pay.
- **Exactness:** traces are not byte-identical, but an identical-baseline control diverges MORE
  (max rel 2.9e-3 by step 11, 1.1e-1 by step 150) than `=0` does from the baseline (4.0e-4 by
  step 11, 2.8e-2 by step 150). Exact within the atomics noise floor, as the config claims.
- Conclusion: lever 1 is real but **VRAM-gated**. On 32 GB cards it pays only after the activation
  footprint is cut (the memory half of lever 2: fused CE `grad_w`, router retention, HC carrier) or
  at a per-GPU batch that costs as much as it gives. On 96 GB cards it is free.

### A3. Where the step time actually goes (kernel census, 2026-08-21)

Method: `MORPH_PROF_WINDOW` kineto window on the REAL `train.py` step, arms `tul_a0`
(n_core 6) and `tul_a3` (n_core 0), batch 14 x seq 1024. Kernel rows only (self-CPU == 0),
so no CPU-op double counting. Artifacts `ignore/perf/lever2*`.

| bucket | A0 ms/step | calls | A3 (no core) | core = A0 - A3 |
| --- | --- | --- | --- | --- |
| generic pointwise elementwise | **237.5** | 9356 | 107.3 | 130.2 |
| cuBLAS GEMM | 199.6 | 790 | 78.3 | 121.3 |
| custom fused Triton (HC/CCA/CSA/window) | 199.1 | 1032 | 53.5 | 145.6 |
| optimizer | 25.3 | 1 | 24.4 | 0.9 |
| reduce/norm, index, memcpy, softmax | 36.0 | 1623 | 18.7 | 17.3 |
| **total self-CUDA** | **805.0** | | 319.5 | 485.5 |

Two facts kill the older "launch-bound" model of this step:

1. **GPU-busy is 890 ms of a 904 ms wall step (98.5%).** The 2026-07-03 pass fixed the
   launch problem. What is left is work, not latency.
2. **Pointwise elementwise is the largest bucket** — larger than the GEMMs. Useful work is
   83.7 TFLOP/step; at the measured 219 TFLOPS cuBLAS ceiling the floor is 382 ms, so
   kernel-perfect is x2.1 and the elementwise bucket is most of that gap.

The elementwise kernels, by full name (the kineto table used to truncate these; fixed):

| ms/step | calls | kernel |
| --- | --- | --- |
| 76.0 | 1283 | `CUDAFunctor_add<float>` — fp32 adds |
| 45.0 | 1938 | `bfloat16_copy_kernel_cuda(lambda(float))` — fp32 to bf16 casts |
| 36.3 | 1515 | `BinaryFunctor<float,float,float>` |
| 25.3 | 193 | `CUDAFunctor_add<float>` at 131 us each — the `[V,d]` head |
| 23.5 | 1336 | `direct_copy_kernel_cuda` |

**Root cause: the residual carrier `[B,S,4,C]` is fp32, not bf16** (probed: 64 MiB per copy
at batch 4, so 224 MiB at batch 14). Three sites promote it:

1. the embeddings emit fp32, so `transformer.py` `carrier::expand_contig` starts fp32;
2. `e = self.input_norm(x)` at the core entry returns fp32 (norms upcast);
3. `DiagonalInjection.forward` computes `A * h_ctx + dt * e_ctx` against fp32 parameters
   and then `torch.cat`s, which promotes the whole carrier.

Cast at those three sites and the carrier is bf16 end to end. Measured at a PINNED loop
depth of 6 (identical work in every arm — the Poisson draw alone is worth up to 14%, which
is larger than every effect under test, and it invalidated the first two attempts):

| carrier | bptt=4 | bptt=2 | bptt=1 | peak VRAM (bptt=4) |
| --- | --- | --- | --- | --- |
| fp32 (today) | 1212.9 ms | 904.8 | 742.9 | 21.49 GB |
| bf16 | 1081.8 ms | 803.9 | 675.3 | 19.21 GB |

**bf16 carrier = x1.12 and -2.28 GB at matched `bptt_depth=4`**, reproducible to ~1% over
two rounds. `bptt_depth` is the most expensive single knob in the step (x1.63 from 4 to 1)
but Wolfe ruled it out on quality grounds [W, 2026-08-21]: the 4-deep window stays.

Precision floor: bf16, not lower. `lorentz_fraction=0.25`, so a quarter of the carrier
channels are Lorentz tangent coordinates. The manifold math (`_project_to_hyperboloid`,
`_log_map_origin`) runs on fp32 parameters inside `lm_weight()` and autocast leaves it
there; only the transport of the resulting tangent vector becomes bf16.

Two other measured items:

- **torch.compile scope.** Compiling the whole `MORPHBlock` instead of only `layer.mlp`
  did NOT reproduce a speed win over three interleaved rounds (mixed sign). It is worth
  -0.76 GB of peak. Re-measure at pinned depth before deciding.
- **Ternary QAT cost.** 43 modules, 214.8 M parametrized elements (76% of the 284 M model).
  `torch.nn.utils.parametrize` recomputes the quantizer on EVERY `module.weight` access;
  one pass over every parametrized weight costs 7.63 ms eager vs 0.021 ms inside
  `parametrize.cached()`. At ~2.75 passes per step that is ~21 ms. Nothing in the tree uses
  `parametrize.cached()`.

### A4. FP8: not running, and one confirmed bug

`base.yaml` has `fp8: false`, and neither FP8 path can run today:

- `morph/model/fp8_scope.py` needs **torchao, which is not installed** in the venv. It has
  never executed.
- `CMSBlockLinear.enable_fp8()` (raw `torch._scaled_mm`, no dependency) has **no caller**
  anywhere in `morph/` or `tests/`.

**Confirmed bug (latent).** `_get_cached_fp8_weight()` recasts only when
`self.weight._version` changes. That is the right key for a plain `nn.Parameter`. Ternary
QAT registers a `parametrize` parametrization on the SAME weight, and a parametrized
`.weight` is recomputed on every access — a new tensor whose version starts at 0. Measured
(`scratchpad/probe_fp8_cache.py`):

```
plain nn.Parameter:   step 0 -> _version 2, step 1 -> 3, step 2 -> 4
with ternary QAT:     step 0 -> _version 0, step 1 -> 0, step 2 -> 0   (values changed: True)
shadow original after 3 steps: _version 5
```

So FP8 on a ternary-parametrized Linear would cast once and hit the cache forever: the FP8
weight freezes at its step-0 value while the shadow keeps learning. No error, no NaN — the
layer silently stops contributing. Fix: key the cache on
`parametrizations.weight.original._version` when the module is parametrized.

**The per-layer FP8/ternary exclusion is over-stated.** The FP8 note forbids the overlap
because "a ternarized Linear's forward weight is already ternary — there is no bf16 weight
to cast." That is an argument about redundant COMPRESSION and it misses that FP8 is also a
RATE mechanism. A ternary weight `{-1,0,+1}x gamma` under dynamic per-tensor scaling maps to
exactly +-448 and 0 in e4m3 — lossless on the weight operand, so only the activation is
newly quantized. Ternary-plus-FP8 is therefore the SAFEST FP8 arm available, not a
forbidden one.

Size it honestly: core-loop GEMM is 121 ms of the 805 ms step and MLP is ~85% of it, so a
perfect 2x on the MLP GEMM is about **x1.07** end to end. The FP8 note's own
"single-digit-to-low-double-digit %" is right; the "<= x1.3" in section A above was
optimistic and is withdrawn.

### A5. Round-2 task list (ordered by measured value / risk)

| # | Task | Expected | Class | Gate |
| --- | --- | --- | --- | --- |
| 1 | **bf16 carrier.** Cast at the three promotion sites (A3). Baked in, no runtime flag. | **x1.12, -2.28 GB (measured)** | B (numerics) | dtype invariant test: the carrier is bf16 at every core block under autocast; then a paired quality arm vs `tul_a0` on val CE. Watch residual accumulation over 40+ updates. |
| 2 | **Re-test lever 1 after task 1.** -2.28 GB may make `ckpt_grad_iters` affordable at batch 14; re-measure the 0.61 GB per retained row-iteration. | up to x1.21 if it now fits | exact | pinned-depth A/B + peak VRAM. |
| 3 | **Fix the FP8 weight-cache key** (A4). Belongs on master whether or not we use FP8. | correctness | bug fix | test that steps the optimizer and asserts the cached FP8 weight changed. |
| 4 | **Make FP8 reachable and measure it.** Wire `enable_fp8()` to a config key; arm = ternary backbone + FP8 on the same CMS linears (exact in e4m3). Install torchao only if the nn.Linear path is also wanted. | ~x1.07 | B | the FP8 note section 8 gates + a loss curve. Blocked by task 3. |
| 5 | **`fused_ce` grad_w accumulation.** Replace `grad_w += (probs_c.t() @ x_c).float()` with `grad_w.add_(probs_c.t() @ x_c)` — drops a 201 MB fp32 temp per chunk, 14 chunks per step. | ~9 ms (~1%) | **exact** (bf16 to fp32 upcast is exact either way) | bit-exactness against the current path. |
| 6 | **LM-head weight rebuild.** 42 copies of `[1024,49280]` plus 42 of `[49280,1024]` per step, ~32 ms total, from `lm_weight()` and the per-chunk cast. Build the head weight once per step. | ~2-4% | exact if cached within a step | bit-exactness; the gradient must still reach `euc_embed` and the Lorentz table. |
| 7 | **`parametrize.cached()` around fwd+bwd** (A3). Must not survive the optimizer step. | ~21 ms (~2%) | A/B (accumulation order) | loss-trace gate; assert the cache is empty at `optimizer.step()`. |
| 8 | **Re-measure whole-block torch.compile at pinned depth.** Cheap; -0.76 GB is already banked. | unknown | A | pinned-depth A/B. |
| 9 | **stk `dds`/`sdd` microbench at d2048, density 0.25** (section B). Unchanged: this is the 30B-vs-70B token-budget gate, not a throughput lever. | budget decision | n/a | vs cuBLAS dense at the same shape. |

Tasks 1, 3 and 5 are independent and can land in any order. Task 2 depends on 1; task 4
depends on 3.

### B. The pruning claim, stated honestly

Pruning to 0.25 density cuts MLP GEMM FLOPs 4× on paper, and MLP is ~85% of per-layer
FLOPs at d2048 (3·d·d_ff = 34.6M vs ~6M for CCA attention). ReMoE at `activation_ratio`
0.5 halves it again. So the routed regime (70% of the run) could cost ~0.3× the dense
per-token FLOPs. **That saving is not measured anywhere.** At d768 the carve netted only
−14 ms of a ~790 ms step (dense MLP −62 ms, BCSR kernels +48 ms), and the router added
+60–80 ms ([perf-pass-phase1](../../implemented/process/2026-07-03-perf-pass-phase1.md)).
"3.5B effective → 1B effective" is true for parameter count; whether the stk `dds`/`sdd`
kernels turn that into wall-clock at d2048 block counts (16×44 = 704 blocks per gate_up
vs 352 at d1024) is the single most important unknown for the cloud budget. It decides
whether $400 buys ~30B or ~70B tokens.

Gate: microbench `morph/sparse/stk` dds/sdd at d2048 shapes, density 0.25, and the
router, against cuBLAS dense at the same shape, before any cloud money is spent.

### C. Multi-GPU

`train.py` has no DDP today; this is greenfield. Data-parallel is linear at best:
~3.5× on 4 GPUs, ~7× on 8. It raises per-GPU efficiency only indirectly, by giving the
VRAM for lever 1.

Communication estimate (not measured): d1024 fp32 grads 1.1 GB → ring all-reduce ~1.7 GB
per GPU per step, ~85 ms at ~20 GB/s PCIe P2P, overlapped with a ~0.5 s backward → near
zero visible. d2048 ~850M params → ~6 GB per GPU per step, ~150 ms with a bf16 comm
hook, mostly overlapped → ~5–10% visible. NVLink does not pay for itself (table below).

MORPH-specific items a generic DDP wrap gets wrong:

1. **Poisson-depth straggler.** `_sample_depths` (`transformer.py:616`) draws per rank.
   The sum of depths over 14 sequences has ~11% std, so the slowest of 8 ranks runs
   ~15% longer than the mean every step and every all-reduce waits for it. Draw depths
   from a rank-shared generator so every rank sees the same depth multiset. Free, exact.
2. **Prune-saliency divergence.** `block_score_ema` (`block_sparse.py:453`) accumulates
   from local grads. Without an all-reduce before each prune event, ranks prune different
   blocks and the replicas desync. Carve and route topology: compute on rank 0, broadcast.
3. Loader rank-sharding on the RAM/mmap placement path; rank-0-only wandb, eval, ckpt.
4. `torch.compile` + DDP + the optgraph static-graph path; the GradScaler `found_inf`
   becomes a collective.
5. Checkpoints pushed off-box every `ckpt_every`; rented boxes die (97–99.5% reliability).

Develop and measure this on a rented **4×3090 box at $0.41/hr** (under $10/day), not on
the train box.

### D. Cost comparison (Vast.ai listings, 2026-08-21, "plus bandwidth")

Relative speed = bf16 dense fp32-accumulate tensor peak relative to the 5090 (the 5090
figure is measured at 219–223 TFLOPS; the others are vendor peaks, with a haircut for
the bandwidth-bound half of the step on 24 GB / ~880 GB/s cards).

| Box | $/hr | $/GPU-hr | rel. speed | speed per $/hr | Notes |
|---|---|---|---|---|---|
| 4×RTX 3090 (US, PCIe 3) | 0.412 | 0.10 | 0.32 | **3.2** | best per $; 3× wall-clock; no FP8; dev box |
| 1×RTX 5090 (KR) | 0.349 | 0.35 | 1.0 | 2.9 | single GPU; 8-day cap |
| 1×RTX 5090 (US, PCIe x8) | 0.361 | 0.36 | 1.0 | 2.8 | |
| **4×RTX 5090 (SI, PCIe 5 x16, 54 GB/s)** | 1.769 | 0.44 | 1.0 | **2.3** | 11-day cap, 99.5% |
| 8×A100 SXM4 40 GB (CA) | 4.273 | 0.53 | ~1.15 | 2.2 | no FP8; 95.7% reliability |
| 8×RTX 4090 | 2.946 | 0.37 | ~0.7 | 1.9 | 24 GB, 883 GB/s |
| 4×RTX 4090 (ES) | 1.343 | 0.34 | ~0.7 | 2.1 | |
| 8×A100 SXM4 40 GB (US-GA) | 6.741 | 0.84 | ~1.15 | 1.4 | |
| 8×RTX 6000 Ada (TR) | 4.809 | 0.60 | ~0.8 | 1.3 | |
| 4×B200 (US-VA) | 24.259 | 6.06 | ~8 | 1.3 | |
| 8×H100 SXM (US) | 23.476 | 2.93 | ~3.2 | 1.1 | NVLink; 2× worse per $ |
| 1×B200 (US-OR) | 5.321 | 5.32 | ~8 | 1.5 | |
| 8×RTX PRO 6000 Blackwell (CA) | 9.622 | 1.20 | ~1.1 | 0.9 | only if 96 GB is required |

8×5090 blades were all rented at the time of the survey; expect ~$3.5/hr by scaling the
4× price.

### E. What $400 buys

4×5090 at $1.769/hr → 226 h ≈ 9.4 days (inside the 11-day cap). Reserve ~15% for setup,
eval, checkpoint I/O → ~770 productive GPU-hours. Bandwidth is billed: ship a
pre-tokenized corpus, not raw text.

| Model | tok/s per GPU | Tokens for $400 |
|---|---|---|
| d1024 (284M, ~0.9B effective/token), today | 17.7k | ~49B |
| d1024 after the ×2.2 stack | ~39k | ~108B |
| d2048 cloud (~3.5B effective/token), today, dense | ~5k | ~14B |
| d2048 after ×2.2, dense | ~11k | ~30B |
| d2048 after ×2.2, **if** the routed regime realizes ~0.3× FLOPs for 70% of the run | ~25k blended | **~70B** |

200B tokens is not inside $400 on any hardware (~$1,500 at d1024 with the stack; ~$5,500
at d2048 dense). 70B at d2048 is reachable only if section B's gate passes.

### F. Research-grade levers (must be tested; not in the ×2.2)

These change numerics or the training recipe. Each needs an ablation arm. Ranked by
expected size.

1. **FP8 with ternary weights** (lever 4). The weight side is exact; the open question
   is activation scaling (per-tensor delayed vs per-block) through the looped core,
   where errors compound `ρ^T`.
2. **Sparse-kernel efficiency at 0.25 density** (section B). If stk cannot reach >2×
   over cuBLAS at d2048, alternatives: 2:4 structured sparsity on the tensor cores
   (hardware 2× on SM120, but 0.5 density not 0.25), or block size 256 for fewer,
   fatter BCSR tiles.
3. **Core-loop CUDA graph** — blocked locally by dynamic `[n_active, S, n, C]` shapes
   and VRAM ([structural-2x-hunt](2026-07-03-structural-2x-hunt.md)); on multi-GPU with
   smaller per-GPU batch and lever 1 the VRAM blocker lifts. Pad `n_active` to a fixed
   bucket to lift the shape blocker. Exact.
4. **Cross-iteration KV sharing (CLA)** — plumbing exists (`cla_capture`/`cla_kv`);
   prior arm `cla_iter1_b4`. Removes K/V projection + CCA conv from T−1 of T iterations.
   Parked pending ablation.
5. **Router cost** (regime 4, 70% of the run): route once per loop-iteration group or
   share gates across iterations; bf16 router. Class B.
6. **HC carrier width** n=4 → n=2: halves the residual's memory traffic (the
   bandwidth-bound half of the step). The n2-vs-n4 quant A/B in remaining-kernel-work
   decides; re-run at d1024+.
7. **Sequence-length curriculum** (short seq early): attention is 1.8% of FLOPs, so
   this buys little here; skip unless the data side wants it.
8. **Fewer gradient iterations (`bptt_depth` 4 → 3)** cuts backward FLOPs of the loop by
   25% — this is a loop-budget change and is explicitly out of scope for now; listed so
   nobody re-derives it.

### G. Iso-depth scaling law (verified 2026-08-21)

`arXiv:2604.21106`, Schwethelm, Rueckert, Kaissis, *How Much Is One Recurrence Worth? Iso-Depth
Scaling Laws for Looped Language Models* (Apr 2026): `L = E + A(N_once + r^phi N_rec)^-alpha + B D^-beta`
with **phi = 0.46** (full BPTT), **0.38** (truncated BPTT), **0.65** (hyper-connections). With HC,
r = 8 iterations of the 8-layer core is worth `8^0.65 = 3.9` cores of capacity, not 8: the cloud
model executes 72 layer-passes per token for ~39 layer-equivalents of capacity. This is the
quantitative form of "the loop is the cost" and the quality-side number for any future
`bptt_depth` or `mean_depth` decision (out of scope now). Goes in `docs/references.md` section 1.

Also verified: `arXiv:2607.14427` (Logan, Jul 2026) per-token fixed-point exit is an
**inference** result (4.94 mean loops for depth-8 quality, 135M on one 4090), not a training
lever. `arXiv:2606.31796` (CHERRY, Kwon and Park, Jun 2026) reports that selected-token
supervision "collapses free generation" — evidence against Rho-1-style selective loss here.

## Alternatives considered

- **Keep the bit-exact launch-cut path from 2026-07-03.** It produced 5% and the
  command-buffer stall analysis says the stack is worth ≤ 2× only at the cloud shape.
  Lost because the measured MFU is already 47%: launch cuts cannot give the asked 1.5–4×.
- **Rent H100/B200 for NVLink and FP8 transformer engine.** 2–2.5× worse per FLOP-dollar
  than 5090s; this model's all-reduce fits PCIe. Lost on price.
- **Rent 3090s for the train run.** Best per dollar (3.2) but 3× the wall-clock, which
  busts the duration caps and the 2–3 day goal. Kept as the dev box.
- **Go straight to 8×5090 without a dev run.** Lost because DDP is greenfield and the two
  MORPH-specific desync bugs (depth straggler, prune saliency) would burn paid hours.
- **Attention-kernel work first** (Wolfe's initial guess). Attention score/value FLOPs
  are 1.8% of the step at seq 1024 and the skinny projections were already fused in
  July. Lost on the numbers.

## Acceptance criteria

1. Lever 1 gated: loss trace within the 5.9e-4 noise floor over 200 steps vs
   `ckpt_grad_iters=-1`, and tok/s reported with `perf/flop_proxy` and peak alloc.
2. A3-region profile exists with a ms breakdown of the ~165 ms non-layer time.
3. stk dds/sdd microbench at d2048 / density 0.25 vs cuBLAS dense, same shapes, reported
   as a ratio. This decides the 30B-vs-70B row.
4. DDP on 4×3090: scaling efficiency measured; depth multiset identical across ranks
   (assert); prune masks identical across ranks after a prune event (assert).
5. $25 pilot on 1×5090: image boots, corpus ingests, checkpoint leaves the box.
6. Only then the $400 run, with the model size chosen from section E.

## Risks

- The relative-speed column is vendor peaks scaled by one measured card; the 3090 and
  4090 rows could be off by ±20%.
- Comm overlap assumes PCIe P2P works on the rented GeForce box; if P2P is disabled the
  all-reduce goes through host memory at ~10 GB/s and d2048 sees ~20% visible.
- TUL is validated dense only; its interaction with prune/carve/route is untested.
- Rented boxes die; an unpushed checkpoint is lost money.
- The pruning FLOP saving (section B) may not materialize; plan against the 30B row
  until the gate passes.
