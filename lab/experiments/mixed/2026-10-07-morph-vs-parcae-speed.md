# Experiment: why MORPH trains 2x slower than Parcae at the same shape

Status: mixed

Date: 2026-10-07 08:09 CDT (written while ladder rung L0 runs; no rung result seen).
GPU budget: 3 hours from 08:02 (Wolfe).

## Question

At d 1024, 4/6/4 layers, batch 6, seq 1024, on the same rows and GPU, the MORPH winner arm
(`lxtul_pointer`) trains at 11.8k tok/s and the Parcae testbed's LXTUL + pointer at 22.2k
(runner means, steps >= 200). Plain MORPH 10.3k, plain Parcae 21.8k. Where does the time go?

## Measured before this file (08:02-08:08, not predictions)

- FLOPs per step (FlopCounterMode, eager, batch 3 x 2): MORPH LXTUL + pointer 22.35 TFLOP;
  Parcae LXTUL + pointer 22.4 TFLOP model + 5.5 TFLOP Muon. The work is the same.
- torch.profiler, 4 steps after 40 (compile on, production configs), ms per step:
  MORPH 594 wall, 451 GPU busy, 31.3k kernels; Parcae 275 wall, 267 busy, 8.8k kernels.
  GEMM 148 vs 160. Copy/cast 94 vs 17, aten elementwise 71 vs 18, custom Triton (HC, CCA)
  67 vs 0, idle 143 vs 9 (115 of the 143 in gaps under 50 us: launch-bound).

## Hypothesis

The gap is execution, not work: MORPH runs most non-GEMM code eager (only the MLP is
compiled), carries a 4-stream fp32 HC residual, and casts it to bf16 at every projection.
That multiplies small memory-bound kernels, and the host cannot launch them fast enough.

## Predictions (cumulative ladder from `lxtul_pointer`, 420-step bench, ms/step from steps 199-419)

| Rung | Change | Predicted saving |
|---|---|---|
| L1 | probes off (`grad_probe_every 0`, `loop_cot_probe false`) | 15-35 ms |
| L2 | + `compile_blocks`, static core | 60-110 ms |
| L3 | + ternary off | 10-25 ms |
| L4 | + `hc_streams 1` | 50-120 ms (may refuse to build on the slot loop) |
| L5 | + dropout 0 | 5-15 ms |

- After L5 MORPH is still at least 40 ms slower than Parcae's 277 ms: CCA, the masked eager
  attention, the fp32 carrier and AdEMAMix remain.
- No single rung is worth more than half the gap (about 120 ms).

## Method

`/home/wolfe/morph-scratch/perf/r1007/` (private): `flops_patch.py`, `prof_generic.py`,
`kcat.py` (kernel categories), `bench.sh` (420 steps, eval and checkpoints off, GPU lock,
CPU quiet). Worktree `/home/wolfe/morph-wt-lsel` at `18f3676d`. Speed only; no loss claim.

## Method amendments

- 2026-10-07 08:45 CDT, after the ladder: added three rungs not in the prereg, each on top of
  L1 + L2 (no prediction was written for them): O1 the existing opt-in speed keys
  (`model.ce_softmax_kernel`, `model.ce_compact_rows`); O2 O1 + `model.hc_use_kernel false`;
  O3 O2 + `model.tg_scoped_kernels false`. Also added: a 12-step `TORCH_LOGS=graph_breaks`
  run of L2, and profiles of L2, L5, O1, O3.

## Results

Artifacts: [`results/2026-10-07-morph-vs-parcae-speed/`](../results/2026-10-07-morph-vs-parcae-speed/)
(the 300 MB chrome traces stay in private scratch, `TRACES.txt`). Profiled numbers are 4 steps
after 40 and run about 15 % slow on MORPH (the profiler adds host time per launch).

**Work is equal.** Model FLOPs per step at batch 6: MORPH 22.35 TFLOP, Parcae 22.4 TFLOP
(+5.5 TFLOP of Muon Newton-Schulz). MORPH's GEMMs run at about 151 TFLOPS (148 ms per step);
the GEMMs are not the problem.

**The ladder** (cumulative from `lxtul_pointer`, ms per step over steps 200-400):

| Rung | ms/step | tok/s | saving | predicted | held? |
|---|---|---|---|---|---|
| L0 production | 516.2 | 11903 | | | |
| L1 probes off | 513.6 | 11963 | 2.6 | 15-35 | no |
| L2 + compile_blocks, static core | 482.1 | 12745 | 31.5 | 60-110 | no |
| L3 + ternary off | 468.3 | 13120 | 13.8 | 10-25 | yes |
| L4 + hc_streams 1 | 434.5 | 14140 | 33.8 | 50-120 | no |
| L5 + dropout 0 | 428.5 | 14338 | 6.0 | 5-15 | yes |
| O1 = L2 + CE kernel + compact rows | 444.7 | 13815 | 71.5 vs L0 | | |
| O2 = O1 + HC eager references | 492.8 | 12469 | -48.1 vs O1 | | |
| O3 = O2 + all kernels eager | 519.5 | 11827 | -74.8 vs O1 | | |

Parcae LXTUL + pointer: 276.8 ms (22.2k tok/s, its runner mean). After L5 MORPH is still
151.7 ms slower (prediction "at least 40" held); no rung saved more than 120 ms (held).

**The step is host-serial.** CPU busy per thread (union of op intervals, profiled):

| Trace | forward thread | autograd thread | sum | wall | GPU busy | kernels/step |
|---|---|---|---|---|---|---|
| MORPH L0 | 270.6 | 301.5 | 572 | 594 | 451 | 31.3k |
| MORPH L5 | 194.6 | 291.9 | 487 | 497 | 348 | 24.9k |
| MORPH O1 | 218.0 | 264.1 | 482 | 492 | 388 | 22.8k |
| Parcae LXTUL + ptr | 161.4 | 109.7 | 271 | 275 | 267 | 8.8k |
| MORPH plain | 178.5 | 391.5 | 570 | 589 | 566 | 27.9k |
| Parcae plain | 171.1 | 112.1 | 283 | 285 | 282 | 4.8k |

The forward and the backward run one after the other, and on every MORPH LXTUL trace their
CPU time sums to the wall clock while the GPU idles 100-150 ms. The step costs about 20 us of
host time per kernel launch, and removing launches is what buys time: L0 to L5 removed 6.4k
launches and 87.7 ms (13.7 us each). That is why the GPU-work rungs (HC, ternary) bought less
than predicted. Plain MORPH is limited on both sides (GPU 96 % busy and CPU 570 ms).

**Where the GPU time and the launches are** (L0, forward + backward charged to the forward
range through autograd sequence numbers, ms per step):

| Region | MORPH | Parcae | MORPH launches | Parcae launches |
|---|---|---|---|---|
| core loop + gain hinge (blocks + glue) | ~186 | ~87 | 20.7k | 5.3k |
| prelude (live) | ~54 | ~11-17 | ~1.6k | ~0.2k |
| EMA twin prelude (no grad) | 20.8 | ~5 | 0.9k | ~0.2k |
| coda | ~49 | 19.5 | ~1.6k | 0.2k |
| span decoder + its CE | 61.6 | 26.0 | 0.6k | 0.2k |
| main CE + pointer (+ embed on Parcae) | 50.4 | 53.3 | 0.7k | 1.8k |
| optimizer + clip | 7.5 | 50.5 | 0.5k | 0.3k |

By sublayer, summed over all MORPH regions: MLP + its HC wrapper 120.7 ms (GEMM 64.6; HC
Triton 21.3, inductor 13.4, elementwise 11.4, casts 7.9); attention 99.5 ms (GEMM 18.2 +
attention kernel 12.7; CCA/conv Triton 30.0, casts 23.2, elementwise 7.2); HC wrapper on
attention 30.3; eager RMSNorm 18.4. The core attention alone launches 10.2k kernels per step
(about 170 per call, forward + backward, mean 5.6 us): casts, strided copies, fills, cats,
fp32 grad adds, and an `arange` + boolean mask rebuild on every call
(`core_attention_kernels.txt`).

**Compile is cut by design.** With `compile_blocks` a MORPHBlock breaks at every hand-written
Triton dispatcher wrapped in `torch.compiler.disable`: HC post 18, HC pre-map 15, segment
causal conv 10, CCA conv 8, CCA prologue 4 (60 breaks in 12 steps). So inductor fuses small
pieces (5.2k inductor kernels per step at 4.6 us mean). Swapping the kernels for their torch
references (O2, O3) removes the breaks (60 -> 10) and runs compiled code (10.4k inductor
kernels), but the references are a different, costlier algorithm (GEMM 141 -> 225 ms, 6.7k
GEMMs, attention off SDPA) and the step got slower.

## Verdict

Mixed. The hypothesis's core held: the work is equal and the gap is execution, not
architecture size. Three of five rung predictions missed low, because I modelled the gap as
GPU memory traffic. It is mainly host launch rate: the MORPH LXTUL step is host-serial at
about 20 us per launch, with 3.5x Parcae's launches.

## Updated hypothesis

MORPH's LXTUL step time is set by its kernel count. The count comes from (1) the slot loop's
eager attention (TG-restricted, mask rebuilt per call, fp32 <-> bf16 casts) at 256-cell
shapes, where each kernel is a few microseconds; (2) the HC and CCA Triton dispatchers that
cut every compiled block into fragments; (3) eager RMSNorm and eager CE heads. Predicted
levers, in order: one captured graph per loop pass (CUDA graphs, the 2026-10-04 note's item
1); register the HC / CCA / conv Triton kernels as `torch.library` custom ops so a block
compiles whole; a fused kernel for the strict-mask attention (Parcae uses `flex_attention`
for the same masks). The GPU-side floor today is O1's 388 ms busy (profiled); reaching
Parcae's 277 ms also needs the casts, HC traffic and CCA work cut.
