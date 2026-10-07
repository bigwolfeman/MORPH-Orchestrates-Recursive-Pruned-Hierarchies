# Experiment: why MORPH trains 2x slower than Parcae at the same shape

Status: planned

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
