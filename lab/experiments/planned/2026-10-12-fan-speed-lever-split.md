# Fan speed lever split: bf16 carrier, hinge rows and FP8 forward alone at 5k

Status: planned

## Question

The round-2 package (`failures/2026-10-12-fan-speed-round-2.md`) was 14.8 % faster but lost
0.013 CE@6 and 34 % of K1-K6. Which of its three levers costs the loop?

## Hypothesis

The bf16 carrier costs the loop: it rounds the looped state at every store, across up to 8
passes, and depth use lives in small per-pass changes to that state. The hinge rows change only
the regulariser's estimate and cost nothing. FP8 is in between.

## Predictions

Ruler: speed base CE@6 3.8985, K1-K6 +0.0053 (one run); locked recipe 3.8954 / +0.0051. Each
arm is the speed base (`nowta_pointer_fast_spdcells_speed`) plus ONE override, 5000 steps,
seed 1, 480-row sweep. Wall ms/step from the benches: speed base 214.6, package 186.9.

| Arm | Override | CE@6 best guess | K1-K6 best guess | Pass (both): CE@6 <= 3.9005, K1-K6 >= +0.0045 |
|---|---|---|---|---|
| bf16 carrier alone | `model.hc_carrier_bf16=true` | 3.9080 | +0.0038 | fails |
| hinge rows alone | `model.slot_gain_rows=2` | 3.8990 | +0.0052 | passes |
| FP8 forward alone | `model.fp8_ternary_fwd=true` | 3.9010 | +0.0047 | fails CE, passes K1-K6 |

A lever that passes both lines goes into the next package. If no single lever fails, the
package's loss comes from their combination, and the next step is the pair runs.

## Method

Tree `/home/wolfe/morph-scratch/abq/tree-l` (perf/speed-r2 at 1bf79dd5, the round-2 snapshot,
gate already passed). Three 5k runs one at a time under `gpu.lock`, WANDB offline, each followed
by `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` and `abq/r2bench/stab.keep.py`.
