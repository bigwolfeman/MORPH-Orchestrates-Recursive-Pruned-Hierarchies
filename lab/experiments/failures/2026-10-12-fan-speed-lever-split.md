# Fan speed lever split: bf16 carrier, hinge rows and FP8 forward alone at 5k

Status: failure

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

## Results

480 rows, seed 1, one run per arm. Log: `chain-split.txt`. Trainer tok/s is the fixed wall rate
(the speed base's 5k run predates the fix, so its rate is not comparable; use the benches).

| Arm | CE@1 | CE@6 | Pass CE@6 <= 3.9005 | K1-K6 [95 % CI] | Pass >= +0.0045 | grad-norm max/median | gain_est 1000-5000 |
|---|---|---|---|---|---|---|---|
| speed base (ruler) | 3.9038 | 3.8985 | | +0.0053 [+0.0049, +0.0058] | | 7.9 | 0.884-0.902 |
| bf16 carrier alone | 3.9048 | 3.8996 | yes | +0.0052 [+0.0047, +0.0057] | yes | 7.4 | 0.889-0.902 |
| hinge rows alone | 3.9019 | 3.8975 | yes | +0.0044 [+0.0039, +0.0048] | no (by 0.0001) | 8.4 | 0.888-0.912 |
| FP8 forward alone | 3.9081 | 3.9057 | no | +0.0024 [+0.0021, +0.0027] | no | 8.1 | 0.890-0.903 |

Cell cos -0.27 on all three (diagnostic).

## Verdict

Failure: two of three predictions missed. The bf16 carrier, predicted to fail, holds both CE@6
and K1-K6 at the speed base's values. FP8 forward is the lever that costs the loop: K1-K6 falls
55 % and CE@6 rises 0.007, which accounts for the round-2 package's loss. The hinge rows keep CE@6
but read K1-K6 0.0001 under the line (-17 % vs the ruler; the CIs overlap).

Disposition: the bf16 carrier joins the speed base (`nowta_pointer_fast_spdcells_speed_bf16c`).
FP8 forward is rejected for this recipe. The hinge rows stay off; a second run would tell a
-17 % effect from draw noise, and the lever is worth 3 % of the step.

## Updated hypothesis

Depth use rides on small per-pass changes to the cells. FP8 rounds every MLP input of every pass
by about 2.7 % relative (3 mantissa bits, rowwise scale), which is the same order as those
changes, so it erases them; a bf16 store (8 mantissa bits, about 0.4 %) does not. A lever that
adds input noise of more than about 1 % inside the loop should be expected to cost K1-K6.
FP8 outside the loop (prelude, coda) is untested and worth at most about 1 % of the step.
