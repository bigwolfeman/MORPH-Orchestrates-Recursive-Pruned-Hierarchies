# Fan speed package: the inexact speed levers at 5k

Status: failure

## Question

The 2026-10-11 speed work landed three levers that are not bit-identical: the packed span decoder
(`model.spandec_packed`, grad rel 1.7e-5 on the fixture), the LM head on the rows a loss reads
(`model.lmhead_row_cap`, C2, reduction order), and the compiled loop norms
(`model.loop_glue_compile`, step-0 grad norm rel 6.1e-4). A fourth lever changes behaviour on
purpose: the span decoder's target cut to the first 8 tokens of the next span
(`tul.spandec_max_tokens: 8`; the cell lens read the decoders using about 2 tokens). Do the fan
recipe's quality and loop contribution hold with them on?

## Hypothesis

The three inexact levers compute the same function up to rounding, so the 5k result sits inside the
locked recipe's run spread. The 8-token target keeps most of the decoder's value, because the cells
carry about 2 tokens of the next span, and costs at most a little CE.

## Predictions

Locked recipe (`nowta_pointer_fast_spdcells`, 3 runs, seed 1, 480 rows): CE@6 3.8949 / 3.8953 /
3.8954, K1-K6 +0.0052 / +0.0060 / +0.0051, cell cos -0.28. The fan with no decoder: CE@6 3.8993.

| Arm | CE@6 best guess | Pass if | K1-K6 best guess | Pass if | tok/s vs locked (same tree, same bench) |
|---|---|---|---|---|---|
| speed package | 3.8953 | <= 3.8975 | +0.0054 | >= +0.0045 | +8 % or more |
| speed package + 8-token decoder | 3.8965 | <= 3.8975 | +0.0048 | >= +0.0045 | +15 % or more |

Cell cos is a diagnostic only (predicted below +0.3 on both arms). If the 8-token arm fails CE but
passes K1-K6, the 8-token target is rejected and the span cap stays 32.

## Method

Integration branch `perf/speed-pkg` (master 95a2c6e9 + perf/spandec-packed 493f86eb +
perf/hc-kernels 1a184900 + perf/loop-glue 22da0210), frozen as a snapshot in
`/home/wolfe/morph-scratch/abq/tree-k`. Step 1, integration gate: 45 steps of
`nowta_pointer_fast_spdcells` with `training.deterministic=true training.graph_step=true` and every
new key off; the FINAL_SHA256 must equal master's `4a73ebf5...bd70d7`, or the arms do not run.
Step 2: configs `nowta_pointer_fast_spdcells_speed` and `..._speed_sd8`, 5000 steps, seed 1,
WANDB offline, one at a time under `gpu.lock`, then the 480-row depth sweep
(`lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480`). Speed is read as the trainer rate
against the locked recipe's 29,209 tok/s on tree-j (same runner, same 5k length).

## Results

Integration gate (45 deterministic graph-step steps, every new key off): FINAL_SHA256
`4a73ebf5...bd70d7`, equal to master. The first launch never ran the gate (no
`CUBLAS_WORKSPACE_CONFIG`, the trainer refused at startup); the relaunch added it and passed.

480 rows, seed 1, `core_depth_sweep.py`. The tok/s column is the median of the 24 paired
trainer-rate samples (same step, same rows) divided by the locked recipe's sample at that step.
The locked recipe's 29,209 in the prereg is its LAST sample only; its harmonic mean over the run
is 30,145.

| Arm | CE@1 | CE@6 | Pass CE@6 <= 3.8975 | K1-K6 [95 % CI] | Pass >= +0.0045 | tok/s ratio (min, max) | cell cos |
|---|---|---|---|---|---|---|---|
| locked recipe (master, tree-j) | 3.9005 | 3.8954 | ruler | +0.0051 [+0.0047, +0.0055] | ruler | 1.000 | -0.28 |
| speed package | 3.9038 | 3.8985 | no (+0.0010) | +0.0053 [+0.0049, +0.0058] | yes | 1.083 (1.068, 1.103) | -0.08 |
| speed package + 8-token decoder | 3.9265 | 3.9213 | no (+0.0238) | +0.0051 [+0.0047, +0.0056] | yes | 1.178 (1.118, 1.223) | -0.24 |

Harmonic-mean trainer rate: 30,145 / 32,585 / 35,046 tok/s. Raw samples: `trainer-tok-s.txt`.
Final trainer val (a different reader than CE@6): 3.8937 / 3.8894 / 3.9156.

## Verdict

Failure on the CE line for both arms.

- Speed package: K1-K6 and speed held (+8.3 %, the prediction was "+8 % or more"); CE@6 missed by
  0.0010. One run cannot split the levers' bias from draw variance. The three locked-recipe runs
  that agree to 0.0005 ran the same levers on seed 1, so they are near-copies of one trajectory,
  not independent draws; any inexact lever starts a new trajectory, and the spread of new
  trajectories at 5k is not measured on this recipe. Cell cos moved -0.28 to -0.08 (a diagnostic,
  inside the predicted "below +0.3").
- 8-token decoder: CE@6 +0.026 over the locked recipe, worse than the fan with no decoder
  (3.8993). By the prereg's rule the 8-token target is rejected and the span cap stays 32. The
  decoder's value is not in the first 8 tokens of the next span, although the lens read the cells
  carrying about 2 of them.

Disposition (Wolfe's speed-path rule, 2026-10-10: a speed path may be inexact if loop
contribution stays within 5 % of baseline): K1-K6 +0.0053 vs +0.0054 mean of the three ruler
runs, -2 %. The code merges to master with every key default off (the gate proves the off path
byte-identical); `nowta_pointer_fast_spdcells_speed.yaml` is the base for further speed work,
with this CE caveat attached. The locked recipe config is unchanged.

## Updated hypothesis

The three levers compute the same function up to rounding, and the +0.001 CE@6 is a trajectory
draw, not a bias. Untested: it needs two runs of the speed package, or one run per lever. The
8-token decoder hypothesis is refuted: cutting the target to 8 tokens removes most of what the
decoder trains into the cells.
