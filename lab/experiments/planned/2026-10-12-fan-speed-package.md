# Fan speed package: the inexact speed levers at 5k

Status: planned

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
