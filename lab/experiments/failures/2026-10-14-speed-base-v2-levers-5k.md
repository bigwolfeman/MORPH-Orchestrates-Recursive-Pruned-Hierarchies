# Speed base v2 at 5k: diet stack + HC backward on tensor cores + glue folds

Status: failure

## Question

The FLOP diet stack (`nowta_pointer_fast_spdcells_speed_bf16c_cellp2_dec1`, one-block span decoder
+ cell mean 2) passed at 3 seeds (`successes/2026-10-13-flop-diet-5k.md`). Two rounding-only speed
keys merged the same night, each gated on the old parent:

- `model.hc_bwd_fused` (merge 9e0e8517): the HC entry backward's projection term on tensor cores.
  Bench on the parent: 158.50 -> 157.01 ms.
- `model.glue_fold` (merge e4b5d738): compiled glue (span decoder norms + SwiGLU, pointer mass,
  tied-head table, injection term, diagonal injection) and an exact pad + cast. Bench on the
  parent: 160.67 -> 151.53 ms.

Do both keys together on the stack keep CE@6 and K1-K6, and what is the new step time?

## Hypothesis

Both keys change rounding only (each bounded against fp64 in its tests), so the trained model is
the stack's within seed noise. The glue fold's saving shrinks a little on the stack, because the
one-block decoder has half the decoder glue.

## Predictions

Control: the stack's own 3 runs, same seeds (seed 1 3.8999 / +0.0051, seed 2 3.8939 / +0.0096,
seed 3 3.9058 / +0.0044; mean 3.8999 / +0.0064).

| Arm | CE@6 best guess (3-seed mean) | K1-K6 best guess | bench ms, pinned pass 3 |
|---|---|---|---|
| stack (bench only) | | | 130 |
| stack + both keys | 3.900 | +0.0062 | 121 |

Decision rule (Wolfe's paired reading for rounding-only changes, 2026-10-13):

- PASS if the mean seed-paired CE@6 difference is <= +0.003 AND the 95 % t-interval of the three
  seed-paired K1-K6 differences includes zero or lies above it.
- FAIL otherwise, or on a stability miss: `tul/gain_est` above 0.95 after step 60, or grad-norm
  max/median above 30 (one recovered spike inside the 1000-step warmup is reported, not failed).
- The cell knob (M1-M4, M2-M4) is reported, not gated.

If it passes, this config becomes the speed base and the morning report says so.

## Method

Master at the commit adding this file, `git archive` to `abq/tree-lev2`, per-tree
`TORCHINDUCTOR_CACHE_DIR` (a shared cache crashed another tree's gates on 2026-10-14).
`abq/chain-lev.sh`: (1) the key-off byte gate on this tree (45 deterministic graph steps of
`nowta_pointer_fast_spdcells`, FINAL_SHA256 must be 4a73ebf5...). (2) `bench_step.py`, arms stack /
stack + both keys, `model.slot_gain_pass_force=3`, 300 steps, skip 100, 2 repeats. (3) Seeds 1-3,
5000 steps, then `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3`, `stab.keep.py`, and
`cell_count_sweep.py --cells 1,2,4 --cell-extend cyclic`.

## Results

Key-off byte gate on this tree: 4a73ebf5 (match). Bench, 2 repeats, pinned pass 3: stack 129.01 ms
(129.29, 128.73), stack + both keys 122.13 ms (122.14, 122.13), 1.056x. flame5 (M pinned to 2):
119.86 ms, 90.6 TFLOPS achieved (`morph-scratch/perf/flame5/LEDGER.md`).

| Seed | CE@6 | vs stack | K1-K6 | vs stack | M1-M4 | M2-M4 | gain_est max | grad-norm max/median |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.9006 | +0.0007 | +0.0072 | +0.0021 | +0.0004 | -0.0001 | 0.9021 | 7.5 |
| 2 | 3.8998 | +0.0059 | +0.0061 | -0.0035 | +0.0000 | -0.0004 | 0.9116 | 16.5 |
| 3 | 3.9094 | +0.0036 | +0.0055 | +0.0011 | +0.0003 | -0.0000 | 0.9037 | 7.6 |

Mean seed-paired CE@6 difference +0.0034 (sd 0.0026). Mean K1-K6 difference -0.0001, 95 %
t-interval [-0.0075, +0.0073], which includes zero. 3-seed means: CE@6 3.9033, K1-K6 +0.0063.
Against the 9-run pooled control of the parent family (3.9013): +0.0020.

## Verdict

FAIL by the letter: the CE@6 line is +0.003 and the mean paired difference is +0.0034. K1-K6
and stability pass.

What the rule can see here. Both keys change rounding only, and MORPH runs decorrelate within
about 11 steps under any rounding change, so a seed pair shares its data order and nothing else.
A paired difference then has the spread of two independent runs: about sqrt(2) x 0.0036 = 0.005
per seed, 0.003 for a 3-seed mean. +0.0034 is about one such standard deviation. The pooled
reading (+0.0020 against the 9-run family) is inside the line. The stack's own seed 2 (3.8939)
is the second-lowest CE@6 of the 16 diet-family runs of 2026-10-13/14 (the lowest is cell mean 2
seed 3, 3.8883), which inflates the paired differences.
So the data neither shows a cost nor rules out one of about 0.003. Same situation as K1
(2026-10-13), which Wolfe accepted under his paired reading. The keys stay default off and the
speed base stays the stack until Wolfe calls it.

Prediction check: CE@6 3.900 predicted, 3.903 measured; K1-K6 +0.0062 predicted, +0.0063
measured; 121 ms predicted, 122.1 ms measured.

## Updated hypothesis

The keys are rounding-only (each bounded against fp64 in its tests), so the expected cost is zero.
A 3-seed paired test cannot resolve a 0.003 effect on this family. If Wolfe wants a firmer
reading, the efficient test is more seeds on both arms (6 per arm gives a mean-difference sd near
0.002), not a different gate.
