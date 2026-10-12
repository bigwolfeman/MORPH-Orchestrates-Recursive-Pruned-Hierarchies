# bf16 carrier against matched controls

Status: success

## Question

Two bf16-carrier 5k runs read CE@6 3.8996 and 3.9060. Is the 0.0064 spread draw noise, and does
the bf16 carrier cost CE against controls run under the same eval regime?

## Hypothesis

Fresh 5k draws of this recipe spread by several thousandths of CE@6, and the bf16 carrier's
mean sits within 0.003 of the fp32-carrier mean.

## Predictions

Three new runs, all with `training.eval_keep_graphs=true training.eval_diag_every=4` (the RNG is
forked around eval, so each run's training stream depends only on its seed and config):

| Arm | Seed | CE@6 best guess | K1-K6 best guess |
|---|---|---|---|
| fp32 carrier control | 1 | 3.9010 | +0.0054 |
| fp32 carrier control | 2 | 3.9010 | +0.0054 |
| bf16 carrier | 2 | 3.9025 | +0.0055 |

With the existing bf16 seed-1 run under the same regime (3.9060 / +0.0058):

- Draw spread: |control seed 1 - control seed 2| CE@6. Best guess 0.004.
- Decision: the bf16 carrier rejoins the speed base if the bf16 mean CE@6 (2 runs, seeds 1 and
  2) minus the control mean CE@6 (2 runs, seeds 1 and 2) is <= 0.003, AND the bf16 mean K1-K6 is
  >= 95 % of the control mean. Otherwise it stays out, and the note says by how much.
- Stability on every run: grad-norm max/median <= 30, `tul/gain_est` inside 0.85-0.95.

## Method

Master at the commit adding this file, snapshot `/home/wolfe/morph-scratch/abq/tree-n` (same
code as the second bf16 run). Configs: control = `nowta_pointer_fast_spdcells_speed` + the two
eval overrides + `training.seed`; bf16 = `nowta_pointer_fast_spdcells_speed_bf16c`
(it carries the eval settings) + `training.seed=2`. 5000 steps each, one at a time under
`gpu.lock`, then the 480-row sweep and `abq/r2bench/stab.keep.py`.

## Results

All four runs under the same eval regime (graphs kept, probe block every 4th eval, RNG forked
around eval). Log `chain-bf16ctl.txt`; the bf16 seed-1 run is `failures/2026-10-12-bf16-carrier-second-run.md`.

| Arm | Seed | CE@1 | CE@6 | K1-K6 [95 % CI] | grad-norm max/median | gain_est 1000-5000 |
|---|---|---|---|---|---|---|
| fp32 carrier | 1 | 3.9102 | 3.9061 | +0.0041 [+0.0037, +0.0045] | 7.1 | 0.890-0.901 |
| fp32 carrier | 2 | 3.9093 | 3.9029 | +0.0064 [+0.0059, +0.0068] | 8.1 | 0.891-0.902 |
| bf16 carrier | 1 | 3.9118 | 3.9060 | +0.0058 [+0.0053, +0.0062] | 6.6 | 0.889-0.902 |
| bf16 carrier | 2 | 3.9093 | 3.9040 | +0.0052 [+0.0048, +0.0057] | 7.8 | 0.890-0.902 |

- Means: fp32 CE@6 3.9045, K1-K6 +0.00525; bf16 CE@6 3.9050, K1-K6 +0.00550.
- bf16 - fp32 CE@6: +0.0005 (rule <= 0.003: held). bf16 K1-K6 / fp32: 105 % (rule >= 95 %: held).
- Draw spread between the two fp32 seeds: CE@6 0.0032 (guess 0.004), K1-K6 0.0023.
- Stability held on every run.

## Verdict

Success: the predictions held. The bf16 carrier rejoins the speed base. Two readings change how
every later 5k check here must be read:

1. K1-K6 spreads 0.0023 between two seeds of one config, 44 % of its mean. A single run cannot
   resolve the 5 % loop-contribution rule; it needs about 2 seeds per arm and a rule on means.
2. All four runs here read CE@6 3.903-3.906, while the six earlier seed-1 runs of the fan
   recipe and its speed variants (eval regime: graphs released, probe block every eval, no RNG
   fork) read 3.8949-3.8996. Seed 1 is in both groups. The gap (about 0.007) is either a draw
   effect of the old regime's seed-1 family or a cost of the new eval regime. Next:
   `lab/experiments/planned/2026-10-12-eval-regime-check.md`.

## Updated hypothesis

The bf16-stored carrier is free at 5k (CE@6 +0.0005, K1-K6 +5 %, both inside the draw spread).
