# bf16 carrier against matched controls

Status: planned

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
