# lxtul_pointer_fast at 5k against the eager 5k runs

Status: planned

## Question

Trained from step 0 for 5000 steps, does the fast recipe (`morph/configs/lxtul_pointer_fast.yaml`,
23.0k tok/s) land where the eager production recipe lands on CE and loop contribution?

## Hypothesis

Every key in it passed its own check (byte gates, fp64 parity, fixed-weight readouts, paired
1000-step continuations). Over 5000 steps from init the effects do not compound beyond the
run-to-run spread of the eager recipe.

## Predictions

Eager 5k runs (same rows, same readout): winner CE@6 3.9280 / K1-K6 +0.0155, rerun 3.9187 / +0.0095.
- CE@6 (480 rows) between 3.914 and 3.933 (the eager pair's range widened by 0.005); best guess 3.924.
- K1-K6 (ce_tokens) at or above +0.0075; best guess +0.011.
- Trainer final val within 0.010 of the eager rerun's 3.9085.
- Wall clock: about 5000 x 0.267 s plus evals, under 30 minutes of training.

## Method

`--config-name lxtul_pointer_fast` at the commit of this file, seed 1 (default), WANDB offline,
`wandb.name=lxtul-pointer-fast-5k`. Readout `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3`
on the final checkpoint.
