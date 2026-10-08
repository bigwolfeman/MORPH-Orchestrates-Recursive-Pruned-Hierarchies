# lxtul_pointer_fast at 5k against the eager 5k runs

Status: success

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

## Results

Artifacts: [`results/2026-10-08-lxtul-pointer-fast-5k/`](../results/2026-10-08-lxtul-pointer-fast-5k/).

| run | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | spandec K1-K6 | final val |
|---|---|---|---|---|---|
| eager winner (2026-10-06) | 3.9435 | 3.9280 | +0.0155 | +0.0133 | |
| eager rerun (2026-10-08) | 3.9281 | 3.9187 | +0.0095 | +0.0113 | 3.9085 |
| fast recipe (this run) | 3.9353 | 3.9248 | +0.0104 [+0.0098, +0.0110] | +0.0099 | 3.9161 |

- CE@6 3.9248: inside 3.914-3.933. Held.
- K1-K6 +0.0104: at or above +0.0075. Held.
- Final val 3.9161: 0.0076 from 3.9085, inside 0.010. Held.
- Speed: 24.1-24.5k tok/s logged by the trainer late in the run; 35 minutes wall clock from
  launch to the final checkpoint (compile, evals and checkpoints included), against about 70
  for the eager 5k run.
- One val reading moved (4500: 3.9533, 4750: 4.1047, final 3.9161), recovered by the end.

## Verdict

Success: all three quality predictions held. The fast recipe lands inside the eager recipe's
run-to-run spread on CE and loop contribution at about twice its speed.

## Updated hypothesis

The combined speed keys do not compound into a quality change over 5000 steps from init. One run
per arm: the spread between two eager runs (0.006 K1-K6, 0.009 CE@6) is the resolution here.
