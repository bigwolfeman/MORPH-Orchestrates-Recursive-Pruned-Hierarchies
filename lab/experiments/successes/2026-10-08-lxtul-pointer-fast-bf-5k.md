# lxtul_pointer_fast with the block-glue keys at 5k

Status: success

## Question

Adding `hc_region_fused`, `cca_prologue_tiled` and `inject_fold` (rounding only; fixed-weight
CE@6 +4.2e-5, K1-K6 -0.42 %) to the fast recipe: does a 5k run from init still land inside the
eager recipe's spread?

## Hypothesis

Rounding-level changes do not compound over 5000 steps beyond the run-to-run spread.

## Predictions

References: eager winner 3.9280 / +0.0155, eager rerun 3.9187 / +0.0095 (val 3.9085), fast recipe
without these keys 3.9248 / +0.0104 (val 3.9161).
- CE@6 between 3.914 and 3.933; best guess 3.923.
- K1-K6 at or above +0.0075; best guess +0.011.
- Final val within 0.010 of 3.9085.
- Wall clock launch to final checkpoint under 33 minutes (the run without them took 35).

## Method

`--config-name lxtul_pointer_fast` at the commit of this file (the config now carries the three
keys), `wandb.name=lxtul-pointer-fast-bf-5k`, WANDB offline. Readout `core_depth_sweep.py --depths
1,6 --rows 480 --batch 3` on the final checkpoint.

## Results

Artifacts: [`results/2026-10-08-lxtul-pointer-fast-bf-5k/`](../results/2026-10-08-lxtul-pointer-fast-bf-5k/).

| run | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | spandec K1-K6 | final val | wall clock |
|---|---|---|---|---|---|---|
| eager winner | 3.9435 | 3.9280 | +0.0155 | +0.0133 | | |
| eager rerun | 3.9281 | 3.9187 | +0.0095 | +0.0113 | 3.9085 | 65 min |
| fast, no block-glue keys | 3.9353 | 3.9248 | +0.0104 | +0.0099 | 3.9161 | 35 min |
| fast with block-glue keys (this run) | 3.9319 | 3.9229 | +0.0091 [+0.0084, +0.0097] | +0.0132 | 3.9129 | 31.3 min |

- CE@6 3.9229 inside 3.914-3.933. Held.
- K1-K6 +0.0091 at or above +0.0075. Held.
- Final val 3.9129, 0.0044 from 3.9085. Held.
- Wall clock 31.3 minutes (09:43:27 -> 10:14:46), under 33. Held. Trainer rate late in the run
  29.8k tok/s.
- The val readings at 4500 / 4750 (3.9515 / 4.1026) repeat the other fast run's (3.9533 / 4.1047):
  they follow the eval batches, not the run.

## Verdict

Success: all four predictions held.

## Updated hypothesis

The final fast recipe trains inside the eager recipe's spread at 2.1x its wall-clock speed for a
5k run (2.3x per step in the bench). One run per arm; resolution is the eager run-to-run spread.
