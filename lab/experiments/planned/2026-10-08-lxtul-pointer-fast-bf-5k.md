# lxtul_pointer_fast with the block-glue keys at 5k

Status: planned

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
