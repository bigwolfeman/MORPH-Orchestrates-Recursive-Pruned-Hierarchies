# bf16-carrier speed base: second 5k run

Status: planned

## Question

The bf16-carrier speed base (`nowta_pointer_fast_spdcells_speed_bf16c`) held quality on one 5k
run (lever split: CE@6 3.8996, K1-K6 +0.0052). Does a second run hold it? This run also carries
the two exact-for-training eval changes (graphs kept across eval; the probe block on every 4th
eval with the RNG forked around eval), so its training stream is a new draw, not a copy of the
first run.

## Hypothesis

The bf16 store (8 mantissa bits, about 0.4 % per store) is below the size of the loop's per-pass
changes, so a second draw lands inside the speed base's range.

## Predictions

| Quantity | Best guess | Pass if |
|---|---|---|
| CE@6 (480 rows) | 3.8995 | <= 3.9005 |
| K1-K6 | +0.0051 | >= +0.0045 |
| grad-norm max / median, steps >= 200 | 7.5 | <= 30 |
| `tul/gain_est` steps 1000-5000 | 0.885-0.905 | inside 0.85-0.95 |
| wall time of the 5000-step run (start to final eval) | about 1250 s | below the round-2 run's 1470 s |

Cell cos is a diagnostic. If CE@6 or K1-K6 fails, the bf16 carrier leaves the speed base until
a third run decides it.

## Method

Master at the commit that adds this file, snapshot `/home/wolfe/morph-scratch/abq/tree-n`.
`nowta_pointer_fast_spdcells_speed_bf16c`, 5000 steps, seed 1, WANDB offline, under
`gpu.lock`; then `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` and
`abq/r2bench/stab.keep.py`. Wall time from the chain log's timestamps.
