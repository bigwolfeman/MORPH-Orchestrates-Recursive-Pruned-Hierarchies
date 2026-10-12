# bf16-carrier speed base: second 5k run

Status: failure

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

## Results

Log `chain-bf16b.txt`. Eval beside the pool: peak 5.75 GiB, 6.24 GiB free after eval.

| Quantity | First bf16 run (lever split) | This run | Pass if | Held |
|---|---|---|---|---|
| CE@1 | 3.9048 | 3.9118 | | |
| CE@6 | 3.8996 | 3.9060 | <= 3.9005 | no (+0.0055) |
| K1-K6 [95 % CI] | +0.0052 [+0.0047, +0.0057] | +0.0058 [+0.0053, +0.0062] | >= +0.0045 | yes |
| grad-norm max / median | 7.4 | 6.6 | <= 30 | yes |
| `tul/gain_est` 1000-5000 | 0.889-0.902 | 0.889-0.902 | 0.85-0.95 | yes |
| wall time, train start to exit | (not comparable) | 1196 s | < 1470 s | yes |
| cell cos | -0.27 | -0.29 | diagnostic | |

## Verdict

Failure on CE@6. Loop contribution, stability and wall time held; the second bf16 draw sits
0.0064 CE@6 above the first. The pass line (locked recipe + 0.005) was set from three runs that
were near-copies of one trajectory; how far two FRESH trajectories of one config land apart at
5k is not measured on this recipe, so this run cannot tell a bf16 cost from draw spread. Per the
prereg the bf16 carrier leaves the speed base until the next experiment decides it.

## Updated hypothesis

The bf16 store costs no loop contribution (K1-K6 +0.0052, +0.0058 against +0.0053). Whether it
costs CE is open; the next run measures the draw spread with matched controls:
`lab/experiments/planned/2026-10-12-bf16-carrier-matched-controls.md`.
