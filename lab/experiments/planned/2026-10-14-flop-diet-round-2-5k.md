# FLOP diet round 2 at 5k: hinge cadence on the stack, and a half-slot decoder

Status: planned

## Question

Round 1 (`successes/2026-10-13-flop-diet-5k.md`) tests a one-block span decoder and a Poisson(2.0)
cell count. Two new keys (merge 2957ade6, both default off, key-off byte gate 4a73ebf5) cut more:

- hinge cadence: `model.slot_gain_every: 4`. The one-pass gain hinge (1.069 TFLOP, 9.5 ms
  kernel-sum on a side stream, flame4) runs on every 4th step with 4x its lambdas.
- half-slot decoder: `tul.spandec_slot_keep: 0.5`. The two-block packed decoder trains on a random
  token-budget half of each row's slots, each with its full next-span target.

Do these keep CE@6 and the loop's depth use at 5k?

## Hypothesis

The hinge holds the map's gain near 0.89; a push on 1 step in 4 at 4x strength holds the same mean
but lets the gain drift for 3 steps. The map moves slowly (gain_est spread 0.88-0.90 over a run), so
the cadence costs nothing measurable. A half-slot decoder gives the same expected gradient to the
cells with more variance; at 2 blocks it keeps the decoder's capacity, so it may cost less than the
one-block decoder.

## Predictions

Control: as in round 1, the 9 filed runs pooled (CE@6 3.9013, sd 0.0036; K1-K6 +0.00519, sd 0.0011).

| Arm | Config | CE@6 best guess (2-seed mean) | K1-K6 best guess | bench ms, pinned pass 3 |
|---|---|---|---|---|
| diet stack + hinge every 4 | `..._cellp2_dec1_ge4` | 3.902 | +0.0048 | 133 |
| cell mean 2 + half-slot decoder | `..._cellp2_keep05` | 3.903 | +0.0048 | 136 |

Round 1's bench on the same day, for reference: speed base 157.4, diet stack 139.6 ms.

Decision rule per arm, as in round 1: PASS if the 2-seed mean CE@6 minus 3.9013 is <= +0.003 AND
the mean K1-K6 is >= +0.0035; GREY if CE@6 is in (+0.003, +0.006]; FAIL otherwise. Stability:
`tul/gain_est` above 0.95 after step 60 is a FAIL. Grad-norm max/median is reported against the
controls' 6.7-8.2; a single recovered spike during the 1000-step warmup is noted, not failed
(round 1's stack seed 1 had one at step 240, max/median 45.5, loss and gain unmoved).

## Method

Master at the commit adding this file, `git archive` to `abq/tree-diet2`. `abq/chain-diet2.sh`
starts when `abq/chain-diet.sh` logs `done`: (1) `bench_step.py` arms base / stack / ge4 / keep05,
cellp configs, `model.slot_gain_pass_force=3` for the base, stack and keep05 arms (the ge4 arm
cannot pin a pass on the hinge's off steps, so it runs with the pin too, which pins the on steps),
300 steps, skip 100, 2 repeats. (2) Each arm on seeds 1 and 2, 5000 steps, then
`core_depth_sweep.py --depths 1,6 --rows 480 --batch 3`, `stab.keep.py`, and
`cell_count_sweep.py --cells 1,2,4 --cell-extend cyclic`.

## Results

## Verdict

## Updated hypothesis
