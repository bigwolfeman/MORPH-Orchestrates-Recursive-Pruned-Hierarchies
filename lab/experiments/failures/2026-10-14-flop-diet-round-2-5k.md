# FLOP diet round 2 at 5k: hinge cadence on the stack, and a half-slot decoder

Status: failure

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

Amendment 2026-10-14 05:20. Hinge every 4 reads GREY on seeds 1-2, so it gets seed 3 as the rule
says (`abq/chain-diet4.sh`, same tree, with a per-tree Inductor cache dir; the shared cache
crashed another tree's gates the same night). The half-slot decoder arm fails by the letter and
gets no third seed.

## Results

Bench (median of 2 repeats, pinned pass 3, 2026-10-14 03:42): speed base 146.29 ms, stack 129.96,
stack + hinge every 4 123.91, cell mean 2 + half-slot decoder 123.05.

| Arm | Seed | CE@6 | K1-K6 | M1-M4 | M2-M4 | gain_est max | grad-norm max/median |
|---|---|---|---|---|---|---|---|
| stack + hinge every 4 | 1 | 3.9093 | +0.0118 | +0.0006 | -0.0004 | 0.9062 | 7.0 |
| stack + hinge every 4 | 2 | 3.9028 | +0.0089 | -0.0000 | -0.0001 | 0.9217 | 7.9 |
| cell mean 2 + half-slot decoder | 1 | 3.9083 | +0.0037 | +0.0007 | -0.0002 | 0.9050 | 9.5 |
| cell mean 2 + half-slot decoder | 2 | 3.9067 | +0.0061 | -0.0001 | -0.0004 | 0.9035 | 7.5 |

2-seed means against the pooled control (3.9013, +0.00519): hinge every 4 CE@6 +0.0048 (GREY),
K1-K6 +0.0104 (200 %); half-slot decoder +0.0062 (FAIL by the letter), +0.0049. Seed-paired
against the stack (round 1, same seeds): hinge every 4 costs +0.0094 and +0.0089 CE@6, and moves
K1-K6 by +0.0067 and -0.0007. The half-slot decoder arm reads +0.0084 and +0.0128 against the
stack.

Seed 3 of hinge every 4: CE@6 3.9009, K1-K6 +0.0042, M1-M4 -0.0001, M2-M4 -0.0003, gain_est max
0.9098, grad-norm max/median 8.2. Against the stack's seed 3 (3.9058): -0.0049.

| Arm | Seeds | CE@6 mean (vs 3.9013) | K1-K6 mean | paired vs stack, CE@6 | bench ms |
|---|---|---|---|---|---|
| stack + hinge every 4 | 1-3 | 3.9043 (+0.0030) | +0.0083 (160 %) | +0.0045 (+0.0094, +0.0089, -0.0049) | 123.9 |
| cell mean 2 + half-slot decoder | 1-2 | 3.9075 (+0.0062) | +0.0049 (94 %) | +0.0106 | 123.1 |

## Verdict

Neither arm passes. Both are faster than predicted (123.9 and 123.1 ms against 133 and 136
predicted) and both cost more CE@6 than predicted (3.904 and 3.908 against 3.902 and 3.903).

Hinge every 4 stays GREY at 3 seeds: +0.00303, over the +0.003 line by 0.00003. That is a tie
with the line, not a measured difference, so the call goes to Wolfe. What it buys: 6.0 ms (4.6 %)
against the stack, and K1-K6 +0.0083 against the stack's +0.0064. What it costs: +0.0045 CE@6
paired against the stack, with one seed of three going the other way. It does not go into the
speed base without Wolfe's call.

The half-slot decoder FAILS (+0.0062). It also reads 0.011 worse than the stack, which reaches a
similar speed (130 ms) by the one-block decoder instead.

## Updated hypothesis

The hinge's cadence moves two things against each other: a lighter hinge lets the loop use depth
more (K1-K6 up on 2 of 3 seeds) and costs token CE. This matches the older reading that the
loop-holding terms (fixed-point term, gain hinge) hold the loop still
(`.agents/notes/implemented/architecture/2026-09-07-fixed-point-objective-as-the-loop-stability-term.md`).
The decoder needs every slot's target each step; a slot subsample costs as much as a block, and
the block cut is the cheaper of the two at equal speed.
