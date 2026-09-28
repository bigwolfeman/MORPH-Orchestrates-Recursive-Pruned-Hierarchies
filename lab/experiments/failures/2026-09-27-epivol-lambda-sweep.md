# epivol lambda sweep on LX-Fan + WTA at 3000 steps

Status: failure

Date: 2026-09-27 16:54 (frozen before any arm trained).
Parent: [`../failures/2026-09-26-lx-credit-arms.md`](../failures/2026-09-26-lx-credit-arms.md), arm (b).
Wolfe (2026-09-27): "test some more values for epivol ... 3k runs at a small sweep ... 0.01 0.05 0.11".

## Question

epivol (`tul.fan_repel_mode: epivol`: epiplexity of the between-stream deviations plus
within-slot volume, passes 1-2) has only run at lambda 0.1. It was added before WTA. WTA now
gives each cell its own winner-alone CE, which is a second push toward distinct cells. How does
the fan's CE, depth use and cell spread move with lambda when WTA is on?

## Hypothesis

H: with WTA on, epivol is mostly redundant. The depth use comes from the codes, so K1-K6 does
not move with lambda. The term costs a little CE, so lower lambda reads slightly better CE,
and the cells spread less.

## Predictions (mine, orchestrator, before any arm trained)

All at 3000 steps, 480 sweep rows, coda readouts. The lambda-0.1 point is
`lxtul-lxfan4-wta-fp01-s2` step_3000 (seed 2; the sweep arms are seed 1).

- **P-1**: no arm detonates. 80 % for 0.05 and 0.11, 70 % for 0.01.
- **P-2**: K1-K6 of all four points (0.01, 0.05, 0.1, 0.11) lie within a band 0.006 wide. 60 %.
- **P-3**: depth-6 CE is ordered 0.01 <= 0.05 <= 0.11 (lower lambda, lower CE), and the
  0.01-to-0.11 difference is under 0.0137 (inside the seed spread). 45 %.
- **P-4**: cell rank at pass 1 (`fan` rank instrument) falls with lambda: 0.01 reads lower
  than 0.11. A diagnostic, not a pass/fail line. 65 %.
- **P-5**: 0.11 and the seed-2 0.1 point read within 0.0137 of each other in CE. They differ
  by 10 % in lambda, so this pair is a seed-noise reading at 3k. 75 %.

Pass: P-1 and P-2 hold. P-3 is read for direction only.

## Method

- Configs `morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01_ev001.yaml`, `_ev005`,
  `_ev011`: (b) with `tul.fan_repel_lambda` 0.01 / 0.05 / 0.11, `training.steps: 3000`,
  `training.lr_decay_steps: 5000`. Composed on CPU: each resolved config differs from (b) in
  those three keys and wandb.name only. Every steps-derived value was audited (ev001's header):
  AdEMAMix t_alpha / t_beta3 resolve to 1600 / 3500 (not null), lr == min_lr, TST, prune and
  route are off, eval and checkpoint cadences are absolute. So steps 0-3000 of each arm run
  (b)'s schedule exactly; only lambda differs.
- Runner `runner_steps.sh` (4th queue field = steps), queue `queue_next.txt` lines 2-5 after
  the seed-2 run; line 5 reads out s2's step_3000.pt (readout only, never trains). Guard
  `next_guard.sh`.
- The runner's gap and score readouts reference plain 5k and fp01 5k. Across steps those
  absolute values mean little; the sweep reads the four 3k points against each other.
- Runs only after Wolfe frees the GPU and says go.

## Results

Filed 2026-09-28. Three arms at 893be0f, seed 1, 3000 steps each, 2627-2634 tok/s, peak 21.09
GB. The lambda-0.1 point is `lxtul-lxfan4-wta-fp01-s2` step_3000 (seed 2). Readouts are on the
480 sweep rows, at the coda, at depth 6. Artifacts:
[`../results/2026-09-27-epivol-lambda-sweep/`](../results/2026-09-27-epivol-lambda-sweep/).

| lambda | seed | Tripwire max | K1-K6 | K3-K6 | CE minus the 0.1 point, same rows | Worth zero / shuffle | Cell rank pass 1 | `core_gain_t0` at 3000 (max) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | 1 | 27 @ 2393 | +0.0301 [+0.0289, +0.0314] | +0.0006 | -0.0203 [-0.0234, -0.0171] | 0.163 / 0.146 | 2.579 | 8.4 (14.0) |
| 0.05 | 1 | 76.6 @ 738 | +0.0099 [+0.0091, +0.0107] | -0.0003 | -0.0204 [-0.0233, -0.0178] | 0.165 / 0.139 | 2.838 | 3.4 (3.4) |
| 0.1 (s2 @ 3000) | 2 | (s2, healthy) | +0.0082 [+0.0076, +0.0088] | | 0 | 0.156 / 0.136 | | 11.8 |
| 0.11 | 1 | 52.2 @ 2409 | +0.0167 [+0.0157, +0.0177] | +0.0017 | -0.0127 [-0.0154, -0.0102] | 0.148 / 0.131 | 2.730 | 12.0 (14.1) |

0.01 minus 0.11, same rows: CE -0.0076 [-0.0101, -0.0051]. Gaps to plain 5k (cross-step, for
reference only): 0.01 +0.4424, 0.05 +0.4423, 0.11 +0.4499, s2 @ 3000 +0.4627. The stage-2 scorer
and the notul pair exit 1 on fan arms (no parallel head), as in the seed-2 filing. The
3000-step cell rank of the s2 point is not logged: the trainer's final-val line is at 5000.

- **P-1 holds.** No arm detonated.
- **P-2 fails.** K1-K6 spans +0.0082 to +0.0301, a band 0.022 wide against a predicted 0.006.
  The seed-1 arms alone span 0.020.
- **P-3 holds in direction, not strictly.** 0.01 and 0.05 tie (0.0001 apart, inside each other's
  interval); both beat 0.11. The 0.01-to-0.11 difference, 0.0076, is inside 0.0137.
- **P-4 holds.** Cell rank at pass 1: 0.01 reads 2.579, 0.11 reads 2.730. It is not monotone:
  0.05 reads the highest, 2.838.
- **P-5 holds, barely.** 0.11 vs the seed-2 0.1 point: 0.0127 < 0.0137.

## Verdict

Failure: P-2, the pass line's depth prediction, failed. K1-K6 is not flat in lambda. The
lowest lambda, 0.01, reads +0.0301, the largest K1-K6 in the LX-Fan family so far, and at 3000
steps, not 5000. The 0.01-vs-0.11 K1-K6 difference is 0.0134. That is about twice the one
seed-to-seed spread measured on this recipe (0.0069, the seed-2 filing), so it is suggestive
and not settled. 0.05 reads +0.0099, below 0.11, so the curve is not monotone in lambda; with a
seed spread near 0.007, 0.05 and 0.11 are not separable. Nearly all of 0.01's depth use sits in
passes 1 to 3 (K3-K6 +0.0006).

CE moves the direction the hypothesis said: less epivol, better CE, by 0.0076 from 0.11 to
0.01 on the same rows, inside the seed spread. At 3000 steps neither CE nor worth separates the
arms beyond one seed.

## Updated hypothesis

epivol at 0.1 costs depth use rather than buying it when WTA is on. WTA already makes the cells
distinct (cell rank 2.58 at lambda 0.01 is still far above the register's 1.24), and the
diversity term pushes the cells apart along directions the reader does not use across passes.
Next: lambda 0.01 at a second seed, and lambda 0 (epivol off, WTA on), both at 3000 steps, so
the 0.0301 reading gets a seed band and a floor.
