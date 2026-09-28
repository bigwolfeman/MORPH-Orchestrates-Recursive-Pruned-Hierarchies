# Planned: epivol lambda grid 2 (0, 0.0001, 0.001, 0.005, 0.015) and lambda 0.01 at seed 2

Status: planned

Date: 2026-09-28 09:37 (frozen before any arm trained).
Parent: [`../failures/2026-09-27-epivol-lambda-sweep.md`](../failures/2026-09-27-epivol-lambda-sweep.md).
Wolfe (2026-09-28): "We should do that 2nd 0.01 seed. We should also expand our grid. Lamda = 0,
0.015 0.005 0.001 and 0.0001".

## Question

Grid 1 at 3000 steps read K1-K6 +0.0301 at lambda 0.01 (seed 1), against +0.0099 at 0.05,
+0.0167 at 0.11 and +0.0082 at 0.1 (seed 2). The only measured seed spread for K1-K6 on this
recipe is 0.0069. Is the lambda-0.01 reading real, and where does depth use sit between the
term off (lambda 0) and 0.015?

## Hypothesis

H: with WTA on, epivol costs depth use. Low lambda (0 to 0.015) earns more K1-K6 than high
lambda (0.05 to 0.11), and the term-off arm earns as much as 0.01.

## Predictions (mine, orchestrator, before any arm trained)

All at 3000 steps, 480 sweep rows, coda readouts. "Low group" = seed-1 arms at lambda 0, 0.0001,
0.001, 0.005, 0.01, 0.015. "High group" = 0.05 and 0.11 (grid 1).

- **P-1**: no arm detonates. 80 %.
- **P-2**: lambda 0.01 at seed 2 reads K1-K6 above +0.0082, the seed-2 lambda-0.1 point at
  3000 steps (same seed). 70 %.
- **P-3**: lambda 0.01 at seed 2 reads K1-K6 in [+0.012, +0.030]. Seed 1 is likely a high draw.
  55 %.
- **P-4**: the median K1-K6 of the low group is above +0.0167, the highest high-group reading.
  55 %.
- **P-5**: lambda 0 reads K1-K6 of at least +0.015. 55 %.
- **P-6**: lambda 0 and lambda 0.0001 read within 0.007 in K1-K6 and within 0.0137 in CE on the
  same rows. 75 %.
- **P-7**: every low-group arm is within 0.0137 of lambda 0.01 (seed 1) in depth-6 CE on the same
  rows. 75 %.
- **P-8**: cell rank at pass 1 is lower at lambda 0 than at 0.01 (2.579). A diagnostic, not a
  pass line. 60 %.

Pass: P-2 and P-4 hold.

## Method

- Configs in `morph/configs/`: `tul_slot_spandec_strict_lxfan4_wta_fp01_ev001_s2` (ev001 with
  seed 2) and `..._ev0`, `_ev00001`, `_ev0001`, `_ev0005`, `_ev0015` (lambda 0, 0.0001,
  0.001, 0.005, 0.015). Every one is (b) with steps 3000 and lr_decay_steps 5000, as in grid 1.
  Composed on CPU: each resolved config differs from (b) only in lambda, steps, lr_decay_steps,
  wandb.name, and (ev001_s2) seed.
- lambda 0 keeps `fan_repel_mode: epivol`: the term's instruments run, and transformer.py adds
  the term to the loss only when lambda > 0.
- Order: ev001_s2, ev0, ev0005, ev0015, ev0001, ev00001. Runner `runner_steps.sh`, queue
  `queue_next.txt`, guard `grid2_guard.sh` (the sustained-tripwire rule, kill by PID).
- Paired readouts as in the parent: each arm against ev001 (seed 1) on the same rows, ev001_s2
  against the seed-2 lambda-0.1 point at 3000, and ev0 against ev00001.
- The stage-2 scorer and notul pair exit 1 on fan arms (no parallel head); they are not readouts
  of this experiment.
