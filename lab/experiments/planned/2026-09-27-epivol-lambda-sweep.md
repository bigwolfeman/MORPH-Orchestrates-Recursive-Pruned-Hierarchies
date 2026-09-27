# Planned: epivol lambda sweep on LX-Fan + WTA at 3000 steps

Status: planned

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
