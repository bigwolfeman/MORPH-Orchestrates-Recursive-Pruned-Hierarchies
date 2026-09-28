# Planned: three-seed bands for epivol lambda 0, 0.01 and 0.1 at 3000 steps

Status: planned

Date: 2026-09-28 16:44 (frozen before any of the five new runs trained).
Parents: [`../failures/2026-09-27-epivol-lambda-sweep.md`](../failures/2026-09-27-epivol-lambda-sweep.md),
[`2026-09-28-epivol-lambda-grid2.md`](2026-09-28-epivol-lambda-grid2.md) (in progress, paused).
Wolfe (2026-09-28): "those seed runs should go to the front of the queue."

## Question

Grid 2's first two arms: lambda 0.01 at seed 2 read K1-K6 +0.0026 against seed 1's +0.0301, so
one config spans 0.0275 over two draws. Every lambda-vs-depth reading in grids 1 and 2 sits
inside that spread. Lambda 0 read cell rank 1.51 at pass 1 against 2.58 at lambda 0.01, a large
effect on one draw. With three draws per point, which of these effects survive: depth, CE,
cell rank, the pass-0 write?

## Hypothesis

H: epivol's effect on cell rank is real and large; its effect on K1-K6 at 3000 steps is smaller
than the draw noise.

## Draws

| lambda | seed 1 | seed 2 | seed 3 |
| --- | --- | --- | --- |
| 0 | ev0 (done: +0.0088) | **ev0-s2 (new)** | **ev0-s3 (new)** |
| 0.01 | ev001 (done: +0.0301) | ev001-s2 (done: +0.0026) | **ev001-s3 (new)** |
| 0.1 | **ev01 (new)** | s2 step_3000 (done: +0.0082) | **ev01-s3 (new)** |

## Predictions (mine, orchestrator, before any of the new runs trained)

- **P-1**: none of the five new runs detonates. 85 %.
- **P-2**: cell rank at pass 1 (`fan/stream_rank_t1`) is below 1.9 on all three lambda-0 draws
  and above 2.3 on all six lambda-0.01 and 0.1 draws. 75 %. (The s2 step_3000 point has no
  logged 3000-step rank; P-2 counts the draws that log one.)
- **P-3**: no lambda separates on depth: each lambda's three-draw mean K1-K6 lies inside the
  [min, max] K1-K6 range of each of the other two lambdas. 60 %.
- **P-4**: mean depth-6 CE at lambda 0.01 is below the mean at lambda 0, by less than 0.0137.
  55 %.
- **P-5**: the pass-0 write (25-step median of `loop/core_gain_t0`) at step 3000 is below 5x on
  all three lambda-0 draws, and above 5x on at least two of the three lambda-0.1 draws. 65 %.
- **P-6**: the K1-K6 range over three seeds exceeds 0.010 for at least two of the three
  lambdas. 65 %.

Pass: P-2 and P-3 hold.

## Method

- New configs in `morph/configs/`: `tul_slot_spandec_strict_lxfan4_wta_fp01_ev01` ((b) at 3000
  steps with grid 1's pins, seed 1), `..._ev01_s3`, `..._ev0_s2`, `..._ev0_s3`,
  `..._ev001_s3`. Composed on CPU: each differs from (b) only in steps, lr_decay_steps, lambda,
  seed and wandb.name as its name says.
- The seed-2 lambda-0.1 draw is `lxtul-lxfan4-wta-fp01-s2` step_3000: a 5000-step run whose
  schedule matches the 3000-step configs through step 3000 (lr == min_lr, t_beta3 3500, cadences
  absolute). Its ckpt_every is 3000, not 2500; saving a checkpoint does not touch the loss.
- Order: ev01, ev0-s2, ev001-s3, ev0-s3, ev01-s3, then grid 2's held arms. Same runner, readouts
  and sustained-tripwire guard as grid 2.
- Per lambda: mean and [min, max] of K1-K6, depth-6 CE paired against ev001 (seed 1) on the same
  480 rows, cell rank at passes 1 and 6, pass-0 write at 3000.
