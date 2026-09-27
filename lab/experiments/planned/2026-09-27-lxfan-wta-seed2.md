# Planned: LX-Fan + WTA (b) at a second seed

Status: planned

Date: 2026-09-27 16:54 (frozen before the run trained).
Parent: [`../failures/2026-09-26-lx-credit-arms.md`](../failures/2026-09-26-lx-credit-arms.md), arm (b)
`lxtul-lxfan4-wta-fp01`, and [`../failures/2026-09-27-lx-soft-fan-retry.md`](../failures/2026-09-27-lx-soft-fan-retry.md).
Wolfe (2026-09-27): "lets do a second seed of b."

## Question

(b) is LX codes (K=4 fixed simplex codes, per-span mixture loss) on the write-all fan with cell
WTA 1.0 and epivol 0.1. At seed 1 and 5k it read K1-K6 +0.0162 [+0.0153, +0.0172] against a2's
+0.0057 (the same fan and WTA, no codes), and gap to plain +0.2193. The soft-fan retry (codes,
no WTA) read K1-K6 +0.0161 and 0.0303 nats worse CE than (b). So one draw says the codes buy
the depth use and WTA buys the CE. Is (b)'s depth use a property of the recipe or of the draw?

## Hypothesis

H: the depth use is the recipe's. A second draw earns K1-K6 above a2's +0.0057, holds CE within
the cross-arm seed spread (0.0137) of seed 1, and does not detonate.

## Predictions (mine, orchestrator, before any step of the run)

- **P-1**: no detonation (sustained tripwire never fires; the run writes step_5000.pt). 85 %.
- **P-2**: the 25-step median of `loop/core_gain_t0` stays below 5 for the whole run (seed 1
  held near 3x; every soft fan passed 10x). 75 %.
- **P-3**: K1-K6 (coda, 480 sweep rows, same readout as the parent) lower CI bound above
  +0.0057. 70 %.
- **P-4**: K1-K6 point value in [+0.010, +0.022]. 60 %.
- **P-5**: gap to plain at depth 6 in [+0.2056, +0.2330] (seed 1's +0.2193 +- 0.0137). 70 %.
- **P-6**: one cell alone reads at most 0.08 nats behind the joint read (seed 1: 0.049; soft
  fan: 0.164). 65 %.

Pass: P-1 and P-3 hold. P-5 is read for direction only (Wolfe: 5k CE is not a verdict).

## Method

- Config `morph/configs/tul_slot_spandec_strict_lxfan4_wta_fp01_s2.yaml`: (b) with
  `training.seed: 2` and `training.ckpt_every: 3000` (the extra step_3000.pt is the
  lambda-0.1 point of [`2026-09-27-epivol-lambda-sweep.md`](2026-09-27-epivol-lambda-sweep.md)).
  Composed on CPU: the resolved config differs from (b) in seed, ckpt_every and wandb.name only.
- Runner `/home/wolfe/morph-scratch/abc/runner_steps.sh`, queue `queue_next.txt` line 1,
  guard `next_guard.sh` (the parent's sustained-tripwire rule, kill by PID). Readouts: depth
  sweep 1-6 on 480 rows, paired gap vs plain 5k, stage-2 score vs fp01 5k, notul pair, worth.
- Paired vs (b) seed 1 5k on the same 480 rows; the per-cell read as in the soft-fan retry.
- Runs only after Wolfe frees the GPU and says go.
