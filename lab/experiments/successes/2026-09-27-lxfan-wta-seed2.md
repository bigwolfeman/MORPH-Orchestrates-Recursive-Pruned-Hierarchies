# LX-Fan + WTA (b) at a second seed

Status: success

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

## Results

Filed 2026-09-28. Run `lxtul-lxfan4-wta-fp01-s2` at 893be0f, 5000 steps, 2558 tok/s, peak
21.09 GB, final trainer val_loss 4.4536. Readouts are on the 480 sweep rows, at the coda, at
depth 6. Artifacts: [`../results/2026-09-27-lxfan-wta-seed2/`](../results/2026-09-27-lxfan-wta-seed2/).

| Reading | seed 1 (b) | seed 2 | Source |
| --- | --- | --- | --- |
| Tripwire | HEALTHY, max 270 | HEALTHY, max 38.5 @ 746 | `chain.runlog.txt` |
| K1-K6 | +0.0162 [+0.0153, +0.0172] | +0.0093 [+0.0088, +0.0100] | `s2_vs_s1.runlog.txt` |
| K3-K6 | +0.0020 [+0.0017, +0.0024] | +0.0009 [+0.0007, +0.0011] | same |
| Gap to plain 5k | +0.2193 [+0.2038, +0.2368] | +0.2375 [+0.2215, +0.2549] | `gap_s2.runlog.txt` |
| Seed 2 minus seed 1, same rows | | +0.0178 [+0.0152, +0.0203] | `s2_vs_s1.runlog.txt` |
| Worth zero / shuffle | 0.200 / 0.170 | 0.197 / 0.177 | `worth_s2.json` |
| One cell alone minus joint read | 0.0488 | 0.0718 | `cell_and_write.txt` |
| Oracle gap / pick0 | 0.0876 / 0.370 | 0.1124 / 0.322 | same |
| Cell rank pass 1 / pass 6 | 2.924 / 2.538 | 2.924 / 2.793 | same |
| `core_gain_t0`, 25-step median at 5000 | 4.2 (max 4.4) | 23.4, still rising | same |

The stage-2 scorer (`lxtul_e_stage2_score.py`) exits 1 on every fan arm: it needs
`tul.spandec_parallel`, which LX-Fan refuses. It failed the same way on seed 1. The notul pair
reads the scorer's output, so it failed too. Neither is a reading of this run.

- **P-1 holds.** No detonation.
- **P-2 fails.** The pass-0 write rose almost linearly from 2.0 at step 1000 to 23.4 at 5000
  and had not levelled off. The run stayed healthy anyway (tripwire max 38.5).
- **P-3 holds.** The K1-K6 lower bound +0.0088 is above a2's +0.0057.
- **P-4 fails.** +0.0093 is just below the predicted [+0.010, +0.022].
- **P-5 fails.** Gap +0.2375 is 0.0045 past the top of [+0.2056, +0.2330]; seed 2 is 0.0178 worse
  than seed 1 on the same rows, which is more than the 0.0137 cross-arm seed spread.
- **P-6 holds.** 0.0718 <= 0.08.

## Verdict

Success on the preregistered pass line (P-1 and P-3), with three of six predictions failed.
The depth use survives a second draw in a weaker form: seed 2 earns K1-K6 +0.0093, 1.6x a2's
single draw, not the 2.8x that seed 1 read. The two seeds give one seed-to-seed K1-K6 spread for
this recipe: 0.0069. That is larger than the gap between (b) seed 1 and the soft retry
(0.0001), so "the codes buy the depth" still stands on arms that differ by less than one seed of
noise in K1-K6. a2 has one draw, so the "above a2" line compares against a point, not a band.

The pass-0 write is not held near 3x by WTA. Seed 2 grew it to 23x, past the soft fans'
12-14x plateau, and did not detonate. So a large pass-0 write alone does not predict the soft
fans' detonation, and WTA does not stop the growth. The claim "every WTA fan held it near 3x"
in the soft-fan retry filing and in `docs/slot-cells-distinct-vs-blurred.md` was one draw; both
carry a dated amendment.

## Updated hypothesis

The LX codes on a WTA fan earn depth at about 0.009 to 0.016 nats K1-K6 at 5k, and the true
seed-to-seed spread of K1-K6 on this recipe is near 0.007. No single-draw K1-K6 difference under
about 0.01 in this family can be read as an arm effect. What separates the soft fans that
detonated from the WTA fans is not the size of the pass-0 write.
