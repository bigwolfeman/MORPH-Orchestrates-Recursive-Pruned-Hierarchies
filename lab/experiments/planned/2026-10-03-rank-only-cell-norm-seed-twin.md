# Planned: does the rank-only head + per-stream cell norm earn depth on a second seed?

Status: planned

Date: 2026-10-03 14:05 CDT, before the run. Config `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm_s2`:
the seed-1 arm at training seed 2, nothing else changed (composed configs differ only in
`training.seed` and `wandb.name`). Wolfe (2026-10-03): "lets do a 2nd seed".

## Question

Seed 1 ([filing](../failures/2026-10-03-slot-cell-pass-norm-pair.md)): K1-K6 +0.0215
[+0.0205, +0.0225], ledger CE 4.353, gap to plain +0.272, zero-cell worth +0.176, greedy
rep-4 0.560, and mean-ablating the cells' shared direction left K6-K1 unchanged (-0.0219
-> -0.0217). It is the lead arm. Is it a seed-1 accident? MORPH runs decorrelate within
tens of steps at a fixed seed, and the LX-Fan seed spread was about 0.007 on K1-K6.

## Predictions (mine, orchestrator)

- **T-1**: no detonation (HEALTHY or one recovered spike). 85 %.
- **T-2**: K1-K6 above +0.0108 (half of seed 1), CI above 0. 70 %.
- **T-3**: K1-K6 within 0.007 of +0.0215. 45 %.
- **T-4**: the earning is directional: mean-ablating v changes K6-K1 by under 30 % of its
  shipped value. 70 %.
- **T-5**: ledger CE within 0.03 of 4.353. 65 %.
- **T-6**: tok/s within 5 % of 9800. 85 %.

Pass rule: T-2 and T-4 hold.

## Method

Seed 2, 5000 steps, runner_steps.sh. Readouts: depth sweep, gap, worth; the exploration
ledger; the LayerNorm probe (geometry and --vablate); the repetition eval.
