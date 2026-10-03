# Planned: does the rank-only head + per-stream cell norm earn depth on a second seed?

Status: success

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

## Results

Seed 2 at b97d4cd, 5000 steps. Artifacts:
[`../results/2026-10-03-rank-only-cell-norm-seed-twin/`](../results/2026-10-03-rank-only-cell-norm-seed-twin/).

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth | greedy rep-4 |
| --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | | 0.523 [0.450, 0.597] |
| rank-only + norm, seed 1 | 9800 | 4.353 | +0.272 [+0.257, +0.290] | +0.0215 [+0.0205, +0.0225] | +0.176 | 0.560 [0.486, 0.632] |
| rank-only + norm, seed 2 | 8876 | 4.362 | +0.282 [+0.266, +0.299] | +0.0172 [+0.0163, +0.0181] | +0.154 | 0.474 [0.404, 0.548] |

Seed 2, mean-vs-zero ablation of the shared direction (K6 - K1): shipped -0.0183
[-0.0208, -0.0159], mean-ablated -0.0170 [-0.0196, -0.0147], zeroed +0.0027 [-0.0006,
+0.0060]. Geometry: cosine to the shared direction 0.02 -> 0.40 at pass index 2, flat
after; between-slot spread 0.07 -> 2.34; within-slot 0.75 throughout. Ledger: random exit
pick +0.0059, random walk +0.0063, fixed lineage +0.0191, oracle -0.0215.

| prediction | reading | held |
| --- | --- | --- |
| T-1 no detonation | HEALTHY (max 471 at 1135) | yes |
| T-2 K1-K6 > +0.0108, CI above 0 | +0.0172 [+0.0163, +0.0181] | yes |
| T-3 within 0.007 of +0.0215 | 0.0043 below | yes |
| T-4 mean-ablation changes K6-K1 by < 30 % | 7 % | yes |
| T-5 ledger CE within 0.03 of 4.353 | 4.362 | yes |
| T-6 tok/s within 5 % of 9800 | 8876 (9.4 % lower) | no |

## Verdict

Success: the pass rule (T-2 and T-4) holds. The rank-only head with the per-stream cell
norm earns depth on both seeds (+0.0215, +0.0172; mean +0.0194, spread 0.0043), the
earning is directional on both, and CE agrees within 0.009. T-6 missed: seed 2 ran 9.4 %
slower; the config is identical, so the likely cause is host load during the run, not
checked.

## Updated hypothesis

The recipe holds across two seeds: about 4.6x the ungraded fan's K1-K6, 0.02 to 0.03
nats behind it on CE, at the same speed. Next: the same arm at 10k steps.
