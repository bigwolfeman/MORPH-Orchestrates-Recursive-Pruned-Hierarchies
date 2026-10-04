# Planned: the lead arm at 10k steps

Status: success

Date: 2026-10-04 06:20 CDT, before the run. Wolfe (2026-10-04): "do a 10k training on our current best arm."
Arm: rank-only head + per-stream cell norm,
`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm`, seed 1, run to 10000 steps
through a 10k config that changes only `training.steps`, `wandb.name` and two speed keys
that are byte-identical to master by gate (`training.ademamix_fused_fp32: true`,
`model.ckpt_grad_iters`, value set from the training-speed pass 2 memory curve). The
non-bit-identical `model.ce_softmax_kernel` stays off.

## Question

At 5k the arm earns K1-K6 +0.0215 (seed 2: +0.0172), all of it directional, ledger CE
4.353 ([seed twin](../successes/2026-10-03-rank-only-cell-norm-seed-twin.md)). Does the
loop's contribution grow with training, as the fixed-point arm's did (+0.0125 at 5k ->
+0.0177 at 10k), and does it stay directional?

## Predictions (mine, orchestrator)

- **L-1**: no detonation (HEALTHY or one recovered spike). 80 %.
- **L-2**: K1-K6 at 10k above +0.0150 with CI above 0. 75 %.
- **L-3**: K1-K6 at 10k above the 5k value, +0.0215. 50 %.
- **L-4**: ledger CE at 10k below 4.353 by more than 0.08. 80 %.
- **L-5**: mean-ablating the cells' shared direction changes K6-K1 by under 30 %. 70 %.
- **L-6**: tok/s at least 1.25x the 5k run's 9800. 75 %.

Pass rule: L-2 and L-5.

## Method

runner_steps.sh, 10000 steps. Readouts at 10000: the depth sweep, gap and worth; the
exploration ledger; the LayerNorm probe (geometry, --vablate); the repetition eval. The 5k
checkpoint of this run (ckpt_every 2500) gives a same-run 5k reading for the K-curve.

**Method amendment, 2026-10-04 07:41 CDT, before the run.** Config
`tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm_10k`: `model.ckpt_grad_iters: 4`
(20.2 GiB reserved; chosen over k = 0 for daytime desktop slack) and
`training.ademamix_fused_fp32: true`, both byte-identical to master by gate (passes 1 and 2,
993dc50). No opt-in key that changes numerics is on. Expected step about 501 ms
(12258 tok/s); L-6's 1.25x bar is 12250.

## Results

Run at b3f2f2e, 10000 steps, seed 1, 12402 tok/s, peak 20.08 GB. Artifacts:
[`../results/2026-10-04-rank-only-cell-norm-10k/`](../results/2026-10-04-rank-only-cell-norm-10k/).

| reading | 5k (seed 1) | 10k |
| --- | --- | --- |
| K1-K6 | +0.0215 [+0.0205, +0.0225] | +0.0236 [+0.0225, +0.0246] |
| ledger CE (480 rows) | 4.353 | 4.0975 |
| gap to plain at the SAME step (K6 vs plain K6) | +0.272 vs plain 5k | +0.356 [+0.336, +0.379] vs plain 10k |
| zero-cell worth | +0.176 | +0.233 |
| K6 - K1: shipped / mean-ablated v | -0.0219 / -0.0217 | -0.0244 / -0.0230 |
| cos to v at pass index 2; between-slot / within-slot | 0.36; 2.45 / 0.84 | 0.33; 2.53 / 0.85 |
| greedy rep-4, distinct-4 | 0.560, 0.300 | 0.536 [0.468, 0.603], 0.343 |
| tok/s | 9800 | 12402 |

The runner's own gap line (+0.017) is against the plain model at 5k, not 10k: this arm at
10k roughly equals plain at 5k, the same pattern as the fixed-point arm on 2026-09-26. The
matched-step gap above uses `/home/wolfe/morph-scratch/plain-10k/sweep_plain-panel-norm-match_10000.json`.

Ledger at 10k: random exit pick +0.0088, random walk +0.0090, teacher +0.0038, fixed
lineage +0.0183, single cell +0.0136, oracle -0.0235.

Tripwire: preclip/total above 1e4 at steps 3829, 3831, 3844 and 8187 (max 3.89e5 at
3829); all recovered. Val loss kept falling through the first episode (3750 4.528, 4000
4.406, 4250 4.301) and the run ended at val 4.011.

| prediction | reading | held |
| --- | --- | --- |
| L-1 HEALTHY or one recovered spike | four spike steps in two episodes, all recovered | no |
| L-2 K1-K6 > +0.0150, CI above 0 | +0.0236 [+0.0225, +0.0246] | yes |
| L-3 K1-K6 above +0.0215 | +0.0236, CI above it | yes |
| L-4 ledger CE below 4.273 | 4.0975 | yes |
| L-5 mean-ablation changes K6-K1 by < 30 % | 6 % | yes |
| L-6 tok/s >= 12250 | 12402 | yes |

## Verdict

Success: the pass rule (L-2 and L-5) holds. The loop's contribution grew a little from 5k
to 10k (+0.0215 -> +0.0236, CIs separate) and stays directional; the slot channel's worth
grew (+0.176 -> +0.233). L-1 missed: four recovered spike steps, not one. The gap to a
plain model at the same step WIDENED (+0.272 -> +0.356), as it did for the fixed-point arm
(+0.265 -> +0.333): more training helps plain more than this arm.

## Updated hypothesis

The recipe keeps its loop contribution with training, but the cross-span channel still
loses ground to a plain model that reads every earlier token. The CE gap is the open
problem, not loop contribution. Repetition is unchanged (rep-4 overlaps every arm).
