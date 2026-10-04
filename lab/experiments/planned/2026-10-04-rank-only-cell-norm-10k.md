# Planned: the lead arm at 10k steps

Status: planned

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
