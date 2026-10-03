# Planned: does a per-pass cell norm keep the loop's real depth and kill its magnitude channel?

Status: planned

Date: 2026-10-03 02:20 CDT, before either arm trains past its 45-step smoke.
Note: [`2026-10-03-slot-cell-pass-norm.md`](../../../.agents/notes/proposed/architecture/2026-10-03-slot-cell-pass-norm.md).
Configs at d724f65: `tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1_cnorm` (parent: detached
weight 1), `tul_slot_spandec_strict_fan4_all_fp01_lsel_joint_rf_lam1_rank_cnorm` (parent: rank-only
head). One key each: `tul.slot_cell_pass_norm: rms`.

## Question

Huginn and HRM-Text both normalise their carried state at the end of every step, and both
move it perpendicular to a small shared bias direction
([Huginn](../successes/2026-10-02-huginn-loop-geometry.md),
[HRM-Text](../failures/2026-10-03-hrm-text-loop-geometry.md)). MORPH's slot cells are not
normalised. In the pulled arms they grow 4x along a shared bias at pass index 2, and in the
rank-only arm the whole K1-K6 is one amplitude along it
([LayerNorm note](../../../.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md)).
Wolfe (2026-10-02): "magnitude isn't true diversity." With the magnitude channel removed,
which arm still earns depth?

## Hypothesis

The detached weight-1 arm's depth earning is directional (mean-ablating its shared
direction left its K-curve unchanged), so it survives the norm. The rank-only arm's earning
is a magnitude, so the norm removes it.

## Predictions (mine, orchestrator)

Parents (ledger, 480 rows): detached weight 1 CE 4.447, K1-K6 +0.0333, tok/s 9818;
rank-only head CE 4.380, K1-K6 +0.0052, tok/s 9635.

- **N-1**: neither arm detonates (tripwire HEALTHY or a single recovered spike). 85 %.
- **N-2**: detached weight 1 with the norm keeps K1-K6 above +0.0167 (half its parent's),
  CI above 0. 55 %.
- **N-3**: rank-only with the norm drops K1-K6 below +0.0026 (half its parent's). 60 %.
- **N-4**: detached weight 1 with the norm: CE within 0.03 of 4.447 or better. 60 %.
- **N-5**: detached weight 1 with the norm: at pass index 2 under 50 % of the cells' mean
  squared step lies along the shared cell direction (parent 0.93), read by
  `lab/divergence/ln_common_mode_probe.py`. 65 %.
- **N-6**: both arms within 5 % of their parent's tok/s. 85 %.

Pass rule: N-2 and N-3 both hold.

## Method

Seed 1, 5000 steps each, `runner_steps.sh` at d724f65, detached arm first. Readouts: the
runner's depth sweep, gap to plain and worth; then the exploration ledger and the LayerNorm
probe (cell geometry and `--vablate`) on both step-5000 checkpoints; then the generation
eval for repetition.
