# Planned: does a per-pass cell norm keep the loop's real depth and kill its magnitude channel?

Status: failure

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

## Results

Both arms ran 5000 steps at d699450, seed 1. Artifacts:
[`../results/2026-10-03-slot-cell-pass-norm-pair/`](../results/2026-10-03-slot-cell-pass-norm-pair/).
CE is the ledger's shipped CE on the 480 sweep rows.

| arm | tok/s | ledger CE | gap to plain | K1-K6 | zero-cell worth | greedy rep-4 |
| --- | --- | --- | --- | --- | --- | --- |
| ungraded fan | 9797 | 4.334 | +0.254 | +0.0042 | | 0.523 [0.450, 0.597] |
| detached weight 1 | 9818 | 4.447 | +0.366 | +0.0333 | | 0.568 [0.501, 0.633] |
| detached weight 1 + norm | 9809 | 4.405 | +0.325 [+0.308, +0.343] | +0.0119 [+0.0109, +0.0130] | +0.096 | 0.493 [0.432, 0.553] |
| rank-only head | 9635 | 4.380 | +0.299 | +0.0052 | +0.166 | 0.559 [0.493, 0.622] |
| rank-only head + norm | 9800 | **4.353** | **+0.272 [+0.257, +0.290]** | **+0.0215 [+0.0205, +0.0225]** | +0.176 | 0.560 [0.486, 0.632] |

Mean-vs-zero ablation of the cells' shared direction at the write (96 rows, K6 - K1, 95 % CI):

| arm | shipped | mean-ablated | zeroed |
| --- | --- | --- | --- |
| rank-only head (no norm, 2026-10-02) | -0.0052 | +0.0003 | +0.0216 |
| rank-only head + norm | -0.0219 [-0.0242, -0.0199] | -0.0217 [-0.0241, -0.0196] | -0.0112 [-0.0141, -0.0085] |
| detached weight 1 (no norm) | -0.0326 | -0.0329 | -0.0246 |
| detached weight 1 + norm | -0.0113 [-0.0132, -0.0095] | -0.0114 [-0.0134, -0.0095] | +0.0150 [+0.0121, +0.0178] |

Ledger, rank-only + norm: random exit pick +0.0049, random walk +0.0051, fixed lineage
+0.0069, oracle -0.0215. Detached weight 1 + norm: random exit +0.0074, fixed lineage
+0.0516 (cell 0 is a poor cell: +0.0504), oracle -0.0263, teacher -0.0008.

Cell geometry (probe units, the read state): the detached arm + norm still jumps onto its
shared direction at pass index 2 (cosine 0.02 -> 0.92, between-slot 0.10 -> 1.09). The
rank-only arm + norm reaches cosine 0.36, and its between-slot spread grows 9x (0.27 ->
2.45) while within-slot stays 0.8. The norm acts on each of the 4 HC streams; the probe
reads the combined state, which can still grow when the streams align.

| prediction | reading | held |
| --- | --- | --- |
| N-1 no detonation | detached: one-step excursion 5.6e4 at 2654, recovered; rank: HEALTHY | yes |
| N-2 detached + norm K1-K6 > +0.0167 | +0.0119 | no |
| N-3 rank-only + norm K1-K6 < +0.0026 | +0.0215 (4x its parent) | no, opposite sign |
| N-4 detached + norm CE within 0.03 of 4.447 or better | 4.405 | yes |
| N-5 detached + norm step at pass index 2 < 50 % along the shared direction | step share not logged; cosine to it 0.92 at pass index 2 | no (read from the cosine) |
| N-6 tok/s within 5 % | 9809 / 9800 | yes |

## Verdict

Failure on the pass rule (N-2 and N-3 both missed). The hypothesis was wrong about the
rank-only arm in the useful direction. Without the norm its loop contribution was one
magnitude (mean-ablation erased it). With the norm it earns 4x more depth (+0.0215), all
of it directional (mean-ablation changes nothing), and it is the best latent-selected arm
on CE so far: 4.353, 0.019 behind the ungraded fan, against 0.046 before. Removing the
magnitude channel made the loop find direction. The detached arm lost depth (+0.0333 ->
+0.0119) and gained CE (4.447 -> 4.405); it still writes a shared direction at pass 2.

## Updated hypothesis

The norm turns a magnitude-only loop into a directional one. It does not stop an arm
whose pull already writes a shared bias from doing so, likely because the norm is per HC
stream and the read state is the streams' combination. Next: the rank-only head + norm is
the lead arm (K1-K6 +0.0215, CE 0.019 behind the ungraded fan, 9800 tok/s). Seed twin
first, then a 10k run; and a norm on the combined read state as a one-key variant.
