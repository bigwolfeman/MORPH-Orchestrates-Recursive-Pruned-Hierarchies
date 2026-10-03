# Planned: does fixing the read state's scale stop the detached arm's pass-2 jump?

Status: failure

Date: 2026-10-03 09:37 CDT, before the arm trains past its smoke. Config
`tul_slot_spandec_strict_fan4_all_fp01_lsel_det_rf_lam1_cnorm_read` (one key over the per-stream norm
arm: `slot_cell_pass_norm: rms_read`; 3000 steps). Wolfe (2026-10-03): "try #2 out to 3k steps
then we will reassess."

## Question

Under the per-stream norm the detached weight-1 arm still jumps onto one shared direction
at pass index 2 (probe cosine 0.02 -> 0.92) and its K1-K6 fell from +0.0333 to +0.0119
([filing](../failures/2026-10-03-slot-cell-pass-norm-pair.md)). My guess: each stream's RMS
is fixed, but the stream mean (what the head, router, epivol and readout read) can still
grow when the 4 streams align. `rms_read` fixes the mean's RMS directly. Does the jump go?

## Predictions (mine, orchestrator)

Control: the per-stream norm arm at step 3000 (val loss 4.626; its K-curve and probe exist
only at 5000).

- **D-1**: no detonation (HEALTHY or one recovered spike). 80 %.
- **D-2**: probe cosine of the read cells to their shared direction at pass index 2 below
  0.5 (per-stream arm 0.92 at 5000). 60 %. This is the direct test of the guess.
- **D-3**: val loss at step 3000 within 0.05 of 4.626, or better. 60 %.
- **D-4**: K1-K6 at 3000 above 0 with CI above 0. 75 %. Above +0.0119 (the per-stream arm at
  5000). 35 %.
- **D-5**: tok/s within 5 % of 9809. 85 %.

## Method

Seed 1, 3000 steps, runner_steps.sh. Readouts: the runner's depth sweep, gap and worth at
3000; the exploration ledger; the LayerNorm probe (geometry and --vablate). The K-curve
and probe have no 3000-step control, so D-2 and D-3 decide; D-4 is a pointer.

## Results

Run at 59bd74e, 3000 steps, seed 1, 9826 tok/s. Artifacts:
[`../results/2026-10-03-slot-cell-read-norm/`](../results/2026-10-03-slot-cell-read-norm/).

Read-state geometry per pass (LayerNorm probe, 96 rows; "along" and "perp" split the read
cell on its shared direction v):

| pass index | read norm | cos to v | along v | perp | between-slot | within-slot |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 30.98 | 0.004 | 1.6 | 30.9 | 1.7 | 30.9 |
| 1 | 31.02 | 0.042 | 5.3 | 30.6 | 1.7 | 30.9 |
| 2 | 31.16 | 0.636 | 20.7 | 23.3 | 22.1 | 8.7 |
| 3 | 31.44 | 0.924 | 29.1 | 12.0 | 11.5 | 3.5 |
| 5 | 31.50 | 0.956 | 30.1 | 9.2 | 8.5 | 3.0 |

Mean-vs-zero ablation of v at K6: mean +0.0011 [+0.0001, +0.0021], zero +0.0168
[+0.0145, +0.0191]; K6 - K1 shipped -0.0092, mean-ablated -0.0089, zeroed +0.0059.
Ledger (480 rows): shipped CE 4.611, random exit pick +0.0203, teacher +0.0122, fixed
lineage +0.0031, oracle -0.0241. Greedy rep-4 0.609 [0.543, 0.671], distinct-4 0.216
(per-stream norm arm at 5000: 0.493, 0.291).

| prediction | reading | held |
| --- | --- | --- |
| D-1 no detonation | one-step excursion 1.07e4 at 2521, recovered | yes |
| D-2 cos to v at pass index 2 < 0.5 | 0.636 (0.92 to 0.96 from pass index 3) | no |
| D-3 val loss at 3000 within 0.05 of 4.626 | 4.6135 | yes |
| D-4 K1-K6 > 0 / > +0.0119 | +0.0063 [+0.0053, +0.0072] / no | yes / no |
| D-5 tok/s within 5 % of 9809 | 9826 | yes |

## Verdict

Failure: D-2, the direct test, missed. My guess was wrong. The read norm fixes the read
state's scale exactly (norm 31.0 to 31.5 at every pass), and the cells still go onto one
shared direction: with the scale fixed, they ROTATE onto it. The part off that direction
shrinks from 30.9 to 9.2, and the within-slot spread from 30.9 to 3.0. So the detached
weight-1 arm's pass-2 event is a direction collapse, not growth by stream alignment.
Norms decide only whether it shows up as growth (no norm) or rotation (read norm).

## Updated hypothesis

The collapse belongs to the latent pull on the detached arm, not to the missing norm. The
rank-only head (no pull) under the per-stream norm reaches cos 0.36 and spreads its slots
apart 9x, with K1-K6 +0.0215. The lead arm stays the rank-only head + per-stream norm.
