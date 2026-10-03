# Planned: does fixing the read state's scale stop the detached arm's pass-2 jump?

Status: planned

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
