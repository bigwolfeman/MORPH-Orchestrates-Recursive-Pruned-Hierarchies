# The span decoder reads the winner cell

Status: planned

## Question

On LXTUL + pointer the span decoder is 0.69 of the loop core's gradient (credit matrix,
`lab/experiments/results/2026-10-10-lxtul-credit-matrix/`). It grades the MEAN of the 4 exit
cells, but the coda reads only the router's winner cell, so the 3 losers carry 2.6x the winner's
part of that gradient. If the decoder grades the winner cell (`tul.spandec_read: winner`), do CE
and the loop's own contribution change?

## Hypothesis

Two accounts, and this run separates them.

- Mismatch is waste: the loop's main teacher grades a readout the model never uses. Grading the
  read cell aligns the decoder with the coda, so CE improves and K1-K6 holds or rises.
- Mean is a selection pool: all 4 exit cells start from the same reset winner and share weights,
  so the decoder's gradient through the losers keeps every candidate decodable, which the router
  needs. Taking it away lets the losers decay: the gap between the best cell and the routed cell
  grows, and CE gets worse.

My best guess is a small CE gain with K1-K6 unchanged, but I hold it weakly.

## Predictions

References, `lxtul_pointer_fast` from init at 5k, seed 1, 480 rows (2026-10-08 filings): CE@6
3.9248 / 3.9229 (mean 3.9239), K1-K6 +0.0104 / +0.0091 (mean +0.0098), final val 3.9161 / 3.9129,
`fan/oracle_gap` 0.0309 / 0.0347, `fan/mixed_ce` 4.5499 / 4.5415. Spread between independent 5k
runs: about 0.006 in K1-K6, 0.002 in CE@6.

Read as the mean of two arm runs:

- CE@6 best guess 3.918. At or below 3.918 (0.006 better, 3x the reference spread) = the winner
  read helps. At or above 3.930 = it harms. Between = no CE effect.
- K1-K6 best guess +0.011. Below +0.006 = harm (the C1d rule). Above +0.016 = help.
- Final val best guess 3.910.
- `fan/oracle_gap` best guess 0.040. At or above 0.045 is the selection-pool signature (losers
  decayed relative to the best cell).
- Trainer rate late within 2 % of the references' (no new compute: one gather).

Decision rule: help or no CE effect, with K1-K6 at or above +0.006: I propose to Wolfe that
`lxtul_pointer_fast` take `spandec_read: winner`. Harm on CE or K1-K6: the mean read is doing
work for the selection; file it, keep the mean, and the next arm is a relaxed read (winner plus a
small loser weight).

## Method

`--config-name lxtul_pointer_fast_spdwin` (= `lxtul_pointer_fast` + `tul.spandec_read: winner`),
from a frozen `git archive` of the commit that adds this file. Two runs, both seed 1 (the
references' seed; MORPH runs decorrelate within a few steps at a fixed seed), run one after the
other under `gpu.lock`, WANDB offline. Readout per run: `core_depth_sweep.py --ckpt
spdwin=lxtul_pointer_fast_spdwin=<step_5000> --depths 1,6 --rows 480 --batch 3`, the references'
rows and command. The fan readings come from each run's final val line.
