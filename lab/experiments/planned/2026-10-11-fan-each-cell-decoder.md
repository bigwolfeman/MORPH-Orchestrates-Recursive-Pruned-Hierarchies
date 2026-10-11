# Ungraded fan with the span decoder grading each cell

Status: planned

## Question

Why does the span decoder cost the ungraded fan 0.022 CE@6 at 5k
(`successes/2026-10-11-fan-no-decoder-replicate.md`)? Two accounts:
- MEAN READ: the decoder grades the MEAN of the 4 cells, while the coda reads the cells one by
  one as distinct parts (cell 0 carries copying; `failures/2026-10-10-cell-lens.md`). The mean
  target pulls the cells toward a shared job.
- DECODER GRADIENT: any decoder gradient into the cells costs the coda, however it reads them.

## Hypothesis

The mean read is the cost. Grading each cell alone (`tul.spandec_read: each`) keeps the cells
distinct and keeps the decoder's depth signal.

## Predictions

Measured references (two runs each): fan with mean-read decoder 3.9212 CE@6, K1-K6 +0.0042;
fan without decoder 3.8993, K1-K6 +0.0025. Same-config spread 0.020 CE@6 (0.003 on these arms).
- CE@6 best guess 3.910, inside [3.895, 3.925].
- K1-K6 best guess +0.006, at least +0.003 (the each-cell grade on LXTUL gave +0.0166).
- Cell cos below 0. Trainer rate 15-25 % below the fan with the mean-read decoder (folding the 4
  cells into the decoder's batch; the LXTUL each-cell arm was 24 % below its reference).

Decision rule (one run): CE@6 at or below 3.905 with K1-K6 at least +0.004 supports the mean
read account and makes the each-cell decoder the fan's decoder candidate (second run owed). CE@6
at or above 3.915 supports the decoder-gradient account: drop the decoder. Between: inconclusive;
a second run decides.

## Method

New config `nowta_pointer_fast_spdeach.yaml` = `nowta_pointer_fast` + `tul.spandec_read: each`.
The guard in `TULConfig._check_spandec_read` is narrowed: `each` now needs `slot_cells > 1`
only, because it reads the register's cells and never the router's pick (`winner` keeps its
latent-selected-loop requirement). Tests `tests/test_spandec_read_each.py` 9 passed CPU (the loss
and per-cell gradient tests now also run on the ungraded fan; the old guard raises on both).
Run in `/home/wolfe/morph-scratch/abq/tree-f` (round 3's tree, with the narrowed guard copied
in), 5000 steps, seed 1, the A/B runner and its 480-row sweep.

Method amendment 2026-10-11 03:31: run 1 landed in the "supports the mean read account" box of
the decision rule (numbers withheld here until filing), which owes a second run. Run 2
(`nowta-spdeach-r2`, same config, tree and seed) is queued now; the file is filed on both runs,
against the same Predictions, judged on the two-run mean.
