# Ungraded fan with the span decoder grading each cell

Status: failure

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

## Results

Run 2026-10-11, two runs, 480 rows. Artifacts: `../results/2026-10-11-fan-each-cell-decoder/`.

| Fan arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| each-cell decoder, run 1 | 3.9092 | 3.9034 | +0.0059 [+0.0054, +0.0063] | 0.186 | +0.81 | 18,551 |
| each-cell decoder, run 2 | 3.9093 | 3.9029 | +0.0064 [+0.0059, +0.0069] | 0.188 | +0.82 | 18,549 |
| mean-read decoder (2 runs) | | 3.9212 | +0.0042 | 0.18 | -0.09 / -0.11 | 24,665 |
| no decoder (2 runs) | | 3.8993 | +0.0025 | 0.19 | -0.28 / -0.27 | 28,230 |

Two-run mean: CE@6 3.9032, K1-K6 +0.0062, span-decoder CE K1-K6 +0.0465.

Predictions:
- CE@6 in [3.895, 3.925], best guess 3.910: 3.9032. Held.
- K1-K6 at least +0.003, best guess +0.006: +0.0062. Held.
- Cell cos below 0: +0.81 / +0.82. Failed.
- Rate 15-25 % below the mean-read decoder: 24.8 % below. Held.

Decision rule: the two-run mean (3.9032, +0.0062) lands in "supports the mean read account".

## Verdict

Failure, on the hypothesis's own mechanism. The each-cell grade removes the mean-read cost on CE
(0.018 better than the mean-read decoder, two runs each side), but it does NOT keep the cells
distinct: they move from cos -0.1 to +0.81. Graded alone on the same next span, the four cells
learn the same job. The two-run spreads are 0.0005 CE@6 here and 0.0003 without the decoder, so
the 0.004 CE@6 gap to the fan without the decoder is likely real at this recipe.

Trade at 5k, against the fan without the decoder: +0.004 CE@6 and 34 % slower, for 2.5x the
K1-K6 (+0.0062 against +0.0025) and cells that are near copies.

## Updated hypothesis

The decoder's cost to the fan comes from the target it grades, not from its gradient as such:
a mean target costs 0.022 CE@6 and keeps the cells apart; a per-cell target costs 0.004 and
collapses them. Any decoder that grades every cell on the SAME span pulls them together. A decoder
that keeps diversity would have to grade the cells as a set (for example the best cell per span,
or the combination the coda reads), not each cell alone. Wolfe wants the diversity kept, so the
fan without the decoder stays the proposal.
