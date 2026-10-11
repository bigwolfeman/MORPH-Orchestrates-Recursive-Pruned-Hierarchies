# Ungraded fan with the span decoder grading the best cell per span

Status: failure

## Question

Can a span decoder train the ungraded fan's cells without collapsing them? Measured, two runs
each: the mean-read decoder costs 0.022 CE@6 and keeps the cells apart (cos -0.10); the
each-cell decoder costs 0.004 CE@6 and collapses them (cos +0.81,
`failures/2026-10-11-fan-each-cell-decoder.md`). No decoder: 3.8993 CE@6, K1-K6 +0.0025, cos
-0.28.

## Hypothesis

Grading the cells as a set keeps them distinct. Under `tul.spandec_read: best` each span's
decoder loss is the minimum over the 4 cells, and only that cell gets the gradient, so a cell is
trained only on the spans it already decodes best (specialisation). The risk is rich-get-richer:
one cell wins most spans and the decoder trains it alone.

## Predictions

- Cell cos (`val/slot_cell_pairwise_cos`) below +0.3 (best guess 0.0).
- CE@6 in [3.893, 3.915], best guess 3.903.
- K1-K6 at least +0.003, best guess +0.005.
- The largest final win share (`spandec_best_win_c{i}`) in [0.35, 0.90]; the smallest above 0.02.
- Trainer rate within 5 % of the each-cell decoder (18,550 tok/s): the decoding work is the same.

Decision rule: cos below +0.3 and CE@6 at or below 3.905 with K1-K6 at least +0.004 makes the
best-cell decoder the fan's decoder candidate (second run owed). Cos at or above +0.3 means the
set grade does not keep diversity either; the fan without the decoder stays the proposal.

## Method

`tul.spandec_read: best` (branch feat/spandec-best: `_spandec_best_of_cells` in
`morph/model/transformer.py`; per-span token-mean CE from the decoder's own label-logprob kernel,
argmin over cells, gather, token-weighted mean; `spandec_ce_row_cap` supported). Tests
`tests/test_spandec_read_best.py` 10 passed CPU (hand reference on the ungraded fan and LXTUL,
exact-zero gradient to losing cells, win stats, row cap on/off, refusals); sabotage (mean over
cells instead of min) fails 4. Config `nowta_pointer_fast_spdbest.yaml`. Run from a frozen
snapshot of the commit adding this file (`/home/wolfe/morph-scratch/abq/tree-g`), 5000 steps, seed
1, the A/B runner and its 480-row sweep. This tree includes the 2026-10-11 byte-identical speed
levers (C3, E, Q, C4; C2 off), so its tok/s is not comparable 1:1 to tree-f arms; CE is.

Method amendment 2026-10-11 04:24 (committed before the rerun was queued): the first launch (tree-g, master + this change) ran out of
GPU memory at the first validation, step 200 (22 GB in CUDA-graph private pools; the
2026-10-11 speed levers, likely C4's second recorded body, leave less headroom than the tree the
references trained on). No numbers were read. Rerun as `nowta-spdbest-h` in
`/home/wolfe/morph-scratch/abq/tree-h` = tree-f (the references' tree) + this change's diff,
with `model.spandec_ce_row_cap=false`, because tree-f has no `label_logprob_static_rows`. The
row cap changes only the decoder CE's sum order. In tree-h `tests/test_spandec_read_best.py`
gives 9 passed, and 1 failed: the cap-on test, by NameError for that missing helper.
Predictions unchanged; the tok/s prediction now compares tree-f to tree-f.

## Results

Run 2026-10-11 in tree-h, one run, 480 rows. Artifacts: `../results/2026-10-11-fan-best-cell-decoder/`.
The runner's own sweep failed (it does not pass the run's override, so the model built with the
row cap on and hit the NameError the amendment names). The sweep was rerun by hand with
`model.spandec_ce_row_cap=false` on the same checkpoint. The win shares were never logged by the
trainer (a gap in the build, fixed in the same change as this filing); they were read offline
from the step-5000 checkpoint on 40 val batches (`best_wins.py`).

| Fan arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| best-cell decoder (1 run) | 3.9129 | 3.9066 | +0.0063 [+0.0058, +0.0069] | 0.187 | +0.61 | 15,994 |
| each-cell decoder (2 runs) | | 3.9032 | +0.0062 | 0.19 | +0.81 | 18,550 |
| no decoder (2 runs) | | 3.8993 | +0.0025 | 0.19 | -0.28 | 28,230 |
| mean-read decoder (2 runs) | | 3.9212 | +0.0042 | 0.18 | -0.10 | 24,665 |

Win shares (val): 0.253, 0.245, 0.239, 0.263. Span-decoder CE K1-K6 +0.0610.

Predictions:
- Cell cos below +0.3: +0.61. Failed.
- CE@6 in [3.893, 3.915]: 3.9066. Held.
- K1-K6 at least +0.003: +0.0063. Held.
- Largest win share in [0.35, 0.90], smallest above 0.02: largest 0.263. Failed (no cell took
  over; the shares are near uniform).
- Rate within 5 % of the each-cell decoder: 14 % below. Failed. Confound: this run had the
  decoder's row cap off, the each-cell runs had it on.

Decision rule: cos at or above +0.3, so the fan without the decoder stays the proposal.

## Verdict

Failure. The set grade does not keep the cells distinct. The cells share the spans almost evenly,
and they still move from -0.28 (no decoder) to +0.61. That is less than the each-cell grade
(+0.81) but far from the no-decoder fan. CE@6 is 0.007 worse than no decoder (one run; inside
noise of the each-cell arm). The rich-get-richer risk did not happen.

## Updated hypothesis

Any span-decoder target that asks a cell to decode the next span pulls the cells toward the
same content, even when each span trains only one cell: four cells that each win a quarter of
the spans still learn one shared code (cos +0.61). The fan's cells stay distinct only when
nothing but the coda trains them, and the coda uses them as distinct parts (cell 0 copies). So
the fan's depth signal cannot come from a next-span decoder without paying in diversity. Wolfe
wants the diversity kept: the fan without the decoder stays the proposal. Not tested: a decoder
that reads the cells the way the coda does (all four as a set, `spandec_reads_cells`).
