# Ungraded fan with the span decoder grading the best cell per span

Status: planned

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
