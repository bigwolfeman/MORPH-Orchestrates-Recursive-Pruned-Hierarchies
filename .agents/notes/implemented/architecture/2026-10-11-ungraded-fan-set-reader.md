# Agent Note: Ungraded fan with a set-reader span decoder as the fan recipe

Status: implemented

## Problem

Wolfe adopts the ungraded fan (4 independent cells, the coda reads all 4, no router, no reset)
and wants its cell diversity kept. The span decoder it inherits from LXTUL grades the MEAN of the
cells and costs it 0.022 CE@6 at 5k. Dropping the decoder keeps the diversity but halves the
small depth signal (K1-K6 +0.0042 to +0.0025). Two decoder variants that grade cells one at a time
(each cell, best cell per span) collapse the cells (cos +0.81, +0.61).

## Decision

Wolfe, 2026-10-11: the fan recipe is `morph/configs/nowta_pointer_fast_spdcells.yaml`,
`nowta_pointer_fast` + `tul.spandec_reads_cells: true`. This was his original design. The span
decoder cross-attends to the 4 cells as an unordered set (`_SpanDecCross`,
`morph/model/tul_spandec.py`), the way the coda reads them; its `z` conditioning stays the mean.
The decoder is training-only. New loop and speed work starts from this config; the LXTUL base
([2026-10-04](2026-10-04-lxtul-primary-candidate.md)) is superseded as the loop base.

Measured at 5k, 480 rows, seed 1:

| Run | Tree | CE@6 | K1-K6 | cell cos | trainer tok/s |
|---|---|---|---|---|---|
| run 1 | tree-h | 3.8949 | +0.0052 | -0.28 | 23,962 |
| run 2 | tree-h | 3.8953 | +0.0060 | -0.28 | 23,933 |
| master | tree-j (58ef105e) | 3.8954 | +0.0051 | -0.28 | 29,209 |

Filings: `lab/experiments/failures/2026-10-11-fan-set-reader-decoder.md` (CE prediction missed
low), `successes/2026-10-11-fan-set-reader-on-master.md`, `successes/2026-10-11-set-reader-cell-lens.md`
(cell 0 carries 70 % of the copied worth; novel worth needs the combination, 2.4x the best
single cell).

## Alternatives considered

- Fan without the decoder: 3.8993 CE@6, K1-K6 +0.0025, cos -0.28, 28.2k tok/s on the pre-lever
  tree (the set reader read 23.9k there, so the decoder costs about 15 % of throughput). The fallback.
- Mean-read decoder (the inherited default): 3.9212, +0.0042, cos -0.10.
- Each-cell decoder (`spandec_read: each`): 3.9032, +0.0062, cos +0.81. Collapses the cells.
- Best-cell decoder (`spandec_read: best`): 3.9066, +0.0063, cos +0.61 (one run). Collapses them.
- One-cell loop: 3.9013, +0.0021. No diversity by construction; Wolfe expects 4 cells to win
  after about 2B tokens.
- LXTUL (latent-selected loop): 3.919-3.932, +0.009-0.010; its cells become copies from pass 3
  (diversity term on passes 1-2 only, reset to the winner every pass).

## Consequences

- Every run here is seed 1. The 0.0005 agreement is run-to-run nondeterminism, not seed
  variance. The margin over the no-decoder fan (0.004 CE@6) has no second seed behind it.
- The fan gives up LXTUL's 1-cell-at-inference option: its 4 cells are independent lineages
  and the coda reads all of them, so decode pays the 4-cell loop.
- Loop contribution is small (K1-K6 +0.005 against LXTUL's +0.02). The trade is CE and cell
  diversity at 5k for depth use.
- The cached and graphed generators refuse the fan (`_check_supported`, `tul.fan_k > 0`); a
  cached decoder for the write-all read is open work in the phase-2 tracker.
- The set reader does not depend on the cell count; the coda's per-cell `W_prefix[i]` and
  prefix positions do. Variable cell count
  ([2026-10-05](../../proposed/architecture/2026-10-05-lxtul-variable-cell-count.md)) needs a
  coda write that is shared across cells.
- 5k only; the decoder's share may change with training length.
