# Agent Note: Ungraded fan with a set-reader span decoder as the fan recipe

Status: proposed

## Problem

Wolfe is adopting the ungraded fan (4 independent cells, coda reads all 4, no router, no reset)
and wants its cell diversity kept. The span decoder it inherits from LXTUL grades the MEAN of the
cells and costs it 0.022 CE@6 at 5k. Dropping the decoder keeps the diversity but halves the
small depth signal (K1-K6 +0.0042 to +0.0025). Two decoder variants that grade cells one at a time
(each cell, best cell per span) collapse the cells (cos +0.81, +0.61).

## Proposal

Make `nowta_pointer_fast_spdcells.yaml` the fan recipe: `nowta_pointer_fast` +
`tul.spandec_reads_cells: true`. The decoder cross-attends to the 4 cells as a memory, as the
coda reads them. Measured at 5k, two runs (`lab/experiments/failures/2026-10-11-fan-set-reader-decoder.md`):
CE@6 3.8951, K1-K6 +0.0056, cell cos -0.28, 23.9k tok/s on tree-h (the pre-lever tree).

## Alternatives considered

- Fan without the decoder: 3.8993 CE@6, K1-K6 +0.0025, cos -0.28, 28.2k tok/s. The fallback; 15 %
  faster, half the depth signal, CE 0.004 worse.
- Mean-read decoder (the inherited default): 3.9212, +0.0042, cos -0.10.
- Each-cell decoder (`spandec_read: each`): 3.9032, +0.0062, cos +0.81. Collapses the cells.
- Best-cell decoder (`spandec_read: best`): 3.9066, +0.0063, cos +0.61 (one run). Collapses them.
- One-cell loop: 3.9013, +0.0021. No diversity by construction; Wolfe expects 4 cells to win
  after about 2B tokens.
- LXTUL (latent-selected loop): 3.919-3.932, +0.009-0.010; its cells become copies from pass 3
  (diversity term on passes 1-2 only, reset to the winner every pass).

## Acceptance criteria

- Wolfe's call.
- The set reader keeps cos below +0.3 and CE@6 at or below the no-decoder fan on any further
  run at 5k (two runs here).
- The cell lens on this checkpoint (`lab/divergence/cell_lens.py`) shows the cells still split
  the work (copy cell, combination value on novel tokens).
- MET 2026-10-11: master (with the speed levers and the e651c95b pool fix) runs it through
  validation, 260 steps, exit 0, 34.0k tok/s in training (tree-h: about 24k at 5k steps).

## Risks

- 5k only; the decoder's share may change with training length.
- Memory: the validation OOM on master was a pinned graph pool, fixed in e651c95b. A full 5k run
  on master passed (3.8954 CE@6, 29.2k tok/s, `successes/2026-10-11-fan-set-reader-on-master.md`).
- Decode cost: the decoder is training-only, so it does not touch inference.
- Related: [2026-10-04-lxtul-primary-candidate.md](../../implemented/architecture/2026-10-04-lxtul-primary-candidate.md)
  is the current locked base; this note does not supersede it until Wolfe adopts the fan.
