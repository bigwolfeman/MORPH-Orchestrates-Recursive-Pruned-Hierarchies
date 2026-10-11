# Cell lens on the set-reader fan

Status: planned

## Question

Do the set-reader fan's cells still split the work the way the ungraded fan's did
(`failures/2026-10-10-cell-lens.md`: cell 0 carried 0.126 of the 0.170 copied-token worth; the
novel-token worth, 0.123, came mostly from the cells together; first-token-class NMI 0.051)?

## Hypothesis

The set reader keeps the cells distinct (cos -0.28) because it reads them as the coda does, so
the coda's split survives: one copy cell and a combination code for novel tokens.

## Predictions

Patch at depth 6, coda only, 240 rows (`--part patch`):
- One cell carries at least 50 % of the all-cells copied-token worth (zero minus all).
- All-cells novel-token worth at least 2x the best single cell's novel-token worth.
- All-cells CE at least 0.005 below the per-span best single cell (the cells combine).
- The top cell's win share in [0.35, 0.65].
Special (`--part special`): the largest NMI above the permutation null's p95.

## Method

`lab/divergence/cell_lens.py` (master) `--part patch --depths 1,6 --n-val 40` on
`fan=nowta_pointer_fast_spdcells=/home/wolfe/morph-scratch/abq/tree-j/checkpoints/morph/ab-spdcells-master/step_5000.pt`
with `/home/wolfe/morph-scratch/abq/lens/batches.pt` (packed for nowta_pointer_fast, prefix_k
4), under gpu.lock; then `--part special` on its JSON (CPU).
