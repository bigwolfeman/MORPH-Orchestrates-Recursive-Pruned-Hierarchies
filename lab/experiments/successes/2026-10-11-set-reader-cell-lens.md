# Cell lens on the set-reader fan

Status: success

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

## Results

Run 2026-10-11 on the master checkpoint (`ab-spdcells-master/step_5000.pt`), 240 rows, 12,401
spans. Artifacts: `../results/2026-10-11-set-reader-cell-lens/` (the patch JSON, about 10 MB of
per-span columns, stays in `/home/wolfe/morph-scratch/abq/lens/patch_spdcells.json`).

Depth 6, CE in nats per token, coda only, no pointer head. "Worth" = CE with no cell written
minus CE with the given cells written.

| | cell 0 | cell 1 | cell 2 | cell 3 | all four | per-span best single | none |
|---|---|---|---|---|---|---|---|
| all tokens CE | 4.6084 | 4.5914 | 4.7525 | 4.7254 | 4.5198 | 4.5281 | 4.6742 |
| copied tokens CE | 3.6490 | 3.6671 | 3.8223 | 3.8049 | 3.5948 | 3.6031 | 3.7764 |
| novel tokens CE | 5.7236 | 5.6659 | 5.8337 | 5.7954 | 5.5950 | 5.6034 | 5.7178 |
| win share | 0.58 | 0.30 | 0.06 | 0.07 | | | |

Copied worth: all four 0.182, cell 0 alone 0.127 (70 %). Novel worth: all four 0.123, best
single (cell 1) 0.052. Cells 2 and 3 alone are worse than no cell on both token kinds.
Specialisation NMI: first-token class 0.0406, length 0.0115, copy ratio 0.0057, difficulty
0.0023; null p95 at most 0.0009.

Beside the plain ungraded fan (`failures/2026-10-10-cell-lens.md`, depth 6): total worth 0.154
vs 0.148; copied 0.182 vs 0.170; novel 0.123 vs 0.123; cell 0's copied share 70 % vs 74 %.

Predictions:
- One cell carries at least 50 % of the copied worth: cell 0, 70 %. Held.
- All-cells novel worth at least 2x the best single cell's: 0.123 vs 0.052, 2.4x. Held.
- All-cells CE at least 0.005 below the per-span best single cell: 0.0083. Held.
- Top win share in [0.35, 0.65]: 0.58. Held.
- Largest NMI above the null's p95: 0.0406 vs 0.0009. Held.

## Verdict

Success. The set reader keeps the ungraded fan's division of labour: cell 0 is the copy cell,
the novel-token value lives in the combination, and the cells combine beyond the best single
cell per span. Its total worth is 0.006 nats higher than the plain fan's, all of it on copied
tokens. One change: cell 1 carries more alone (novel worth 0.052 vs about 0.02), while cells
2 and 3 are useful only with the others.

## Updated hypothesis

The set-reader decoder trains the cells toward the same division the coda uses, so its depth
signal (K1-K6 +0.005) comes without the collapse the single-target decoders caused. One
checkpoint, one lens run; the plain fan's numbers are also one checkpoint.
