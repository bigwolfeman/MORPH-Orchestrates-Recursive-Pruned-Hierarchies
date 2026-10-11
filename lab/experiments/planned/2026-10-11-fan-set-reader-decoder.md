# Ungraded fan with a span decoder that reads the cells as a set

Status: planned

## Question

Every next-span decoder tried on the ungraded fan so far either cost CE or collapsed the cells:
mean read 3.9212 CE@6 at cos -0.10; each cell 3.9032 at +0.81; best cell 3.9066 at +0.61; no
decoder 3.8993 at -0.28 (`failures/2026-10-11-fan-each-cell-decoder.md`,
`failures/2026-10-11-fan-best-cell-decoder.md`). Does a decoder that reads the 4 cells as a set,
by cross-attention as the coda does (`tul.spandec_reads_cells`), keep them distinct and pay less?

## Hypothesis

The coda uses the cells as distinct parts. A decoder that reads them the same way can be served
by distinct cells, so its gradient does not need to pull them together. Its z conditioning is
still the mean (the option's design), so part of the mean-read cost may remain.

## Predictions

- Cell cos below +0.3, best guess -0.05.
- CE@6 in [3.900, 3.925], best guess 3.912.
- K1-K6 at least +0.003, best guess +0.005.
- Trainer rate 0-10 % below the mean-read decoder (24,665 tok/s): one cross-attention per
  decoder layer over 4 cells.

Decision rule: cos below +0.3 and CE@6 at or below 3.905 with K1-K6 at least +0.004 makes the
set reader the fan's decoder candidate (second run owed). Otherwise the fan without the decoder
stays the proposal.

## Method

Config `nowta_pointer_fast_spdcells.yaml` = `nowta_pointer_fast` + `tul.spandec_reads_cells:
true` (existing option, zero-init cross-attention, so step 0 is the mean-read decoder). Built on
CPU in tree-h: 306.7M params. Run in `/home/wolfe/morph-scratch/abq/tree-h` (tree-f + the
best-cell diff; the reference arms' code), row cap as the recipe sets it (on; this option does
not use the best-cell path, so the missing helper is not reached), 5000 steps, seed 1, the A/B
runner and its 480-row sweep.

Method amendment 2026-10-11 (after run 1, before run 2 is queued): run 1 landed in the
decision rule's "candidate" box, which owes a second run. Run 2 (`nowta-spdcells-r2`, same
config, tree and seed) is queued now; the file is filed on both runs, against the same
Predictions, judged on the two-run mean.
