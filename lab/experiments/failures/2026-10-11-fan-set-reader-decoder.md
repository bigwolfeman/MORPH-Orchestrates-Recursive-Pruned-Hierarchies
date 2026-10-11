# Ungraded fan with a span decoder that reads the cells as a set

Status: failure

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

## Results

Run 2026-10-11 in tree-h, two runs, 480 rows. Artifacts: `../results/2026-10-11-fan-set-reader-decoder/`.

| Fan arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| set-reader decoder, run 1 | 3.9001 | 3.8949 | +0.0052 [+0.0047, +0.0056] | 0.177 | -0.28 | 23,962 |
| set-reader decoder, run 2 | 3.9013 | 3.8953 | +0.0060 [+0.0055, +0.0065] | 0.180 | -0.28 | 23,933 |
| no decoder (2 runs) | | 3.8993 | +0.0025 | 0.19 | -0.28 | 28,230 |
| mean-read decoder (2 runs) | | 3.9212 | +0.0042 | 0.18 | -0.10 | 24,665 |
| each-cell decoder (2 runs) | | 3.9032 | +0.0062 | 0.19 | +0.81 | 18,550 |
| best-cell decoder (1 run) | | 3.9066 | +0.0063 | 0.19 | +0.61 | 15,994 |

Two-run mean: CE@6 3.8951, K1-K6 +0.0056, span-decoder CE K1-K6 +0.033. The one-cell loop: 3.9013.

Predictions:
- Cell cos below +0.3: -0.28 / -0.28. Held.
- CE@6 in [3.900, 3.925], best guess 3.912: 3.8951. Failed, low: 0.017 better than the guess and
  below the range.
- K1-K6 at least +0.003: +0.0056. Held.
- Rate 0-10 % below the mean-read decoder: 2.9 % below. Held.

Decision rule: cos below +0.3, CE@6 at or below 3.905, K1-K6 at least +0.004, on both runs. Met.

## Verdict

Filed under failures because the CE prediction missed, low, by a wide margin: my model of what
a set reader costs was wrong. The decision rule is met on two runs that agree to 0.0004 CE@6.

What it shows: a next-span decoder does NOT have to collapse the fan's cells. The mean read
(one target from the mean), the each-cell grade (four cells, one target each) and the best-cell
grade (the winner per span) all pulled the cells together or cost CE. The set reader, which
cross-attends to all four cells as the coda does, keeps them as distinct as no decoder (cos
-0.28), is the best fan arm on CE@6 (0.004 below no decoder, with run spreads of 0.0004 and
0.0003), doubles K1-K6 over no decoder, and costs 15 % tok/s against no decoder.

## Updated hypothesis

A decoder's target pulls the cells toward what its READ needs. A read that takes one vector per
grade (the mean, or one cell) is served best by cells that each hold the whole span, so they
converge. A read that takes the cells as a set, like the coda's, is served by cells that hold
distinct parts, so the decoder's gradient and the coda's agree. Proposal (Wolfe decides): the
ungraded fan with the set-reader decoder is the fan recipe
(`.agents/notes/proposed/architecture/2026-10-11-ungraded-fan-set-reader.md`). Untested: the
lens on this checkpoint (does cell 0 still carry copying), and the 20k horizon.
