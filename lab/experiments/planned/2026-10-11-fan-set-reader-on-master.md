# Fan with the set-reader decoder on the master tree

Status: planned

## Question

Does the fan + set-reader recipe (two runs on tree-h: CE@6 3.8951, K1-K6 +0.0056, cell cos
-0.28, `failures/2026-10-11-fan-set-reader-decoder.md`) reproduce on master, with the 2026-10-11
byte-identical speed levers (C3, E, Q, C4) and the graph-pool fix (e651c95b), and at what speed?

## Hypothesis

The levers are byte-identical on the training step, so the result is a third draw from the same
distribution, only faster.

## Predictions

- CE@6 in [3.885, 3.905], best guess 3.895.
- K1-K6 +0.003 to +0.008.
- Cell cos below 0.
- Trainer rate (third tok/s reading) at least 28,000 (tree-h read 23,950; the 260-step check on
  master read 34.0k at step 200).
- No out-of-memory at any validation.

## Method

`nowta_pointer_fast_spdcells`, 5000 steps, seed 1, from a frozen snapshot of the master commit
adding this file (`/home/wolfe/morph-scratch/abq/tree-j`), the A/B runner and its 480-row sweep.
