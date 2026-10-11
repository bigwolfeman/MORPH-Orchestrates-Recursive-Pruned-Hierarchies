# Ungraded fan without the span decoder: second run

Status: planned

## Question

Does the ungraded fan's 0.022 CE@6 gain from dropping the span decoder hold on a second run?
Round 3 (`failures/2026-10-10-credit-panel-round-3.md`) measured 3.8994 on one run against the
fan with the decoder at 3.9212 (two runs). The same-config spread is 0.020, so one run cannot
say it.

## Hypothesis

The span decoder costs the ungraded fan CE at 5k: its gradient pulls the cells toward the mean
read the decoder grades, while the coda reads the cells one by one.

## Predictions

- Run 2 CE@6: best guess 3.905, inside [3.885, 3.925].
- Run 2 K1-K6: +0.001 to +0.004.
- Two-run mean CE@6 at least 0.010 below the fan with the decoder (3.9212), that is at or below
  3.911.
- Cell cos (`val/slot_cell_pairwise_cos`) below 0.

Decision rule: two-run mean at or below 3.911 means propose to Wolfe that the fan recipe drops
the span decoder. Above 3.911 means no 5k difference; the decoder question stays open.

## Method

The same config and tree as round 3: `nowta_pointer_fast_nospandec` in
`/home/wolfe/morph-scratch/abq/tree-f`, 5000 steps, seed 1, the A/B runner and its 480-row
sweep. It does not use the 2026-10-11 speed levers, so it is a pure replicate.
