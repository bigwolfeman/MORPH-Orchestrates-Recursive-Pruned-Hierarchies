# Ungraded fan without the span decoder: second run

Status: success

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

## Results

Run 2026-10-11, one run, 480 rows. Artifacts: `../results/2026-10-11-fan-no-decoder-replicate/`.

| Arm | CE@1 | CE@6 | K1-K6 [95% CI] | write worth | cell cos | trainer tok/s |
|---|---|---|---|---|---|---|
| ungraded fan, no span decoder, run 1 | 3.9018 | 3.8994 | +0.0024 [+0.0020, +0.0027] | 0.188 | -0.28 | 28,157 |
| ungraded fan, no span decoder, run 2 | 3.9017 | 3.8991 | +0.0026 [+0.0022, +0.0029] | 0.185 | -0.27 | 28,304 |

Two-run mean CE@6 3.8993, K1-K6 +0.0025. The fan with the decoder: 3.9212 (two runs, 3.9196 and
3.9227). The one-cell loop: 3.9013 (two runs).

Predictions:
- Run 2 CE@6 in [3.885, 3.925], best guess 3.905: 3.8991. Held.
- Run 2 K1-K6 +0.001 to +0.004: +0.0026. Held.
- Two-run mean at or below 3.911: 3.8993. Held.
- Cell cos below 0: -0.27. Held.

## Verdict

Success. The two runs agree to 0.0003 CE@6. Dropping the span decoder makes the ungraded fan 0.022
better on CE@6 (two runs each side, each side's spread 0.003 or less) and 14 % faster. The cost is
K1-K6: +0.0025 against +0.0042. By the decision rule, the proposal to Wolfe is that the fan recipe
drops the span decoder. The fan without the decoder and the one-cell loop tie (3.8993 against
3.9013, inside noise).

## Updated hypothesis

The span decoder costs the ungraded fan CE at 5k. It grades the MEAN of the cells, while the
coda reads the cells one by one and uses them as distinct parts (cell 0 carries cross-span copying;
`failures/2026-10-10-cell-lens.md`). The mechanism is not tested: a fan whose decoder grades each
cell (`spandec_read: each`) would separate "the decoder's gradient" from "the decoder grades the
mean". Next: that arm on the ungraded fan, if Wolfe wants the depth signal back.
