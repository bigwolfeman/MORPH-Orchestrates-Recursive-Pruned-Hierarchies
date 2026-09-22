# Planned: seed twins of the single-block reach arm — learned trade or run draw

Status: planned

Date: 2026-09-22 (frozen before any GPU step; committed before the runner lines are
appended). Follow-up named in
[`failures/2026-09-21-span-reach-split.md`](../failures/2026-09-21-span-reach-split.md)
("a twin ... would separate the two"), held there until Step 1 read; Step 1 has read
(`2026-09-22-lxtul-r-step1.md`). Two plain arms, about 100 minutes on the 5090, no code
change beyond two seed configs.

## Question

`budget-web-reach1-coda` (reach 1 at coda[0] only, seed 1) reads 0.044 nats WORSE than
`budget-web-span-h` (seed 1) at 5k, depth 6, with a structured offset shape: 0.39 better on
the first token after a boundary, 0.12 on the second, 0.04 on the third, then worse from
the fifth on and 0.106 worse from the eighth. Eval swaps showed the mask is harmless to a
model trained without it (−0.001) and the trained model uses the read at every offset
(removing it costs 0.108). Is the tail deficit a property of training with one
previous-span block (it reproduces at another seed), or a run draw (it does not)?

## Hypothesis

Learned. The coda's first block is the one place the previous span can be read, so the
model spends that block's attention on it; the gain on the boundary tokens is large and
the remaining two coda blocks cannot recover the own-span long-range read the block used
to do. Against it: the same-config cross-commit pair already drifts 0.048 at offset 8+
(`span_budget_profile_drift_*`), half the deficit, so a draw of twice that size is not
excluded by one arm.

## Method

Two arms at HEAD, `training.seed: 2`, otherwise the `budget_root` recipe (seq 1024, batch 6,
5,000 steps, warmup 1000, `norm_match`, prune / carve / route off):

| arm | config | what |
|---|---|---|
| `budget-web-span-s2` | `budget_web_span_s2` | own span only, seed 2 (twin of span-h) |
| `budget-web-reach1-coda-s2` | `budget_web_reach1_coda_s2` | reach 1 at coda[0] only, seed 2 (twin of reach1-coda) |

Runner kind `plain`, sweeps at 2500 and 5000, 480 rows. Readouts at 5k, depth 6, paired
on `tok_index`: coda-s2 − span-s2 (the effect at seed 2, whole-arm and per offset via
`span_budget_profile.py`); span-s2 − span-h (the seed floor, whole-arm and per offset);
coda-s2 − coda (the coda arm's own seed floor).

## Predictions (frozen)

- **P-1 (survival).** Both arms HEALTHY to 5,000. **90 %.**
- **P-2 (the seed floor).** |span-s2 − span-h| whole-arm under **0.06**, and its 8+ bin
  under **0.06** in magnitude. **75 %.** The cross-commit pair read 0.041 / 0.048.
- **P-3 (the effect reproduces, the headline).** coda-s2 − span-s2 at offset 1 at or
  below **−0.25** AND at 8+ at or above **+0.05**. **60 %.** Learned: both signs
  reproduce. If offset 1 reproduces and 8+ reads inside [−0.05, +0.05]: the tail deficit
  was a draw and the boundary gain is the learned part (25 %). Neither reproduces: 15 %.
- **P-4 (the whole-arm sign).** coda-s2 − span-s2 whole-arm at or above **+0.01**
  (the one-block read a net loss again). **55 %.**

## Binding

- P-3 holds: the single-block read is a learned trade; the LXTUL-R note's two-channel
  design must not route history through ONE coda block (it needs the reach at every coda
  block or a separate channel), and the Step 0 filing gets a dated correction saying so.
- P-3's second branch (boundary gain reproduces, tail deficit does not): the Step 0
  "lower bound" of the previous-span value is re-read from the seed-2 pair as
  span-s2 − coda-s2, and the relay share (P-8 there) is computed from it with a dated
  correction.
- P-2 fails: the seed floor at 5k is above 0.06 and every n=1 whole-arm comparison on
  this panel is re-labelled as inside the noise; a dated correction goes on the Step 0
  filing and on `memory` (the 0.05 floor).

## Not verified before launch

- No GPU step at seed 2 on this panel; the smoke is the first.
- Whether `training.seed` alone changes the data order on this loader (if the row order
  is seed-independent, the twins differ by init and dropout only; the filing states
  which by reading the first-batch token ids of the two run logs if printed, else marks
  it unknown).
