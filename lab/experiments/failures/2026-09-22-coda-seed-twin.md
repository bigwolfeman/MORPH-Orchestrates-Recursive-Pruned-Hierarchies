# Planned: seed twins of the single-block reach arm — learned trade or run draw

Status: failure

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

## Results

Two arms on the runner at 599ae1d, 2026-09-22 03:21 to 05:03 (28 s and 29 s smokes, 5,000
steps each at about 1.98 steps/s). Readouts as Method: the runner's sweeps at 2500 and
5000, `paired_vs_ruler.py` on `tok_index` (481 blocks), `span_budget_profile.py` per
offset. Artifacts: `../results/2026-09-22-coda-seed-twin/`. Data order is seed-independent
(no shuffle in `morph/training/data.py`; the only fixed generator in the trainer feeds the
compile-warmup buffer), so the twins differ by init, dropout and depth draws only.

| arm | verdict (`preclip/total` max @ step) | final val | CE d1 | CE d6 |
|---|---|---|---|---|
| `budget-web-span-s2` | HEALTHY, 19.1 @ 3005 | 4.4240 | 4.4344 | 4.4014 |
| `budget-web-reach1-coda-s2` | HEALTHY, 20.9 @ 612 | 4.3060 | 4.3256 | 4.2893 |
| `budget-web-span-h` (seed 1) | HEALTHY, 14.1 @ 1284 | 4.4168 | 4.4300 | 4.3975 |
| `budget-web-reach1-coda` (seed 1) | HEALTHY, 38.3 @ 2048 | 4.4539 | 4.4680 | 4.4414 |

Paired depth-6 readings (A − B, positive = A worse):

| reading | value |
|---|---|
| span-s2 − span-h (the seed floor, P-2) | **−0.0039 [−0.0062, −0.0013]**; every offset bin inside ±0.01 (8+: −0.004) |
| coda-s2 − span-s2 (the effect at seed 2, P-3 / P-4) | **−0.1121 [−0.1168, −0.1077]** |
| coda-s2 − coda (the coda config's own seed spread) | **−0.1521 [−0.1559, −0.1487]** |
| coda-s2 − reach1 (ordering) | +0.0712 [+0.0683, +0.0741] |
| coda-s2 − reachall (the far budget's upper bound, re-read) | +0.1971 [+0.1872, +0.2081] |

Per offset, coda-s2 against span-s2 (negative = coda better), depth 6: **−0.076** at 0,
**−0.469** at 1, −0.231 at 2, −0.172 at 3, −0.151 at 4, −0.129 at 5, −0.113 at 6, −0.095 at 7,
**−0.069** at 8+. The seed-1 pair read +0.003, −0.392, −0.121, −0.039, −0.005, +0.016,
+0.046, +0.068, **+0.106** on the same bins.

Scoring:

- **P-1 HOLDS.** Both arms healthy; tripwire maxima 19 and 21.
- **P-2 HOLDS.** The seed floor on the plain span arm is 0.004 whole-arm and under 0.01 in
  every bin. The 0.041 between `span` at cc4e034 and `span-h` at 66fc75c is therefore NOT
  seed noise; it is code or nondeterminism drift across ten days of commits on a path the
  tests call bit-identical at reach 0, and it is now an open item.
- **P-3 FAILS.** The boundary gain reproduces and grows (−0.469 at offset 1 against
  −0.392) but the tail deficit does not: 8+ reads −0.069 (coda BETTER), against the clause
  "at or above +0.05". Not the second branch either (that needed 8+ inside ±0.05): the
  seed-2 arm is better than its span twin at every offset.
- **P-4 FAILS.** Whole-arm −0.112, against "at or above +0.01".

## Verdict

**Failure** (P-3 and P-4 fail, 2 of 4 hold): the hypothesis "learned trade" is refuted.
The seed-1 `budget-web-reach1-coda` arm was a bad draw OF THAT CONFIG: two seeds of the
single-block-reach config differ by **0.152** nats at depth 6 on the same rows, while two
seeds of the plain span config differ by 0.004. At seed 2 one non-looped block reading the
previous span is worth 0.112 nats, at every offset, with the largest part on the first
three tokens after a boundary. The Step 0 filing's coda paragraph, its "learned
single-block trade" reading and the brackets it declined to read are corrected by a dated
note there; the design note's Outcome log gets the same correction.

What this does to Step 0's brackets (seed 2 for the coda arm, seed 1 for the rest; the
plain seed floor is 0.004): previous-span value in **[0.112, 0.179]**; far budget in
**[0.126, 0.197]**; relay share **0.071** (inside the prereg's [0.02, 0.12]); ordering
span ≥ coda ≥ reach1 ≥ reachall holds; the sum 0.112 + 0.197 + 0.053 = 0.362 against the
0.358 endpoint.

What one pair still cannot say: whether the seed-1 coda run sits in a distinct basin (a
bimodal config) or on the tail of a wide one. A third seed would say; it is not queued,
because the brackets above are read at seed 2 with the plain floor as the error bar, and
the LXTUL-R line does not depend on it. The warning stands for any design that routes
history through ONE block: its outcome varied by 0.15 nats across two seeds at 5k.

## Updated hypothesis

A previous-span read at one non-looped block is worth about 0.11 nats at 5k when training
lands well, on every token of the span, and the split of the 0.40-nat cross-span budget is
roughly a third previous span (0.11 to 0.18), a third to a half further back (0.13 to
0.20) and the rest local routes and relay. The single-block read is the one arm on this
panel with a large seed variance; the plain span arm's is 0.004. Seed variance of a
config is a property to measure before reading any n=1 comparison on it, and the plain
panel's 0.004 is the floor to quote, not the cross-commit 0.041.
