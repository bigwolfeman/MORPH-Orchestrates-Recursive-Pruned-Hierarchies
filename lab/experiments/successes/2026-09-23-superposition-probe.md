# Planned: does a trained slot cell hold a superposed frontier, a committed point, or a blur?

Status: success

Date: 2026-09-23 21:31 (frozen before any GPU read). Instrument:
`lab/divergence/superposition_probe.py` (the 3c instrument of the Reasoning by
Superposition re-read,
[`reasoning-by-superposition.md`](../../../docs/references/looping-depth/latent-exploration/reasoning-by-superposition/reasoning-by-superposition.md)).
Wolfe's go: 2026-09-23 ("I like 1 and 2 of the agents recommendation we should do this").
No training.

## Question

Reasoning by Superposition (arXiv 2505.12514) keeps a frontier of candidates in ONE
continuous thought and lets later evidence select among them. Our slot cells are
deterministic after LXTUL-GK (the learned noise switched itself off). Does a trained cell
already hold continuation content for more than the one branch it favours
(superposition), only for its favourite (committed point), or nothing span-specific
(blur)? The answer decides whether the loop's depth problem is representational.

## Hypothesis

A cell trained by the span decoder's CE is asked for a distribution over the next span,
so it can hold a superposition. The existing worth profile says the cell helps tokens deep
into the span (ruler, within-row shuffle: 0.32 nats at offset 1, 0.03 at offsets 16+).
My guess is that most of that help is branch-independent (topic and document content),
which reads as "superposition" under a rule that does not separate the two.

## Arms (checkpoints at step 5000, trained configs, no overrides)

| arm | config | why |
|---|---|---|
| ruler `slot-spandec-strict` | `tul_slot_spandec_strict` | the deterministic reference |
| `lxtul-gk1` | `tul_slot_spandec_strict_gk1` | GK, K = 1 (noise collapsed) |
| `lxtul-gk4-shared` | `tul_slot_spandec_strict_gk4` | GK, K = 4 (noise collapsed), read at gram prior seed 0 |
| `slot-spandec-strict-fan4-all` | `tul_slot_spandec_strict_fan4_all` | four cells per slot: room for a frontier |

## Instrument

Per scored span s (1 <= s < max_slots), `t0` is the label at offset 0, the first
position that reads the own cell (slot s−1's cells). Worth = CE(control) − CE(own) on the
same rows. Controls: `xrow` (another row's cells: document removed), `row` (another slot
of the same row: document kept, span-specific content removed), `zero`, and `xrow2` (the
decode-clustering null). Two bucketings of the rank of `t0`:

- **own bucket** (the spec): rank under the own cell. Biased toward "committed" because a
  good cell also ranks `t0` first.
- **context bucket** (added at review, 2026-09-23): rank under the `xrow` cell, which
  cannot know how good the own cell is. Top1 / top2 are the branches the context alone
  favours.

Ratio = worth(top2) / worth(top1) on offsets 1..n, block-bootstrap CI over 1,024-token
stream blocks. Rule on a ratio: superposition >= 0.5, committed < 0.25 (or top2 worth
< 0), intermediate between; blur when every bucket's CI sits inside ±0.01 nats.

**The scored reading is the `row` control on the context bucket.** The `xrow` control's
worth includes the document, which helps whichever branch came true, so an `xrow` ratio
near 1 cannot tell a frontier from a topic code. The spec's own-bucket `xrow` verdict is
reported and predicted, but it does not decide.

480 validation rows, batch 3, seed 0, the sweep's stream and packer (the npz pairs with
the sweeps by stream index), eager path (`model.use_kernels=false`, the trained
`tg_scoped_kernels`), on the 5090, one arm at a time.

## Predictions (frozen)

- **P-1 (spec rule, ruler).** The own-bucket `xrow` verdict is `superposition`: **65 %.**
- **P-2 (context bucket, document removed, ruler).** `xrow` context ratio >= 0.5: **70 %.**
- **P-3 (the scored reading, ruler).** `row` worth on offsets 1..n over all buckets has a
  CI above 0: **90 %**; the `row` context ratio >= 0.5: **45 %**; < 0.25: **25 %.**
- **P-4 (GK reads like the ruler).** Both GK arms' `row` context ratios within ±0.15 of
  the ruler's: **65 %.**
- **P-5 (four cells hold more of a frontier).** fan4-all's `row` context ratio exceeds the
  ruler's by more than 0.10: **35 %.**
- **P-6 (decode clustering).** Ruler `excess_frac_ge2_positive` over the `xrow2` null
  above 0.05: **20 %.**
- **P-7 (depth into the span).** Ruler `row` worth on the context top2 bucket at offsets
  8..15 has a CI above 0: **55 %.**

## Verdict rules

The verdict is on the `row` context ratio of the ruler (P-3), read with P-4 and P-5.

- **>= 0.5 (superposition):** the cell already holds continuation content for the
  context's runner-up branch. The loop's flat depth is then not a missing frontier; the
  RbS mechanism that uses depth (one hop per pass across a graph) has nothing to expand
  on this data. Next: a task or target where the frontier must grow per pass.
- **< 0.25 or top2 < 0 (committed):** the cell commits to one branch. RbS's set-valued
  training (CE over the frontier's members) is the lever to design.
- **blur:** the cell's span-specific content is too small to read; the reader or target is
  the limit (see `the-reader-was-the-limit`).
- **intermediate:** report, no design change.

A clause that the instrument cannot score (for example too few top2 spans: fewer than 50
stream blocks or a top1 CI touching 0) is a protocol failure, not a pass.

## Method

The builder's suite (30 tests, 9 sabotages caught) plus my review: I added the context
bucket and its planted test (31 pass), and four sabotages of my own (context bucket read
from the own cell, no shuffle, worth sign, offset 0 in the offsets 1..n reading) each
fail the suite. CPU smokes on 2 rows ran all four arms end to end (no verdict read from
them). Artifacts: JSON in `../results/2026-09-23-superposition-probe/`, npz outside the
repo.

## Results (filed 2026-09-23 21:40)

Artifacts: [`../results/2026-09-23-superposition-probe/`](../results/2026-09-23-superposition-probe/)
(one JSON and one run log per arm, `posthoc_strat.json`; the token npz files stay in
`/home/wolfe/morph-scratch/superpos/`). Probe at 626d0c5 on the 5090, 95 to 125 s per arm.
Every arm: 480 rows, 490 stream blocks (fan4-all 500), 24,339 scored spans (fan4-all
24,815), about 486k scored tokens. Own-bucket span counts on the ruler: top1 8,587, top2
1,726, other 14,026 (context bucket: 6,183 / 1,286 / 16,870).

Worth on offsets 1..n, 95 % CI:

| arm | row, all tokens | row ratio, context bucket | row ratio, own bucket | xrow ratio, context | xrow ratio, own (spec verdict) | decode excess |
|---|---|---|---|---|---|---|
| ruler | +0.096 [0.092, 0.100] | 1.741 [1.553, 1.938] | 1.476 [1.332, 1.629] | 1.202 [1.120, 1.289] | 1.183, superposition | 0.047 |
| lxtul-gk1 | +0.095 [0.091, 0.099] | 1.764 [1.574, 1.973] | 1.414 | 1.267 [1.182, 1.357] | 1.112, superposition | 0.046 |
| lxtul-gk4-shared | +0.099 [0.095, 0.103] | 1.838 [1.643, 2.037] | 1.389 | 1.238 [1.156, 1.323] | 1.159, superposition | 0.044 |
| fan4-all | +0.105 [0.101, 0.109] | 1.780 [1.582, 1.973] | 1.447 | 1.172 [1.082, 1.255] | 1.115, superposition | 0.052 |

Ruler worth per context bucket (row control): top1 +0.0646 [0.0596, 0.0695], top2 +0.1125
[0.1020, 0.1238], other +0.1081 [0.1033, 0.1127]. Of the context-top2 spans, 46 % have
`t0` ranked first by the own cell (context top1: 93 %; other: 13 %). The zero control
gives the same ordering (context ratio 1.308 [1.227, 1.391]).

P-7, `row` worth on the context top2 bucket at offsets 8..15 (from the npz,
`lab/divergence/superposition_strat.py`): ruler +0.0889 [+0.0747, +0.1033], gk1 +0.0851,
gk4-shared +0.0962, fan4-all +0.0947.

Scorecard:

| clause | reading | verdict |
|---|---|---|
| P-1 spec rule on the ruler says superposition | superposition, ratio 1.183 | held |
| P-2 ruler xrow context ratio >= 0.5 | 1.202 [1.120, 1.289] | held |
| P-3 ruler row worth CI above 0; row context ratio >= 0.5 (competing outcome < 0.25) | +0.096 [0.092, 0.100]; 1.741 [1.553, 1.938] | held, both; the < 0.25 outcome did not occur |
| P-4 both GK row context ratios within ±0.15 of the ruler's | +0.023, +0.097 | held |
| P-5 fan4-all row context ratio above the ruler's by > 0.10 | +0.039 | failed |
| P-6 ruler decode excess > 0.05 | 0.047 | failed |
| P-7 ruler row worth, context top2, offsets 8..15, CI above 0 | +0.0889 [+0.0747, +0.1033] | held |

**Post-hoc (not preregistered).** Every ratio is above 1, in all controls and both
bucketings. I checked whether token difficulty explains that: with the offsets 1..n
tokens stratified into deciles of their zero-cell CE, the ruler's row context ratio is
1.752 [1.563, 1.949] (gk1 1.767, gk4-shared 1.846, fan4-all 1.803). Per decile it is 0.99
in the easiest decile and 1.47 to 2.31 in the other nine. Token difficulty does not
explain it.

## Verdict

Success on the predictions; the rule's label needs a qualification.

By the frozen rule the verdict is **superposition** on every arm: the cell's span-specific
content (document held fixed) helps the continuation when the context's runner-up first
token came true, and it helps deep into the span (P-7). Two outcomes are ruled out. The
cell is **not a committed point**: one committed to its favourite branch would help less,
or hurt, when another branch came true, and it helps more. It is **not a blur**: the
span-specific worth is 0.096 nats per token with a tight CI.

What the rule does NOT establish is a frontier of branches in the RbS sense. A frontier
predicts a ratio at or below 1 (the runner-up is held with less weight than the
favourite). The reading is 1.5 to 1.8 at fixed token difficulty. The more likely reading
is that the cell carries span content that is independent of which first token comes
true, and that this content is worth more on spans the context could not settle (almost
half of the context-top2 spans are ones the cell itself ranked first). A first-token
branch is mostly not a semantic branch, so this instrument cannot separate "holds several
branches" from "holds what every branch shares". That is the prereg's hypothesis in a
narrower form: the help is branch-independent even with the document held fixed.

The GK arms read the same as the ruler (P-4): training on noisy rollouts did not change
what the cell holds. Four cells per slot do not hold more of it (P-5).

## Updated hypothesis

The deterministic slot cell is already a useful, uncommitted code of the next span (0.1
nats per token beyond the document, worth still 0.09 at offsets 8..15). The prereg's rule
therefore points away from "the cell is missing a frontier" as the reason the loop does
not use depth: one pass already writes content that serves every first-token branch, and
passes 2 to 6 add nothing measurable to it (K1−K6 +0.0016 to +0.0036 on these arms). The
depth question moves to what a SECOND pass could add that the first cannot write, which
is the question the Lean design (in progress, `lab/theory/`) is asked to answer. A
semantic-branch version of this instrument (continuations clustered by meaning, spec
3c.2 / 3c.3) was not built; build it only if a design needs the frontier distinction.
