# Planned: does a trained slot cell hold a superposed frontier, a committed point, or a blur?

Status: planned

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
