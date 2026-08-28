# Planned: is the plan SPAN-SPECIFIC, or an interchangeable constant?

Status: planned
Instrument: `plan_shuffled` in `lab/divergence/slot_path_worth.py`
Replaces the instrument of: ../failures/2026-08-28-plan-content.md and its bounded re-run

## Why a new instrument, and why this is not a third attempt at the old one

The blind-decoder probe failed its own pre-registered power check TWICE, from opposite
sides. Attempt 1 (1,654 fit examples) MEMORISED: POSITION train/eval gap 0.8872 nats.
Attempt 2 (~41k examples, 125 epochs) did not memorise but could not FIT: POSITION beat the
unigram floor by **0.0414 nats** against a pre-registered 0.30 line, and PLAN stayed WORSE
than POSITION in every attempt and on every checkpoint. An instrument that clears unigram by
0.04 nats cannot detect a 0.20-nat signal. Its budget is spent and it is reported inadequate.

The root cause is structural, not a tuning miss: a blind decoder must **learn language from
scratch** to read z. That is a far harder task than the question being asked.

This instrument does not. It uses **the model's own coda** — which was trained to read z —
as the reader, so nothing has to be learned. It is a new measurement, not a re-run of the
old one at different settings.

## The question it answers, which `plan_off` cannot

`plan_off` zeroes what `prefix_project` writes and asks "is the slot path used at all".
Measured: 0.0386 (TG3b) to 0.0874 (TG2) nats of ce_main. That number cannot separate:

| world | shuffling the plan across slots | zeroing it |
|---|---|---|
| z carries SPAN-SPECIFIC content | costs about as much as zeroing | costs a lot |
| z is a useful CONSTANT | costs ~0 | costs a lot |

The second world is what "the plan is EMPTY" means in the only sense that matters: the coda
gains from the slot POSITIONS, not from what any particular span wrote there.

`plan_shuffled` permutes whole slots within each row. Every value the coda reads is still a
genuine plan this model produced on this batch — only the correspondence between a plan and
its own span is destroyed. That is what makes it a CONTROL rather than a corruption, and it
is the lesson of `seed_bagmean`, whose out-of-distribution shock swamped the signal it was
built to isolate (prediction B2, falsified 3.6–7.3×).

## Metric

**Specificity fraction = (cost of shuffling) / (cost of zeroing)**, both on ce_main against
the same `full` baseline, same batches, same checkpoint.

## Predictions (frozen 2026-08-28 07:45, before the condition has run on any checkpoint)

Arms: tg2-s1, tg2-s2, tg3b-s1, tg3b-s2, ctrlworth-s3, at step 3000.

**P1 (the control is sane).** Shuffling costs ≥ 0 on every arm, within noise. A NEGATIVE
cost — the coda doing better on someone else's plan — would mean the condition is broken,
not that the plan is harmful.

**P2 (specificity is LOW).** Mean specificity fraction across the four restricted seeds is
**below 0.5**: shuffling costs less than half what zeroing costs. Reasoning, and it is a
real prior rather than a hedge: the memory `morph-tul-plan-is-empty` records that the plan
is trained to predict ONE token and carries ~0.07 nats of its span, and `plan_off` on TG3b
costs only 0.0386 nats total. There is very little span-specific content available for the
shuffle to destroy.

**P3 (seeds agree).** The two TG2 seeds' specificity fractions agree within 0.25, and the
two TG3b seeds likewise. If they do not, n=2 is not enough and the panel is unreadable.

**P4 (softening does not change specificity).** TG3b's mean specificity fraction is within
0.25 of TG2's. The mask changes how much the coda NEEDS the plan (already measured: plan
worth fell 56%); it should not change what fraction of the plan's value is span-specific.

## Decision rule

- **Specificity ≥ 0.5** → the plan IS span-specific. The coda reads real per-span content,
  the plan is not empty, and the reader — not the target — is where the remaining value is.
  A flow-matching objective on span i+1 becomes well-motivated.
- **Specificity ≤ 0.2** → the plan is an interchangeable constant. The slot path's whole
  measured worth comes from the POSITIONS, not the content. This is the EMPTY verdict the
  blind decoder could not deliver, and it makes PROVENANCE and the TARGET the only lever:
  nothing is reading z's content because there is no content to read.
- **Between** → report the fraction and say so; the next instrument would need to bound the
  content directly rather than by ablation.
- **P1 fails** → fix the condition before reading anything else.

## Method

Add `plan_shuffled` to the existing worth panel and re-run `slot_path_worth.py` on the five
checkpoints above. No training. The permutation is seeded (`seed=0`) so the panel is
reproducible; rows are permuted independently.

Derangement is NOT enforced: at S=64 the expected fixed-point count is 1, so ~1.6% of slots
keep their own plan and the measured cost is biased DOWNWARD by about that much — far below
the effect sizes in question. Named here rather than hidden.

Tests: three cases in `tests/test_slot_seed.py`, verified to FAIL under both an off-by-K
index error (which would scramble WITHIN slots instead of across them) and an identity
permutation (a silent no-op that would report "not span-specific" for every arm).
`tests/test_slot_seed.py` → 22 passed.
