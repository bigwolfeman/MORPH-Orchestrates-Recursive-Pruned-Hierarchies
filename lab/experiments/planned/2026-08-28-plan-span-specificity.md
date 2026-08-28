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

## Results (filled 2026-08-28 09:45). ALL FOUR PREDICTIONS HELD — and the control inverts the programme.

All at step 3000, ce_main, 8 fixed eval batches.

| arm | restriction | aux losses | zeroing costs | shuffling costs | **specificity** |
|---|---|---|---|---|---|
| ctrlworth-s3 | **OFF** | ON | +0.0148 | +0.0096 | **65.1%** |
| tg1-s1 | ON | ON | +0.0637 | +0.0019 | **3.0%** |
| tg2-s1 | ON | OFF | +0.1232 | −0.0001 | −0.1% |
| tg2-s2 | ON | OFF | +0.0516 | +0.0002 | 0.3% |
| tg3b-s1 | ON (soft) | OFF | +0.0407 | +0.0002 | 0.6% |
| tg3b-s2 | ON (soft) | OFF | +0.0365 | +0.0000 | 0.1% |
| cap64-s2 | ON | OFF | +0.0454 | +0.0001 | 0.2% |

**P1 HELD** — every shuffle cost is ≥ 0 within noise (the one negative is −0.0001).
**P2 HELD**, and not marginally: mean specificity over the four restricted seeds is **0.2%**
against a predicted line of 50%.
**P3 HELD** — seeds agree (tg2 −0.1/0.3%, tg3b 0.6/0.1%).
**P4 HELD** — TG3b's mean (0.35%) is within 0.25 of TG2's (0.1%). Softening the mask changed
how much the coda NEEDS the plan but not what fraction of it is span-specific.

### The finding

**In every restricted arm the plan is an interchangeable constant.** The coda gains
0.037–0.123 nats from the slot path, and **essentially none of it depends on which span
wrote the plan.** Hand the coda a different span's plan and its token predictions are
unchanged to four decimal places. The slot path's value is STRUCTURAL — it comes from the
positions — not informational.

### The control inverts the programme's premise, and the confound is controlled

`ctrlworth-s3` is **65.1%** span-specific. Its plan is worth far LESS in total (0.0148 vs
0.0637–0.1232) but most of that value is real per-span content.

The obvious objection is that the control differs in TWO ways: no restriction AND aux losses
on (`emit_weight`/`plast_weight` default to 0.5, and the emit loss trains z directly to
predict the next span's first token — exactly span-specific content). **tg1-s1 settles it.**
TG1 is `tul_a1` plus the restriction, with the aux losses left ON, so tg1-s1 vs ctrlworth-s3
differ by the RESTRICTION alone:

    aux ON, restriction OFF   ctrlworth-s3   65.1%
    aux ON, restriction ON    tg1-s1          3.0%

A 20x collapse from the restriction alone. Aux losses do contribute — within the restricted
family they lift specificity from ~0.3% to 3.0%, about 10x — but that is an order of
magnitude short of what the restriction removes.

(Accepted, already-recorded confound: TG arms do not build the compressed branch's
compressor/indexer, so tg1 vs ctrl also carries that parameter delta. It is the same
confound noted in ../failures/2026-08-27-tg-restriction.md and is not plausibly worth 62
points of specificity.)

### Why this matters more than the CE numbers

The restriction was adopted to make the plan LOAD-BEARING. Measured, it does the opposite of
what was intended at the level that matters:

- it makes the coda depend on the slot **positions** more (plan worth 0.0148 → 0.0637–0.1232)
- it makes the coda depend on the plan's **content** ~20x less (65.1% → 3.0%)

When tokens cannot see earlier spans, the slot positions become a structural crutch and the
model stops reading what is in them. **The restriction destroyed exactly the thing it was
introduced to create.** The earlier "plan worth rose 2–10x, the restriction moves the plan"
reading — already withdrawn on 2026-08-28 as structurally confounded — is now not merely
unsupported but backwards.

### What this does NOT establish

The shuffle asks whether the CODA uses span-specific content. It cannot tell "z contains no
span-specific information" apart from "z contains it and the coda ignores it". Both give a
shuffle cost of zero. The blind decoder was supposed to separate those and failed
(../failures/2026-08-28-plan-content.md).

**Retraction of a claim I made from this panel earlier today:** I said the slot's own emit
head "does notice which span wrote the plan" because `ce_emit` moved 0.36 under shuffling on
cap64-s2. That is not supportable — cap64 is `tul_tg2`-based with `emit_weight = 0`, so its
emit head is untrained and its `ce_emit` (13.7 nats, worse than the 10.8 of a uniform
distribution over the 49k vocabulary) is not a readable signal. On arms where the head IS
trained the number does behave: ctrlworth-s3 shuffles at 81% of its zeroing cost on ce_emit,
tg1-s1 at 7.8% — the same story as ce_main, from a head that was actually trained.
