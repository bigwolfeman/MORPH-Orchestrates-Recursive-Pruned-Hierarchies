# Planned: is the TUL plan EMPTY, or FULL BUT UNREAD?

Status: planned
Probe: `lab/divergence/plan_content_probe.py`   Spec: TG-WORKLIST.md A3
Written and committed BEFORE the probe was run on any real checkpoint.

## Question

Every plan measurement in this campaign reads z THROUGH the coda (plan-off ablations),
which cannot separate "z has no content" from "z has content the coda ignores". This
probe reads z directly with a throwaway decoder on frozen weights, and asks a second
question for free: is z a SUMMARY of its own span, or a PLAN for the next one?

## Hypothesis

Under `tg_restrict` the slot attends its ENTIRE own span through all four prelude
blocks, so z necessarily absorbs span i. In TG2 `emit_weight=0`, so the slot has NO
direct next-span supervision at all — its content is shaped only by being useful to the
coda. Therefore z should read much more like a summary of span i than a plan for span
i+1, and the plan's thinness is a TARGET problem, not a capacity problem.

## Predictions (frozen — do not edit during the run)

Numbers are nats/token on held-out batches. The deciding quantity is
`SHUFFLED - PLAN` ("nats saved by the real z"); the summary quantity is
`PLAN - SUMMARY`. Both are positive when the named effect is present.

- **P1 (not empty):** both TG2 seeds show `SHUFFLED - PLAN >= 0.05`. z is not inert.
- **P2 (it is a summary):** both TG2 seeds show `PLAN - SUMMARY >= 0.20` — z
  reconstructs its OWN span substantially better than the next one.
- **P3 (the restriction put it there):** TG2's `SHUFFLED - PLAN` exceeds the
  unrestricted control's (`ctrlworth-s3`) by `>= 0.05`.
- **P4 (readable at all):** the two TG2 seeds agree within `0.10`. If they do not, the
  n=1 rule applies and NOTHING below is readable.

I am NOT predicting whether `SHUFFLED - PLAN` clears 0.20 (the FULL band). That is the
number this probe exists to find out, and guessing it would be fitting the threshold to
a hunch.

## Decision rule

- P4 fails -> void the panel, add seeds, change nothing else.
- `SHUFFLED - PLAN >= 0.20` (FULL) -> the plan has content the coda is not reading. Next
  work is the READER (injection, per-layer projections), and the flow-matching TRAINING
  loss becomes well-motivated because there is content to sharpen.
- `SHUFFLED - PLAN <= 0.05` (EMPTY) -> next work is PROVENANCE and the TARGET. TG4a/TG4b
  are already queued for provenance; the target fix is a next-span objective.
- P2 holds (whatever P1 does) -> headline finding: z is a summary, not a plan. The TUL
  thesis has a TARGET problem, and any next-span objective (flow matching on span i+1,
  continuous MTP) is attacking the right thing.
- P3 fails -> the restriction is not what put content into z, which weakens the whole TG
  line independently of the other numbers.

## Method

Frozen `requires_grad_(False)` + `.eval()`; asserted, and a parameter fingerprint is
compared before/after the decoder fit so a leak through the tied `lm_weight()` cannot
pass silently. z is the output of `prefix_project` — what the coda actually reads — mean
over HC streams, concat over `prefix_k`. Blind non-autoregressive decoder, own fresh
unembedding, identical size/steps/LR/seed across all four conditions, fit and eval
batches disjoint (asserted). SHUFFLED permutes z across rows. Offsets past a target
span's true length are masked and counted; each row's last valid slot is excluded in
EVERY condition so N is identical. Corpus-unigram floor reported alongside.

Checkpoints, all at matched step 3000:
`tg2-s1`, `tg2-s2` (config `tul_tg2`), `ctrlworth-s3` (config `tul_a1`, unrestricted).

Self-test (`tests/test_plan_content_probe.py`, 16 tests): a synthetic positive control
where z encodes the target must show a large saving, and a noise-z negative control must
show ~0. A probe that cannot separate those two cannot be trusted here.

## METHOD AMENDMENT 2026-08-28 (before any probe produced data)

Added a **memorization gate** to `plan_content_probe.py`
(`memorization_gate`, `CANARY_MAX = 0.5`). Predictions above are UNCHANGED; this amends
the Method only, and it is recorded here because the protocol requires an amendment to
carry its date and reason rather than appear silently in the code.

**Reason.** `fit_decoder`'s docstring already named the failure: with too little weight
decay the decoder MEMORIZES the fit set, its train loss collapses below the marginal
entropy, and eval loss on fresh z comes out WORSE than SHUFFLED — which flips the SIGN of
the deciding number, not merely its size. `weight_decay=1e-2` was found on a SYNTHETIC
sweep at a toy vocabulary. Nothing yet shows it holds at 49k vocab on a real checkpoint,
and this probe was built to settle a question the campaign has circled for weeks. A
silently sign-flipped answer is the worst outcome available.

**Mechanism.** POSITION is the canary and costs nothing extra: it is handed NO z, so it
cannot legitimately learn anything z-specific, while sharing every other condition's
decoder size, step budget, LR and seed. Its train/eval gap is memorization capacity that
ALL four conditions had to spend. Above 0.5 nats the panel is REFUSED rather than
reported — the `score_arms.py` convention of refusing a verdict the method cannot support.

**Why this cannot be fitted to a result.** No probe has run. The three chained runs
(tg2-s1, tg2-s2, ctrlworth-s3) start only after the round-2 training queue drains.

**Tests.** Three cases in `tests/test_plan_content_probe.py`, each verified to FAIL when
the gate is stubbed to always-readable. Full file: 19 passed.

**If the gate fires**, the recorded outcome is "method could not distinguish", the run is
filed under `failures/` per the protocol, and the next planned file raises `--weight-decay`
or shrinks the decoder. It is NOT a licence to re-run until the gate happens to pass.

## Results (filled 2026-08-28 07:10): REFUSED BY THE MEMORIZATION GATE. Status: failure.

    checkpoint      PLAN     SUMMARY  SHUFFLED  POSITION  unigram(i+1)  POSITION train
    tg2-s1         8.0267    8.0462   8.0267    8.0008      8.1484         7.1137
    tg2-s2         8.0266    8.0461   8.0266    8.0008      8.1484         7.1137
    ctrlworth-s3   8.0084    8.0156   8.0298    8.0008      8.1484         7.1137

    MEMORIZATION GATE  POSITION eval 8.0008 - train 7.1137 = +0.8872 nats
                       (refuse above 0.50) -> REFUSED, on all three checkpoints

**The gate did its job.** The printed band reads EMPTY on all three, and
`SHUFFLED - PLAN` is −0.0000 / +0.0000 / +0.0214. Without the gate, added at 02:2x on the
same day and BEFORE any of this data existed, the honest-looking report available here was
"the plan is EMPTY on both restricted seeds" — a headline conclusion drawn from a broken
instrument. It is refused instead.

**Diagnosis, and it is not subtle.** The probe fitted a **13,251,840-parameter** decoder on
**1,654 examples** (z_dim 2048, `--fit-batches 6` at batch 6). That is ~8,000 parameters per
example. Two independent symptoms confirm memorization rather than measurement:

- POSITION, which is handed NO z at all, scores 8.0008 — **better than PLAN's 8.0267**. A
  condition with strictly less information should not win. Extra input dimensions bought
  extra capacity to memorize the fit set, and that cost more than z was worth.
- Every condition sits within 0.15 nats of the unigram floor (8.1484). The decoder barely
  learned anything at all, in any condition.

`weight_decay=1e-2` was found on a synthetic sweep at toy vocabulary, exactly as the
`fit_decoder` docstring warned, and it does not hold at 49k vocab.

**Per this file's own rule**, the outcome is "the method could not distinguish", filed under
`failures/`, and the re-run gets a NEW planned file with a bounded, pre-declared procedure
rather than a licence to re-run until the gate happens to pass. See
../planned/2026-08-28-plan-content-rerun.md.

**Nothing about the plan's content is known.** Every downstream decision gated on
EMPTY-vs-FULL — the flow-matching objective in particular — stays gated.
