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


## Attempt 2 (the redirect) and the FINAL verdict on this instrument, 2026-08-28 08:10

Settings: fit 150 / eval 20 batches, steps 20000, decoder-batch 256, lr 1e-3
(~41k fit examples, ~125 epochs). Full precision, from the output JSONs:

    checkpoint    PLAN         SUMMARY      SHUFFLED     POSITION     unigram(i+1)
    tg2-s1        7.49422125   7.49451206   7.49733750   7.42210594   7.4635
    tg2-s2        7.49418440   7.49440176   7.49432029   7.42210594   7.4635

    MEMORIZATION GATE  POSITION eval 7.4221 - train 7.3271 = +0.0950 -> OK (passes)

**Scored against the frozen predictions:**

- **R1 HELD** (both attempts). The gate passes.
- **R2 FAILED.** PLAN (7.4942) is ABOVE POSITION (7.4221) by 0.072 nats. A condition given
  strictly MORE information still does WORSE. True in both attempts, on every checkpoint.
- **R3 FAILED.** POSITION beats the unigram floor by **0.0414 nats** against a
  pre-registered 0.30 line.
- **R4 HELD** (the two restricted seeds agree), but on a degenerate quantity.
- **A2-R5 FAILED** — it required BOTH the gate passing AND ≥0.30 over unigram. Exactly the
  case it was written for: the gate passes while the fit does not.

**VERDICT, per the bounded procedure: the blind-decoder probe cannot be fitted at this
scale. EMPTY-vs-FULL stays OPEN. No attempt 3. `CANARY_MAX` was never moved.**

An instrument that clears a unigram model by 0.04 nats cannot detect a 0.20-nat signal.
`SHUFFLED - PLAN` came out +0.0031 and +0.0001 — deep inside the EMPTY band — and that
number is NOT reported as a result, because the same instrument cannot show that offset
alone predicts anything either, and offset alone certainly does.

### A false alarm I raised and then resolved, recorded so it is not re-raised

The 4-decimal console table printed tg2-s1 and tg2-s2 as IDENTICAL on PLAN, POSITION and
both train losses. Two different seeds should not do that, and the natural suspicion is that
`load_checkpoint` is not taking effect. It is. Checked two ways:

1. The two checkpoints genuinely differ — **410 of 494 tensors differ**, with max absolute
   differences up to 0.90 (`embed.bigram.lambdas`).
2. At full precision PLAN, SUMMARY and SHUFFLED all differ between the seeds; only the
   5th decimal onward separates them, which 4-decimal printing hid.

**POSITION is bit-identical across checkpoints, and that is CORRECT** — POSITION is handed
no z, so with the same data, the same seed and the same decoder init it MUST reproduce
exactly. It is a determinism check passing, not a bug. The rounding, not the loading, was
the problem.

### Why this instrument was the wrong shape, for whoever revisits it

The blind decoder has to **learn language from scratch** to read z: predict K tokens of a
span over a 49k vocabulary from one vector, with no token context. That is a much harder
task than the question being asked, and it is why the decoder sits within 0.04 nats of a
unigram model after 125 epochs on 41k examples. The window between "too few examples, it
memorises" (attempt 1, gap 0.8872) and "enough examples, it cannot fit" (attempt 2, 0.0414
over unigram) may not exist for this design at this vocabulary.

The replacement uses the model's OWN coda — already trained to read z — so nothing has to be
learned: ../planned/2026-08-28-plan-span-specificity.md.
