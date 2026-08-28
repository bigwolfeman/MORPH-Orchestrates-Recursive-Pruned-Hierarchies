# Planned: plan-content probe, re-run with a decoder that can actually be fitted

Status: planned
Supersedes the method of: ../failures/2026-08-28-plan-content.md (REFUSED by its own gate)

## What failed, and the fix

The first panel fitted a **13,251,840-parameter** decoder on **1,654 examples** — ~8,000
parameters per example. The memorization gate refused it at a POSITION train/eval gap of
0.8872 nats against a 0.50 line. Two symptoms confirm it: POSITION (no z) BEAT PLAN, and
every condition landed within 0.15 nats of the unigram floor.

The fix is data volume first, capacity second, and the order matters: 1,654 examples is
absurd for this decoder regardless of regularisation, so more data is the honest first move.

## BOUNDED PROCEDURE — declared now so this cannot become a fishing expedition

Exactly two attempts are authorised. Each changes ONE thing, in a pre-declared direction,
for a stated reason. If attempt 2 is refused, the probe DESIGN is inadequate and that is the
reported result — no attempt 3, no threshold adjustment, and the `CANARY_MAX = 0.5` line is
NOT moved at any point.

- **Attempt 1 — more data.** `--fit-batches 150 --eval-batches 20 --steps 3000` (~41,000 fit
  examples, a 25x increase; steps raised so the larger set is actually traversed). Nothing
  else changes.
- **Attempt 2, ONLY if attempt 1 is refused — less capacity.** Attempt 1's settings plus
  `--hidden-dim 64 --weight-decay 0.1`. The 49k-vocab output layer dominates the parameter
  count and cannot shrink without changing the metric, so hidden width and decay are the
  only capacity knobs that leave the measurement intact.
- **If attempt 2 is refused**: report "the blind-decoder probe cannot be fitted at this
  scale", leave EMPTY-vs-FULL OPEN, and do not quote any band from any attempt.

## Predictions (frozen 2026-08-28 07:20, before either attempt runs)

**R1 (the gate passes on attempt 1).** POSITION's train/eval gap falls below 0.50 nats with
25x the data. If it does not, capacity — not data volume — is the binding constraint.

**R2 (POSITION stops winning).** PLAN's nats/token is BELOW POSITION's on all three
checkpoints. A condition given strictly more information must not do worse once the decoder
is no longer memorizing. This is the sanity check that the first panel failed, and it is
scoreable independently of the EMPTY/FULL bands.

**R3 (the decoder clears the unigram floor properly).** POSITION beats unigram(i+1) 8.1484
by at least 0.30 nats. In the refused panel it managed 0.15. Span-offset alone predicts a
great deal — the first token after a boundary is heavily constrained — so a decoder that
cannot show that is not reading anything.

**R4 (the restricted arms agree).** tg2-s1 and tg2-s2 give `SHUFFLED - PLAN` within 0.05 of
each other. They agreed to four decimals in the refused panel, but on a degenerate number.

**Deliberately NOT predicted:** whether `SHUFFLED - PLAN` clears the 0.20 FULL band or falls
under the 0.05 EMPTY band. That is the question the probe exists to answer, and guessing it
is how a threshold gets fitted to a hunch. The bands themselves are unchanged from the
original pre-registration.

## Decision rule (unchanged bands, restated so the re-run scores against the same lines)

- `SHUFFLED - PLAN >= 0.20` → the plan is FULL, the coda is not reading it, next work is the
  READER, and a flow-matching training loss becomes well-motivated.
- `SHUFFLED - PLAN <= 0.05` → the plan is EMPTY, next work is PROVENANCE and the TARGET.
- Between → inconclusive; report as such and name what would separate it.
- `PLAN - SUMMARY >= 0.20` → z is a summary of its own span, not a plan for the next, and
  that is a headline finding regardless of the other number.
- **Any of the above is quotable ONLY if the memorization gate passed.**

## Method

Same three checkpoints at matched step 3000 — `tg2-s1`, `tg2-s2` (restricted) and
`ctrlworth-s3` (unrestricted control) — same extraction point (`prefix_project` output, what
the coda actually reads), same four conditions, same frozen-weights guards
(`assert_frozen`, `param_fingerprint` before/after, `assert_disjoint_batches`, fresh
unembedding). GPU is free; the round-2 queue and TG3b have both completed.

## ATTEMPT 1 RESULT + AMENDMENT (2026-08-28 07:17) — the failure mode CHANGED

Attempt 1 on `tg2-s1` (fit 150 / eval 20 / steps 3000, ~120k eval tokens):

    PLAN      7.5956   SUMMARY  7.5991   SHUFFLED 7.5956   POSITION 7.5386
    unigram(i+1) 7.4635        unigram(i) 7.4645
    train losses: PLAN 7.6214   SUMMARY 7.4359   SHUFFLED 7.6215   POSITION 7.5538
    MEMORIZATION GATE  POSITION eval 7.5386 - train 7.5538 = -0.0152  -> OK

**R1 HELD.** The gate passes with room to spare: the 0.8872-nat gap is now −0.0152. Data
volume was indeed the binding constraint on memorization.

**R2 FAILED.** PLAN (7.5956) is not below POSITION (7.5386). A condition given strictly
more information is still doing worse.

**R3 FAILED, and it is the diagnostic one.** POSITION was predicted to beat unigram(i+1) by
≥ 0.30 nats. It is **0.075 nats WORSE** (7.5386 vs 7.4635). **Every condition loses to a
unigram model.** Train loss ≈ eval loss in every condition, so this is not overfitting — it
is a decoder that has not been fitted at all.

R2 and R3 were pre-registered precisely to catch a panel whose gate passes but whose
instrument has no power. They fired. **The bands are NOT quotable from attempt 1.**

### The amendment, and why it is not fishing

The declared attempt 2 is "less capacity" (`--hidden-dim 64 --weight-decay 0.1`). That was
written when the observed failure was MEMORIZATION. The observed failure is now the
OPPOSITE — underfitting — so running attempt 2 as written would knowingly spend a run
making a underfitted decoder smaller. Following the letter of the procedure would produce a
result I already know is useless.

Attempt 2 is therefore redirected to **more optimization, not less capacity**:
`--steps 20000 --decoder-batch 256 --lr 1e-3`. At attempt 1's settings the decoder saw
3000 × 32 = 96k draws over ~41k examples — **2.3 epochs**. The redirect gives 5.1M draws,
about 125 epochs.

What keeps this honest, stated explicitly:

- The redirect is driven by a MEASURED diagnosis (train ≈ eval, and every condition below
  the unigram floor), not by an unwanted band. The band is not the thing being changed.
- `CANARY_MAX` stays **0.50**. The decision bands (0.20 FULL / 0.05 EMPTY) are untouched.
- The budget stays at **two attempts total**. This redirect IS attempt 2; there is no
  attempt 3.
- R2 and R3 remain live as pass/fail gates on attempt 2. If the decoder still cannot beat
  unigram, the reported result is that the blind-decoder probe cannot be fitted at this
  scale, and EMPTY-vs-FULL stays OPEN.

**A2-R5 (new, frozen now).** Attempt 2's POSITION beats unigram(i+1) by ≥ 0.30 nats AND its
memorization gate passes. Both, or the probe design is reported as inadequate. Failing one
while passing the other means the valid window between underfitting and memorization is too
narrow for this decoder, which is itself the finding.
