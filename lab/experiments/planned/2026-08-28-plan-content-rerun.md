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
