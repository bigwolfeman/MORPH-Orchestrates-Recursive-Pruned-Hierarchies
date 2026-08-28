# Planned: TG round 2 — the slot SEED (TG4a/TG4b) and the softness of the restriction (TG3)

Status: planned
Spec: ../../../docs/tul-tg-spec.md   Worklist: ../../divergence/TG-WORKLIST.md (A1, A2)
Prior: ../failures/2026-08-27-tg-restriction.md   Paper: arXiv 2512.25026 (Thought Gestalt)

## Two process failures, stated before anything else

**1. TG4a ran with no pre-registration.** The arm was launched at 01:18 on 2026-08-28
straight off worklist item A1. Its step-3000/3500 numbers therefore exist BEFORE any
prediction was written, and no prediction below is scored against them. TG4a seed 1 is
recorded here as OBSERVATION, not as a passed or failed test. The rule the tree states is
"filling predictions after seeing data is a process failure: delete the file and start
over" — so this file does not pretend otherwise, and TG4a's headline claim gets no
prediction credit at all.

**2. TG3 is being run against its own pre-registered gate.** The prior experiment's
decision rule reads: "P1 fails on both seeds (> 5.15) → run TG3 once before any verdict."
P1 HELD (tg1-s1 ce_main@3000 = 4.8104, well under 5.15), so the trigger for TG3 did NOT
fire. TG3 was queued anyway, two seeds rather than the rule's "once".

**METHOD AMENDMENT 2026-08-28, with the reason, per the experiments protocol.** The
justification is worklist A2: the ce_main deficit measured against the control band
(0.17–0.42 nats) is large enough that "is the restriction simply too tight" became worth
one cheap read rather than a contingency. That is a real argument, and it is also exactly
the kind of after-the-fact re-justification a decision rule exists to resist. It is
recorded here so the record shows the rule was OVERRIDDEN, not satisfied. Two consequences
are accepted: TG3's numbers are EXPLORATORY, and the prior experiment's rule that ambiguity
should buy ONE longer run rather than more short seeds remains unsatisfied — round 2 spends
~3.5 GPU-hours on six short arms instead.

## Question

Round 1 left the loop under its 0.05-nat line and the restricted arms 0.17–0.42 nats of
ce_main behind the control band. Two independent levers remain untested:

- **The SEED.** `pooling_probe` on tg2-s1@3500 confirms the plain-mean law (slope −0.470,
  r² 0.922): slot-seed signal falls 0.516 (spans 4–5) to 0.210 (spans 24–32) against a
  constant ‖E_slot‖ = 0.238. Under the restriction the slot already attends its whole span
  through 4 prelude blocks, so the bag-mean may be redundant AND diluting. TG4a deletes it
  (`slot_seed: e_slot`); TG4b replaces it with a projection of the span's LAST token
  (`slot_seed: boundary`).
- **The SOFTNESS.** TG3 lets a token also see the PREVIOUS span.

## The confound this round exposed, and why it changes what gets measured

`loop_off` leaves the slot carrying its own INPUT, so what "no-loop" falls back to is
ARM-DEPENDENT: a bag_mean arm falls back to a span summary, a TG4a arm falls back to a
CONSTANT. Read side by side, the column credits the loop for the seed's deletion. Measured
on tg4a-s1/step_3000, the naive column reads 0.0921 nats — ~25× the control band — and that
number is an artefact of the fallback, not a loop result.

`lab/divergence/slot_path_worth.py` therefore gains a `no-loop, bag-mean seed` condition
(`seed_bagmean`, added 2026-08-28) that forces the bag-mean fallback whatever the arm was
trained on, so the loop column means the same thing in every arm. Both columns are UPPER
bounds on a non-bag_mean arm — the forced fallback is out of distribution for weights never
trained on it — and the predictions below use the new column only for a DIRECTIONAL claim,
never as the loop's worth.

## Predictions (frozen at 02:05 on 2026-08-28, before any TG3 or TG4b step)

Honesty note on timing: tg4a seed 2 began at 01:54 and had produced eval points up to
step ~250 when these were written; its step-3000/3500 worth numbers did not exist. TG3 and
TG4b had produced nothing at all. Metric is **ce_main** throughout, from
`slot_path_worth.py` at step 3000 unless stated. Control band: 4.4586–4.5459.

**A2 (TG4a seed agreement).** tg4a-s2 ce_main@3000 lands within 0.15 of seed 1's 4.8094,
i.e. in [4.659, 4.959]. If it does not, TG4a is n=1 and no TG4a claim is readable.

**T1 (TG3 recovers CE).** Relaxing the mask restores context, so on BOTH seeds tg3
ce_main@3000 < 4.8104 (tg1-s1, the arm TG3 is built from).

**T2 (TG3 gives the plan back).** Tokens regain a direct route to the previous span, so on
BOTH seeds tg3 plan worth@3000 < 0.0637 (tg1-s1's). This is the prediction that makes the
confound correction falsifiable: if plan worth is largely a function of what the mask
FORBIDS rather than of what the plan CONTAINS, softening the mask must lower it.

**T3 (TG3 keeps the takeover).** TG3 is built on tul_tg1, so it keeps the aux losses that
O5 and TG2 identified as the takeover fuel. At least 1 of 2 seeds fires the takeover rule
(core share > 0.5 on more than 30% of the last 50 probed steps).

**T4 (the loop still does not earn its line).** No round-2 arm reaches loop worth ≥ 0.05 on
ce_main at step 3500, on either seed, read from the bag-mean-seed column where it exists.

**B1 (TG4b sits with TG4a).** Both delete the bag-mean and boundary restores only one
token's worth of span signal, so tg4b ce_main@3000 lands within 0.10 of tg4a's seed-matched
value, on both seeds.

**B2 (the confound is real, not merely arguable).** On every non-bag_mean arm and both
checkpoint steps, `full − no-loop [bag-mean]` < `full − no-loop [own seed]`. This is the
directional claim the new condition exists to test; a violation means `seed_bagmean` is
measuring something other than what its docstring says.

**B3 (single-objective still removes the takeover).** TG4a and TG4b are tul_tg2-based, so
0 of their 4 seeds fire the takeover rule.

**B4 (no arm reaches the band).** min ce_main@3000 over all six round-2 arms stays above
4.60 — still clear of the control band's 4.5459 upper edge.

## Decision rule

- **T2 fails** (plan worth does NOT fall when the mask softens) → the confound correction is
  wrong; plan worth tracks CONTENT after all, and the round-1 verdict's original wording was
  right. Reinstate it and say so.
- **T2 holds and the plan-content probe reports EMPTY** → plan worth is a mask artefact,
  the whole "restriction makes the plan load-bearing" line is retired, and the target
  (span i+1) becomes the only live lever.
- **B2 fails** → stop using the new column and fix `seed_bagmean` before any seed claim.
- **T4 holds** (expected) → the loop has now missed its line under every seed, objective and
  mask variant tried. File the loop negative and stop paying for short arms.
- Any arm reaching the control band → that arm, not the loop, is the result; re-plan.

## Method

Six arms, `tg4a`/`tg3`/`tg4b` × seeds 1,2, 3500 steps, batch 6, `use_kernels=false`,
`ademamix_alpha_cap=3.5`, `grad_probe_every=1`, ckpt every 500 — flags identical to round 1
so the control band stays comparable. Sequential on the 5090 (UPS). Driver:
`/home/wolfe/morph-scratch/tg2_arms.sh`. Worth passes at steps 3000 and 3500. Takeover from
`score_arms.py` unchanged.

NOT controlled, and named rather than hidden: TG3 is tul_tg1-based while TG4a/TG4b are
tul_tg2-based, so TG3 differs from the seed arms in BOTH mask softness and objective count.
TG3 is readable only against TG1, never against TG4a/TG4b.

## Results — TG4a, both seeds (filled 2026-08-28 02:31, arms tg3/tg4b still running)

All ce_main, from `slot_path_worth.py`; takeover from `score_arms.py` unchanged.

| arm | ce_main@3000 | @3500 | no-loop [own seed] 3000→3500 | no-loop [bag-mean] | no-plan 3000→3500 | end core share |
|---|---|---|---|---|---|---|
| tg4a-s1 | 4.8094 | 4.7146 | 0.0921 → 0.1172 | not measured (pre-patch) | 0.2042 → 0.1934 | 0.0011 held |
| tg4a-s2 | 4.7735 | 4.6903 | 0.0764 → 0.1333 | 0.5589 → 0.4794 | 0.0953 → 0.0988 | 0.0010 held |

**A2 HELD.** tg4a-s2 ce_main@3000 = 4.7735, inside the predicted [4.659, 4.959]. TG4a is
readable as a 2-seed arm.

**B3 on track.** 2 of 2 tg2-based seeds held, core shares 0.0011 / 0.0010 — the two lowest
of the campaign, and far more seed-stable than ce_main (spread 0.036) or plan worth
(spread 2.1x). Awaiting tg4b's two seeds for the full 0-of-4.

**B4 holding.** min ce_main@3000 so far 4.7735, still clear of the band's 4.5459.

**B2 FAILED, and the failure is the informative one.**

| step | no-loop [own seed] | no-loop [bag-mean] | predicted | measured |
|---|---|---|---|---|
| 3000 | 0.0764 | 0.5589 | bag-mean SMALLER | 7.3x LARGER |
| 3500 | 0.1333 | 0.4794 | bag-mean SMALLER | 3.6x LARGER |

Forcing the bag-mean is worse than zeroing the plan outright (no-plan 0.0953 / 0.0988). The
prediction assumed the information a bag-mean carries would outweigh the distribution shift
of injecting a seed the weights never trained on. It does not, by 3.6-7.3x.

Per this file's own decision rule ("B2 fails → stop using the new column"), the column is
NOT repaired and NOT used as a loop worth. It is relabelled in the instrument as what it
actually measures — an OOD-shock number, i.e. how far an arm's downstream weights have
specialised to their own seed. No eval-time substitution can do better: swapping a seed the
weights never saw measures the shock, not the loop.

**The standing consequence for the campaign's metric.** Loop worth is comparable only
WITHIN a fixed `slot_seed`. Across seed modes it needs matched TRAINING, not a smarter
ablation. The confound that motivated the column is REAL — tg4a's naive 0.0921 is still not
a loop result — but the repair is not available at eval time.

**A second reason the naive column is uninterpretable on an e_slot arm.** At step 3500,
`no-loop` (0.1333) costs MORE than `no-plan` (0.0988): removing less hurts more. `E_slot` is
a constant the coda has learned to READ, so an uninformative-but-valid plan actively
misleads it while zeroing is clean. The fallback is not merely uninformative, it is harmful.

**T4 is unscoreable on TG4a** for the same reason, and this is recorded rather than
resolved: the own-seed column crosses 0.05 (0.0764-0.1333), but on this arm that column does
not measure the loop. T4 stays open, to be read on TG3 (a bag_mean arm, where the column
means what it always meant).

**What TG4a says about the SEED, which is what the arm was for.** ce_main@3000 averages
4.791 over TG4a's two seeds against 4.794 over TG2's (4.8763 / 4.7121). Deleting the
bag-mean moved CE by 0.003 nats — nothing, well inside the seed spread. The pooling-law
argument predicted the bag-mean was diluting; removing it neither helped nor hurt.
