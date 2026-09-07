# Success: ARC E11 — the fixed-point objective at 5000 ramped steps against notul (price and earning)

Status: success
Date: 2026-09-07 06:10 (frozen before launch; the E10 binding rule's follow-up)
Arc: `2026-09-04-loop-contribution-arc.md`. Follows E10a (`planned/2026-09-07-arc-e10-loop-
loss-terms.md`: 6 of 6 warmup-0 draws healthy at max `preclip/total` 11–18 where the
controls detonated by step 419).

## Question

The terminal fixed-point term (λ · mean ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's last
iteration, λ 1.0) is the first mechanism in this tree that holds the early detonation
without an LR ramp. E1's lesson is that a stability dial can be an earning dial in the wrong
direction: a map held at its fixed point may be a map that computes nothing past iteration
1. What does the term cost, and what does it do to the loop's contribution, on the shipped
recipe (ramp 1000, flat 1e-4, 5000 steps) against the matched notul run on the same rows?

## Method

`notul_fp.yaml` = `notul` + `core_fixed_point_lambda 1.0`, seed 1, seq 1024, batch 6, 5000
steps, sustained tripwire. Readouts at 2500 and 5000: `token_depth_sweep.py --profile` at
depths 1, 3, 6, 8 on the arc's 480 rows; paired bootstraps against
`results/2026-09-04-arc-e0/sweep_notul_5000.json` (and the E0 tree's notul at 2500 if it
exists, else the 5000 ruler only); `score_arc_e0.py` at (1, 6); the `train/fixed_point`
trace. Output `arc/results/2026-09-07-arc-e11/`. Runner `arc/run_e11.sh` after E4. ~1 h.

## Predictions (frozen)

- **P11a.** Reaches 5000 with the sustained tripwire silent: **85 %**.
- **P11b (the price).** E11 at depth 6 minus notul at depth 6, paired on the 480 rows at
  5000, is above +0.05 nats: **55 %**.
- **P11c (the earning).** E11's token K1−K6 at 5000 is below notul's +0.037 with
  non-overlapping CIs: **60 %** (a settled map is a free ride).
- **P11d (no price).** The paired difference at depth 6 is within ±0.02: **25 %**.
- **P11e.** `train/fixed_point` stays below 0.05 for the whole run: **80 %**.

## Binding

- P11d TRUE ⇒ the term is a free stability mechanism on the plain loop: candidate for
  `base.yaml` alongside the ramp (belt and braces), and the block-loop rerun (E7's fix
  list) carries it on the slot path once ported.
- P11b TRUE and P11c TRUE ⇒ the term buys stability by emptying the loop (E1 again, from
  the other side); it stays a diagnostic, and the lever moves to the directional hinge
  (E10b) or the bounded write.
- P11a FALSE ⇒ the term holds the early window only; file it under the divergence README's
  regime table as a partial.

## Not verified before launch

The term under the ramp (all E10a draws were warmup 0); λ 1.0 is the assay's guess; the
notul ruler at 2500 may not exist on disk (the 5000 ruler does).

## Results (2026-09-07 09:17; `arc/run_e11.sh` on worktree bd9696c; draw 08:22–09:14, 5000 steps, HEALTHY, max `preclip/total` 62 at 509; files in `results/2026-09-07-arc-e11/`)

Val: final 4.0016; last four 4.1589, 3.9418, 4.0031, 4.0387 (notul run at 5000: 4.1380).
Forced depth on the 480 rows at 5000, with notul at the same step (`paired_ci.txt`):

| depth | 1 | 3 | 6 | 8 |
|---|---|---|---|---|
| E11 (fixed-point λ 1.0) | 3.993 | 3.973 | 3.972 | 3.973 |
| notul | 4.008 | 3.972 | 3.971 | 3.973 |
| E11 − notul, paired | −0.0142 [−0.0172, −0.0110] | +0.0003 [−0.0027, +0.0032] | **+0.0009 [−0.0022, +0.0040]** | +0.0006 [−0.0024, +0.0038] |

K-differences: E11 K1−K3 +0.0206, K3−K6 +0.0010 [+0.0006, +0.0013], K1−K6 +0.0216 [+0.0206,
+0.0225]; notul K1−K3 +0.0352, K3−K6 +0.0016, K1−K6 +0.0367 [+0.0356, +0.0379]; the paired
difference in K1−K6 is −0.0151 [−0.0167, −0.0136]. At 2500: 4.476 / 4.454 / 4.454 / 4.456.
`train/fixed_point`: 0.109 at step 13 (the one value above 0.05), median 0.005, 0.0024 at
4999.

Scored:

- **P11a TRUE** (5000 steps, tripwire silent, max 62).
- **P11b FALSE** (the price is +0.0009, not > 0.05).
- **P11c TRUE** (K1−K6 0.0216 against 0.0367, non-overlapping; the paired difference is
  −0.015). But the mechanism is the opposite of the one predicted: the deep readout is
  unchanged (+0.0009 at depth 6) and the SHALLOW readout improved (−0.014 at depth 1).
  The loop did not empty; the model depends less on its iterations for the same result.
- **P11d TRUE** (within ±0.02 at the trained depth; prior 25 %).
- **P11e FALSE by the letter** (one reading of 0.109 at step 13; below 0.05 from then on).

## Verdict

Three of five, and the one that matters (P11d, the 25 % prior) held: the terminal
fixed-point objective is a FREE stability mechanism on the plain loop at 5k. It holds the
early detonation without an LR ramp (E10a: 0 of 6 against 4 of 7 controls tonight, 17 of
24 in September), and under the shipped ramp it costs nothing at the trained depth, runs
healthy, and lowers the model's dependence on depth by improving its shallow readout. The
E1 fear (a stability dial that empties the loop) did not materialise: depth 6 is the same
model, depth 1 is a better one.

Not verified: 20k behaviour (one 5k run, one seed); the slot path (the term is plain-loop
only; `_tul_core` needs the same finishing-slice term); interaction with the deep draw
(E8-16's detonation at 3446 was at mean 16 / bptt 8 without this term); whether λ 1.0 is
near a cliff (no other λ was run); generation quality (gen_every 0).

## Updated hypothesis

The early detonation is a settling failure: the map organises expansive in the first ~300
steps because nothing asks it to converge, and once asked (one relative-change term at the
last iteration) it does not, at no cost to CE. Proposal: ship `core_fixed_point_lambda 1.0`
in `base.yaml` beside the ramp (belt and braces), port the term to `_tul_core`, and put it on
the block-loop rerun (`.agents/notes/proposed/architecture/2026-09-07-fixed-point-objective-
as-the-loop-stability-term.md`). Wolfe's call on the default.
