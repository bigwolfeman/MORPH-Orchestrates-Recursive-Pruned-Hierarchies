# Planned: ARC E11 — the fixed-point objective at 5000 ramped steps against notul (price and earning)

Status: planned
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
