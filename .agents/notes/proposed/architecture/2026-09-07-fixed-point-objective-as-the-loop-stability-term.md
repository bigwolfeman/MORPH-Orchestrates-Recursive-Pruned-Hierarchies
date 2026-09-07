# Agent Note: the terminal fixed-point objective as the loop's stability term

Status: proposed

## Problem

Every hold on the looped core's early detonation in this tree is a schedule or an optimizer
cap (the 1000-step LR ramp, `ademamix_alpha_cap 1.0`), and the one map-side term that
worked at mean 6 (the finite-difference gain hinge on the slot loop) lost at a deep draw
(`lab/experiments/failures/2026-09-07-arc-e7-block-loop.md`). Widening the ρ < 1 diagonal
carry to the whole residual (Parcae's mechanism) did not touch the detonation
(`failures/2026-09-07-arc-e9-widen-the-carry.md`).

## Proposal

Ship `model.core_fixed_point_lambda 1.0` in `base.yaml` beside the ramp: λ · mean over
samples of ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's LAST core iteration, training only,
plain loop (`_core_region`). Evidence, 2026-09-07:

- Under warmup 0 (the detonation assay, `lab/divergence/DIVERGENCE-README.md`): 0 of 6
  draws detonated (max `preclip/total` 11–18) against 4 of 7 controls the same night and
  17 of 24 in September (`successes/2026-09-07-arc-e10-loop-loss-terms.md`, E10a). The
  relative change sits at 0.004–0.02 from step 0: the map never leaves the settled regime.
- Under the shipped ramp at 5000 steps (`successes/2026-09-07-arc-e11-fixed-point-
  ramped.md`): +0.0009 [−0.0022, +0.0040] nats against notul at the trained depth on 480
  paired rows, healthy (max 62), depth-1 CE 0.014 BETTER, loop dependence K1−K6 0.022
  against 0.037.

Then port the same finishing-slice term to `_tul_core` (per-slot depths, the same sorted
active-set structure) so the block-loop rerun carries it, and log `train/fixed_point` as a
standing regime instrument beside `preclip/total`.

## Alternatives considered

- The directional gain hinge (STARS-style). Its one-shot form was inert on the plain loop
  (the reading sat at 1.1–1.5 along a batch-averaged direction); the within-step
  power-iterated form (E10c) detonated 2 of 6 at STARS' λ 1e-3, though its reading leads
  the onset by 20–40 steps (`successes/2026-09-07-arc-e10-loop-loss-terms.md`). It costs
  four extra core-step applications per step (1.29x the fixed-point draw's wall clock);
  the fixed-point term costs one elementwise op.
- Keeping the ramp alone. It is measured (0 of 9) but it is a schedule: it cannot act on
  a map that goes expansive later (E7 at 2712, E8-16 at 3446, both under the ramp).
- Parcae's carry constraint. Refuted for the early window (E9).

## Acceptance criteria

- The `base.yaml` flip is bit-identical at λ 0 (tests pass: `tests/test_core_loop_aux.py`).
- One more seed of E11 (or the 20k continuation) before the default changes: a single 5k
  draw is the current evidence.
- The `_tul_core` port passes the same contract tests on the slot path and holds one
  slot-loop arm (Y2 with the term, warmup 0, 6 draws) before the block-loop rerun.

## Risks

A term that asks for settling can, at a different λ or scale, make the loop a free ride
(E1's lesson); at λ 1.0 and 5k it did not, but 20k and the deep draw are unmeasured. The
term is on the last iteration only, so a map that is expansive early and settles late
satisfies it; the E7 failure (one huge first iteration, then rotation) is exactly that shape
and the term has not been tested against it.
