# Agent Note: the terminal fixed-point objective as the loop's stability term

Status: implemented

## Problem

Every hold on the looped core's early detonation in this tree was a schedule or an optimizer
cap (the 1000-step LR ramp, `ademamix_alpha_cap 1.0`), and the one map-side term that
worked at mean 6 (the finite-difference gain hinge on the slot loop) lost at a deep draw
(`lab/experiments/failures/2026-09-07-arc-e7-block-loop.md`). Widening the ρ < 1 diagonal
carry to the whole residual (Parcae's mechanism) did not touch the detonation
(`failures/2026-09-07-arc-e9-widen-the-carry.md`). A schedule cannot act on a map that
goes expansive after the ramp ends (E7 at 2712, E8-16 at 3446, both under the ramp).

## Decision

`base.yaml` ships `model.core_fixed_point_lambda: 1.0` (2026-09-07, Wolfe: "ship the
fix"): λ · mean over samples of ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's LAST core
iteration, training only, in `_core_region`. That code path is the plain model AND the
paid TUL loop (`tul.tokens_through_core: true`, the shipped forward), so the term is on
for the production recipe from step 0 through TUL activation. `_forward_tul` consumes the
stash the same way the plain path does (`fixed_point` / `fp_weighted` in the output dict,
subtracted from every reported loss by train.py, logged as `train/fixed_point`). Under
SCSE the denominator is the absolute state h* + Δ, so the term is the same quantity on
both carriers.

The slot loop (`_tul_core`) has no finishing-slice term yet. A slot-loop model with the
term on REFUSES TO BUILD (build-time raise in `MORPHTransformer.__init__`), and
`tul_short.yaml`, the root of every slot-loop arm, sets it to 0 with the reason. The
refuted gain penalty (`core_gain_lambda`) refuses to build on any TUL model.

Evidence, 2026-09-07:

- Under warmup 0 (the detonation assay, `lab/divergence/DIVERGENCE-README.md`): 0 of 6
  draws detonated (max `preclip/total` 11–18) against 4 of 7 controls the same night
  (Fisher one-sided p 0.049) and 17 of 24 in September
  (`lab/experiments/successes/2026-09-07-arc-e10-loop-loss-terms.md`, E10a). The relative
  change sits below 0.05 from step 20; the survivors read 0.41 nats better at 800 than the
  surviving controls.
- Under the shipped ramp at 5000 steps (`successes/2026-09-07-arc-e11-fixed-point-
  ramped.md`): +0.0009 [−0.0022, +0.0040] nats against notul at the trained depth on 480
  paired rows, healthy (max 62), depth-1 CE 0.014 BETTER, loop dependence K1−K6 0.022
  against 0.037.
- The paid-loop wiring: `tests/test_core_loop_aux.py` (λ 0 bit-identical, the paid loop's
  loss = weighted CE + `fp_weighted`, gradient reaches the core, eval and label-less
  forwards leave no stash, the slot loop and the gain term refuse to build) and a 12-step
  `tul_smoke` run on the GPU with the shipped default (see the commit).

## Alternatives considered

- The directional gain hinge (STARS-style). Its one-shot form was inert on the plain loop
  (the reading sat at 1.1–1.5 along a batch-averaged direction); the within-step
  power-iterated form (E10c) detonated 2 of 6 at STARS' λ 1e-3, though its reading leads
  the onset by 20–40 steps. It costs four extra core-step applications per step (1.29x the
  fixed-point draw's wall clock); the fixed-point term costs one elementwise op.
- Keeping the ramp alone. It is measured (0 of 9) but it is a schedule: it cannot act on
  a map that goes expansive later.
- Parcae's carry constraint. Refuted for the early window (E9).
- Porting to `_tul_core` before shipping. Rejected for now: the slot loop is not the
  shipped forward, the port needs its own assay (Y2 with the term, warmup 0, 6 draws),
  and a build-time refusal plus the `tul_short.yaml` override keeps the two paths honest
  until it lands.
- Making the term silently inert on the slot loop (the `[slot-levers] INERT` pattern).
  Rejected: an inert stability term on the one path that still spikes is exactly the
  silent gap this tree has been bitten by before.

## Consequences

- The production recipe carries a second hold beside the ramp. 20k behaviour, the deep
  draw, the prune/carve/route conjunction and seq 4096 are UNMEASURED with it on; the
  only evidence is one 5k seed and six 1200-step draws. The first production-length run
  reads `train/fixed_point` beside `preclip/total`.
- `train/fixed_point` is a standing regime instrument: it peaks 0.2–0.6 inside the first
  20 steps and sits at 0.003–0.02 on a healthy loop.
- Open: the `_tul_core` port (per-slot depths, the same sorted active-set structure) so
  the block-loop rerun and the think-once panel carry it; whether λ 1.0 is near a cliff
  (no other λ was run); the E7 failure shape (one huge first iteration, then rotation),
  which a last-iteration term has not been tested against.
