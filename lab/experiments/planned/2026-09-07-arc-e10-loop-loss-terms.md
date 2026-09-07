# Planned: ARC E10 — two loop loss terms under the detonation assay (terminal fixed point; directional gain hinge)

Status: planned
Date: 2026-09-07 (frozen before any smoke; Wolfe: "try the top 3 or 4")
Arc: `2026-09-04-loop-contribution-arc.md`. Note:
`.agents/notes/proposed/architecture/2026-09-07-loop-stability-mechanisms-from-the-literature.md`.
Shares E9's assay and E9's three control draws.

## Question

E7's Jacobian says the loop's failure is a map whose top singular direction runs to 95
while its typical gain reads 0.85, and whose state drifts rather than settles. Two
published mechanisms target that directly with one loss term each: a terminal
fixed-point objective (2608.18222: a settling loop is depth-safe; removing the term
induces drift) and a Jacobian penalty along the top singular direction (STARS 2605.26733:
‖Jv‖² by a power-iteration JVP; here by finite difference along a power-iterated
direction). Does either remove the early detonation on the plain loop?

## Method

The E9 assay: `notul`, warmup 0, 1200 steps, six draws per arm (seeds 301–306), abort rule
`preclip/total > 1e4` at step ≥ 200, control = E9's three `notul_carry_ctx_wu0` draws.

- **E10a** `notul_fp_wu0.yaml`: `core_fixed_point_lambda 1.0` (λ · mean over samples of
  ‖h_T − h_{T−1}‖² / ‖h_T‖² at each sample's last iteration).
- **E10b** `notul_pgain_wu0.yaml`: `core_gain_lambda 0.01`, `core_gain_target 20`,
  `core_gain_eps 0.02`, `core_gain_direction power` (hinge on the finite-difference gain
  along a direction updated by one power step per training step).

Readouts per draw: verdict and first crossing, max `preclip/total`, val at 400/800/1200,
`train/fixed_point` and `train/core_gain_est` / `core_gain_max` traces. Output
`arc/results/2026-09-07-arc-e10/`. Runner `arc/run_e10.sh` after E9. ~13 min per draw,
~2.6 h for twelve.

Cost: E10a adds one elementwise term per finishing sample; E10b adds two extra core-step
applications at one iteration per step (about 1/3 of a core pass at mean 6).

## Predictions (frozen)

- **P10a.** At most 1 of 6 fixed-point draws detonates: **35 %**.
- **P10b.** At most 1 of 6 directional-hinge draws detonates: **40 %**.
- **P10c.** On the hinge arm, `core_gain_est` on the surviving draws sits below 20 by
  step 600 and stays there (the hinge holds σ_max): **55 %**.
- **P10d.** On the fixed-point arm, `train/fixed_point` falls below 0.05 by step 600
  (the loop settles) on the surviving draws: **50 %**.
- **P10e (the price).** Surviving draws' val CE at 1200 is within 0.05 of the surviving
  control draws' (or of the ramped notul at 1200 if no control survives): **45 %** for
  each arm.
- **P10f.** The three E9 control draws detonate ≥ 2 of 3 (shared with P9b): **70 %**.

## Binding

- P10a or P10b TRUE (with P10f TRUE) ⇒ that term is a mechanism-level hold on the early
  detonation; it goes to 5000 ramped steps against notul on the 480 rows (CE at the
  trained depth, the depth sweep, and the Jacobian sweep) before any default changes.
  If both hold, the cheaper one (E10a) is the candidate default and E10b the belt.
- Both FALSE and E9 FALSE ⇒ the early detonation is not a map-side property at all at
  this LR; the ternary trigger and the optimizer are next, and the bounded write
  (2605.18797) is the remaining structural candidate.
- P10c TRUE and P10b FALSE ⇒ σ_max is not the quantity either (the hinge held it and the
  run still detonated); read the probe record for what moved.

## Not verified before launch

Both terms under torch.compile on the GPU (the smoke is the first run); the λ choices
(1.0 and 0.01) are scale guesses from the E7 Jacobian, not measured; the power direction
is shared across the batch and updated once per step, so it lags a fast-rotating map;
the fixed-point term's gradient can be satisfied by a trivially contractive map (E1's
lesson) — the 5000-step follow-up, not the assay, reads that.
