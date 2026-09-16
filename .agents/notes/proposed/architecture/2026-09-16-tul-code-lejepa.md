# Agent Note: LeJEPA on the code — the flow loss trains the encoder in full, SIGReg guards the rank

Status: proposed

Date: 2026-09-16. Prereg: `lab/experiments/planned/2026-09-16-tul-code-lejepa.md`. Follows
the rejected [`2026-09-15-tul-code-conditioned-thinker.md`](../../rejected/architecture/2026-09-15-tul-code-conditioned-thinker.md)
(its A1 arm at λ 0.1 with a rank tripwire) and the reference
[`docs/references/regularization-objectives/lejepa/lejepa.md`](../../../../docs/references/regularization-objectives/lejepa/lejepa.md).
Supersession check: the rejected note's A1 is superseded by this design (it stays in
`rejected/` for its measurements); no other active note claims how the encoder is trained.

## Problem

The code is a verbatim copy of the next span, and a copy's unpredictable share bounds
every thinker (thinker-only, cfg, jepa, XM all read the sample as an unconditional draw).
The one arm that let the prediction loss shape the encoder (jepa, λ 0.1) did move the
code toward predictability (rank 70 → 19, flow share 0.31 → 0.27, a one-step sample 2.5
nats better on a trusting coda) but with a rank tripwire as the only guard the code
collapsed toward a lower-rank copy and the gains died in rollout. That is the textbook
JEPA failure: the prediction objective alone drifts to dimensional collapse, and the
heuristics (partial gradient, an abort) are ad hoc.

## Proposal

LeJEPA's objective on the code (arXiv 2511.08544): `code_target_lambda 1.0` (the flow
loss's gradient reaches E in full, no stop-gradient) plus `code_sigreg_lambda 0.05` with
1024 directions, SIGReg applied per cell index over the valid slots' codes. The coda's
CE through the truth cells keeps the code decodable; SIGReg keeps its distribution
isotropic Gaussian, which the paper proves minimises downstream prediction risk and which
forbids dimensional collapse by construction. Inference is unchanged.

## Alternatives considered

- **Sweep λ_target on the jepa arm (0.03, 0.3) with the rank abort**: a weaker version of
  the same idea with a heuristic guard; rejected in favour of the principled guard.
- **VICReg variance/covariance terms on the code**: the same class (a moment penalty), no
  distributional guarantee, quadratic in the dimension; SIGReg is linear and bounded.
- **A discrete plan code (VQ)**: a different object; kept as the next lever if this one
  leaves the code a copy.
- **SIGReg alone with the stop-gradient (λ_target 0)**: regularises a target the
  prediction loss cannot move; pointless.

## Acceptance criteria

P-LJ1 (rank ≥ 40 throughout), P-LJ3 (ce_tf in [0.5, 2.5]) and P-LJ5 (paired gap ≤ +0.50)
on one seed move the note to `implemented/`; P-LJ1 holding with P-LJ3 failing moves it to
`rejected/` with the reading that the CE anchor wins over the full gradient.

## Risks

- λ 0.05 is the paper's image-domain default; on ≈ 200 codes per cell per step the
  statistic is noisier, and the first slot-state SIGReg arm died on a badly picked λ.
- The full gradient lets the flow loss reshape the code while the coda reads samples in
  phase 3; the rank tripwire is the only guard against a late collapse.
