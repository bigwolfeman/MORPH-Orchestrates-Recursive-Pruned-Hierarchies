# Planned: A2c — the capped paid loop (contingency for the 20k lottery)

Status: planned
Date: 2026-09-02 (frozen ~14:20, BEFORE any capped run and BEFORE the
outcome of tul-a2-20k attempt 2 is known)

## Question

Does the l2cap spectral projection (sigma_max <= 1.5, post-step, core
linears) let the paid loop train reliably — and does it keep the earning?

## Hypothesis

The paid-axis detonation is map expansivity: the beta1=0 optimizer walks
the core's weights across the rho=1 manifold and the loop amplifies the
excursion T-fold (the nested-dynamical-system frame). A hard sigma cap
bounds the map's linear gain each step, which the slot-axis evidence says
is compatible with depth-earning (tul_l2 earned 0.233 nats WITH this exact
cap). Counter-evidence to respect: the 2026-08-24 takeover-cure campaign
showed spectral caps do NOT stop alignment-driven takeover — but that was
a different pathology (rank collapse, not gradient explosion), and the
detonations here are explosions.

## Method

Config `tul_a2c` = `tul_a2` + exactly ONE delta:
`training.spectral_project_cap=1.5`. Trigger rule (frozen): this arm runs
ONLY if tul-a2-20k attempt 2 fails (vanilla A2 then stands 0/2 at 20k, 1/3
overall). If attempt 2 completes, A2c is optional follow-up work, not a
substitute — the walkover 20k stands. If triggered: 20k, panel flags,
ckpt_every 2000, prune, a2_depth_sweep, no gen samples, ONE retry allowed
(the 5k convention, since a capped detonation would itself be decisive).
The smoke gate for a capped arm must see "Core spectral PROJECTION ON" —
note this INVERTS the projection-absent gate every prior arm used.

## Predictions (frozen)

- **P-C1 (stability).** A capped A2 20k completes with no div-guard abort,
  within two draws: 75% (per-draw ~55-60% if the cap only half-works; the
  hypothesis says higher).
- **P-C2 (earning survives the cap).** Capped 20k a2-sweep K1-K6 >= 0.10:
  55% (l2's slot-axis precedent says caps and earning coexist).
- **P-C3 (CE cost).** Capped final val CE within 0.05 of the best
  completed uncapped A2 comparison available at matched step: 50%.

## Binding

P-C1 TRUE and P-C2 TRUE => the cap is the paid axis's missing guard; it
returns to the recipe for paid arms and the 20k readout proceeds against
tul-20k/notul-20k. P-C1 FALSE => the explosion is not (only) map
expansivity — move to optimizer-side interventions (alpha_cap, t_beta3,
beta1>0 for paid arms) with a fresh prereg. P-C2 FALSE with P-C1 TRUE =>
the cap trades earning for stability on the paid axis too — the
identity-escape tension is real and the next arm searches the cap value.

## Not verified before run

Whether the projection's cost profile changes under A2's fused kernels
(l2 ran the slot loop); the cap's interaction with tokens_through_core is
untested by anyone.
