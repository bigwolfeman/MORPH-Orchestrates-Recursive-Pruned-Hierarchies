# Planned: P1c — does target whitening unlock the planner, and does true flow matching match EDM?

Status: planned
Arc: [`.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md`](../../../.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md)
Predecessor: [`../failures/2026-08-28-tulfm-p1.md`](../failures/2026-08-28-tulfm-p1.md)
Doctrine: [`../../../docs/tul-fm-probing.md`](../../../docs/tul-fm-probing.md) §6–7

## Question

P1b wrote real conditional content (within-row top-1 0.0423, chance 0.0201, controls at
chance) but missed the 0.06 gate. The subspace diagnostic says the planner spends its
capacity on the ~999 dead target dimensions. Two questions, one panel:
(1) does whitening the targets to their live subspace unlock the missing signal?
(2) does a TRUE conditional flow-matching objective (straight-line interpolation,
velocity target — Wolfe directive) reproduce the EDM arm's result under the identical
retrieval gate?

## Hypothesis

Anisotropy is the binding constraint; the objective family is second-order at this scale.

## Method

Three arms, one variable each vs P1b (which is the fourth cell: edm + raw). All:
frozen `a3-s2/step_4500`, 4000 steps, batch 8, seed 0, `training.loss_scale: auto`,
identical probe (8 held-out batches, both scopes, both controls). Configs:
`tulfm_p1c_edm_white` (edm, whiten rank 64, sigma_data 1.0),
`tulfm_p1c_cfm_raw` (cfm, raw unit-norm, source_std 0.031),
`tulfm_p1c_cfm_white` (cfm, whiten rank 64, source_std 1.0).
Whitened arms are probed in whitened space; controls keep cross-arm comparison fair.
NOTE `loss_scale: auto` is a second change vs P1b in every arm; it is accepted because
P1b's grad-clip saturated ~150x every step, making "one variable" already false — the
saturation itself was an uncontrolled variable. Its predicted effect is C5.

**Method amendment, 2026-08-28 (pre-launch, before any P1c arm has taken a step):**
the builder's CPU measurement on the real checkpoint found `cfm_raw`'s post-scaling
gradient norm is |g| ~= 11-40 (its residual lives in 1024 dims; the white arms' live in
64-128 and measure |g| ~= 0.6). At `grad_clip=1.0` that arm would saturate the clip
every step — the exact condition C5's premise excludes. `cfm_raw` therefore runs with
`training.grad_clip=50` (CLI override, logged to wandb); the other arms are untouched.
Reason: preserve C5's premise for all arms rather than knowingly launch one that
violates it. Predictions are unchanged.

## Predictions (frozen 2026-08-28, before any P1c arm has taken a step)

- **C1 — whitening unlocks the gate.** `edm_white` within-row top-1 ≥ **0.06** and
  MRR ≥ **0.12**. If `edm_white` lands BELOW P1b's 0.0423, the anisotropy hypothesis is
  falsified as the binding constraint.
- **C2 — objective family is second-order.** |top1(cfm_white) − top1(edm_white)| ≤
  0.25 × max of the two (within-row).
- **C3 — whitening dominates objective.** min(edm_white, cfm_white) > cfm_raw (within-row top-1).
- **C4 — controls stay honest in every arm.** Untrained ≤ 2× chance, shuffled-context
  ≤ 1.5× chance, both scopes, all three arms.
- **C5 — the clip saturation disappears.** With loss_scale auto, fewer than 20% of
  steps hit grad_clip in every arm (P1/P1b: ~100%).

## Decision rule

| Outcome | Action |
|---|---|
| Any arm passes C1's bars with C4 held | P1 gate PASSES (retroactively). Write the P2 plan (coda under token scarcity). |
| Best arm beats P1b but misses 0.06 | ONE pre-authorized follow-up: rerun the best arm at 12000 steps, nothing else changed. Then verdict, no further P1 iterations. |
| All arms ≤ P1b | Anisotropy hypothesis dead. The arc-note rejection question returns to Wolfe with no further P1 proposals from me. |
| C4 fails anywhere | That arm is void; diagnose before reading its other numbers. |
