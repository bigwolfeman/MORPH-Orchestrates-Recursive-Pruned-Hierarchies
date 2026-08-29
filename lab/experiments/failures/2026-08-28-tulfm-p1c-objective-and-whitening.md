# Planned: P1c — does target whitening unlock the planner, and does true flow matching match EDM?

Status: failure — C1 missed; the gate metric is capped at ~2.5x chance across five configurations
Arc: [`.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md`](../../../.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md)
Predecessor: [`2026-08-28-tulfm-p1.md`](2026-08-28-tulfm-p1.md)
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

## Results (filled 2026-08-28; runs tulfm-p1c-{edm-white,cfm-raw,cfm-white}, + cfm-white-12k)

Within-row lineup, chance 0.0201, P1b baseline 0.0423. Probe JSONs in
`results/2026-08-28-tulfm-p1/`.

| arm | top-1 | top-5 | MRR | med rank /50 | untrained | shuffled |
|---|---|---|---|---|---|---|
| edm_white (4k) | 0.0421 | 0.1568 | 0.1264 | 19.9 | 0.0203 | 0.0173 |
| cfm_raw (4k, clip 50) | 0.0371 | 0.1694 | 0.1275 | 19.8 | 0.0226 | 0.0158 |
| cfm_white (4k) | 0.0510 | 0.1935 | 0.1452 | 17.5 | 0.0203 | 0.0168 |
| **cfm_white (12k)** | **0.0516** | 0.2048 | 0.1517 | 16.6 | 0.0203 | 0.0160 |

- **C1 FAILED, decisively and informatively**: edm_white 0.0421 = P1b's 0.0423.
  Whitening moved the EDM arm by nothing. Per this file's own falsification clause,
  the ANISOTROPY HYPOTHESIS IS DEAD as the binding constraint.
- **C2 HELD**: |0.0510 − 0.0421| = 0.0089 ≤ 0.0128. True CFM and EDM find the same
  signal — the objective family is second-order.
- **C3 HELD** (0.0421 > 0.0371), weakly given C1.
- **C4 HELD** in all arms, both scopes.
- **C5 HELD**: 1-2 of 200 logged steps touched the clip (P1/P1b: ~100%).
- Follow-up (pre-authorized): cfm_white at 12000 steps → top-1 0.0516. PLATEAU.
  Depth metrics improved mildly (top-5 .1935→.2048, MRR .1452→.1517); the gate metric
  did not move.

## Verdict: FAILURE (C1 missed; the 0.06 gate was never reached on any axis)

Five configurations now bracket the same number: sigma_data x2, objective family x2,
whitening x2, steps x3 — all land within 0.037-0.052 within-row top-1. The signal is
REAL in every arm (controls at chance, shuffle kills it) and CAPPED at ~2.5x chance.

## Updated hypothesis

The cap is not a planner-side artifact — every planner-side knob has now been varied
and the number does not move. The remaining explanation is INFORMATION: how much the
frozen A3 context representation (a 4.0-nat, 207M, seq-1024 backbone) actually
determines the next span's pooled representation. Raising it means a stronger or
longer-context backbone, or wiring the coda so the plan trains against token loss
under scarcity (P2) — a different experiment with a different gate, not a P1 iteration.
Per the frozen decision rules of BOTH P1 files, P1 is closed: mechanism proven
(conditional, causal, stable, cheap, no-BPTT), magnitude under gate.
