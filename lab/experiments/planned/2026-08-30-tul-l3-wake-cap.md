# Planned: L3-WAKE-CAP — is the σ cap the waking agent?

Status: planned
Date: 2026-08-30, frozen BEFORE the uncapped wake arm's post-run depth sweep was read
(tul-l3wake was mid-training at freeze). Config: `tul_l3wake_cap.yaml` (= tul_l3wake +
training.spectral_project_cap 1.5). Wolfe: "the diffusionBlocks retrain probably
should use l2cap."

## Question

Same DB init (tul-l3 step_4500), same AdamW full-BPTT wake schedule, plus the hard
σ≤1.5 projection. Against the uncapped wake arm this separates: standard training
wakes the loop (both arms wake) / the cap is the waking agent (only this arm wakes) /
the DB init is an inert basin (neither wakes).

## Reference numbers (fixed)

l3 pre-wake: depth-FLAT (4.3928 @1 vs 4.3917 @6, 48 rows), CE @4250 4.3519, plan
content unused (shuffle profile +0.013 flat). l2cap-from-scratch: 0.233-nat curve,
CE 4.3489, worth 0.146. Uncapped wake arm (running at freeze): prereg PW2 gave it
35% to develop a ≥0.02-nat curve.

## Predictions (frozen)

- **PWC1.** Completes with S1 clean: 90%.
- **PWC2 (binding).** Post-run depth sweep shows CE(1) − CE(6) ≥ 0.02 nats: 55%
  (vs 35% for the uncapped twin — the difference IS my claim that the cap is the
  active ingredient). Decision rule: capped wakes AND uncapped does not ⇒ cap is
  the waking agent, wake recipe adopts it; both wake ⇒ standard training suffices
  and the cap is a booster; neither wakes ⇒ DB init is an inert basin for depth —
  from-scratch l2cap stays the recipe and DB init is dropped from the loop program.
- **PWC3.** Reaches ≥ 0.10 nats of depth (half of from-scratch l2cap) within the
  4500-step wake budget: 30%.
- **PWC4.** worth_shuffle ≥ 0.04 sustained (ladder definition): 40%.

## Method

Queued after L2-TRUNC (single-tenant GPU). Panel flags, seed 1, wandb
tul-l3wake-cap. No smoke (tul_l3wake path smoked tonight; cap is the tested
projection). Post-run: core_depth_sweep + tul_samples + prune. Artifacts →
lab/experiments/results/2026-08-30-tul-l3-wake-cap/.
