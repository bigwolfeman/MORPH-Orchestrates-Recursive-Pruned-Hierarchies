# Planned: L3-WAKE-CAP — is the σ cap the waking agent?

Status: failure
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

---

## Results (filed 2026-08-30; artifacts in `../results/2026-08-30-tul-l3-wake-cap/`)

Run tul-l3wake-cap (wandb 2c95q3ut), exit 0, σ pinned 1.50 the whole run.

- **PWC1 (90%): HELD.** S1 clean.
- **PWC2 (55%, binding): FAILED — wrong side of my prior.** Depth sweep flat:
  4.2221 @1 vs 4.2213 @6 (|Δ|=0.0008 < 0.02). With the uncapped twin ALSO flat
  (|Δ|=0.0004), the three-way rule resolves: **neither wakes ⇒ DB init is an inert
  basin for depth. From-scratch l2cap stays the recipe; DB init is dropped from the
  loop program.**
- **PWC3 (30%): failed (right side).** 0.000 nats of depth vs the ≥0.10 bar.
- **PWC4 (40%): failed (right side).** Streak 2 at the end (0.044, 0.052) — same
  shape as the uncapped twin; misses the 3-consecutive bar.
- The cap was FREE: CE @4250 4.2009 vs uncapped 4.2044 (inside replicate spread);
  generation slightly healthier than uncapped (topk50 rep4 0.063/distinct3 0.873 vs
  0.100/0.828) but nowhere near from-scratch l2cap (0.045/0.931, greedy 0.61/0.34).

## Verdict

Failure — the binding prediction (cap wakes the loop, 55%) was wrong. The pair's
answer is clean: contractivity control must be present WHILE the loop structure
forms; applied post-hoc it neither wakes the loop nor hurts anything. The one-pass
DB solution is a basin end-to-end training does not leave at this LR/budget.

## Updated hypothesis

Same as the wake filing's: depth-earning structure forms during training under the
cap, or not at all. The open budget-matched question (9000-step from-scratch l2cap
vs DB→wake 4500+4500 at CE 4.2044) is the next decision point for the loop program.
