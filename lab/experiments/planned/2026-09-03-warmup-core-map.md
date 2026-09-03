# Planned: is the warmup core map closer to identity? (the shallow-path reading, measured)

Status: planned
Date: 2026-09-03 (frozen 00:40, before the probe runs; trigger: Wolfe — "warmup
alone shouldn't change the dynamics that much right? this is very peculiar")

## Question

Under `training.warmup=1000` the loop's K1−K6 earning is one third of the
flat schedule's at both 2500 and 5000, while the model is better at every
depth (K1 0.19 better, K6 0.077 better at 5000). Reading: the ramp lets the
prelude/coda path organize first and the loop does less per iteration. Is
that visible in the operator? Flat A2 at iter 0: typical gain 1.052 (2500)
and 1.134 (5000), worst direction 55 and 97, drifting outward.

## Method

`lab/divergence/jac_ladder.py --config-name tul_a2 --overrides
training.batch_size=2,model.use_kernels=false,training.warmup=1000 --ckpt-dir
checkpoints/morph/tul-a2-wu5k --iters 0,3 --power-iters 200`, same batch and
seed as the 2026-09-02 ladder. Inserted into the chain after gla1, before
gla2 (~10 min). Output lab/experiments/results/a2_jac_ladder_wu5k.json.

## Predictions (frozen)

- **P-M1.** wu5k typical gain (rms_step) at iter 0 is below flat A2's at the
  same step, at BOTH 2500 (< 1.052) and 5000 (< 1.134): **65%**.
- **P-M2.** wu5k worst-direction sigma_step at iter 0 at 5000 is below flat
  A2's 97.3: **60%**.
- **P-M3.** wu5k still drifts outward (sigma_step 5000 > 2500): **60%**; the
  ramp changes the level, not the direction of travel.

## Interpretation

M1 TRUE ⇒ the loop does less per iteration under the ramp and the shallow-
path reading stands; the loop's share is a property of who organized first.
M1 FALSE (map as expansive as flat) ⇒ the map is not where the earning
difference lives; look at the prelude/coda (per-block CE attribution) next.
Data-order dependence is a separate question: two warmup draws at data seeds
2 and 3 after the pair, sweep at 2500.

## Not verified before run

The `training.warmup` override is irrelevant to the model build (LR only) and
is passed so the loaded config matches the checkpoint's; nothing else new.
