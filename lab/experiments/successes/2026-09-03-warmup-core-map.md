# Planned: is the warmup core map closer to identity? (the shallow-path reading, measured)

Status: success
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

## Results (2026-09-03 01:10, jac_ladder on tul-a2-wu5k step_2500 / step_5000; conv <= 4e-6)

| ckpt | iter | sigma_step (worst) | rms_step (typical) | per-block rms | alignment |
|---|---|---|---|---|---|
| wu5k 2500 | 0 | **25.7** | **0.994** | 1.025–1.036 | 0.83 |
| flat A2 2500 | 0 | 55.0 | 1.052 | 1.036–1.057 | 0.82 |
| wu5k 5000 | 0 | **26.0** | **1.053** | 1.041–1.054 | 0.81 |
| flat A2 5000 | 0 | 97.3 | 1.134 | 1.054–1.086 | 0.78 |
| wu5k 2500 | 3 | 14.4 | 0.918 | ~1.02 | 0.81 |
| wu5k 5000 | 3 | 18.9 | 0.936 | ~1.03 | 0.78 |

Artifact: lab/experiments/results/a2_jac_ladder_wu5k.json.

- **P-M1 (65%): TRUE.** 0.994 < 1.052 and 1.053 < 1.134.
- **P-M2 (60%): TRUE.** 26.0 < 97.3 (and 25.7 < 55.0 at 2500).
- **P-M3 (60%): TRUE, barely.** sigma 25.7 -> 26.0 (+1%) and rms 0.994 ->
  1.053 (+6%), against the flat run's +77% / +8%. Outward, but slowly.

## Verdict

**SUCCESS.** Under the ramp the core map is near identity: at 2500 its
typical gain is slightly contractive (0.994) and by 5000 it is 5% expansive,
against 5% and 13% on the flat schedule; its worst direction is 26 in both
places, against 55 and 97. Per-block gains are lower too (1.03 vs 1.05 at
2500; 1.05 vs 1.07 at 5000), so the map's anisotropy, not just its level,
is smaller. The alignment ratios are the same (0.78–0.83 everywhere), so
this is a magnitude difference, not a directional one.

Reading, now measured: the ramp lets the prelude/coda path organize first,
the loop does less per iteration (a near-identity map cannot move the state
much in six applications), and K1−K6 falls for that reason. The same
near-identity map is why the warmup run is stable: the ρ^T cliff is far
away. The loop's earning and the loop's instability were the same quantity
on the flat schedule, the gain of the core map above 1. Whether the ramped
map keeps drifting outward (rms +6% per 2500 steps here) and the earning
grows with it is the pair's earning-over-time curve at 10k/15k/20k.

## Updated hypothesis

Loop earning under this architecture tracks the core map's gain above 1,
and the gain is set in the early window by whichever path organizes first.
A knob that raises the gain later (after the codebook and the shallow path
have formed) may buy earning without the early detonation; the first
candidate is the LR after the ramp (a higher post-ramp LR, or a shorter
ramp, e.g. 300 steps) with the tripwire on.
