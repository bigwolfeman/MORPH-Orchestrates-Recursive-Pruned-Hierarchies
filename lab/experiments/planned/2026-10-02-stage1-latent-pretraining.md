# Planned: can the slot loop learn a span latent with no coda, and do its passes add to it?

Status: planned

Date: 2026-10-02 14:28 CDT (frozen before any arm trained beyond 45-step smokes).
Design note: [`2026-10-02-staged-latent-pretraining.md`](../../../.agents/notes/proposed/architecture/2026-10-02-staged-latent-pretraining.md).
Configs (commit 962dd35): `tul_latent_pre_prelude_l2`, `tul_latent_pre_prelude_nce`,
`tul_latent_pre_prelude_wta4`, `tul_latent_pre_final_l2`, `tul_latent_pre_ema_l2`.
Wolfe (2026-10-02): "Right, we may get blurred means. We should try those variants."

## Question

Stage 1 trains the prelude and the strict slot loop on a latent of the next span alone:
no coda, no token CE. Two questions decide whether stage 2 runs. Is the latent learnable
from the past at all (retrieval of the true next span among the same row's spans, above
chance)? And do the loop's passes add to it (retrieval at pass 6 above pass 1)? The arms
vary the loss (L2, InfoNCE, a 4-cell winner-takes-all), the target (the frozen plain
model's prelude or final hidden state, standardised per coordinate) and include the live
EMA target as the collapse control.

## Hypothesis

The latent is weakly learnable: retrieval sits a few times above chance, highest under
InfoNCE. The passes add little: the 2026-09-17 code-only arm met its target in one pass,
and per-pass targets have been met in one step since 2026-09-13. The plain-prelude target
is low-dimensional (standardised participation ratio 6.4), so its retrieval saturates early.

## Predictions (mine, orchestrator, before the runs)

Chance (logged): same-row 0.0172, all-row 0.0029 at the smoke's batch; the run logs its own.

- **S-1**: no arm detonates. 85 %.
- **S-2**: plain-prelude L2: same-row retrieval at the exit at least 2x chance. 55 %.
- **S-3**: InfoNCE's same-row retrieval at the exit is above L2's. 70 %.
- **S-4**: in at least one frozen-target arm, same-row retrieval at pass 6 exceeds pass 1 by
  more than 2 percentage points. 30 %.
- **S-5**: winner-takes-all's best-of-4 same-row retrieval exceeds single-cell L2's. 60 %.
- **S-6**: final-hidden target: same-row retrieval at the exit at least 2x chance. 50 %.
- **S-7**: live-EMA control: target rank at step 3000 below half its first logged value. 70 %.
- **S-8**: every frozen-target arm trains at >= 1.15x the one-rollout loop's tok/s (11574). 70 %.

Gate (from the design note): stage 2 runs on an arm with same-row retrieval at the exit at
least 2x chance AND pass 6 above pass 1, both with a CI. The CIs come from a 480-row
post-hoc probe on the step-3000 checkpoints (to be built before filing; the val logs give
point values only).

## Method

- Seed 1, 3000 steps each, `runner_steps.sh` at 962dd35, in the order plain-prelude L2,
  InfoNCE, winner-takes-all, final hidden, live EMA. The runner's coda readouts (depth
  sweep, score, worth) do not apply to a coda-free model and are ignored.
- Readings from the val logs at every eval: per-pass retrieval (same-row, all-row) with
  chance, R^2, prediction rank, target rank; winner-takes-all's best-of-4 and cell spread.
