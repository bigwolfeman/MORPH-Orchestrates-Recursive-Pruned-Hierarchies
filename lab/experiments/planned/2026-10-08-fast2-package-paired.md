# FAST2 package plus online teacher, paired against production

Status: planned

## Question

Does the whole speed package as committed on perf/graph-step at efb5bf7a (graph-captured step,
compiled blocks, fused TG attention, HC fused grad, ternary step cache, no checkpoint recompute,
compiled fan twin, fp32 fan diversity terms, bf16 lsel pick) plus the online-prelude router
teacher keep the loop's contribution within the owner's 5 % rule over a 1000-step continuation?

## Hypothesis

Every key but the online teacher changes rounding only; rounding chaos does not move a 1000-step
continuation outside the paired noise. The online teacher changes the router's target by at most
the EMA lag (teacher pick agrees 93-98 % per pass), and the model CE does not depend on it while
`fan_lsel_train_follow` is `router`.

## Predictions

Baseline: the two no-change continuations of the same checkpoint (`mixed/2026-10-08-paired-continuation-noise.md`):
K1-K6 +0.0086 / +0.0091, CE@6 4.0461 / 4.0468.
- K1-K6 (ce_tokens, 480 rows) of each package continuation within 0.001 of the baseline mean
  (+0.00885): best guess +0.0085.
- CE@6 within 0.003 of the baseline mean (4.04645).
- Pass rule: both package continuations inside both bounds. One outside: rerun that arm once more.
  Both outside in the same direction: the package fails; bisect online teacher vs the rest next.

## Method

Two continuations, c and d, from `checkpoints/morph/lxtul-pointer-rerun/step_2500.pt`,
`training.resume=<ckpt> training.steps=3500`, recipe FAST2 (`/home/wolfe/morph-scratch/perf/graph/BRIEF.md`)
plus `model.fan_target_online=true`, tree = git archive of efb5bf7a, WANDB offline. Readout as the
baseline: `core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on each final checkpoint.
