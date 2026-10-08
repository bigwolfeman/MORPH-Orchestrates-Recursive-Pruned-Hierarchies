# Slot-loop compaction, paired against the FAST2 package

Status: planned

## Question

Do the two compaction keys (`model.slot_depth_stratified=true model.slot_compact=gather`, f1ac1b9c)
keep CE and the loop's contribution over a 1000-step continuation? They change the training
function twice: the depth draw becomes stratified per row (each slot's marginal unchanged, the
joint changed), and frozen cells serve the attention input cached at their last live pass instead
of a fresh pass (fixed-weight loss rel 2.6e-4, up to 8 of 80 winner picks flipped).

## Hypothesis

Neither change moves CE or K1-K6 beyond the paired spread: the marginal depth law is what K1-K6
reads, and the cache difference is a small per-pass perturbation the fixed-point term already damps.

## Predictions

Baseline: package continuations c, d, e (CE@6 4.0461 / 4.0457 / 4.0471; K1-K6 +0.0088 / +0.0199 /
+0.0082) and eager a, b, f (CE@6 4.0461 / 4.0468 / 4.0462; K1-K6 +0.0086 / +0.0091 / +0.0098).
Pooled six: CE@6 mean 4.04633, K1-K6 median +0.00895.
- CE@6 of each compaction run within 0.002 of 4.04633; best guess 4.0465.
- Mean K1-K6 of the three runs at or above +0.0075 (no drop larger than about 16 % of the pooled
  median); best guess +0.0090.
- Resolution, stated now: three pairs cannot resolve the owner's 5 % (0.00045); this test detects
  a K1-K6 drop of about 0.0015 or more. A pass means "no drop that large", not "within 5 %".
- Pass rule: both predictions hold. CE miss on one run only: rerun it once. Mean K1-K6 below
  +0.0075: run the stratified-only decomposition arm (three runs) to split draw vs cache.

## Method

Tree: git archive of the commit of this file (compaction f1ac1b9c included). Three continuations
g, h, i from `checkpoints/morph/lxtul-pointer-rerun/step_2500.pt`, `training.resume=<ckpt>
training.steps=3500`, recipe FAST2 + `model.fan_target_online=true` (as the package runs) +
`model.slot_depth_stratified=true model.slot_compact=gather`, WANDB offline. Readout
`core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on each final checkpoint.
