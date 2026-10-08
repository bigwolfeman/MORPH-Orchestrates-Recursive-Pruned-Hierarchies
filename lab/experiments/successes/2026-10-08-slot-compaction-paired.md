# Slot-loop compaction, paired against the FAST2 package

Status: success

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

## Results

Artifacts: [`results/2026-10-08-slot-compaction-paired/`](../results/2026-10-08-slot-compaction-paired/).
All three runs logged "Resumed model+scaler+RNG from checkpoint step 2500".

| run | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | trainer val |
|---|---|---|---|---|
| compaction g | 4.0578 | 4.0480 | +0.0098 [+0.0091, +0.0104] | 4.0926 |
| compaction h | 4.0561 | 4.0468 | +0.0093 [+0.0086, +0.0099] | 4.0915 |
| compaction i | 4.0592 | 4.0472 | +0.0120 [+0.0113, +0.0127] | 4.0920 |
| mean | | 4.04733 | +0.01037 | 4.0920 |
| eager a/b/f mean | | 4.04637 | +0.00917 | 4.0904 |
| package c/d/e mean | | 4.04630 | +0.01230 | 4.0906 |

- CE@6 within 0.002 of 4.04633: 0.0017, 0.0005, 0.0009. Held.
- Mean K1-K6 at or above +0.0075: +0.01037. Held.
- Speed in these runs: about 18.8k tok/s at step 2600 (package arm 18.2k), reserved 19.05 GB
  (package 21.0 GB).

## Verdict

Success: both predictions held; no K1-K6 drop of the size this test can see (about 0.0015).

## Updated hypothesis

Compaction keeps the loop's contribution. It may cost a little CE: every compaction run sits
above every eager and package run on trainer val (4.0915-4.0926 against 4.0897-4.0919) and its
CE@6 mean is 0.0010 above the pooled baseline. The size is about 0.001-0.0016 nats, inside the
prediction bound and at the edge of what three runs resolve. If it is real, the stratified joint
law or the cached frozen-cell input is the cause; the stratified-only arm would split them.
