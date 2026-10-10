# CCA prologue rounds once: paired continuations against the bf16-per-op kernel

Status: success

## Question

The Triton `fused_cca_prologue` forward rounded to bf16 after every op (q/k relative error
3.9e-3 / 4.7e-3 against fp64 under training autocast, where the eager path it replaces gives
3.1e-3 rounded to bf16 and the one-rounding floor is 1.66e-3), while its backward recomputed the
forward in fp32: the gradient of a different function. The fix computes in fp32 and rounds once
at the store (`.agents/notes/implemented/bug-fix/2026-10-10-cca-prologue-rounds-once.md`). One
training-shape forward + backward on the lxtul_pointer step_5000 checkpoint moves the loss by
+5.2e-4 and the grad vector by 1.5e-2 relative (the accepted per-row -> tiled reorder moved it
0.94e-2). Over a 1000-step continuation, do CE@6 and K1-K6 stay where the old kernel had them?

## Hypothesis

The fix makes the forward the function the backward differentiates and lowers the forward's
rounding noise to the bf16 floor. That is a correctness gain at the 1e-3 level of q/k, far below
what moves a loss at 1000 steps, so CE@6 and K1-K6 stay inside the baseline's own spread. If
anything moves, the matched gradient should help, not hurt.

## Predictions

Baseline: the three fresh-pass continuations filed in
`lab/experiments/successes/2026-10-10-hinge-fresh-pass-paired.md` (same checkpoint, recipe and
readout; the tree be624816 differs from this one only by the kernel fix plus code that is inert
in this recipe: the disco arm, the decoders): CE@6 4.0482 / 4.0480 / 4.0487 (mean 4.04830);
K1-K6 +0.0117 / +0.0121 / +0.0124 (mean +0.01207); trainer val 4.0942 / 4.0938 / 4.0941. Paired
noise over three draws is 0.0012 in K1-K6; the fixed-pass arm's spread was 0.0027.

- CE@6 of each continuation within 0.002 of 4.04830; best guess 4.0480.
- Mean K1-K6 of the three at or above +0.0094 (no drop larger than 0.0027 below the baseline
  mean) and at or below +0.0148; best guess +0.0121.
- Trainer val of each within 0.003 of 4.0940; best guess 4.0938.

Resolution: as in the baseline filing, this detects a K1-K6 shift of about 0.0015-0.0027 or more
and a CE@6 shift of about 0.001-0.002; a pass means no shift that large.

Decision rule: all hold -> merge fix/cca-prologue-precision to master. CE@6 misses on one run
only -> rerun that run once. Mean K1-K6 below +0.0094 or CE@6 misses on two runs -> report to
Wolfe before merging; the kernel stays fixed in the tree either way only on his word.

## Method

Tree: `git archive` of the commit that adds this file (fix/cca-prologue-precision). Three
continuations from `/home/wolfe/morph-wt-graph/checkpoints/morph/lxtul-pointer-rerun/step_2500.pt`,
`--config-name lxtul_pointer training.resume=<ckpt> training.steps=3500`, the baseline's recipe
key for key (FAST2 + `model.fan_target_online=true model.slot_depth_stratified=true
model.slot_compact=gather`, `model.slot_gain_pass_force=-1`), WANDB offline, under `gpu.lock`.
Readout: `lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on each final
checkpoint. Chain: `/home/wolfe/morph-scratch/prologue/paired/chain.sh <commit>`. Artifacts to
`lab/experiments/results/2026-10-10-cca-prologue-rounds-once-paired/` when filed.

## Results

Run 2026-10-10 13:02-13:51, tree `git archive 2e434a60`. Artifacts:
`lab/experiments/results/2026-10-10-cca-prologue-rounds-once-paired/` (`chain.txt`, `sweep_*.json`).
Every run drew the baseline's exact pass sequence (counts 108, 119, 135, 129, 131, 119, 130, 128).

| Run | CE@6 | vs 4.04830 | K1-K6 (ce_tokens, 480 rows) | trainer val | spandec_ce K1-K6 |
|---|---|---|---|---|---|
| prologue 1 | 4.0487 | +0.0004 | +0.0107 [+0.0101, +0.0114] | 4.0930 | +0.0179 |
| prologue 2 | 4.0495 | +0.0012 | +0.0102 [+0.0095, +0.0109] | 4.0951 | +0.0159 |
| prologue 3 | 4.0508 | +0.0025 MISS | +0.0101 [+0.0094, +0.0108] | 4.0961 | +0.0179 |
| prologue 3 rerun (decision rule) | 4.0494 | +0.0011 | +0.0101 [+0.0094, +0.0108] | 4.0947 | +0.0182 |
| baseline mean (fresh-pass arm) | 4.04830 | | +0.01207 | 4.0940 | +0.0131 (fresh 1, 3) |

Against the predictions:
- CE@6 within 0.002: runs 1 and 2 hold; run 3 missed by 0.0005; under the stated rule (a miss
  on one run only -> rerun that run once) its rerun reads +0.0011 and holds.
- Mean K1-K6 at or above +0.0094: +0.01033 with run 3 or with its rerun. Held (best guess +0.0121).
- Trainer val within 0.003 of 4.0940: 4.0930 / 4.0951 / 4.0961 / 4.0947. Held.

## Verdict

Success under the decision rule; the fix merges. Stated plainly: every reading leans the same
way, against the fix, by about the test's resolution: CE@6 +0.0011 (mean of 1, 2, 3 rerun), K1-K6
-0.0017 (1.4x the 0.0012 paired noise), trainer val +0.0007, and the original run 3 missed. The
span decoder's own depth curve moves the other way (spandec_ce K1-K6 about +0.0175 vs +0.013).

## Updated hypothesis

The fix is a correctness fix (the old forward was not the function its backward differentiated)
and its cost here is at the edge of detection. One untested reading: the step_2500 checkpoint
adapted for 2500 steps to the old kernel's per-op rounding, so a 1000-step continuation under the
exact kernel starts slightly off the function its weights fit, and the shift is a resume
transient. The test that separates "transient" from "the exact kernel trains worse" is a pair of
5k runs from scratch, one per kernel, read on CE@6 and K1-K6 at 5k; not run.
