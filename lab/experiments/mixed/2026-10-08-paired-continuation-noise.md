# Paired continuation noise: can one checkpoint pair resolve 5 % of K1-K6?

Status: mixed

## Question

Two continuations of ONE production checkpoint, same code, same seed, same data order, no change:
how far apart are their K1-K6 and CE@6 after 1000 steps? This is the noise of the paired design
that the 2026-10-08 speed plan uses to test function-changing speed cuts against Wolfe's rule
(loop contribution within 5 % of baseline, about 0.0006 of K1-K6).

## Hypothesis

Chaos decorrelates MORPH runs within ~11 steps, but both continuations start from the same weights
and optimizer state, so most of the slow drift that separates two 5k runs (0.006 in K1-K6, filed
2026-10-08) has no time to build.

## Predictions

- |delta K1-K6| (ce_tokens, 480 rows) at the end: between 0.0005 and 0.004; best guess 0.002.
- |delta CE@6|: between 0.001 and 0.006; best guess 0.003.
- Decision rule written now: if |delta K1-K6| <= 0.0006, one paired continuation per cut can test
  the 5 % rule; if it is 0.0006-0.002, 4 pairs per cut; above 0.002, the paired design cannot test
  5 % at affordable cost, and function-changing cuts are judged on a looser stated bound.

## Method

Checkpoint: `checkpoints/morph/lxtul-pointer-rerun/step_2500.pt` (the 2026-10-08 production
rerun, `lxtul_pointer`, eager, seed 1). Two runs, identical: `--config-name lxtul_pointer
training.resume=<ckpt> training.steps=3500 wandb.name=paired-noise-{a,b}`, WANDB_MODE offline,
eager production recipe (no speed keys), tree `perf/graph-step` at the commit of this file.
Readout of each final checkpoint: `lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480
--batch 3`. Results to `results/2026-10-08-paired-continuation-noise/`.

## Results

Artifacts: [`results/2026-10-08-paired-continuation-noise/`](../results/2026-10-08-paired-continuation-noise/).
Both continuations logged "Resumed model+scaler+RNG from checkpoint step 2500"; final trainer
val 4.0901 / 4.0906.

| continuation | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | spandec K1-K6 |
|---|---|---|---|---|
| a | 4.0547 | 4.0461 | +0.0086 [+0.0079, +0.0092] | +0.0118 |
| b | 4.0559 | 4.0468 | +0.0091 [+0.0084, +0.0097] | +0.0093 |
| gap | | 0.0007 | 0.0005 | 0.0025 |

- |delta K1-K6| = 0.0005: inside the predicted 0.0005-0.004, at its low edge. Held.
- |delta CE@6| = 0.0007: below the predicted 0.001-0.006. Missed low.

## Verdict

Mixed: one prediction held, one missed low (the paired design is quieter than predicted).

## Updated hypothesis

Pairing removes about 90 % of the run-to-run K1-K6 spread (0.0005 here vs 0.006 between two
independent 5k runs). By the rule written before the run (<= 0.0006), one paired continuation
per cut can test the owner's 5 % rule; at this checkpoint 5 % of K1-K6 is about 0.00045, so the
paired noise is at that level, and two pairs per cut give a margin. One pair is one sample of the
paired noise; the span-decoder K1-K6 gap (0.0025) shows the readouts differ in noise.
