# Paired continuation noise: can one checkpoint pair resolve 5 % of K1-K6?

Status: planned

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
