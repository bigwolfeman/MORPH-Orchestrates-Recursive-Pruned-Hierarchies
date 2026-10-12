# Does the new eval regime move training?

Status: planned

## Question

Six earlier seed-1 runs of the fan recipe family (eval regime A: graphs released around eval,
probe block every eval, no RNG fork) read CE@6 3.8949-3.8996. Four runs under eval regime B
(graphs kept, probe block every 4th eval, RNG forked around eval), seeds 1 and 2, read
3.9029-3.9061. Does regime B cost CE, or is regime A's seed-1 family a lucky draw?

## Hypothesis

Regime B changes no training computation: graph keeping is bit-identical over 120 deterministic
steps, and the fork only changes which random numbers training draws. So the gap is draw luck,
and regime A at seed 2 on today's tree lands with regime B's runs.

## Predictions

One run: `nowta_pointer_fast_spdcells_speed` (fp32 carrier), seed 2, regime A
(`training.eval_keep_graphs=false training.eval_diag_every=1`), today's tree. Compare with
regime B's fp32 seed-2 run (CE@6 3.9029, K1-K6 +0.0064).

- Best guess CE@6 3.904. If CE@6 >= 3.900: no regime effect (draw luck), regime B stays.
- If CE@6 <= 3.898: the regime moves training; regime B is pulled from the speed base and the
  cause is bisected (graph keeping vs the RNG fork vs the probe cadence).
- Between the two: inconclusive, one more seed of each regime.

## Method

Master at the commit adding this file, snapshot `/home/wolfe/morph-scratch/abq/tree-o`. 5000
steps, WANDB offline, under `gpu.lock`, then the 480-row sweep and `stab.keep.py`.
