# Does the new eval regime move training?

Status: success

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

## Results

The first launch died at step 1000 in a power loss (10:33); the relaunch ran from scratch at
18:11 on the same tree. Log `chain-regime.txt`.

| Run | Eval regime | Seed | CE@1 | CE@6 | K1-K6 [95 % CI] | grad-norm max/median |
|---|---|---|---|---|---|---|
| fp32 carrier (this run) | old: release, every eval, no fork | 2 | 3.9127 | 3.9076 | +0.0051 [+0.0046, +0.0056] | 7.7 |
| fp32 carrier (matched controls) | new: keep, every 4th, fork | 2 | 3.9093 | 3.9029 | +0.0064 [+0.0059, +0.0068] | 8.1 |

## Verdict

Success: CE@6 3.9076 >= 3.900, so the new eval regime costs nothing; on seed 2 the old regime
reads 0.0047 HIGHER. The earlier seed-1 runs at 3.8949-3.8996 were one family of near-copied
trajectories, not the recipe's typical value. Regime B stays in the speed base.

## Updated hypothesis

Fresh 5k trajectories of the fan speed recipe land between CE@6 3.903 and 3.908 (five runs:
3.9029, 3.9040, 3.9060, 3.9061, 3.9076), a spread of about 0.005. Pass lines for later speed
checks are set against the mean of matched fresh runs, never against the old seed-1 family.
The trainer rate on this run was 27.4k against about 29k on its sibling; two builders were
resuming CPU-side work at the same time, and this note does not separate those.
