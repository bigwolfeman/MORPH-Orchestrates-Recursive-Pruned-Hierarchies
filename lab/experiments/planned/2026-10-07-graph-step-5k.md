# Experiment: the graph-captured step trains like the eager winner at 5k

Status: planned

Date: 2026-10-07 (written before the run).

## Question

Does `lxtul_pointer_graph` (the winner on the graph-captured step: probes off, compile_blocks
static, CE softmax kernel, capturable optimizer, graph_safe forward, graph_step) reach the same
quality at 5k as the eager `lxtul_pointer` run of 2026-10-06, at a higher speed?

## Hypothesis

Only execution changed. The non-bit-identical parts (dense slot-column attention, the CE
kernel) change roundings at the 1e-6 level, which is chaos, not bias.

## Predictions

- CE@6 on the 480 sweep rows within 0.009 (the measured seed spread) of the eager run's 3.9280.
- K1-K6 within 0.007 of the eager run's +0.0155 (two-seed spread of the LXTUL base).
- Runner tok/s mean over steps >= 200 at least 12.8k (eager run: 11.8k); peak reserved <= 26 GiB.
- No detonation (preclip/total > 1e4) beyond one recovered spike.

## Method

One seed (seed 1, as the eager run). Worktree /home/wolfe/morph-wt-graph at the commit that adds
this file. Readout: `lab/divergence/core_depth_sweep.py --depths 1,6 --rows 480 --batch 3` on
step_5000 of both runs, from the same tree. A miss on CE by more than 0.009 is read against one
seed only and needs a seed twin before any verdict on bias.
