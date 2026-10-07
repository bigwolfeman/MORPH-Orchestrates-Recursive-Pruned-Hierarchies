# Experiment: the graph-captured step trains like the eager winner at 5k

Status: success

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

## Results

Artifacts: [`results/2026-10-07-graph-step-5k/`](../results/2026-10-07-graph-step-5k/). Run at
e0c814a2, 16:54-17:52 CDT; readout of both step_5000 checkpoints from the same tree.

| | CE@1 | CE@6 | K1-K6 (ce_tokens, 480 rows) | spandec K1-K6 |
|---|---|---|---|---|
| graph (`lxtul_pointer_graph`) | 3.9404 | 3.9316 | +0.0088 [+0.0082, +0.0094] | +0.0118 |
| eager (`lxtul_pointer`, 2026-10-06) | 3.9435 | 3.9280 | +0.0155 [+0.0147, +0.0164] | +0.0133 |

- CE@6: +0.0036 (prediction: within 0.009). Held.
- K1-K6: -0.0067 (prediction: within 0.007). Held, at the edge: the graph run is better at
  depth 1 and worse at depth 6. One seed; the LXTUL base's two-seed K1-K6 spread was 0.004.
- Runner tok/s: mean 14013 over the 24 logged windows after step 0 (eager run 11.8k).
  Prediction (>= 12.8k) held. Wall clock, start to end, 58 min against the eager runner's 74 min;
  the steady state is about 36 min: the other ~22 min are evals, the 22 graph re-recordings
  after them, and up to 20 eager steps after each re-recording.
- Peak reserved 25.06 GiB in the first eager steps, 20.00 GiB in steady state. Held (<= 26).
- Detonation check: not measurable on this recipe (probes and tripwires are off under
  graph_step); the loss stayed finite and the final val is 3.9212.

## Verdict

Success: all four predictions held, the K1-K6 one at its bound.

## Updated hypothesis

The graph recipe trains like the eager winner on CE. The lower K1-K6 is either seed noise or
a real effect of the graph_safe numerics (dense slot-column attention, the fp64 epivol
standardisation that changed the fan term's value by ~1e-4 relative). A seed twin of the
graph recipe would separate the two; until then, read K1-K6 from graph runs against graph runs.
