# The span decoder grades each cell alone

Status: planned

## Question

The span decoder (0.69 of the loop core's gradient) grades the MEAN of the 4 exit cells. Two other
reads are queued: the winner cell (`2026-10-10-spandec-reads-winner.md`). This arm grades each cell
ALONE and averages the 4 losses (`tul.spandec_read: each`), so every candidate gets the gradient of
its own decodability. Under the mean read all 4 cells get one identical cotangent (a consensus
target); under `each` they get different ones. Does a per-candidate text signal help the loop and
the selection?

## Hypothesis

If the selection needs a pool of candidates that are each decodable, `each` is the right target:
CE improves over the mean read, the gap between the best cell and the routed cell
(`fan/oracle_gap`) stays small, and K1-K6 holds. It costs 4x the decoder's compute, so a win here
is a credit result, not yet a recipe.

## Predictions

References, `lxtul_pointer_fast` at 5k, seed 1, 480 rows: CE@6 mean 3.9239 (3.9248 / 3.9229),
K1-K6 mean +0.0098 (+0.0104 / +0.0091), `fan/oracle_gap` 0.0309 / 0.0347. Spread: 0.002 CE@6,
0.006 K1-K6. One run.

- CE@6 best guess 3.916. Help at or below 3.918; harm at or above 3.930.
- K1-K6 best guess +0.011. Harm below +0.006; help above +0.016.
- `fan/oracle_gap` best guess 0.030 (at or below the references').
- Trainer rate late: 15 to 25 % below the references' 29.8k tok/s (the decoder runs on 4x the rows).

## Method

`--config-name lxtul_pointer_fast_spdeach`, from a frozen `git archive` of a commit that contains
this file and the knob, 5000 steps, seed 1, WANDB offline, one run under `gpu.lock` through
`/home/wolfe/morph-scratch/abq/runner.sh`. Readout: `core_depth_sweep.py --ckpt
spdeach=lxtul_pointer_fast_spdeach=<step_5000> --depths 1,6 --rows 480 --batch 3`.
