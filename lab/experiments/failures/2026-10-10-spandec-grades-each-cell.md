# The span decoder grades each cell alone

Status: failure

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

## Results

Run 2026-10-10, 5000 steps, seed 1, fast recipe, 480 rows, one run. Artifacts:
`lab/experiments/results/2026-10-10-spandec-grades-each-cell/` (`sweep-spdeach.json|log`,
`queue-log.txt`).

| Reading | Prediction | Result | References |
|---|---|---|---|
| CE@6 | best guess 3.916, help at or below 3.918 | 3.9159 (CE@1 3.9325) | 3.9239 mean |
| K1-K6 | best guess +0.011, help above +0.016 | +0.0166 [+0.0158, +0.0174] | +0.0098 mean |
| `fan/oracle_gap` | best guess 0.030, at or below the references | 0.0427 | 0.0309 / 0.0347 |
| Trainer rate late | 15 to 25 % below 29.8k (22.3k to 25.3k) | 18,267 (39 % below) | 29.8k |

Other readings: write worth (`val/plan_worth_zero`) 0.170, cell pairwise cos 0.98, span-decoder
CE K1-K6 +0.0445, final `val/ce_tokens` 3.9103, `fan/mixed_ce` 4.5373.

Held by the number: CE@6 and K1-K6 (both on the "help" side of their lines). Failed: the
`fan/oracle_gap` (above the references, the wrong side) and the trainer rate (39 % below, not
15 to 25 %).

## Verdict

Failure, and the help readings are inside noise. CE@6 is 0.008 better than the reference mean and
K1-K6 is 0.0068 higher. A single run resolves only about 0.015 to 0.020 of CE@6 and 0.008 of
K1-K6, so neither shift is readable. The oracle gap and the rate predictions failed on their
own.

The cells did not become distinct under the per-cell grade: cos 0.98 (LXTUL 0.95). The hypothesis
said the gap between the best and routed cell stays small. It grew from 0.031 / 0.035 to 0.043.

What the method could not distinguish: a real K1-K6 gain of about 0.007 from the 0.008 same-config
noise.

## Updated hypothesis

Grading each cell alone does not widen the cells and does not shrink the oracle gap. It costs 39 %
of the rate. K1-K6 of +0.0166 is the highest K1-K6 in the panel and the span-decoder K1-K6 of
+0.0445 is the highest in the panel (next: +0.0273), so the decoder uses the loop more here. Whether that
reaches the token CE is unresolved.

Next experiment: two more runs of the each-cell grade (three in all) and read the mean K1-K6 and
CE@6 against the references' spread. Given the 39 % rate cost, I do not propose it for the recipe
before that.
