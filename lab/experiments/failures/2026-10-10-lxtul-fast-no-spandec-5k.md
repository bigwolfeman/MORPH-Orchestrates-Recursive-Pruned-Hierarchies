# lxtul_pointer_fast without the span decoder at 5k

Status: failure

## Question

The span decoder (`tul.spandec`) is a training-only target. It costs 5.68 of 21.67 TFLOP per
step (26 %; the 9.1 TFLOP target in the phase-2 tracker leaves it out). What do loop
contribution and CE do on the fast recipe when it is off? Tracker item C1d, approved by Wolfe on
2026-10-10.

## Hypothesis

The latent teacher (the factor fan's EMA code of the next span) and the coda's token CE train
the loop without the span decoder. On Parcae LXTUL the decoder was worth +0.006 CE@6 for 16 %
of the throughput (24.4k -> 28.2k tok/s, one 5k seed each). On MORPH the speed share is larger
because the step is GPU-bound under `graph_step`. The span decoder was the target that first
moved the slot channel (2026-09-11), so part of K1-K6 may depend on it.

## Predictions

References, `lxtul_pointer_fast` from init at 5k, 480 rows: 3.9248 / +0.0104 (val 3.9161, 35
min) and 3.9229 / +0.0091 (val 3.9129, 31.3 min, trainer rate 29.8k tok/s late). Independent
runs spread about 0.006 in K1-K6.

- Trainer rate late in the run at or above 33k tok/s; best guess 36k.
- Wall clock launch to final checkpoint under 29 minutes; best guess 26.
- CE@6 (ce_tokens, depth 6, 480 rows) at or below 3.936; best guess 3.930.
- K1-K6 (ce_tokens) at or above +0.0060; best guess +0.008.
- Final val at or below 3.925; best guess 3.919.

Decision rule: if CE@6 and K1-K6 both hold, I propose to Wolfe to take the span decoder out of
`lxtul_pointer_fast` after a second seed. If K1-K6 falls below +0.0060, the decoder carries
loop contribution: it stays, and the speed work goes into its cost instead.

## Method

`--config-name lxtul_pointer_fast_nospandec` (= `lxtul_pointer_fast` + `tul.spandec: false`;
`model.spandec_ce_row_cap` cleared because it acts only on the decoder), from a frozen
`git archive` of the commit that adds this file. `wandb.name=lxtul-pointer-fast-nospandec-5k`,
WANDB offline, under `gpu.lock`. Readout: `core_depth_sweep.py --ckpt
nospan=lxtul_pointer_nospandec=<step_5000> --depths 1,6 --rows 480 --batch 3`, the same rows and
command as the references. One seed (seed 1, the references' seed).

## Results

Run 2026-10-10, tree `git archive 6a33958e`, seed 1, launch (GPU lock taken) 10:43:50, final
checkpoint 11:10:16. Artifacts: `lab/experiments/results/2026-10-10-lxtul-fast-no-spandec-5k/`
(chain log with the readout line, trainer rate lines, final val line).

| Reading | Prediction | Result | Span-decoder references |
|---|---|---|---|
| Trainer rate, late | >= 33k tok/s (guess 36k) | 34.7k (flat 34.66k-34.71k from step 800) | 29.8k |
| Wall clock | < 29 min (guess 26) | 26.4 min | 35 / 31.3 min |
| CE@6 (480 rows) | <= 3.936 (guess 3.930) | 3.9927 | 3.9248 / 3.9229 |
| K1-K6 (ce_tokens) | >= +0.0060 (guess +0.008) | +0.0020 [+0.0014, +0.0026] | +0.0104 / +0.0091 |
| Final val | <= 3.925 (guess 3.919) | 3.9806 | 3.9161 / 3.9129 |

CE@1 3.9947. The two speed predictions held; the three quality predictions missed by far more
than the 0.006 run-to-run spread of K1-K6 (the drop is 0.0071-0.0084) and the CE gap is 0.068-0.070.

## Verdict

Failure. The span decoder carries both CE (0.07 nats at 5k) and most of the loop's contribution
(K1-K6 +0.0098 mean -> +0.0020). Under the decision rule it stays in `lxtul_pointer_fast`, and
its cost (5.68 of 21.67 TFLOP per step, about 16 % of wall clock: 34.7k vs 29.8k tok/s) becomes a
speed target instead of a cut.

## Updated hypothesis

On MORPH the latent teacher and the coda's token CE do not replace the span decoder as a training
signal for the loop: the whole-span target is still the term that makes passes 2-6 useful, as it
was on 2026-09-11 when it first moved the slot channel. One seed only; the gap is about ten times
the spread, so a second seed is not needed to keep the decoder. Next: make the decoder cheaper,
not absent (tracker C1d follow-up: its CE row cap, fewer layers, or a decoder on a subset of
slots per step), each as its own preregistered arm.
