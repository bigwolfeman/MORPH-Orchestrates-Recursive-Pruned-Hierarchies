# lxtul_pointer_fast without the span decoder at 5k

Status: planned

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

## Verdict

## Updated hypothesis
