# LXTUL + pointer: a DITTO phase against self-reinforcing repetition

Status: planned (written 2026-10-07 05:17 CDT, before any run; tests/test_tul_ditto.py 12 passed, 3
sabotages caught; code at 387a033f)

## Question

The pointer head lifts LXTUL's sampled repetition from 0.041 to 0.29 seq_rep_4 at T 0.7 /
top-k 40 (plain 0.215, real text 0.020), and See et al. coverage made it worse (0.374;
lab/experiments/failures/2026-10-07-lxtul-pointer-coverage.md): coverage counts attended
positions, and an LM loop copies from fresh ones. The loops are the self-reinforcement DITTO
describes (Xu et al. 2022, arXiv 2206.02369): once a phrase repeats, each copy raises the
next. Wolfe (2026-10-07): "the foot gun of over repitition will still bite later ... I think we
should just do it." Does a DITTO phase cut the pointer model's repetition, and at what CE cost?

## Method

`tul.ditto_rows: 3`, `tul.ditto_lambda: 0.5` (`morph/configs/lxtul_pointer_ditto.yaml`): 3 of each
batch's 6 rows are rebuilt as pseudo-repetition rows (`tul_layout.pack_ditto_row`: the real
text up to a randomly chosen span, then that span repeated to fill the row; the row consumes
the same stream tokens as the row it replaces). On copy n >= 1 the CE is replaced by
-log(1 - |p_n - 0.5 sg(p_{n-1})|) on the final pointer-mixed probability, mean over those
positions, added to the CE mean at weight 1 (the paper's equal mix and lambda). The paper
fine-tunes 10k steps at 750M; here it is the same 1000-step phase as the coverage round, from
the same LXTUL + pointer 5k checkpoint (weights only, fresh optimizer, 200-step ramp), so the
control is the existing `lxtul_pointer_ft` phase (lxtul-pointer-ft, step 1000).

Readouts: the runner's sweep (re-trained rows, phase-vs-phase only, as in the coverage round's
02:18 amendment), the head-off check, the trainer's final val (clean), wandb
`tul/ditto_p_ratio` (sum p_n / sum p_{n-1} on DITTO positions; the loss drives it to 0.5) and
`tul/ditto_rows_built`, and `gen_diversity_eval.py --decodes greedy,sample_t07_k40 --n 24
--gen_len 256 --gen_batch 8` with the same prompts and seeds as the coverage round, so the
control, plain and LXTUL-without-head rows of that eval are the comparison.

Control (from the coverage round): trainer val 3.9548; T 0.7 seq_rep_4 0.291; greedy 0.909;
sweep K1-K6 +0.0180. Plain: T 0.7 0.215, greedy 0.873.

## Predictions

1. The DITTO loss learns: `tul/ditto_p_ratio` over the last 100 steps at most 0.7 (75 %).
2. T 0.7 seq_rep_4 at least 20 % below the control's (at most 0.233) (60 %).
3. T 0.7 seq_rep_4 at or below plain's 0.215 (45 %).
4. Greedy seq_rep_4 at least 0.05 below the control's (at most 0.859) (45 %).
5. CE cost: trainer final val within +0.02 of the control's 3.9548 (75 %).
6. Sweep K1-K6 within 0.004 of the control's +0.0180 (60 %).
7. No raise, no tripwire; `tul/ditto_rows_built` = 3 on every logged step (80 %).

## Results

## Verdict

## Updated hypothesis
