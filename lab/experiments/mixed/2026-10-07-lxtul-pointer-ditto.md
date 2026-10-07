# LXTUL + pointer: a DITTO phase against self-reinforcing repetition

Status: mixed (filed 2026-10-07 06:14 CDT; written 2026-10-07 05:17 CDT, before any run; tests/test_tul_ditto.py 12 passed, 3
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

Artifacts: `lab/experiments/results/2026-10-07-lxtul-pointer-ditto/` (sweep, head-off sweep,
gap, write worth, train curve, smoke log, the DITTO series from wandb run `0rirjess`, the
spike's probe rows, and the generation json / examples / log; the json also holds the
coverage round's rows for the same 24 prompts). One seed.

The smoke's first check read a wandb summary file that offline wandb 0.28 does not write and
reported FAIL; the run itself was clean (rc 0). The offline record file shows the term live:
tul/ditto 0.415 -> 0.213 over 40 steps, 3 rows built, 721-2062 DITTO positions per step. The
chain resumed at the queue step (no code change).

| reading | control | DITTO |
| --- | --- | --- |
| trainer val, final (clean) | 3.9548 | 4.0254 (+0.071) |
| sweep CE@6 (re-trained rows) | 3.8671 | 3.9601 (+0.093) |
| sweep K1-K6 | +0.0180 | +0.0294 [+0.0239, +0.0338] |
| head off CE@6 / head worth | 4.3316 / 0.465 | 4.4105 / 0.450 |
| head off K1-K6 | +0.0252 | +0.0269 |
| write worth zero / shuffle | +0.157 / +0.154 | +0.165 / +0.172 |
| tripwire | HEALTHY | DETONATED at 477 (1.6e7), recovered by 483 |
| tul/ditto_p_ratio, first 100 -> last 100 steps | | 1.016 -> 0.785 (0.73 at step 980) |
| tul/ditto, first 100 -> last 100 steps | | 0.223 -> 0.020 |

The spike is in the slot-loop core (core.0-2 carry 1e7 at steps 477-478; prelude 372) and
recovers within 6 steps; val kept falling (4.2188 at 500, 4.1480 at 750, 4.0254 final). The CE
cost sits in the body, not in a muted head: the head's worth fell only 0.015, the head-off CE
rose 0.079.

Generation (the coverage round's 24 prompts, seeds and settings):

| model | greedy seq_rep_4 | T 0.7 seq_rep_4 | T 0.7 rep_l | T 0.7 distinct_4 | T 0.7 gen-PPL |
| --- | --- | --- | --- | --- | --- |
| real text | | 0.020 | 0.346 | 0.959 | 96.41 |
| LXTUL, no head | 0.744 | 0.041 | 0.526 | 0.901 | 13.66 |
| plain | 0.873 | 0.215 | 0.626 | 0.752 | 6.47 |
| pointer phase, control | 0.909 | 0.291 | 0.668 | 0.690 | 11.92 |
| pointer phase, coverage | 0.925 | 0.374 | 0.696 | 0.607 | 9.89 |
| pointer phase, DITTO | 0.821 | 0.140 | 0.590 | 0.820 | 14.48 |

Samples (gen.md): greedy loops still start but now break after a few copies and drift ("The
2007 election." six times, then other sentences), the decay the paper describes. Sampled text is
varied. A new artifact appears: junk subword runs ("studymbok of the worldmbok", "2011111"),
probably mass pushed off a repeated token landing on junk; not measured.

Per prediction:

1. ditto_p_ratio over the last 100 steps at most 0.7: FALSIFIED (0.785; still falling, 0.73 at
   step 980).
2. T 0.7 seq_rep_4 at most 0.233: HELD (0.140, 52 % below the control).
3. T 0.7 seq_rep_4 at or below plain's 0.215: HELD (0.140).
4. Greedy seq_rep_4 at most 0.859: HELD (0.821; plain 0.873).
5. Trainer val within +0.02 of the control: FALSIFIED (+0.071).
6. Sweep K1-K6 within 0.004 of the control: FALSIFIED, upward (+0.0294 vs +0.0180).
7. No raise, no tripwire, 3 rows built every step: FALSIFIED on the tripwire (one recovered
   spike train at 477); 3 rows built on all 50 logged steps.

## Verdict

Mixed. DITTO is the first guard that works on the pointer's repetition: sampled repetition
halves and lands below plain, greedy looping drops below plain, and the head keeps its copying
worth (0.450 vs 0.465). It is not free at this budget: +0.071 nats of clean val, and the cost
is in the model body. Two design choices plausibly carry that cost, both fixable: the DITTO rows
REPLACE half the real rows (half the real text per step during the phase), and a pseudo row
feeds the slot loop dozens of identical spans (near-zero slot-state rank, the condition earlier
work tied to the slot loop's spike mode; one spike train appeared at 477). The ratio was still
falling at step 1000, so the phase was also short of the paper's 10k fine-tune.

## Updated hypothesis

DITTO is the copy channel's repetition guard. Next, in order: (a) add DITTO rows ON TOP of the
real batch instead of replacing real rows, so the phase sees the same real text as the
control; (b) cut the pseudo row's copies to a few (the paper's loss needs only copy n vs n-1),
which also bounds identical slot states; (c) a longer phase or DITTO mixed into the main run
at a low rate, then a seed twin. Check the junk-subword artifact with a counted metric.
