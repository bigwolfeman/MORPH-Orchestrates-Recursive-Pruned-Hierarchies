# LXTUL + pointer: the See et al. coverage phase against repetition

Status: failure (filed 2026-10-07 05:08 CDT; written 2026-10-07 00:53 CDT, before any run; tests/test_tul_pointer.py 14
passed, 3 coverage sabotages caught)

## Question

A copy head can feed on its own output in free generation and loop; CE cannot see it. Wolfe
(2026-10-07): "even if it were a good test the foot gun of over repetition will still bite later.
You mentioned a solution from one of the original papers. I think we should just do it." The
Parcae generation test could not answer it: sampled at T = 1 on 5k checkpoints, every arm,
real text included, repeats too little to rank (parcae docs/experiments/failures/
2026-10-06-parcae-pointer-generation.md). Does See et al.'s coverage (arXiv 1704.04368 §2.3)
cut the pointer model's repetition, and at what CE cost?

## Method

Coverage (`tul.pointer_coverage_lambda`, morph/model/tul_pointer.py): per head, the coverage of
key j at query i is the attention earlier queries paid it; it enters the score through a learned
per-head scalar (zero-init) and the loss adds lambda * mean_h sum_j min(alpha, coverage) per
token. The paper's recipe: lambda 1, as a SEPARATE final phase. One forced deviation: the
paper's coverage is recursive, ours reads the coverage-free attention for the score term (the
loss reads the final attention); train and generation compute the same function.

Two 1000-step phases from the LXTUL + pointer 5k checkpoint (`training.init_from`, weights only,
fresh optimizer, 200-step ramp):

| run | config |
| --- | --- |
| pointer phase, no coverage (control) | `lxtul_pointer_ft` |
| pointer phase, coverage lambda 1 | `lxtul_pointer_cov` |

Readouts: the runner's 480-row sweep (CE@6, K1-K6, gap to plain), the head-off check, and a
repetition eval built for this question: `gen_diversity_eval.py --decodes
greedy,sample_t07_k40` (T = 0.7, top-k 40; the T = 1 mode cannot show repetition at 5k),
24 prompts, 256 new tokens, on both phases, the 5k pointer checkpoint, LXTUL without the head
(the 5k lead checkpoint) and the plain 5k model.

Method amendment, 2026-10-07 02:18 CDT, after the control phase trained and before the coverage
phase trained. `training.init_from` resets the step AND the train stream to doc 0, so each phase
re-trains the first docs of OpenWebText, which are the 480 rows that `core_depth_sweep.py` and the
gap readout score (skip 0). The control's sweep gap reads -0.209 against -0.150 at 5k while the
trainer's val (doc 50,000+, clean) got 0.035 WORSE (3.9202 -> 3.9548): the sweep rows are
contaminated for both phases. Readings change as follows. Prediction 2 is read on the trainer's
final val loss (clean) and reported beside the sweep CE@6 (contaminated, but matched between the
two phases). No sweep number from a phase is compared with a 5k checkpoint's. The generation eval
reads its prompts at the trainer's val offset (skip 50,000), so predictions 3-5 are unaffected.
Both phases run as configured; adding `training.data_skip_batches` now would unpair them.

Method amendment, 2026-10-07 03:09 CDT, before any generation. The first generation eval ran out of GPU memory
on the first LXTUL model (24 rows x 384 tokens in one batch; the eager generator also runs the
training-only span decoder). It reruns with `--gen_batch 8`: the same 24 prompts, seeds and
decodes, generated 8 rows at a time. Each row has its own seeded sampler, so the change touches
outputs only through batch-size GPU numerics.

## Predictions

1. The coverage loss falls during the phase: `tul/pointer_cov` over the last 100 steps at most
   0.6x its mean over steps 1-100 (75 %). (The paper: 0.5 -> 0.2.)
2. CE cost: coverage CE@6 within +0.02 of the control's (70 %).
3. T = 0.7: coverage seq_rep_4 at least 20 % lower (relative) than the control's (55 %).
4. Greedy: coverage seq_rep_4 at least 0.05 below the control's (50 %).
5. The foot gun exists: at T = 0.7 the control's seq_rep_4 is above LXTUL-without-the-head's
   (65 %).
6. No run raises or trips the detonation tripwire.

## Results

Artifacts: `lab/experiments/results/2026-10-07-lxtul-pointer-coverage/{control,coverage,gen}/`
(sweeps, head-off sweeps, write worth, train curves, the coverage loss from wandb run
`h7gjd1or`, the generation json, examples and logs, and the OOM log of the first eval attempt).

Training and CE (one seed per phase; "sweep" rows are re-trained by both phases, see the
02:18 amendment, so only phase-vs-phase differences are read there):

| reading | control | coverage |
| --- | --- | --- |
| trainer val, final (clean) | 3.9548 | 3.9725 (+0.018) |
| sweep CE@6 | 3.8671 | 3.8859 (+0.019) |
| sweep K1-K6 | +0.0180 | +0.0221 |
| head off CE@6 / head worth | 4.3316 / 0.465 | 4.3249 / 0.439 |
| head off K1-K6 | +0.0252 | +0.0312 |
| write worth zero / shuffle | +0.157 / +0.154 | +0.137 / +0.155 |
| tripwire | HEALTHY, max 8.41 | HEALTHY, max 9.21 |
| coverage loss, mean of steps 1-100 -> last 100 | | 0.304 -> 0.030 |

The coverage loss starts at 0.52 (the paper's 0.5) and is at 0.03 by step 250. The learned
per-head coverage gains end at -0.001, -0.0003, -0.087, -0.014: the score term is barely used;
the loss did the work by moving attention.

Generation (24 prompts at the trainer's val offset, 128-token prompt, 256 new tokens; seq_rep_4
/ rep_l / distinct_4 / gen-PPL under the plain 5k scorer). Real text: seq_rep_4 0.020, rep_l 0.346.

| model | greedy seq_rep_4 | T 0.7 k 40 seq_rep_4 | T 0.7 rep_l | T 0.7 distinct_4 | T 0.7 gen-PPL |
| --- | --- | --- | --- | --- | --- |
| pointer 5k | 0.891 | 0.307 | 0.672 | 0.671 | 12.11 |
| pointer phase, control | 0.909 | 0.291 | 0.668 | 0.690 | 11.92 |
| pointer phase, coverage | 0.925 | 0.374 | 0.696 | 0.607 | 9.89 |
| LXTUL, no head | 0.744 | 0.041 | 0.526 | 0.901 | 13.66 |
| plain | 0.873 | 0.215 | 0.626 | 0.752 | 6.47 |

The pointer's loops (gen.md): greedy repeats whole sentences ("The 2012 election, which was also
the first of its most recent years ...") and short phrases ("to be done to be done"); sampled,
it stutters ("public's public's public's") and re-copies one phrase ("fair bet", "theatrical
utilities") over and over.

Per prediction:

1. Coverage loss at most 0.6x: HELD (0.10x).
2. CE cost within +0.02: HELD on both readings (+0.018 val, +0.019 sweep), with no margin.
3. T 0.7 seq_rep_4 at least 20 % lower: FALSIFIED. It is 29 % HIGHER (0.374 vs 0.291).
4. Greedy seq_rep_4 at least 0.05 lower: FALSIFIED (0.925 vs 0.909).
5. The foot gun exists (control above LXTUL without the head at T 0.7): HELD (0.291 vs 0.041).
   Context the prediction did not name: plain repeats at 0.215, so the pointer lifts LXTUL from
   far below plain to 1.35x plain; greedy looping is at plain's level for every 5k model.
6. No raise, no tripwire: HELD for both training runs. The first generation eval ran out of
   memory (24 rows x 384 tokens in one eager TUL batch) and was rerun with `--gen_batch 8`
   (amendment 03:09).

## Verdict

Failure. Coverage did its training job and made repetition worse. The mechanism is a category
error for a language model. See et al. summarize a FIXED source; repetition there means
attending the same source positions again, which coverage counts. In LM self-copy the source is
the text so far and grows with the output: a loop "A B C A B C" copies cycle n from cycle n-1,
whose positions no query has attended yet, so their coverage is about zero and the loop is
invisible to the term. On real text the loss is cheap to meet (0.52 -> 0.03 in 250 steps). The
way the heads met it, attending positions with no coverage, favours the newest positions,
which in free generation are the model's own output: the loss trained the head toward the
pattern behind the loops.

## Updated hypothesis

The pointer's repetition is real under sampling (0.29 against plain's 0.21 and real text's
0.02) and is the self-reinforcement DITTO describes (Xu et al. 2022, arXiv 2206.02369): once a
phrase repeats, each copy raises the next. A repetition fix for a language model must see the
repeat itself, not attention positions. Candidates, both from the LM literature: DITTO
(pseudo-repeated sentences, loss -log(1 - |p_n - lambda p*_{n-1}|), lambda 0.5, no model
rollouts; Wikitext-103 greedy Rep-4 44 % -> 22 % with PPL 25.7 -> 24.3) and sequence-level
unlikelihood (Welleck et al. 2019, arXiv 1908.04319; greedy seq-rep-4 0.442 -> 0.058 but it
needs greedy rollouts during training, which the eager generator makes impractical at 36 s per
row). Coverage stays in the tree, default off, as a measured negative.
