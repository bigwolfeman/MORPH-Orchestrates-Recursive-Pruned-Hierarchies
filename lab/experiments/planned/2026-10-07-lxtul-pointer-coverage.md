# LXTUL + pointer: the See et al. coverage phase against repetition

Status: planned (written 2026-10-07 00:53 CDT, before any run; tests/test_tul_pointer.py 14
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

## Verdict

## Updated hypothesis
