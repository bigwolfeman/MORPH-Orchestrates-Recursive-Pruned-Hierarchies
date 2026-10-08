# Parcae pointer: does the copy head make generation repeat?

Status: failure (written 2026-10-06 23:39 CDT, before any decode of the 32 prompts; a
4-prompt smoke is running on the 3070 to test the code path)

## Question

The pointer head scores well on CE by copying the token that followed an earlier match. In free
generation the head reads the model's OWN output as context, so a copy can feed the next copy
and the text can loop or paste long runs of the prompt. CE cannot see this (See et al. 2017 added
a coverage loss for it). Wolfe (2026-10-06): "We need generation samples. This method of scoring
bigrams may cause repetition."

## Method

`lxtul/generate.py` on the 3070 (8 GB, eval only): 32 prompts of 128 tokens from MORPH's
validation rows, 128 new tokens, depth 6, greedy and nucleus p = 0.95 at T = 1, seed 1234.
Recompute per step; an LXTUL arm re-packs the sequence every step and reads the next token at
the last real token; a pointer arm reads the full mixture (`PointerHead.mixed_at`, tested equal
to the training rule at every vocabulary entry). Arms: plain-5k, plain-pointer-norm-5k,
lxtul-5k, lxtul-pointer-norm-5k, lxtul-pointer-cellkey-5k. REAL = the true continuations.
Metrics (module docstring): seq_rep_4, rep_l, ctx_copy_4 (continuation 4-grams already in the
context), max_copy (longest verbatim run from the context), distinct_4, gen_ppl under plain-5k.

Amendment (2026-10-06 23:43 CDT, before the 32-prompt run): the predictions below were written before
any decode. The 4-prompt smoke (two pointer arms only) then finished before this file was
committed, so its numbers were seen before the commit; the predictions were not edited.

## Predictions

"Twin" = the same backbone without the head (plain + pointer vs plain; LXTUL + pointer vs LXTUL).

1. Greedy: every arm loops (seq_rep_4 above 0.3), as every MORPH 5k arm does (80 %).
2. Greedy: each pointer arm's seq_rep_4 at least 0.05 above its twin's (55 %).
3. Sampled: each pointer arm's seq_rep_4 within 0.01 of its twin's (60 %).
4. Sampled: each pointer arm's ctx_copy_4 above its twin's (75 %).
5. Sampled: each pointer arm's ctx_copy_4 at most 1.5x REAL's (55 %).
6. Sampled: each pointer arm's mean max_copy at most 2 tokens above its twin's (60 %).

Decision rule, set now: if a sampled pointer arm's seq_rep_4 sits more than 0.02 above its
twin's, or its ctx_copy_4 above 2x REAL's, the head needs a repetition guard (coverage term, or a
gate penalty on copying the model's own recent output) before the MORPH port ships.

## Results

Run 2026-10-06 23:41 CDT on the 3070, finished before 00:07, at 3aabdc9 (`results/2026-10-06-parcae-pointer-generation/`:
gen.json, and gen.md with six prompts' continuations from every arm). Mean over 32 prompts,
bootstrap 95 % CI.

Sampled, nucleus p = 0.95, T = 1:

| arm | seq_rep_4 | ctx_copy_4 | max_copy | distinct_4 | gen_ppl |
| --- | --- | --- | --- | --- | --- |
| REAL | 0.0258 [0.0088, 0.0525] | 0.0582 [0.0210, 0.1233] | 7.0 | 0.9665 | 52.4 |
| plain | 0.0070 | 0.0125 | 3.9 | 0.9895 | 44.9 |
| plain + pointer | 0.0105 | 0.0220 | 4.5 | 0.9845 | 63.4 |
| strict LXTUL | 0.0013 | 0.0015 | 2.8 | 0.9930 | 79.6 |
| strict LXTUL + pointer | 0.0105 | 0.0260 | 5.1 | 0.9858 | 73.2 |
| strict LXTUL + pointer + cell keys | 0.0068 | 0.0192 | 4.5 | 0.9915 | 69.7 |

Greedy:

| arm | seq_rep_4 | ctx_copy_4 | max_copy |
| --- | --- | --- | --- |
| plain | 0.7845 | 0.8025 | 85.3 |
| plain + pointer | 0.8013 | 0.8540 | 97.1 |
| strict LXTUL | 0.4665 | 0.4700 | 36.1 |
| strict LXTUL + pointer | 0.7948 | 0.8630 | 98.3 |
| strict LXTUL + pointer + cell keys | 0.7715 | 0.8458 | 93.0 |

| # | prediction | result |
| --- | --- | --- |
| 1 | greedy: every arm seq_rep_4 above 0.3 | holds (lowest 0.4665) |
| 2 | greedy: each pointer arm at least 0.05 above its twin | fails for plain (+0.017); holds for both LXTUL arms (+0.33, +0.31) |
| 3 | sampled: each pointer arm's seq_rep_4 within 0.01 of its twin | holds (+0.0035, +0.0092, +0.0055) |
| 4 | sampled: each pointer arm's ctx_copy_4 above its twin | holds |
| 5 | sampled: each pointer arm's ctx_copy_4 at most 1.5x REAL's | holds (at most 0.026 vs 0.087) |
| 6 | sampled: max_copy at most 2 above the twin | fails for strict + pointer (+2.3); holds for the other two |

Decision rule: no sampled pointer arm sits more than 0.02 above its twin on seq_rep_4 (largest
+0.0092) or copies 4-grams above 2x REAL (largest 0.026 vs 0.116). No repetition guard is needed
by the rule set before the run.

gen_ppl is scored by plain-5k, which favours its own samples (44.9, below REAL's 52.4), so the
column ranks arms by closeness to plain, not by quality.

## Verdict

Failure on 2 and 6 as written, but not in the direction the question feared. Under sampling, the
head makes text copy MORE from its context, and every arm still copies less than real text:
strict LXTUL without the head copies 4-grams 40x less often than real text (0.0015 vs 0.058),
and the head moves it to 0.019-0.026. Repetition within the continuation stays below real text
in every sampled arm. Under greedy decoding the head makes strict LXTUL loop like plain (0.79 vs
plain 0.78); strict LXTUL without the head loops less only because it cannot copy at all. The
head adds 0.017 to plain's greedy loop rate.

## Updated hypothesis

The pointer does not create a repetition problem beyond plain's own greedy loops at 5k. Strict
LXTUL without a copy channel under-copies (a symptom of the same gap the CE probe found). One
thing to watch at longer training and on MORPH: strict + pointer's longest verbatim run (5.1
tokens sampled) is above its twin's and plain + pointer's; real text reaches 7.0.
