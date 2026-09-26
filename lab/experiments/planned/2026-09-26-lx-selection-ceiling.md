# Planned: how much could any selector over the LX rollouts buy (credit concentration and the oracle ceiling)

Status: planned

Date: 2026-09-26 15:06 (frozen before the script existed and before any number of this probe was
computed. The unit tests run on the tiny CPU model and on synthetic arrays only.)
Requested by: external council reviewers (space-bunny, mimo-flash), 2026-09-26, after
[`../failures/2026-09-26-lx-carry-stage0.md`](../failures/2026-09-26-lx-carry-stage0.md).
Parent readings: [`../successes/2026-09-25-lxtul-fp01-10k.md`](../successes/2026-09-25-lxtul-fp01-10k.md)
(the stored arrays this re-scores).

## Question

LXTUL-E runs K = 4 slot-loop rollouts per row, each with a fixed code, and the coda decodes
each one. Training minimises the exact per-span mixture
`-log (1/4) sum_k exp S_k(s)`, `S_k(s)` the sum of rollout k's log-probs over span s's
scored tokens; eval reads the per-token Bayes mixture that restarts at uniform in every
span. Every "selector" design (a value head, a gate, a learned prior over the codes) tries
to put the weight on the right rollout before the evidence arrives. Two numbers decide
whether any of them can pay:

1. How concentrated is the per-span credit `c_k(s) = softmax_k S_k(s)`? If the four
   rollouts explain every span about equally well, width is ensemble averaging and there is
   nothing to select.
2. What is the ceiling? An ORACLE that knows, for each span, which rollout has the largest
   `S_k(s)` and uses it for every token of that span is the best any per-span selector can
   do. Its gap to today's Bayes read is the most a learned selector could buy.

## Method

- Instrument: `lab/divergence/lx_selection_ceiling.py` (to be written; tests in
  `tests/test_lx_probes.py`). It reuses `lx_carry_stage0.rescore` (row rebuild, stream-index
  join, today's read recomputed, EOS handling, offsets from `_earning.py`) and the Stage 1
  scorer's paired block bootstrap (`_ci`, stream blocks of 1,024, 2,000 resamples, seed 0).
- Data: `/home/wolfe/morph-scratch/lxtul-10k/score.tokens.npz`, labels `fp01_5k` and
  `fp01_10k`, forced depths 1 and 6, all 480 packed validation rows (501,106 scored coda
  tokens per checkpoint). CPU only, no forward pass, no fitting, so no split.
- Span: a run of one `bag_id` on a packed row that holds at least one scored token (the
  unit of the training mixture and of the eval restart). The first span of a row and the
  tail dump bin are spans too; their tokens sit in the offset bin "first_span_or_dump".
- Credit readings, per checkpoint and depth, one value per span: `KL(c || uniform) =
  log 4 - H(c)` in nats, `H(c)`, the fraction of spans whose argmax rollout has credit
  > 0.5 and > 0.9, the argmax rollout's share of spans; per checkpoint, the fraction of
  spans whose argmax at depth 6 equals the argmax at depth 1 (the same code u_k at both
  depths), beside the chance level `sum_k p1(k) p6(k)` from the two marginals. CIs: the same
  block bootstrap with one unit per span (block of its first scored token).
- Per-token CE of four reads, per checkpoint and depth:
  - Bayes: today's uniform-restart per-token read (recomputed, checked against the stored
    `coda_mix`);
  - span mixture: `-log (1/4) sum_k exp S_k(s)` per span, divided by the token count. By
    the chain rule of a mixture, a span's Bayes-read tokens SUM to this value exactly, so
    overall the two are one number; the script ASSERTS it per span. The span mixture has no
    other per-token split, so it is reported overall only;
  - oracle best-of-4: per span, `-lp_{k*}` on every token, `k* = argmax_k S_k(s)`;
  - best single fixed rollout: the one k with the lowest total NLL over all spans (chosen
    on the same rows; 4 candidates, so the in-sample optimism is small).
  Differences with paired CIs: Bayes - oracle (the selection ceiling), best-fixed - Bayes,
  best-fixed - oracle; overall, at offset 0, and in every `_earning.BINS` bin.
- Identities checked (each raises): oracle <= span mixture <= oracle + log 4 per span;
  span mixture - oracle = `log(4 c_max(s))` per span; oracle <= best fixed overall.
- Stated before the run: the ceiling per TOKEN is bounded by `log 4 x (spans / tokens)`,
  because a span's gap is at most log 4 = 1.386 nats and a span holds about 20 scored
  tokens. The script prints this bound beside the measured gap.

- **Amended 2026-09-26 15:10, after the first run (reason stated, Predictions untouched).** A
  lookahead oracle beats the span mixture even when the four rollouts are exchangeable
  token noise, since the max of four noisy span sums lies above their log-mean-exp. So
  "Bayes - oracle" has a floor that does not come from distinct hypotheses. A post-hoc
  diagnostic now reports it: the token-shuffle null permutes the rollout axis independently
  at every token (5 draws, seed 0), keeping each token's four values and removing any
  rollout advantage that persists across a span. It reports the null's mean KL and overall
  Bayes - oracle, and observed minus null. It is a diagnostic only. R-1 and the reading
  rules stay as frozen.

## Predictions

From the external council reviewers (space-bunny, mimo-flash), 2026-09-26:

- **R-1 (credit).** Mean `KL(c || uniform)` per span lies in **[0.05, 0.20] nats**: the
  rollouts broadly agree, and width is ensemble variance reduction. Graded at depth 6 on
  both checkpoints (holds only if both points lie in the range); depth 1 is reported beside
  it.
- **R-2 (ceiling).** No numeric prediction was given for the oracle gap. The reviewer's
  stated claim is the reading rule below: the oracle-to-Bayes gap is the ceiling any
  learned selector (value head, gate, prior) could buy.

## Reading rules

- Bayes - oracle < 0.02 nats per token (depth 6, both checkpoints) => the rollouts are
  interchangeable for a per-span selector, and selector / prior designs over the fixed codes
  are closed.
- Mean KL < 0.2 nats => width is ensemble averaging, not a choice among distinct
  hypotheses.
- A ceiling at or above 0.02 does not open the selector lane by itself: the oracle uses the
  span's own tokens (lookahead), so a causal selector gets only part of it. The offset-0 bin
  shows how much of the ceiling sits where a causal read has no evidence yet.
- Argmax agreement across depths near chance says which rollout wins is not a property of
  the code at a given span; well above chance says it is.
