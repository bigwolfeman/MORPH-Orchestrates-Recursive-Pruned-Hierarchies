# Experiment: TUL-Code at 20k — horizon, the frozen-E rollout arm, and the marginal

Status: failure

Committed before launch. Predictions frozen. Follows
[`failures/2026-09-14-arc-tul-code.md`](../failures/2026-09-14-arc-tul-code.md) (the 5k
panel: the encoder code is a verbatim copy of the span, the sample is close to an
unconditional draw, one draw costs 0.7 nats against the strict ruler, and 5k cannot
separate "the target is unpredictable" from "the conditioning is not learned yet").
Wolfe 2026-09-14: "much better than it looks, it is just undertrained"; approved the
three arms below.

## Question

At 20k, with the K-sample marginal as the likelihood, does the sampled code stop costing
against the deterministic slot loop, and does training the coda on guesses at every slot
(E frozen from the phase-3 switch) close the topic gap?

## Hypothesis

H1 (horizon): the thinker's conditioning is still arriving at 5k (band-0 flow ratio 0.49
vs a 0.64 blind floor, still falling), so the one-draw gap to the ruler narrows by 20k.
H2 (distribution): the 5k coda saw a perfect code on half its training slots and never
had to hedge; with `code_rollout_p: 1.0` it trains on the eval distribution, leans on the
past codes for the topic, and lands near the ruler on one-draw CE.
H3 (metric): one-draw CE undercounts a latent-variable LM; the K = 8 marginal sits well
below the one-draw average, and is the number to read against the ruler.

## Method

Arms, all at seq 1024, batch 6, seed 1, 20,000 steps with the queue overrides
`training.steps=20000 training.ademamix_t_beta3=20000 training.ckpt_every=5000`, sweeps
at 5000/10000/15000/20000, KIND slot, wandb `morph-tul`, at the commit carrying this file:

| arm | config | role |
|---|---|---|
| `tul-code-20k` | `tul_code.yaml` | the design, phases at 2000 / 10000 |
| `slot-spandec-strict-20k` | `tul_slot_spandec_strict.yaml` | the ruler twin at 20k |
| `tul-code-rollout1-20k` | `tul_code_rollout1.yaml` | phase 3 with a sample on EVERY slot (E frozen from 10000) |

Instruments: the runner's K-curve sweep and worth profile per checkpoint; `val/loss`
(one draw, k = 8), `val/ce_tf`, `val/code_gap`, `val/ce_marginal` and `val/ce_single_mean`
(K = 8, new at 1ddb34c), `train/code_fm_rel` and the four t bands; paired gaps against the
ruler twin on the shared tokens (`paired_vs_strict_ruler_20000.json`, the 5k procedure);
`code_subspace_probe` and `code_span_samples` (ten cuts, seeds 1 and 2, the same cuts as
the 5k panel) on the 20k checkpoints of both code arms and the ruler twin.

Predictions are read on the 20k checkpoints; the 5k/10k/15k readings are the trajectory.

## Predictions (frozen)

Paired numbers are on the shared tokens against `slot-spandec-strict-20k` at 20k.

- P-A (H1). `tul-code-20k` one-draw `ce_k1` − ruler narrows from +0.70 (5k) to ≤ +0.35
  at 20k, CI excluding +0.35. 50 %.
- P-B (H1). The band-0 flow ratio of `tul-code-20k` at 20k ≤ 0.40 (5k: 0.49). 60 %.
- P-C (H2). `tul-code-rollout1-20k` one-draw `ce_k1` is ≥ 0.20 below `tul-code-20k`'s at
  20k, paired, CI excluding 0. 65 %.
- P-D (H2). `tul-code-rollout1-20k` one-draw `ce_k1` − ruler within ±0.15 at 20k. 50 %.
- P-E (H2). On the ten cuts, `tul-code-rollout1-20k` keeps a topic thread (a named
  entity or subject from the context) in ≥ 3 of 10 sampled continuations, judged by hand
  and recorded per cut; `tul-code-20k` in ≥ 1 of 10 (5k: 0 of 10). 55 %.
- P-F (H3). For `tul-code-20k` at 20k, `val/ce_single_mean − val/ce_marginal` ≥ 0.30
  nats (the draws differ materially). 60 %.
- P-G (H3). `tul-code-rollout1-20k`'s `val/ce_marginal` at 20k is BELOW the ruler twin's
  `val/loss` on the same eval stream. 40 %.
- P-H. All three arms healthy to 20k (no tripwire, no DIV-GUARD), code arms ≥ 2.5× the
  ruler's tok/s. 85 %.

## Binding

- P-C and P-D fail together → the coda's training distribution was not the lever; the
  target is. Then the next arm is a code defined by what the past determines, not a copy
  of the span (a spec change first).
- P-A fails and P-B holds → conditioning improves but the draw still costs: the metric
  question (P-F, P-G) decides whether the design is judged on the marginal or dropped.
- P-G holds → TUL-Code is a better LM than the slot loop in the only sense that applies
  to a latent-variable model; the follow-on is the 0.9-mix / low-LR-E scheduled-sampling
  arm Wolfe named, and generation quality with the diversity guard.

## Not verified before launch

- The marginal instrument ran on the CPU contract (K = 1 equals the seed-0 one-draw CE,
  Jensen holds, seeds differ) and on a 12-step Spark smoke; never on a trained checkpoint.
- The 20k horizon on any code arm; memory and rate come from the 5k runs (13.1 GB peak,
  29–33k tok/s).
- One seed per arm. Nothing here is a verdict across seeds.

## Results

All three arms ran to 20k healthy (no tripwire, no DIV-GUARD; largest pre-clip gradient
172 at tul-code-20k's phase-3 switch, 58.7 on rollout1, 27.5 on the ruler). Artifacts:
`lab/experiments/results/2026-09-14-arc-tul-code-20k/` (Spark k-sweeps at every
checkpoint, subspace probes and span samples at 20k, the val series, the paired scoring
`paired_vs_strict_ruler_*.json`, the P-E judgment, the norm-flag probe).

Paired on 501,106 identical tokens against `slot-spandec-strict-20k` at 20k (one draw,
k = 1 unless said):

| arm | one draw − ruler | encoder code − ruler | k16 − k1 (8-draw marginal) |
| --- | --- | --- | --- |
| tul-code-20k (p 0.5) | +0.626 [+0.618, +0.634] | −2.550 (bare z) | +0.014 [+0.012, +0.016] |
| tul-code-rollout1-20k (p 1.0) | +0.359 [+0.353, +0.364] | +0.291 | +0.005 [+0.003, +0.007] |
| rollout1 − tul-code-20k | −0.268 [−0.272, −0.263] | | |

Through phase 2 the k-curve is steep on both seeds (tul-code-20k −1.25 at 5k, −1.29 at
10k; rollout1 −0.98 at 5k, −0.42 at 10k) and flat within 5000 rollout steps on both.
The thinker's flow loss reaches its floor by step 4000 (band ratios 0.48 / 0.33 / 0.23 /
0.21 at 20k, within 0.05 of their 4k values). The sample residual over the code's
variance at 20k: 1.65–1.92 (tul-code-20k), 1.49–1.64 (rollout1, E frozen from 10k); the
sample is worth the top 32 of 1024 code directions (head-32 ce_tf 4.35 vs sampled 4.38).
Under p 1.0 the coda drops the cells within 250 steps of the switch (encoder code =
sample = 4.92 at 10250) and the cell-blind coda then beats the half-trusting one by 0.27
nats. `val/ce_single_mean − val/ce_marginal` at 20k: 0.005–0.015 on tul-code-20k.
Span samples: tul-code-20k 0 / 10 keep a topic thread, rollout1 1 / 10, the ruler 4 / 10.
Rates: 30k tok/s in phase 2 (2.6× the ruler's 11.7k), 24k in phase 3 (2.05×).

A defect found by the panel and fixed in the same day (commit bed41d7): a truth cell
reached the phase-3 coda at RMS sqrt(1 + noise²) and a sampled cell at RMS 1, eval fed
bare z, and the coda learned the norm as the sample flag (tul-code-20k at 20k: bare z
1.25 nats, the trained statistic 0.35, a sample at the truth norm +0.01). The
`encoder code − ruler` column above for tul-code-20k is the bare-z reading; every other
number is unaffected (samples are at RMS 1 in training and eval alike).

Predictions: P-A FAIL (+0.626, needed ≤ +0.35). P-B FAIL (band-0 0.48, needed ≤ 0.40).
P-C HOLD (−0.268, CI excludes 0). P-D FAIL (+0.359, needed within ±0.15). P-E FAIL (1 / 10
and 0 / 10). P-F FAIL (0.005–0.015, needed ≥ 0.30). P-G FAIL. P-H: health holds, the
phase-3 rate clause fails (2.05× < 2.5×).

## Verdict

Failure: one prediction of eight held (P-C), and it held for the wrong reason — the
rollout coda wins by ignoring its cells, not by learning how much of a guess to trust.
The binding case "P-C and P-D fail together" did not fire literally, but its conclusion
does: the coda's training distribution is not the lever. A code model whose sample is an
unconditional draw is worse than no code at all (+0.27 nats), the ruler's deterministic z
is worth 0.36 nats over no code at this horizon, and the thinker stopped learning at
step 4000. H1 (horizon) is refuted for the thinker and confirmed for the coda side only.

## Updated hypothesis

The sampled code carries nothing span-specific for two independent reasons: the target
is a lossless copy of a sentence the past does not determine (so p(z | past) is broad and
its mean is content-free), and the sampler does not sharpen toward the conditioning it
does have (band 0 at 0.48 against a 0.64 blind floor). Next: the two arms of
`.agents/notes/proposed/architecture/2026-09-15-tul-code-conditioned-thinker.md` —
classifier-free guidance on the thinker and predictability pressure on E — preregistered
in `planned/2026-09-15-tul-code-conditioned-thinker.md`; the bar is "beats the cell-blind
coda", i.e. a one-draw gap under +0.36 against the ruler. The renorm resume
(`planned/2026-09-15-tul-code-renorm-r10k.md`) reads whether the norm flag also shaped
the p 0.5 coda's discount.
