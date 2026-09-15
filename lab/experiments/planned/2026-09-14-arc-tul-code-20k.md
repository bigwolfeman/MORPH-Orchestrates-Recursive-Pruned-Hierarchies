# Experiment: TUL-Code at 20k — horizon, the frozen-E rollout arm, and the marginal

Status: planned

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

(Empty until the runs finish.)
