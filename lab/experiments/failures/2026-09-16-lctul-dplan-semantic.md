# Experiment: LCTUL-D plan arm — a coarse discrete code, scored on generation

Status: failure

Date: 2026-09-16 (frozen before any GPU step of the arm). Follows
`2026-09-16-lctul-d-first-arm.md` (the 72-bit copy: context share 0.02 % at 5k, CE k-curve
inverted +0.16 from k = 1 to 8) and the theorems in `lab/theory/lctul_euler_depth/`.
Wolfe, 2026-09-16: "kill the current run, let's get this up and running", with a metric and
measured controls first.

## Question

With a coarse code (four symbols of 6 bits per span) and the verdict read on GENERATION,
does the model's own sampled plan make the written span semantically closer to the true
span than a foreign plan, does more sampler rounds help, and does it beat the deterministic
slot write of the strict ruler and the plain model on the same cuts?

## Hypothesis

On likelihood a sampled latent of the next span cannot beat a deterministic summary of the
past, so CE is the wrong verdict. On generation, a coarse plan is a few separated modes,
where the theorems let the sampler's rounds act, and committing to one plan can make the
greedy continuation more on-topic than a mean write.

## Metric

`lab/divergence/code_semantic_probe.py` on the SAME 120 validation cuts for every model
(cuts from the `tul_code_d` boundary rule and packer; `--cuts_config` for the plain model):
greedy continuation of the open span, MiniLM cosine to the true span (`cos_true`),
content-word overlap, paired bootstrap over cuts, and distinct-2 per condition as the
diversity guard. Conditions on the arm: OWN@k for k in 1, 2, 4, SHUF (a foreign sample at
k 4), ZERO, ORACLE. Controls (OWN only): the strict ruler `slot-spandec-strict-20k`
step 20000 (kind slot), the plain `norm-match-20k` step 20000 (kind plain), and the flow arm
`tul-code-cfg` step 20000 (kind code, k 1, 2, 4, 8). Cross-model differences are paired
over cuts from the saved `per_cut` arrays.

## Predictions (frozen)

Arm `tul-code-dplan` (`tul_code_dplan.yaml`, 20k steps, batch 6, the panel extras), one seed.

- P-P1 (own plan beats a foreign plan): at 20k, `OWN@4 − SHUF` on cos_true > 0 with the
  95 % interval clear of 0. (Flow arm at 10k: +0.026 [+0.007, +0.045]; thinker-only: 0.)
- P-P2 (rounds pay): `OWN@4 − OWN@1` on cos_true > 0 with the interval clear of 0.
- P-P3 (the bar): `OWN@4` on cos_true within 0.02 of the strict ruler's OWN on the same cuts,
  paired. (A win over the ruler is not predicted.)
- P-P4 (diversity guard): distinct-2 of OWN@4 within 0.05 of the ruler's.
- P-P5 (codes used): `tul/vq_perplexity` >= 16 of 64 at 5k.
- P-P6 (context share): the denoiser ELBO with the past minus without >= 5 % of the with-past
  number at 20k (the 72-bit arm: 0.02 % at 5k).
- P-P7 (CE for the record only): `val/ce_tf` at 20k above the 72-bit arm's (a coarser code
  carries less); no verdict on CE.

## Method

Queue line at the head of `recon_arms.txt` (commit recorded at insertion), the running
`tul-code-d` killed at Wolfe's instruction. Spark: the three controls run once with the
extended probe (kind slot / plain / code); the arm at 10k and 20k with `--k_list 1,2,4
--steps 4`. Pairing across models by a small script over the `per_cut` arrays.

## Results

Arm `tul-code-dplan` ran 2026-09-16 21:54 to 2026-09-17 00:25 (wandb `xkinv18f`, commit
ef27c3c, 20k steps, no spikes). Artifacts: `lab/experiments/results/2026-09-16-lctul-dplan/`
(semantic probe JSON/TXT at 10k and 20k, the three controls under `controls/`, the pairing
outputs `semantic_pairs_{10000,20000}.json` and `controls/semantic_pairs_controls.json`, the
marginal sweeps at 5k/10k/15k/20k, the 20k context-share pair). Pairing script:
`lab/divergence/code_semantic_pair.py` (1095dea).

Semantic probe at 20k, MiniLM cosine of the greedy continuation to the true span, 120 cuts:

| Condition | cos(true) | distinct-2 |
| --- | --- | --- |
| dplan OWN@1 | 0.148 | 0.380 |
| dplan OWN@2 | 0.158 | 0.375 |
| dplan OWN@4 | 0.158 | 0.371 |
| dplan SHUF (foreign sample) | 0.149 | 0.356 |
| dplan ZERO (no plan) | 0.161 | 0.393 |
| dplan ORACLE (true plan) | 0.189 | 0.399 |
| flow arm `tul-code-cfg` OWN@8 | 0.159 | 0.309 |
| strict ruler OWN | 0.187 | 0.437 |
| plain OWN | 0.230 | 0.490 |

| Prediction | Reading at 20k | Holds |
| --- | --- | --- |
| P-P1 OWN@4 − SHUF > 0, CI clear | +0.008 [−0.006, +0.023] | no |
| P-P2 OWN@4 − OWN@1 > 0, CI clear | +0.009 [−0.008, +0.028] | no |
| P-P3 OWN@4 within 0.02 of the ruler, paired | −0.030 [−0.054, −0.008] | no |
| P-P4 distinct-2 within 0.05 of the ruler | 0.371 against 0.437 (−0.066) | no |
| P-P5 perplexity ≥ 16 of 64 at 5k | 29.7 (35 codes used); 53.4 at 15k | yes |
| P-P6 context share ≥ 5 % | ELBO rel 0.878 with the past, 0.906 without: 3.1 % | no |
| P-P7 `val/ce_tf` at 20k above the 72-bit arm's | 3.89 at 19750; the 72-bit arm read 3.96 at 5k against this arm's 4.29 at 5k | yes |

Other readings, for the record:

- The oracle bound is low. The TRUE 24-bit plan lifts cosine by +0.040 [+0.018, +0.062]
  over a foreign one and lands exactly on the ruler (+0.002 [−0.027, +0.030]). A perfect
  thinker on this code could at most match the ruler; the flow arm's 1024-d oracle read
  0.691 (+0.543 over foreign).
- The empty plan is as good as the own plan: OWN@4 − ZERO −0.003 [−0.019, +0.013].
- The CE k-curve turned the right way once phase 3 settled: 8-draw marginal k4 − k1 is
  −0.0136 [−0.0164, −0.0108] at 15k and −0.0119 [−0.0155, −0.0082] at 20k (inverted at 5k
  and 10k). Encoder 3.764, k1 4.003, k4 3.991 at 20k.
- The coda learned to read samples: sampled val loss minus true-plan val loss fell 0.64 (5k)
  → 0.52 (10k) → 0.22 (15k) → 0.26 (19750).
- The thinker predicts 12 % of the plan's bits from everything it sees (rel 0.878 of the
  uniform floor); the past is 3.1 % of that. The share climbs with a coarser code (0.02 %
  at 72 bits, 1.1 % on the flow code, 3.1 % at 24 bits) but the whole plan gets less
  worth reading as it gets coarser.
- At 10k (first step of phase 3) every own-plan number was worse: OWN@4 −0.051 against the
  ruler, ZERO ahead of OWN@4 by 0.015.

## Verdict

Failure. Four of the five semantic predictions fail and P-P6 fails. On generation the
sampled plan does no better than a foreign plan or no plan at all, rounds add nothing the
interval can see, and the arm sits 0.030 below the strict ruler and 0.072 below the plain
model on the same cuts. The only depth signal is 0.012 nats on CE, real but tiny, and it
does not reach the generated text. What the method could not distinguish: the oracle line
says the code itself is the limit (a perfect thinker matches the ruler, no more), so this
arm cannot say whether a masked denoiser would show a semantic k-curve on a code that
carried more of the span. The 72-bit and 1024-d codes carried more and were less
predictable from the past; no code tried so far is both.

## Updated hypothesis

The rate of a next-span code trades against its predictability: a code the past can
predict (24 bits) carries too little to move the generated span, and a code that carries
the span (72 bits, 1024-d) is not predictable from the past. Under the 0.40 nats/token
cross-span budget the two do not meet on web text. Scoring depth on generation, not CE, was
the right call (CE k-curves stayed near zero while the semantic gaps stayed near zero too),
and the semantic probe with the strict ruler and plain controls is the instrument to keep.
