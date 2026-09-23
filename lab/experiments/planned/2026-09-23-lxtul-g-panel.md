# Planned: LXTUL-G, the stochastic contractive slot loop trained as a latent-variable model

Status: planned

Date: 2026-09-23 11:07 (frozen before any build or GPU step). Note:
[`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../.agents/notes/proposed/architecture/2026-09-23-lxtul-gram-stochastic-loop.md)
(Proposal + Paper refresh). Design synthesis:
`/home/wolfe/morph-scratch/arc/notes/2026-09-23-lxtul-g-paper-synthesis.md`. Wolfe
(2026-09-23): build the arms with subagents, then run them.

## Question

The slot loop's central failure is the coda ignoring the loop. If the loop is a
stochastic, contractive recursion whose steps are proposed at train by a posterior that
sees the next span (GRAM shape: per-pass learned Gaussian step, KL to the loop's own prior
with KL balancing 0.8, beta 0.1), does the coda use the loop's latent, does the loop learn
to propose it from context alone, and does depth or width (N prior samples) earn on tokens
as an outcome?

## Hypothesis

During training the latent carries information about the next span that the tokens do
not, so the coda's cheapest route to the loss goes through it; the KL teaches the prior
to propose it. Depth use is not charged and should emerge only if the prior needs several
contractive passes to reach what the posterior proposes. The risk is the exposure gap:
the coda trained on posterior cells may not be served by prior cells.

## Arms

All on `tul_slot_spandec_strict` (seq 1024, batch 6, seed 1, norm_match, ramp 1000,
fixed-point 1.0 computed on the deterministic part), 5000 steps, recon runner, at the commit
carrying the build.

| arm | config | one factor against |
|---|---|---|
| `lxtul-g` | `tul_slot_spandec_strict_gram.yaml` | the strict ruler: the GRAM step, posterior, KL |
| `lxtul-g-meanfree` | `tul_slot_spandec_strict_gram_meanfree.yaml` | lxtul-g: both mean heads fixed at 0 (stochasticity only, GRAM control A1) |
| `lxtul-g-d1` | `tul_slot_spandec_strict_gram_d1.yaml` | lxtul-g: `tul.slot_depth_fixed: 1` (the depth-1-trained twin) |
| `slot-spandec-strict-fan4-all-sd` | `tul_slot_spandec_strict_fan4_all_sd.yaml` | fan4-all: spectral decoupling, an L2 penalty on the coda's token logits (Gradient Starvation) |

Existing partners (sweeps on disk): `slot-spandec-strict` @5000 (ruler),
`slot-spandec-strict-fan4-all` @5000.

## Instruments

Read at 5000 (and 2500 where the runner sweeps) on the same held-out rows:
`ce_post` (coda on one posterior sample), `ce_prior@1`, `ce_iw@N` for N in {1, 4, 16}
(the exact Bayesian per-token read over N prior samples = the multi-sample bound), the
identity null (N copies of one sample; `ce_iw@N` must equal `ce_prior@1`), `ce_elbo`,
`ce_zero` (cells zeroed), per-pass KL mean and distribution, sigma/RMS. The runner's
sweep reads the token K-curve on ONE seeded prior sample (same noise per pass at every
forced depth). Paired CE against the ruler on identical tokens.

## Predictions (frozen)

Ruler at 5k: token K1-K6 +0.0016, worth(zero) 0.1865. Slot-loop floor [-0.0001, +0.0033].

- **P-1 (no collapse).** lxtul-g mean per-slot KL (summed over passes) at 5k above 0.5
  nats: **60 %.**
- **P-2 (the exposure gap closes).** lxtul-g `ce_prior@1 - ce_post` smaller at 5000 than
  at 2500: **65 %.**
- **P-3 (the coda uses the latent).** lxtul-g worth(zero) on a prior sample at 5k above the
  ruler's 0.1865 by more than 0.05: **35 %.**
- **P-4 (width is search, not noise).** lxtul-g width gain `ce_prior@1 - ce_iw@4` above
  0.010, AND larger than lxtul-g-meanfree's width gain: **35 %.**
- **P-5 (depth as an outcome).** lxtul-g token K1-K6 on the seeded prior sample above
  +0.005 with the CI clear of it: **30 %.**
- **P-6 (no price).** lxtul-g `ce_iw@4` paired against the ruler at depth 6 within
  +0.020: **50 %**; better than the ruler: **25 %.**
- **P-7 (training depth).** lxtul-g `ce_iw@4` better than lxtul-g-d1's `ce_iw@4` by
  more than 0.010: **35 %.**
- **P-8 (spectral decoupling).** fan4-all-sd token K1-K6 above fan4-all's +0.0049 by more
  than 0.005: **20 %.**
- **P-9 (cost).** lxtul-g tok/s at step 200 at or above 0.85x the ruler's (~11.7k): **75 %.**

## Verdict rules

No clause passes or fails on a cosine. The verdict rests on the exposure-gap numbers, the
worth profile, the width gain against the mean-free floor, the token K-curve and the
paired CE. P-3 or P-4 holding with P-5 failing files as: the latent is used and explores
across samples but depth does not emerge. P-1 failing (KL collapsed) with P-3 failing
files as posterior collapse; the fallback (free bits 1 nat per slot) is a new prereg.

## Method

Build by opus subagents on worktrees at HEAD (model + instruments + tests + smokes;
spectral decoupling separately), merged by me, gate tests on CPU, then the runner. The
posterior-leak perturbation test is a build gate: in prior mode, the cells must not move
when the next span's tokens change. Artifacts under
`../results/2026-09-23-lxtul-g/` (json and runlog only).

## Method amendment (2026-09-23 11:52, after the build review, before any run)

Predictions unchanged. What the review of the two builds fixed or recorded:

- **fan4-all-sd lambda is 2e-7, not the builder's 1e-5.** Measured on the trained 5k ruler
  and fan4-all-fp0 checkpoints, mean ||z||^2 per token is 1.03e6 / 1.04e6 (per-logit rms
  4.6); 1e-5 would be a 5.1-nat penalty (125 % of CE). 2e-7 is 0.10 nats, about 2.5 %.
- **The fixed-point term reads u_T** (the deterministic part), so it puts no gradient on
  the LAST pass's sigma. At depth >= 2 it still reaches earlier passes' steps through
  h_{T-1}; left in by design (the loop is meant to stay contractive around its noise) and
  watched through `tul/gram_sigma_ratio_prior` / `_post`. If sigma/r falls toward the
  1e-4 floor, that is the cause to test first.
- **The Bayesian read resets at every span.** Each span's tokens are reweighted over the N
  joint prior samples by that span's own earlier tokens only. It is a valid causal
  predictor and equals the multi-sample bound per span; it is not the whole-row
  sequential importance weight.
- Build commits: gram e75e78c, spectral decoupling 3221610, merge (this commit's parent).
  Review: 44 + 131 + 58 tests passed on the branches, 185 on the merge; five sabotages
  caught; 40-step GPU smokes of gram and meanfree exit 0.

## Amendment: a fifth arm (2026-09-23 12:24, while lxtul-g ran, before its readouts)

`lxtul-g-b1` (`tul_slot_spandec_strict_gram_b1.yaml`): lxtul-g with `tul.gram_beta`
1.0. The first arm's training log at step 2720 reads a KL of 43 nats per slot (1.8 per
coda token), posterior sigma/r 0.21 -> 0.09 and a runner val loss on prior samples of 5.28
at 2500: the posterior is paid to copy the next span at beta 0.1. At beta 1 the training
objective is the ELBO, an upper bound on the NLL. The frozen predictions P-1..P-9 stay
scored on `lxtul-g`; this arm has its own, frozen now:

- **P-10.** lxtul-g-b1 `ce_prior@1 - ce_post` at 5k smaller than lxtul-g's: **75 %.**
- **P-11.** lxtul-g-b1 `ce_iw@4` better than lxtul-g's `ce_iw@4` (paired): **60 %.**
- **P-12.** lxtul-g-b1 mean KL per slot at 5k above 0.5 nats (not collapsed): **50 %.**
