# Planned: LXTUL-G, the stochastic contractive slot loop trained as a latent-variable model

Status: failure

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

## Amendment: two controls pulled from the queue (2026-09-23 13:27)

`lxtul-g-meanfree` and `lxtul-g-d1` are removed from the runner queue before they
started. Both control the beta-0.1 posterior-trained design, and lxtul-g's 5k sweep shows
that design is teacher-forced across passes (prior-sample token CE 5.0956 at depth 6
against the ruler's 4.3474; span-decoder CE 8.66 -> 9.24 with depth), so their readings
would measure a known exposure gap, not search against noise. Their configs stay in the
tree. P-4 and P-7 are therefore NOT RUN. The design moves to training on K prior rollouts
under the multi-sample bound (Wolfe's go, 2026-09-23 13:27): new prereg
`lab/experiments/planned/2026-09-23-lxtul-gk-multisample.md`.

## Results (filed 2026-09-23 16:06)

Artifacts: [`../results/2026-09-23-lxtul-g/`](../results/2026-09-23-lxtul-g/) (sweeps, worth,
slot state, the probe JSONs, `paired_5000.json`, run logs). Probe: `lab/divergence/lxtul_g_probe.py`
on the Spark at 1416d4d, 192 validation rows (200,719 tokens), seeds 0.. for the prior samples.
Paired CIs bootstrap over 1,024-token stream blocks against the ruler `slot-spandec-strict` @5000 depth 6.

The exposure gap (one prior sample minus one posterior sample, same rows):

| arm | step | ce_post | ce_prior@1 | ce_iw@4 | ce_iw@16 | ce_zero | gap | KL / token |
|---|---|---|---|---|---|---|---|---|
| lxtul-g | 2500 | 4.4798 | 5.1905 | 5.0881 | | 4.8893 | 0.7108 | 1.87 |
| lxtul-g | 5000 | 4.0270 | 5.0110 | 4.8523 | 4.7957 | 4.6862 | 0.9840 | 2.81 |
| lxtul-g-b1 | 5000 | 3.9934 | 6.0620 | 5.5896 | 5.2901 | 4.8693 | 2.0686 | 3.58 |

The identity null (`ce_iw_identity@4`) equals `ce_prior@1` exactly on both arms.

Paired against the ruler at depth 6 (on the probe's tokens):

| reading | lxtul-g | lxtul-g-b1 |
|---|---|---|
| ce_iw@4 | +0.5901 [+0.5796, +0.6012] | +1.3275 [+1.3105, +1.3447] |
| ce_iw@16 | +0.5336 [+0.5239, +0.5437] | +1.0280 [+1.0135, +1.0424] |
| ce_zero (cells zeroed) | +0.4241 [+0.4150, +0.4338] | +0.6072 [+0.5943, +0.6205] |
| ce_post (reads the answer) | −0.2352 [−0.2447, −0.2254] | −0.2687 [−0.2801, −0.2577] |
| ce_iw@4, depth 1 minus depth 6 | +0.0123 [+0.0093, +0.0150] | −0.2674 [−0.2732, −0.2616] |
| ce_prior@1, depth 1 minus depth 6 | −0.0010 [−0.0055, +0.0037] | −0.3481 [−0.3546, −0.3418] |

Runner sweeps (one seeded prior sample, 480 rows): lxtul-g K1−K6 −0.1340 at 2500 and
−0.0050 [−0.0083, −0.0018] at 5000. b1 K1−K6 −0.0001 [−0.0003, +0.0001] at 2500 and
−0.3580 at 5000. The lxtul-g worth profile at 5000: a shuffled cell costs +1.12 nats at a
span's first token, and a zeroed cell is 0.60 nats BETTER than the own prior cell there.

b1's training history (wandb `wpnauq56`). Steps 0 to ~2750: the posterior sat on the prior
(KL per token ≤ 0.016), val 4.58 to 4.68, and the 2500 sweep read 4.6878 at depth 6
against the ruler's 4.6567. From step 3000: KL per token 0.29 → 3.57, the pre-noise state
RMS `u_rms` 1.7 → 21.3, grad norm 1.5 → 27, val 4.58 → 5.88, and `loss/total` 9.26 → 12.88.
The optimizer climbed its own objective by ~3.6 nats over the last 2500 steps.

fan4-all-sd at 5000: K1−K6 +0.0043 [+0.0038, +0.0047] (fan4-all: +0.0049 [+0.0044, +0.0055]).
Depth-6 CE paired: +0.0141 [+0.0117, +0.0163] against fan4-all, −0.0327 [−0.0356, −0.0299]
against the ruler. Tok/s at step 200: lxtul-g 11,712, b1 11,657, fan4-all-sd 7,709, ruler 11,759.

| clause | reading | verdict |
|---|---|---|
| P-1 KL per slot > 0.5 | 58.85 nats | held |
| P-2 gap smaller at 5000 than 2500 | 0.7108 → 0.9840 | failed |
| P-3 worth(zero) on a prior sample > 0.2365 | −0.325 | failed |
| P-4 width vs mean-free | control pulled | not run |
| P-5 K1−K6 > +0.005 | −0.0050 [−0.0083, −0.0018] | failed |
| P-6 ce_iw@4 within +0.020 of the ruler | +0.5901 | failed |
| P-7 training depth vs d1 | control pulled | not run |
| P-8 fan4-all-sd K1−K6 > +0.0099 | +0.0043 | failed |
| P-9 tok/s ≥ 0.85x ruler | 0.996x | held |
| P-10 b1 gap smaller than lxtul-g's | 2.0686 vs 0.9840 | failed |
| P-11 b1 ce_iw@4 better than lxtul-g's | +0.7374 [+0.7184, +0.7549] worse | failed |
| P-12 b1 KL per slot > 0.5 | 74.84 nats | held |

## Verdict

Failure. The latent does not collapse (P-1, P-12), and the coda does read it: a shuffled
cell costs 1.12 nats at a span's first token. But the coda reads the POSTERIOR's cell.
Training takes a posterior step at every pass and eval takes prior steps, so the reader is
trained on cells the deployed loop never produces. The gap grows with training (P-2), a
prior cell does more harm than an empty one (P-3), and more prior passes do more harm on
b1 (P-5). At beta 1 the ELBO did not hold the posterior near the prior; the run went
unstable after step 3000 and ascended its own loss. Spectral decoupling costs 0.014 nats
against fan4-all and leaves its K-curve where it was (P-8).

Two readings are worth keeping. Width earns inside the Bayesian read (lxtul-g 0.159 nats
at 4 samples and 0.215 at 16; b1 0.47 and 0.77), and under that read depth 6 beats depth 1
by 0.0123 [+0.0093, +0.0150] on lxtul-g while the single-sample curve is flat. Both
readings sit on a trajectory the model was not trained to produce.

## Updated hypothesis

A stochastic slot loop has to be trained on the rollouts it will be deployed with. The
next arm trains on K prior rollouts under the multi-sample bound, which is the deployed
Bayesian read's own loss, with no posterior and no KL: LXTUL-GK,
[`2026-09-23-lxtul-gk-multisample.md`](../planned/2026-09-23-lxtul-gk-multisample.md). If its width
gain survives and the depth-under-width reading grows, that is the loop contributing
through search. Spectral decoupling is closed as a lever on the fan4-all read.

## Addendum (2026-09-23 16:13): b1 at step 2500

The probe finished after filing. lxtul-g-b1 at 2500: ce_post 4.6084, ce_prior@1 4.6092,
ce_iw@4 4.6089, ce_iw@16 4.6089, ce_zero 4.7247, KL 0.018 per token. The gap is 0.0008
and width gains 0.0003: in its collapsed phase b1 is a near-deterministic loop whose
cells the coda uses (zeroing them costs 0.116 nats). No verdict changes.
