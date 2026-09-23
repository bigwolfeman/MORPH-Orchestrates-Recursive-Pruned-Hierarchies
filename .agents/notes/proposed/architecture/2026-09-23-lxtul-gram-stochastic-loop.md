# Agent Note: LXTUL as one stochastic, contractive slot loop trained as a latent-variable model (GRAM shape)

Status: proposed

## Problem

The slot loop's central fight is loop contribution against the coda ignoring the loop.
Measured directly: without a restriction geometry, the loop's side-head K-curve is 33x to
106x the coda's token K-curve on the same rows (A1: `ce_emit` 0.542 vs `ce_main` 0.005);
with the mask, the coda converts about a tenth of it at a CE price of 0.13 nats or more
(`/home/wolfe/morph-scratch/arc/notes/2026-09-23-slot-loop-large-readings.md`, from
`lab/experiments/successes/2026-09-04-arc-e4-mask-under-constraint.md` and the E13/E14
filings). The loop computes things with depth; the coda has a cheaper route to the loss.

Every lane that tried to FORCE the loop's use failed: per-pass targets are met in one step,
restriction geometries tax CE, the relay chain is a tax refund, teacher-forced denoising is
a bypass. The 2026-09-22/23 panel closed two more explanations: a direct per-token read of
the exit state (`tul.bcast`, `bcast_layers: all`) is used and worth about 0.015 nats with
the K-curve unchanged, and the strict loop's value over its depth-1 twin is a constant
0.0075 from 5k to 20k (`lab/experiments/failures/2026-09-22-arc-slot-token-like-read.md`).
And LCTUL-J settled that with a stop-gradient cell and a token-trained coda, a change to
the target never reaches the reader (`lab/experiments/failures/2026-09-22-lctul-ema-target.md`).

Wolfe's design rules (2026-09-23): TUL's goal is to amortize loop cost; LXTUL, the prime
target, is the insight that TUL's structure allows latent EXPLORATION per span; absolute
CE against no-TUL matters less than the coda using the loop; depth use must be EMERGENT
from the flow through a mostly contractive loop, not a hard objective; wire the pieces into
ONE loop.

## Proposal

Build LXTUL as GRAM (Generative Recursive Reasoning, arXiv 2605.19376; reading note
`docs/references/looping-depth/latent-exploration/gram/gram.md`) on the slot loop.

```
LXTUL-G: one stochastic, contractive slot loop
├── each pass: h_{t+1} = f(h_t) + eps_t, eps_t ~ N(mu, sigma^2) from a learned head
│   f is the existing core (fixed-point term on, so the map stays mostly contractive)
├── prior    q_prior(eps_t | h_t)                  sees the context only (the loop itself)
├── posterior q_post(eps_t | h_t, next span)       sees the span the slot precedes (train only)
├── training: the coda decodes the span from the POSTERIOR trajectory's exit, token CE,
│   gradient flows into the loop and both heads (no stop-gradient on the cell)
├── KL( q_post || q_prior ) per pass, summed, with free bits per slot
├── K streams = K samples of the same loop (replaces seeded streams and the epivol term)
└── inference: K prior samples per span, written to K prefix cells; the coda's per-token
    read chooses (fan_mix all); a value head over trajectories is the later option
```

Why the coda must use it: during training the latent carries information about the next
span that the tokens cannot supply, so the cheapest route to the loss goes THROUGH the
latent. That is the mechanism by which VAE decoders are made to use latents (Bowman 2015,
Optimus, GRAM), and it needs no cut on the coda. Why the loop learns to explore: the KL
makes the prior propose what the posterior used, and the prior's variance is the
exploration the K samples draw from. Nothing is assigned to a pass; depth use is read as an
outcome (K-curve, per-pass KL, prior-sample CE by depth).

GRAM's own ablations are the reason to build it whole: randomness alone does not help
(N-Queens 50.27), and a deterministic target-conditioned map scores 0.00.

Readings (instruments, not loss terms):
- `ce_prior` vs `ce_post`: the coda on a prior sample against the coda on the posterior.
  The gap is the exposure gap; it must close with training.
- The oracle over K prior samples against the per-token read (the fan's instruments).
- Token K-curve on prior samples; per-pass KL; the fraction of slots at the free-bits floor.
- Worth profile (zero / shuffle) of the written cells.

## Alternatives considered

- **Spectral decoupling** (an L2 penalty on the coda's logits; Gradient Starvation, arXiv
  2011.09468, `docs/references.md` section 9). Targets the token path explaining the loss
  first and starving the latent. Cheap, a soft penalty, not a cut. Kept as a companion arm
  on the fan4-all base, not as the design.
- **Token-state dropout sweep** (Bowman word dropout; the frozen prereg
  `lab/experiments/planned/2026-08-27-token-tax-headroom.md`). A soft tax on the coda's
  cheap path. Closer to the forcing Wolfe ruled out; held.
- **Per-candidate credit from the reader's gain** (LTC K x M advantage, Quiet-STaR
  REINFORCE, Bifurcation energy). Addresses credit, not whether the latent is informative.
  A possible second stage for the K samples.
- **Decode-then-encode realignment / Gumbel-Softmax on the VQ code** (DiscoLoop, arXiv
  2607.00341). Puts the latent in a format the reader already reads. Parked.
- **Keep fan4-all as is** (seeded streams, epivol, write-all, winner-take-all credit;
  K1-K6 +0.0049, reader recovers 0.056 of 0.099 oracle nats). It is the composition of the
  existing knobs; its streams carry nothing about the future the coda cannot get, which is
  the missing piece.
- **The hindsight fitted z** (the posterior without the KL): worth -1.48 nats with the
  answer, +0.25 worse than the loop when fitted causally
  (`lab/experiments/failures/2026-09-12-arc-latent-z-gradient.md`). This is the trap the
  KL exists to close.

## Acceptance criteria

To be frozen in the prereg. The shape: the coda uses the latent (worth of the written
cells well above the ruler's 0.19 nats on prior samples); the exposure gap `ce_prior` -
`ce_post` closes over training; the per-token read over K prior samples recovers part of
the oracle; the token K-curve on prior samples rises off the +0.002 floor. No clause on a
cosine; no clause that requires a per-pass behaviour.

## Risks

- **Exposure gap.** Train on the posterior, deploy on the prior: the fitted-z trap. The KL
  is the mechanism; the reading is `ce_prior` against `ce_post`.
- **Posterior collapse.** The KL prices the latent to zero and the coda ignores it again.
  Free bits per slot, and possibly KL annealing.
- **Posterior leak.** The posterior sees the next span; the coda must never read it except
  through the sampled latent. A perturbation test on the next span's tokens with the latent
  fixed is the gate (the 2026-09-22 rule: geometry is verified by perturbation).
- **Cost.** The posterior needs an encoding of the next span per slot at train (E from
  LCTUL, or a pooled prelude read of the span); the K prior samples multiply the coda's
  prefix cells at inference, as fan4-all already does.
- **Paper facts** come from the repo's reading note, not a fresh read; re-read the paper
  before the prereg.

Drafted 2026-09-23 with Wolfe ("Yes, this is very good. Save it to a doc."). Sources:
`/home/wolfe/morph-scratch/arc/notes/2026-09-23-latent-exploration-prior-work.md`,
`/home/wolfe/morph-scratch/arc/notes/2026-09-23-slot-loop-large-readings.md`,
`docs/9-26-TUL-run-history-IMPORTANT.md`.

## Paper refresh (2026-09-23 10:39)

An opus agent read GRAM (re-read from the PDF, all appendices), Variational Reasoning
(arXiv 2509.22637), ReGuLaR (2601.23184), Search Inertness (2607.19635) and Emergent Search
and Backtracking (2602.08100) in full. Reading notes are under
`docs/references/looping-depth/latent-exploration/`; the design synthesis is
`/home/wolfe/morph-scratch/arc/notes/2026-09-23-lxtul-g-paper-synthesis.md`. What it
changes in the proposal above, before any prereg:

- **Posterior input.** An attention pool over the NEXT span's prelude states, copy-initialised
  from the prior head so the KL is exactly 0 at step 0 (GRAM Eq. 12-13, VR Table 4, ReGuLaR
  Table 10).
- **KL.** Per pass, summed over passes (the exact ELBO under full BPTT), with GRAM's KL
  BALANCING at 0.8 and beta 0.1. GRAM uses no free bits and no annealing, and neither do
  the other four; free bits move from the design to a fallback (1 nat per slot) if the KL
  collapses. This corrects the Proposal's "free bits per slot".
- **Noise.** A learned mean and variance, scaled by each slot's detached RMS, so the step
  cannot escape by scale (the fan4-all-noise failure).
- **The fixed-point term must be computed on the deterministic part u_T, not on h_T.**
  On h_T it includes the Gaussian step and pays the model to shrink sigma to zero.
- **Selection at inference: the exact Bayesian per-token read.** Decode the coda once per
  prior sample; weight sample n at token j by softmax over n of its log-likelihood of the
  span's EARLIER tokens. Causal, no training, and its summed log-likelihood equals the
  multi-sample bound exactly, so the deployed CE and the exposure-gap instrument are one
  number. It also decodes each sample alone, the format the coda trains on. Cost: N coda
  passes per span at inference; the one-pass N-cell read (`fan_mix: all`) is the cheaper
  alternative, scored against it as its ceiling. A GRAM-style value head is an instrument
  until its ranking beats chance.
- **Controls** (from Search Inertness and GRAM): width gain against span entropy with a
  mean-free arm (stochasticity only) and an identical-sample null beside it; and the
  first-pass contingency (is the outcome decided at pass 1) on a depth x N factorial.
- **GRAM facts corrected** in the re-read note: ARC-AGI-1 52.0 and ARC-AGI-2 11.1 (the
  earlier note read the LLM bars); no official code; how the posterior reads the target is
  not stated.

## Outcome of the posterior-trained arms (2026-09-23 16:06)

Filed as a failure: [`2026-09-23-lxtul-g-panel.md`](../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md).
The posterior-plus-KL training forward is teacher-forced across passes. At beta 0.1 the
exposure gap grew from 0.71 to 0.98 nats between steps 2500 and 5000, and a prior cell
read 0.32 nats worse than a zeroed cell. At beta 1 the posterior sat on the prior until
step ~2750, then the run went unstable and climbed its own loss. The Gaussian step itself
stays: the note remains `proposed`, and its training objective moves to K prior rollouts
under the multi-sample bound (LXTUL-GK,
[`2026-09-23-lxtul-gk-multisample.md`](../../../../lab/experiments/planned/2026-09-23-lxtul-gk-multisample.md)).
The posterior, the KL and KL balancing are the parts this outcome rejects.
