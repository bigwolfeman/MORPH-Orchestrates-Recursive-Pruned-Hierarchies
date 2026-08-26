# Agent Note: Explorative Modeling applies to the TUL plan vector, not to the token head

Status: rejected — reader alignment measured at 1.36-1.45 (1.0 = random) and flat across the onset; the readers do not pull the plan apart

> **REJECTED the same day it was written, by its own acceptance test.** The mechanism below
> — that `h_i` is a compromise between conflicting readers — is measured and false. See
> [../../../../lab/experiments/failures/2026-08-24-tul-reader-gradient-conflict.md](../../../../lab/experiments/failures/2026-08-24-tul-reader-gradient-conflict.md).
> Normalised reader alignment is **1.36 to 1.45** across the whole onset, where 1.0 is K
> independent random directions: the readers agree slightly BETTER than random, with a mean
> pairwise cosine of +0.02, and the value is FLAT while the core share goes from 0.017 to
> 0.961.
>
> The accounting below is also wrong in a way worth keeping. The slot's own label carries
> 2.8 % of its span's LOSS WEIGHT but delivers about **half the gradient** reaching `h_i`
> (`route_frac` 0.45 to 0.64). Loss weight is not gradient share — the direct route is one
> projection into one coda position, while the reader route passes through attention, which
> attenuates it. **Do not repeat the "the core is trained 97 % indirectly" claim.** It is
> reasoning about the wrong quantity, and it is the reason this note existed.
>
> What is NOT refuted, and was flagged in Risks below before the test ran: the
> across-dataset form. Under teacher forcing the next span is given, so the probe measures
> disagreement among readers of ONE known continuation. XM's one-to-many coupling is about
> the same state serving different continuations in different EXAMPLES. That needs a
> data-construction experiment, not a probe, and nothing here speaks to it. Lever 3 below
> (XM on the plan) rests on that untested form alone; levers 1 and 2 rested on the refuted
> mechanism and should not be built on this note's reasoning.

## Problem

MORPH dismissed Gladstone, Ji & Du, *Explorative Modeling: Unlocking a Third Pretraining
Axis and End-to-End Generation* (arXiv:2607.27372, 29 Jul 2026) on the grounds that MORPH
trains autoregressive next-token prediction, and the paper's mode-blurring disease is a
property of reconstructive models with low generative expressivity.

That dismissal is correct for the token head and wrong for the looped core, and the paper
says so itself. Its own escape clause:

> "factoring generation does not remove this generative expressivity limitation entirely,
> as even highly scalable generative modeling approaches, such as diffusion and
> autoregression, can leave modes uncaptured when single predictions inside their factored
> procedure face many valid targets at once."

XM defines **generative expressivity** `E` as the number of distinct modes a training
objective lets a prediction keep, and shows direct regressors have `E = 1`: "even with
unlimited parameters and data, their best possible output (loss minimizer) is still a
single blurred mean of all the modes." The cause is one-to-many coupling — "a single input
is typically coupled to many valid targets across the dataset."

**The TUL slot state `h_i` is a single prediction facing many valid targets at once.**
Measured on the shipped configuration:

* Its own label carries **2.8 %** of its span's loss weight — one token, `t_1(span i+1)`,
  at weight 0.5 against a mean span weight of 18.04 (spec §5, `_tul_half_weights`). At
  `span_cap` 8 this rises to 7.8 %.
* The other **97 %** reaches the core only through the coda's attention from token
  positions to the slot's `prefix_k` positions, and those contributions are **summed** onto
  one vector in `R^1024`. A point estimate has no way to represent "70 % this, 30 % that".
* The slot's input is already a plain mean of its span's token embeddings, whose deviation
  from the corpus mean falls as `1/sqrt(L)` — measured slope −0.473 / −0.504 / −0.527 at
  three caps, see
  [lab/experiments/successes/2026-08-24-tul-span-pooling-law.md](../../../../lab/experiments/successes/2026-08-24-tul-span-pooling-law.md).

So the plan is blurred on the way in and graded by a sum of conflicting demands on the way
out, while the token head downstream of it keeps `E > 1` and looks perfectly healthy. The
measured post-loop slot-state effective rank of **1.7 to 4.8 in 1024 dimensions** is what a
blurred plan looks like.

## Proposal

Treat `E` as a property of the PLAN, not only of the output distribution, and raise it.
Three levers, increasing in ambition, listed so the cheap one is tried first:

1. **Raise the direct route.** Give the slot more than one token of label, so Route A is not
   2.8 %. Must stay in token space: the spec forbids regressing onto the slot state (LCM,
   CoCoMix, BT §4.2) and forbids decoding a span from one vector with no token path
   (Huginn). A multi-label CE over the next span's token SET satisfies both.
2. **Stop the summing.** More `prefix_k` channels so different readers pull on different
   vectors rather than onto one.
3. **XM on the plan.** Sample `K` candidate `h_i` — K Poisson depths, or K perturbations —
   score each by the resulting span loss, and train only the best. This is XM's own
   mechanism transplanted from the output to the latent, and it raises the plan's
   expressivity from 1 to at least K.

## Alternatives considered

* **Keep the dismissal.** Defensible for the head and it is what we did. Rejected because
  the paper's own text names the case that bites us, and because the credit-assignment
  numbers above were not known when the dismissal was made.
* **Blame pooling alone and fix the boundary rule.** `span_cap` 32 -> 8 raises slot-input
  effective rank 1.97x, config-only. Kept as a cheap lever, rejected as THE answer: pooling
  costs about 2x of input rank while the loop costs about 10x, so the input is not where
  most of the diversity dies.
* **Blame the optimizer's slow accumulator.** Measured and refuted —
  [lab/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md](../../../../lab/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md).
* **Blame under-determination of the core map.** Measured and refuted —
  [lab/experiments/failures/2026-08-24-tul-core-underdetermined.md](../../../../lab/experiments/failures/2026-08-24-tul-core-underdetermined.md).
  The input span is nearly full rank and the concentration reproduces on the token path.
* **Adopt XM at the token head.** Rejected: the head already has `E > 1` through softmax
  CE. The paper's gains there are for reconstructive objectives MORPH does not run.

## Acceptance criteria

Before any of the three levers is built, the mechanism must survive its own test. Define,
for the readers `r` of slot `i`,

    conflict_i = || sum_r g_r || / sum_r || g_r ||        g_r = d(loss at r) / d(h_i)

normalised by `sqrt(K)`, since K independent random directions give `conflict = 1/sqrt(K)`.

* **Confirmed** if the normalised alignment is at or below 1 (readers no better than random)
  AND falls as the takeover develops, AND `||g_A||` (the direct label) is a small fraction
  of `sum_r ||g_r||`.
* **Refuted** if readers are strongly aligned (normalised alignment well above 1), which
  would mean the summed gradient has a clear direction and `h_i` is not a compromise.

## Risks

* **Teacher forcing is the strongest counter-argument and is not yet ruled out.** Within one
  training example the next span is GIVEN, so `h_i`'s target is determined; the one-to-many
  coupling appears only across the dataset. Whether that suffices to force the blur is what
  the acceptance test decides, not what the argument decides.
* **`h_i` is a representation, not an output.** XM's `E` counts modes of a distribution, and
  the coda downstream of `h_i` still has a softmax. The transfer is by geometry — a
  deterministic point, coupled one-to-many, optimised by a summed objective — not by the
  paper's literal definition. Do not cite XM as if it proved this.
* Lever 1 changes the objective, so it contaminates `preclip/core_share` the way the
  spectral penalty did. Score it with a measure that survives a region-local loss term.
* Lever 3 costs K forward passes of the core per step. The core is 9-19x cheaper per
  position than the token path, so this may be affordable, but it is unmeasured.
