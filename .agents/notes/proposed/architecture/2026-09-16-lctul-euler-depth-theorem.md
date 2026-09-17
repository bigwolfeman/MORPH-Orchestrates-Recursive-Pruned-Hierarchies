# Agent Note: The LCTUL k-curve is flat because a flow sampler on a Gaussian-like code moves one scalar per step

Status: proposed

Date: 2026-09-16. Lean development:
[`lab/theory/lctul_euler_depth/`](../../../../lab/theory/lctul_euler_depth/README.md)
(32 theorems, `lake build` exit 0, axiom check clean, no `sorry`). Spec:
[`docs/tul-code-spec.md`](../../../../docs/tul-code-spec.md). Design record:
[`2026-09-14-tul-span-code.md`](2026-09-14-tul-span-code.md). Schedule under test:
[`2026-09-16-lctul-ladir-recipe.md`](2026-09-16-lctul-ladir-recipe.md). Information side:
[`2026-09-13-information-view-of-the-slot-loop.md`](2026-09-13-information-view-of-the-slot-loop.md)
and `lab/theory/tul_information/`, which this development reuses. Supersession check: no
active note makes a claim about the Euler k-curve; the span-code note records the flat
curve as a measurement and this note gives its cause. Nothing archived.

## Problem

Every LCTUL arm reads the same inference-depth curve. From `k = 1` to `k = 2` the 8-draw
marginal CE falls by 0.05 to 0.11 nats. From `k = 2` to `k = 16` it moves by at most 0.02
nats (thinker-p3 at 55k: 4.676, 4.613, 4.610, 4.611, 4.613 at `k = 1, 2, 4, 8, 16`). The
thinker's sample sits at 1.7 to 1.9 times the code's variance from the true code in every
principal subspace, where an independent draw would read 2.0. Forty thousand thinker-only
steps, best-of-K selection, LeJEPA and a full flow gradient into the encoder did not move
either reading. The open question was whether this is a training failure or what the
design predicts.

## Proposal

Treat the flat k-curve as a theorem about the design, not a defect of a run. The Lean
development proves, for the sampler `euler_sample` as written in `morph/model/tul_code.py`:

1. `k1_endpoint_of_fm_optimal`: the flow loss at `t = 0` with an independent source is
   minimised only by the field `m - z0`, and one Euler step then returns the conditional
   mean `m` from every source draw. The `k = 1` sample is a constant, and it lies inside the
   RMS shell (`norm_wmean_le_of_shell`). It is a different object from every code the coda
   trained on. The `k = 1 → 2` step is mean against sample, and it happens once.
2. `eulerEndpoint_gauss`: on the isotropic-Gaussian field the `k`-step endpoint is
   `μ + λ_k • z0` for every `k ≥ 1`, with `λ_1 = 0`, `λ_2 = 1/2`, `λ_k > 0` after that. The
   mean is exact at every depth, the noise direction is the source's, and `k` sets one
   scalar radius. Nothing about the span enters through `k`.
3. `eulerEndpoint_dirac`: a code the context determines is sampled exactly in one step.
4. `euler_error_bound` and `euler_error_bound_curvature`: the Euler endpoint differs from
   the exact one by at most the curvature of the field along the path times `e^L / (2k)`.
   Zero curvature, zero difference at every `k`. For an L²-optimal field the curvature exists
   only where the conditional law has separated modes.
5. `residual_ratio_conditional_draw`: a sampler that lands on the conditional law reads
   `2 (1 - R²)` on the subspace probe, where `R²` is the share of the code's variance the
   context explains. The measured 1.7 to 1.9 is `R²` between 0.05 and 0.15, which is what
   a 0.40-nat-per-token cross-span budget can explain of a 4.4-nat-per-token span.
6. `sample_mi_le`: the sampled cell at any `k` carries at most `I(tape, seed; span)`. No
   depth can move the coda's floor below `H(span | tape, seed)`.

The numerical companion `sim/kcurve.py` runs the sampler on the exact
conditional-expectation field and shows the two regimes: a Gaussian conditional gives a
one-scalar family; a mixture whose modes sit about seven within-mode standard deviations
apart gains 8 to 27 nats of log-density from `k = 2` to `k = 16` and its mode-hit rate goes
to 1.00, while a mixture whose modes sit two standard deviations apart gains 3 nats by
`k = 4` and nothing after.

Reading for LCTUL. The code is a near-verbatim copy of a span with about 50 nats of
entropy given the past. Its conditional is a mixture of about `e^50` near-orthogonal
unit-RMS codes of nearly equal weight, blurred by the noise augmentation at 0.5 to 1.0 on a
unit scale: modes one to two standard deviations apart. The field a thinker can learn on it
is the affine mean field of theorem 2 (the residual says so), and theorem 2 gives that
field a one-scalar k-curve. The flat k-curve is the design's prediction.

Consequences the note asks Wolfe to accept:

- The queued LaDiR chain (noise 3.0, frozen coda, thinker on its own tape) lowers the mode
  separation against blur from about two standard deviations to about one third. Under
  theorems 2 and 4 it can change what the frozen coda tolerates, not the k-curve past
  `k = 2`. Prediction, to be read on the chain: `k16 - k2` within 0.02 on every stage.
- A code that earns Euler depth must be a code of something the past nearly determines,
  with a few separated outcomes, read by a coda whose tolerance radius is smaller than the
  separation. LaDiR's latent is that (a reasoning step given the problem and the earlier
  steps, scored by a discrete accuracy). A copy of a web-text span is not.
- Flow matching's field is a conditional expectation at every `t`. A field that does not
  resolve the modes collapses toward the affine mean field, and MSE regression onto a
  2048-dimensional copy of a high-entropy span does not resolve `e^50` modes. Changing the
  training schedule, the batch, the selection rule or the encoder's gradient leaves the
  object unchanged.

## Alternatives considered

- **Keep training the thinker longer or larger.** Rejected by the measurement the theorems
  explain: the thinker-only arm reached its floor by step 10000 and the residual is where a
  correct sampler of this conditional sits. More steps cannot move a scalar family.
- **Read the k-curve on a trusting coda instead.** The phase-2 coda reads `k16 - k1` of
  −1.0 to −1.9 nats at 10 to 14 nats absolute. Theorem 1 explains it (the mean is off the
  shell), and theorem 2 says every `k ≥ 2` is one family; the reading is the `k = 1 → 2`
  event on a coda that cannot use any of them.
- **A different sampler (higher-order integrator, more steps).** Theorem 4: a better
  integrator reaches the exact flow sooner, and the exact flow of an affine field is
  `μ + (s1/s0) z0`. It changes the radius, not the family.
- **Best-of-K or guidance.** Measured flat (`2026-09-15-tul-code-xm.md`); theorem 6 says
  why: the selected draw is still `F(ctx, z0)` and carries at most what the context does.
- **A discrete or low-entropy code.** Not built. It is the one change the theorems point
  at: a code of a decision the past nearly determines (which of a few continuations, a VQ
  code with few codewords, a plan bit) rather than a copy of the span. Recorded as the
  candidate next arm, not decided here.

## Acceptance criteria

The note moves to `implemented/` when Wolfe accepts the reading and one of the two
predictions is read: the LaDiR chain's `k16 - k2` stays within 0.02 nats on every stage
(the theorem's prediction), or a low-entropy discrete-code arm reads a k-curve with
`k16 - k2` beyond 0.05 nats (the theorem's converse). It moves to `rejected/` if the LaDiR
chain reads `k16 - k2` beyond 0.05 nats on a frozen coda, which would mean the code's
conditional has separated modes the theorems did not credit it with.

## Risks

- The Gaussian field is assumed, not derived, inside Lean (standard Gaussian conditioning);
  the finite-weight model stands in for Gaussian measures. The conclusions rest on the
  affine form and the measured residual, and the README names every gap.
- "About `e^50` modes one to two standard deviations apart" is an estimate from the ruler's
  CE, the span length, the measured pairwise cosine and the noise level; it is not measured
  directly in code space. A direct measurement (the multimodality of `p(z | ctx)` in the
  top-128 subspace) would sharpen it.
- One seed per arm behind every number the note explains.
