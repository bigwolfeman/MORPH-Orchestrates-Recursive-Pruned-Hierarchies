# Agent Note: LCTUL-D — a discrete code and a masked denoiser, because the flow thinker never learned the conditional

Status: proposed

Date: 2026-09-16. Spec: [`docs/tul-code-spec.md`](../../../../docs/tul-code-spec.md) §16.
Prereg: [`lab/experiments/planned/2026-09-16-lctul-d-first-arm.md`](../../../../lab/experiments/planned/2026-09-16-lctul-d-first-arm.md).
Follows [`2026-09-14-tul-span-code.md`](2026-09-14-tul-span-code.md) (LCTUL, the design this
one keeps) and [`2026-09-16-lctul-euler-depth-theorem.md`](2026-09-16-lctul-euler-depth-theorem.md)
(the proofs). Supersession check: [`2026-09-16-lctul-ladir-recipe.md`](2026-09-16-lctul-ladir-recipe.md)
changes the SCHEDULE of the flow thinker and stays proposed with its chain queued; this note
changes the CODE and the OBJECTIVE and does not supersede it. [`2026-09-13-discrete-thought-vq.md`](2026-09-13-discrete-thought-vq.md)
built the quantiser this note reuses (on the slot loop's exit state; never run) and stays as is.

## Problem

Two measured facts and one theorem close the flow thinker as the way to earn depth:

1. The thinker is context-blind. On `tul-code-cfg` at 20k the flow loss with the past
   removed (the trained null condition on every row) is 0.3161 of the null floor; with the
   past it is 0.3125. The past is worth 1 % of the loss
   (`lab/experiments/results/2026-09-16-lctul-context-blind-probe/`). That is what the sample
   residual of 1.7 to 1.9 times the code variance was saying since 2026-09-14.
2. The coda cannot use such a sample: a coda trained on truth codes decodes it to word salad
   (9 to 13 nats), a coda trained on samples learns to ignore the cell (semantic OWN−SHUF
   +0.009, CI spanning 0).
3. Euler depth cannot pay on it. k = 1 is the conditional mean (proved), and on a
   Gaussian-like conditional every k ≥ 2 endpoint is mean + λ_k · noise (proved): one scalar
   moves. The code, a noisy copy of a 50-nat span in 1024 dimensions, is that object.

The cause is the objective. A velocity regression on a high-entropy continuous code spends
its gradient on noise removal; the conditional is a sliver of it. Wolfe's reading (flow
matching on text is data-starved) was right and is now a number.

## Proposal

Keep LCTUL's geometry (strict coda, E on the next span, the doubled slot sequence, the
phases, the instruments) and change three objects, `tul.code_discrete: true`:

- The code is N = prefix_k · groups symbols from a cosine codebook (`TULThoughtVQ` on E's
  pooled vector). The coda reads rmsnorm(lift(symbols)) with the straight-through gradient
  into E. Rate control is LaDiR's symbol substitution (`code_sub_p` 0.3), not Gaussian noise.
- The thinker is a masked denoiser over the symbols (MDLM / LLaDA objective on the doubled
  slot sequence). Its loss is the ELBO of −log p(code | past) in nats per span: every unit of
  its gradient is about the past, and the number is readable against the uniform floor
  N · log C and against the null-conditioned pass (the context share, the instrument that
  found fact 1).
- The sampler is k rounds of parallel unmasking (MaskGIT). k = 1 is a real one-shot sample,
  k = N is one symbol per round. Depth resolves the dependence among a span's symbols, which
  is what the k-curve then measures.

What this buys, in words: the thinker's task becomes categorical prediction, the one thing
this model learns from a few hundred million tokens; depth exists by construction where the
symbols of a span depend on each other; and a sample is a sample at every k.

## Alternatives considered

- **FSQ instead of the VQ-VAE codebook.** No codebook loss, no collapse. Deferred: the VQ
  module is in the tree with its own tests and collapse instruments (`vq_perplexity`); FSQ
  is one config away once the arm reads.
- **Keep flow matching on a low-dimensional continuous code** (project the cells to 64 dims
  for the field). Cuts the noise-removal share but leaves the affine-field theorem in force:
  depth would still change one scalar on a unimodal conditional.
- **An autoregressive thinker over the symbols.** Same objective, fixed order, k ≡ N. Rejected
  for v1: no depth dial, and the point is to read the k-curve.
- **Discrete flow matching / discrete diffusion with a uniform corruption.** The same family;
  the absorbing (MASK) form is the one with the cleanest ELBO and the standard sampler.
- **Lower the Gaussian noise on the continuous code.** Cheap, but it does not touch fact 1
  (the context share is set by the objective, not by the noise level).
- **Cut the LaDiR chain.** Under the theorems it can improve the coda's tolerance of samples
  and not the k-curve. Left queued for Wolfe's call; it answers his frozen-coda question.

## Acceptance criteria

At 5k steps of `tul_code_d` (one seed; a reading, not a verdict), in the prereg:

1. The codes are used: `vq_perplexity` ≥ 64 of 512.
2. The code is read: `val/ce_tf` at least 0.15 below the strict ruler.
3. The thinker is conditional: the context share (denoiser nats with the past minus with the
   null condition) ≥ 10 % of the with-past number (the flow thinker: 1 %).
4. Depth resolves dependence: the 8-draw marginal `ce_k1 − ce_k8` ≥ 0.02 with a CI clear of 0.
5. Rate ≥ 0.8× the flow arm's tok/s.

The note moves to `implemented/` when the arm trains on master with D1–D12 green and 1–4
read, whatever they read. It moves to `rejected/` if 3 fails on two seeds.

## Risks

- The codebook can collapse (perplexity → 1) or E can route around the bottleneck; the
  instrument is `vq_perplexity` and the 2026-09-13 note's decisions 1 and 2 are the guards.
- The 1/t ELBO estimator is heavy-tailed at small t; the trainer clamps t ≥ 1e-3 and the
  band ratios show where the loss sits. If the variance hurts, a t floor of 1/N is the knob.
- A span's symbols may be nearly independent given the past (then the k-curve is flat for an
  honest reason: one round is enough). The context share (criterion 3) is the primary
  reading; criterion 4 is secondary.
- One seed per arm; nothing in the first panel is a verdict.
