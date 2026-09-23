# How Model Growth, Recursion, and Boundary Operators Influence Scaling Exponents: reading note

Read 2026-09-23, for the LXTUL-G panel
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../.agents/notes/proposed/architecture/2026-09-23-lxtul-gram-stochastic-loop.md),
prereg [`2026-09-23-lxtul-g-panel.md`](../../../../lab/experiments/failures/2026-09-23-lxtul-g-panel.md)).

Short answer for LXTUL-G: this is a scaling-law paper about looped transformers as a way
to add depth. It has no latent variable, no posterior, no KL and no sampling. It says
nothing about the exposure gap, posterior collapse or KL weighting. What it offers MORPH
is smaller and listed at the end.

## Citation (verified)

Zixi Chen (NYU; work done as an intern at Q Labs), Akshay Vegesna (Q Labs), Samip Dahal
(Q Labs), Andrew Gordon Wilson (NYU, Q Labs). *How Model Growth, Recursion, and Boundary
Operators Influence Scaling Exponents*.
[arXiv 2609.19107v2](https://arxiv.org/abs/2609.19107) [cs.LG]. v1 2026-09-16, v2
2026-09-17 (v2 is the latest on 2026-09-23). 44 pages. No venue on the PDF or the arXiv
page. The arXiv page links code at `github.com/qlabs-eng/scaling-exponents` (not checked).

Read in full: all 44 pages of the v2 PDF, main text and Appendices A to D. The text came
from `pdftotext`. Plot values that appear only inside figures could not be read; every
number below is stated in the text, a table or a caption.

## Local cache

Not copied into the repo (the brief allowed only this note and the manifest line).
Fetched from `https://arxiv.org/pdf/2609.19107v2` on 2026-09-23: 1,842,841 bytes,
44 pages, SHA256 `56e3ead36234eb1a3364fe87021dcb505af1e71eb556a00adea3f507a7a5a832`.
Session copy: `/tmp/claude-1000/.../scratchpad/papers/2609.19107v2.pdf`. Suggested cache
name: `ignore/papers/2609.19107v2-scaling-exponents-growth-recursion.pdf`.

## What it does

It asks whether an architecture change can move the scaling EXPONENT, not only the
constant. It fits a separate compute-optimal recipe for each of eight architectures (base
hyperparameters, tokens per parameter, growth timing, a learning-rate rule) and trains
each as a ladder from about 1e18 to 1e20 FLOPs on FineWeb. All eight are members of one
prelude, core, coda family (the Huginn / Parcae shape). The claims:

1. A boundary operator (normalize the state, add back the prelude output) between core
   passes AND before the coda raises the exponent, at no parameter or FLOP cost.
2. Model growth (start at 2 core passes, go to 4 late in training) raises the exponent
   more. Tied growth reuses one core; untied growth copies the trained core.
3. Untying the weights moves only the constant.
4. With repeated data (100M tokens, 10 epochs) the optimal loop count grows with compute,
   and tied looping beats adding parameters.
5. They explain all of it through "computational depth": the number of blocks that
   change the prediction, measured by a logit-lens proxy.

## The mechanism

The family (Eq. 2), with `P`, `R`, `C` the prelude, core and coda:

    e   = P(s)
    h_k = R_{theta_k}( phi(h_{k-1}, e) ),   k = 1..K
    y   = C( rho(h_K, e) )

Executed depth is `P + K*C + D` blocks. `h_0 = 0`. `K` is FIXED, not sampled. Gradients
flow through every pass (Section 3, last paragraph of "Model growth"). Width = 128 x depth.

The boundary operator (Eq. 3), used as both `phi` and `rho`:

    BO(h, e) = Norm(h) + alpha * e

Normalizing lets each pass write at full relative weight (the "curse of depth" remedy);
adding `e` keeps each pass conditioned on the input. Prior looped models re-inject `e`
only between core passes. This paper also applies it before the coda, and that part
matters (Table 4 below).

Growth (Algorithm 1): train with `K_0` passes; after `g` steps, copy the trained cores
(untied) or just run the tied core more times, and continue with `K_f = m K_0`. The paper
uses 2 to 4. `rho` is the fraction of tokens after growth; the average recurrence is
`K_rho = 2(1 - rho) + 4 rho` (Eq. 9).

The scaling law and the comparison (Eq. 1 and Section 2):

    L(C) = E + A (C / C_0)^(-gamma)
    pi(l) = C_hat_A(l) / C_hat_B(l)       compute multiplier of B over A at loss l

A flat multiplier across budgets is a constant gain. A rising one is an exponent gain.
`E` is fitted once on Vanilla (Huber); the other arms are an affine fit of
`log(L - E)` on `log C`. Slope standard error is below 1e-3 on every arm.

KL effective depth (Section 6, Figure 6): decode the residual stream after each block with
the logit lens; the first block after the KL peak whose decoded distribution is within
2 nats of the final output. The authors call it a proxy: it misses unused middle blocks.

Setup (Table 1, A.1): FineWeb, GPT-2 tokenizer, context 2048, batch 524,288 tokens, Muon
for matrices and AdamW for embeddings and head, pre-norm, RoPE, SwiGLU, QK-norm, zero-init
output projections. Most runs on one 8xH100 node. Ladders d6 to d20 (Vanilla, 120M to
1.8B stored parameters) and d6 to d18 (looped).

## The numbers that matter

**Exponents, FineWeb (Figure 2 / 3 legends).** Vanilla 0.111, Deep Vanilla 0.111,
Operator-1 0.114, Loop-2 0.114, Untied-2 0.114, Deep Vanilla Grow 0.114, Loop-Grow 0.116,
Untied-Grow 0.117.

**Compute multipliers over Vanilla (Section 4.2).**

    Operator-1     1.12x at 1e18  ->  1.25x at 1e20   (operator only, same params and FLOPs)
    Untied-2       1.19x          ->  1.34x
    Untied-Grow    1.30x          ->  1.55x
    Loop-Grow                         1.36x at 1e20   (Vanilla's parameter count)
    Deep Vanilla   flat near 1.08x                    (shape only, no operator)

Growth over Untied-2 rises 1.09x to 1.16x, so growth is an exponent gain. Untied over
tied is 1.08x to 1.06x (fixed) and 1.16x to 1.14x (grown): a constant.

**Extrapolation (Section 4.3, Figure 4).** A 7.4B Untied-Grow (d26, starts at 5B
parameters) at 1.23e21 FLOPs, 8x the largest fitted budget, lands 0.03 below the forecast
loss. CORE 0.3865 against a forecast of 0.3837 and GPT-3 13B's 0.3852, which cost
2.31e22 FLOPs. The "20x" is indicative only: different data and a separate evaluation
pipeline. Multiplier over Vanilla on FineWeb-Edu: 1.6x at 1e20, 1.8x at 1.23e21,
projected 2.7x at 1e25.

**Boundary-operator ablation (Table 4; d8, 1B tokens, width 1024, executed depth 11,
each arm tuned separately).**

    arm                       phi (core)   rho (before coda)   loss
    Deep Vanilla              h            h                   3.2772
    Deep Vanilla + norm       Norm(h)      Norm(h)             3.2781
    Deep Vanilla + injection  h + a e      h + a e             3.2690
    Loop-2                    BO           BO                  3.2704
    Loop-2 no coda inj        BO           Norm(h)             3.2912
    Untied-2                  BO           BO                  3.2563
    Untied-2 no coda inj      BO           Norm(h)             3.2731

Injection before the coda is worth 0.021 (tied) and 0.017 (untied). Normalization alone
is worth nothing (+0.0009); injection alone is worth 0.008. At 1e20, norm-only,
injection-only or no-coda-injection each drop Untied-2's multiplier from 1.34x to
1.17-1.19x (B.1, Figure 15a). Fixing the prelude and coda at 2 and 3 blocks and scaling
only the core pushes the multiplier below 1 at the largest budgets.

**How many passes (A.2.2, A.2.3).** On fresh data at matched compute, jointly scaling
size and tokens, the fitted optimum is 1 to 2 passes at every budget, tied or untied;
K = 3, 4, 6 are worse (Figure 8). At a FIXED model size, the optimal K rises with the
budget (Figure 9), with `E_{K*} = 0.82 TPP_{K1}^0.15` (tied, Eq. 4). Growth target: from
2 passes, growing to 4 is best at every budget; 6 and 8 raise KL effective depth without
lowering loss.

**Growth timing (A.2.3, A.4.3).** Mean post-growth fraction `rho`: Loop-Grow 0.170,
Untied-Grow 0.271, Deep Vanilla Grow 0.544. The minima are broad; a fixed `rho = 0.30`
changes loss by -0.0009 to +0.0019. Grown models prefer a smaller start trained on more
tokens (tokens per parameter 6 to 7-8, Table 7 / Table 9).

**Computational depth (Section 6, Figure 6).** At 1e20: Untied-2 24 against Deep Vanilla
20 (same executed blocks); Untied-Grow 36 against Untied-2 24. Tied and untied curves
coincide, so weight sharing does not change it. Multi-epoch: loop scaling 18 against
depth scaling 14.

**Random recurrence and extra test-time passes (B.4, Figures 16d and 17).** K drawn
uniformly from {2..6} each step after growth (mean 4), evaluated at K = 4: slightly WORSE
than fixed Loop-Grow at matched compute. It does flatten the extra-pass penalty: at k = 8
the loss moves -0.0003 to +0.0028 against +0.008 to +0.042 for fixed Loop-Grow. Five of
seven random-recurrence checkpoints have a minimum at k = 5, worth under 0.001, and more
passes give nothing. "Random recurrence chiefly reduces the penalty for extra passes
rather than providing sustained test-time scaling."

**Recipe transfer (A.4.1 Table 6, B.2, Figure 16b).** A new architecture run on Vanilla's
tuned recipe loses 7.2e-3 (Loop-2), 7.8e-3 (Operator-1), 9.8e-3 (Deep Vanilla) and
13.3e-3 (Untied-2) at d8 (the text says 7 to 26e-3). On the ladder, Operator-1 on
Vanilla's recipe turns its exponent gain into a constant gain.

**Data-constrained (Section 5, Appendix D).** 100M unique FineWeb tokens, 10 epochs. The
optimal loop count rises from 1.4 to 6.7 across budgets; the gap to Operator-1 grows from
0.00 to 0.11. Loop scaling at fixed weight decay reaches the tuned size-scaling loss
(3.40 at 7.5e18) with 2.2x less compute; tuning weight decay on top gains at most 0.02.
Optimal weight decay tracks stored depth (0.4 at d6 to 1.6 at d16-d18) and is nearly flat
in K. Untied looping does not beat Operator-1 here, so weight sharing is the regularizer.
At 4 epochs the optimum stays at 1 to 2 passes.

**Optimizer (B.5).** Muon's gain over Adam is constant under width-only scaling but
shrinks with scale under joint width and depth scaling.

## What LXTUL-G should take

The exposure-gap question (prior samples at ~5.24 val against the ruler's ~4.4, posterior
KL 43 nats per slot at beta 0.1) gets no direct help from this paper. Section 3 fixes
`h_0 = 0`, a fixed K and a deterministic core; there is no latent to encode, no KL to
weight and no prior to follow a posterior. For that problem the GRAM re-read
([`gram-2026-09-23-reread.md`](../latent-exploration/gram/gram-2026-09-23-reread.md), the
"guide only N(mu, 0) = 0.00" row) and
[`variational-reasoning.md`](../latent-exploration/variational-reasoning/variational-reasoning.md)
stay the sources. Three smaller things do carry over.

- **Read the P-6 and P-7 margins against a tuning floor.** Table 6 prices "new
  architecture on the old recipe" at 7e-3 to 13e-3 nats at d8, and Figure 16b shows it can
  hide a real gain. The LXTUL-G arms run the ruler's learning rate and schedule with new
  heads. P-6 allows +0.020 and P-7 asks for 0.010. A miss inside about 0.013 is inside
  the paper's measured recipe-transfer loss, so it is not an architecture verdict until
  the gram arms get at least a learning-rate check. This is the most concrete transfer.
- **Depth growth is an untested schedule for "depth use must emerge".** Section 6's
  reading is that a fixed-depth model pays for depth it cannot use yet, and growth adds it
  late (at 50-80 % of training here). For LXTUL-G this suggests a later arm: the slot loop
  at depth 1-2 while the prior learns to follow the posterior, then grown. The paper does
  not test anything like this with a latent-variable loop, and B.4 says sampled depth
  (MORPH's Poisson draw) and growth are different objects: sampled was slightly worse in
  compute but flatter in test-time depth. Wolfe's standing rule (sampled depth is
  superior, never rank a fixed rung against it) applies. This is a candidate for a
  prereg, not a recommendation.
- **The coda read of the exit cell.** Table 4 says what the coda sees at the boundary
  matters: normalize-and-inject before the coda is worth 0.017-0.021 nats in their loop.
  My inference, not tested in the paper: the LXTUL-G coda is trained on posterior exit
  cells and served prior ones. A `Norm` on the cell before the coda reads it would remove
  the scale part of that train/serve difference (sigma/r shrank 0.21 to 0.09 on the
  posterior). It would not remove the content part, which is the 43-nat KL. Worth a
  one-line check of the cell RMS under `gram_mode post` against a prior sample before
  building anything.

## What does not transfer

- **The latent-variable problem.** No KL, posterior, prior, sampling or exploration
  anywhere in the paper. Nothing here on over-encoding, collapse, beta or KL balancing.
- **The loop it studies.** A token-level loop: every position passes through the core,
  closer to MORPH's paid loop than to the slot loop. Fixed K, `h_0 = 0`, dense
  weights, Muon. The slot loop runs one state per span, Poisson depth, norm_match
  ternary, AdEMAMix, and a coda that reads prefix cells.
- **"The optimum is 1 to 2 passes on fresh data" is a compute-optimality result, not a
  depth verdict.** It holds when size and tokens are scaled together at matched FLOPs. At
  fixed size, more compute favors more passes (Figure 9), and with repeated data up to
  6.7. It does not say web text needs no depth, and must not be quoted that way (Wolfe,
  2026-09-12).
- **Coda injection is a bypass in MORPH's terms.** It adds a second route from the input
  to the coda. In the paper that lowers loss. In the slot loop the coda's cheap route
  around the loop is the central failure. MORPH already injects an input term at every
  coda block (`x0_injects` in `_back_region`, `morph/model/transformer.py`, read for this
  note). More of it works against "the coda uses the loop".
- **KL effective depth as the depth instrument.** A logit-lens proxy that the authors
  flag as blind to unused middle blocks. MORPH's forced-depth token K-curve on identical
  rows measures the loop's contribution directly; keep it. The slot state is not decoded
  per pass, so the logit lens has no direct target on the slot loop anyway.
- **Exponent claims.** They need ladders from 1e18 to 1e20 FLOPs with a recipe fitted per
  architecture. The 5090 panel is one size at 5k to 20k steps. The constant-versus-exponent
  lens does support MORPH's rule that short-horizon CE is not a verdict, but no MORPH
  panel can measure an exponent.
- **The random-recurrence result is a consistency check, not a lever.** Sampled depth
  flattened extra passes and gave under 0.001 at k = 5 and nothing after (B.4). That
  matches MORPH's own reading of a sampled-depth loop that stops earning after a few
  passes. It adds a second setting where the same shape appears; it does not say why.
- **Multi-epoch regularization and the Muon finding.** MORPH trains single-epoch on web
  text with AdEMAMix. Neither applies now.
