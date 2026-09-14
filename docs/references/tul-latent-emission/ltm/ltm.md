> Reading filed 2026-09-14 for docs/tul-code-spec.md. Produced by a fetch-and-summarise pass over the
> arXiv HTML/PDF, not a hand transcription; items the reader flagged as unverified are marked in the text.

# Latent Thought Models (LTM) — reading notes

Paper: "Latent Thought Models with Variational Bayes Inference-Time Computation" (originally
titled "Scalable Language Models with Posterior Inference of Latent Thought Vectors"), Kong,
Zhao, Xu, Pang, Wang, Honig, Si, Li, Xie, Xie, Wu. arXiv:2502.01567. Fetched via the HTML
version (`arxiv.org/html/2502.01567`); quotes below are as returned by that fetch and section
numbers are the paper's own.

## 1. The latent thought vectors

z is layered: **z = (z_1, ..., z_L)**, one sub-vector per Transformer layer L (Section 2.1).
N_z is "the total number of latent vectors" across all layers, not per layer: LTM-Small/Medium
use N_z=24 (a 6-layer model, so 4 per layer), LTM-Large uses N_z=96 (a 12-layer model, so 8
per layer). Each z_l has the same dimension as the hidden state, 512 in every experiment
(Appendix A.1: "All LTMs have 512 hidden dimensions, 8 attention heads... The latent thought
vector z shares the same dimensionality as the hidden vectors"). Representationally the paper
frames z under the "language of thought hypothesis": the vectors are meant to act as "words" of
an internal cognitive language that guides token generation, not as a single sentence summary
(Section 1).

## 2. Prior, conditioning, and the encoder question

**Prior**: fixed isotropic Gaussian, not learned. "an isotropic Gaussian prior over the latent
thought vectors z ~ N(0, I)" (Section 2.1).

**Conditioning**: cross-attention, not a prefix or concatenation. Each decoder layer l takes its
own z_l as keys/values while the token stream x supplies the queries: "z_l ... through
cross-attention" with z providing "keys and values while the input x offers the queries"
(Section 2.1).

**Encoder**: confirmed NOT present. There is no amortized inference network mapping x to z.
Instead the paper uses **classical variational Bayes with local variational parameters** (μ_i,
σ_i²) per example, fit by direct gradient ascent (Adam) on the ELBO for each sequence,
independently of every other sequence. This is the "fast" inner loop in Algorithm 1. The paper
also defines, but does not use as the shipped method, a Langevin-dynamics posterior sampler:

Eq. 4: `z^{τ+1} = z^τ + s∇_z log p_β(z^τ|x) + √(2s)ε^τ`

Ablations found Langevin gave "lower generation quality than VB" (Section 3.4), so the shipped
inference is gradient-based coordinate ascent on the variational parameters, reparameterized
sampling from q, not raw Langevin on z itself.

## 3. Training objective and dual-rate optimization

Generative model: `p_β(x) = ∫ p_β(x|z) p(z) dz` (Eq. 2), with
`p_β(x|z) = ∏_{n=1}^N p_β(x^{(n)} | z, x^{(<n)})` (Eq. 1) — an ordinary autoregressive decoder
additionally conditioned on z via cross-attention at every layer.

ELBO, per sequence (Eq. 5):
`L(β, μ, σ²) = E_{q(z|x)}[log p_β(x|z)] − D_KL(q(z|x) ‖ p(z))`, with `q(z|x) = N(μ, σ²)`
(diagonal covariance), z sampled via the reparameterization trick.

Algorithm 1 ("Fast-Slow Learning of LTM"), verbatim structure:
1. Sample a mini-batch {x_i}.
2. **Fast loop**, per example, T_fast steps: initialize μ_i, σ_i²; each step samples
   z ~ q_{μ_i,σ_i²}(z|x_i), computes L_i (Eq. 5), and updates μ_i, σ_i² with Adam at η_fast.
3. **Slow step**: average L_i over the batch, update decoder weights β with AdamW at η_slow.

Numbers given: T_fast = 16 for Small/Medium, T_fast = 64 for Large. η_fast is high and itself
scheduled, "linearly increasing from 0.3 to 0.34" (order-of-magnitude example given in text as
"e.g., 0.3"). η_slow = 4×10⁻⁴ with cosine decay. The KL term against the fixed N(0,I) prior is
what keeps z from being an unconstrained per-example code — there is no separate free-bits or
KL-annealing schedule reported for LTM itself (annealing is mentioned only for the VAE
baseline, which still collapsed — see §6).

`tfpt` (training compute per token, Section 2.4), given as an explicit cost formula, not a
single number:
`tfpt = O((T_fast+1_slow)·L·(N²H + N·N_z·H + N·H²) + (T_fast+1_slow)·N·V·H)`
— i.e. compute scales with T_fast because every fast step is a full forward+backward through
the decoder to update z, before one slow step updates β. This is the paper's stated reason LTM
has an extra, controllable compute axis that plain autoregressive models do not.

## 4. What z sees at test time — precise

Two distinct regimes, and the paper is explicit that they differ:

- **Perplexity/ELBO evaluation** (scoring a fixed sequence x): the SAME classical-VB fast loop
  as training is run on x, i.e. μ, σ² (hence z) are fit BY GRADIENT ASCENT DIRECTLY ON THE
  SEQUENCE BEING SCORED, exactly as in training. This is why the paper reports "perplexity
  upper bounds" (stated in the abstract, echoed for the diffusion-model comparisons too) — the
  ELBO is a variational upper bound on −log p(x), the same relationship as a VAE's
  reconstruction+KL bound. **z is not blind to the tokens whose likelihood is being reported at
  eval time; it is fit on them, same as at train time.** The fetch could not surface a sentence
  describing a held-out-continuation perplexity protocol distinct from this — if one exists it
  is not stated in the text the fetch returned.
- **Generation**: this is where the paper draws a real train/test information boundary.
  Conditional generation infers z ONLY from the prefix x, then decodes the continuation y from
  that fixed z: `p_β(y|x) = ∫ p(z|x) p_β(y|x,z) dz = E_{p(z|x)}[p_β(y|x,z)]` (Eq. 6, the
  "principled" form), approximated in practice by
  `p_β(y|x) ≈ E_{q(z|x)}[p_β(y|x,z)]` (Eq. 7) — z is fit with the same fast-loop machinery but
  using ONLY x (the prompt), never y (the continuation). Unconditional generation instead draws
  z straight from the prior: `p_β(x) = E_{p(z)}[p_β(x|z)]` (Eq. 8). The paper notes, as future
  work not the shipped method, "an alternative sampling scheme is to incorporate each newly
  generated token into the prefix and then updating z through variational inference" —
  i.e. re-fitting z online as generation proceeds, explicitly flagged as "more computationally
  intensive" and not what the reported generation results use.

So: generation-time z is causal (prefix-only). Eval-time PPL z is not causal with respect to
the scored span — it is a reconstruction-style upper bound, the same convention VAEs use.

## 5. The three scaling axes

Axes named: model size, training compute per token (tfpt, driven mostly by T_fast), and number
of latent vectors N_z. Configurations: LTM-Small (38M params, T_fast=16, N_z=24, tfpt=4.07G),
LTM-Medium (51M, T_fast=16, N_z=24, tfpt=5.52G), LTM-Large (76M, T_fast=64, N_z=96,
tfpt=32.2G). Qualitative claim (Section 3.2 / Fig. 4): "models with more inference steps
achieve greater sample efficiency and become more compute-efficient beyond certain thresholds."
The HTML fetch could not recover Figure 4's plotted per-step perplexity values as a table (it
is a figure, not prose text) — only the configuration list above and the qualitative trend
were recoverable this way; treat exact PPL-vs-T_fast and PPL-vs-N_z curve values as **not
recoverable from this fetch** and, if needed exactly, get them from the PDF figure directly.

## 6. Collapse / decoder ignoring z

The paper ran a VAE baseline (amortized encoder network instead of per-example VB) and reports
outright failure: "we observe severe posterior collapse, even with careful annealing on the
KL-divergence term in ELBO" (Section 3.4), with symptoms "repetitive prior samples and low
entropy distributions." Their diagnosis for why: an amortized encoder "is more likely than the
classical variational Bayes to take the easy route and only minimize the KL term," i.e. the
encoder network finds it cheaper to output something close to the prior (ignoring x) than to
produce an informative z, especially against a strong autoregressive decoder that can already
get x^{(<n)} for free. Classical VB with local, per-example (μ_i, σ_i²) avoids this because
"the initial mismatch between the inference model and the true posterior" that plagues a
freshly-initialized encoder network does not apply — there is no shared encoder to fail to
learn; each example gets its own from-scratch gradient-ascent fit every time. The paper does
not report a measured KL value or an explicit "rate" (bits used by z) — the collapse discussion
is qualitative (VAE fails, VB does not), not an ablation curve of KL vs. training step.

## 7. Results vs. AR baselines, and what didn't work

Table 1: at matched tfpt, LTM beats GPT-2 at the same/larger parameter count on zero-shot
perplexity across PTB, WikiText, LM1B, LAMBADA, AG News, PubMed, Arxiv. LTM-Medium (51M) is
reported to roughly match GPT-2-Large (utilizing "6.7% of GPT-2-Large parameters"), and
LTM-Large's reported reductions are "52.2% and 91.7% reductions in perplexity compared to
state-of-the-art results at GPT-2 scale" (exact baselines for each of those two numbers were
not resolved by the fetch — treat as a headline claim to re-verify against the actual table
rows before citing). Table 2 (generation quality, MAUVE): LTM-Large scores 0.974
(multinomial)/0.972 (greedy) vs. SEDD (a diffusion baseline) at 0.957 and GPT-2-Medium nucleus
sampling at 0.955. What did NOT work: the VAE/amortized-encoder variant (posterior collapse,
§6 above) and Langevin-dynamics posterior sampling (worse generation quality than the VB fast
loop, Section 3.4).

## 8. Multimodality, sampling vs. point estimate, diversity

z is a distribution, not a point estimate: q(z|x) = N(μ, σ²) with a learned diagonal σ², and
sampling uses the reparameterization trick both in training and at eval/generation time, so
there is per-example uncertainty rather than a deterministic code. The paper does not discuss
whether q(z|x) is multimodal (it is Gaussian by construction, so no — multimodality of the true
posterior is not modeled) and offers no explicit diversity metric for z itself; the only
diversity-adjacent evidence is the MAUVE generation-quality numbers in Table 2, which are about
text quality/diversity, not latent-space diversity.

## What z is in LTM, in three sentences

z is a set of per-layer Gaussian latent vectors, cross-attended into every decoder layer as
keys/values, fit fresh for each sequence by a from-scratch gradient-ascent loop on the ELBO
rather than by any learned encoder network. The KL term against a fixed N(0,I) prior is the
only thing stopping z from becoming an unconstrained code, and the paper's central empirical
claim is that this per-example optimization (not an amortized network) is what keeps the
decoder from ignoring it, where the amortized-encoder version collapsed outright. At test time
z is fit causally (prefix only) for generation but fit on the full scored sequence, VAE-style,
when the number reported is a perplexity upper bound — these are two different protocols under
one name.

## Open questions the paper does not answer

- No measured KL/rate curve for the shipped VB model — collapse is discussed only as a
  contrast against the VAE baseline, not as an ablation on VB itself under a weak or strong
  decoder.
- No stated free-bits or KL-warmup schedule for the shipped model (only mentioned for the
  failed VAE baseline), so it's unclear how tightly the KL term is actually enforced.
- Figure 4's exact per-step and per-N_z perplexity curve values were not recoverable from the
  HTML fetch (figure, not prose) — get these from the PDF directly if the precise shape of the
  scaling curve matters.
- No discussion of what happens with a much stronger/deeper decoder relative to N_z and
  T_fast — i.e. no stated threshold at which the decoder is expected to start ignoring z again
  (the VAE-vs-VB comparison is the only lever tested, not decoder capacity vs. z capacity).
- No explicit ICL/GSM8K numeric table was resolved from the fetch (pass@5 on a 1K test set is
  named, exact accuracy numbers were not) — re-check Section 3.3 / Figure 5 directly for exact
  values before citing.
- No description of whether/how the per-example local parameters (μ_i, σ_i²) are cached,
  reused, or discarded between epochs — each fast loop is described as re-initialized, but the
  paper does not state the initialization scheme in the text recovered.
