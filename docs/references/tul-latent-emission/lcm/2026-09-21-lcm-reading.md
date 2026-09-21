> Reading filed 2026-09-21, against the state of the TUL slot-loop work on that date (the `code_target`
> regression arms, the vq8/vq4 discrete-thought arms, and the fan4 multi-stream arms). Produced by a
> full read of two sources: the repo's markdown dump of the arXiv HTML
> ([`lcm.md`](lcm.md), 1173 lines) and a `pdftotext -layout` render of the v2 PDF archived at
> `ignore/papers/2412.08821v2-large-concept-models.pdf`. Every table in this note is quoted from the
> PDF render, because the HTML dump interleaves LaTeX markup into the table cells. Tables 3, 4 and 7
> were spot-checked cell-for-cell across both sources and agree exactly. Figures are read off the
> plotted axes only where the body text gives no number, and those lines say so.

# Large Concept Models: Language Modeling in a Sentence Representation Space

arXiv 2412.08821v2 (15 Dec 2024). LCM team, FAIR at Meta: Loïc Barrault, Paul-Ambroise Duquenne,
Maha Elbayad, Artyom Kozhevnikov (core, alphabetical) and 17 others. Code at
`github.com/facebookresearch/large_concept_model`.

## 1. The unit: what a "concept" is, and how sentences are cut

A concept is defined as "an abstract atomic idea", language- and modality-agnostic (§1). For the
paper it is operationalised as **one sentence**, embedded by the **frozen** SONAR encoder to a single
**1024-dimensional** vector (Appendix A gives the width; Figure 1 marks encoder and decoder frozen
with a star). SONAR covers 200 text languages, 76 speech-input languages and English speech output
(Table 1). SONAR itself was trained as an encoder/decoder with a fixed-size bottleneck instead of
cross-attention, on a criterion combining 200-language machine translation, denoising auto-encoding
and an explicit MSE loss at the bottleneck (§2.1). Nothing in the LCM trains it.

**Segmentation** (§2.2). Two segmenters compared, SpaCy (rule-based) and SaT / Segment any Text
(token-level boundary prediction), each also in a **capped** variant that breaks long sentences at the
next best split point. The quality metric is **AutoBLEU**: BLEU between the reference segment and the
text you get by encoding that segment to SONAR and decoding it back. Study size: 10k documents,
approximately 500k sentences (§2.2).

The fidelity result, Figure 3: with a cap at **200 characters**, SaT Capped is slightly but
consistently ahead of SpaCy Capped. Both uncapped segmenters underperform at every length, and the
body text names the knee explicitly: the underperformance "is especially pronounced for sentences
exceeding **250 characters**". Reading the left panel of Figure 3, uncapped average AutoBLEU falls
from roughly 0.90 near 100 characters to roughly 0.65–0.70 by 400–500 characters; the capped panel
(right) stays in the 0.85–0.93 band out to its 200-character limit. The 0.90/0.65 pair is read off
the plot, not printed in the text.

The same 250-character wall reappears from the other direction in the fragility study (§2.5.2,
Figure 14 left panel): AutoBLEU itself "drops only by 1–2% for long sentences" but the *fragility*
score falls much faster with length, and the authors conclude "using a max sentence length over 250
can be **extremely challenging** for SONAR and the LCM model". So the ~250-character cap is a
fidelity finding twice over: once in raw reconstruction, once under perturbation.

Training-corpus scale and shape (Appendix A): about **4 billion documents → 310 billion sentences**,
averaging **27 tokens and 88 characters per sentence**, from a bit over 889 TB of raw text. Storage
cost of the frozen-encoder choice: 1 TB of raw text becomes 15–20 TB of fp16 SONAR embeddings, but
loading precomputed embeddings runs at over 20k embeddings/s/GPU against 300–400/s to encode on the
fly.

Chosen recipe: **SaT Capped** (§2.2).

**Not stated**: any per-length AutoBLEU table (Figure 3 is a plot only). **Not stated**: what cap
value in characters was actually used to build the pre-training corpus — the text says "capping at
200 characters" for the *analysis* and then "we prepare the LCM training data with SaT Capped"
without repeating the number.

## 2. Base-LCM: the MSE regression baseline, and every number that kills it

**Architecture** (§2.3.1, Figure 4). A plain decoder-only Transformer over concept positions, wrapped
in a `PreNet` and a `PostNet`:

- `PreNet(x) = normalize(x) W_pre^t + b_pre` maps SONAR-dim → `d_model` (Equation 1).
- `PostNet(x) = denormalize(x W_post^t + b_post)` maps back (Equations 2–3).
- `normalize(x) = (x − µ)/σ`, `denormalize(x) = µ + σx` (Equation 4). µ and σ come from a **robust
  scaler** fitted on randomly sampled SONAR vectors across corpora: it removes the median and scales
  by the interquartile range. This is a fixed, data-fitted affine map, not a learned norm.

Size and hyper-parameters (§2.4.1): 32 layers, `d_model` = 2048, 16 heads, RoPE, pre-norm RMSNorm,
SwiGLU, dropout 0.1, approximately 1.6B trainable parameters, 250k steps on 32 A100s with total batch
229k concepts, on Fineweb-edu, documents wrapped at 128 sentences.

**Loss** (Equations 5–6): pure MSE onto the next ground-truth SONAR vector,
`L = E_x [ Σ_n ‖f(x_<n; θ) − x_n‖² ]`. No sampling, no contrastive term, no token-level term.

Stopping (§2.3.1): documents are suffixed with the encoded sentence "End of text."; generation stops
when cosine to that vector, or cosine to the previous generated vector, exceeds a threshold. Both
thresholds are **0.9**.

**The averaging argument, stated by the authors** (§2.4.1, discussion of Table 3): "when many
plausible next sentence continuations are possible, Base-LCM generates their **average** in SONAR
space (instead of sampling one of plausible modes) which may not correspond to any relevant point in
SONAR space." The motivation section says the same thing up front (§2.3): "a given context may have
many plausible, yet semantically different, continuations", so the model must learn a conditional
*distribution*, not a point.

**Metric definitions** (§2.4.1) — these are the instruments, and they are the reason the argument is
provable rather than rhetorical:

- **ℓ2**: `‖x̂_n − x_n‖₂` in SONAR space. Exactly the training loss of Base-LCM.
- **ℓ2-r** (round-trip): `‖encode(decode(x̂_n)) − x_n‖₂`. Decode the prediction to text, re-encode it,
  then measure. The gap `ℓ2-r − ℓ2` is the out-of-distribution meter: the decoder pulls an
  off-manifold prediction to the nearest plausible embedding, and the size of that pull is the
  evidence that the prediction was not a real sentence.
- **CA** (contrastive accuracy): "the ratio of embeddings in a batch that are further away (in terms
  of ℓ2) from the predicted embedding x̂_n than the ground truth x_n", with x_n and its **two
  neighbouring** ground-truth embeddings excluded from the comparison pool. It is a retrieval
  accuracy against in-batch negatives, and the authors note it "naturally assigns higher penalty for
  large ℓ2 values in the regions with high density of sentence embeddings" — it is scale-free where
  raw ℓ2 is not.
- **PAR** (paraphrasing): `max_{m<n} CS(x̂_n, x_m) / max_{m<n} CS(x_n, x_m)`. Above 1 means the
  prediction copies or paraphrases the context more than the reference sentence does.
- **MI** (mutual information): `MI = (1/|ŝ_n|) · (log p_LM(ŝ_n) − log p_LM(ŝ_n | s_{n−k:n−1}))` with
  **k = 10** previous ground-truth sentences, `p_LM` estimated by **GPT-2**, a newline prepended so
  the first token gets a probability, normalised per token and length-weighted when averaged over a
  dataset. This is a *context-usage* meter: how much cheaper the generated sentence becomes once the
  scorer sees the ten sentences before it.

**Table 3** (pre-training evaluation, teacher-forced next-concept prediction; test splits of
ROC-stories, C4, Wikipedia-en, Gutenberg; dataset statistics in Table 2):

| Model | ROC ℓ2 | ROC ℓ2-r | ROC PAR | ROC CA | ROC MI | C4 ℓ2 | C4 ℓ2-r | C4 PAR | C4 CA | C4 MI |
|---|---|---|---|---|---|---|---|---|---|---|
| Base-LCM | **0.177** | 0.237 | 1.847 | 72.4% | 0.062 | **0.204** | 0.261 | 1.964 | 69.1% | **−0.105** |
| One-Tower | 0.236 | 0.236 | 1.939 | 80.2% | 0.977 | 0.279 | 0.273 | 2.239 | 77.1% | 1.110 |
| Two-Tower | 0.233 | 0.231 | 2.088 | 80.6% | **1.137** | 0.265 | 0.261 | 2.265 | 75.4% | **1.134** |
| Quant-LCM-c | 0.236 | 0.237 | 1.683 | 76.0% | 0.610 | 0.279 | 0.283 | 2.013 | 77.2% | 0.715 |
| Quant-LCM-d | 0.240 | 0.246 | 1.871 | **81.1%** | 0.682 | 0.270 | 0.282 | 1.808 | 75.0% | 0.359 |

| Model | Wiki ℓ2 | Wiki ℓ2-r | Wiki PAR | Wiki CA | Wiki MI | Gut ℓ2 | Gut ℓ2-r | Gut PAR | Gut CA | Gut MI |
|---|---|---|---|---|---|---|---|---|---|---|
| Base-LCM | **0.229** | 0.283 | 1.770 | 69.6% | 0.071 | **0.207** | 0.264 | 1.780 | 67.8% | **−0.184** |
| One-Tower | 0.324 | 0.311 | 2.087 | **80.9%** | 1.202 | 0.284 | 0.281 | 2.051 | **75.1%** | **0.725** |
| Two-Tower | 0.307 | 0.297 | 2.079 | 78.8% | **1.307** | 0.267 | 0.267 | 2.077 | 73.0% | 0.684 |
| Quant-LCM-c | 0.306 | 0.317 | 1.842 | 79.5% | 0.744 | 0.269 | 0.281 | 1.774 | 72.1% | 0.419 |
| Quant-LCM-d | 0.295 | 0.311 | 1.592 | 76.0% | 0.323 | 0.276 | 0.290 | 1.599 | 72.0% | 0.153 |

Read the first column against the last. Base-LCM wins ℓ2 on **all four** corpora and is the only model
that does, "expected since Base-LCM effectively optimizes ℓ2 during training" (§2.4.1). It
simultaneously loses **every other metric on every corpus**:

- **ℓ2-r does not improve with ℓ2.** On ROC-stories, Base-LCM's ℓ2 is 0.177 against One-Tower's
  0.236 — a 25% better regression — yet their round-trips are 0.237 and 0.236, a tie. Read the *gap*
  `ℓ2-r − ℓ2` rather than the level. Base-LCM: **+0.060, +0.057, +0.054, +0.057** on ROC, C4,
  Wikipedia-en and Gutenberg. Two-Tower: **−0.002, −0.004, −0.010, 0.000** on the same four. The
  MSE model's prediction moves when you decode it and encode it back, by about a quarter of its own
  ℓ2; the diffusion model's does not move at all. That gap is the direct evidence that the regressed
  point is **off the manifold of real sentences** — which is what an average of several modes is.
- **MI collapses to zero or below.** 0.062 / −0.105 / 0.071 / −0.184 against 0.7–1.3 for the
  diffusion models. A negative MI means the generated sentence is *less* likely under GPT-2 once the
  ten preceding sentences are shown than it is unconditionally. The regression target is met and the
  output carries no usable dependence on the context.
- **CA is 3–8 points below every alternative** (72.4 / 69.1 / 69.6 / 67.8 against 72.0–81.1).

**Table 4** (instruction tuning on the stories subset of Cosmopedia, scored on a held-out test split;
`smaLlama` is a 1.4B token-level Llama, 24 layers, 16 heads, d=2048, trained on the same Fineweb-edu
and finetuned on the same Cosmopedia; Coherence is the reference-free Jwalapuram et al. 2022
classifier, sigmoid-normalised at temperature 3.0):

| Model | R-L ↑ | Coherence ↑ |
|---|---|---|
| Base-LCM | 23.69 | 0.482 |
| One-Tower | 33.40 | 0.968 |
| Two-Tower | 33.64 | 0.938 |
| Quant-LCM-c | 30.87 | 0.847 |
| Quant-LCM-d | 28.01 | 0.704 |
| smaLlama | **34.88** | **0.984** |

Base-LCM is **9.7 ROUGE-L points** and **0.486 coherence** below One-Tower, and 11.2 / 0.502 below the
matched token-level Llama. Coherence 0.482 on a scale calibrated so "certainly incoherent" is near 0
and "certainly coherent" near 1 means the MSE model's output is a coin flip. The authors note the
ordering of R-L and Coherence "correlate with the model ordering based from MI scores" — the cheap
pre-training instrument predicted the expensive downstream one.

The conclusion (§8) states it without hedging: "we have first shown that directly minimizing the MSE
loss in the embedding space does not yield good results."

**Not stated**: any perplexity of the decoded text. The authors say explicitly that LCMs "cannot
produce the probability explicitly" and that this is why they built the ℓ2/CA/MI battery (§2.4.1). MI
uses GPT-2 log-probabilities but is reported as a *difference* of per-token log-probs, never as an
absolute PPL. **Not stated**: any measurement of Base-LCM's prediction *rank* or effective
dimensionality — the "average in SONAR space" claim is argued from the ℓ2/ℓ2-r/CA/MI pattern, not
from a spectrum.

## 3. Diffusion LCMs: One-Tower and Two-Tower

Both are **auto-regressive over concepts** and diffusive **within** one concept: `p_θ(x_n | x_<n)`,
one sentence at a time, each sentence produced by iteratively denoising a fresh Gaussian draw (§2.3.2).

**Forward process** (Equations 7–9): variance-preserving Gaussian, `x^t = α_t x^0 + σ_t ε`, with
`α_t² = sigmoid(λ_t)`, `σ_t² = sigmoid(−λ_t)`, `λ_t = log(α_t²/σ_t²)` the log-SNR.

**Prediction target**: **x0-prediction**, not ε-prediction — "we predict the noiseless state and
optimize the simple reconstruction loss" (Equation 16),
`L(t,θ) = E‖x⁰ − µ_θ(α_t x⁰ + σ_t ε, t)‖²`, with `L(θ) = E_{t~U(0,1)}[ω(t) L(t,θ)]`. Default
`ω(t) = 1 ∀t`. Note what this means: the per-step objective is **MSE onto the clean target**, the same
loss Base-LCM fails with. The difference is not the loss, it is that the input is a *noised* sample
at a *randomly drawn* noise level, so the model learns a family of denoisers rather than one
conditional mean.

**Noise schedules** (§2.3.2, Figure 5): cosine (default, Nichol and Dhariwal with s = 0.008,
Equation 12); quadratic (Equation 13), Quadratic-1 with (β0, βT) = (0.001, 0.0012) and Quadratic-2
with (0.02, 0.022); and a **sigmoid schedule introduced in this paper** (Equation 14),
`α_t² = f(t)/f(0)`, `f(t) = sigmoid(δ − γ logit(t))`, where γ sets the *scale* of the log-SNR
distribution p(λ) and δ its *centre*. All schedules are rescaled to enforce zero terminal SNR
(`β_T = 1`, following Lin et al. 2024). All use **T = 100** training timesteps.

**Classifier-free guidance** (Equation 20): `∇_x log_γ p(x|y) = (1−γ) ∇_x log p(x) + γ ∇_x log p(x|y)`,
with y the preceding embeddings. Unconditional training is achieved by dropping self-attention with
probability **0.15** (One-Tower) or dropping cross-attention mask rows at rate `p_cfg` = 0.15
(Two-Tower).

**Inference** (§2.3.2). Start from `x^T ~ N(0, σ_init² I)` — "the quality of the sampled output is
sensitive to the initial noise scale". Train on **T = 100** discretized timesteps, generate with
**S = 40**, selecting steps by the *trailing* method of Lu et al. 2022. Guidance rescaling of Lin et
al. 2024, and epsilon-scaling (Ning et al. 2023) by a scalar `λ_eps` to fight exposure bias. Defaults
used throughout: **S = 40, g_scale = 3, g_rescale = 0.7, σ_init = 0.6, λ_eps = 1.00045** (§2.4.1).

**One-Tower** (§2.3.3, Figures 6-left and 7): one 32-block transformer, 32 heads, FFN inner size
8192, `d_model` = 2048, learned position embeddings, causal multi-head self-attention. Each input
embedding is concatenated with its diffusion-timestep embedding. Training trick: the sequence
**interleaves noisy and clean** embeddings with an attention mask that lets the noisy position attend
only to clean context, so every sentence in a document is denoised in one forward pass, each at its
own independently sampled timestep.

**Two-Tower** (§2.3.4, Figures 6-right and 8): a **contextualizer** (5 layers, causal self-attention,
RoPE) encodes the clean context; a **denoiser** (13 layers, no positional embeddings) cross-attends
to it and iteratively denoises `x_n`. Shared `d_model` = 2048, 16 heads, SwiGLU, RMSNorm, dropout 0.1.
Every denoiser block, cross-attention included, is modulated by **AdaLN**: `[β, γ, α] = SiLU(embed(t))
W_t + b`, `y = x + α · Block((1+γ) x + β)` (Equations 21–22), with W and b zero-initialised so each
residual block starts as the identity. The timestep is a 256-dimensional frequency embedding through
a two-layer SiLU MLP. The denoiser's **self-attention attends only to the current position** — it does
not see the noised context — and is retained only for architectural consistency and possible future
multi-vector denoising. Context is shifted by one and a zero vector is prepended so position 0 is
predictable.

**Ablation verdicts.**

- Architecture (Table 3, Table 4): "to tackle the next sentence prediction task in the SONAR space,
  diffusion-based methods give clearly better results compared to all other models". Between
  One-Tower and Two-Tower there is **no consistent difference** across metrics and datasets. The 7B
  scale-up chose Two-Tower purely for memory footprint, because of the shallower contextualizer
  (§3).
- Inference hyper-parameters (§2.4.2, Figure 10, Two-Tower on C4). Raising `g_scale` raises MI **and**
  PAR (more paraphrase of the prefix) while ℓ2 from the ground-truth continuation **increases** —
  guidance buys context-dependence at the cost of distance to the true next sentence. `σ_init`
  between **0.5 and 0.7** gives the best MI; higher `σ_init` makes individual generated sentences
  longer, and ℓ2 does not reflect this trend at all.
- **Number of denoising steps** (Figure 10, third panel, `g_scale` = 1.5, `σ_init` = 0.6, S swept from
  about 20 to about 80 at three `λ_eps` values): "With more inference steps, we can improve the
  prefix-suffix mutual information, but there is **diminishing returns** to increasing the inference
  cost with little qualitative improvement." Reading the panel, MI rises from roughly 0.7–0.8 near
  S = 20 to roughly 1.1–1.2 near S = 80, with the three `λ_eps` curves separating mainly in the
  large-S regime. Those two MI values are read off the plot; the body text gives **no numeric
  step-count table**. So: more steps help monotonically on MI, and the paper still ships S = 40.
- Noise schedules (§2.4.3, Table 5, Two-Tower, C4 and Wikipedia-en):

| Model | C4 ℓ2 | C4 ℓ2-r | C4 PAR | C4 CA | C4 MI | Wiki ℓ2 | Wiki ℓ2-r | Wiki PAR | Wiki CA | Wiki MI |
|---|---|---|---|---|---|---|---|---|---|---|
| Cosine | 0.265 | 0.261 | 2.265 | 75.4% | 1.134 | 0.307 | 0.297 | 2.079 | 78.8% | 1.307 |
| Quadratic-1 | 0.268 | 0.264 | 2.341 | 75.7% | **1.252** | 0.309 | 0.300 | 2.202 | 79.1% | **1.409** |
| Quadratic-2 | 0.270 | 0.265 | 2.320 | 76.2% | **1.252** | 0.312 | 0.303 | 2.185 | 79.7% | 1.399 |
| Sigmoid(1.5, −1) | 0.257 | 0.259 | 2.226 | 74% | 1.083 | 0.298 | 0.292 | 2.110 | 77% | 1.271 |
| Sigmoid(1.5, −2) | 0.277 | 0.267 | 2.291 | 77.2% | 1.173 | 0.321 | 0.303 | 2.179 | 80.3% | 1.308 |
| Sigmoid(0.8, −1) | **0.252** | **0.255** | 2.053 | **70.6%** | **0.936** | **0.285** | **0.283** | 1.883 | **71.7%** | **1.127** |
| Sigmoid(3.5, 0) | 0.307 | 0.265 | 2.347 | **80.3%** | 1.154 | 0.347 | 0.303 | 2.187 | **83.7%** | 1.288 |

  The two extreme sigmoids are the load-bearing rows and they are a clean controlled repeat of the
  Base-LCM lesson **inside the diffusion family**. The **peaked** schedule (δ=0.8, γ=−1), which
  concentrates training on a narrow band of log-SNR near −1, gets the **best ℓ2 in the table**
  (0.252 / 0.285) and the **worst CA** (70.6% / 71.7%) and the **worst MI** (0.936 / 1.127) — the
  authors say it "focuses on regressing to the target (lower ℓ2), **akin to a Base-LCM**". The
  **wide** schedule (δ=3.5, γ=0), trained on a flat spread of noise levels, gets the **worst ℓ2**
  (0.307 / 0.347) and the **best CA** (80.3% / 83.7%), because "trained on a wider spectrum of
  log-SNR [it] learns to **contrast** more than to **regress**". Figure 11 confirms the CA advantage
  holds across every guidance scale while MI tracks cosine.
- Loss weighting (§2.4.4, Table 6, Two-Tower, cosine schedule). Clamped-SNR weighting (Equation 17)
  at (λmin, λmax) = (0, 10) and (0.001, 5) does **not** help: CA 74.8% and 73.4% against the baseline's
  75.4% on C4, MI 1.107 and 1.094 against 1.134. Fragility weighting (Equations 18–19, 24–25),
  `ω(x⁰) = sigmoid(a·F(x) + b)` with a 3-layer MLP F trained on 50M sentences to approximate the
  fragility score and **a = −4, b = 3.5** (Figure 12), is the only one that moves anything: **CA
  76.5% on C4 (+1.1) and 80.7% on Wikipedia-en (+1.9)**, at the cost of worse ℓ2 (0.2815 / 0.321) and
  worse MI (1.103 / 1.193). The paper defaults back to `ω(t) = 1` anyway.

**Inference cost** (§2.5.1, Figure 13): theoretical FLOPs for Two-Tower at S = 40 against Llama2-7B.
The LCM scales substantially better with total context because it runs on a sequence at least an
order of magnitude shorter (footnote 2 assumes 10–20 tokens per sentence). The crossover is named:
"For extremely short sentences (**less than 10 tokens**), an LLM is more computationally efficient."

**Not stated**: any ablation of S at *training* time, or of T. **Not stated**: a step-count table.
**Not stated**: any result for denoising more than one concept at a time — the Two-Tower denoiser
keeps its self-attention layers explicitly "for the possible extension of denoising multiple vectors
at once", which is future work.

## 4. Quantized LCM: Quant-LCM-d and Quant-LCM-c

This is the closest structural analogue to the MORPH vq8/vq4 arms, so here is everything the paper
says.

**Motivation** (§2.3.5): "all possible text sentences (of less than a given number of characters) are
a **cloud of points rather than a real continuous distribution** in the SONAR space." Quantization is
pursued for two reasons: it matches that discreteness, and it "enables the natural use of temperature,
top-p or top-k sampling, to control the level of randomness and diversity". The authors state they
"tried to come up with an architecture as close as the diffusion LCM models, to be able to compare
approaches" — this is a controlled swap, not a different system.

**The quantizer** (§2.3.5). Residual Vector Quantization (RVQ, Zeghidour et al. 2021): quantize, take
the residual, quantize the residual with a fresh codebook, repeat. Implementation is FAISS (Douze et
al. 2024) doing iterative k-means on residuals, using Improved RVQ (IRVQ, Liu et al. 2015) with
**beam size 1** for memory. Trained on **15 million English sentences** from Common Crawl with
**n_codebooks = 64** and **n_units_per_codebook = 8192**. So one sentence is **64 codes drawn from 64
separate 8192-entry codebooks**, coarse to fine.

**How good is that code** (Figure 9, AutoBLEU on the FLORES devtest set, sweeping the number of
codebooks used from about 5 to 64): AutoBLEU "consistently improves as the number of codebooks
increases, reaching around **70%** of the auto-encoding BLEU score achieved with continuous SONAR
embeddings, when using **all 64 codebooks**." The 70% figure is the body text's and it describes the
**base** decoder, since the paragraph precedes the decoder fine-tuning. Reading the plot, the
unquantized topline sits near 80 AutoBLEU and the base-decoder curve reaches roughly 55 at 64
codebooks, which is consistent with 70%. The finetuned-decoder curve runs visibly above the base one
at every codebook count; I will not put a number on it from the plot. Only the 70% is a quoted
value; the rest of this paragraph is read off Figure 9.

That is the headline for a discrete-thought arm: **64 codebooks × 8192 units — 2^768 nominal
combinations — still loses 30% of the auto-encoding BLEU.**

**Decoder adaptation** (§2.3.5). The SONAR decoder is fine-tuned on quantized representations over
**1.2M English sentences**. To make it robust to *partial* codes, training randomly picks a codebook
count `k ∈ [⅔ · n_codebooks, n_codebooks]` with probability **p = 0.3** and feeds the cumulative sum
up to k. Figure 9 shows the resulting lift. This matters: the reader is *retrained on the code
distribution it will actually see*, including truncated codes.

**Quant-LCM architecture** (§2.3.5). Built on **One-Tower**, unchanged except that the noisy input is
replaced by the **intermediate quantized representation** (the running cumulative sum of centroid
embeddings of the first k−1 codebooks) and the diffusion-timestep embedding is replaced by the
**codebook-index embedding**. Generation starts from the **zero vector** and iteratively adds predicted
residual centroid embeddings until all codebooks have been consumed. The number of refinement
iterations is therefore fixed by the codebook count, and each iteration has a **defined target**: the
residual that the next codebook is supposed to explain.

**Quant-LCM-d (discrete targets).** A softmax over **8192** outputs — not 64 × 8192 — with the
codebook index fed to the model as input instead, for parameter efficiency. Training samples a
codebook index `k ∈ [1, n_codebooks]` uniformly, builds the cumulative sum of the first k−1 centroids
as input, and uses the true unit from codebook k as the cross-entropy target. Inference predicts the
next codebook's unit left-to-right, adds its centroid, repeats. Classifier-free guidance **on logits**
(Gafni et al. 2022) via left-context dropout at training time. Uses the improved quantization-adapted
decoder in all ablations.

**Quant-LCM-c (continuous targets).** Same inputs, but the output is a continuous SONAR vector
trained with **MSE** against the target embedding, given the left context *and* the intermediate
quantized representation. At inference you either add the nearest centroid to the predicted residual
`r̂`, or sample one with temperature β (Equation 23): `p(c_i | r̂) = exp(−β‖c_i − r̂‖₂) / Σ_k
exp(−β‖c_k − r̂‖₂)`.

**Sampling settings** (§2.4.1). Single-sentence prediction: `top_k = 1, g_scale = 2` for Quant-LCM-d;
`top_k = 1, g_scale = 3` for Quant-LCM-c. Multi-sentence generation: `temperature = 1, top_k = 3,
g_scale = 1` for Quant-LCM-d; `temperature = 0.005, top_k = 5, g_scale = 1.5` for Quant-LCM-c,
"as higher guidance or lower temperature setups led to **repeated generated sentences**". Note the
asymmetry: the continuous variant needs temperature 0.005, effectively deterministic, to avoid
collapse into repetition, while the discrete variant runs at temperature 1.

**Results.** From Table 3 and Table 4 above, three orderings, all consistent:

1. **Both Quant variants beat Base-LCM by a large margin** on MI (C4: 0.715 and 0.359 against
   −0.105; Gutenberg: 0.419 and 0.153 against −0.184), on ROUGE-L (30.87 and 28.01 against 23.69) and
   on coherence (0.847 and 0.704 against 0.482). Quantizing and predicting codes is a real fix for
   the conditional-mean failure.
2. **Both Quant variants lose to diffusion** on MI at every corpus (best Quant 0.744 on
   Wikipedia-en against Two-Tower's 1.307) and on the downstream pair (30.87/0.847 against
   33.64/0.938). CA is the exception: "We do not notice any significant difference in CA scores
   between diffusion LCMs and Quant-LCM variants" — Quant-LCM-d even posts the single best CA in
   Table 3 (81.1% on ROC-stories).
3. **Quant-LCM-c beats Quant-LCM-d** on MI at three of the four corpora (C4 0.715 against 0.359,
   Wikipedia-en 0.744 against 0.323, Gutenberg 0.419 against 0.153; ROC-stories inverts, 0.610
   against 0.682) and on the
   downstream pair (30.87/0.847 against 28.01/0.704). The authors' hypothesis: "predicting codebook
   indices with cross-entropy loss is **harder** than MSE objective where Quant-LCM-c can more easily
   learn combination of left-context vectors for next sentence embedding." So within the quantized
   family, the **continuous** regression on the residual wins — and that is not a contradiction of
   §2 because the residual target is *conditioned on the partial code*, so it is no longer a
   marginal conditional mean.

**The authors' own post-mortem on quantization** (§6): "The limited performance of the Quant-LCM
approaches presented in this paper may be explained by the fact that **SONAR space was not trained to
be efficiently quantizable**, yielding a significant number of codebooks and a large amount of units
per codebook. Therefore, the current SONAR quantization suffers from the exponentially increasing
number of RVQ units combinations which **does not solve the data sparsity/uniqueness issue**."

**Not stated**: any sweep of `n_codebooks` or `n_units_per_codebook` as a *model* ablation — Figure 9
sweeps codebook count for reconstruction only, never for LCM quality. **Not stated**: a Quant-LCM at
7B; only Two-Tower was scaled (§3). **Not stated**: codebook usage, perplexity or dead-code
statistics. **Not stated**: whether the RVQ codebooks were re-fit after decoder fine-tuning.

## 5. Fragility of the SONAR space

This is the paper's own audit of the frozen-encoder assumption, and it is the section a
regression-onto-a-frozen-encoder arm should read first.

**The claim** (§2.5.2): "the homogeneous Euclidean geometry of any latent representation will not
perfectly match the underlying text semantics. This is evidenced by the fact that a small perturbation
in the embedding space may result in a **drastic loss of semantic information** after decoding."

**The definition** (Equations 26–28):
`fragility(w) := −E_{α~U([0,1]), ε~N(0,I)} [score(w, w_{α,ε})]`, where
`x_{α,ε} = denormalize(√(1−α) · normalize(x) + √α · ε)` and `w_{α,ε} = decode(x_{α,ε})`. Note the
perturbation is applied in the **normalized** coordinates, so it is scale-independent, and it is
"similar to the variance-preserving noising used in diffusion LCMs" — deliberately the same noise the
denoiser is trained against.

**Two score functions**: AutoBLEU between original and perturbed decode, and **external cosine
similarity** using an encoder unrelated to SONAR — **mGTE** (Zhang et al. 2024) — which is "typically
more robust to paraphrasing" than BLEU. Using an outside encoder is the point: you cannot audit
SONAR's geometry with SONAR.

**Scale**: **50M random text fragments**, each perturbed at **9 noise levels α ∈ {0.1, 0.2, …, 0.9}**.

**Findings.**

- **A noise-trained decoder is dramatically more robust** (Table 7, AutoBLEU on 10k-sentence random
  subsets, Flores on the full dev split):

| Model | Flores | CNN DailyMail | Gutenberg | C4 |
|---|---|---|---|---|
| Base SONAR decoder | 79.5 | 75.9 | 70.5 | 75.7 |
| Finetuned SONAR decoder | **88.0** | **87.6** | **85.6** | **87.5** |

  That is **+8.5, +11.7, +15.1 and +11.8 AutoBLEU** from one change: adding random noise vectors to
  the SONAR embeddings during decoder training (Equation 27). And that is *raw* reconstruction, on
  clean embeddings — the robustness training did not cost clean-input fidelity, it bought it. Figure 14
  right panel shows the gap widens with α: both AutoBLEU and cosine "decrease at a markedly slower
  rate for the Finetuned decoder than for the Base one as the amount of noise increases". Reading the
  panel, base-decoder AutoBLEU falls from about 0.8 at α ≈ 0 to near 0.0 by α ≈ 0.9 while the
  finetuned decoder still holds; cosine similarity falls more slowly than BLEU for both.
- **Fragility is length-dependent and the cliff is the same ~250 characters.** Figure 14 left panel,
  α-averaged: "Compared to the Auto-Encoding BLEU metric (which drops only by **1–2%** for long
  sentences), fragility is **more sensitive** to the length of sentences and drops faster for both
  similarity metrics. This shows that using a max sentence length over **250** can be extremely
  challenging for SONAR and the LCM model." Reading the left panel, α-averaged AutoBLEU falls from
  about 0.52 near 25 characters to about 0.22 near 250; the external cosine falls from about 0.855 to
  about 0.725 over the same span.
- **Short is not automatically safe**: "even if short sentences are on average more robust, splitting
  a long sentence in the **wrong place** may result in shorter but **more fragile** sub-sentences."
- **The spread across samples is large** (Figure 15): the α-averaged score distribution "exhibits a
  large spread of fragility scores across SONAR samples". Fragility is a per-sample property, not a
  global constant.
- **What the 5% most fragile embeddings are**: "hyperlinks, references, unique ids, code-switched or
  numerical entries… artifacts that the SONAR models were not exposed to during training or where the
  SONAR tokenizer fails." Also "short but complex technical phrases can be more fragile than common
  language phrases of similar length." Fragility is therefore usable as a **data filter**.

**What the authors conclude about frozen encoders** (§6, "Choice of the embedding space"): SONAR "is
trained to sustain a **local** geometry (sentences with very similar meanings are geometrically close)
with **no special guarantees for sentences that are only loosely related**. Yet, predicting next
sentences distribution requires the space to operate well **globally**." And: "Using a frozen encoder
represents some interesting trade-offs. Any frozen encoder which is learned in a different data
context, and with no a-priori strong connection to LCM modeling, may be **suboptimal** compared to
encoders that are learned in an end-to-end fashion (with the loss coming from the decoder). At the
same time, learning an encoder within end-to-end training can be challenging and the resulting space
is not guaranteed to result in good semantic representations."

**Not stated**: a fragility number for the *LCM's own predictions* — the study measures the encoder's
space, not the model's outputs. **Not stated**: any numeric table of fragility against length or α;
Figures 14 and 15 are plots only. **Not stated**: whether the fragility-filtered corpus was used for
the 7B run (fragility appears as a loss *weight* ablation in §2.4.4, which was then dropped).

## 6. Scaling to 7B, and where the LCM loses

**The 7B model** (§3). Two-Tower chosen over One-Tower "given its smaller memory footprint,
particularly when processing long contexts with a shallower contextualizer tower". 5 contextualizer
layers, **14** denoiser layers (up from 13), `d_model` = **4096**, 32 heads, everything else as the
1.6B. Pre-trained on **2.3B documents = 2.7T tokens = 142.4B concepts**, for **124k steps on 256
A100s** with total batch **1M concepts**, context extended to **2048 concepts** (from 128). AdamW
(β1, β2) = (0.9, 0.95), ε = 1e-5, weight decay 0.1, cosine LR with **10,000-step warm-up to
LR = 3e-4**, gradient clipping at **norm 10** "to improve training stability". Instruction tuning:
loss on answer sentences only, answers suffixed with "End of response.", **389M sentences of which
53M are targets**, LR 3e-5 cosine, **7 epochs**, batch 262K sentences. Called `Two-Tower-7B-IT`.

**Metrics** (Table 8): R-L (ROUGE-L), OVL-3 (proportion of source word 3-grams present in the output,
an extractiveness meter), REP-4 (portion of duplicated word 4-grams, a repetition meter), CoLA
(sentence-level linguistic-acceptability classifier, Krishna et al. 2020 — fluency), SH-4 and SH-5
(SEAHORSE Q4 "fully attributable to the source" and Q5 "captures the main ideas"), and local coherence
as average cosine between sentence n and n+2.

**Table 10, CNN DailyMail:**

| Model | Paradigm | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ | SH-4 ↑ | SH-5 ↑ |
|---|---|---|---|---|---|---|---|
| Ground truth | — | 100.00 | 0.170 | 0.684 | 0.850 | 0.683 | 0.586 |
| T5-3B | SFT | 37.56 | 0.174 | 0.854 | 0.946 | 0.773 | 0.503 |
| Gemma-7B-IT | IFT | 31.14 | 0.245 | 1.032 | 0.963 | 0.740 | 0.560 |
| Mistral-7B-v0.3-IT | IFT | 36.06 | 0.200 | 0.780 | 0.972 | 0.780 | **0.676** |
| Llama-3.1-8B-IT | IFT | 34.97 | 0.248 | 0.928 | **0.973** | 0.763 | 0.692 |
| **Two-Tower-7B-IT** | IFT | 36.47 | **0.177** | **0.757** | **0.767** | 0.723 | **0.459** |

**Table 10, XSum:**

| Model | Paradigm | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ | SH-4 ↑ | SH-5 ↑ |
|---|---|---|---|---|---|---|---|
| Ground truth | — | 100.00 | 0.108 | 0.399 | 0.987 | 0.352 | 0.418 |
| T5-3B | — | 17.11 | 0.221 | 0.671 | 0.939 | 0.680 | 0.450 |
| Gemma-7B-IT | IFT | 18.20 | 0.177 | 0.620 | 0.769 | 0.546 | 0.446 |
| Mistral-7B-v0.3-IT | IFT | 21.22 | 0.162 | 0.480 | 0.922 | 0.633 | 0.621 |
| Llama-3.1-8B-IT | IFT | 20.35 | 0.186 | 0.501 | 0.941 | 0.687 | 0.658 |
| **Two-Tower-7B-IT** | IFT | **23.71** | **0.106** | 0.464 | **0.683** | **0.358** | **0.284** |

**Read the wins and the losses honestly.** The LCM wins ROUGE-L on XSum (23.71, the best in the
table, ahead of Llama-3.1-8B-IT's 20.35) and is second on CNN DailyMail (36.47 against T5-3B's 37.56,
ahead of all three instruction-tuned 7–8B LLMs). It is the most **abstractive** of the
instruction-tuned models (OVL-3 0.177 on CNN DailyMail against 0.200–0.248 for the LLMs, though
T5-3B's 0.174 is lower still; and 0.106 on XSum, the lowest in that table and below even the ground
truth's 0.108). It is also the **least repetitive** model in both tables (REP-4 0.757 on CNN
DailyMail, closest of any model to the ground truth's 0.684; 0.464 on XSum).

Then the losses, and they are large:

- **Fluency. CoLA 0.767 on CNN DailyMail against 0.946–0.973 for every LLM, and 0.683 on XSum against
  0.922–0.941 for Mistral and Llama.** That is a 0.18–0.26 absolute gap. The authors' mitigation is
  that even the human ground truth scores 0.850 (CNN DM), below the LLMs — which is an argument that
  CoLA is biased toward LLM text, and also an admission that the LCM is below the *human* score too.
- **Semantic coverage. SH-5 0.459 on CNN DailyMail against 0.503–0.692, and 0.284 on XSum against
  0.446–0.658.** The LCM is last on SH-5 in both tables, below the ground truth in both (0.586 and
  0.418).
- **Source attribution. SH-4 0.723 (CNN DM, last of the model rows) and 0.358 (XSum, last).**

The authors' reading: "This may be explained by model-based metrics that are more biased towards LLM
generated content." That is a plausible partial explanation and it is not a measurement.

**Long-context summarization** (Table 11, LCFO, ~5k-word documents, summaries at 20/10/5% of input;
WR = word count ratio to source):

| LCFO.5% | WR | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ | SH-4 ↑ | SH-5 ↑ |
|---|---|---|---|---|---|---|---|
| Gemma-7B-IT | 0.107 | 25.21 | 0.151 | 4.711 | 0.688 | 0.357 | 0.174 |
| Mistral-7B-v0.3-IT | 0.512 | 21.36 | 0.532 | 5.997 | **0.854** | **0.656** | 0.296 |
| Llama-3.1-8B-IT | 0.076 | **37.67** | 0.190 | 2.767 | **0.931** | 0.488 | **0.314** |
| Two-Tower-7B-IT | 0.060 | 26.88 | 0.162 | **2.473** | 0.796 | 0.628 | 0.196 |

| LCFO.10% | WR | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ | SH-4 ↑ | SH-5 ↑ |
|---|---|---|---|---|---|---|---|
| Gemma-7B-IT | 0.150 | 29.25 | 0.164 | 6.427 | 0.667 | 0.377 | 0.194 |
| Mistral-7B-v0.3-IT | 0.549 | 25.00 | 0.537 | 6.289 | 0.848 | 0.660 | 0.306 |
| Llama-3.1-8B-IT | 0.128 | **42.85** | 0.243 | 3.804 | **0.907** | 0.486 | 0.310 |
| Two-Tower-7B-IT | 0.089 | 29.38 | 0.202 | **3.00** | 0.791 | 0.623 | 0.183 |

| LCFO.20% | WR | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ | SH-4 ↑ | SH-5 ↑ |
|---|---|---|---|---|---|---|---|
| Gemma-7B-IT | 0.257 | 33.32 | 0.201 | 9.188 | 0.603 | 0.425 | 0.239 |
| Mistral-7B-v0.3-IT | 0.493 | 28.82 | 0.527 | 5.806 | 0.858 | 0.658 | 0.293 |
| Llama-3.1-8B-IT | 0.179 | **46.92** | 0.272 | 4.783 | **0.888** | 0.485 | **0.315** |
| Two-Tower-7B-IT | 0.140 | 31.74 | 0.253 | **3.664** | 0.779 | 0.613 | 0.187 |

The LCM beats Mistral and Gemma on R-L at 5% and 10% and sits just under Gemma at 20%, and it has the
**lowest REP-4 at every length**. Llama-3.1-8B-IT is **10.8, 13.5 and 15.2 R-L points ahead** at
5/10/20% — the authors flag this as possibly "training data contamination for Llama-3.1-8B-IT, or…
the other two LLMs struggle to handle the long input context", which is a hypothesis, not a
measurement. The LCM's SH-5 is last or near-last again (0.196 / 0.183 / 0.187).

**Summary expansion** (§4.1, Table 12) is where the LCM loses cleanly:

| CNN DailyMail | WR | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ |
|---|---|---|---|---|---|
| Gemma-7B-IT | 6.8 | 35.54 | 0.801 | 2.104 | 0.951 |
| Mistral-7B-v0.3-IT | 6.4 | 34.24 | 0.817 | 2.063 | **0.959** |
| Llama-3.1-8B-IT | 8.5 | **37.76** | 0.822 | 2.582 | 0.844 |
| Two-Tower-7B-IT | 6.3 | 30.85 | 0.726 | 2.911 | **0.474** |

| XSum | WR | R-L ↑ | OVL-3 ↑ | REP-4 ↓ | CoLA ↑ |
|---|---|---|---|---|---|
| Gemma-7B-IT | 19.5 | 17.89 | 0.963 | 10.238 | 0.116 |
| Mistral-7B-v0.3-IT | 1.6 | **29.31** | 0.893 | 2.268 | **0.939** |
| Llama-3.1-8B-IT | 19.8 | 28.84 | 0.915 | 2.543 | 0.898 |
| Two-Tower-7B-IT | 7.1 | 23.82 | 0.561 | **1.542** | 0.603 |

**CoLA 0.474 on CNN DailyMail** is the worst single number the LCM posts anywhere in the paper — a
fluency score below one half, against 0.844–0.959 for the three LLMs, a **0.37–0.49 gap**. The
authors own it: "the CoLA results show that this comes along with **lower fluency**, especially for
CNN DailyMail", and attribute it to the SONAR decoder being "trained on a translation task that tends
to paraphrase". On R-L the LLMs lead by 3.4–6.9 (CNN DM) and 5.0–5.5 (XSum).

**Zero-shot multilingual** (§4.2, Figure 16, XLSum, 45 languages, 42 reported after dropping Pidgin,
Serbian-Latin and Uzbek-Cyrillic which SONAR does not support). The LCM saw **English only** and no
XLSum training data at all:

- English: **23.5 against Llama-3.1-8B-IT's 20.7 ROUGE-L** (+2.8).
- Average over the **six** languages officially supported by Llama and present in XLSum (English,
  French, Hindi, Portuguese, Spanish, Thai): **20.2 against 19.7** (+0.5).
- Best single language: **Vietnamese at 30.4**.
- Low-resource languages above 20 ROUGE-L: Southern Pashto, Burmese, Hausa, Welsh; also named as
  performing well: Somali, Igbo, Kirundi.

This is the paper's strongest result and it is a direct consequence of the frozen multilingual
encoder, the same design choice §6 criticises for everything else.

**Explicit planning** (§4.3, LPCM). A One-Tower LCM trained multitask to also emit a **break concept**
(paragraph boundary) and a **plan concept** (a one-sentence synthetic topic description of the coming
paragraph, generated by Llama-3.1-8B-IT), then conditioned on it. Data: paragraphs split by Segment
Any Text, each forced to **under 10 sentences**, small consecutive paragraphs merged; **320M
paragraphs, 1.5B concepts, ~30B tokens**. Coherence judged by Llama-3.1-8B-IT on a 0–5 scale
(Appendix D prompt), validated against the Jwalapuram et al. human-judgement set where it agreed with
annotators better than their own coherence model.

| | Llama-3.1-8B-IT judge ↑ |
|---|---|
| LPCM | **2.82 ± 0.62** |
| Baseline One-Tower | 2.74 ± 0.70 |

Table 13. A **+0.08** difference, significant by paired t-test **at the 99.9% level** (footnote 11).
Same parameter count, same data, same steps. Statistically real, practically tiny, and both sit near
the middle of a 0–5 scale.

**Not stated**: any human evaluation, anywhere in the paper. Every "quality" judgement is a
classifier (CoLA, SEAHORSE, Jwalapuram coherence) or an LLM judge (LPCM only). **Not stated**: any
non-generative benchmark — no MMLU, no MCQ, no code, no math. **Not stated**: any perplexity
comparison against the token LLMs. **Not stated**: wall-clock or measured throughput; Figure 13 is
theoretical FLOPs only.

## 7. Limitations the authors state themselves (§6)

Grouped exactly as they group them.

**Choice of the embedding space.**
1. SONAR was trained on bitext machine-translation data with "rather short sentences". Consequence
   one: it "is trained to sustain a **local** geometry… with **no special guarantees for sentences
   that are only loosely related**. Yet, predicting next sentences distribution requires the space to
   operate well **globally**."
2. Consequence two: SONAR "auto-encodes surprisingly well texts containing links, references, or
   merely numbers or code data. Yet, such texts tend to be **fragile**", a distribution mismatch
   against normal LLM pre-training corpora, so "the accurate prediction of the sentences containing
   such a content (**non-negligible** in LCM pre-training data) will be hard for any LCM SONAR based
   model. For instance, the **factuality** of fragile generated sentences may easily be compromised."
3. The frozen-encoder trade-off, quoted in §5 above: frozen may be suboptimal against end-to-end, but
   end-to-end "can be challenging", is less data- and compute-efficient, and risks modality
   competition.

**Concept granularity.**
4. "the manifold of possible next sentences is **very wide**, attributing a proper probability to each
   of such sentences is much harder (even with a modeling within the latent space) tha[n] to the
   discrete set of tokens."
5. "Combinatorial complexity of possible next sentences grows **exponentially** with the maximum
   character length… long sentences (**>120 characters**) could reasonably be considered as several
   concepts. However, any finer splitting of such sentences does not necessar[il]y separate well these
   concepts. This shows the limitation of a **fixed size embedding representation for one sentence**.
   Text splitting… or **one-to-many mapping of a sentence into several embeddings** is a major future
   direction of research." Note the 120-character figure here sits *below* the 200-character cap the
   data pipeline actually used.
6. **Data sparsity**: "the large majority of sentences are merely **unique**" across the whole corpus.
   Higher-level representations trade lossless encoding (named entities, numbers) against abstraction.
   "SONAR offers semantic representations with good auto-encoding quality but still certainly
   sacrificing generalization capacities."
7. Mitigation sketched, not done: split or re-encode text into "conceptual units which are more
   commonly shared across source documents", in the spirit of stemming or lemmatization for words.

**Continuous versus discrete.**
8. "sentences in the SONAR space, despite being represented as continuous vectors, remain **discrete
   combinatorial objects**. This makes diffusion modeling **struggle** on the text modality (either at
   word or sentence embedding level)."
9. "The **contrastive** nature of cross-entropy loss based on softmax outputs… plays a critical role
   for many downstream task[s] where higher accuracy is required (e.g. MCQ tasks, code or math
   generation). On the opposite, continuous diffusion modeling **does not allow to integrate such a
   contrastive objective**."
10. The Quant-LCM post-mortem quoted in §4 above, ending: "This indicates once again the importance of
    developing a **new representation space**, either continuous or discrete, for the Large Concept
    Model."

From the conclusion (§8), three further admissions: next-sentence prediction is "substantially more
challenging" than next-token prediction for three named reasons (unbounded sentence space against a
~100k vocabulary; more inherent ambiguity even given long context; no normalised output
distribution); "small modeling errors could yield predictions in the embedding space which **do not
correspond to valid sentences**"; and the ability "to sample multiple embeddings and associate a
score", which would enable beam search over sentences, is named as **not implemented**. The paper
closes with "we acknowledge that there is still a long path to reach the performance of current
flagship LLMs."

## 8. What this says to TUL

Twelve bullets, each tying one number to one MORPH item. `(a)` = the `code_target` MSE/InfoNCE
regression arms, `(b)` = the vq8/vq4 discrete-thought arms, `(c)` = the fan4 multi-stream arms,
`(d)` = the standing house rule against decoding a span from one vector with no token path.

1. **(a) The conditional mean is the predicted outcome of an MSE target, and the paper measures it
   on four corpora.** Base-LCM wins ℓ2 everywhere (0.177 / 0.204 / 0.229 / 0.207, Table 3) and posts
   MI of 0.062 / −0.105 / 0.071 / −0.184 — at or below zero. MORPH's `code_target` cell reading as a
   rank-16 conditional mean found in one pass is the same event with a different instrument. The
   paper's framing is worth adopting verbatim: the failure is not that the regression is bad, it is
   that **the regression succeeded and the target was a mean**.
2. **(a) Ship CA and MI as MORPH instruments; they are cheap and they separate what CE does not.**
   CA is in-batch retrieval accuracy against the ground truth with the two temporal neighbours
   excluded; MI is `(1/|ŝ|)(log p_GPT2(ŝ) − log p_GPT2(ŝ | previous 10 sentences))`. In Table 3 these
   two are the only pre-training metrics that rank the models the same way Table 4's downstream
   ROUGE-L and coherence do. MORPH already has a per-span latent and a decoder, so both port with no
   new training. MI in particular would answer directly whether a slot cell makes the next span
   cheaper *given context*, which is the question the K1−K6 curve does not ask. (inference: that the
   port is mechanical — the paper never applies these to a looped model.)
3. **(a) A frozen encoder whose input is live is not the paper's setup, and the difference explains
   MORPH's collapse.** SONAR is frozen *and* its input is raw text, so its output is a genuinely
   fixed target. MORPH's frozen encoder pooled the live prelude, so the target moved with the model
   and the codes collapsed (own cosine 0.61 → 0.99, oracle CE 13 nats). The paper's own frozen-encoder
   caveat (§6) is about *suboptimality*, never about *collapse* — it had no way to see that failure,
   because it cannot happen in their architecture. The MORPH fix (a frozen deep copy, not a
   registered submodule) is the thing that restores the paper's precondition. (inference: the paper
   does not discuss moving-target encoders at all.)
4. **(b) 64 codebooks × 8192 units still loses 30% of auto-encoding BLEU** (Figure 9, body text).
   MORPH's vq8/vq4 use 512 codes with K = 8 or 4 per span. If a 1024-dim SONAR vector needs 64
   coarse-to-fine codes over 8192-entry books to retain 70% of its reconstructible content, then a
   4-or-8-code, 512-entry budget is a far tighter bottleneck than the LCM's, and the vq arms should
   report a reconstruction ceiling — the code's own AutoBLEU-equivalent — **before** any CE contrast is
   read. Without that ceiling an arm cannot distinguish "the loop cannot use the code" from "the code
   cannot carry the span".
5. **(b) Quantizing beat plain MSE by a wide margin, and that is the encouraging half.** Quant-LCM-c
   and -d post MI 0.715 and 0.359 on C4 against Base-LCM's −0.105, ROUGE-L 30.87 and 28.01 against
   23.69, coherence 0.847 and 0.704 against 0.482 (Tables 3, 4). The discrete route is a real fix for
   the conditional-mean failure mode `(a)` hit. It is the right next move after `(a)`, and the paper
   says so with numbers.
6. **(b) But the discrete-target variant lost to the continuous-residual one.** Quant-LCM-d trails
   Quant-LCM-c on MI at three of four corpora (0.359 vs 0.715 on C4, 0.153 vs 0.419 on Gutenberg,
   0.323 vs 0.744 on Wikipedia-en) and on both downstream metrics, and the authors' hypothesis is that
   "predicting codebook indices with cross-entropy loss is harder than [the] MSE objective". Directly
   relevant to vq8/vq4: the softmax-over-codes arm is the *harder* of the two discrete formulations,
   and a **residual-MSE conditioned on the partial code** is a cheaper arm that scored better in the
   only controlled comparison in print. Worth running as the vq control.
7. **(b) The reader must be retrained on the code distribution, including truncated codes.** LCM
   fine-tuned the SONAR decoder on 1.2M sentences of quantized representations, and during that
   fine-tune randomly truncated the code at `k ∈ [⅔·64, 64]` with p = 0.3 (§2.3.5), producing the
   Table 7 lift. MORPH's own "the reader was the limit" finding says the same thing from the other
   side. Any vq arm whose coda was not trained on partial codes is measuring the coda, not the code.
8. **(b) Quantization did not fix uniqueness, and the authors say why.** §6: the RVQ combinations grow
   exponentially and "do[] not solve the data sparsity/uniqueness issue" — most sentences in a corpus
   are unique, so a code that is nearly as expressive as the vector inherits the same sparsity. A
   MORPH vq arm that *does* work will be working because K = 4 or 8 codes is a genuine bottleneck that
   forces sharing, which is a different mechanism from LCM's 64-code near-lossless target. That is a
   reason for optimism about small K, not pessimism. (inference: the paper never tries a
   deliberately-lossy small-K code.)
9. **(c) The fan's missing piece is named in the paper's own future work.** §8: "the ability to sample
   multiple embeddings and **associate a score** would enable beam search to find the best sequence of
   sentences" — listed as not done. MORPH's fan4 cashed 0.056 of a 0.099 oracle headroom, which is
   exactly the gap between having K candidates and being able to rank them. The literature offers no
   solution to copy here; this is open ground.
10. **(c) Guidance scale is the paper's diversity-vs-fidelity dial, and it moves MI and ℓ2 in opposite
    directions** (§2.4.2, Figure 10, first panel): raising `g_scale` raises MI and PAR while raising
    ℓ2 from the true continuation. A fan of deterministic streams has no equivalent dial. If the fan
    is meant to cover a multimodal conditional, the paper's answer is that coverage comes from
    *sampling with a controllable temperature*, and the two variants that sample (diffusion,
    Quant-LCM) both beat the one that does not. (inference: mapping `g_scale` onto a fan mixture
    weight is my extrapolation, not the paper's.)
11. **(d) The house rule is confirmed with a number: coherence 0.482.** Base-LCM is precisely "decode
    a span from one regressed vector with no token path", and Table 4 scores it at 0.482 on a scale
    calibrated so incoherent ≈ 0 and coherent ≈ 1, against 0.968 for the sampler and 0.984 for the
    matched token-level Llama. The rule stands, and the citation is now a specific cell rather than a
    general impression.
12. **(d) MORPH's span cap is inside the safe zone, and the noised-decoder trick is the mitigation to
    copy if a cell ever drives a decode alone.** LCM's own corpus averages 88 characters and 27 tokens
    per sentence (Appendix A) and its fidelity knee is 250 characters (§2.2, §2.5.2); MORPH's 32-token
    cap lands near 100–130 characters, well under. Separately, training the decoder on **noised**
    embeddings lifted AutoBLEU 79.5 → 88.0 on Flores and 70.5 → 85.6 on Gutenberg (Table 7) at no cost
    on clean inputs — a nearly free robustness gain for any MORPH reader that has to consume an
    imperfect cell.

**One honest counterweight to bullets 1–2 before they get over-applied.** LCM's denoising steps are
*not* flat: MI rises with S across the swept range (Figure 10, third panel) with diminishing returns.
The structural reason each of those steps earns is that each has a **defined, trained target** — the
clean `x⁰` recovered from a specific noise level `t` drawn during training — and the same is true of
Quant-LCM's iterations, where step k predicts the residual codebook-k explains. MORPH's passes 2–6
have no per-pass target, and MORPH has already measured that per-pass targets, when added, are met in
one step. So the paper supports "an iterative refinement over a span latent can earn depth" and
supplies the two mechanisms that made it earn, but it does **not** supply evidence that a
weight-shared loop with a single terminal loss will. Treating LCM's step-count curve as a prediction
for MORPH's K-curve would be reading across two different training signals. (inference: the
"defined target per step" contrast is mine; the paper never compares against an untargeted loop.)
