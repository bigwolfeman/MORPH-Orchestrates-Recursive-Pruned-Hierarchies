> Reading filed 2026-09-14 for docs/tul-code-spec.md. Produced by a fetch-and-summarise pass over the
> arXiv HTML/PDF, not a hand transcription; items the reader flagged as unverified are marked in the text.

# LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning

arXiv 2510.04573 (verified by search; fetched https://arxiv.org/abs/2510.04573 and https://arxiv.org/html/2510.04573v3). Authors: Haoqiang Kang, Yizhe Zhang, Nikki Lijing Kuang, Nicklas Majamaki, Navdeep Jaitly, Yi-An Ma, Lianhui Qin.

Note: fetched through a tool that renders and summarizes the HTML paper rather than a raw-text parse. Numbers and structure are reliable; exact equation symbols (sub/superscripts) are approximate.

## 1. The latent

One block Z^(b) = {z1^(b), ..., z_Lb^(b)} is **one sentence** of the chain-of-thought. The paper: "we then split c into individual sentences, each treated as a block of latent tokens with block size L_b." L_b is ablated at 1, 2, 4, 6, 8; GSM8K reconstruction improves with L_b and is near-perfect at **L_b = 6**, with diminishing/redundant returns beyond that (Appendix C, Figure 6). So one latent token is not one word or one reasoning step; a block of ~6 latent tokens together stands for one CoT sentence.

## 2. The encoder (VAE)

Section 3.1: "our VAE encoder is initialized from a pretrained LLM and fine-tuned with all parameters, along with L_b learnable embeddings." The last hidden state (at the L_b query-embedding positions) is projected to mean mu and variance sigma^2; sampling is z = mu + sigma ⊙ eps, eps ~ N(0, I). Input is the CoT sentence text itself, not the question.

Loss (Equation 1), standard beta-VAE: L_beta-VAE = E_{q_phi(z|x)}[-log p_theta(x|z)] + beta * KL(q_phi(z|x) || p(z)). Numeric beta is **not stated**.

Decoder: "The decoder is a frozen pretrained LLM that conditions on the sampled Z^(b) to reconstruct the corresponding block of text" — frozen, autoregressive, sentence-level reconstruction.

Training stage: VAE trained **first and separately**: "We train the two components separately: the VAE is first trained to learn latent representations of thought tokens, after which the reasoning model is trained to predict these thought tokens." Not end-to-end joint.

Decodability trick (Section 3.2.1): robustness augmentation during VAE training — latent Gaussian noise z'_i = z_i + eta_i, eta_i ~ N(0, k^2 I), **k = 3**; input token substitution with probability **p = 0.3**. This makes the latent tolerant of the imperfect latents the diffusion model will later hand it, not just the encoder's own clean output.

## 3. The diffusion/flow model over latents

Flow matching, not DDPM: z_t = (1-t) z_0 + t*eps; velocity field u_theta trained against target u* via L_FM = E[||u_theta(z_t,t) - u*(z_t,t)||^2] (Equation 2). Ablation (Table 4) shows flow matching beats epsilon-, x0-, and v-prediction.

Conditioning: on question Q, with **block-causal, within-block-bidirectional** attention: "Within each block, tokens attend bidirectionally; across blocks, attention is strictly causal." Blocks are generated in "block diffusion" order (Section 2.3): "each block is denoised iteratively while conditioning on all previously generated blocks" (Appendix B.4). So generation is block-autoregressive, and within each block the L_b latent tokens are jointly denoised over several flow steps — neither fully parallel nor fully token-AR.

Step count as a quality dial: **yes, measured directly.** Figure 5: 5→10 denoising steps gives "+11.7 points in accuracy on average of 7 benchmarks"; 10→30 steps adds "+4.8 additional points" (range tested 5–50). More steps helps with diminishing returns, and is treated as an inference-time compute dial.

## 4. The decoder (latent block → text)

Two decode paths. The **VAE decoder** (frozen, autoregressive) reconstructs a sentence from its latent block and is used for VAE training/validation. At inference the paper skips paying for it on every step for efficiency: after a special `<SOA>` token, "the same model autoregressively predicts answer text tokens" directly from the full latent sequence, p_psi(y_t | q, Z^(<=B), y_<t), trained with cross-entropy (Equation 3). So there is a token-level AR path, but it is used only for the final answer — the CoT itself stays latent; only the last-mile answer is decoded token-by-token. A special-token loss (Equation 4) handles `<SOA>`/`<BOT>`-style boundary classification.

## 5. Training cost and losses

Joint reasoning-model loss (Equation 5): L = lambda_FM * L_FM + lambda_Ans * L_Ans + lambda_Spec * L_Spec (flow-matching + answer CE + special-token CE). **Exact weight values are not stated.**

Two-stage training of the reasoning model:
1. **Teacher-forcing**: trained against oracle VAE-encoded latents Z^(1:B) from real CoT data.
2. **Rollout**: "Instead of conditioning on oracle latents, the model generates its own latents Z-tilde^(1:B) from random noise using fewer denoising steps (50→10)," flow-matching loss still active. This unrolls the model's own noisier, fewer-step generation during training instead of always feeding ground truth.

VAE and reasoning model are two separate training stages (§2/§3), not end-to-end.

## 6. Collapse / leakage

**No dedicated section on posterior collapse, KL vanishing, or answer leakage** was found. What is said, adjacent to it:
- Stage-2 rollout (self-generated noisy latents rather than always-oracle) is explicitly linked to avoiding "latent collapse as in Coconut w/o curriculum learning" — Coconut's collapse mode is cited analogically as the failure rollout training avoids, not measured directly on LaDiR's own VAE.
- VAE-side noise/token-substitution augmentation (k=3, p=0.3) is framed as making the latent "resilient to noise," not explicitly an anti-KL-collapse device, but it stops the decoder from depending on a razor-thin encoding.
- No statement was found about the encoder leaking the final answer; the encoder only sees CoT sentence text, not the question+answer pair, which structurally limits leakage but is not argued explicitly.
- **Not stated**: numeric beta, a direct KL-collapse measurement, an explicit leakage probe.

## 7. Results

**Math reasoning, LLaMA-3.1-8B** (Table 1): LaDiR MATH 45.2, GSM8K 84.2, avg over 7 benchmarks 41.8, vs CoT-SFT (43.1 / 84.5 / 40.4) and Coconut (37.3 / 68.3 / 31.9). "w/o Stage 2" ablation drops to 30.7 / 57.8 / 32.6 — the rollout stage is load-bearing. Framed as "~2% average improvement over the best prior latent approach."

**Countdown planning** (Table 2): CD-4 Pass@1 76.6 / Pass@100 96.4 / CD-5 Pass@1 38.5, vs LLaMA-8B SFT baseline 46.7 / 65.3 / 8.9 — "over 30% absolute improvement in both Pass@1 and Pass@100."

**Diffusion steps** (Figure 5): 5→10 steps +11.7 pts avg, 10→30 steps +4.8 pts more.

**Latent tokens per block** (Figure 6): GSM8K accuracy rises with L_b up to ~6, then flattens.

**What did not work**: epsilon-, x0-, v-prediction parameterizations all underperform flow matching (Table 4). Oracle-only teacher forcing without Stage-2 rollout badly hurts accuracy (30.7 vs 45.2 MATH).

Code generation is named as a third eval domain in the abstract; no numbers for it surfaced in this extraction pass.

## 8. Diversity / multimodality

A named, first-class mechanism (Section 3.4), not incidental:
- **Wider initial noise**: an inflated initial noise scale sigma-tilde^2 "broaden[s] the distribution of starting points" — diversity is injected via the initial noise prior, not assumed from the model's own multimodality.
- **Diversity gradient guidance** (Equation 6), a repulsive term over a batch of concurrently sampled latents z_i: F(z_i) = sum_{j != i} 2*(1 - ||z_i-z_j||_2^2/sigma^2) * exp(-||z_i-z_j||_2^2/sigma^2) * (z_i-z_j), weighted by a time-decayed gamma_t = gamma_max*(t/T). Structurally an RBF-kernel repulsive force (SVGD-flavored) that pushes parallel latent samples apart during early denoising and fades to zero near t=0, so it shapes exploration without corrupting the final sample.
- Ablated jointly (Figure 4): sigma-tilde^2 = 2 and gamma_max in [0.3, 0.5] reported as the accuracy/diversity balance point.
- Payoff: Pass@100 96.4% vs 65.3% AR on Countdown-4. The tool's extraction also reported a "7.3 vs 3.0 unique solutions" figure whose exact source table was not pinned down — flag for re-verification before treating as a hard citation.
- Abstract framing: diversity "enables the generation of multiple diverse reasoning trajectories... allowing the model to plan and revise holistically," contrasted with AR decoding's inability to revisit/refine earlier tokens.

## What z is in LaDiR, in three sentences

z is a block of L_b (~6) continuous latent tokens produced by a frozen-decoder, learnable-query VAE encoder fine-tuned to compress exactly one CoT sentence into that block, trained upstream and independently of the reasoning model. At inference the reasoning model never writes that sentence's tokens: it flow-matches a velocity field that iteratively denoises a block of latents (block-causal across sentences, bidirectional within one), conditioned on the question and every prior block, and only the final answer span is decoded back to text autoregressively. Diversity is not claimed as an intrinsic property of a naturally multimodal z; it is engineered at inference time by widening the initial noise and adding an explicit inter-sample repulsive term during denoising.

## Open questions the paper does not answer

- No numeric beta (KL weight) for the VAE loss.
- No numeric lambda_FM / lambda_Ans / lambda_Spec for the joint reasoning-model loss.
- No direct measurement of whether KL/posterior collapse occurs in LaDiR's own VAE — the Coconut citation is analogical, about a different (non-VAE) method's failure mode, not a measurement here.
- No explicit probe of whether the sentence-level encoder leaks information from later CoT or the final answer.
- No code-generation numbers surfaced despite code generation being named as an eval domain.
- The "7.3 vs 3.0 unique solutions" diversity number needs re-verification against the primary table before being cited as a hard fact.
