# Paper note: Next Concept Modeling (NCM)

- **Title:** Next Concept Modeling (NCM), Hazy Research and Stanford
- **arXiv or URL:** arXiv 2610.14162. Code: github.com/HazyResearch/ncm
- **Local paper text:** none archived
- **Date first noted:** 2026-10-14

## 1. Why we reference it

- 2026-10-14: Our nearest arm is LCTUL-D (discrete latent codes, masked denoiser thinker). NCM uses a similar idea: a frozen multiscale residual VQ tokenizer and a generator that makes code levels coarse to fine.
- 2026-10-14: The paper's block-diagonal read (each level reads only its own block) supports the strict geometry we use in the slot loop.
- 2026-10-14: Overlap with LXTUL is low. NCM has no loop, no autoregressive token stream, and no slots.

## 2. What the paper shows

Read in full on 2026-10-14. Section numbers not recorded.

- 2026-10-14 (stage 1): Multiscale residual VQ-VAE. 0 encoder blocks (embeddings only), 1 decoder block, K levels, 4096 codes per level, nested dropout 0.5.
- 2026-10-14 (stage 2): About 100M transformer. Level k is predicted in parallel from the cumulative reconstruction of levels below k. Block-diagonal attention: each level reads only its own block. All levels train in one teacher-forced forward, with no BPTT.
- 2026-10-14: Uniform code corruption on inputs, decreasing per level, gives a large Gen-PPL gain (the paper's exposure fix).
- 2026-10-14: Block-diagonal (state-only read) beats block-causal (history read): Gen PPL 15.41 vs 19.63, and it is faster.
- 2026-10-14 (weak spots): No likelihood or CE is reported. The ELBO is derived but never given. OWT Gen PPL is 83 vs 35 for an AR sample. The token budget counts L positions per sample, not the sum over levels, so the paper understates FLOPs (our reading). K is fixed for every input.

## 3. Our speculation

- 2026-10-14 Speculation: The block-diagonal result favors a span code that each level reads from its own state only. This is untested on our models.
- 2026-10-14 Speculation: Because the paper reports no likelihood, our own CE numbers are the only comparison we have for a span code.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-14: Our measurement: our span NCM on Parcae (branch lxtul/ncm) costs more to predict than it saves. Level 1 costs 0.226 nats per token and saves 0.118 nats per token. Paper's claim: the paper reports no likelihood, so there is no number to compare. Source: filing docs/experiments/failures/2026-10-14-parcae-span-ncm-5k.md (on branch lxtul/ncm; this file is not on master in this checkout).
