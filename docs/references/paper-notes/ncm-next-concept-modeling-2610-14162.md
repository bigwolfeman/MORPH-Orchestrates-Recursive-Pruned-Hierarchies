# Paper note: Next Concept Modeling (NCM)

- **Title:** Next Concept Modeling (NCM), Hazy Research and Stanford
- **arXiv or URL:** arXiv 2610.14162. Code: github.com/HazyResearch/ncm
- **Local paper text:** docs/references/looping-depth/ncm-next-concept-modeling/2610.14162.txt (pdftotext, 2026-10-16)
- **Released weights (checked 2026-10-16):** https://huggingface.co/hazyresearch/ncm-checkpoints (MIT, not gated). Each model is a frozen residual-VQ tokenizer (tokenizer.pt) plus a generator (generator.pt) and config.yaml. Released: LM1B small, medium, large (large: 36 layers, width 1280, 12 levels of 8192 codes); Sudoku easy, medium, hard; TinyStories; XSum; WMT14. NOT released: OpenWebText (`owt/` holds only TODO.md, though the repo README links it). The GitHub repo (MIT, last commit 2026-10-14) holds eval and sampling code; no training script was seen at top level.
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
- 2026-10-16 (sec. 3.4, Fig. 4, eq. 10, App. B, App. F.1): Level lengths grow l_1 < ... < l_K = L (ablations: K = 12, linear 1 to 128). The generator trains ALL levels in one forward, so one training example is N = sum_k l_k positions (block-diagonal attention). For a linear 1-to-L schedule that is about K * L / 2 positions (LM1B, K = 12, L = 128: about 6x the L positions an AR model processes). The paper's token budgets count full-length text positions and say "These budgets measure token exposure, not FLOPs" (App. B accounting paragraph). So the speed gain is at generation (K sequential steps instead of L); training costs more per text token than AR at the same width and depth (our reading of the shapes, not a measurement).
- 2026-10-16 (App. A): The ELBO uses the complete hierarchy; minimizing L_NCM maximizes it over theta. So a likelihood bound per sequence exists and can score multiple-choice candidates.
- 2026-10-16 (App. B, Table 4): OWT uses the GPT-2 tokenizer and L = 1024; LM1B uses bert-base-uncased and L = 128; TinyStories a custom BPE, L = 128.
- 2026-10-14 (weak spots): No likelihood or CE is reported. The ELBO is derived but never given. OWT Gen PPL is 83 vs 35 for an AR sample. The token budget counts L positions per sample, not the sum over levels, so the paper understates FLOPs (our reading). K is fixed for every input.

## 3. Our speculation

- 2026-10-14 Speculation: The block-diagonal result favors a span code that each level reads from its own state only. This is untested on our models.
- 2026-10-14 Speculation: Because the paper reports no likelihood, our own CE numbers are the only comparison we have for a span code.

- 2026-10-16 (Table 8, App. E.2): Gen PPL under GPT-2-large. TinyStories: data 9.67, AR sample 8.89, NCM 15.41 (12 NFE). LM1B: data 53.51, NCM 93.05, no AR row. OWT: data 14.15, AR sample 35.45, NCM 83.12 (32 NFE). NCM beats the diffusion and flow baselines at far fewer NFE; it does not reach AR sampling quality on any dataset with an AR row.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-16: Agreement. Our run of the released checkpoints (their evaluate.py, 5 seeds x 256 samples, GPT-2-large; morph-scratch/abq/ncm-ref/REPORT.md): TinyStories Gen PPL 16.68 (paper 15.41); LM1B small 94.69, medium 88.96, large 81.25 (paper 93.05, size not matched). 11 network passes per 128-token batch (level 0 is drawn from a stored unigram). LM1B sampling uses a hand-tuned greedy pick with frequency and adjacency penalties. The prompt prefix is one token; prompts gave word salad on LM1B and were often ignored on TinyStories. TinyStories samples are story-shaped with local repeats ("is is", "too too"). Training cost shape from their code: 642 generator positions per 128-token example (about 5x AR), attention pairs about 6.4x.

- 2026-10-14: Our measurement: our span NCM on Parcae (branch lxtul/ncm) costs more to predict than it saves. Level 1 costs 0.226 nats per token and saves 0.118 nats per token. Paper's claim: the paper reports no likelihood, so there is no number to compare. Source: filing docs/experiments/failures/2026-10-14-parcae-span-ncm-5k.md (on branch lxtul/ncm; this file is not on master in this checkout).
