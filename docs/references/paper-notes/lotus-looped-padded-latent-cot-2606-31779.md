# Repo note: LOTUS (looped padded transformer with parallel latent supervision)

- **Title:** LOTUS, code for "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (Ying Fan, Anej Svete, Kangwook Lee; 2026). Type: REPO with its paper.
- **arXiv or URL:** https://github.com/yingfan-bot/lotus (MIT); paper arXiv 2606.31779 (v2, 2026-07-13). Read: README in full, `args/base/lotus.yaml`, the head and the loop section of `scripts/lotus.py` (1620 lines, skimmed), and the paper text through the ablations (Sections 4.1 to 4.3, Tables 1 to 8). Appendices not read. No code was run.
- **Local paper text:** docs/references/looping-depth/lotus-latent-reasoning-looped-2606-31779/ (README.md, lotus.py, lotus.yaml, paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: Wolfe's priority is "the span needs a target for credit assignment". LOTUS gives a looped model K blocks of latent positions and puts a direct cross-entropy on each block's gold step tokens through the base LM head. That is a span-level target read back through the LM head, with a post-loop readout.
- 2026-10-20: It is a measured result on a looped model at 3B with ablations on loop count and readout timing.

## 2. What the paper shows

- 2026-10-20 (Sec 3, `lotus.py` forward loop): the prefix is [question, K x c latent tokens, answer]. For R iterations all latent positions are recomputed with embedding = original embedding + previous hidden state (`latent_injection_mode: add`; the `replace` mode is a code option). Loss = answer CE + lambda_step * CE of each block's latent positions against the gold CoT-step tokens, read from the final iterate h(R). A curriculum replaces CoT steps by latents (`max_latent_stage 3`, `epochs_per_stage 3`) after a CoT fine-tune stage 1. Fine-tuning of Llama-3.2 and GPT-2, GSM8K-Aug only.
- 2026-10-20 (Table 1, 3 seeds): Llama-3.2-3B GSM8K explicit CoT 71.5, LOTUS 70.0 +/- 0.9, CODI 60.8, no-CoT 38.3. README: 70.05 % for the released 3B model. Llama-1B: CoT 58.4, LOTUS 57.3. GPT-2: CoT 42.7, LOTUS 44.1.
- 2026-10-20 (Table 2, 3): thought-phase latency 2.5x lower than CoT on math-expression steps; 6.9x lower on natural-language steps (963.6 to 140.8 ms), with accuracy 68.13 vs 68.41.
- 2026-10-20 (Table 4): latent supervision design at 3B: none 63.3; CODI only 64.4; per-iteration readout through the LM head 68.2; post-loop readout 70.0.
- 2026-10-20 (Tables 5, 6, 7): trained loop count R = 2, 3, 4, 5, 6 gives 14.6, 23.2, 52.6, 68.1, 70.0 %. Inference-time R on the R = 6 model: 1 to 7 gives 22.7, 40.0, 55.0, 63.5, 68.7, 70.0, 69.3. Latents per block c = 1, 5, 10, 25, 30 gives 49.7, 67.5, 68.4, 70.0, 70.0 %.
- 2026-10-20: failure modes the authors list: LOTUS-aux (auxiliary decoder) degrades at small scale; one token per step is too narrow (49.7 %); needs CoT-annotated data and a CoT-tuned stage 1.

## 3. Our speculation

- 2026-10-20 Speculation: the trained R sweep is the clearest number: depth is worth a lot (14.6 to 70.0 %), and it is worth it because every latent block has its own gold target. The target comes from a CoT annotation that web text does not have. A span-only Parcae over punctuation spans has no such gold step. A near analogue is the NEXT span's own tokens, which is what a decoder trained later would give.
- 2026-10-20 Speculation: Table 4 (post-loop beats per-iteration readout) agrees with our rule that only the main loss buys depth and no pass-indexed terms.
- 2026-10-20 Speculation: the block-parallel layout (K blocks, c positions each, all looped together) is the same object as our slot positions plus fan cells. LOTUS has no router or selector; its c = 1 to 5 jump is like our cell-count result (1 cell costs +0.005 for us).
- 2026-10-20 Speculation: R is fixed at train and test (no Poisson). Inference R = 7 is slightly below R = 6 (69.3 vs 70.0), so the model trained at one depth does not extrapolate; this matches "training depth moves CE, eval depth does not".

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper: more loop iterations raise accuracy steeply. Ours: K1-K6 is small on web text (+0.0054 on the fan). The tasks differ (GSM8K with gold steps vs next-token on web text), so the gap itself is the finding: depth pays when the latent has its own target.
