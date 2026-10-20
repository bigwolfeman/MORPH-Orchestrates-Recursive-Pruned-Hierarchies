# Paper note: DiscoLoop (looping discrete embeddings and continuous hidden states)

- **Title:** DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning (Hengyu Fu, Tianyu Guo, Zixuan Wang, Hanlin Zhu, Jason D. Lee, Jiantao Jiao, Stuart Russell, Song Mei; 2026, paper)
- **arXiv or URL:** 2607.00341 (v1 2026-07-01, v2 2026-07-27). Read in full, main text and Appendix B training notes; Appendix A (three-hop) not read line by line.
- **Local paper text:** docs/references/looping-depth/discoloop-2607-00341/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is a measured looped-LM result (440M, 20B tokens, loop 4) for a change to what the recurrence carries between passes. Our loop carries only the continuous state through a contractive diagonal injection.
- 2026-10-20: It names a failure our tests may share: the state after pass 1 is decodable but not aligned with the token embedding the next pass expects (cosine 0.27 to 0.33).

## 2. What the paper shows

- 2026-10-20 (Sec 3.1, Table 1): symbolic two-hop task, 500 entities per graph, K = 2 loops. Vanilla looped transformer: train 100 %, ID test 71.1 %, OOD test 8.3 %. Stage-1 (read after loop 1) two-hop train accuracy 8.8 %, so the second loop does the composition.
- 2026-10-20 (Sec 3.2, Table 1b, Fig 2): logit lens after loop 1 gives the right bridge token with probability 1.000, but cos(H1, W[b]) is 0.327 (ID) and 0.266 (OOD). A training-free swap at one position, H <- (1-a) H + a Norm(W[argmax]), lifts OOD from 8.3 % to 25.9 % at a = 0.1 and both splits to about 100 % at a = 0.5.
- 2026-10-20 (Eq 4, 5): DiscoLoop adds to the state after each non-final pass a gated soft decode-then-encode vector: H~ = H + alpha * RMSNorm(softmax(W h / tau) @ W). Gate fixed at 1 (symbolic) or a learned sigmoid with d+1 parameters. No extra loss. It costs one LM-head projection and one embedding mixture per position per pass.
- 2026-10-20 (Sec 5.1, Fig 3): symbolic task, DiscoLoop reaches near 100 % ID and OOD with a phase change near epoch 500; vanilla loop stays near 70 % ID and below 10 % OOD; the non-looped stack of K layers stays under 20 % ID and 0 % OOD. Natural-language versions: DiscoLoop about 100 % ID and about 95 % OOD (Fig 4).
- 2026-10-20 (Sec 5.3, Table 2, App B, Fig 7): 440M, 24 layers, d 1024, loop 4, 20B tokens, 6:4 FineWeb-Edu to FineMath, Muon + AdamW. Seven-task zero-shot average: vanilla loop 49.3, PonderLM 49.8, DiscoLoop 50.5 (ARC-C 30.5 to 32.6, LAMBADA 35.1 to 37.6, SciQ 79.5 to 81.5). Training loss: vanilla loop is lowest EARLY; DiscoLoop passes it at about 13B tokens and leads by at most about 1e-2 in global loss at the end.
- 2026-10-20: failure modes and limits the authors give: moderate scale only; no length generalization test; no continual-pretraining test; one seed per row as far as the text states. The LM table has no variance.

## 3. Our speculation

- 2026-10-20 Speculation: at 5k steps (about 31M tokens) we sit in the regime where the paper's vanilla loop is AHEAD (DiscoLoop crosses at about 13B tokens, 400x our budget). A 5k K1-K6 check would probably read the adaptation cost, not the benefit.
- 2026-10-20 Speculation: the mismatch story fits our "span states carry context in ONE pass" finding only loosely. A coarse or span state is a latent that is not a token, so a decode-then-encode channel has no natural target there. It fits the plain token-level Parcae loop and the pointer-head arms better, where the pass output is a next-token distribution.
- 2026-10-20 Speculation: cost. The extra projection is a V x d matmul per pass per position. With the StarCoder2 vocabulary (49k) that is a large add to the GPU-bound step, and training speed is our top priority. A top-k or sampled-softmax Phi would be needed.
- 2026-10-20 Speculation: the Stage-1 collapse (loop 1 stops answering two-hop alone) is the same kind of event as K1-K6 growing: the model stops doing all work in pass 1. It is a useful probe for a 2-hop synthetic panel on the testbed.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper: the late-training loss lead appears only after 13B tokens. Our rule (5k K1-K6 is the go/no-go, "K1-K6 grows with tokens") would reject a design whose benefit starts that late. Do not read a flat 5k K1-K6 as a verdict on this method.
