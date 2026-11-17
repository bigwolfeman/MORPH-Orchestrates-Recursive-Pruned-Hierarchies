# Paper note: LPT (shared first-pass memory in looped transformers)

- **Title:** The Surprising Effectiveness of Shared Memory in Looped Transformers (arXiv 2610.02383v1, IBM and Cornell)
- **arXiv or URL:** arXiv 2610.02383
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Later passes read the first pass's full KV cache. Close to our "private depth" question (item 3): does a pass need its own tokens' deep state, or can it read pass 1?

## 2. What the paper shows

Read 2026-11-17 via a summarising fetch of arxiv.org/html/2610.02383 (first 100k of 111k characters).

- 2026-11-17 (method, Sec 3): pass 1 (state recursion) writes a full-context KV cache. Later passes (prediction recursions) attend to that shared cache plus a sliding window of their own cache (omega = 64), one softmax. H-LPT does the same on the full-attention layers of a hybrid with Gated DeltaNet layers.
- 2026-11-17 (models): 151.9M, 454.2M, 1.01B; FineWeb-Edu 20B tokens (S, M) and 50B (L); context 4096; 2, 3, 5 recursions (Sec 4, Table 2). The intro says "11B", Table 2 says 1.01B.
- 2026-11-17 (gain vs control): LPT vs loop-all-tokens LT at the same recursions: val ppl lower by 0.38 to 1.09, accuracy +0.4 to +4.8, 2.0 to 4.7x less KV memory, under 1 percent more FLOPs. S, R = 5: 14.91 vs 16.00 (Table 1). LT stops improving after 3 recursions, LPT keeps improving through 5. Controls: standard R = 1 model (the depth-1 reference), parameter-matched deeper-narrower model. No equal-FLOP control (Sec 8).
- 2026-11-17 (answer to question a, first part): they do NOT freeze the first-pass cache. The closest test is detaching gradients through the later reads (Sec 6, App D, Table 6). Detached LPT is much worse: 33.21 vs 14.91 ppl at S, R = 5; at M, R = 3 it fails to train. At S, R = 3 it is 16.14 vs 16.32 for LT. So the gain depends on gradient flowing from later passes into the first-pass cache.
- 2026-11-17 (question a, second part): training cost not reported. Savings are inference memory and decode FLOPs. FlexAttention skips masked blocks; nothing on training compute, memory, or wall clock.
- 2026-11-17 (timing): end-of-training numbers only. No curve of when gain appears.
- 2026-11-17 (window ablation, Fig 4): LPT beats LT at every window, even omega = 0.

## 3. Our speculation

- 2026-11-17 Speculation: LPT works because pass 1 is a trained producer for later passes (the gradient is the point). That is shared work, not private depth. It is a different claim from our item 3 and does not contradict it: their later passes may gain by reading pass-1 neighbours, but the gain also needs pass 1 to be trained for that job. Our frozen-neighbour test froze neighbours in a model that was not trained to read them.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 3: partly. LPT gets a lead from reading neighbours' pass-1 state, where we saw other tokens gain about 0 from reading a deep state. The key difference is that LPT trains for it, and the neighbour state is pass 1 (shallow), not deep. Worth a direct check.
- Item 9: the detach ablation agrees that gradient flow through the shared read matters.
