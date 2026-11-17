# Paper note: BeNice! (KV-sharing for looped models)

- **Title:** BeNice! Closing Looping's Inference-Memory Gap via KV-Sharing (arXiv 2610.31633v1, Prairie, Ryan, Berg-Kirkpatrick, Fu)
- **arXiv or URL:** arXiv 2610.31633
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Parcae-style models (same family as ours). Tests whether later loops can share one KV cache without losing quality.

## 2. What the paper shows

Read 2026-11-17 via a summarising fetch of arxiv.org/html/2610.31633 (first 100k of 170k characters).

- 2026-11-17 (models): 140M, 370M, 770M, 1.3B, Parcae-style, over 100B climbmix tokens (Sec 5.1). Huginn 3.5B and Ouro appear only as comparisons.
- 2026-11-17 (pretraining, Sec 4.1): Poisson depth per sequence, truncated BPTT depth floor(mu_rec/2). Stochastic vs static depth (Table 2): shared-KV quality drop is about 0.01 to 0.02 ppl for stochastic and 1.31, 2.70, 3.10, 2.84 for static (140M to 1.3B). Depth variability itself helps shared-KV.
- 2026-11-17 (failure, Sec 3.2, Table 3): sharing fails at the first layer of the first recurrence, a token-independent basis shift. At 770M: unshared 11.86, naive patching 11.87, shared 34.24.
- 2026-11-17 (fix, Sec 4.2): post-training loss CE(final) + gamma * CE(first-recurrence queries and residuals with final-recurrence KV). Shared vs unshared drop: 770M standard finetune +9.75 ppl, BeNice +0.01; 1.3B +7.73 vs -0.01 (Table 5). Core benchmark gap 1.3B: -3.33 to +0.04 (Table 4).
- 2026-11-17 (question a): later passes read the FINAL pass's KV (not a frozen first-pass cache). The paper does not say "loop gain" and does not test a first-pass freeze. After the fix, quality with sharing is within noise of unshared. So a read of a shared cache keeps quality, but only after a trained fix. Appendix J (LoRA) was not in the text we got.
- 2026-11-17 (cost): inference only. 1.3B, batch 64, ctx 2048: 63.82 to 21.60 GB (-66.2 percent); serving at 2 req/s: throughput +23.6 percent, p90 TTFT 139.2 to 23.2 s. No training-cost comparison. The two-term loss adds cost, unmeasured.
- 2026-11-17 (inconsistency): Sec 5.3 gives both 22 and 2 req/s for the cache-hit result.

## 3. Our speculation

- 2026-11-17 Speculation: the aux loss here is a post-training patch for cache sharing, not a depth-buying term. Its CE cost is not reported, so item 9 is untested here.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None. It does not test token-private gain (item 3). Nothing to say on items 1, 2, 4 to 8.
