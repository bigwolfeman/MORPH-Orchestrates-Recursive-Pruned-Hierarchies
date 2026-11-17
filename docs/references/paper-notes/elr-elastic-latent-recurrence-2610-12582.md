# Paper note: Elastic Latent Recurrence (ELR)

- **Title:** One Model to Loop Them All: Elastic Latent Recurrence for Language Models (arXiv 2610.12582v1, NVIDIA)
- **arXiv or URL:** arXiv 2610.12582
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Names the "one-loop tax" (a looped model's depth-1 result is worse than a flat model's). That is our depth-1 twin problem seen from the other side.

## 2. What the paper shows

Read 2026-11-17 via a summarising fetch of arxiv.org/html/2610.12582 (first 100k of 125k characters).

- 2026-11-17 (models): GDP-Hybrid-4B dense, 1T tokens; GDP-Hybrid-30B-A3B MoE, 300B tokens; GPT-0.5B ablations, 300B tokens; Koi-4B product model (7T tokens).
- 2026-11-17 (recipe): input / shared recurrent / output blocks. Gated adaptive-delta update with learned per-token per-channel step (init about 0.35). Mixed-depth training. Loop-conditioned modulation, zero-initialised. Delayed mixed-depth (train 1 loop first, switch later). Route replay for MoE. Last-loop-only KV cache (Sec 3).
- 2026-11-17 (one-loop fix, Sec 3.2, 6.4): a fraction f of data-parallel ranks runs ONE loop on every microbatch. Without it, one-loop loss is 0.0225 above the baseline at GPT-0.5B. f = 0.15 is the smallest tested value that beats baseline. The big runs use f = 0.10. Sampling the same one-loop marginal without the override stays above baseline, so the cause is not the marginal alone.
- 2026-11-17 (depth-1 vs full depth): larger f buys one-loop gain at the cost of deeper loops (Sec 6.4). So fixing depth 1 DOES move the full-depth result, and the direction is a trade, not a free win.
- 2026-11-17 (gain): 4B: 1 loop 65.13 vs flat 65.65; 1 to 4 loops +7.25 points. 30B-A3B: 1 loop 66.08 vs flat 65.04; 1 to 4 loops +4.75. Two and three loops capture 79 and 96 percent of the 4B gain. Control: non-recurrent, matched params, data and tokens (Sec 4.1). The intro and abstract quote different quantities (margin over baseline 6.7 / 5.8 vs 1-to-4 gain); the paper does not reconcile them.
- 2026-11-17 (cost): mixed depth throughout = 1.62x layer compute at 4B; delayed switch at 75 percent = 1.16x (GPT-0.5B: 1.79x vs 1.20x). Route replay cuts update time 11.7 percent. Last-loop KV: KV memory 1x not 2.5x, score change 0 to about -3 points (Sec 5.3).
- 2026-11-17 (not read): last 25k characters, which may include more ablations.

## 3. Our speculation

- 2026-11-17 Speculation: "delayed switch" is a cheap way to get a depth-1 model that is already good, then let the loop add. It costs 1.16x, not 1.6x. Whether the loop then earns at full depth is the open point. Their data says a trade exists.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- Item 2 (load is not depth): the one-loop override works because a depth-1 route can do the job. It is the same fact. Overrides move load to pass 1.
- Item 7: delayed switch matches "value forms late", but they did not measure gain over time. Their evidence is end-of-run.
