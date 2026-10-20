# Paper note: Hidden Decoding at Scale (sequence-length scaling instead of loops)

- **Title:** Hidden Decoding at Scale: Latent Computation Scaling for Large Language Models (WeChat AI Team, Aiwei Liu and 46 others; 2026, paper)
- **arXiv or URL:** 2607.08186 (v1, 2026-07-09). Read main text through Section 6 (Tables 2 to 6, Figures 4 to 8). Appendices B to I not read.
- **Local paper text:** docs/references/looping-depth/hidden-decoding-2607-08186/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is a no-loop way to buy per-token compute, built because loops do not fit pipeline parallelism. Each token becomes n streams with their OWN embedding tables, only the last stream takes the next-token loss, and the earlier streams' KV stays as context. Our particles result says copies need a persistent learned identity; this is that, at 617B.
- 2026-10-20: Its Table 5 is a measured case of our rule "only the main loss buys depth".

## 2. What the paper shows

- 2026-10-20 (Sec 2.1, Eq 2): n embedding tables E_1..E_n interleave into a length-nL sequence; loss only on stream n. Cyclic replication init when doubling n (E_{n+k} <- E_k). Only embeddings grow (Qwen3-8B: 1.2B to 3.1B at n = 4).
- 2026-10-20 (Sec 2.2, Table 4, 21B MoE, 27 layers): Stream-Factorized Attention makes most layers attend within a stream and mixes streams at few layers. Average over 13 benchmarks: baseline 54.33, 1 full cross-stream layer 55.88, 4 full layers 56.49, all 27 full 56.95. BBH 67.0 to 72.0 at 4 layers.
- 2026-10-20 (Sec 4.2, Fig 4a): a 4x longer sequence costs 5.1x per batch at 80B (256k to 1M) and 4.4x at 617B (32k to 128k), versus 16x for dense attention. Table 2 (early SFT, matched non-HD): GPQA-D 89.1 to 91.2, HLE 33.6 to 35.4 at 617B; SciCode 45.8 to 50.0 at 80B. HD runs on about 5.3 % of the full token path in continued pretraining (Table 1).
- 2026-10-20 (Sec 4.3): Qwen3-8B mean gain grows with n (about +1.9, +2.8, +3.4 for n = 2, 4, 8, read from Fig 4b); 80B MMLU 85.1 (n=1) to 87.5 (n=8), Pile-test BPB 0.386 to 0.378.
- 2026-10-20 (Sec 5.1, Table 5, 6B MoE, n = 2): loss baseline 1.908; all-token loss 1.880; sum of streams 1.877; HD (last-stream loss only) 1.874; HD n = 3 1.857. Loss on every stream is WORSE than loss on the last stream only.
- 2026-10-20 (Sec 5.2, Table 6): shared KV across streams (PHD-like) drops the 6-task average 64.23 to 63.46 (1 full layer) and 64.66 to 63.90 (4 full layers). Probes at n = 8: same-token stream cosine 0.987 at input, 0.637 mid-network, 0.783 at the last layer; the final stream puts up to 0.52 attention mass on stream 0; intermediate-stream LM-head probes disagree with the final top-1 in up to 63 % of positions and have higher entropy (final 2.09 bits).
- 2026-10-20 (Sec 4.5, Fig 5): decode throughput is 83 to 88 % of baseline at batch 1 and 27 to 61 % at batch 16 to 32 on long inputs. Decode speed is not our concern.
- 2026-10-20: limits stated by the authors: all gains are against matched non-HD models after SFT only; the 617B long-context stage used fewer HD tokens than the baseline; no compute-matched comparison against a looped model; no variance on the pretraining tables.

## 3. Our speculation

- 2026-10-20 Speculation: this is a FLOP-for-sequence trade where a loop is a FLOP-for-depth trade. At the same extra FLOPs, per-stream KV gives later tokens access to earlier tokens' intermediate states, which a plain Parcae loop with a diagonal injection does not give (the loop re-reads its own pass state at the same position only).
- 2026-10-20 Speculation: Table 5 backs our rule from the other side. Directly supervising the helper streams hurts. It also says the helper streams need no per-pass target to be useful; the main loss reached them through attention.
- 2026-10-20 Speculation: for training speed it is not free. Cost per batch is 4.4 to 5.1x at n = 4, so the extra compute is about as large as a loop of depth 4, with no sequential dependency. On one 5090 it would use the GPU more evenly than a Poisson loop (no per-row active-set shapes), which may matter given our host-serial step.
- 2026-10-20 Speculation: it supports "identity matters": the streams differ only by their embedding tables at the start and drift apart to cosine 0.64 by mid-network.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper: more streams keep helping up to n = 8 at 8B and 80B scale. Our cell-count run found no arm gains above 4 cells (`lab/experiments/successes/2026-10-13-fan-variable-cell-count-v1.md`). The two are not the same object (streams are full token copies, our cells are slot states), so this is a prompt for a test, not a contradiction.
