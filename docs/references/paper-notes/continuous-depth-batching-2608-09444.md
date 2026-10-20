# Paper note: Continuous depth batching for depth-adaptive looped LM inference

- **Title:** Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching (Schwethelm, Rueckert, Kaissis; TUM, Imperial, HPI)
- **arXiv or URL:** 2608.09444 (v1 2026-08-10, v2 2026-09-25). Type: paper.
- **Local paper text:** docs/references/looping-depth/continuous-depth-batching-2608-09444/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: Serving-side scheduler for looped LMs with per-token exit (Ouro 1.4B, Huginn 3.5B). Decode speed is NOT our priority and we train, not serve. Noted so nobody reads it twice.
- 2026-10-20: Read abstract, intro, the speedup bound and latency model sections, and the experiment headings. Online-serving tables and appendices were not read.

## 2. What the paper shows

- 2026-10-20: tokens exiting at different loop steps break a uniform batch. CDB forms new batches between loop steps, refills freed slots with new tokens, keeps looped KV caches, and predicts exits early to prepare batches asynchronously (Algorithm 2; minimum coda batch size K).
- 2026-10-20 (Sec. 4 to 5): the speedup bound is set by mean exit depth and the fraction of compute outside the loop; fully looped Ouro reaches up to 99 % of the estimated maximum speedup, Huginn (non-looped prelude and coda) 83 to 96 %, where refill may cost more than it saves. Recurrent-step latency is flat (weight-reload bound) below a saturation batch size B*(L) and linear above it.
- Failure mode: large non-looped parts (embedding, LM head, unshared blocks) slow and complicate scheduling.

## 3. Our speculation

- 2026-10-20 Speculation: none for training. Our Poisson draw is per row at train time, not per-token exit.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None found.
