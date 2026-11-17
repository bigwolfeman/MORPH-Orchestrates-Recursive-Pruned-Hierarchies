# Paper note: LoopCD (contrastive decoding for looped transformers)

- **Title:** Decoding Looped Transformers Better for (Almost) Free (arXiv 2610.02185)
- **arXiv or URL:** arXiv 2610.02185
- **Local paper text:** none archived
- **Date first noted:** 2026-10-02

## 1. Why we reference it

- Full reading is in [`../looping-depth/2026-10-02-loopcd-contrastive-decoding.md`](../looping-depth/2026-10-02-loopcd-contrastive-decoding.md). Verdict there (Wolfe, 2026-10-02): decode-only, no training savings, kept for insights. This note adds only what that file does not hold.

## 2. What the paper shows (additions only)

Re-read 2026-11-17 via a summarising fetch of arxiv.org/html/2610.02185 (first 100k of 149k characters).

- 2026-11-17: the first-iteration prediction trails the final prediction by 4.0 to 23.7 points on the seven-benchmark mean. Accuracy rises steadily with iteration.
- 2026-11-17: divergence from the final prediction: Ouro-1.4B 0.71 bits after pass 1, 0.013 after pass 3. Huginn 3.0 nats after step 1, 0.04 after step 16.
- 2026-11-17: Parcae starts from a deterministic state, so its first state is usable in both guidance spaces. Parcae-1.3B ARC-Challenge LoopCD-Logits gain falls from 3.50 to 1.02 points as the reference pass moves deeper. No per-loop CE or ppl curves in the paper.
- 2026-11-17: limits: FLOPs are arithmetic counts, not wall clock. Negative contrast strength hurts. The paper suggests models trained with intermediate predictions in mind (Ouro) may gain more. No training-time change reported.

## 3. Our speculation

- None added.

## 4. Corrections

- None.

## 5. Where our findings disagree

- Item 9: the paper's idea that pass 1 and pass K disagree enough to contrast is a statement that pass-1 and deep states differ. It uses no aux loss, so it does not conflict.
