# Paper note: HybridAgents (latent reasoning steps)

- **Title:** HybridAgents (Coconut-style latent steps with an adapter and self-conditioning)
- **arXiv or URL:** arXiv 2610.18861
- **Local paper text:** none archived
- **Date first noted:** 2026-10-14

## 1. Why we reference it

- 2026-10-14: Wolfe asked how close it is to our work. Its latent steps are close to our think-once TUL decode idea (cheap decoding after one thinking pass).
- 2026-10-14: It supports our note that teacher forcing is a bypass.

## 2. What the paper shows

Read on 2026-10-14. Section and table numbers not recorded.

- 2026-10-14 (mechanism): Last hidden state h goes through HybridLink, R(h) = h + W2 gelu(W1 h), and then becomes the next input embedding. Each latent step is one full forward pass (a time axis, not depth). A latent run exits when the nearest token of R(h) is the end-of-latent token. Runs are 16 to 256 steps.
- 2026-10-14 (SFT target): R(h) regresses onto the embeddings of the real chain-of-thought tokens (cosine plus MSE). Then a self-conditioned second pass runs on its own latents (+8.7 AIME25 at 14B).
- 2026-10-14 (adapter ablation): Identity adapter 71.4 vs 88.2 on MATH500 (wording from our memory note; check the table before citing).
- 2026-10-14 (parity): SFT parity with text SFT. 14B MATH500 84.2 vs 83.8. 4B mixed.
- 2026-10-14 (snap): Snapping each latent to its nearest token costs only 1 to 2 points. The latents are near tokens.
- 2026-10-14 (compute): 51.2 to 13.9 TFLOP per question at 4B, from fewer steps.
- 2026-10-14 (RL): Latents are recorded and replayed. The ratio covers text tokens only. HybridLink is frozen, so credit does not flow through latents.

## 3. Our speculation

- 2026-10-14 Speculation: Their latents have a text target. Our OWT pretraining has no such target, so the same method may not transfer to our setting as is.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None recorded yet.
