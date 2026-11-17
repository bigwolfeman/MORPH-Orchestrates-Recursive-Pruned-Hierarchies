# Paper note: Contrastive latent representation learning with denoising score matching (ScoreCLR)

- **Title:** Contrastive latent representation learning with denoising score matching (ScoreCLR)
- **arXiv or URL:** arXiv 2611.18469
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Checked as a possible way to train a per-sentence latent without collapse and without large batches.

## 2. What the paper shows

Read 2026-11-17 from the arXiv HTML. The fetch tool read only the first 100k characters, so late appendices were not read.

- 2026-11-17 (Sec. 3.1, 3.3, Eq. 1): Loss = invariance term between two augmented views, a repulsion term from a learned score (denoising score matching at random noise levels), and a Gaussian prior (L2).
- 2026-11-17 (Sec. 4.1, App. C.2): ResNet-18 encoder, small unconditional score MLP (0.28M params). Pretraining on CIFAR-10 and ImageNet-100 only. No text.
- 2026-11-17 (Table 1): Linear probe 88.87 vs SimCLR 89.26 on CIFAR-10. 71.23 vs 72.39 on ImageNet-100. Slightly worse.
- 2026-11-17 (Table 3): Effective rank of the projector output 229/256 vs 96 for SimCLR (CIFAR-10). Collapse resistance is real on images.
- 2026-11-17 (Sec. 4.7, Fig. 1): Repulsion cost is linear in batch, InfoNCE quadratic. 7.6 ms vs 228 ms at B=32768. Accuracy does not improve at large batch, and the score net needs more steps as batch grows (Table 5).
- 2026-11-17 (Table 9): The score network is unconditional. Conditioning it on the other view cost 3.3 to 5.6 points.
- 2026-11-17: The regression target is noise, fresh each query. The ENCODER target is still the other view, so the invariance term is a deterministic pull.

## 3. Our speculation

- 2026-11-17 Speculation: It could keep a sentence code spread out at small batch (the score term is an entropy push, cost linear in batch). But it is a view-invariance method. A sentence latent in TUL needs a target that depends on the PAST and predicts the FUTURE. A deterministic predictor of the next-span code still meets the invariance term in one pass, so it would not force depth. The stochastic part is only the noise inside the repulsion, not the prediction target.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-11-17: Agrees with item 5 and item 9. No result here that a loop earns on it. The paper's own conditioning ablation says a conditional score hurts.
