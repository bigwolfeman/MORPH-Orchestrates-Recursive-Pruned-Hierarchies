# Paper note: Denoising Recursion Models (one step forward, K steps back)

- **Title:** One Step Forward and K Steps Back: Better Reasoning with Denoising Recursion Models (Chris Cameron, Wangzheng Wang, Nikita Ivanov, Ashmita Bhattacharyya, Didier Chetelat, Yingxue Zhang; 2026, paper)
- **arXiv or URL:** 2604.18839 (v1, 2026-04-20). Read the main text through Section 6 (Table 1, Sections 3 to 5). Appendices (F.x ablations, Sudoku and Maze) not read.
- **Local paper text:** docs/references/looping-depth/denoising-recursion-models-2604-18839/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It states the credit-assignment argument in loop terms: backpropagating through a window of recursions gives multi-step credit assignment, and a noise level sets how hard the start state is. It claims this replaces deep supervision in TRM.

## 2. What the paper shows

- 2026-10-20 (Sec 3.2, Eq 6): DRM starts the loop from the embedded corrupted target (masking, cosine schedule) and trains k recursion steps against the clean target, loss only at step k. Inference starts fully masked and re-masks between steps. SPRM instead noises the hidden state at truncation points.
- 2026-10-20 (Table 1, ARC; TRM codebase, 7M params): ARC-Easy no pretrain: TRM 45.7, DRM 50.5, SPRM 50.7. ARC2-Eval no pretrain: TRM 6.3, DRM 9.6, SPRM 10.1. With NVARC pretraining: TRM 12.5, DRM 16.7. 14M: TRM 21.8, DRM 24.9, 4B LLM (NVARC) 22.6 (same data).
- 2026-10-20 (Sec 5): TRM without deep supervision drops 45.7 to 34.7 on ARC-Easy; DRM with no deep supervision beats TRM. Standard masked diffusion with no recursion gets about 0 % (7M) and under 1 % (70M); a 70M plain transformer gets 25.5 %. The authors say the core ingredient is backpropagation through a recursion window, and DRM adds a modest gain over that.
- 2026-10-20 failure modes: DDPM on embeddings failed (embeddings learn to be recoverable from noise); SPRM trails TRM on the reARC pretrain (11.8 vs 12.4 at 7M, 17.4 vs 18.8 at 14M); confidence-based re-masking gave no gain; significance only on a few cells (permutation test p < .05 marked); evaluation aggregates predictions over all checkpoints and 1000 augmentations; all tasks are fixed-size grids.

## 3. Our speculation

- 2026-10-20 Speculation: not a next test. The corruption level is a depth curriculum in disguise (easy-to-hard start states), and depth curricula are on our closed list. Its one transferable claim, "backprop through the recursion window is the ingredient", agrees with our l2cap full-BPTT finding.
- 2026-10-20 Speculation: for span-CALM, masking 15 % of tokens before the encoder (already in our setup) is the nearest analogue of the forward corruption.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper: removing deep supervision costs TRM 11 points (45.7 to 34.7). Ours: no deep-supervision arm earned depth on this testbed. Different tasks (ARC grids vs OpenWebText); no conflict is shown.
