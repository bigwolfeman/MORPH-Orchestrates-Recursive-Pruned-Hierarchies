# Paper note: GRAM (Generative Recursive Reasoning Models)

- **Title:** Generative Recursive Reasoning (Junyeob Baek, Mingyu Jo, Minsu Kim, Mengye Ren, Yoshua Bengio, Sungjin Ahn; NeurIPS 2026, paper)
- **arXiv or URL:** 2605.19376 (v1 2026-05-19; abs page shows v4 2026-10-15). Read the main text through the ablation table (Sections 1 to 4.4, Tables 1 to 3, Figures 3 to 5). Appendices not read.
- **Local paper text:** docs/references/looping-depth/gram-generative-recursive-reasoning-2605-19376/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It puts a Gaussian transition on a looped state, trains it with an ELBO whose posterior sees the target, and uses K parallel trajectories at test time. That touches three of our dead ends: learned-sigma stochastic loops, deep supervision, and teacher-fed posteriors.

## 2. What the paper shows

- 2026-10-20 (Sec 2, Eq 4 to 9): z_t = u_t + eps_t, eps_t ~ N(mu(u_t), sigma^2(u_t)); noise only on the high-level half of an HRM/TRM-style (h, l) state. Training uses deep supervision over N_sup steps, gradient only through the last transition, loss = CE + KL(posterior head || prior head). The posterior head sees the target y; inference samples the prior and majority-votes N trajectories.
- 2026-10-20 (Sec 4.1, Fig 3, Fig 4): Sudoku-Extreme, 3-seed mean: GRAM 86.6 at N = 1 vs TRM 83.6; 93.6 at N = 5, 96.1 at N = 20. ARC-AGI-1 50.3 vs TRM 44.6; ARC-AGI-2 9.5 vs 7.8. At near-equal FLOPs: GRAM 96.1 % at 8.48 TFLOPs vs TRM 92.1 % at 8.19 TFLOPs (320 iterations).
- 2026-10-20 (Table 1): 8x8 N-Queens accuracy at N = 1: GRAM 99.5, TRM 66.8, HRM 78.7, AR 96.3, MDLM 96.1; 10x10: GRAM 89.7, TRM 17.5. Deterministic recursive models recover at most 26.7 % of valid solutions; GRAM 89.7 % at N = 20.
- 2026-10-20 (Table 3, read from text extraction, layout is uncertain): adding stochastic guidance to the plain looped TF without deep supervision gives 65.1 vs 61.7 on Sudoku; the full model with deep supervision and hierarchy gives 86.6 / 96.1. "Stochasticity only" gives 96.0 on Sudoku but 62.1 on N-Queens at N = 20; "guide only" gives 0.0.
- 2026-10-20 failure modes the authors report: needs the hierarchical TRM layout plus deep supervision for the full gain; truncated gradient makes the ELBO a biased surrogate (their words); all tasks are small fixed-size grids; no language model result.

## 3. Our speculation

- 2026-10-20 Speculation: its single-sample gain over TRM is a training effect (stochastic states seen during training), which is the part we have tried: learned-sigma loops died on the likelihood term in our runs. GRAM survives because the KL ties the prior to a posterior that sees y, which is the teacher-forced bypass we already filed as a failure for the ELBO stochastic loop.
- 2026-10-20 Speculation: the useful piece for us is the width axis (N samples, majority vote), which is the same object as our fan or particles. At 96.1 % vs 92.1 % at matched FLOPs it beats depth on a puzzle with several valid paths. Web text next-token has one label per position, so the voting step has no analogue unless the target is a span with many valid continuations.
- 2026-10-20 Speculation: no direct next test. Score 1.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper: deep supervision is part of the recipe (TRM base). Our rule: no pass-indexed depth losses, and deep supervision on this testbed is a closed dead end. The paper does not ablate it in the part we read.
