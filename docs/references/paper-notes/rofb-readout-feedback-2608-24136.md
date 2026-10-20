# Paper note: RoFB, Readout Feedback for steering recurrent reasoners at inference

- **Title:** Steering Recurrent Reasoners at Inference Time with Readout Feedback (Kamiya, Koyama, Jeong, Uchiyama, Kubo, Hayashi, Suzuki, Matsuo)
- **arXiv or URL:** 2608.24136 (v1 2026-08-25, v2 2026-10-02). Type: paper.
- **Local paper text:** docs/references/looping-depth/readout-feedback-rofb-2608-24136/ (paper.txt, paper.pdf 3.8 MB, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: A test-time-only intervention on small recurrent puzzle solvers (AKOrN, ItrSA++, TRM) that couples tokens by their readout probabilities. It is not about language modelling and not about training. Included for completeness of the recurrent-reasoner picture.
- 2026-10-20: Read abstract, Sec. 1, Sec. 5.1 (performance), Sec. 6 and conclusions. Mechanism sections (5.2) and appendices only grepped.

## 2. What the paper shows

- 2026-10-20 (Sec. 5.1, Fig. 3): RoFB helps in 4 of 6 model-task pairs (AKOrN on Sudoku and Maze, ItrSA++ on Maze, TRM on Sudoku), +0.6 % on ItrSA++ Sudoku, nothing on TRM Maze (both saturate). ItrSA++ on Maze: 83.4 % at (Kvote=1, T=128) vs best baseline 82.1 % at (64, 128), about 1/62 of its FLOPs. TRM on Sudoku: 68.4 % to 74.2 % at 16 outer iterations. Per-step cost within +20 %.
- 2026-10-20 (Sec. 5.2 summary): feedback acts as repulsion (leave unsolved states) plus readout-dependent weighting (retain solved states); random direction control of equal norm does not help (TRM 74.7 vs 74.2).
- Failure modes (Sec. 6): no validated predictor of when it helps; confidence and class separation can be high for wrong answers (45 of 97 confident terminal predictions incorrect on a case); not tested on language models, where a large vocabulary may dilute the coupling.

## 3. Our speculation

- 2026-10-20 Speculation: weak link only. The mechanism needs a small class set per token and no training; our losses and decoders work on 49k-vocab tokens and spans. No test follows.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None found.
