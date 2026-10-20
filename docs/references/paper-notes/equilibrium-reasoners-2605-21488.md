# Paper note: Equilibrium Reasoners (EqR): Learning Attractors Enables Scalable Reasoning

- **Title:** Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning (Benhao Huang, Zhengyang Geng, Zico Kolter, CMU)
- **arXiv or URL:** 2605.21488 (v1, 2026-05-20); code at github.com/locuslab/EqR (not read)
- **Local paper text:** docs/references/looping-depth/equilibrium-reasoners-2605-21488/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20
- **Triage score:** 2. Residual-vs-CE diagnostic and RI/NI-at-test-time give a no-training probe; SOT conflicts with our no-aux-loss rule; puzzles only.

## 1. Why we reference it

- 2026-10-20: It is the cleanest published recipe for training a weight-tied latent loop to run far past its training depth (16 trained, 1024 tested). It also gives a free diagnostic, the fixed-point residual, and a statement of when depth and breadth help. Our particles filing (K hypotheses, collapse, no depth) and the "depth is per-token refinement" finding are the two things it speaks to.
- 2026-10-20: All results are puzzle tasks (Sudoku-Extreme, Maze, Mini-ARC) with exact-match accuracy on a fixed grid. There is no language-model result. Treat it as a mechanism paper, not as evidence about OpenWebText loops.

## 2. What the paper shows

Read in full: main text and the appendix sections on training dynamics (A.1, A.2) and the tables. Appendices A.3 to A.6, C, D were read only by grep; the Mini-ARC and Transformer-backbone transfer (A.5) and seed diagnostics (A.6) were not read.

- 2026-10-20 (Table 1): at matched weights, feedforward models stay at 1.8 to 2.6 % on Sudoku-Extreme (4, 16, 64 layers); weight-tied iteration reaches 84.8 % (TRM) and EqR 99.8 %. Maze: 0.0 % feedforward, 44.9 % TRM, 93.0 % EqR.
- 2026-10-20 (Table 2, Sudoku, one run per row): vanilla feedforward 2.6 eval; weight-tied 32.6; plus segmented online training (SOT) and depth x16 74.7; plus hierarchical recurrence 76.5; plus ACT training 84.8. Trajectory supervision (losses at many iterations, one optimizer step at the end) at 16x gives 47.1 vs SOT 74.7 (Table 8c). SOT = supervise the end of each segment and take an optimizer step right away, carry detached. In-segment gradient truncation costs 74.7 to 67.2 with a single latent (Table 8d).
- 2026-10-20 (Table 3): randomized initial state (RI) lifts Maze 44.9 to 68.6 and Sudoku 84.8 to 86.0; RI plus per-step noise (NI, beta 0.01, damping 0.05) gives Sudoku 86.4 and Maze 82.2 at D=16, B=1.
- 2026-10-20 (Table 4, Fig 3): depth 16 to 64 at B=1: Sudoku 86.4 to 93.0, Maze 82.2 to 88.9. Adding breadth B=128 restarts: 99.8 and 93.0. Breadth helps only once depth is at least about 4 outer steps (168 layers unrolled). Lower fixed-point residual tracks lower error (Fig 1).
- 2026-10-20 (Fig 8): picking the restart with the smallest residual over the last 3 iterations beats majority vote only after RI+NI training. For the plain TRM baseline the residual can mark a spurious attractor and top-1 convergence loses to majority vote.
- 2026-10-20 (Table 5): an ACT head cuts mean function evaluations from 1024 to 58.7 at D=1024 (accuracy 96.1 to 95.3).
- Failure modes reported: a low residual only certifies convergence, not correctness (Eq. 7 needs a correct attractor and a positive logit margin); no correct attractor means neither depth nor breadth helps (Fig 6a); learnable initializers did not beat a fixed Gaussian on held-out accuracy (Table 10).

## 3. Our speculation

- 2026-10-20 Speculation: the residual diagnostic is free for us. We already run eval depth 1 to 6. Computing ||f(z)-z|| at each depth on the 480 fixed rows, and correlating it with per-row CE at depth 6 minus depth 1, tests whether our Parcae loop is in the paper's "aligned landscape" regime or in a regime where residual falls and CE does not.
- 2026-10-20 Speculation: our contractive diagonal injection already makes the pass map close to a fixed-point iteration, so a small residual may be built in (the 0.87 floor note in CLAUDE.md). A residual that falls only because the injection decays the state would look like convergence and carry no information. The test must separate "residual falls" from "CE falls".
- 2026-10-20 Speculation: SOT is a form of per-segment credit assignment with more optimizer steps, not a new loss. For us a segment is a pass group. It is close to the dead-end "deep supervision (HRM-style)" and to "pass-indexed auxiliary losses", because every segment end carries the loss. Do not run it as a depth lever. Its one clean lesson is the 47.1 vs 74.7 gap: stale-parameter trajectories hurt, so any multi-loss loop scheme should update between segments.
- 2026-10-20 Speculation: RI plus NI is the same family as our dead "noise-seeded particles". The difference is that EqR uses breadth only at test time, with the same trained map, and selects by residual. Our particle failure was collapse during training. This paper does not address collapse of persistent K states.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: depth far past training (16 to 1024 iterations) keeps lowering residual and error. Our measurement: eval depth beyond the trained Poisson mean 6 (max 8) is not where our K1-K6 lives, and our loop gains are small (+0.0377 plain). The paper's tasks have one exact answer per grid; ours is a token distribution. No contradiction, but no transfer either.
- 2026-10-20: Paper claim: truncated gradients with detached carry plus SOT beat long backprop. Our truncated BPTT depth 4 vs full BPTT history (l2cap recipe uses full BPTT) points the other way for slot loops. The paper changes the optimizer step, which we did not.
