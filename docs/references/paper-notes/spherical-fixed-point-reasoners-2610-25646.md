# Paper note: Spherical Fixed-Point Reasoners (SFPRM): adaptive loop halting on the hypersphere

- **Title:** Spherical Fixed-Point Reasoners: Adaptive Loop Halting on the Hypersphere (Zela et al., EPFL / Lamarr / Amazon AGI)
- **arXiv or URL:** 2610.25646 (v1, 2026-10-19). Type: paper. Read: full main text (Sections 1 to 5), Appendix C.6, C.7 (Table 6), D (limitations) and the halting text of C.2. Proof appendices A.1 to A.7 and the hyperparameter tables were not read line by line.
- **Local paper text:** docs/references/looping-depth/spherical-fixed-point-reasoners-2610-25646/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It is the most direct statement we have of a looped block with an injection term whose strength decides whether the loop has one attractor. Our core loops with a contractive diagonal injection, and our own measurement says the slot map's 0.87 is the injection floor. This paper has a theorem and a sweep for the same trade.
- 2026-10-20: It is a puzzle paper (Sudoku, Maze, ARC). It has no language-model result. Treat it as mechanism evidence, not as a CE result.

## 2. What the paper shows

- 2026-10-20 (Section 3.2, eq. 5 to 10): SFPRM is the Fixed-Point Reasoner (FPRM, Movahedi 2026) on the nGPT unit sphere. Per pass: pull the state a fraction lambda along the arc to the normalized input (Slerp), apply a weight-tied nGPT block, then rotate a fraction eta toward the proposal (geodesic damping; eta shrinks by gamma when the residual stalls for P passes). Halt when the worst-token chordal residual 1 - cos(z, z~) is under 0.01 (8.1 degrees). Training is deep supervision with truncated BPTT over K-step segments and a detach between segments, copied from FPRM.
- 2026-10-20 (Theorems, Section 3.2.1): a fixed point exists for any lambda in (0,1) with no contraction assumption (Brouwer on a cap). Angular damping converges locally if every Jacobian eigenvalue has Re < 1 and eta is under 2(1-Re)/|zeta-1|^2 (min over eigenvalues). Strong injection (lambda above a lambda_0 that depends on f and x) gives a unique global attractor. Damping cannot fix modes with Re >= 1; "training must remove" those (Remark 1).
- 2026-10-20 (Table 2, Section 4.1): 6.83M-parameter backbone, single seed per run ("every run uses a single seed", Appendix D). Sudoku-Extreme 96.6 vs FPRM 94.2 (our retrain, Table 4) and 93.0 published; Maze-Hard 87.9 vs 87.0; Mini-ARC Pass@1 59.1 vs FPRM 55.0; ARC-1 47.4 vs 47.5 (a tie). The Sudoku gain of 2.4 is on a different training build (5k puzzles x 200 augmentations) than FPRM's published one, so it is not a matched pair; Table 4 holds the matched control.
- 2026-10-20 (Section 4.2, Figure 3, Figure 10): both models saturate accuracy within 20 to 35 iterations on Maze; every later loop is wasted. SFPRM halts on 992 of 1000 mazes at a mean of 108 passes; FPRM halts on 950 at 130. On Sudoku: 452 of 512 halt at a mean of 33 passes vs 422 at 89 (text of C.2). The text also says exact accuracy is still rising after the mean halt, so the threshold fires early.
- 2026-10-20 (Section 4.2, Figure 4): overriding lambda at inference on the trained model (learned lambda 0.47), raw map, 8 starts, 32 puzzles: for lambda >= 0.8 all eight end states coincide; at 0.47 the raw map reaches an attracting fixed point on 91 % of puzzles; at 0.1 it does not converge; for lambda >= 0.7 it converges in a few steps to a unique fixed point that no longer solves the puzzle.
- 2026-10-20 (Table 6, Mini-ARC ablation, one run each): full 59.1; no QK-norm 26.2 (-32.9); no damping (eta = 1) 47.7 (-11.4); fixed lambda 0.1 49.7 (-9.4); ACT halting instead of chordal residual 54.4 (-4.7); fixed eigen learning rates 55.0; no logit scale 55.7; no weight norm 56.4.
- 2026-10-20 (Section 4.3): the angle between a token's state and its own input embedding encodes task structure (Maze: walls 90 degrees, empty 113, path 102; empty and path cells share the same input token, so the loop wrote the difference). On Sudoku, the final angle separates solved from unsolved puzzles (clues about 52 vs 65 degrees).
- 2026-10-20 (Section 4.2, Table 5): swapping Anderson acceleration for the damped step at inference halts earlier at within one point of accuracy; Broyden halts as early but diverges later. Training with these solvers is left as future work.
- Failure modes and limits the authors state: one seed per cell; no tuning beyond the FPRM recipe; Theorem 1's step-size window is certified only for a fixed eta, the patience rule is a heuristic; Theorem 2 covers only strong injection; no language modeling.

## 3. Our speculation

- 2026-10-20 Speculation: the lambda sweep (Figure 4a) is the same trade as our injection floor. Strong injection gives one attractor but the attractor stops solving the task (lambda >= 0.7), and the trained lambda (0.47) sits where the raw map still converges on 91 % of puzzles. Our 0.87 floor says the same thing from the other side: a block that does not respond leaves the map at the injection's own gain. Their result suggests the useful setting is an injection that is learned, not fixed (fixed 0.1 costs 9.4 points on Mini-ARC).
- 2026-10-20 Speculation: damping by eta is a per-row step size on a loop, shrunk by a patience rule. Our Poisson depth draw is a depth distribution, not a step size, so the two are different dials. Their no-damping row (-11.4) says an undamped loop that has a ρ near 1 mode hurts. Our contractive diagonal injection already damps by construction, so this is not a gap in our loop.
- 2026-10-20 Speculation: the paper's early-halting gain (2.7x fewer passes at equal accuracy) is a puzzle-scale effect where the answer settles in 20 to 35 passes. Our loop contribution is K1-K6 +0.0377 on web text, where the loop refines each token and nothing "settles". Halting was a cost lever there, not a CE lever, and decode speed is not our priority.
- 2026-10-20 Speculation: the sphere keeps the state at unit norm, so the state cannot carry scale. Our measured HC carrier and carrier-norm work (per-pass cell RMSNorm in LXTUL) already holds scale fixed by a norm layer. This paper's claim that FPRM stores the clue mask in the norm and SFPRM in the direction is a parallel to our "LN common mode is plain-only" memory and says nothing about CE.
- 2026-10-20 Speculation: the anchor-angle readout (angle of the state to its own input embedding) is a cheap per-token instrument for "did the loop change this token". We could read it on the plain Parcae loop with no model change.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Paper claim: loop passes beyond the settling point (20 to 35 on Maze) are wasted and an angular residual halts them. Our measurement: on web text the plain loop earns +0.0377 K1-K6 by per-token refinement, and the span-only CALM backbone shows loop K1-K6 of only +0.002 to +0.003, with the backbone's contribution to CE (about 0.43 nats per token) unchanged across decoder quality. No settling test exists on our testbed, so the claim is neither supported nor refuted for text.
- 2026-10-20: Paper claim: SFPRM's input-anchor angle becomes a readable code of task structure and predicts correctness. Our measurement: slot states sit at effective rank 1.7 to 4.8 in 1024 dimensions (2026-08 audit), so a per-token angle to the input may be dominated by a shared direction. Untested.

## 6. Triage

Score 2. A mechanism and a sweep on injection strength and damping in a looped block, with numbers, but on puzzles at 6.8M parameters, one seed per row, and no language-model or span-level result. It names no test that changes what we run next.

Test ideas (control and predicted behavior):
1. Anchor-angle instrument on the plain Parcae loop. Control: the same 480 paired rows at depths 1 to 8. Read the per-token angle between the pass state and the prelude output. Prediction: tokens with the largest angle change between depth 1 and depth 6 are the tokens where K1-K6 is largest; if the angle is flat and K1-K6 is not, the instrument is blind on text.
2. Learned vs fixed injection strength on the slot-core DiagonalInjection (log_A already learned and has not left init). Control: the current learned-A run. Arm: A fixed at a stronger value. Prediction from Figure 4a: a stronger injection collapses the map toward one attractor and K1-K6 falls; a weaker one raises K1-K6 only if the blocks respond. Report with "floor + blocks' part" as in the READ-BEFORE-TUNING note.
