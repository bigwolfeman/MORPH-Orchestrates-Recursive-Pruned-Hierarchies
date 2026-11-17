# Paper note: Parallel-in-time training of RNNs for dynamical systems reconstruction (GTF-DEER)

- **Title:** Parallel-in-Time Training of Recurrent Neural Networks for Dynamical Systems Reconstruction
- **arXiv or URL:** arXiv 2605.12683
- **Local paper text:** none archived
- **Date first noted:** 2026-11-17

## 1. Why we reference it

- 2026-11-17: Asked whether a weight-tied loop over DEPTH can train with passes in parallel or below full BPTT cost.

## 2. What the paper shows

Read 2026-11-17 from the arXiv HTML. Only the first 100k of 264k characters were read.

- 2026-11-17 (Sec. 2, 3.1.3, 4.1): GTF-DEER = DEER (Newton fixed-point solve of the whole trajectory, a linear recurrence per Newton step, solved by a parallel associative scan in O(log T)) plus generalized teacher forcing.
- 2026-11-17 (Sec. 3.1.3, 5.2): Full Jacobian costs O(M^3 T) per Newton step. Diagonal (quasi-DEER) costs O(M T) but degrades gradients when the state is partly observed.
- 2026-11-17 (Sec. 5.1, 5.3): Up to 870x over sequential training on Lorenz-63. T = 32768 on Lorenz-96 took about 10 minutes vs an estimated 34 hours.
- 2026-11-17 (Prop. 1, Sec. 6): Convergence needs a teacher-forcing contraction condition (assumes M <= N). Average case O((log T)^2) for contracting dynamics, worst case O(T log T).
- 2026-11-17: Models are small shallow PLRNNs on low-dimensional chaotic systems. No transformer, no weight-tied depth.

## 3. Our speculation

- 2026-11-17 Speculation: A depth loop h_{k+1} = f(h_k) at one position has the same recurrence form with T = 6 to 8 steps. Parallel scan over 6 steps saves little (log 6 vs 6), and a Newton step needs the d x d Jacobian of a 768-wide transformer block, which costs more than the passes it replaces. Our loop is also scored only at the end, with no per-step teacher signal, so the forcing part has no counterpart. It does not transfer.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-11-17: None. It does not touch items 1-9. The contraction condition is related to our gain and contractivity work but gives no new lever.
