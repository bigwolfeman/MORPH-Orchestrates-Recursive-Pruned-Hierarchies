# Paper note: The Dynamics of Continuous Mixture Collapse in Language Models

- **Title:** The Dynamics of Continuous Mixture Collapse in Language Models (Ali Backour, MIT; preprint, under review)
- **arXiv or URL:** 2609.02049v1 (2026-09-02)
- **Local paper text:** docs/references/looping-depth/latent-exploration/continuous-mixture-collapse-2609-02049/ (PDF and reading file); extracted text in ignore/papers/2609.02049v1-continuous-mixture-collapse.txt (private)
- **Date first noted:** 2026-10-18 (summarised from an uncached PDF on 2026-09-23; read in full 2026-10-18)

## 1. Why we reference it

- 2026-10-18: It gives a dynamical account of when a recurrent read keeps or loses a mix of
  two candidates: a coupling `L_t` against a threshold of 2, and a shared field `b_t`. Below
  the threshold every mix washes out to one state; above it the initial majority takes over.
- 2026-10-18: For the coarse-level particle design (K copies of the coarse loop
  `x^{t+1} = F(D x^t + Delta B z)` from random starts, read as a set by a teacher-forced AR
  coda; Lean claims in `lab/theory/tul_exploration/TulExploration/Particles.lean`): Lean C1
  (spread at most `rho^T` under contraction) has the shape of its Theorem 2, and the
  several-basins claim (P2 in `Particles/Collapse.lean`) has the shape of its Theorem 1.
  Its Theorem 3 puts a price on holding a K-way mix in ONE state.

## 2. What the paper shows

- 2026-10-18 (Section 2, Figure 1): one slot holding 50 / 25 / 25 of three colour words is
  answered 99.5 % "red" (six triples, ten wordings). Over 325 three-way mixes, Qwen3.5-4B
  and Gemma-4-E4B-it send most of the simplex interior to one near-pure answer.
- 2026-10-18 (Section 2, Eq. 1 and 2, Table 1): final-layer CALIB, trained against the mean
  of five random inits of the same architecture: Qwen3.5-0.8B 0.632 / 0.818, Qwen3.5-4B
  0.332 / 0.732, Qwen3.5-27B 0.130 / 0.611, gemma-4-E4B-it 0.475 / 0.765, gemma-4-31B-it
  0.160 / 0.569. Training makes it worse, more so at scale.
- 2026-10-18 (Section 3, Eq. 3 to 7): with exactly linear transport of the final hidden
  state, the softmax makes a geometric mix `q_w ∝ q_A^w q_B^(1-w)`, and with
  `u_t = 2 w_t - 1`, `L_t = (Delta_A - Delta_B)/2`, `b_t = (Delta_A + Delta_B)/2` the
  feedback is `u_{t+1} = tanh((b_t + L_t u_t)/2)`, the Curie-Weiss iteration.
- 2026-10-18 (Section 3.2, Eq. 9): with `b = 0` and constant `L`, the balanced state is
  stable for `L < 2` and unstable for `L > 2`, where two stable states `+-u*(L)` appear.
  `L = 4` gives `u* ≈ 0.96`, `L = 8` gives `0.9993`.
- 2026-10-18 (Theorem 1, Eq. 10, proof in Appendix B): if `u_0 ≠ 0`, `b_t = 0` and
  `L_t >= L_min > 2` for all `t`, the sign of `u_0` is kept forever and
  `liminf |u_t| >= u*(L_min)`.
- 2026-10-18 (Theorem 2, Remark 1, Appendix B): if `0 <= L_t <= L_max < 2`, then
  `|u_T| <= (L_max/2)^T |u_0| + (1/2) sum_t (L_max/2)^(T-1-t) |b_t|`; with `b_t = 0`,
  `u_t -> 0` from any start. With `|b_t| <= b_max`, `limsup |u_T| <= b_max / (2 - L_max)`.
- 2026-10-18 (Section 4, Eq. 11, Figure 3): two pure branches run under the model with greedy
  successors and the mix exists only in logit space. Starts just above and below 1/2 split to
  opposite branches. Measured `L_t` sits mostly above 2 in both families, and `|b_t|` is much
  smaller (medians, 1000 items, 50 steps).
- 2026-10-18 (Section 4.1, Figure 4): a feedback temperature sets `L_eff = L_t / tau`.
  Sweeping it moves the same items from polarisation to washout, with the turn near
  `L_eff = 2`.
- 2026-10-18 (Theorem 3, Eq. 16 to 19): a C^1 corrector `f(a, r(x))`, `r(x) in R^m`, that
  keeps even one interior mix fixed near a context `x0` needs `m >= rank D_x Phi`, and
  `m >= K - 1` if the context moves the output in all `K - 1` mixture directions. The
  full-rank condition is not measured (Section 7).
- 2026-10-18 (Section 6): feeding the hidden state back (Coconut) removes the
  recursive-softmax source "but is not by itself sufficient"; damping the gain trades
  polarisation for washout.

## 3. Our speculation

- 2026-10-18 Speculation: Theorems 1 and 2 do not apply to our coarse loop as stated. Eq. 7
  needs a vocabulary softmax and autoregressive feedback inside the loop, and the coarse loop
  has neither. The Theorem 2 proof shape does transfer: a contraction plus a bounded additive
  forcing. K particles share the same map and the same injection `Delta B z`, so the
  injection cancels in any difference between two particles. The injection plays the role of
  the field `b_t`: it sets where all particles go, and it cannot hold them apart. That is
  Lean C1.
- 2026-10-18 Speculation: the takeover of Theorem 1 needs the mix weights to pass back
  through a softmax. Our particles do not read each other, and the coda reads them
  teacher-forced, so the coda's attention weights never return to the loop. That channel is
  absent from the forward. What remains is per particle: under contraction all particles fall
  to one state; with an expansive direction and saturation each particle falls into the basin
  its random start points at, chosen by the start (the "sign of `u_0`") and not by the
  context. A context-dependent field can break that symmetry; the paper proves nothing for a
  nonzero field above threshold. Training gradients through the coda's read are a separate
  path that could pull particles together (Lean C5); the paper says nothing about it.
- 2026-10-18 Speculation: on multiple stable mixes the paper is narrow. It proves two stable
  states for two components at zero field above threshold, both polarised, each holding one
  bit about the start. Nothing for K > 2 under recursion. So it gives no support for "K
  particles land in K context-dependent basins". That has to come from the coarse map
  itself.
- 2026-10-18 Speculation: a testable prediction for the spiral testbed. On the Anchor spiral
  each level's output changed only 3 to 6 % when its incoming update was removed, though the
  update's RMS was 29 to 70 % of h0's (CPU probe on `spiral-cycle-end-5k-s1`, memory note
  spiral-levels-ignore-incoming-update, 2026-10-16). Read as a map from the incoming state, that
  coarse map is strongly contractive, so particles from random starts would collapse within
  about one pass. The strict spiral, whose coarse levels carry context (zeroing costs
  +0.126), is the better host. Measuring the coarse map's gain along the direction between
  two particles, before building the arm, would test this.
- 2026-10-18 Speculation: Theorem 3 is the strongest formal support in these four papers for
  the "separate particles when the computation is nonlinear" half of Wolfe's criterion. One
  state holding a K-way mix through a context-dependent map must carry at least `K - 1`
  extra context-dependent degrees of freedom. K particles pay that as K states, each of
  which only has to hold one candidate.

## 4. Corrections

- 2026-10-18: Our 2026-09-23 summary (in the Reasoning by Superposition reading file) said
  that under Theorem 2 "a mixture not re-supplied by the field `b_t` goes to 0 at a geometric
  rate". `u -> 0` is the 50 / 50 state: every starting mix ends at the same point. It is
  washout, not loss to one component. The bound formula it quoted is correct, and Theorem 2
  also needs `L_t >= 0`.
- 2026-10-18: The same summary said "above the threshold (Theorem 1) one component takes
  over". Theorem 1 needs `b_t = 0` exactly and `L_t >= L_min > 2` at every step, and the
  component that takes over is the initial majority.
- 2026-10-18: The same summary called the setting "soft-token feedback in pretrained LLMs".
  Section 2 injects one soft token with no recursion. Sections 3 and 4 form the mix only in
  logit space between two pure branches, so transport is linear by construction.

## 5. Where our findings disagree

- 2026-10-18: Agreement in shape, in a hidden-state loop with injection, which the paper does
  not cover. Our measurement: on `slot-spandec-strict-prev-reach1` @ 5000, with the loop's
  cross-cell read cut after pass 0, content the loop had carried in decayed under the cell's
  own later passes (planted copy two spans back: 0.148 / 0.073 / 0.050 / 0.035 nats of
  benefit at depth 1 / 2 / 3 / 6), while the cell's own span, re-supplied by the injection
  every pass, improved (0.181 to 0.291). At depth 16 every planted distance of two or more
  sat near 0.05. Paper's claim: under subcritical coupling the part of the state not
  re-supplied by the field decays geometrically, and a persistent field holds a residual set
  by the field (Theorem 2, Remark 1). Source:
  `lab/experiments/failures/2026-09-19-hop-distance-plateau-and-dilution.md`, Verdict item 2
  (measured 2026-09-19).
