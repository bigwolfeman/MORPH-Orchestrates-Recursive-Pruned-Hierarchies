# Paper note: Looped Reasoning Finishes Earlier Than You Think (QRR)

- **Title:** Looped Reasoning Finishes Earlier Than You Think (Park and Rusch, ELLIS Tuebingen / MPI-IS / Liquid AI)
- **arXiv or URL:** 2610.18174 (v1, 2026-10-13)
- **Local paper text:** docs/references/looping-depth/qrr-quotient-recurrent-reasoner/2610.18174.txt (pdftotext of v1, read in full 2026-10-16)
- **Date first noted:** 2026-10-16

## 1. Why we reference it

- 2026-10-16: It is a dynamics paper about looped reasoners (FPRM, HRM, TRM, EqR) on puzzles. Three overlaps with our work:
  - Its training recipe is the HRM recipe that the cycling-spiral deep-credit arms and `plain_deep` test on Parcae
    (A.3): the state is carried across deep-supervision segments, each segment runs no-grad steps and then a
    truncated-BPTT window, a loss follows each segment, and the state is detached before the next one.
  - Its halting rule (a quotient residual under a threshold for a persistence window, eq. 19) works, and the halting
    step tracks puzzle difficulty. This bears on Wolfe's PonderNet-like gate over the cycle schedule (deferred).
  - Its A.4.4 result (suppressing an oscillating mode early costs accuracy) agrees with our fixed-point-term finding.
- Not an overlap: it is not a language model. No CE, no next-token task, 7M parameters, and loops run for hundreds to
  thousands of steps to a fixed point. Our loops run 6 passes and never reach one.

## 2. What the paper shows

- 2026-10-16 (sec. 3, Lemma 3.1, Thm 3.2): near a solution root, a slow oscillating mode (modulus r) that couples into
  the rest of the state (B != 0) gives the rest its own rate r. Settling time is log(1/delta) / (-log r). If r = 1 the
  state never settles. This is exact linear algebra (Sylvester decomposition).
- 2026-10-16 (sec. 3.2, Fig. 2, Fig. 6): an intervention test on the trained nonlinear FPRM (a flip mode, angle median
  179.7 deg) and HRM (a slow spiral, median 3.9 deg). The measured response and the settling gain match the local
  Jacobian prediction. The median margin r - rho(A) is 0.0176 (FPRM) and 0.035 (HRM) (Fig. 8).
- 2026-10-16 (A.4.4, Table 4): halving the FPRM flip mode BEFORE the output stabilizes costs 5.9 pp exact accuracy
  (CI [-13.1, -0.6]). An energy-matched non-oscillating control costs -1.1 pp (not significant). The same suppression
  AFTER stabilization costs 0.0 pp. So the oscillation carries computation early and is only a delay late.
- 2026-10-16 (sec. 4, A.3): QRR. The state is complex (channel pairs), with one U(1) phase per token. The trunk is
  equivariant to it: complex-linear Q, K and V, attention scores Re(U q^H k) with a relative phase transport, a gate on
  |W_g h|, and the input injected through a phase carrier. The readout is invariant. A second-order (momentum) update
  with one learned global beta. A detached gauge re-phasing each step.
- 2026-10-16 (Table 1): Sudoku-Extreme 97.9 exact (FPRM reproduced 83.7, EqR 89.9, TRM 75.7, HRM 55.4); Maze-Unique
  97.0; Mini-ARC 62.4; Maze-Hard 85.8 (FPRM 86.4 official). Per-step cost 1.5x FPRM, memory 1.3x. End-to-end rollout
  p50 1.9 s vs 5.4 s.
- 2026-10-16 (Table 2): halting step p50 27 vs FPRM 122 on Sudoku; false halts 0.1 % vs 8.9 %. Root reached at a
  median 122 vs 234 layer-forwards.
- 2026-10-16 (Table 3, ablations on Sudoku): a single global U(1) gets 91.3 % but never halts (coverage 0). No momentum:
  87.8 %. The FPRM trunk with momentum: 77.0 %, below the FPRM trunk without it (83.7 %, Table 1). So the momentum
  helps only on the equivariant trunk.
- 2026-10-16 (A.4.5, Fig. 9): the halting step rises with difficulty (Spearman 0.356 against blank count; FPRM 0.201).

## 3. Our speculation

- 2026-10-16 Speculation: the rate theorem is asymptotic near a root. Our 6-pass LM loops are far from any root, so the
  "slow mode delays settlement" mechanism probably does not bind there. Our problem is the reverse: the loop has
  little to do after a pass or two. The paper does not test the first passes of a loop.
- 2026-10-16 Speculation: gauge motion is invisible to the readout and cannot force the quotient state (Prop. A.2). So
  it carries no information. It is not "latent exploration" in Wolfe's LXTUL sense. It removes a nuisance from the
  convergence test and from the dynamics.
- 2026-10-16 Speculation: the halting rule may be the useful part for us. A residual "mod the symmetries the readout
  ignores" is a better stop signal than raw state change. The analogue for a span loop is a residual measured through
  the coda's read (what the coda would decode), not on the raw state.
- 2026-10-16 Speculation: a momentum carry on Parcae's loop (second-order dynamics, one learned beta) is cheap. The
  Table 3 row "FPRM trunk + momentum, 77.0 %" says it can hurt on a plain trunk. That row is evidence against a
  momentum-only arm.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-16: Agreement, not disagreement. Our measurement: the fixed-point term (it charges ||h_T - h_{T-1}||^2, the
  motion between passes) held the strict slot loop still. With it off, coda K1-K6 was +0.0123 (seed 1) and +0.0077
  (seed 2), at 5k. Paper's claim: suppressing motion before the output stabilizes costs accuracy (A.4.4). Source:
  lab/experiments/successes/2026-09-24-lxtul-stage3-map.md, successes/2026-09-25-lxtul-nofp-followup.md.
- 2026-10-16: Different regime. Our measurement: MORPH's E7 loop is a power iteration. Iteration 0 sets the scale, and
  the state converges in about 3 steps with rank 181 to 9. Paper: its puzzle loops settle over hundreds of steps,
  limited by one slow mode. Source: lab/experiments/failures/2026-09-07-arc-e7-block-loop.md.
