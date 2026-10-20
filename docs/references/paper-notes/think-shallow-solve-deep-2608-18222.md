# Paper note: Think Shallow, Solve Deep (finite-time dynamical regime predicts test-time depth)

- **Title:** Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth (Viakhirev, Borodin, Almutairi, Barannikov, Abramov, Mkrtchian)
- **arXiv or URL:** 2608.18222 (v1, 2026-08-18, cs.LG). Type: paper.
- **Local paper text:** docs/references/looping-depth/think-shallow-solve-deep-2608-18222/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It measures the inner map of a looped net (spectrum, Lyapunov exponents, settle ratio) on every trained operator and flips the regime with ONE terminal fixed-point term. That is the same term we hold (`model.core_fixed_point_lambda`), and the same identity collapse we call the identity-escape law. It is the closest outside evidence for our nested-dynamical-system view.
- 2026-10-20: Scope warning. All tasks are small algorithmic ladders (Sudoku, reachability, carry addition), d=256, 800 examples per tier, 16 steps. Nothing is a language model except a Huginn-3.5B measurement.

## 2. What the paper shows

Read in full (main text Sec. 1 to 7; appendices not read line by line).

- 2026-10-20 (Sec. 4, 5.3): removing input recall collapses the operator to identity, sigma_max = 1.0000, EM 0.00. Removing structure keeps settling and still scores 0.00 ("settling is necessary, not sufficient").
- 2026-10-20 (Sec. 4, Fig. 3, Table 1): removing only the terminal fixed-point term (||z_T - z_{T-1}||, scaled by the update gate) flips the regime settle to drift (lambda_max -0.06 to +0.10) with sigma_max (~5.6) unchanged. Unseen-hard Sudoku goes 0.34 to 0.03; reachability loses up to 0.37 EM by 128 steps. Adding only this term to a generic recurrence on carry addition: best 0.12 to 0.88, overthink damage 0.91 to 0 (18 seeds).
- 2026-10-20 (Sec. 5.4): among 164 competent runs, none of the 17 settling runs lose more than 0.004 EM to extra depth; 30 % of marginal and 22 % of drifting runs lose more than 0.1. Cluster-bootstrap keeps the safety result; pooled conversion is NOT cluster-significant.
- 2026-10-20 (Sec. 6.1, Prop. 2): sigma_max is a correlate, not a lever. Successful operators are strongly non-normal (sigma_max/rho about 4 to 5.4), so the top singular value is a transient amplification. Forcing typical gain to expand (tau >= 2) blows sigma_max to 14 to 33 and kills training. rho-control (asymptotics only) is the control that helps (cites STARS).
- 2026-10-20 (Sec. 6.2): spread of the top-8 Lyapunov spectrum (lambda_1 - lambda_8) predicts extrapolation with AUC 0.846 over 304 runs, on par with training accuracy (0.851); within settling runs it separates "settled into an algorithm" from "merely settled" (AUC 0.872). Failure runs have eight nearly identical exponents.
- 2026-10-20 (Sec. 7): Huginn-3.5B: sigma_eff = 1.000 +/- 0.003 over 2,585 tokens, none settle; answer freezes without tracking difficulty (guess-freezing); carry probe 0.69 at r=8 then 0.00 for r >= 32. A label-free LoRA that pulls deep states (r ~ U[16,128]) to the frozen base's own r=8 state restores depth-safety (digit copy 0.00 to 1.00, 10/10 seeds) and gives NO conversion. The fixed-point term itself "does not transfer to this scale".
- Failure modes reported: identity, marginal, drift, settle-but-wrong. Training 32 steps instead of 16 gave 0.43 to 0.48 at 128.

## 3. Our speculation

- 2026-10-20 Speculation: this agrees with our held stability pair (1000-step ramp plus a weak fixed-point term). Their ablation says the term protects depth-safety; our measure says it is free at 5k. They do not say it buys CE, and neither do we.
- 2026-10-20 Speculation: "no-recall gives sigma_max = 1.0000 identity" matches the identity-escape law and the 0.87 injection floor in our map. Their recall is our DiagonalInjection.
- 2026-10-20 Speculation: their non-normality result (sigma_max a transient, rho the asymptote) is a clean reason our spectral caps failed and why "target rho" was the right place but sigma the wrong quantity. It matches our 2026-08-24 correction.
- 2026-10-20 Speculation: the Lyapunov spread is a cheap instrument we lack. K1-K6 says whether depth pays; spread would say whether a pass has differentiated directions.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-20: Their depth use needs a settling map and rises past the training horizon on fixed-size algorithmic ladders. Our K1-K6 on web text is small (+0.0377 plain, +0.0054 fan) and our eval depth is capped at the train max of 8, so we cannot test their extrapolation axis. Not a contradiction, a different regime.
