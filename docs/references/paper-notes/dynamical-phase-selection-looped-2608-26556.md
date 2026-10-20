# Paper note: Dynamical phase selection controls compute scaling in looped transformers

- **Title:** Dynamical phase selection controls compute scaling in looped transformers (Gunn Kim, Sejong University)
- **arXiv or URL:** 2608.26556 (v1, 2026-08-27, cond-mat.dis-nn). Type: paper (physics letter style, 4+ pages; companion supplement cited as [10], not read).
- **Local paper text:** docs/references/looping-depth/dynamical-phase-selection-2608-26556/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: It treats a trained looped map as a dynamical system and says initialization, not the task, selects the bifurcation type, which sets the relaxation time. That is the kind of claim our nested-dynamics model makes (the optimizer is blind to the inner map). The test bed is a toy. Triage score 1.

## 2. What the paper shows

Read: the full main text (about 3.8k words). Not read: the supplement ([10]) with the derivations and per-network tables.

- 2026-10-20 (Setup): one pre-norm attention+MLP block, d=16 to 20, state z with one evidence token, residual scale and latent decay, trained to hold a decision after evidence is withdrawn. 15 networks (5 memory pressures x 3 inits) all reach held-out accuracy 1.000. Linear part has rho(A)=0.90 < 1 by construction.
- 2026-10-20 (Fig 1): at one setting, ten inits give 5 fold, 1 Neimark-Sacker, 1 rigid, 3 monostable networks. Phase depends strongly on init, weakly on objective. 8 of 10 are in their final phase by step 250 while loss falls 4 to 24x more.
- 2026-10-20 (Table I, 7 fold networks): relaxation exponent 0.5004 +/- 0.0019 (predicted 1/2); tau*(1-lambda_max) -> pi measured 3.146 +/- 0.017; workload tail P(tau > N) ~ N^-2 (fitted -2.130 +/- 0.196, Hill 2.108 +/- 0.183). Neimark-Sacker network: fold law absent; bounded non-stationary motion; escape 29 % above the crossing.
- 2026-10-20 Failure modes / limits: one toy task, tiny d, a stationarity penalty, decay term added so fixed points are isolated; the classification is "of the models explored, not an exclusion theorem".

## 3. Our speculation

- 2026-10-20 Speculation: it is consistent with our finding that the same loop recipe can land on different depth use (measured seed spread of K1-K6 is 44 % of its mean), and that the injection decay `A` stays at init. It gives no handle for a 1024-wide LM with a diagonal injection; the fold analysis needs a task with a decision under withdrawn input.
- 2026-10-20 Speculation: the paper reinforces our "spectral caps do not target the disease" correction. The rho(A)<1 bound did not prevent a unit multiplier of the full Jacobian.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None.
