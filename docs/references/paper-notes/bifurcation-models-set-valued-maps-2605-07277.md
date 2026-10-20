# Paper note: Bifurcation Models (set-valued solution maps with weight-tied dynamics)

- **Title:** Bifurcation Models: Learning Set-Valued Solution Maps with Weight-Tied Dynamics (Caleb Jore, Jialin Liu)
- **arXiv or URL:** 2605.07277 (v1, 2026-05-08, cs.LG). Type: paper.
- **Local paper text:** docs/references/looping-depth/bifurcation-models-2605-07277/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-20

## 1. Why we reference it

- 2026-10-20: Our particles filing found that K hypotheses in a loop collapse unless each has a persistent learned identity, and that noise-seeded particles are a dead end. This paper gives a clean small-scale case of the same collapse in a weight-tied recurrent map, and it measures the price of the fix. No language model. Triage score 2.

## 2. What the paper shows

Read: abstract, Sections 1 to 4 text, Tables 1, 2, 4, the collapse discussion. Not read: the proofs in the theory section and appendices (so "almost everywhere regular" is the authors' claim, unchecked).

- 2026-10-20 (Sec 2, theory): a set-valued map with locally Lipschitz branches can be represented by a weight-tied map whose different starts settle in different equilibria. Hand-picked single-branch selectors can be arbitrarily irregular; the learned selector is a.e. regular.
- 2026-10-20 (Sec 3, Table 2, frustrated Ising, recurrent GNN, 200 test graphs x 20 trajectories): energy objective, recurrent GNN: test energy -0.983, 13.6 +/- 5.8 distinct solutions of 20. Vanilla GNN with energy loss: -0.750, 1 solution. Gurobi-label fitting: test energy -0.092 (recurrent) and -0.059 (vanilla), 1 solution. No branch labels were used for the multi-solution arm.
- 2026-10-20 (Sec 4, Table 4, Allen-Cahn): energy-only recurrent model collapses to ONE solution cluster even though the map can hold many. An explicit diversity term over M sampled starts lifts clusters 1.00 -> 16.11 at weight 1.2, and raises test residual error 9.19e-4 -> 5.74e-3. Fitting one IMEX label is worse than energy-only on residual and energy.
- 2026-10-20 Failure modes: diversity does not emerge unprompted on Allen-Cahn; diversity weight trades against accuracy.

## 3. Our speculation

- 2026-10-20 Speculation: energy-only collapse on Allen-Cahn is the same event as our particle collapse. The objective rewards any one good branch, so all starts fall into the easiest one. The Ising success needs a frustrated landscape that really has many equal-energy optima; web text next-span prediction may have one dominant branch per context, so there is nothing to bifurcate to.
- 2026-10-20 Speculation: their starts differ only by the initial state, which is the "noise-seeded" setting we list as dead. That fits: they needed an explicit diversity term (accuracy cost) where we needed a persistent learned identity. Our fan diversity terms were gamed; their term is a pairwise start-to-output spread and the paper does not report gaming.
- 2026-10-20 Speculation: for span-CALM, the target is a latent of a VAE. A set of valid next-span latents is a set-valued target. A bifurcation-style loop would need a loss that scores the best of K (a min over particles), not the mean.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None. Agreement with the particles result: collapse without a force against it.

## Test ideas (score 2)

- 2026-10-20 Test: on the particle panel, control = mean-over-particles loss. Arm = best-of-K (min over particles) loss on the span target with the same K and identities. Prediction: particles differ more (pairwise latent spread) than the control; if spread rises with no CE@6 or K1-K6 change, the min loss only moves the diversity metric (we must not pass or fail on a cosine, so read CE and K1-K6).
- 2026-10-20 Test (cheap, offline): Allen-Cahn-style check of whether our web-text next-span target has more than one mode per context. Take 480 fixed rows, decode N samples from a trained model per context, count clusters in the decoder's latent space. One cluster means a bifurcation loop has nothing to find.
