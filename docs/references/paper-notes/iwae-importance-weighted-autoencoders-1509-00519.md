# Paper note: Importance Weighted Autoencoders (IWAE)

- **Title:** Importance Weighted Autoencoders (Yuri Burda, Roger Grosse, Ruslan Salakhutdinov; ICLR 2016)
- **arXiv or URL:** 1509.00519 (v4, 2016-11-07). The id and authors were read from the paper's own first page.
- **Local paper text:** docs/references/smc-latent-objectives/iwae-1509-00519/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-19

## 1. Why we reference it

- 2026-10-19: Our mixture particles loss `-log mean_k p(row | eps_k)` is the k-sample bound `L_k` (Eq. 8) with the prior as the proposal (so the weight is `p(x | h)` alone). This note records the original bound, its monotonicity in k, and what its gradient does. FIVO, AESMC and VSMC all treat it as the no-resampling baseline.

## 2. What the paper shows

- 2026-10-19 (Section 3, Eq. 8 and 9): `L_k(x) = E_{h_1..h_k ~ q(h|x)} log (1/k) sum_i p(x, h_i)/q(h_i | x)`. It lower-bounds `log p(x)` by Jensen. k = 1 is the VAE bound.
- 2026-10-19 (Theorem 1, Eq. 10, proof in Appendix A): `log p(x) >= L_{k+1} >= L_k`. If `p(h, x)/q(h | x)` is bounded, `L_k -> log p(x)` as k grows. The paper gives no rate in k and says nothing about sequence length T. It treats single observations.
- 2026-10-19 (Section 3 text, Appendix B, pointer only): the log of the mean weight "does not suffer from high variance" the way a raw importance-sampling estimate of `p(x)` does. I did not read Appendix B's argument in full.
- 2026-10-19 (Section 3.1, Eq. 11 to 14): the gradient is `sum_i w_tilde_i grad log w_i`, where `w_tilde_i` are the normalized weights. The weights act as a soft selection over samples. There is no resampling and no discrete choice, so there is no score-function term. Footnote 1: the VAE trick of splitting out the KL term gives a lower-variance update that has no analogue for k > 1, so IWAE updates "may be higher variance". In their runs the two rules were indistinguishable at k = 1.
- 2026-10-19 (Section 3.1, end): a cheaper estimator picks ONE sample i with probability `w_tilde_i` and back-propagates only through it. It needs k forward passes and one backward pass, about 3x fewer multiply-adds at large k, "at the cost of increased variance in the updates, but empirically we have found the tradeoff to be favorable". The chosen sample is not resampled into new particles. It is a stochastic estimate of the same gradient.
- 2026-10-19 (Section 5.1, Table 1): MNIST and Omniglot (binarized 28 x 28), one stochastic layer (50 units) or two (100 and 50), Adam, 3280 passes, evaluation by `L_5000`, k in {1, 5, 50}. MNIST NLL (one layer): VAE 86.76 / 86.47 / 86.35, IWAE 86.76 / 85.54 / 84.78. MNIST two layers: VAE 85.33 / 85.01 / 84.78, IWAE 85.33 / 83.89 / 82.90. Omniglot one layer: VAE 108.11 / 107.62 / 107.80, IWAE 108.11 / 106.12 / 104.67. Omniglot two layers: VAE 107.58 / 106.31 / 106.30, IWAE 107.56 / 104.79 / 103.38. Best MNIST: IWAE two layers, k = 50, -82.90.
- 2026-10-19 (Section 5.2, Table 1 and 2): latent "activity" `A_u = Cov_x(E_{u ~ q(u|x)}[u])`, active if `A_u > 0.01`. Active units, MNIST one layer: VAE 19, 20, 20 and IWAE 19, 22, 25 for k = 1, 5, 50. Only about 20 to 25 of 50 are active. Two-layer second layers use fewer than 10. Continuing a VAE with the IWAE objective raised active units (19 to 22) and lowered NLL (86.76 to 84.88). Continuing IWAE with the VAE objective did the reverse (25 to 23 units, 84.78 to 86.02). This suggests inactive units are driven by the objective, not just by optimization.
- 2026-10-19 (proposal): `q(h_l | h_{l-1})` is a diagonal Gaussian whose mean and variance are network outputs of the previous layer, reparameterized by `h_l = Sigma^{1/2} eps + mu` (Eq. 4). Conditioned on the observation x (and the previous layer) and learned. Per sample, independent across the k samples. No ancestor, no resampling, no sequence structure.
- 2026-10-19 (variance collapse): the paper does not discuss learned variances shrinking to zero. The log-weight gradient (Eq. 15) has a `-grad log q` term, and the text says it "encourages the recognition network to have a spread-out distribution over predictions". That is the entropy-like term of the p/q ratio.

## 3. Our speculation

- 2026-10-19 Speculation: the IWAE finding that more samples raise the number of used latent units (Section 5.2) predicts that our K > 1 loss should keep more of the `eps` channel informative than a K = 1 loss. We have not measured that. Our end-of-row posterior max at 0.985 to 0.998 at K = 2 says the weights pick one particle almost every row, which is the regime where IWAE reduces to K = 1 for that row.
- 2026-10-19 Speculation: their "pick one sample by weight, backprop through it" estimator (Section 3.1) is a cheap way to run K = 8 or more if memory bounds our coda passes. It is not the same as resampling.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None recorded yet.
