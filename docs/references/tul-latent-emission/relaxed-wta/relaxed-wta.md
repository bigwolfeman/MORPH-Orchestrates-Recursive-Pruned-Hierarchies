# Learning in an Uncertain World: Representing Ambiguity Through Multiple Hypotheses

- **Authors:** Christian Rupprecht, Iro Laina, Robert DiPietro, Maximilian Baust, Federico Tombari, Nassir Navab, Gregory D. Hager
- **Year:** 2017 (arXiv v1 1 Dec 2016; stored PDF is v3, 22 Aug 2017)
- **Venue:** ICCV 2017 (arXiv comment "ICCV 2017"; PDF header names the conference)
- **arXiv:** https://arxiv.org/abs/1612.00197
- **PDF:** [relaxed-wta.pdf](relaxed-wta.pdf)

## Summary

A network predicts M hypotheses f_theta^j(x). Each label is assigned to its closest
hypothesis, which splits the output space into M Voronoi cells. This is the multiple
hypothesis prediction (MHP) framework, a gradient-descent version of Lloyd's method.

- **Meta loss (Eq. 11).** `M(f(x_i), y_i) = sum_{j=1}^{M} delta(y_i in Y_j(x_i)) * L(f_j(x_i), y_i)`,
  with Kronecker delta. This is hard winner-take-all: only the winning head gets a gradient.
  It works on top of any base loss L (l2, Tukey bi-weight, etc.).
- **Problem.** A head with a bad initialization can win no samples and never update.
- **Relaxation (Eq. 12).** Replace delta with
  `delta_hat(a) = 1 - eps` if a is true, `eps / (M - 1)` otherwise, with 0 < eps < 1.
  The winner gets weight 1 - eps, every other head gets eps/(M-1), and the weights sum to 1.
- **Values used.** eps = 0.05 in all experiments. They also drop out whole predictions with
  probability 1 %, which adds some randomness to winner selection.
- **Architecture.** The simplest conversion of an existing net is to replicate the output
  layer M times with different initializations; the cost is negligible.
- **Finding.** Accuracy under an oracle pick of the best hypothesis rises with M (pose PCP
  59.7 % for 1 head, 62.8 % for 10), and the spread of hypotheses tracks uncertainty.
