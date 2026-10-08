# Multiple Choice Learning: Learning to Produce Multiple Structured Outputs

- **Authors:** Abner Guzman-Rivera, Dhruv Batra, Pushmeet Kohli
- **Year:** 2012
- **Venue:** NIPS 2012 (Advances in Neural Information Processing Systems 25); page fetched at papers.nips.cc
- **URL:** https://papers.nips.cc/paper_files/paper/2012/hash/cfbce4c1d7c425baf21d6b6f2babe6be-Abstract.html (no arXiv)
- **PDF:** [mcl.pdf](mcl.pdf) (from papers.nips.cc, file/cfbce4c1d7c425baf21d6b6f2babe6be-Paper.pdf)

## Summary

Train M predictors so that a set of M outputs contains one good answer, instead of making
each predictor good alone. Only the best of the M is penalized per example.

- **Oracle / hindsight set-loss (Eq. 4).** `L(Y_i_hat) = min_{y_hat in Y_i_hat} l(y_i, y_hat)`.
  The set pays only for its most accurate member. "Predicting a set that contains even a
  single accurate output is better than predicting a set that has none." Replacing min with
  max or mean would punish diversity.
- **Weakness.** If one prediction equals the ground truth, the loss is 0 whatever the others
  are, so the loss is poorly conditioned.
- **Min-hinge upper bound (Eq. 6).** `H~_i(W) = min_{m in [M]} hinge_i(w_m)`: the min over the
  M per-predictor hinge losses (Eq. 2). Not convex, but it upper-bounds the set-loss.
- **Objective (Eq. 7).** `min_W 1/2 ||W||^2 + C * sum_i H~_i(W)`.
- **Optimization.** Coordinate descent: each example is assigned to the predictor that
  currently has the lowest loss on it, then each predictor is retrained on its assigned
  examples (a k-means-like alternation; see Section 3.2 and the flag variables in Eq. 9).
  Only the winner of each example gets its gradient.
