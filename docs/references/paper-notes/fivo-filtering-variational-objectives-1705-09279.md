# Paper note: Filtering Variational Objectives (FIVO)

- **Title:** Filtering Variational Objectives (Chris J. Maddison, Dieterich Lawson, George Tucker, Nicolas Heess, Mohammad Norouzi, Andriy Mnih, Arnaud Doucet, Yee Whye Teh; NeurIPS 2017)
- **arXiv or URL:** 1705.09279 (v3, 2017-11-12). The id and author list were read from the paper's own first page.
- **Local paper text:** docs/references/smc-latent-objectives/fivo-1705-09279/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-19

## 1. Why we reference it

- 2026-10-19: Our mixture particles (Parcae testbed, `lxtul/spiral.py` "Mixture particles", `lxtul/plain_pointer.py::_mixture_nll`) are K copies of the last loop level, each fed `z + sigma * eps_k * rms(z)`, with `eps_k` held for every pass. The loss is `-log mean_k p(row | eps_k)`. That is the IWAE estimator with the prior as proposal and no resampling. FIVO is the same estimator plus a resampling step every chunk. It is the paper that tells us what the resampled version optimizes, what its gradient is, and which part of that gradient to drop.
- 2026-10-19: A 20-step smoke read an end-of-row posterior max of 0.985 to 0.998 at K = 2. That is weight degeneracy. FIVO's argument (Section 4) is that resampling is the cure for this on long sequences.

## 2. What the paper shows

Measured and proved results only. Numbers come from `paper.txt`.

- 2026-10-19 (Definition 1, Eq. 3): a Monte Carlo objective (MCO) is `L_N = E[log p_hat_N(x)]` for any unbiased positive estimator of `p(x)`. ELBO is one weight. IWAE (Eq. 4) is the mean of N i.i.d. weights `p(x,z^i)/q(z^i|x)`.
- 2026-10-19 (Proposition 1): (a) `L_N <= log p(x)`. (b) If the estimator is strongly consistent and `log p_hat_N` is uniformly integrable, `L_N -> log p(x)` as N grows. (c) Under a bounded first inverse moment, `log p(x) - L_N = (1/2) var(p_hat_N / p) + O(sqrt(g(N)))`, with `g(N)` the 6th central moment (Eq. 5). The bound's gap is half the relative variance of the estimator. This is the paper's tightness theorem.
- 2026-10-19 (Section 3, Section 4, text only): the relative variance of a naive importance sampling estimator "tends to scale exponentially in the number of steps", while a particle filter's "can scale more favorably ... linearly in some cases", citing Cerou et al. 2011 and Berard et al. 2014 [14, 15]. The paper does not prove this itself. It also gives no IWAE degeneracy experiment with a numeric effective sample size. Its evidence is the final-bound comparison in Table 1.
- 2026-10-19 (Algorithm 1, Eq. 6 and 7, Definition 2): FIVO is `E[log p_hat_N(x_1:T)]` where `p_hat_N` is the product over t of the weighted mean of incremental weights `alpha_t = p_t(x_t, z_t | x_1:t-1, z_1:t-1) / q_t(z_t | x_1:t, z_1:t-1)`. Each particle proposes from `q_t` conditioned on its own ancestor path z_1:t-1 and on observations up to t. Resampling is the only difference from IWAE: "if resampling is removed, then FIVO reduces to IWAE" (Section 4). Resampling schemes named: every step, or when ESS `(sum_i w_i^2)^-1 < N/2`. Experiments use ESS < N/2.
- 2026-10-19 (Section 4.1, Eq. 8): the full gradient has three kinds of terms: the pathwise term `grad log p_hat_N`, terms from each distribution that depends on the parameters, and (with adaptive resampling) terms for the resample decision. With reparameterized `z_t^i` and a fixed resampling schedule, `grad L = E[ grad log p_hat_N + sum_t sum_i I(resample at t) * log(p_hat_N(x_1:T)/p_hat_N(x_1:t)) * grad log w_t^i ]`. The second sum is the resampling term. The paper follows only the first term, a biased estimator. Reason given: the resampling terms "contribute to the majority of the variance of the estimator".
- 2026-10-19 (Section 6.3, Figure 2 left): the resampling term "often" raises gradient variance "by 6 orders of magnitude". With it (plus a moving-average baseline linear in T), "models trained without the resampling gradient term outperformed ... by a large margin" on train and held-out data, and "many runs with resampling gradients failed to improve beyond random initialization". The authors call this "an empirical result", and say biased gradients can in principle diverge.
- 2026-10-19 (Proposition 2, Section 4.2): FIVO is sharp (reaches `log p(x_1:T)` at the optimal q) only if `p(z_1:t-1 | x_1:t) = p(z_1:t-1 | x_1:t-1)`. In that case the optimal `q_t` does not condition on future observations x_t+1:T. In general FIVO is not sharp, unlike ELBO and IWAE whose optimal q uses all of x.
- 2026-10-19 (Section 6.4, Table 2): a smoothing proposal that sees a backward RNN over future observations helped IWAE (JSB train bound -5.77 to -4.63) but "did not reliably improve" FIVO (JSB -4.08 to -3.83, Nottingham -2.29 to -2.34, MuseData -5.80 to -5.87). Smoothed IWAE still lost to FIVO. Text: "consistent with the view of resampling as a greedy process that responds to each intermediate distribution as if it were the final".
- 2026-10-19 (Section 6, Table 1): model is a VRNN with an LSTM, factorized Gaussian latents and Gaussian proposals `q_t(z_t | h_t(...), x_t)` in the residual parameterization. Bounds are L_128 on the test set. N in {4, 8, 16}. Polyphonic music (nats per timestep, 64 units; JSB 32): at N = 4, ELBO / IWAE / FIVO are Nottingham -3.00 / -2.75 / -2.68, JSB -8.60 / -7.86 / -6.90, MuseData -7.15 / -7.20 / -6.20. At N = 16, JSB -8.63 / -7.41 / -6.72 and MuseData -7.18 / -7.13 / -5.89. The text states that FIVO with N = 4 beat IWAE and ELBO at N = 4, 8, 16 on all four sets. TIMIT (nats per sequence relative to ELBO N = 4): 64 units N = 4 IWAE -160, FIVO 5,691; 256 units N = 4 IWAE 11,054, FIVO 17,822; 256 units N = 16 IWAE 13,069, FIVO 21,536. Sequence lengths are not stated as one number. TIMIT frames are 200 samples at 3.1 s average.
- 2026-10-19 (Section 6.5, Figure 2 right): the KL from q to the prior collapses in ELBO-trained models on JSB (N = 16) and stays high under FIVO "even while achieving a higher log-likelihood bound".
- 2026-10-19 (Section 6.4 and 4): computation matching. ELBO runs batch 4N to match FIVO and IWAE at batch 4. Resampling costs O(N) with the alias method.

Not verified from the text: the exact mechanism of the appendix proofs (read only as far as the main body and the proof pointers), and any ESS curve for IWAE.

## 3. Our speculation

- 2026-10-19 Speculation: our row-mode mixture loss is a single-step FIVO with T = 1, so resampling has nothing to act on. A FIVO-style version needs a sequence of intermediate weights. Our token-mode and the per-position sequential Bayes predictive give one weight per chunk, which is the object FIVO's `alpha_t` is built from. Resample at chunk boundaries and only the carried-forward particle indices change.
- 2026-10-19 Speculation: use the Section 6.3 recipe as the default. Resample with ESS < K/2, detach the ancestor choice (no resampling-gradient term), reparameterize `eps_k` so the pathwise term flows. This is the biased estimator they trained all their models with.
- 2026-10-19 Speculation: our proposal is the prior (`eps ~ N(0, I)`, ratio p/q = 1). FIVO's gain from learned `q_t` does not apply until we give `eps_k` a learned, observation-conditioned proposal. With the prior as proposal the filter is a bootstrap filter, and with K = 2 resampling will still discard one particle at nearly every chunk.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- 2026-10-19: Our measurement: end-of-row posterior max 0.985 to 0.998 at K = 2 in a 20-step smoke (no resampling, prior proposal). Paper's claim: without resampling the weights degenerate and the IWAE bound stays loose on long sequences (Section 4, via [14, 15]; Table 1). This agrees in direction. It is a smoke, not a measurement of the bound's gap. Source: Wolfe and the Parcae testbed smoke, branch lxtul/particles, 2026-10-19 (no filing path yet).
