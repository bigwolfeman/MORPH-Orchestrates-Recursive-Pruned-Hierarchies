# Paper note: Auto-Encoding Sequential Monte Carlo (AESMC)

- **Title:** Auto-Encoding Sequential Monte Carlo (Tuan Anh Le, Maximilian Igl, Tom Rainforth, Tom Jin, Frank Wood; ICLR 2018)
- **arXiv or URL:** 1705.10306 (v2, 2018-04-05). The id and authors were read from the paper's own first page. This v2 is the ICLR version. A v1 preprint exists and differs.
- **Local paper text:** docs/references/smc-latent-objectives/aesmc-1705-10306/ (paper.txt, paper.pdf, SOURCE.md)
- **Date first noted:** 2026-10-19

## 1. Why we reference it

- 2026-10-19: It is the second independent derivation of the SMC bound (with FIVO and VSMC). It adds two things we need. First, a KL form of the bound's gap that says what a perfect objective would force the intermediate weights to be. Second, a measured warning: a tighter bound (larger K) makes the proposal's gradient noisier. This matters if we learn a proposal for `eps_k`.
- 2026-10-19: Mixture particles use K draws of `eps_k` as an IWAE-style bound. The K-vs-gradient-noise result applies directly to any learned `sigma` or learned `eps` proposal we add.

## 2. What the paper shows

- 2026-10-19 (Section 2.2, Algorithm 1, Eq. 1): SMC with proposals `q_t(x_t | y_1:t, x_1:t-1)` per particle, ancestor `a_{t-1}^k ~ Discrete(normalized w_{t-1})`, weights `w_t^k = f(x_t^k | ancestor) g(y_t | x_1:t^k) / q_t(...)`. Estimator `Z_SMC = prod_t (1/K) sum_k w_t^k`. Resampling is at every step. The paper says resampling "allows the algorithm to focus on promising particles in light of new observations, avoiding the exponential divergence between the weights of different samples that occurs for importance sampling as T increases". It cites Del Moral 2004 ch. 9 and does not run a degeneracy experiment of its own.
- 2026-10-19 (Section 2.3, Eq. 2 and 3): IWAE bound `ELBO_IS = E_{Q_IS}[log Z_IS]`, `Z_IS = (1/K) sum_k p(x_k, y)/q(x_k | y)`. K = 1 is the VAE.
- 2026-10-19 (Section 3.1, Eq. 5 and 6): `ELBO_SMC = E_{Q_SMC}[log Z_SMC]`, with `Q_SMC` the joint over particles and ancestor indices.
- 2026-10-19 (Section 3.2 and Appendix A): gradient. The reparameterized pathwise term is kept. The REINFORCE term for discrete ancestor indices is dropped: "the additional term in the estimator has problematically high variance ... despite introducing a small bias". Appendix A (A.1 full REINFORCE, A.2 REINFORCE plus reparameterization, and further variants) compares estimators. I read the derivations, not the variance tables, so I cannot give a number for the variance gap.
- 2026-10-19 (Section 3.3, Claim 1, Eq. 10, 11): any log of an unbiased non-negative estimator has gap `log Z - KL(Q || P)` on an extended space, with `P = Q * Z_hat / Z`. For IS this gives `P_IS` as a mixture over which of K samples is the posterior draw (Eq. 12, 13). For SMC it gives Eq. 14, 15.
- 2026-10-19 (Proposition 1): `Q_IS = P_IS` iff `q(x | y) = p(x | y)`. (Proposition 2): for K > 1, `P_SMC = Q_SMC` iff (1) the intermediate targets `pi_t(x_1:t)` are the marginals `p(x_1:t | y_1:T)` of the final posterior and (2) each `q_t` is the conditional `p(x_1:t | y_1:T) / p(x_1:t-1 | y_1:T)`. So a perfect SMC bound needs proposals that condition on all of y_1:T (future observations) and intermediate targets that are smoothing marginals, "not just the correct individual proposals". Most models cannot satisfy condition 1, so the bound is not made tight in general. This differs from FIVO's Proposition 2, which says that under an independence condition the optimal `q_t` uses no future observations. The two statements are about different targets: AESMC uses smoothing marginals as `pi_t`, FIVO uses the filtering targets `p(z_1:t | x_1:t)`.
- 2026-10-19 (Section 4, Figure 1, Eq. 16): tighter bounds hurt proposal learning. For IWAE, the signal-to-noise ratio of the proposal gradient is `O(sqrt(1/K))` (from Rainforth et al. 2017). The mean gradient shrinks faster than its standard deviation. At K > 10 in the Section 5.2 model, the gradient sign is about as likely positive as negative. The model gradient `grad_theta` does not shrink with K.
- 2026-10-19 (Section 4.1): alternating ELBOs (ALT). Use SMC with large `K_theta` for the model and IS with small `K_phi` for the proposal.
- 2026-10-19 (Section 5.1, Figure 2): linear Gaussian SSM, T = 200, true theta = (0.9, 1.0), bootstrap proposal, 500 steps at lr 0.01. SMC beats IS and 10000 particles beat 10 for both theta and log marginal likelihood speed.
- 2026-10-19 (Section 5.2, Figure 3): Gaussian unknown-mean model, `x = 2.3`, proposal `N(mu_q, sigma_q^2)` with free `(mu_q, log sigma_q^2)`, initialized at 0.01. Optimizing `ELBO_IS` w.r.t. phi: the learned `(mu_q, sigma_q^2)` get further from the analytic posterior as K grows (the plot axis runs K = 1 to 10000; the exact K set is not recorded here). On the LGSSM, ALT with `(SMC, 1000)` for theta and `(IS, 10)` for phi gives a better phi than SMC 1000 for both.
- 2026-10-19 (Section 5.3, Figure 4): Moving Agents video, N = 5000 sequences (1000 test), T = 40, 32 x 32 frames, half occluded, VRNN, K in {10, 20, 40}, 3 seeds. Final test `max(ELBO_IS, ELBO_SMC)`: IWAE -357.3 / -356.6 / -356.2 and AESMC -356.7 / -356.1 / -356.1. ALT "was much less stable than IWAE and AESMC" there. The gains are small: 0.1 to 0.6 nats on a value near -357.
- 2026-10-19 (proposal form): `q_t` conditions on `y_1:t` and the particle's own ancestor path `x_1:t-1`. No future observations are used in any experiment. The Moving Agents VRNN proposal is a network of the VRNN state and the current frame (Appendix C.1, not read in full).

## 3. Our speculation

- 2026-10-19 Speculation: the Section 4 result is the reason not to learn `eps` proposals with large K. Our K is 2 to 4 so SNR is fine now. If K grows to 16 or 64 for test-time compute, train any learned `sigma` with an IS bound at small K, as in ALT, and the model with the big-K bound.
- 2026-10-19 Speculation: Proposition 2 says a perfect bound needs the proposal to see the future. In an LM the future tokens are the targets we score. A particle proposal that sees them is a smoothing proposal and cannot run at inference. Only filtering proposals (on `y_1:t`) transfer. FIVO's Table 2 says smoothing did not help FIVO anyway.

## 4. Corrections

- None recorded yet.

## 5. Where our findings disagree

- None recorded yet. AESMC makes no claim about a prior-as-proposal, no-resample estimator beyond the IWAE special case.
