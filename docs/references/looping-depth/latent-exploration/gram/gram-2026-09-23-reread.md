# GRAM re-read for LXTUL-G: the mechanism, the training recipe, and corrections

Read 2026-09-23. This is a second, full read of the paper, done because the LXTUL-G note
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../../.agents/notes/rejected/architecture/2026-09-23-lxtul-gram-stochastic-loop.md))
was drafted from the first reading note ([`gram.md`](gram.md)). This file does not replace
`gram.md`. It adds the recipe details the build needs and corrects three errors in it.

## Citation (verified)

Junyeob Baek, Mingyu Jo (equal contribution), Minsu Kim, Mengye Ren, Yoshua Bengio,
Sungjin Ahn. *Generative Recursive Reasoning*.
[arXiv 2605.19376v2](https://arxiv.org/abs/2605.19376), v1 2026-05-19, v2 2026-05-20
(v2 is still the latest on 2026-09-23). KAIST, Mila, NYU, Universite de Montreal.
Preprint, no venue.

Read in full: the whole PDF, main text and Appendices A to D, including the training
configuration (B.2) and the ELBO-versus-surrogate check (A.3).

Source: the cached PDF from the first read,
[`ignore/papers/2605.19376v2-gram-generative-recursive-reasoning.pdf`](../../../../../ignore/papers/2605.19376v2-gram-generative-recursive-reasoning.pdf),
SHA256 `c53be1192d945358fc0fa31cab4c9d8ffc09fbd686340161de504ebeb24de0f7`.

Code: the project page says "Code (coming soon)" on 2026-09-23. No official code exists.
Two community reimplementations exist (`ad3002/gram`, `DeadByDawn101/GRAM-MLX`). Neither is
the authors' code, so neither is a source for the unstated details below.

## Corrections to `gram.md`

1. **ARC-AGI numbers were wrong.** `gram.md` gives GRAM 66.7 / 16.0 and TRM 55.7 / 9.7 on
   ARC-AGI-1 / ARC-AGI-2. Those are the LLM bars in Figure 3 (GPT 5.2 low 55.7 / 9.7,
   Grok-4-thinking 66.7 / 16.0). Table 8 gives the recursive models:

       ARC-AGI-1   HRM 40.3   TRM 44.6   GRAM 52.0
       ARC-AGI-2   HRM  5.0   TRM  7.8   GRAM 11.1

2. **The mechanism ablation has one more row.** "w/ direct prediction" scores
   63.43 / 61.44 (Sudoku / N-Queens). `gram.md` dropped it.
3. **The architecture ablation (Table 3a) was not quoted.** It is the row set closest to
   MORPH, because one row is a flat looped transformer with the stochastic step and no
   deep supervision:

       base (Looped TF)            61.25 / 71.30
       + DS + HR (= HRM / TRM)     55.00 / 87.40 Sudoku,  80.70 / 72.90 N-Queens
       + SG                        65.64 / 86.30
       + DS + SG                   73.90 / 100.00
       + DS + HR + SG (= GRAM)     93.96 / 99.69

   DS is deep supervision, HR the two-level recursion, SG the stochastic guidance. SG alone
   on a flat loop adds 4.4 points on Sudoku and 15.0 on N-Queens.

The Sudoku row (Looped TF 61.3, HRM 55.0, TRM 87.4, GRAM 97.0) in `gram.md` is right.

## The mechanism, exactly as written

The state is `z = (h, l)`. One transition runs the fast part `K` times with `h` held, then
makes a deterministic proposal for the slow part and adds a learned Gaussian step:

    l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x)        k = 1..K          (Eq. 6)
    u_t     = f_H(h_{t-1}, l_t)                                      (Eq. 7)
    eps_t   ~ p_theta(eps_t | u_t) = N(mu_theta(u_t), sigma_theta^2(u_t) I)   (Eq. 8)
    h_t     = u_t + eps_t                                            (Eq. 9)

- Noise enters the slow state `h` only. Footnote 3: noise in `l` was tried and did not
  help.
- `h` has shape `[B, L, D]`, so the step is a diagonal Gaussian per position and per
  channel.
- The decoder reads `h_T` only.
- `z_0` is drawn once from N(0, I) and stored in the checkpoint. It is fixed, not
  resampled.
- `mu_theta`, `sigma_theta`, `mu_phi`, `sigma_phi` are each a SwiGLU MLP (Table 4). The
  activation that keeps sigma positive (softplus, exp) is not stated.

**Prior and posterior.** Both are Markov chains over the same states, share `z_0`, and
share the transition module (`f_L`, `f_H`). They differ only in the noise distribution:

    prior      p_theta(eps_t | u_t)           sees the input x through u_t
    posterior  q_phi(eps_t | u_t, y)           sees u_t AND the target y

The text never says how `y` enters `q_phi` (embedding, pooling, concatenation). With no
code released, this detail is unknown.

**The ELBO in noise space (Eq. 13).** Because all randomness is in `eps`, and the
decoder reads only the terminal state:

    L_ELBO = E_q[ log p(y | z_Ttotal, x) ]
             - sum_{t=1}^{Ttotal} E_{q(eps_<t)} KL( q_phi(eps_t | u_t, y) || p_theta(eps_t | u_t) )

This is a per-transition KL, summed over the whole trajectory.

**What is actually trained (Eq. 14).** Not Eq. 13. Training uses deep supervision over
`N_sup = 16` supervision steps of `T = 3` transitions each (`T_total = 48`), and
gradients flow only through the LAST transition of each supervision step. Each supervision
step contributes

    L_GRAM = E_q[ log p(y | z_T^(n), x) ] - KL( q(eps_T^(n) | u_T^(n), y) || p(eps_T^(n) | u_T^(n)) )

so only the final transition's KL of each supervision step is in the loss. The authors
call it a biased, truncated surrogate. Appendix A.3 (Figure 8) shows the full ELBO and the
surrogate both fall monotonically on Sudoku and N-Queens; the gap between them is the KL
of the untrained transitions.

**KL weight and collapse guard (Appendix B.2).** These are the recipe numbers `gram.md`
did not have:

- KL coefficient beta per task: Sudoku 0.1; ARC-AGI-1 / 2 0.04 / 0.1; N-Queens 8x8 / 10x10
  0.07 / 0.045; Graph Coloring 8 / 10 nodes 0.5 / 0.45; MNIST 0.07. Figure 13 also plots
  ARC-AGI-1 at beta 0.1, 0.05 and 0.04.
- **KL balancing with coefficient 0.8** "to prevent posterior collapse", citing
  DreamerV2 / V3. In that form the KL is
  `0.8 * KL(sg(q) || p) + 0.2 * KL(q || sg(p))`: the prior moves toward the posterior four
  times faster than the posterior moves toward the prior.
- **No free bits. No KL annealing.** Neither appears anywhere in the paper.
- Whether the KL is summed or averaged over positions and channels is not stated, so the
  beta values do not carry over to another state size as numbers.

**Other training settings.** AdamW, lr 1e-4, weight decay 1.0, gradient clip 1.0, global
batch 768, EMA 0.9999. D = 512, 8 heads, FFN 512, `f_L` and `f_H` each 2 layers of
attention + SwiGLU (SwiGLU + SwiGLU on Sudoku). K = 6 low-level steps on Sudoku, 4
elsewhere; T = 3. About 10M parameters.

**Depth.** Depth is the number of supervision steps at inference. ACT with a Q-head
(HRM / TRM style) halts each trajectory; the released variant halts when
`sigmoid(q_halt) > 0.5` (A.1). The halt loss does not reach the core. All Table 8 numbers
use 16 supervision steps.

**Width and selection.** N trajectories are drawn from the prior; each terminal state is
decoded. Selection is majority vote or best-of-N by the Latent Process Reward Model
(LPRM, A.2): a value head `v_psi(z_t)` trained jointly by

    L_LPRM = sum_{t=1}^{T} ( v_psi(z_t) - r )^2,   r in [0, 1] = accuracy of the final prediction

and the candidate with the highest terminal value wins. The value head reads the first
token of `h` (B.1).

## The numbers that matter to LXTUL-G

Mechanism ablation, 5 samples (Table 3b), Sudoku / N-Queens:

    GRAM                              93.96 / 99.69
    w/o stochastic guidance            82.87 / 72.91
    stochasticity only N(0, sigma)     94.88 / 50.27
    guide only N(mu, 0)                 0.00 /  0.00
    w/ direct prediction               63.43 / 61.44
    TRM w/ stochastic decoder          82.87 / 71.66
    TRM w/ random init                 78.53 / 71.82

The authors explain the 0.00 row: "deterministic guidance conditioned on the target leads
to severe overfitting". With sigma = 0 the posterior's mean carries the target straight
to the decoder at train time and the prior's mean cannot reproduce it at test time. That
is the exposure gap in its pure form, the same trap as MORPH's hindsight fitted z
(`gram.md`, "What in MORPH already tested this"). The Gaussian's variance is the thing
that closes it.

Width against depth: GRAM at N = 20 and 16 supervision steps reaches 97.0 on Sudoku, above
TRM at 320 iterations (90.5). The 97.0 in Table 8 matches that N = 20 point; the N = 1
value is only in Figure 4 as a curve, not as a number in the text.

Width is worth less when data already covers the diversity (Appendix D.2, Figure 14). On
ARC-AGI-1 without augmentation, accuracy rises with N up to 500. With 50x augmentation,
accuracy is flat from N = 1 to N = 50.

Multi-solution tasks (Table 1, 20 samples for coverage):

    N-Queens 8x8     accuracy / coverage   TRM 66.8 / 36.1   AR 96.3 / 84.8   GRAM 99.7 / 90.3
    N-Queens 10x10                         TRM 17.5 / 2.0    AR 90.0 / 53.2   GRAM 89.7 / 57.5
    Graph Col. 8     conflicts / coverage  AR 19.0 / 83.0    MDLM 2.7 / 84.5  GRAM 2.7 / 85.8
    Graph Col. 10                          AR 61.3 / 40.0    MDLM 12.0 / 48.2 GRAM 3.3 / 51.3

On N-Queens 10x10 the autoregressive transformer has the higher accuracy (90.0 against
89.7). GRAM's lead is in coverage and in constraint satisfaction.

Unconditional MNIST (Table 2): TRM collapses (IS 1.00, FID 303.29). GRAM trained at 16
steps improves with more steps at inference, FID 84.08 at 8 steps to 73.34 at 256.

Cost (Table 7, 8x RTX 4090): Sudoku 2 h, N-Queens 1 to 3 h, MNIST 16 h, ARC-AGI 5 days.
The conclusion names the sequential deep supervision as the barrier to scaling.

An independent warning: the one outside reimplementation, run at 10 to 100x less compute
than the paper's recipe, emitted a wrong grid on 98.9 to 100 % of 6x6 Sudoku puzzles
(arXiv 2607.19635, section 5.1; reading note
[`../search-inertness/search-inertness.md`](../search-inertness/search-inertness.md)). That
says nothing about GRAM's ceiling. It does say the recipe does not work cheaply.

## What LXTUL-G should take

- **The step form.** `h_t = u_t + eps_t`, with `u_t` the deterministic core update and
  `eps_t` a learned diagonal Gaussian from a small MLP head on `u_t`. Noise on the slot
  state only, every pass.
- **One shared core for prior and posterior.** Only the noise heads differ. The posterior
  head reads `u_t` and an encoding of the target (for us, the next span).
- **Learned mean AND learned variance in both heads.** The ablation says both are needed
  on the multi-solution task, and text is multi-solution. A mean-free arm is the
  "stochasticity only" control.
- **The per-pass KL summed over passes.** MORPH runs full BPTT, so the exact Eq. 13 sum
  is affordable where GRAM had to truncate. This removes GRAM's surrogate bias.
- **KL balancing at 0.8 as the collapse guard, beta near 0.1 as the starting weight.**
  These are the paper's actual choices. Free bits is not in the paper.
- **The LPRM as an instrument first.** A value head on the exit state regressed onto a
  detached per-span score. On text the score is the realized span CE under that sample,
  not accuracy.
- **The control rows.** "TRM + stochastic decoder" (output-only randomness), "random
  init" (noise at the entry only) and "stochasticity only" (mu = 0) are the three
  controls that separate the variational step from plain randomness.

## What does not transfer

- **Deep supervision and truncated gradients.** Every GRAM number uses 16 supervision
  steps with gradients through one transition each. MORPH's slot loop is one pass-set with
  full BPTT and no per-step decode loss. The closest GRAM row to MORPH is "+ SG" in
  Table 3a, which is 28 points below full GRAM on Sudoku.
- **The two-level state.** GRAM's gains need HR on Sudoku (73.90 to 93.96). The slot loop
  has one state.
- **The tasks.** Closed-form-checkable puzzles with one target per input, trained to near
  zero error. Next-span prediction on web text has one realized continuation per context
  and irreducible entropy. The LPRM's accuracy target has no text analogue except a
  CE-based score.
- **Beta values as numbers.** The normalisation of the KL is unstated and the state size
  differs, so 0.1 is a starting point, not a transferred constant.
- **The width result's premise.** Figure 14 shows sampling gains vanish when the training
  data already covers the diversity. A web-text model trained on billions of tokens is
  closer to the augmented regime than to the 1000-puzzle regime.
- **How the posterior reads y.** Unstated. LXTUL-G has to choose its own encoder of the
  next span.
