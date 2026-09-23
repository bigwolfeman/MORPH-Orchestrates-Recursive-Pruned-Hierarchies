# GRAM (Generative Recursive Reasoning): reading and local cache

Read 2026-09-18.

## Citation (verified)

Junyeob Baek, Mingyu Jo (equal contribution), Minsu Kim, Mengye Ren, Yoshua Bengio,
Sungjin Ahn. *Generative Recursive Reasoning*.
[arXiv 2605.19376v2](https://arxiv.org/abs/2605.19376), submitted 2026-05-19,
v2 dated 2026-05-20. Affiliations: KAIST, Mila, NYU, Universite de Montreal.
The PDF states "Preprint"; no venue is claimed. Project page:
<https://ahn-ml.github.io/gram-website>.

The brief's short title was "Generative Recursive Reasoning" with the acronym GRAM. Both
are correct: the paper expands GRAM as Generative Recursive reAsoning Models.

Read in full: abstract through Section 5, including both ablation tables. Appendices A
through D were skimmed.

## Local cache

- [PDF](../../../../../ignore/papers/2605.19376v2-gram-generative-recursive-reasoning.pdf)
- [Extracted text](../../../../../ignore/papers/2605.19376v2-gram-generative-recursive-reasoning.txt)
- 5,136,087 bytes, 27 pages, 1,497 text lines.
- SHA256: `c53be1192d945358fc0fa31cab4c9d8ffc09fbd686340161de504ebeb24de0f7`.

## What it actually does

Recursive Reasoning Models (HRM, TRM, Looped Transformers) refine one latent state with
a shared transition function. Given the same input they follow ONE trajectory. GRAM makes
that transition stochastic and trains the result as a latent-variable generative model.

The state splits into a slow high-level part h and a fast low-level part l. Within one
transition, l is refined K times deterministically with h held fixed. Then the high level
computes a deterministic proposal and a learned Gaussian perturbation is ADDED to it:

    l_{t,k} = f_L(h_{t-1}, l_{t,k-1}, e_x)      k = 1..K      deterministic
    u_t     = f_H(h_{t-1}, l_t)                               deterministic proposal
    eps_t   ~ N(mu_theta(u_t), sigma^2_theta(u_t) I)          learned guidance
    h_t     = u_t + eps_t                                     the stochastic step

Noise enters only at the high level. They report trying it at the low level and finding
no gain.

Training maximises an ELBO. A variational posterior q_phi sees BOTH the input x and the
target y and proposes the noise; the prior p_theta sees only x; a KL term ties them. In
practice the whole thing runs under deep supervision over N_sup supervision steps, each
of T transitions, with gradients propagated only through the FINAL transition of each
supervision step. The authors are explicit that this makes the objective a biased
truncated surrogate, not the exact ELBO.

At inference there are two scaling axes: DEPTH (more transitions, with ACT halting) and
WIDTH (sample N trajectories from the prior in parallel). Candidates are selected by
majority vote or by a Latent Process Reward Model, a value head v_psi(z_t) regressed on
final prediction accuracy.

## The key numbers

Structured reasoning, single setting:

    task            Looped TF   HRM     TRM     GRAM
    Sudoku-Extreme  61.3        55.0    87.4    97.0
    ARC-AGI-1        -          44.6    55.7    66.7
    ARC-AGI-2        -           7.8     9.7    16.0

Width versus depth on Sudoku: GRAM with N = 20 samples at 16 iterations beats every
deterministic baseline at 320 iterations, including TRM (97.0 against 90.5), at
comparable compute.

Multi-solution coverage, N-Queens 8x8, unique valid solutions found with 20 samples:
Looped TF 23.6, HRM 26.7, TRM 36.1 +/- 22.5, GRAM 90.3 +/- 1.9. Accuracy 99.7 for GRAM
against 96.3 for an autoregressive transformer and 96.1 for MDLM. On Graph Coloring
10-vertex, conflict edges 3.3 for GRAM against 61.3 for AR.

Unconditional generation on binarised MNIST: TRM collapses (FID 303.29); GRAM reaches
FID 73.34 at 256 steps although it was trained at 16, with IS climbing 1.85 to 2.04.

The mechanism ablation is the part that matters to us (5 samples, Sudoku / N-Queens):

    GRAM                                93.96 / 99.69
    w/o stochastic guidance (mu = 0)    82.87 / 72.91
    stochasticity only (N(mu=0, sigma)) 94.88 / 50.27
    guidance only (N(mu, sigma = 0))     0.00 /  0.00
    TRM + stochastic decoder            82.87 / 71.66
    TRM + random z_0 init               78.53 / 71.82

## Where it sits in MORPH's 2x2

Stochastic, K streams, with training that knows about the streams. It is the strongest
published fill of MORPH's empty cell, and it fills it with a variational objective
rather than a diversity penalty.

## What in MORPH already tested this

The K-stream half was run as the Thought Register and collapsed (cells rank 1.24 of 4,
cosine 0.94). The register had no stochastic transition and no posterior, so it is
closest to GRAM's "w/o stochastic guidance" row, which is also their weakest non-broken
variant.

The stochastic half was run as LCTUL. `tul.code` and `tul.code_discrete` are a sampler
over a code, but the sampler is context-blind (past worth 1 % of the flow loss), and
Euler depth on an affine field is provably flat
(`lab/theory/lctul_euler_depth/`). GRAM's guidance is state-dependent AND conditioned
on the target at train time, which is exactly the part LCTUL's field never learned.

The target-conditioned posterior half has been measured, and the measurement is a
warning. `lab/divergence/slot_z_optimize.py` fits a z with the realised next span in the
loss and reaches -1.479 nats against the loop. The CAUSAL version, fitted on teacher
samples and scored on the untouched real span, is +0.252 nats WORSE than the loop
(`fitted-z-used-the-answer`). A posterior that sees the answer is not evidence that the
prior can reach it; GRAM's KL term is precisely the machinery that is supposed to make
that transfer, and it is the piece MORPH has never built.

## What a MORPH arm implementing it would need

Exists:

- `tul.slot_cells` for M mutable cells, with `prefix_k == slot_cells`.
- `tul.recur_gate_noise` and `tul.recur_gate_tau`, which already add noise inside the
  loop, but to a GATE, not to the state.
- `tul.slot_depth_fixed`, `tul.slot_mean_depth`, `tul.slot_max_depth` for the depth axis.
- `tul.code_grade` with `code_grade_k`, which already samples K continuations and ranks
  them with a grader. That is a selector, but it sits OUTSIDE the loop and grades coda
  samples, not latent trajectories.

Does not exist:

- A per-pass Gaussian transition on the slot state with LEARNED mean and variance. The
  slot loop's core step is deterministic.
- A target-conditioned posterior branch. Nothing in `TULSlots` reads the next span at
  train time except the frozen encoder E in `tul.code_target`, and that is a regression
  target, not a proposal distribution.
- A KL term between a prior transition and a posterior transition. `tul.sigreg_lambda`
  is a distributional regulariser on the states, not a per-step KL between two policies.
- A Latent Process Reward Model over slot trajectories. `tul.critic_weight` and
  `tul.egrad_disc_*` are the closest, and both were measured flat on passes 2 to 6.

The honest cost estimate: this is the largest of the nine arms. It needs a second head
for (mu, sigma), a posterior branch that sees the next span, a KL term, and a value head.

## The instrument it implies

Their Figure 4 (right) is the instrument: accuracy plotted against the number of valid
solutions. The MORPH analogue is CE plotted against the entropy of the next span given
the past, estimated from teacher samples. Their claim is that deterministic recursion
degrades as the number of valid continuations rises while a stochastic one does not. On
web text almost every span has many valid continuations, so if that claim transfers the
deterministic slot loop should be worst exactly where the span entropy is highest.

The second instrument is oracle-over-stream: decode N trajectories and score the best.
That is coverage in their terms and pass@N in Parallel-TTS's terms.

## What it does NOT establish for us

It does not establish anything on next-token text. Sudoku, ARC-AGI, N-Queens, graph
colouring and binarised MNIST are all closed-form-checkable tasks with a discrete answer.

It does not show that stochasticity alone works. The "TRM + random init" and
"TRM + stochastic decoder" rows are their own control against that reading, and both are
roughly 15 points below GRAM on Sudoku.

It does not separate width from the variational objective. Every GRAM number carries
both. The one clean factor is the mechanism ablation, and there the pure-noise variant
holds on Sudoku (94.88) and collapses on N-Queens (50.27).

It does not carry a stability story for a shared core under full BPTT. Their gradient is
truncated to the last transition of each supervision step, which is the opposite of the
full-BPTT recipe MORPH runs.


> **Correction 2026-09-23:** a full re-read of the PDF is in [gram-2026-09-23-reread.md](gram-2026-09-23-reread.md). The ARC numbers above came from the LLM bars in Figure 3; GRAM scores 52.0 on ARC-AGI-1 and 11.1 on ARC-AGI-2. Posterior collapse is guarded by KL balancing at 0.8, not free bits or annealing; beta is 0.04 to 0.5 (0.1 on Sudoku).
