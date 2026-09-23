# Variational Reasoning for Language Models: reading note

Read 2026-09-23, for the LXTUL-G build
([`2026-09-23-lxtul-gram-stochastic-loop.md`](../../../../../.agents/notes/proposed/architecture/2026-09-23-lxtul-gram-stochastic-loop.md)).

## Citation (verified)

Xiangxin Zhou, Zichen Liu, Haonan Wang, Chao Du, Min Lin, Chongxuan Li, Liang Wang,
Tianyu Pang. *Variational Reasoning for Language Models*.
[arXiv 2509.22637v2](https://arxiv.org/abs/2509.22637), v2 dated 2025-10-15. Sea AI Lab,
UCAS, CASIA, NUS, Renmin University. No venue on the PDF or the arXiv page.

Read in full: all 35 pages, main text and Appendices A to G, including every derivation.

## Local cache

Not cached into `ignore/papers/` in this pass (the brief allowed only the note files).
Fetched from `https://arxiv.org/pdf/2509.22637` on 2026-09-23: 1,159,527 bytes, 35 pages,
SHA256 `45263477211aea02127a85aeceb0fb5061501c21748b68aff016ea0cc3cd2c78`. Suggested cache
name: `ignore/papers/2509.22637v2-variational-reasoning.pdf`.

## What it actually does

The latent is a discrete thinking trace `z` (tokens), not a continuous state. A question
`x` gives a trace `z` then an answer `y`. The goal is `log P(Y_x | x)`, the probability of
any correct answer, marginal over traces.

**The posterior.** A second copy of the language model, `q_phi(z | x, y')`, sees the
question and an answer hint `y'` wrapped in hint delimiters after `x`. The hint is a
correct answer drawn from the oracle set. It is initialised from the reasoning model
(`phi_0 <- theta_0`) and fine-tuned separately; the two models do not share weights
(Appendix C.1).

**The ELBO (Eq. 2 and 4).**

    log P(Y_x | x) >= E_{q(z|x,y')}[ log pi(Y_x | x, z) ] - KL( q(z|x,y') || pi(z|x) )
                   =  log P(Y_x | x) - KL( q(z|x,y') || P(z | x, Y_x) )

so the ELBO's optimal posterior is the true posterior `P(z | x, Y_x)`, the prior reweighted
by how often each trace yields a correct answer.

**The multi-trace bound (Eq. 5), IWAE style.**

    L^K = E_{z_1..K ~ q} log (1/K) sum_k  pi(z_k, Y_x | x) / q(z_k | x, y')

It tightens as K grows. The gradient for the reasoning model (Eq. 6) is a weighted
likelihood:

    grad_theta L^K = E[ sum_k rho~_k grad_theta log pi(z_k, Y_x | x) ],
    rho_k = [pi(z_k | x) / q(z_k | x, y')]^(1/|z_k|) * E_{y ~ pi(y|x,z_k)} 1(y in Y_x)   (Eq. 8)

with `rho~` the weights normalised over the K traces. Two estimator choices matter:

- The trace likelihood ratio is replaced by its per-token GEOMETRIC MEAN. The raw ratio
  over thousands of tokens has too much variance. This is biased.
- `pi(Y_x | x, z)` is estimated by the ACCURACY of sampled answers (8 per trace), not by the
  likelihood of a reference answer. Theorem 1: the accuracy estimator has lower worst-case
  variance whenever accuracy is at least `1/|Y_x|`, which holds when many answer strings
  are correct.

**The posterior is trained by FORWARD KL, not by the ELBO (Section 2.3, Eq. 9).**
Training `q` by the ELBO minimises the reverse KL `KL(q || P(z|x,Y_x))` with samples from
`q`. In their pilot runs the posterior "may struggle to effectively use hints ... without
collapsing into shortcut reasoning (e.g., directly leaking answer tokens into the thinking
trace)". They switch to

    grad_phi KL( P(z|x,Y_x) || q ) ~= E_{z_1..M ~ pi(z|x)} sum_m w~_m grad_phi log q(z_m | x, y'),
    w_m = pi(Y_x | x, z_m)

which is weighted SFT of the posterior on the PRIOR's own samples, weighted by their
success. It is Reweighted Wake-Sleep (Bornschein and Bengio 2015), an approximation, not a
bound. Appendix A.8 adds that the true posterior is also the minimum-variance behaviour
policy for binary-reward RL.

**How they handle the prior/posterior exposure gap.** Four things, all visible in the
algorithm:

1. The posterior starts as a copy of the prior, so it starts close.
2. Forward KL trains `q` only on traces the prior already produces. `q` covers the prior's
   successful modes and cannot drift to a region the prior never visits.
3. The prior's update weights each posterior trace by `(pi/q)^(1/|z|)`. A trace the prior
   finds unlikely is down-weighted.
4. In the 17k-data runs only the highest-weight trace of 8 is kept, and it is mixed with
   the original data.

**The side result.** RFT and binary-reward RL (including GRPO) are local forward-KL
objectives toward the model's true posterior, weighted by the model's accuracy on the
question (Eq. 10 to 12). The weight biases them toward easy questions. Their objective
drops that weight.

**Pipeline.** One round (T = 1): train `pi_theta0` by SFT; train `q_phi` by Eq. 9; sample
8 traces per question from `q` (temperature 0.7); compute weights with `pi_theta0` and
`q`; train the final model by weighted SFT. They call the SFT variant "Dr. SFT": sum token
losses and divide by a constant, not by the token count.

## The key numbers

Qwen3-4B-Base on Bespoke-Stratos-17k (Table 1), average of the math block / average of the
other block:

    Qwen3-4B-Base            21.38 / 18.26
    General-Reasoner-4B      41.54 / 36.90
    Bespoke-Stratos-4B       51.35 / 40.40
    Ours-PB-GML-4B           55.23 / 45.60
    Ours-PB-Acc-4B           55.72 / 46.12

Conditioning the posterior on the hint (Table 4, same model): without `y'` the traces come
from the initial reasoning model, and the averages fall from 55.72 / 46.12 to 48.18 /
37.80. LCB-Medium falls from 33.68 to 16.63.

Estimator choice (Table 5, Qwen2.5-7B, 1k data), math average: Acc 45.41, GML 45.38,
naive likelihood 43.01; SFT baseline 40.41.

Weighted multi-trace against best-single-trace (Table 8, 1k data), math average:
multi-trace without mixing 45.41, with mixing 44.16. Posterior trained on a disjoint 16k
split (Table 9): 45.17 to 45.65, so the posterior generalises to unseen questions.

Scaling K from 1 to 32 posterior traces improves the final model (Figure 3); the text
gives no numbers. The Pass@K advantage grows with K on hard sets (LCB-Hard) and shrinks on
easy ones (Figure 2).

Training cost: H100s; the posterior trains for 10 epochs; every trace needs 8 sampled
answers for the accuracy estimator.

## What LXTUL-G should take

- **The hint is the answer, not the trace.** Their posterior sees the target output `y'`
  and must invent the latent. That matches LXTUL-G: the posterior sees the next span and
  proposes the pass steps. The Table 4 ablation (55.72 to 48.18 without the hint) is the
  paper's evidence that target conditioning is load-bearing.
- **Start the posterior from the prior.** Initialise the posterior noise head as a copy of
  the prior head with a zero-initialised extra input for the span encoding. At step 0 the
  two distributions are identical and the KL is zero.
- **Forward KL for the posterior is the fallback against a leak.** Their reverse-KL
  posterior leaked answer tokens into the trace. If the LXTUL-G posterior finds a
  shortcut (a step the prior can never produce), the replacement is Eq. 9's form on our
  Gaussians: draw N prior steps, score each by the realized span CE through the coda
  (detached), and fit the posterior head by weighted log-likelihood of those prior draws.
  The posterior then lives inside the prior's support by construction.
- **The multi-sample bound as an instrument.** With the PRIOR as the proposal, Eq. 5
  becomes a plain importance-weighted likelihood of the span:

      ce_iw@N = -(1/L) log (1/N) sum_n exp( -L * ce_n ),   z_n ~ prior

  where `ce_n` is the span's mean token CE under sample n and L the span length. It is an
  honest likelihood (no hindsight choice), it lies between the oracle `min_n ce_n` and the
  oracle plus `log N / L`, and it improves with N only if different samples explain the
  span differently.
- **Per-token normalisation of long log-ratios.** Any importance weight that uses the
  posterior as proposal multiplies `p/q` over T passes and d channels. Their geometric
  mean is the known fix, at a known bias.
- **The accuracy-weighting bias warning.** Any credit scheme that weights samples by a
  success rate emphasises easy spans. If a value head or a per-sample weight enters the
  loss later, normalise it per slot.

## What does not transfer

- **The latent is discrete text.** `z` is thousands of sampled tokens; `q` and `pi` are full
  language models; likelihood ratios are sequence probabilities. The LXTUL-G latent is a
  few Gaussian steps per slot, and its KL is closed form.
- **The training is offline and one round.** Samples come from frozen `q` and `pi_theta0`;
  the final model is trained by weighted SFT on a fixed dataset. LXTUL-G trains prior,
  posterior and reader jointly, online, by reparameterised gradients.
- **The reward is a verifier.** Their weights use answer correctness from math and code
  verifiers. Text spans have only CE.
- **No collapse problem to solve.** The reasoning model must produce the trace to reach the
  answer; there is no bypass around `z`. In MORPH the coda has the token path, which is the
  central problem. This paper has nothing to say about a reader that ignores the latent.
- **No per-step structure.** The trace is one latent; there is no per-pass KL, no depth
  axis and no loop.
