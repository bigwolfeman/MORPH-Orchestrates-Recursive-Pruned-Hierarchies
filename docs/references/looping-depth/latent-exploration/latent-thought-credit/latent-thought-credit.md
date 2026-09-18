# Latent Thought Credit (LTC): reading and local cache

Read 2026-09-18.

## Citation (verified)

Xuyang Zhao, Liting Zhang, Zichen Xu, Yong Chen, Wenjia Zeng, Shiwan Zhao,
Qicheng Li. *Latent Thought Credit: Multi-Answer Credit Assignment for Latent
Reasoning*. [arXiv 2608.01593v1](https://arxiv.org/abs/2608.01593), submitted
2026-08-03. Affiliations: TMCC, Nankai University, and Lingxi (Beijing) Technology.
No venue is claimed in the PDF; the template is AAAI-style.

Read in full: 8 pages, all of it.

## Local cache

- [PDF](../../../../../ignore/papers/2608.01593v1-latent-thought-credit.pdf)
- [Extracted text](../../../../../ignore/papers/2608.01593v1-latent-thought-credit.txt)
- 518,467 bytes, 8 pages, 630 text lines.
- SHA256: `5f6b775d68dcf40d11b70747efc7e298c0d0b8ca68577ca0e2f1964f37335717`.

## What it actually does

The problem is credit assignment. A latent thought is judged only through the answer it
leads to, and that answer's reward mixes the thought's quality with the noise of answer
sampling. One answer is a noisy estimate of a thought's value.

LTC fixes the estimator, not the architecture. For each prompt:

    sample K latent thoughts       (Gumbel-Softmax over the vocabulary, then
                                    an embedding mixture, so each thought is a soft token)
    FREEZE the context after each thought
    sample M answers from each frozen context
    mu_hat_i = mean of the M rewards            the thought's estimated value

    A_think_i = (mu_hat_i - mean_i')/std_i'     over the K thoughts
    A_ans_ij  = (r_ij - mean)/std               over all K*M answers

Latent-thought positions are updated with the thought-level advantage; answer positions
with the answer-level advantage. A third term, thought matching, pushes the current
policy's clean top-k embedding prediction toward the high-credit rollout thoughts,
weighted by a softmax over thought advantages.

The whole thing runs inside GRPO on Qwen2.5-Instruct, with a rollout budget
B = K * M = 8 and the main setting (K, M) = (2, 4).

## The key numbers

Average over GSM8K, MATH, MATH500, MMLU-STEM, ARC-C:
Qwen2.5-3B GRPO 68.45, GRPO-MA 68.59, HRPO 69.28, LTC 70.16.
Qwen2.5-7B GRPO 68.47, GRPO-MA 71.62, HRPO 74.08, LTC 75.31. On 7B the per-task picture
is mixed: HRPO still wins MATH (67.40 against 67.22) and ARC-C (84.00 against 82.50).

Component ablation, GSM8K / MATH deltas from full LTC (84.46 / 58.40):
without thought matching -3.11 / -1.20; without hierarchical credit -2.91 / -2.30;
without Gumbel noise -2.12 / -1.70; without the latent thought -1.22 / -3.80.

Budget allocation on GSM8K: at B = 8, (K, M) = (2, 4) scores 84.46 and (4, 2) scores
82.49, so spending the budget on ANSWER replication beats spending it on more thoughts
by 1.97 points. At B = 16 the balanced (4, 4) is best at 85.15, ahead of (2, 8) at 84.52
and (8, 2) at 83.10. At K = 2, going from M = 1 to M = 4 buys 2.19 points and M = 8 buys
a further 0.06.

Their Table 4 is the number this repository should keep. Fixed-context correctness
variance, 256 prompts:

    setting              K    M    between   within    ratio
    initial policy       4    40   0.0244    0.0376     1.54
    final policy         4    40   0.0043    0.0568    13.10
    final policy, high-K 32   64   0.0065    0.0583     8.99

"Between" is the variance of thought-level mean correctness; "within" is answer-level
variance under a fixed context. Training COLLAPSES between-thought variance by 5.7x
while within-thought noise rises. After training, the K thoughts barely differ.

They read that as motivation for multi-answer averaging: the signal to be extracted is
small relative to the noise, so more answers per thought is the right spend. The reading
that matters to MORPH is the other one.

## Where it sits in MORPH's 2x2

Stochastic, K streams, with the streams trained by RL and a selector implicit in the
advantage. It also supplies the most direct published MEASUREMENT of K-stream collapse
under training.

## What in MORPH already tested this

The collapse. Their between-thought variance falling 0.0244 to 0.0043 is our four cells
going from seeded-apart to rank 1.24 of 4 at cosine 0.94. Ours is a geometry reading,
theirs is a downstream-utility reading, and both say a trained model stops producing
differing latents.

The budget split has a MORPH analogue that has been run. `tul.code_grade` with
`code_grade_k` samples K coda continuations and grades them, which is the M axis.
`tul.slot_cells` is the K axis. LTC says at a fixed budget the M axis pays more, and
MORPH ran the K axis (register, flat) before the M axis.

The one-step optimum shows up here too. Their thought-matching term regresses the
current policy toward high-credit rollout thoughts, which is a conditional mean over
selected samples. It is their largest single contributor on GSM8K (-3.11 without it).
That is a warning for reading `tul.code_grade`: the graded target also regresses the
loop toward a selected mean, so its gains do not demonstrate exploration.

## What a MORPH arm implementing it would need

Exists: `tul.code_grade` (K sampled coda continuations, a grader that cannot see the
graded span, a push toward the best), `tul.code_grade_k`, `code_grade_rows`,
`code_grade_tokens`, `code_grade_loss`, `code_grade_min_distinct2`. That is the M axis
with a fixed thought. `tul.slot_cells` is the K axis.

Does not exist:

- A crossing of the two axes. Today `code_grade` grades continuations of ONE cell. LTC's
  design is K cells, each with M graded continuations, and a per-cell advantage from its
  own M-mean. The forward already writes M cells, so the change is in the grading loop,
  not the model.
- A between-versus-within variance instrument. That is their Table 4 and it is the
  cheapest thing in this whole batch to add: for each of K cells, sample M coda
  continuations, score them, and report the two variances and their ratio. It needs no
  training.
- Any RL. MORPH trains by CE, not by policy gradient with a verifier. LTC's advantages
  need a scalar reward per answer; the language-modelling analogue is the negative CE of
  the true continuation, which turns the "reward" into a differentiable quantity and
  makes the whole GRPO frame unnecessary. That is worth saying out loud before anyone
  ports the algorithm.

## The instrument it implies

Between-stream versus within-stream variance, measured at the start of training and at
the end. If between-cell variance of downstream CE collapses while within-cell variance
holds, MORPH has reproduced their Table 4 and the streams have stopped exploring. This
instrument is strictly better than a rank reading because it is stated in the units the
loop is judged in.

## What it does NOT establish for us

It does not establish that K streams help. Its own budget table says at B = 8 the
2-thought split beats the 4-thought split.

It does not isolate the latent. Removing the latent thought entirely costs 1.22 points
on GSM8K, the SMALLEST of their four ablations there.

It does not carry over to CE training. Every result is RL with a verifiable answer on a
pretrained 3B or 7B instruct model.

Table 4's collapse is a correctness-variance reading on a fixed prompt set, not a
geometric one. It does not say the thoughts are close in embedding space, only that
their downstream success rates are.
