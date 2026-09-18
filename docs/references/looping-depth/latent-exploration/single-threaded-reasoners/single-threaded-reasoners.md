# LLMs are Single-threaded Reasoners: reading and local cache

Read 2026-09-18.

## Citation (verified)

Junhong Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, Hua Wu (Baidu Inc.).
*LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft
Thinking*. [arXiv 2508.03440v4](https://arxiv.org/abs/2508.03440), submitted
2025-08-05, v4 dated 2025-10-16. Accepted at ICLR 2026
([OpenReview forum ASLuOoP78o](https://openreview.net/forum?id=ASLuOoP78o)).

**Venue note.** The v4 PDF is stale on this point. It prints "Preprint" on every page
and the arXiv comment field still reads "11 pages, 6 figures, working in progress". The
ICLR 2026 acceptance comes from OpenReview, not from the cached PDF.

Read in full: abstract through Section 6. The RL-foundations material after Section 6.1
was skimmed.

## Local cache

- [PDF](../../../../../ignore/papers/2508.03440v4-single-threaded-reasoners-soft-thinking.pdf)
- [Extracted text](../../../../../ignore/papers/2508.03440v4-single-threaded-reasoners-soft-thinking.txt)
- 4,223,721 bytes, 16 pages, 1,127 text lines.
- SHA256: `2d84e6e90be39a82959a3564eab206eda1aeadfa05bb7ae7dbd205ac71212da3`.

## What it actually does

Soft Thinking replaces the sampled token with the whole output distribution: the next
input embedding is the probability-weighted mean of vocabulary embeddings. The claim in
the literature is that this keeps several candidates alive.

The paper first measures that vanilla Soft Thinking is WORSE than ordinary sampled
decoding on three 32B reasoning models, then finds out why, then fixes it.

The diagnosis rests on three probes.

1. Three forward passes at the same step: one with the soft token, one with only its
   top-1 token, one with only its second token. The Jensen-Shannon divergence
   JS(P_soft, P_top1) concentrates at 0; JS(P_soft, P_2nd) is near its maximum.
2. Logit Lens at a hand-balanced soft token (0.6 on the first candidate, 0.4 on the
   second). Both candidates' paths are present for the first two or three layers, then
   the first token's path rises to 1.0 while the second decays. The forward pass is a
   PRUNER.
3. ROUGE-L of the top-1-per-step trace against a greedy trace. Soft Thinking scores far
   higher than sampled decoding, so it is behaving greedily.

They call the result the Greedy Pitfall: the soft token feeds back the most confident
path, which reinforces it.

The fix is to put randomness back into the soft token while keeping it soft. Two
candidates are tried:

    Dirichlet(gamma * p)      cannot trade randomness against softness
    Gumbel-Softmax(p, tau)    can; satisfies Luce's choice axiom

With Gumbel noise on log p and a softmax at temperature tau, the resulting vector stays
on the simplex, is unbiased for the underlying categorical, and does not collapse to
one-hot.

## The key numbers

Average over eight benchmarks (AIME24, AIME25, MATH500, AMC23, GPQA-Diamond, HumanEval,
MBPP, LiveCodeBench):

    model                       token sample   soft vanilla   soft Dirichlet   soft Gumbel
    DeepSeek-R1-Distill-Qwen-32B  78.50         72.13          78.36            79.55
    QwQ-32B                       82.35         80.06          81.39            83.04
    Skywork-OR1-32B               82.99         79.21          83.12            83.41

Vanilla Soft Thinking loses to sampled decoding on all three models. Only the
Gumbel-Softmax variant beats it on all three. Defaults: Dirichlet gamma 4.0, Gumbel
tau 0.5, both from a sweep.

The softness-versus-randomness analysis is the mechanism: at small gamma the Dirichlet
sample is random but nearly one-hot; at large gamma it is soft but nearly the original
distribution. Gumbel tau moves softness while keeping JS divergence high.

## Where it sits in MORPH's 2x2

It is the transition from deterministic/1 to stochastic/1, and it is the best-argued
case in this batch that the transition is worth making. It also documents the failure
mode on the deterministic side: a model that averages candidates behaves as if it had
picked the biggest one.

## What in MORPH already tested this

The greedy collapse is what the Thought Register measured on the K-cell side: four cells
that were seeded apart ended at cosine 0.94. Their forward pass prunes a mixture within
one pass; our loop prunes a set of cells across passes. The mechanism is not the same
but the reading is: a shared map applied to a mixture returns something close to its
dominant component.

The stochastic side has been run and failed for a different reason. LCTUL's flow thinker
is stochastic/1 and is CONTEXT-BLIND: removing the past changes the flow loss by 0.0036,
about 1 % (`lctul-thinker-is-context-blind`). Their fix works because the distribution p
being perturbed is already a good conditional. Ours is not.

Gumbel machinery already exists in the tree. `tul.recur_gate_tau` and
`tul.recur_gate_noise` carry a Gumbel-relaxed gate, and `tul.vq_codes` carries the
straight-through path that a discrete relaxation would need.

## What a MORPH arm implementing it would need

The MORPH analogue of a soft token is the discrete-code path, not the continuous cell.
`tul.code_discrete` already makes the code N symbols from a cosine codebook with a
masked denoiser, and `tul.vq_codes` already makes the slot's thought K symbols through a
straight-through quantiser. A Gumbel relaxation would replace the hard nearest-neighbour
snap in that quantiser with `softmax((g + log p) / tau)` over the codebook rows.

What does not exist: a temperature knob on the VQ assignment. `tul.code_target_tau` and
`tul.code_grade_tau` are temperatures on OTHER softmaxes (an InfoNCE and the grader), not
on the code assignment. A new key would be needed, and `KNOWN_TUL_KEYS` refuses unknown
keys by design, so it has to be added deliberately.

What is not needed: any change to the continuous slot loop. Adding Gaussian noise to a
continuous cell is the GRAM and Parallel-TTS move, not this one. This paper's claim is
specifically about a mixture over a DISCRETE alphabet.

## The instrument it implies

Their JS probe, transposed. At each pass, run three readouts off the cell: the cell as
is, the cell replaced by its single nearest codebook entry, and the cell replaced by its
second nearest. If the model's next-pass behaviour under the full cell matches the
top-1 replacement and not the second, the cell is single-threaded in their exact sense.
That probe is cheap, needs no retraining, and works on any existing VQ checkpoint.

The second instrument is the trace-similarity one: decode greedily from the cell at each
pass and measure ROUGE-L against the model's own greedy continuation. A high score means
the cell is carrying the greedy path and nothing else.

## What it does NOT establish for us

It does not establish that stochasticity helps a model that has not already learned a
good conditional. Every gain in the paper is on a 32B model whose output distribution is
a strong predictor; the noise only stops it from over-committing.

It does not establish anything about DEPTH. The reasoning length is whatever the model
generates; no result separates one thinking step from many.

It does not measure K parallel streams. Everything is one trajectory with a perturbed
step.

It does not transfer the mechanism to a continuous cell. The Luce-axiom argument for
Gumbel holds for a categorical distribution over a vocabulary. MORPH's slot cell is not
a categorical unless the VQ path is on.
