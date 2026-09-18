# Latent Thought Flow (LTF): reading and local cache

Read 2026-09-18.

## Citation (verified)

Xiandong Zou, Jing Huang, Jianshu Li, Pan Zhou. *Latent Thought Flow: Efficient Latent
Reasoning in Large Language Models*.
[arXiv 2606.16222v1](https://arxiv.org/abs/2606.16222), submitted 2026-06-15.
Affiliations: Singapore Management University and Ant Group. The PDF says "Preprint";
no venue is claimed. The brief's "~June 2026" date and its GFlowNet description are both
correct.

Read in full: abstract through Section 5 and every table in the main body. Appendix C
was not read.

## Local cache

- [PDF](../../../../../ignore/papers/2606.16222v1-latent-thought-flow.pdf)
- [Extracted text](../../../../../ignore/papers/2606.16222v1-latent-thought-flow.txt)
- 662,555 bytes, 18 pages, 1,045 text lines.
- SHA256: `1367566af1020293ba697082a3bb764cd84105d153252a04251a9ff430a42dca`.

## What it actually does

Every other latent-reasoning method here learns ONE path, or samples paths without
saying what distribution they should follow. LTF specifies the distribution.

A trajectory is a variable-length sequence of continuous thoughts plus a stop decision:

    z_{t+1} ~ N(mu_phi(s_t), diag(sigma^2_phi(s_t)))      Gaussian latent policy
    pi_stop(s_t) = p(<eosr> | s_t)                        adaptive termination

The target distribution is reward-proportional:

    R(tau) = V(tau) * exp(-lambda_c * C(tau))
    V(tau) = verifier(y, y_hat) + exp((1/|y|) log p(y | x, tau))
    C(tau) = T                                            the number of latent steps
    p*(tau | x, y) proportional to R(tau)

A continuous GFlowNet learns a sampler matching p*. The neat trick is that every prefix
is allowed to terminate, so the flow at a prefix is computable analytically from its
immediate-stop reward divided by its stop probability, and no separate flow network is
needed. Sub-Trajectory Balance residuals enforce consistency over all prefix pairs.

Two refinements. Entropy-Weighted SubTB reweights residuals by the length-normalised
differential entropy of the subtrajectory, putting more supervision where the sampler
spreads mass. A reference-prior regulariser anchors early exploration to a branch trained
on teacher rationales, then anneals.

Only a LoRA (rank 128) and the latent head train. The backbone is frozen.

## The key numbers

Fine-tuning, average over GSM8K-Aug, ASDiv-Aug and DU, accuracy / reasoning length:

    backbone        Coconut        CoLaR          ReGuLaR        LTF
    LLaMA-3.2 1B    39.22 / 6.00   47.97 / 5.00   55.56 / 2.04   59.68 / 1.91
    LLaMA-3.2 3B    52.23 / 6.00   58.97 / 4.93   65.12 / 2.10   68.85 / 1.95
    LLaMA-3.1 8B    66.98 / 6.00   70.97 / 5.10   74.40 / 2.13   77.08 / 1.94
    DS-Qwen 1.5B    42.74 / 6.00   49.03 / 5.04   58.06 / 1.98   62.16 / 1.88

Objective ablation, LLaMA 1B, average accuracy / length: GRPO 47.49 / 12.25,
Detailed Balance 55.98 / 7.28, Trajectory Balance 56.80 / 7.51, LTF 59.68 / 1.91. The
GFlowNet objectives beat GRPO by 8 to 9 points AND use half the steps.

Entropy weighting is worth +0.40, +0.67 and +0.95 points at rollout sample sizes
S = 5, 10 and 20. Its benefit grows with S, which is the paper's own evidence that it is
doing something about diversity rather than about optimisation.

Extreme compression, MATH at length 1.00: ReGuLaR 11.98 average, LTF 14.70. On
AQUA-RAT at length 1.00: ReGuLaR 39.47, LTF 43.08.

The number to sit with: the LEARNED average trajectory length is 1.88 to 1.95 steps, and
on two of the three fine-tuning datasets it is 1.17 to 1.24. A sampler that was free to
choose depth, under a reward that pays for accuracy, chose roughly ONE step.

## Where it sits in MORPH's 2x2

Stochastic, one stream, with the DEPTH itself learned. Its distinctive contribution is
not width but a principled way to allocate probability across depths.

## What in MORPH already tested this

The stochastic/1 cell is LCTUL, and it failed in a way LTF's frame explains. The LCTUL
flow thinker is affine (Gaussian-conditional) and context-blind: the past is worth 0.0036
of the flow loss, about 1 %. LTF's sampler is trained against an answer reward, not
against a denoising objective, so its conditional cannot be ignored.

The depth question has been answered locally in LTF's own terms. On MORPH the depth-1
control sits within 0.002 nats of the depth-6 ruler on strict geometry
(`norecur` arm, `per-pass-targets-met-in-one-step`), and matched-compute depth-1 plain is
0.25 nats AHEAD of spandec-mask. A reward that priced compute would pick depth 1 on our
corpus too, which is exactly what LTF's sampler does on theirs.

The one-step optimum is the parent hypothesis of this session, and LTF is the cleanest
external instance of it: nothing about a reward-proportional sampler forces depth, so if
the task does not need depth the sampler discovers that and stops.

## What a MORPH arm implementing it would need

Exists: `tul.gate` with `gate_k_max`, `gate_lambda`, `gate_ponder_lambda`,
`gate_stop_head`, `gate_drives_depth`, `gate_truncate_p` and `gate_scheduled_sampling`,
which is a learned halting head over the slot loop. `tul.slot_depth_fixed` and
`tul.slot_mean_depth` set the sampled-depth regime.

Does not exist:

- Any flow objective. Sub-Trajectory Balance needs a per-prefix reward and a stop
  probability; the gate has the stop probability and there is no reward.
- A reward at all. MORPH trains by CE. The natural language-modelling reward is negative
  per-token CE of the true continuation from a prefix, which makes the balance condition
  computable but also makes the GFlowNet machinery redundant, because that reward is
  differentiable and can be descended directly.
- A learned per-step Gaussian on the slot state. Same gap as GRAM.

The realistic port is NOT a GFlowNet. It is the measurement: give the slot loop a
per-prefix stop reward equal to the CE of stopping at pass t minus a compute penalty, and
ask which t maximises it per span. That is a read, not a training change, and it answers
the same question LTF's sampler answers by training.

Caution for anyone who reaches for `tul.gate`: Wolfe's standing rule is no PonderNet
(`loop-flatness-is-not-a-task-verdict`). A halting arm needs his direction first.

## The instrument it implies

Per-span optimal stopping depth. For each span, read the coda's CE at every pass, subtract
a compute cost per pass, and record the argmax. The distribution of that argmax over spans
is the honest depth-demand curve of the corpus. If it piles up at 1, no sampler over depth
can help, and no per-pass objective can either.

The second instrument is entropy per pass of the latent policy, which is what their
entropy weighting reads. MORPH has no latent policy, so the analogue is the per-pass
spread of the cell across dropout or noise draws, which is the Parallel-TTS probe.

## What it does NOT establish for us

It does not establish that a distributional view fixes flatness. Its own sampler picks
depth 1 to 2.

It does not test width. One trajectory at inference by default.

It does not run on next-token CE. All tasks are verifiable math or data-understanding
with an exact-answer check, which is what makes V(tau) computable.

It does not isolate the GFlowNet from the reference prior. The prior is warm-started from
teacher rationales on every reported run where rationales exist, and the ablation for
dropping it is in the appendix, not the main body.
