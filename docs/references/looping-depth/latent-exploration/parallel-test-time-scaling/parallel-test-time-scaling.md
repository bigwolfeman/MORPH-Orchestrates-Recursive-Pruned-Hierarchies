# Parallel Test-Time Scaling for Latent Reasoning: reading and local cache

Read 2026-09-18.

## Citation (verified)

Runyang You, Yongqi Li, Meng Liu, Wenjie Wang, Liqiang Nie, Wenjie Li. *Parallel
Test-Time Scaling for Latent Reasoning Models*.
[arXiv 2510.07745v4](https://arxiv.org/abs/2510.07745), submitted 2025-10-09,
v4 dated 2026-04-19. The arXiv comment field and the PDF's first line both read
"Accepted at ACL 2026 Main Conference", which confirms the brief's venue.
Affiliations: Hong Kong Polytechnic, Shandong Jianzhu, USTC, HIT Shenzhen.
Project page: <https://github.com/ModalityDance/LatentTTS>.

Read in full: abstract through Section 8. Appendices B through H were not read.

## Local cache

- [PDF](../../../../../ignore/papers/2510.07745v4-parallel-test-time-scaling-latent.pdf)
- [Extracted text](../../../../../ignore/papers/2510.07745v4-parallel-test-time-scaling-latent.txt)
- 2,854,013 bytes, 18 pages, 1,208 text lines.
- SHA256: `74cf224420ff66b56594061e6c7c7caaa1ec8b8b0721a5dafcd2dfb830b23d22`.

## What it actually does

This is a TEST-TIME method. Nothing in the reasoning model is retrained. It takes
existing latent-reasoning checkpoints (Coconut, CODI, CoLaR, Latent-SFT, Render-of-
Thought), makes them stochastic, and aggregates.

Two sampling strategies, chosen from uncertainty theory rather than convenience:

- MC-dropout: keep dropout active at inference with rate p, after the feed-forward in
  each block. Each pass samples a different weight configuration. This is epistemic
  uncertainty.
- Additive Gaussian Noise: perturb each latent thought with `eps ~ N(0, sigma^2 I)`
  directly. This is aleatoric uncertainty.

One aggregator: LatentRM, a scalar head on the backbone that scores a prefix of latent
thoughts. Its training data comes from Monte Carlo rollouts: for every thought in every
sampled trajectory, roll out M stochastic completions and label the thought with the
fraction that end in the right answer.

The objective is the design decision worth copying. Binary cross-entropy per thought,
the obvious choice for a process reward model, works badly. They use a STEP-WISE
CONTRASTIVE loss instead: at each step t, the N candidate thoughts at that step are put
through one softmax and trained against their labels. The supervision is relative among
concurrent candidates, not absolute per candidate.

At inference the sum of per-step logits ranks a trajectory, feeding best-of-N or beam
search.

## The key numbers

Coverage (equivalently pass@N) rises monotonically with N to 64 on Coconut, CODI and
CoLaR across GSM8K-Test, GSM8K-Hard and MultiArith, with diminishing marginal gain.
MC-dropout beats AGN on coverage at nearly every N. At N = 64 Coconut and CODI reach
nearly equal coverage even though CODI is clearly better at N = 1, so sampling narrows
model gaps.

Larger backbones, MC-dropout (their Table 1), deterministic / cov@8 / cov@16:
Latent-SFT 1B on GSM8K 0.445 / 0.585 / 0.649; RoT-4B on MATH500 0.203 / 0.218 / 0.220;
RoT-2B on GSM8K-Hard 0.086 / 0.091 / 0.092. The gains shrink sharply with backbone
quality and task difficulty.

Diversity is measured as mean pairwise cosine dissimilarity of latent thoughts across
steps. Coverage peaks at MODERATE diversity, not maximum. At high diversity AGN holds up
and MC-dropout falls off sharply. t-SNE shows why: dropout drifts directionally into a
dense contiguous region, AGN disperses isotropically. Dropout wins on hard questions
where the correct region is far from the deterministic point; AGN wins on easy ones
where it is near.

LatentRM ablation at best-of-8 (GSM-Test / GSM-Hard):

    Best-of-8 with LatentRM        35.4 / 7.8
    without contrastive (BCE)      33.5 / 7.4
    without stochastic rollouts    30.7 / 6.0
    untrained random scalar head   28.9 / 5.8
    majority voting                33.6 / 6.1

The untrained head is WORSE than no model at all. The selector has to be learned.

Recommended hyperparameters, no per-dataset tuning needed: GPT-2 style backbones
p in [0.1, 0.3] and sigma in [0.5, 0.7]; Llama-1B style p and sigma both in
[0.01, 0.03].

## Where it sits in MORPH's 2x2

Stochastic, K streams, obtained without changing training at all. It is the cheapest
possible probe of the empty cell, because it asks whether an ALREADY TRAINED
deterministic model has anything to explore over.

## What in MORPH already tested this

Nothing. MORPH has never perturbed a trained slot-loop checkpoint and measured whether
the perturbed trajectories reach better readings than the deterministic one. The
deterministic cell is the only cell we have ever read.

The closest thing is the causal-fit probe (`lab/divergence/slot_z_causal_fit.py`,
2026-09-13), which optimises a z against teacher-sampled continuations and scores it on
the untouched real span. It found +0.252 nats WORSE than the loop's own z: no causal
headroom above the one-pass state BY GRADIENT SEARCH. A noise-sampling probe asks the
same question by a different estimator, and a gradient search that finds nothing is weak
evidence that sampling will find nothing, not proof.

## What a MORPH arm implementing it would need

This is by far the cheapest of the nine arms, because it needs no training run.

Exists: any strict slot-loop checkpoint (`slot-spandec-strict` at 5000 steps is the
standing ruler); the offline probe harness under `lab/divergence/`; dropout is already
in the model.

Needs writing, all offline and all in `lab/divergence/`:

- A probe that runs the slot loop N times per row with either dropout active or an
  additive Gaussian on the cell at each pass, and stores the N exit cells per slot.
- Scoring: run the coda on each of the N cells and take the per-span CE. Oracle over
  stream is `min_n CE_n`; the deterministic reading is the existing one.
- A diversity reading: mean pairwise cosine dissimilarity of the N cells at each pass.

Does not exist: any LatentRM. Building a learned selector is a second, larger step and
their ablation says an unlearned one is worse than nothing. The first experiment should
be ORACLE only. If the oracle does not beat the deterministic reading, no selector can
help, and the arm ends there for one GPU-hour.

## The instrument it implies

Coverage, transposed to a language-modelling readout. Their coverage is "at least one of
N trajectories gets the answer right". Ours is the ORACLE per-span CE over N sampled
cells, against the deterministic cell's CE, on identical rows, with the oracle's own
selection bias measured on a held-out half of the span.

The second instrument is their coverage-versus-diversity curve. Sweep sigma and p,
plot oracle gain against mean pairwise cell dissimilarity, and look for their sweet spot.
A curve that is FLAT in diversity says the cell has nothing to explore over. A curve that
rises then falls says there is a scale at which the exploration is real.

## What it does NOT establish for us

It does not establish that a trained-in stochastic loop is better than a perturbed
deterministic one. It never trains with noise.

It does not establish transfer to next-token CE. Every number is answer accuracy on
arithmetic word problems with a verifiable answer, where pass@N is a natural metric.
Per-token CE has no analogue of "at least one of N is correct" without a scoring rule
that has its own bias.

It does not establish that the gains survive a strong backbone. The RoT-4B rows move
0.203 to 0.220 on MATH500 with 16 samples, and GSM8K-Hard moves 0.139 to 0.143.

It does not say anything about depth. The number of latent thoughts is fixed by each
backbone's own recipe.
