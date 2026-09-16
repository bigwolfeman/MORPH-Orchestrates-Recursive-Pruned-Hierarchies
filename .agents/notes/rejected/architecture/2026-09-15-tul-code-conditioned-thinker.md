# Agent Note: TUL-Code — make the thinker's sample carry the span (conditioning and target)

Status: rejected — both levers measured on the panel: guidance is erased by the discounting coda and predictability pressure leaves the code a copy; the sample stays an unconditional draw

Date: 2026-09-15. Owner: Claude (session f9558148) for Wolfe. Parent design:
[`2026-09-14-tul-span-code.md`](2026-09-14-tul-span-code.md); spec `docs/tul-code-spec.md`;
panel readings `lab/experiments/results/2026-09-14-arc-tul-code-20k/` and
`.../2026-09-14-arc-tul-code/`.

## Problem

The 20k panel measured the sampled code as an unconditional draw: its residual against the
encoder's code has 1.7–1.9× the code's variance in every principal subspace (correlation
0.05–0.15), it is worth the top 32 of 1024 code directions (the register), the thinker's
flow loss stops moving at step 4000 (band ratios 0.48 / 0.33 / 0.23 / 0.21 at 20k, blind
floor 0.64 / 0.93 / 0.93 / 0.65), and one draw costs the coda +0.63 nats against the strict
ruler on 501k paired tokens. The coda's discount of a sample is content-based (a sample at
the truth norm reads +0.01; the norm flag that did exist is fixed by `code_noise_renorm`).
Sampler depth is worth −1.0 to −1.3 nats under the 8-draw marginal while the coda trusts
codes (phase 2, two seeds) and 0.00 once the rollout teaches it not to.

Two independent causes, each sufficient:

1. **The target has almost no predictable content.** E's code is a lossless copy of span
   s+1 (ce_tf 0.35). Most of a sentence is not determined by its past, so p(z | past) is
   broad, the L2 flow target is the mean of a broad distribution, and a draw from it is
   nearly independent of the specific truth. The coda has nothing to gain from it.
2. **The sampler under-uses the conditioning it has.** Band 0 (t near the noise end) reads
   0.48 against a blind floor of 0.64, so the field does carry some of the past at the
   start of the path, and more Euler steps help in phase 2; bands 2–3 sit far above what a
   blind rank-60 field would reach, so the late path is poor. Nothing sharpens the sample
   toward the conditional mode.

```
GOAL: a sampled code the coda can use (corr(sample, truth) >> 0.1)
├── A. the TARGET: what E is allowed to put in z
│   ├── A1 predictability pressure on E  (flow gradient reaches E at weight λ; CE anchors content)
│   ├── A2 discrete plan code (VQ, prior = classifier, best-of-K natural)   — large change
│   └── A3 explicit split z = z_pred + z_res                                 — A1 with a seam
├── B. the SAMPLER: how faithful a draw is to the conditioning
│   ├── B1 classifier-free guidance (10 % context dropout in training; guided Euler at w)
│   ├── B2 minibatch-OT noise coupling (nearest of K noise seeds per target)  — straighter paths
│   └── B3 more Euler steps at rollout/eval (k 32)                            — phase 2 says it helps
└── C. the CODA: what it does with a guess                                     — measured: not the lever
```

## Proposal

Two arms, one per cause, 20k steps each on the panel recipe (seq 1024, batch 6, phases at
2000 / 10000, `code_noise_renorm: true`), scored on the phase-2 and phase-3 k-curves, the
sample residual, and the paired gap:

- **B1 `cfg`** — classifier-free guidance. Training: with probability `code_cfg_drop`
  (0.1) a slot's thinker pass sees a null condition: the seed `e` replaced by a learned
  `E_null`, the injection stack zeroed, the clean tape cells masked in the relation.
  Sampling: `v = v_uncond + w · (v_cond − v_uncond)` at `code_cfg_scale` (2.0 first;
  sweepable at eval). Reads: the sample residual should fall below 1.7× and the k-curve
  keep some of its phase-2 dependence after the rollout. Cost: 2× thinker passes per
  sampled slot at rollout/eval. The tree has all three conditioning paths in
  `_tul_code_thinker` (`e`, `inj`, the clean copies via `code_thinker_relation`).
- **A1 `jepa-λ`** — the flow loss gradient reaches E at weight `code_target_lambda`
  (0.1; today it is cut, contract C4). With the coda's CE anchoring content, z settles at
  the part of the span the past can predict plus what the CE still demands: the plan, not
  the copy. Reads: ce_tf rises, code_eff_rank falls, the sample residual falls, the coda's
  own token model carries the words. Collapse guard is the CE term, not the stop-gradient;
  `val/code_eff_rank` and `code_pairwise_cos` are the instruments, with an abort at rank
  < 4. Contract C4 is amended for this arm only.

Then, if either moves the residual: the pair combined, and B3 (k 32) on top.

## Alternatives considered

- **Noise on the truth cell at the sample's residual level** (Wolfe, 2026-09-15). Already
  present at std 0.5; more of it trains the coda to denoise a signal a sample does not carry
  (noisy truth at the sample's residual still correlates with the truth at about 0.6; a
  sample at 0.05–0.15). Rejected as the lever; kept at 0.5.
- **Mixing the truth cell with the thinker's sample** (scheduled sampling): trains the coda
  on the sample's real structure at reduced strength; puts no information into the sample.
  A cheap resume arm if the coda side ever binds again; not first.
- **XM best-of-K rollout** (arXiv 2607.27372): sample K, keep the best by coda likelihood,
  train on it. Selection sharpens the coda's training distribution but the flow's marginal
  is untouched unless the selected seed is regressed (B2 is that idea at the noise end).
  Second wave, after B1.
- **A2 VQ plan code**: the cleanest prior (a classifier) and best-of-K for free, but a new
  head, codebook collapse risks, and a different coda input; too large for a first arm.
- **PonderNet / adaptive depth**: out, per the standing rule.

## Acceptance criteria

- The sample residual over code variance at 20k ≤ 1.2 in the head subspace (from 1.7–1.9).
- marginal(k=16) − marginal(k=1) ≤ −0.10 at 20k (from 0.00 / +0.01).
- One-draw paired gap against the strict ruler at 20k ≤ +0.35 (from +0.63).
- Rate ≥ 2× the ruler's (cfg costs a second thinker pass at rollout).

## Risks

- A1 collapse (z → constant): the CE anchor may not hold at λ 0.1; the abort rule and a λ
  sweep (0.03 / 0.1 / 0.3) are the answer, not a stop-gradient.
- B1 guidance overshoot at w 2.0: samples leave the code manifold; sweep w at eval on one
  checkpoint (no retraining).
- One seed each; MORPH runs decorrelate in 11 steps, so a 0.05-nat read is noise.

## Measured 2026-09-15 (why rejected)

Both levers ran on the panel recipe and both failed their deciding predictions
(`lab/experiments/failures/2026-09-15-tul-code-conditioned-thinker.md`). B1 guidance:
worth 0.6 nats on the phase-2 trusting coda, erased after rollout; residual 1.81–1.84,
gap vs the ruler +0.648, one-draw CE flat in w across [1, 3]. A1 predictability
pressure: the flow gradient reached the encoder (rank 70 → 19 → 31, flow share 0.31 →
0.27, one-step sample 2.5 nats better on the trusting coda at 10k with an inverted
k-curve), and none of it survived rollout: residual 1.84, gap +0.654, sampled val on the
parent's curve to the third decimal, code still a copy (ce_tf 0.32). With the thinker-only
run (40k steps on a fixed target, floor by 10k) this closes conditioning, target motion,
training time and target rank as levers. Kept here because each is a tempting retry; the
open lines are the training rule (Explorative Modeling, `2026-09-15-tul-code-explorative-rollout.md`)
and a code with less capacity by construction (A2 / a bottleneck), which change what the
code IS rather than how it is guessed.
