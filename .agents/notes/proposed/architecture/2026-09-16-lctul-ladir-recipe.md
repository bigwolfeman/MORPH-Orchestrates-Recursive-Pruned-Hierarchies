# Agent Note: LCTUL follows LaDiR's schedule — the code's decoder is frozen for good and the thinker trains on its own tape

Status: proposed

Date: 2026-09-16. Prereg: `lab/experiments/planned/2026-09-16-lctul-ladir-chain.md`.
Follows [`2026-09-14-tul-span-code.md`](2026-09-14-tul-span-code.md) (the design whose
phase schedule this replaces) and the thinker-p3 prereg
(`lab/experiments/planned/2026-09-16-tul-code-thinker-p3.md`, the mirror image: a coda
trained on samples). Supersession check: the span-code note's "three phases in one run"
is the schedule this note argues against; that note stays active as the design record
and gets a pointer here; nothing archived.

## Problem

Every LCTUL run trained the coda on a 50/50 mix of noised truth codes and sampled codes
in phase 3 while the encoder and coda kept moving; the thinker's context was always the
truth tape. Measured on 2026-09-16: a coda that trusts codes (frozen at phase 2) decodes
the thinker's sample to word salad (13 nats per token), and a coda trained on samples
(rollout1, thinker-p3) stops reading the cells within a few hundred steps (truth-code CE
0.38 → 5.3). The code itself is a verbatim copy that tolerates almost no error (noise
0.5, oracle returns the sentence word for word).

LaDiR (arXiv 2510.04573), re-read at Wolfe's request, does three things differently:
the VAE (encoder + frozen pretrained decoder) is trained first and alone under heavy
latent noise (k = 3) and input-token substitution (p = 0.3); the decoder never trains
again and never sees a generated latent in training; the reasoning model's second stage
conditions it on its OWN generated latents for the earlier blocks (its exposure-bias fix
is on the thinker, not the decoder). Its "w/o stage 2" ablation loses 14 points on MATH.

## Proposal

Port the schedule stage for stage as a chain of three runner arms: `tul_code_vae`
(E + coda, noise 3.0 renormed, token-state dropout 0.3, no thinker, 10k), `tul_code_ladir_tf`
(everything but the thinker frozen, flow loss on the truth tape, 30k at batch 12) and
`tul_code_ladir_ro` (same freeze, the thinker's context is its own parallel 8-step
sampled tape, `tul.code_tape_rollout_p 1.0`, 10k). The new knob is the only code change;
`code_rollout_p` (samples to the coda) stays 0 on this line.

## Alternatives considered

- **Wolfe's first framing, coda frozen ~80 % then unfrozen on samples only**
  (`tul_code_thinker_p3`, running): kept as the mirror-image control; on its first 5k
  resumed steps the coda dropped the cells. LaDiR never unfreezes the decoder.
- **Freeze the coda inside one run at a phase switch** (a phase-dependent `train_only`):
  needs the optimizer rebuilt at the switch; the runner chain with
  `resume_fresh_optimizer` does the same with no trainer change.
- **Sequential tape rollout (the paper's exact form)**: S × k thinker passes per step
  (up to 512 at max_slots 64); the parallel draw conditioned on the truth tape gives the
  thinker noisy own-samples as context at k passes and is v1; the sequential form is the
  follow-on if stage 3 moves.
- **Noise 3.0 without renorm (the paper's z + η)**: the coda would then see truth cells
  at RMS 3.16 and samples at 1; the 2026-09-15 norm-flag finding says match them, so the
  noisy truth cell is renormed to RMS 1 like the sample.

## Acceptance criteria

The prereg's P-L1 (the noisy code is readable and not a copy) and P-L3 (the frozen
coda's sampled CE falls by ≥ 1.0 nats through stage 2) on one seed move the note to
`implemented/` (the chain becomes the recipe and the span-code note's phase schedule is
amended); P-L1 holding with P-L3 failing moves it to `rejected/` with the reading that
the code's definition, not the schedule, is the lever.

## Risks

- Noise 3.0 on a unit-RMS renormed code is a signal-to-noise ratio near 1/9; the coda
  may drop the cells in stage 1 (P-L1's low end) and the whole chain reads nothing.
- The frozen token path is 10k steps old for the rest of the chain; every CE is far
  from the ruler's and only within-chain comparisons are meaningful.
- One seed per stage; the chain's later stages inherit any stage-1 accident.
