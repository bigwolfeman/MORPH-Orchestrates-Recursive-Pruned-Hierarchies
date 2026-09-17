# Experiment: LCTUL trained the LaDiR way — VAE first, decoder frozen for good, thinker on its own tape

Status: failure
Date: 2026-09-16
Owner: Claude (session f9558148), for Wolfe ("We should read the lidar paper again" /
"OHHH WE NEED THAT")

## Question

LaDiR (arXiv 2510.04573) trains its latent's encoder and decoder first and alone, with
heavy latent noise (k = 3) and input-token substitution (p = 0.3) so the decoder can read
a NOISY latent, then freezes both for good. Its reasoning model then trains in two stages:
teacher-forced on oracle latents, then conditioned on its OWN generated latents for the
earlier blocks (the flow target stays the oracle). The decoder never reads a generated
latent during training and decodes them at inference. Every LCTUL run so far did the
opposite on both counts: noise 0.5 (a razor-thin verbatim code), the coda trained on
samples in phase 3, and a thinker that always saw the truth tape. On a coda that trusts
codes our samples decode to word salad (13 nats); on a coda trained on samples the cells
are ignored. Does the LaDiR recipe, ported stage for stage, give a frozen coda that can
read the thinker's sample?

Three runs chained through the runner, one seed:

1. `tul-code-vae` (`tul_code_vae.yaml`): E + coda on truth codes, noise 3.0 rms-renormed,
   token-state dropout 0.3, no thinker, 10k steps at batch 6.
2. `tul-code-ladir-tf` (`tul_code_ladir_tf.yaml`): resume 1 at 10k, E + prelude +
   embeddings + slot module + CODA frozen, the thinker alone on the flow loss with the
   truth tape as context, 30k steps at batch 12 (to 40k).
3. `tul-code-ladir-ro` (`tul_code_ladir_ro.yaml`): resume 2 at 40k, same freeze, the
   thinker's context is its own parallel 8-step sampled tape on every row
   (`code_tape_rollout_p 1.0`), 10k steps (to 50k).

## Hypothesis

H-L1: a code learned under noise 3.0 is fat and semantic (the coda cannot rely on fine
detail), so a sample near it in the coda's eyes decodes to text about the right thing;
the frozen coda's sampled CE falls through stage 2 and the k-curve gains a slope; stage 3
lowers the rolled (own-tape) CE below the truth-tape-conditioned one.
H-L0: noise 3.0 destroys the code (the coda learns to ignore the cells in stage 1 and
ce_tf sits at the ruler), or the code survives and the thinker's sample is still an
unconditional draw the frozen coda reads as salad.

## Predictions (frozen)

Reference: the parent's ce_tf 0.35 at 20k (noise 0.5); the strict ruler ≈ 4.4 nats; the
thinker-only arm's frozen-coda sampled CE 13.5 → 13.9 over 40k steps, marginal 11.0 →
11.5, flow probe 0.279 at 50k.

- P-L1. Stage 1 at 10k: `val/ce_tf` (truth code, noise 3.0 renormed) between 1.0 and
  3.5 — the coda reads the noisy code and it is not a copy (0.35) nor ignored (4.4). 55 %.
- P-L2. Stage 1 at 10k: `val/code_eff_rank` ≥ 40 (no collapse under the heavy noise). 60 %.
- P-L3. Stage 2: the frozen coda's sampled `val/ce_tokens` FALLS by ≥ 1.0 nats from the
  first resumed val to 40k (the thinker-only arm rose by 0.4). 45 %.
- P-L4. Stage 2 at 40k: 8-draw marginal k16 − k1 ≤ −0.10 on the frozen coda. 35 %.
- P-L5. Stage 2 at 40k: semantic probe OWN − SHUF cosine to the true span > 0 with the CI
  excluding 0 (the coda trusts codes, so this reads the sample). 35 %.
- P-L6. Stage 3 at 50k: `val/ce_k8_rolled` (the own-tape generation regime) is ≤ the
  stage-2 endpoint's rolled CE − 0.10. 40 %.
- P-L7. Stage 2 at 40k: sampled `val/ce_tokens` on the frozen coda ≤ 4.6 (within 0.2 of
  the no-code ruler: the sample is at least not worse than nothing). 25 %.
- P-L8. All three healthy; stage-2 rate ≥ 28k tok/s at batch 12; stage-3 rate ≥ 18k. 75 %.

## Binding

- P-L1 and P-L3 hold → the code definition was the lever (H-L1): raise the frozen-coda
  budget (stage 2 longer, batch 24) and sweep noise (2.0, 4.0) on stage 1.
- P-L1 holds and P-L3 fails → the noisy code is readable but the thinker's sample is
  still unconditional: the past does not determine this code either; next is a code
  with less capacity by construction (fewer floats, or a discrete code).
- P-L1 fails at the low end (ce_tf ≈ ruler) → noise 3.0 killed the code at this scale;
  re-run stage 1 at noise 1.5 before anything else.

## Method

Queue order after `tul-code-thinker-p3`: vae → ladir-tf → ladir-ro (each line's resume
path exists when the runner reaches it). Runner sweeps at 5k/10k (vae), 20k/30k/40k
(tf), 45k/50k (ro). Spark: marginal at every checkpoint; subspace, flow, span samples and
the semantic probe at 40k (tf) and 50k (ro); `val/ce_k8_rolled` from the trainer's val.
Pairing: the parent and the strict ruler at 20k on the same rows (context, not the verdict:
the token path here is frozen at 10k steps).

## Not verified before launch

- Stage 1 and stage 3 ran 12-step Spark smokes (exit codes in the runner log gate each
  stage); stage 2 is stage 3's config minus the tape knob, not smoked separately.
- The tape rollout draws every slot in one parallel run conditioned on the truth tape;
  LaDiR's sequential draw would cost S × k passes. Recorded in the spec.
- LaDiR's k = 3 is on a KL-regularised latent; ours is on a unit-RMS code with renorm, so
  the effective signal-to-noise is not identical.
- One seed per stage.

## Results

The three stages ran 2026-09-17 00:33 → 08:19 on the 5090 (wandb `za0cyp3w`, `cpi2ru4m`,
`gromu6o1`; commit cbc696f), all healthy, no spikes. Artifacts:
`lab/experiments/results/2026-09-16-lctul-ladir/` (marginal sweeps at 20k/30k/40k/45k/50k,
flow, subspace, span samples and the semantic probe at 40k and 50k, the VAE stage's sweeps).

| Prediction | Reading | Holds |
| --- | --- | --- |
| P-L1 stage 1 `val/ce_tf` in [1.0, 3.5] | 1.34 at 9750 (the coda reads the noise-3.0 code; not a copy, not ignored) | yes |
| P-L2 stage 1 `val/code_eff_rank` ≥ 40 | 57.0 at 10k (61.8 at 40k, 63.0 at 50k) | yes |
| P-L3 stage 2 sampled `val/ce_tokens` falls ≥ 1.0 | 7.69 at the resume → 9.42 one val later → 9.90 at 39750: ROSE 2.2 | no |
| P-L4 stage 2 marginal k16 − k1 ≤ −0.10 | +0.319 [+0.225, +0.428] at 40k (k1 7.66, k16 7.98; encoder 1.33) | no |
| P-L5 stage 2 OWN − SHUF > 0, CI clear | −0.006 [−0.024, +0.011] at 40k (OWN@8 0.111, SHUF 0.116, ZERO 0.132, ORACLE 0.637) | no |
| P-L6 stage 3 `val/ce_k8_rolled` ≤ stage-2 endpoint − 0.10 | the key was never logged by the trainer (a Method error: no such val key exists); the nearest instrument is stage 3's own sampled `val/ce_tokens`, whose context IS the rolled tape: 10.04 at the resume → 10.35 at 49750 (WORSE by 0.31); Spark marginal k8 8.24 (tf@40k) → 8.50 (45k) → 8.53 (50k) | no |
| P-L7 stage 2 sampled `val/ce_tokens` ≤ 4.6 | 9.90 | no |
| P-L8 all healthy; tf ≥ 28k tok/s, ro ≥ 18k | 34.2k / 70.3k / 50.0k tok/s, exit 0 each | yes |

Stage 3 at 50k on the semantic probe: OWN@8 0.109, SHUF 0.115, ZERO 0.131, ORACLE 0.638;
OWN − SHUF −0.006 [−0.026, +0.013]; OWN − ZERO −0.022 [−0.044, −0.001]. Distinct-2 of the
own-sample continuations is 0.89 (word salad; ZERO reads 0.52, the ruler 0.44).

## Verdict

Failure, on the binding's second branch: P-L1 holds and P-L3 fails. The noise-3.0 code is
fat and readable (a frozen coda decodes the TRUE code at 1.3 nats and the oracle continuation
scores 0.64 cosine, the best oracle of any arm), and the thinker's sample still reads as an
unconditional draw: own sample = foreign sample = worse than no code, on CE (7.7 to 8.0
nats against the ruler's 4.4) and on generation. Stage 3's own-tape conditioning made the
sampled CE worse, not better. What the method could not distinguish: whether a longer stage
2 or a sequential (S × k) rollout would have moved the sample toward the coda's region; the
flow loss (`code_fm_rel`) plateaued, so more steps on the same loss are not the answer.
The binding's prescription (a code with less capacity by construction, discrete) was run in
parallel as `tul-code-d` and `tul-code-dplan` and failed the same way
(`2026-09-16-lctul-d-first-arm.md`, `2026-09-16-lctul-dplan-semantic.md`).

## Updated hypothesis

The decoder side was never the problem. Every reader tried (a coda trusting the code, a
coda trained on samples, a frozen LaDiR coda on a noise-3.0 code) reads the TRUE code well.
The thinker's sample carries nothing about the span on every code tried (1024-d at noise
0.5, 1024-d at noise 3.0, 72-bit discrete, 24-bit discrete) because the past does not
determine the next span's code on web text beyond the 0.40 nats/token cross-span budget.
A next-span latent that is sampled rather than summarised is the wrong shape for this
data; the deterministic strict ruler beats every sampled arm on generation.
