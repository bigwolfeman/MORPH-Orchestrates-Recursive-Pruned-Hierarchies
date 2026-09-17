# Experiment: LCTUL-D plan arm — a coarse discrete code, scored on generation

Status: planned

Date: 2026-09-16 (frozen before any GPU step of the arm). Follows
`2026-09-16-lctul-d-first-arm.md` (the 72-bit copy: context share 0.02 % at 5k, CE k-curve
inverted +0.16 from k = 1 to 8) and the theorems in `lab/theory/lctul_euler_depth/`.
Wolfe, 2026-09-16: "kill the current run, let's get this up and running", with a metric and
measured controls first.

## Question

With a coarse code (four symbols of 6 bits per span) and the verdict read on GENERATION,
does the model's own sampled plan make the written span semantically closer to the true
span than a foreign plan, does more sampler rounds help, and does it beat the deterministic
slot write of the strict ruler and the plain model on the same cuts?

## Hypothesis

On likelihood a sampled latent of the next span cannot beat a deterministic summary of the
past, so CE is the wrong verdict. On generation, a coarse plan is a few separated modes,
where the theorems let the sampler's rounds act, and committing to one plan can make the
greedy continuation more on-topic than a mean write.

## Metric

`lab/divergence/code_semantic_probe.py` on the SAME 120 validation cuts for every model
(cuts from the `tul_code_d` boundary rule and packer; `--cuts_config` for the plain model):
greedy continuation of the open span, MiniLM cosine to the true span (`cos_true`),
content-word overlap, paired bootstrap over cuts, and distinct-2 per condition as the
diversity guard. Conditions on the arm: OWN@k for k in 1, 2, 4, SHUF (a foreign sample at
k 4), ZERO, ORACLE. Controls (OWN only): the strict ruler `slot-spandec-strict-20k`
step 20000 (kind slot), the plain `norm-match-20k` step 20000 (kind plain), and the flow arm
`tul-code-cfg` step 20000 (kind code, k 1, 2, 4, 8). Cross-model differences are paired
over cuts from the saved `per_cut` arrays.

## Predictions (frozen)

Arm `tul-code-dplan` (`tul_code_dplan.yaml`, 20k steps, batch 6, the panel extras), one seed.

- P-P1 (own plan beats a foreign plan): at 20k, `OWN@4 − SHUF` on cos_true > 0 with the
  95 % interval clear of 0. (Flow arm at 10k: +0.026 [+0.007, +0.045]; thinker-only: 0.)
- P-P2 (rounds pay): `OWN@4 − OWN@1` on cos_true > 0 with the interval clear of 0.
- P-P3 (the bar): `OWN@4` on cos_true within 0.02 of the strict ruler's OWN on the same cuts,
  paired. (A win over the ruler is not predicted.)
- P-P4 (diversity guard): distinct-2 of OWN@4 within 0.05 of the ruler's.
- P-P5 (codes used): `tul/vq_perplexity` >= 16 of 64 at 5k.
- P-P6 (context share): the denoiser ELBO with the past minus without >= 5 % of the with-past
  number at 20k (the 72-bit arm: 0.02 % at 5k).
- P-P7 (CE for the record only): `val/ce_tf` at 20k above the 72-bit arm's (a coarser code
  carries less); no verdict on CE.

## Method

Queue line at the head of `recon_arms.txt` (commit recorded at insertion), the running
`tul-code-d` killed at Wolfe's instruction. Spark: the three controls run once with the
extended probe (kind slot / plain / code); the arm at 10k and 20k with `--k_list 1,2,4
--steps 4`. Pairing across models by a small script over the `per_cut` arrays.

## Results

(to fill)

## Verdict

(to fill)

## Updated hypothesis

(to fill)
