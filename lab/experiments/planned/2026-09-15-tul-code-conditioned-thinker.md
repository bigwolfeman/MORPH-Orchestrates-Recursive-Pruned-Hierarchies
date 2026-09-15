# Experiment: TUL-Code conditioned thinker — CFG and predictability pressure on E

Status: planned
Date: 2026-09-15
Owner: Claude (session f9558148), for Wolfe ("Agreed. Let's do it.")
Design: `.agents/notes/proposed/architecture/2026-09-15-tul-code-conditioned-thinker.md`

## Question

The 20k panel's sampled code is an unconditional draw (residual 1.7–1.9× the code's
variance; worth the top 32 of 1024 code directions; flow loss at its floor by step 4000;
+0.63 nats one-draw against the strict ruler). Two independent causes: the target has
almost no predictable content (a lossless copy), and the sampler under-uses the
conditioning it has (band-0 flow ratio 0.48 against a 0.64 blind floor). Does either fix
make the sample carry the span?

## Hypothesis

H-B1 (sampler): classifier-free guidance (`code_cfg_drop` 0.1, `code_cfg_scale` 2.0) pulls
the draw toward the conditional mode: the residual falls, the k-curve keeps part of its
phase-2 dependence after the rollout, the gap narrows.
H-A1 (target): letting the flow loss reach E at `code_target_lambda` 0.1 moves z toward
what the past can predict (the plan); ce_tf rises, rank falls, the residual falls, the gap
narrows without collapse (`code_rank_abort` 4).

## Predictions (frozen)

Paired numbers on the shared tokens against `slot-spandec-strict-20k` at 20k; the sample
residual and k-curve from the Spark probes (96 rows, 8 draws); reference `tul-code-20k`:
residual 1.65–1.92 (head, rank 32–256), k16−k1 +0.014 at 20k / −1.29 at 10k, gap +0.626,
ce_tf 0.35 at the trained statistic, rate 30k / 24k tok/s (phase 2 / 3).

- P-1 (B1). `tul-code-cfg` sample residual over z variance at 20k ≤ 1.4 in the rank-128
  head (from 1.83–1.85). 55 %.
- P-2 (B1). `tul-code-cfg` marginal(k=16) − marginal(k=1) at 20k ≤ −0.10. 45 %.
- P-3 (B1). `tul-code-cfg` one-draw `ce_k1` − ruler at 20k ≤ +0.45, CI excluding +0.55.
  45 %.
- P-4 (B1). Guidance sweep on the 20k checkpoint (w ∈ {1, 1.5, 2, 3}): one-draw CE is
  not monotone in w — an interior optimum exists. 60 %.
- P-5 (A1). `tul-code-jepa` reaches 20k without the rank abort (val/code_eff_rank ≥ 4
  throughout). 60 %.
- P-6 (A1). `tul-code-jepa` ce_tf at 20k ≥ 1.0 (the code stops being a copy) AND its
  one-draw `ce_k1` − ruler ≤ +0.45. 40 %.
- P-7 (A1). `tul-code-jepa` sample residual at 20k ≤ 1.4 in the rank-128 head. 50 %.
- P-8. Both arms healthy (no tripwire); `tul-code-cfg` phase-3 rate ≥ 15k tok/s (two
  thinker passes per Euler step at rollout). 80 %.

## Binding

- P-1 and P-3 hold → the sampler was the lever; combine with A1 and k 32 next.
- P-6 and P-7 hold → the target was the lever; sweep λ (0.03 / 0.3).
- Neither residual moves (P-1 and P-7 fail) → the flow parameterisation itself is the
  question (discrete plan code, A2), not its conditioning.
- P-5 fails → λ 0.1 is too strong for the CE anchor; λ 0.03 before any other change.

## Method

Two 20k arms on the panel recipe (`tul_code` lineage: seq 1024, batch 6, warmup 1000,
phases at 2000 / 10000, `code_noise_renorm: true` on both), fresh runs from step 0, one
factor each against `tul-code-20k` plus the renorm (the renorm's own effect is the
`tul-code-renorm-r10k` arm, running first):

- `tul-code-cfg` (`morph/configs/tul_code_cfg.yaml`): `code_cfg_drop` 0.1,
  `code_cfg_scale` 2.0 (rollout and eval samples guided).
- `tul-code-jepa` (`morph/configs/tul_code_jepa.yaml`): `code_target_lambda` 0.1,
  `code_rank_abort` 4.

Runner sweeps at 5k/10k/15k/20k; Spark k-sweep at each checkpoint, subspace probe and
span samples at 20k; `paired_vs_ruler.py` against the ruler twin; for P-4 the k-sweep
re-run with `tul.code_cfg_scale` overrides on the 20k cfg checkpoint.

## Not verified before launch

- No GPU smoke beyond the runner's 12-step one; the guided rollout's memory (two thinker
  passes) is unmeasured at batch 6.
- One seed each.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
