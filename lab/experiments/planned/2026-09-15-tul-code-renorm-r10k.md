# Experiment: TUL-Code phase 3 with the truth cell renormed (resumed from tul-code-20k step 10000)

Status: planned
Date: 2026-09-15
Owner: Claude (session f9558148), for Wolfe

## Question

The 20k panel's phase-3 coda learned a code's RMS as the "this is a sample" flag: a truth
cell reached it at RMS sqrt(1 + 0.5²) = 1.118 (z + noise, no renorm) and a sampled cell at
RMS 1.0, and eval fed bare z (`code_norm_flag_tul-code-20k_20000`: bare z 1.25 nats, the
trained statistic 0.35; a sample scaled to the truth norm +0.01). Does removing that flag
in training (`tul.code_noise_renorm: true`, truth and sample both at RMS 1) change what the
coda does with a SAMPLED code — the sampled gap against the ruler and the k-curve — or is
the coda's discount of samples content-based and unchanged?

## Hypothesis

H1 (content-based discount, expected): the coda separates samples from truth by content,
so the renorm changes the encoder-mode reading only; the sampled gap and the flat k-curve
stay.
H2 (norm-based discount): with the norm flag gone the coda must read samples by content,
so better samples (more Euler steps) are read and the k-curve keeps some of its phase-2
dependence.

## Predictions (frozen)

Paired numbers on the shared tokens against `slot-spandec-strict-20k` at 20k; the k-curve
from `code_marginal_sweep.py` (96 rows, 8 draws) on the Spark.

- P-1. `tul-code-renorm-r10k` one-draw `ce_k1` − ruler at 20k is within ±0.06 of
  `tul-code-20k`'s +0.626 (H1). 60 %. (H2 reads ≥ 0.15 better, CI excluding −0.06.)
- P-2. marginal(k=16) − marginal(k=1) at 20k: |Δ| ≤ 0.02 (H1). 65 %. (H2: ≤ −0.10.)
- P-3. `ce_tf` (encoder mode, now at the trained statistic in both arms) at 20k ≤ 0.60 on
  the renorm arm (the old coda read a renormed noisy truth cell at 0.76 without having
  trained on it). 70 %.
- P-4. Healthy to 20k, no tripwire; rate ≥ 20k tok/s in phase 3 (tul-code-20k: 24k).
  85 %.

## Method

Resume `tul-code-20k`'s `step_10000.pt` (the end of phase 2, the parent's optimizer, RNG
and data position restored: a faithful resume) with `--config-name tul_code_renorm`
(`tul.code_noise_renorm: true`, everything else `tul_code`) for steps 10001–20000 —
phase 3 from the first resumed step, `code_rollout_p` 0.5 as the parent. ONE factor
against the parent: the RMS of the truth cell in phase 3. Runner line at the commit that
carries the knob; sweeps at 15000 and 20000; Spark k-sweep at both, subspace probe and
span samples at 20000; `paired_vs_ruler.py` against the ruler twin's 20000 sweep.

Note on the parent's own numbers: the parent's `ce_tf` and `ce_0` readings before this
commit were taken at bare z (the wrong statistic); the parent is re-read under the new
eval rule where a comparison needs it.

## Not verified before launch

- The resume path with a changed `tul.*` knob: no test covers a faithful resume whose
  config differs from the checkpoint's; the runner's 12-step smoke runs the config fresh.
- One seed.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
