# Planned: hard gamma-freeze discriminator on the A2 paid axis

Status: planned
Date frozen: 2026-09-02 (evening, after the gamma-EMA validation resolved 2/3 DETONATED at draw 2)

## Question

Does the paid-axis (tul_a2) detonation stop when the ternary scale gamma is
frozen at its init value for the whole run?

## Context

- Ternary-off draws: 3/3 healthy (p~0.024 vs the 0.71 base rate). Ternary QAT is
  the trigger surface.
- gamma-EMA at beta=0.99: draws 1 and 2 both DETONATED (7.24e12 / 6.15e9), so
  P-E1 of 2026-09-02-gamma-ema-paid-validation resolves FALSE and the pipeline
  self-stopped before the 20k.
- The lag math: with beta=0.99 gamma still moves ~33% of the way toward a
  sustained drifted mean over 40 steps. The EMA smears the vault, it does not
  stop it. beta=1.0 in the shipped code is a true freeze (update is a no-op),
  so this discriminator needs zero code change.

## Hypothesis

The cusp-vault contagion theory (gamma motion re-thresholds untouched
coordinates) is the trigger. A true freeze removes all gamma motion, so the
freeze arms survive the way ternary-off did.

## Predictions (frozen before launch)

- P-F1: <=1 of 3 freeze draws detonate (probe max preclip/total over steps
  200..2500 > 1e4). Confidence: 40%. The EMA failure is evidence against the
  contagion theory; the lag argument keeps it alive.
- P-F2: healthy freeze draws reach val CE at 2500 within 0.15 nats of clean
  A2's val@2500 (~4.30). Confidence: 60%. Frozen gamma under-scales effective
  weights as shadow weights grow; some CE cost is expected but small at 2500
  steps.

## Method

3 draws, config tul_a2 + training.ternary_scale_ema_beta=1.0, panel flags,
2500 steps, ckpt_every=0, probe verdict identical to the EMA prereg. Draw-1
gates: "TERNARY SCALE EMA ON: beta=1" and "Ternary QAT ON" prints. NO auto-20k
either way — after this discriminator the mechanism verdict and the 20k recipe
are a human-review decision (a frozen-at-init gamma over 20k steps has a real
CE risk; freeze-after-warmup would be the production variant).

## Interpretation table

- Freeze survives (P-F1 true): contagion theory holds; the EMA was
  mechanistically inadequate (lag), not wrong. Production fix candidates:
  freeze-after-warmup or beta>=0.999.
- Freeze detonates (P-F1 false): gamma motion exonerated; trigger is shadow
  weights crossing the 0.5*gamma cusp. Candidates: threshold hysteresis /
  dead-zone widening, or the frozen A2c spectral-cap arm (caveat: the cap
  killed depth-earning in the 2026-08 campaign, which would moot the 20k
  readout).

## Alternatives considered

- Go straight to A2c (spectral_project_cap 1.5): rejected for now — the winner
  recipe removed the cap because it killed depth-earning; running it first
  would spend GPU on an arm whose success wrecks the measurement we want.
- beta=0.999 draws: weaker discriminator than a hard freeze (still lagged, just
  slower); freeze separates the hypotheses cleanly.
