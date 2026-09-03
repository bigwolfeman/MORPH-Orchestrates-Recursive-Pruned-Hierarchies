# Planned: hard gamma-freeze discriminator on the A2 paid axis

Status: failure
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

## Results (2026-09-02, runs tul-a2-frz1/2/3, beta=1.0, 2500 steps each)

Draw-1 gates passed ("TERNARY SCALE EMA ON: beta=1.0", "Ternary QAT ON",
44 modules, 127.8M ternary params). Verdicts from probe.jsonl, max
preclip/total over steps >= 200:

| draw | verdict | max preclip/total | first >1e4 | steps scored | exit |
|---|---|---|---|---|---|
| frz1 | DETONATED | 2.91e15 (step 2015) | step 776 | 200..2040 | 4 (div-guard) |
| frz2 | DETONATED | 1.67e10 (step 1618) | step 201 | 200..2000 | 137 (Wolfe killed it for the UPS at step 2000; verdict was set at step 201) |
| frz3 | DETONATED | 1.01e11 (step 1911) | step 329 | 200..2040 | 4 (div-guard) |

Periodic VAL CE (steps 250..2000), all three draws:

| draw | 250 | 500 | 750 | 1000 | 1250 | 1500 | 1750 | 2000 |
|---|---|---|---|---|---|---|---|---|
| frz1 | 7.48 | 7.67 | 7.41 | 7.33 | 7.37 | 7.50 | 7.35 | 7.52 |
| frz2 | 7.47 | 7.70 | 7.48 | 7.44 | 7.51 | 7.66 | 7.66 | (killed) |
| frz3 | 7.17 | 7.65 | 7.40 | 7.43 | 7.48 | 7.64 | 7.57 | 7.70 |
| clean A2 | 6.85 | 6.37 | 5.76 | 5.39 | 5.23 | 5.23 | 5.05 | 5.18 |

- **P-F1 (40%): FALSE.** 3 detonations / 3.
- **P-F2 (60%): UNMEASURABLE.** No healthy draw. (Reference correction: the
  prereg wrote "~4.30" for clean A2's val@2500; the [VAL 2500] line of
  tul-a2/run.log reads 4.6776. The 4.30 was a mis-transcription of a
  different instrument. It changes nothing here.)

The shape is NOT the paid-axis detonation. Every unfixed and EMA detonation
had a spike train from step ~200-330 on a run that was learning. The freeze
draws never learned: val CE is flat at 7.2-7.7 from the first eval, and
the gradient blowup arrives late on a dead run. Forensics:
$Q/tul-a2-frz{1,2,3}/{run.log,probe.jsonl}; DIVERGED_step_2040.pt under
checkpoints/morph/tul-a2-frz{1,3}/.

## Verdict

**FAILURE, and the discriminator did not discriminate.** The interpretation
table assumed a frozen gamma would at least train. It does not. A gamma
pinned to step-0 mean|W| bounds every effective weight at +-gamma_0 while
the shadow weights grow; once |w| > 0.5*gamma_0 for most coordinates the
codebook is a sign matrix at a fixed tiny scale, and the network cannot grow
its effective norm at all. (Reading, consistent with the flat 7.4 near the
unigram floor; not separately measured.) A run that never learns and blows
up late says nothing about whether gamma contagion triggers a LEARNING run's
detonation, so the "gamma exonerated" branch of the table is NOT earned.

What IS established, with the EMA result: the live rescale of gamma is
load-bearing for learning (beta=0.99 costs 0.30 nats at 2500; beta=1.0
costs everything). Any production fix must keep gamma live or nearly so.

What the method could not distinguish: contagion-through-gamma vs
cusp-crossing-of-shadow-weights, because the only lever tried removes the
codebook's adaptivity along with the contagion path.

## Updated hypothesis

The paid-axis detonation is a property of the winner recipe (notul-20k,
R1, A2, A2s and the EMA draws share it; ~70% per draw), on runs that ARE
learning, with ternary as the trigger surface. The next instruments do not
touch gamma at all: the A2 future-leak probe
(`2026-09-02-a2-future-leak-probe.md`) rules a learned leak in or out, and
the core-Jacobian ladder (`2026-09-02-a2-core-jacobian-ladder.md`) asks
whether the healthy paid map is expansive (edge-of-stability: an optimizer
lever is next) or contractive (a discrete trigger: hysteresis on the code
assignment is next). A freeze-AFTER-warmup discriminator (freeze gamma on a
healthy run at ~step 1000, where the codebook has formed) would be the
honest version of this experiment if the gamma question is still open
after those two. Dense-then-ternary warmup is not a candidate (Wolfe,
2026-09-02).
