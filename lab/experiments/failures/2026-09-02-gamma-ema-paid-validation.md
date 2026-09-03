# Planned: gamma-EMA validation — the cusp-vault fix on the paid axis

Status: failure
Date: 2026-09-02 (frozen ~17:00, before any gamma-EMA run)

## Question

Does the ternary-scale slow-EMA (gamma advanced once per optimizer step,
forward reads the buffer) kill the paid-axis detonation while preserving
training quality — clearing the way for the walkover 20k?

## Hypothesis

The M2G capture showed the detonation is the cusp vault compounding under
clipping (spike train from step ~330, no stale-m2 signature), and 3/3
ternary-off draws were clean. The vault is flip CONTAGION through the
shared live scale: a localized drift yanks gamma=mean|W| and re-thresholds
the untouched majority in one step (unit-pinned:
tests/test_ternary_scale_ema.py::test_vault_suppression — contagion flips
drop >5x under the EMA). beta=0.99 (~100-step horizon) should therefore
collapse the detonation rate with ternary ON. Counter-hypothesis: the live
rescale was also load-bearing protection against uniform drift, and the
EMA trades one instability for another (note: with gamma ~frozen the
effective weights are BOUNDED at ±gamma, so the uniform-blowup direction
is intrinsically damped — this is an argument, not a measurement).

## Method

Same 3-draw protocol as the ternary-off experiment: `tul_a2` +
`training.ternary_scale_ema_beta=0.99`, panel flags, 2500 steps/draw,
probe verdict (DETONATED iff max preclip/total over steps 200..2500 >
1e4). Smoke condition in draw 1: the "TERNARY SCALE EMA ON" print present
AND "Ternary QAT ON" still present. Then, per the frozen binding below,
the 20k runs in the same launch chain.

## Predictions (frozen)

- **P-E1.** Detonation count over 3 draws <= 1: 65%.
- **P-E2.** The healthy draws' step-2500 VAL CE is within 0.10 of the
  clean A2 run's 4.87 K1-context... use the comparable instrument: within
  0.10 of clean-A2's train-window VAL at step 2500 band (read from
  tul-a2/run.log at scoring time; the EMA should cost ~nothing): 70%.

## Binding

Detonations <= 1/3 => launch tul-a2-20k with
`+training.ternary_scale_ema_beta=0.99` (a DISCLOSED recipe amendment to
the walkover arm, mechanism-driven; one retry allowed, forensics
archived). Detonations >= 2/3 => the vault fix is insufficient — stop, do
not burn the 20k, escalate to gamma freeze-after-warmup or the A2c cap.
If the 20k completes: a2_depth_sweep 1..8, readout vs tul-20k (3.8461) /
notul-20k (3.4894) / notul-20k token-axis 0.2072, and the earning bar
K1-K6 >= 0.10 at 20k.

## Not verified before run

The EMA under torch.compile at scale (buffer read inside compiled MLPs —
unit tests are eager CPU; the smoke draw is the live check); interaction
with prune/carve (not active in these runs); CE cost of the ~frozen scale
beyond 2500 steps.

## Results (2026-09-02, runs tul-a2-ema1/2/3, beta=0.99, 2500 steps each)

Smoke passed in draw 1: "TERNARY SCALE EMA ON: beta=0.99" and "Ternary QAT
ON" both present. Verdicts from probe.jsonl, max preclip/total over steps
200..2500:

| draw | verdict | max preclip/total | first >1e4 | last VAL | exit |
|---|---|---|---|---|---|
| ema1 | DETONATED | 7.24e12 (step 1194) | step 282 | 7.66 @2000, div-guard abort @2040 | 4 |
| ema2 | DETONATED | 6.15e9 (step 337) | step 200 | 7.52 @2000, div-guard abort @2040 | 4 |
| ema3 | HEALTHY | 32.7 | never | 4.9815 final @2500 | 0 |

Onset in both detonating draws sits in the same step 200-330 window as
the unfixed A2/A2s/R1 detonations. The EMA changed neither the onset
step nor the shape of the spike train.

ema3 vs clean A2 (tul-a2, same flags, same evaluate() instrument, 20
batches), periodic VAL CE:

| step | 500 | 1000 | 1500 | 2000 | 2250 | 2500 |
|---|---|---|---|---|---|---|
| clean A2 | 6.369 | 5.391 | 5.227 | 5.175 | 4.689 | 4.678 |
| ema3 | 6.627 | 5.750 | 5.576 | 5.492 | 5.019 | 4.982 |
| delta | +0.26 | +0.36 | +0.35 | +0.32 | +0.33 | +0.30 |

- **P-E1 (65%): FALSE.** 2 detonations / 3, the same 2/3 rate as the
  unfixed paid axis (A2 1/2, R1 1/2, A2s 0/2 -> 4/6 before this).
- **P-E2 (70%): FALSE.** The one healthy draw trails clean A2 by 0.30
  nats at step 2500, 3x the 0.10 band, and by 0.26-0.36 at every
  checkpoint from step 500 on. The near-frozen gamma is not free.

Forensics: $Q/tul-a2-ema{1,2,3}/{run.log,probe.jsonl}; DIVERGED_step_2040
checkpoints for ema1/ema2 under checkpoints/morph/tul-a2-ema{1,2}/.

## Verdict

**FAILURE.** beta=0.99 neither lowers the detonation rate nor preserves
CE. The binding's "detonations >= 2/3" branch applies: no 20k launched
(the chain logged "EMA FIX INSUFFICIENT (2/3)" and exited 1).

Two readings remain open, and the method cannot separate them:
(a) the contagion theory is right but beta=0.99 lags too little (over the
~40-step onset window gamma still moves ~33% toward a drifted mean, so it
smears the vault rather than stopping it); (b) gamma is not the trigger
at all and the cusp crossing of individual shadow weights is. The 0.30
nat cost of the near-frozen scale also argues that the live rescale was
doing useful work (reading: it is the ternary codebook's only adaptive
degree of freedom; freezing it at the step-0 mean|W| pins the codebook to
init statistics).

## Updated hypothesis

The next experiment is the hard freeze (beta=1.0), preregistered at
lab/experiments/planned/2026-09-02-gamma-freeze-discriminator.md, which
separates (a) from (b): freeze survives => (a), production fix is
freeze-after-warmup or beta >= 0.999; freeze detonates => (b), gamma is
exonerated and the trigger is the 0.5*gamma cusp itself (hysteresis or
dead-zone on the code assignment). Either way the CE cost of a frozen
gamma must be measured against clean A2 before it goes into a 20k recipe.
