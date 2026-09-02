# Planned: gamma-EMA validation — the cusp-vault fix on the paid axis

Status: planned
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
