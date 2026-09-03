# Planned: A2 core-Jacobian ladder — is the healthy paid loop sitting on the cliff?

Status: planned
Date: 2026-09-02 (frozen ~19:45, before the probe runs; trigger: Wolfe — "Or
maybe this aggressive minima they find is unstable.")

## Question

The winner recipe (retention off, spectral cap 0, ternary on, AdEMAMix
alpha_cap 3.5) is the same for notul-20k (survived 20k), R1 (1/2), A2 (1/2),
A2s (0/2), and the gamma-EMA draws (1/3). Its per-draw detonation rate is ~70%
and it is the only recipe whose loop earns depth. The CLAUDE.md model says
the loss landscape inherits the ρ=1 bifurcation of the inner map and the
cliff steepens like ρ^T. Is the HEALTHY paid loop expansive (an unstable
minimum that a discrete kick sends over), or contractive with a separate
discrete trigger (the ternary cusp) doing all the damage?

## Method

`lab/divergence/jac_ladder.py` (existing; power iteration on JᵀJ of one core
step at the live operating point, fp32, active-set masked, per-block sigmas),
`--config-name tul_a2 --overrides training.batch_size=2,model.use_kernels=false
--iters 0,3 --power-iters 200`, one fixed validation batch, model in train
mode (Poisson depths, as every prior ladder). Checkpoint dirs, one invocation
each:

| dir | checkpoints | override |
|---|---|---|
| tul-a2 | DIVERGED_step_2040.attempt1 (draw 1), step_2500, step_5000 (draw 2, healthy) | none |
| tul-a2-ema1 | DIVERGED_step_2040 | ternary_scale_ema_beta=0.99 |
| tul-a2-ema2 | DIVERGED_step_2040 | ternary_scale_ema_beta=0.99 |
| tul-a2-frz1 | DIVERGED_step_2040 | ternary_scale_ema_beta=1.0 |

Read per checkpoint and iteration: `sigma_step` (worst case), `rms_step`
(typical gain, the number that matches what gradients do), the per-block
product, and alignment = sigma_step / block product. Convergence residual
next to every sigma.

## Predictions (frozen)

- **P-R1 (expansive healthy map).** Healthy A2 (step_2500 AND step_5000)
  has `rms_step` ≥ 1.0 at iter 0: **55%**. Weak prior: the recipe that earns
  is the recipe with no cap, and l2cap's earning needed σ ≤ 1.5 to survive.
- **P-R2 (the sick operator is visibly different).** Every DIVERGED
  checkpoint's `sigma_step` at iter 0 is ≥ 2× healthy step_2500's: **75%**.
  Their raw weights blew up to preclip 1e9–1e15; under ternary the effective
  map scales with γ = mean|W|.
- **P-R3 (outward drift along the healthy trajectory).** On healthy A2,
  `sigma_step(5000) > sigma_step(2500)` at iter 0: **60%**.
- **P-R4 (alignment, the takeover signature).** Alignment at the DIVERGED
  checkpoints ≥ 2× healthy step_2500's: **40%**. On the paid axis the
  cotangent is a sum over ~1152 positions per row, not 50 slots, so I lean
  toward magnitude over alignment here.

## Interpretation (frozen)

- R1 TRUE and R3 TRUE ⇒ "aggressive minimum is unstable" holds: the healthy
  run lives above ρ=1 and drifts outward; the next draw set is the optimizer
  lever (LR 5e-5 or alpha_cap 1.5, 3 draws each) with the earning bar
  K1−K6 ≥ 0.10 as the cost gate. Test 3 from the 19:35 list.
- R1 FALSE (contractive healthy map) ⇒ the detonation is a jump, not a slide;
  the discrete trigger (the 0.5γ cusp under a live γ) stays the prime
  suspect, and a code-assignment hysteresis is the next mechanism test.
- R4 TRUE ⇒ alignment, not magnitude; per the 2026-08-24 result a spectral
  cap cannot help, and the A2c contingency is dead on arrival. This holds
  regardless of R1.
- Dense-then-ternary warmup is NOT a candidate under any outcome (Wolfe,
  2026-09-02: ternary weights organize differently).

## Not verified before run

Memory of the fp32 double-backward at 2×1152 active positions on the 5090
(fallback: batch_size=1); whether `load_checkpoint` accepts a DIVERGED
checkpoint whose weights carry 1e15-scale entries without a NaN guard
tripping; power-iteration convergence at 200 iterations on a residual
Jacobian (the residual is reported and a `rel_change` above 0.05 voids that
sigma).
