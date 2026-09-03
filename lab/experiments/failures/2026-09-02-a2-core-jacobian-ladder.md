# Planned: A2 core-Jacobian ladder — is the healthy paid loop sitting on the cliff?

Status: failure
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

## Results (2026-09-02 20:20-20:57, jac_ladder.py, batch 2, 200 power iters, conv ≤4e-7 everywhere)

| checkpoint | iter | sigma_step (worst) | rms_step (typical) | per-block rms range | per-block sigma range |
|---|---|---|---|---|---|
| tul-a2 step_2500 (healthy) | 0 | 55.0 | 1.052 | 1.036–1.057 | 12.1–30.7 |
| tul-a2 step_5000 (healthy) | 0 | 97.3 | 1.134 | 1.054–1.086 | 17.5–30.7 |
| tul-a2 step_2500 | 3 | 37.5 | 0.933 | ~1.03 | 11.3–19.6 |
| tul-a2 step_5000 | 3 | 44.1 | 0.948 | ~1.04 | 14.2–26.6 |
| tul-a2 DIVERGED 2040 (draw 1) | 0 | 257,187 | 158.6 | 1.004–1.282 | 12.6–284 |
| tul-a2 DIVERGED 2040 | 3 | 8,438 | 5.63 | 1.003–1.026 | 13.3–182 |
| ema1 DIVERGED 2040 | 0 | 41,418 | 34.7 | (blk rms gain 1.074) | |
| ema2 DIVERGED 2040 | 0 | 20,067 | 9.29 | (blk rms gain 1.026) | |
| frz1 DIVERGED 2040 | 0 | 70,751 | 15.0 | (blk rms gain 1.029) | |

Artifacts: lab/experiments/results/a2_jac_ladder_{a2,ema1,ema2,frz1}.json;
logs $Q/a2probes/a2-jac-*.log.

- **P-R1 (55%): TRUE.** Healthy rms_step at iter 0 = 1.052 (2500) and 1.134
  (5000). Iter 3 is mildly contractive (0.93, 0.95).
- **P-R2 (75%): TRUE.** Every DIVERGED sigma_step is ≥ 365x healthy 2500's
  (20,067 / 55.0), not 2x.
- **P-R3 (60%): TRUE.** sigma_step 55.0 → 97.3 and rms_step 1.052 → 1.134
  along the healthy trajectory.
- **P-R4 (40%): TRUE — against the majority prior.** Alignment
  (rms_step / Π rms_blocks): healthy 2500 = 1.052 / 1.279 = 0.82; DIVERGED
  draw 1 = 158.6 / 1.466 = 108; ema1 ≈ 34.7 / 1.53 = 23; ema2 ≈ 9.3 / 1.17 =
  8; frz1 ≈ 15.0 / 1.18 = 13. All ≥ 2x. BUT the per-block worst-direction
  sigmas on the A2 DIVERGED checkpoint also jumped 13 → 284 on blocks 0–1, so
  magnitude moved too.

## Verdict

**FAILURE by protocol (P-R4 resolved against its majority prior), and the
substantive reading is clear.** The healthy paid loop is expansive at its
first iteration and drifts outward while the loss falls: "the aggressive
minimum is unstable" (Wolfe) is measured, not argued. The sick operator is a
low-rank blowup (typical gain 9–159 across the whole step while every block's
typical gain stays 1.0–1.3 ⇒ ten to twenty directions at ~1e5, aligned across
the six weight-shared blocks).

What the method could not distinguish: whether the alignment PRECEDES the
magnitude jump or follows it. Every sick checkpoint here is ~1,700 steps past
onset; on such a checkpoint magnitude and alignment have both moved, and the
prereg's "R4 TRUE ⇒ alignment, not magnitude" deduction does not follow. The
2026-08-24 lesson (a uniform rescale cannot slow an alignment) still stands
on its own evidence; this run neither strengthens nor weakens it.

## Updated hypothesis

Interpretation-table branch R1 ∧ R3 applies: the next draw set is the
optimizer lever through the danger window (a 1000-step LR warmup, or
alpha_cap 1.5 / slower t_alpha), with K1−K6 ≥ 0.10 as the cost gate. To
settle the alignment-vs-magnitude order, the NEXT ladder must run on
PRE-onset checkpoints: a detonating draw with `ckpt_every=50` over steps
200–800, probed at every rung (planned file to be written before that run).
Both open items are indexed in lab/divergence/DIVERGENCE-README.md.
