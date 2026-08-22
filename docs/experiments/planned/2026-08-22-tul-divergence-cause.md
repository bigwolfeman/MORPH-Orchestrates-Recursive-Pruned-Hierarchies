# Experiment: what makes the TUL arms diverge?

Status: planned

## Question

All three arms of the 2026-08-21 gated-TUL bake-off failed. Two of them (`tul_a1`,
`tul_a1r`) carry **no gate at all** and both aborted on the pre-existing DIV-GUARD
after a slow loss detonation. So the divergence is a property of the TUL short recipe,
not of the gate. Which mechanism drives it?

Two candidate causes, not mutually exclusive:

- **(A) Contractivity.** The core map's gain crosses 1, the looped forward amplifies it
  as `ρ^T`, and the backward explodes. `CLAUDE.md`'s iterative-map section names this as
  the working explanation for the β1=0 AdEMAMix detonations, and names `ρ(J_core)` as the
  probe. `morph/training/spectral_penalty.py` exists to control it and is OFF
  (`spectral_penalty_cap: 0.0`, `spectral_penalty_lambda: 0.0`).
- **(B) The optimizer's slow-EMA push.** `ademamix_alpha: 8.0` warms in over
  `ademamix_t_alpha: 1600` steps. Onset is near step 600, inside that horizon.

## What is already measured (from the failed runs, not from this experiment)

- `train/grad_norm` on `tul_gate`: 1.69 @500 → 303 @600 → 9.0e4 @700 → 8.0e5 @800.
  On `tul_a1` it ends at 9.6e10 @5800. Ten orders of magnitude.
- Post-clip `gradnorm/core` = 0.9998 of the total norm; `gradnorm/coda` = 1e-10.
  The clip is one uniform rescale, so this is not "the coda has no gradient" — it is
  the core's gradient being ~1e10 larger, which annihilates every other region.
  **The prelude and the coda stop learning; only the core moves, on an exploded gradient.**
- σ_max of the core MLP linears in `tul-a1/DIVERGED_step_5900.pt`. Ternary QAT is on, so
  the forward uses the ternarized weight, NOT the shadow — both are reported because the
  first pass measured only the shadow and overstated the runaway by ~2x:

  | linear | shadow σ | **effective (ternarized) σ** |
  |---|---|---|
  | core.0.gate_up | 21.4 | **11.1** |
  | core.0.down | 13.7 | 6.6 |
  | core.1–5.gate_up | 5.0–10.2 | 2.8–5.1 |
  | core.1–5.down | 4.3–6.4 | 2.4–3.5 |

  8 of 12 core MLP linears are over the cap of 3 on the effective weight. `base.yaml`
  documents the healthy band as ~1.5 and a cap of 3 as "only fires on runaway"; the new
  `[spec]` logging reads **1.41 at init**, which confirms that calibration.
  `CoreSpectralPenalty` measures the effective weight (it calls `lin(v)`, so the STE is
  live), so the numbers this experiment logs are the effective ones.
- `tul_gate` did not reach DIV-GUARD; it OOMed at step ~1050 (23.58 GB peak + a ~6 GB
  desktop on a 31.4 GB card). Its grad norm was already 8e5, so it was detonating too.

## Method

Four arms, `tul_a1` config (gate OFF — the simplest arm that diverged), 1500 steps,
seed 0, batch 14, `eval_every=250`, `spectral_penalty_log_every=100`. Sequential, one GPU.

| arm | change vs `tul_a1` |
|---|---|
| D0 | none (control; σ logging only, which is bit-exact at λ=0) |
| D1a | `spectral_penalty_cap=3.0 spectral_penalty_lambda=0.1` |
| D1b | `spectral_penalty_cap=3.0 spectral_penalty_lambda=1.0` |
| D2 | `ademamix_alpha=0.0` (same optimizer, slow-EMA push removed) |

New instrumentation, written before the run: `spec/sigma_max`, `spec/sigma_mean`,
`spec/sigma_gate_up_max` and per-linear `spec/sigma/*`, logged every 100 steps on
**every** arm. With `lambda=0` the penalty early-returns an exact zero, so D0 and D2
stay numerically identical to the failed runs.

## Predictions

Written before the runs. D0 restates the failure; D1a/D1b/D2 are the real bets.

1. **D0**: `spec/sigma_max` crosses 3.0 before step 600. `train/grad_norm` exceeds 1e3
   by step 800. Val CE bottoms near step 1000 and is higher at 1500 than at its minimum.
2. **D1a (λ=0.1)**: too weak. σ_max still exceeds 6.0 by step 1500 and grad_norm still
   exceeds 1e3.
3. **D1b (λ=1.0)**: σ_max stays below 4.0 for all 1500 steps, grad_norm stays below 50,
   and val CE at step 1500 is **lower** than D0's at step 1500.
4. **D2 (α=0)**: delays but does not prevent. σ_max still crosses 3.0 by step 1000 and
   grad_norm still exceeds 1e3 by step 1500. Rationale: if (A) is the disease and (B) is
   only the trigger, removing the trigger moves the onset without removing the runaway.
5. The runaway is concentrated in `gate_up`: `spec/sigma_gate_up_max` is the largest
   per-linear σ on D0 at every logged step after 500, and `core.0` is the largest block.

## Decision rule

- D1b holds σ AND beats D0 on val CE → the cause is contractivity; re-run the bake-off
  with the penalty on, and the gate arms become comparable again.
- D2 stable and D1b not → the cause is the optimizer; the fix is an α schedule, not a
  spectral bound.
- Both stable → not separated by this design; the next experiment crosses them.
- Neither stable → neither mechanism is sufficient alone; escalate to a direct
  `ρ(J_core)` measurement on the live forward.


## Amendment 1 (2026-08-22, after D0 and D1a)

Two method defects, both mine, found by running the arms rather than by reading them.

**1. `training.steps` is not a run-length knob.** `ademamix_t_beta3: null` resolves to
`training.steps` (`morph/training/optimizer.py:152`), so `steps=1500` moved the optimizer's
beta3 warmup horizon from 20000 to 1500. D0 is therefore NOT a control for `tul_a1`: it is a
different optimizer trajectory. D0 ran 1500 steps clean (val 5.978 -> 4.756, monotone) where
`tul_a1` had already turned by 1500 (4.806 at 1000 -> 5.462 at 1500). **Prediction 1 is
falsified as stated**: sigma_max was 1.94 at step 600, not >3.0, and never reached 3.0.
Grad norm on the failed arms exploded at step 600 with sigma under 2, so crossing the cap of
3 is not the trigger.

**2. `cap=3.0` cannot bind at this horizon.** The hinge is `relu(sigma - cap)`, and the
control's sigma only reaches 2.8 by step 1500. D1a (lambda=0.1) is consequently a no-op: its
val CE matches D0 to four figures (250/500/1000/1250 = 5.9704/5.2816/4.9121/4.7619 against
D0's 5.9777/5.2856/4.9128/4.7563). Any cap tested at this horizon must be near 2.0.

**Estimator validated, not assumed.** D0 and D1a logged different sigma (2.74 vs 3.00 at step
1400), which looked like a power-iteration artifact from the shared `_v`. Exact `matrix_norm`
on the step-1500 checkpoints says otherwise: D0 = 2.821, D1a = 2.999, and D0's true value at
1400 extrapolates to ~2.73 against a logged 2.74. **The logging is accurate; the weights
really differ.**

That difference is the interesting part. Two runs whose val CE agrees to 0.02% ended 6% apart
in sigma, from nothing but numerical drift. Sigma is close to a free direction with respect to
the loss — which is the concrete, local version of the claim in `CLAUDE.md` that the optimizer
sees only `grad L` and is blind to the core map's gain. One pair is suggestive, not proof.

### Revised method (E arms)

Two arms, `training.steps=20000` so every schedule matches `tul_a1` exactly, stopped by wall
clock near step 6000 (`tul_a1` turned at 1000 and aborted at 5900). `eval_every=500` as in the
bake-off.

| arm | change vs `tul_a1` |
|---|---|
| E1 | none — an exact repro, to establish that the failure reproduces at all |
| E2 | `spectral_penalty_cap=2.0 spectral_penalty_lambda=1.0` — binds from ~step 700 |

### Revised predictions

6. **E1** reproduces the turn: val CE at 3000 is higher than its own minimum, and
   `train/grad_norm` exceeds 1e3 by step 1500.
7. **E2** holds `spec/sigma_max` below 2.3 for the whole run.
8. If E2's val CE at 3000 beats E1's, contractivity control is the fix. If E2 holds sigma but
   its CE turns anyway, sigma is a symptom and the cause is elsewhere — escalate to a direct
   `rho(J_core)` measurement on the live forward.

**Caveat carried forward:** `tul_a1` aborted at 5900 and `tul_a1r` at 2080 on the same recipe
at a different seed. The failure is stochastic with a large spread, so a single E1 arm is weak
evidence about timing. It is the E1-vs-E2 sigma contrast, not E1's abort step, that this
design can actually resolve.
