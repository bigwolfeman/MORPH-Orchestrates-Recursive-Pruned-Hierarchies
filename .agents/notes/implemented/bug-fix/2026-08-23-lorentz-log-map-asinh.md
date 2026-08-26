# Agent Note: compute the Lorentz log map with asinh

Status: implemented

## Problem

`_log_map_origin` in `morph/model/embeddings.py` computed the tangent vector through
`acosh(x₀)` and `sqrt(x₀² − 1)`. On the hyperboloid `x₀ = sqrt(1 + ‖xs‖²)`, so `x₀² − 1`
is a subtraction of two nearly equal numbers whenever `‖xs‖` is small — catastrophic
cancellation, which is why the function carried three separate clamps.

The clamps then produced a second defect. The guard meant to handle the removable
singularity at the origin read:

```python
denom = torch.sqrt(torch.clamp(x0 * x0 - 1.0, min=_EPS))   # _EPS = 1e-6
coeff = torch.where(denom < 1e-4, torch.ones_like(denom), alpha / denom)
```

`denom` is floored at `sqrt(1e-6) = 1e-3`, so `denom < 1e-4` is never true. **The guard was
unreachable.** Measured rather than reasoned: for `‖xs‖ < ~2e-3` the coefficient was a
constant **1.3811** instead of the correct limit **1.0** — a 38 % error, with a
discontinuity at the boundary where it jumped back to 1.

This surfaced from a question about whether the hyperbolic embeddings needed normalising to
avoid the TUL gradient blow-up. They did not — this map **is** the Lorentz channel's
normalisation, and it was the map itself that was subtly wrong.

## Decision

Use the identity `acosh(sqrt(1 + n²)) ≡ asinh(n)` and compute

```python
coeff = asinh(‖xs‖) / ‖xs‖
```

directly. This removes the cancellation, the `acosh`, two of the three clamps and the dead
branch in one change, and it makes the normalising behaviour explicit: the coefficient
decays like `ln‖xs‖ / ‖xs‖`, so embedding norms grow logarithmically and a runaway in the
raw table cannot produce a runaway embedding.

`tests/test_lorentz_log_map.py` gates it with 16 tests. Verified they fail on the old
version: reverting gives **6 failed, 10 passed**; restoring gives **16 passed**.

## Alternatives considered

- **Repair the guard's threshold in place** (compare against `1e-2`, or lower `_EPS`).
  Rejected: it keeps the cancellation that made the clamps necessary, so it fixes the
  symptom and leaves the cause. The threshold would also have to be re-derived for every
  dtype.
- **Leave it alone because the zone is unreachable.** Measured and true today — over a
  diverged checkpoint and a healthy 20k one, **0 of 49169** vocabulary rows have `‖xs‖`
  below 2e-3 and the minimum is 0.072, about 36× above the boundary. Rejected anyway: a
  38 % error and a gradient kink sitting one collapsed-token away from the live path is a
  trap, and the correct form is simpler than the incorrect one.
- **Normalise the Lorentz channel separately** (an explicit norm bound on the table).
  Rejected as unnecessary: the log map already provides logarithmic compression, and adding
  a second mechanism would make the geometry harder to reason about for no measured gain.

## Consequences

- **No effect on trained models.** On the real 20k Lorentz table the two forms differ by at
  most **5.96e-8** in absolute terms, and **0 of 49169** rows differ by more than 1e-6.
  Existing checkpoints behave identically; no re-run is needed.
- Slightly cheaper: one `asinh` and one norm instead of `acosh`, a square, a subtraction,
  a square root and three clamps.
- The `_EPS` constant is now used for exactly one thing — the removable singularity at
  `‖xs‖ = 0` — rather than papering over a conditioning problem.
- **This is not an explanation of the TUL core takeover.** That is an unstable BACKWARD
  operator inside the core stack
  ([`the block-gain result`](../../../../lab/experiments/results/2026-08-23-tul-block-backward-gain.md));
  the embedding region's gradient *follows* the core's by roughly 600 steps rather than
  leading it. This note fixes a latent defect found while checking that hypothesis, and
  rules the embedding out as the driver.
