# Experiment: is the core map under-determined on the slot manifold?

Status: planned

Written 2026-08-24, after the optimizer-state decomposition
(`docs/experiments/failures/2026-08-24-tul-optimizer-state-decomposition.md`) closed the
`alpha * m_slow` line. Nothing here was computed before this file was committed.

## Question

At matched `ademamix_alpha_cap` 3.5, arm A0 loops on 1024 token positions and rises +0.000
over 19500 steps, arm A3 has no core loop and rises +0.000, and arm A1 loops on about 57
slot positions and fails 4 of 4. Every other difference has now been measured and come back
a symptom: the weight spectrum (four arms, all worse than doing nothing), the optimizer
state (17 to 24 %, anti-correlated with harm), sigma_max (a consequence of the forward).

The one thing never varied is the quantity that actually differs: **how many independent
directions the loss constrains the shared core map in.**

**The claim.** For a linear inside the core, the gradient is `sum_i delta_i h_i^T`, so its
ROW SPACE is contained in the span of that linear's own inputs. About 50 slot states at
effective rank 1.7 to 4.8 can only ever push `W` inside a few directions of 1024. The
AdEMAMix update then applies an ELEMENTWISE `1/sqrt(nu)` rescale, which does not preserve
rank, so the applied update leaves that subspace. The weight therefore moves in directions
the loss cannot see and cannot correct. That is what "under-determined" means here, it is
mechanical rather than statistical, and it is directly measurable.

This also retro-explains three results that are otherwise unrelated: capping `sigma_max`
failed because it constrains directions the loss cannot see either way; the failure is
seed-dependent because WHICH unconstrained direction the map wanders into is a coin flip;
and `per_slot_embed` is the only lever that helped on both seeds because it is the only one
that raised the number of input directions.

## Method

**Probe A — do the per-slot embedding rows re-converge?** `TULSlots._seat` gives every row
the same mean plus deterministic jitter from a FIXED generator (`0x5107`), so both arms
start from a bit-identical `E_slot` and only the training seed differs. Raw pairwise cosine
would measure the shared mean, so the rows are CENTRED first. Reported: effective rank
(participation ratio of the centred singular values), mean pairwise cosine of the centred
rows, and centred spread `||E - rowmean||_F / ||rowmean||`. Checkpoints: `s0-slotembed`
(took over at 2225) and `b10-slotembed` (held to 3625), at step 2000 and step 4000.

**Probe B — the subspace measurement.** For each onset rung, load the checkpoint, run ONE
forward and backward on the fixed batch `jac_ladder.py` uses, and for each core MLP linear:

* capture that linear's own inputs with a forward hook, accumulate the Gram `sum_i x_i x_i^T`
* eigendecompose it; `k` = the number of eigenvectors holding 99 % of the energy; `P` = those
  eigenvectors; effective rank = `(sum lam)^2 / sum(lam^2)`
* `frac_g = ||g P||_F / ||g||_F` where `g` is that parameter's grad
* `frac_u = ||u P||_F / ||u||_F` where `u = (g + alpha*m2)/(sqrt(nu/bc2) + eps)` is the
  applied AdEMAMix update, read from the checkpoint's own optimizer state

Then the same on the TOKEN path (`slot_layout=None`) with the same weights, which is the
direct A0-versus-A1 comparison at fixed parameters.

## Predictions

**P1.** `frac_g >= 0.99` at every rung and every core MLP linear. This is true by
construction and is a SELF-TEST: below it, the subspace is wrong and no other number in
probe B may be read.

**P2.** Input effective rank on the slot path is below 20 (of 1024 or 2816) at every rung.

**P3.** `frac_u < 0.5` at every rung: most of the applied update leaves the subspace the
loss constrains.

**P4.** `frac_u` FALLS across the onset, by at least 20 % relative from rung 1700 to 1866.

**P5.** On the TOKEN path with the same weights, input effective rank is at least 10x the
slot path's, and `frac_u` is at least 2x higher.

**P6.** At step 2000 the arm that is about to fail carries LESS row diversity: the centred
effective rank of `s0-slotembed`'s 64 rows is below `b10-slotembed`'s. Both start identical,
so any difference is training.

**P7.** In both arms the centred effective rank FALLS from its initial 64 (the jitter is
isotropic Gaussian, so the rows start full rank).

## What this cannot decide

* Correlation, on all of it. Nothing here shows that adding input directions would prevent a
  takeover; that needs the granularity ladder, and that needs the GPU.
* `frac_u` mixes two effects — the elementwise rescale and the momentum's staleness — and
  this method does not separate them.
* The token-path pass runs A1-trained weights in an A0 configuration, which is not the same
  as an A0-trained model. It measures the operator, not a trained run.
* One batch, one seeded set of Poisson depths, one run's ladder.
