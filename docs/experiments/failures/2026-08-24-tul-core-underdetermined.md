# Experiment: is the core map under-determined on the slot manifold?

Status: failure

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

---

# Results

Ran 2026-08-24. Probes A and B read checkpoints offline; no training run produced any
number below. Instruments `lab/divergence/slot_rows_probe.py` and
`lab/divergence/subspace_probe.py`.

## The claim was wrong, and the probe's own self-test said so in one run

The pre-registered mechanism was that a core linear's gradient `sum_i delta_i h_i^T` is
confined to the span of about 50 slot states at effective rank 3, so the loss constrains
only a few of 1024 directions. **That is false.** The sum runs over batch x slots x loop
iterations x blocks — a few thousand distinct input vectors per linear, not 50 — so the
input span is nearly full rank. Measured on `ROLL_step_1850`: **935 of 1024 directions hold
99 % of the input energy**, and the first version of the probe duly reported `frac_g`
0.9927 and `frac_u` 0.9644 through a 935-dimensional projector, which is a measurement of
nothing.

What IS concentrated is the input ENERGY: participation ratio **11.2 of 1024**. The loss's
curvature along a direction scales with that direction's input energy, so the weight is
strongly constrained in about 11 directions and weakly constrained in the rest. The probe
was rebuilt to report the whole energy CURVE over k rather than one projector, with a
self-test at the far end: every curve must reach exactly 1.0 at k = in_dim.

## Probe A — the rows do not re-converge

`TULSlots._seat` seeds the jitter from a fixed generator, so both arms start from a
bit-identical `E_slot` and only the training seed differs. Rows are CENTRED first, because
they share a common mean by construction and the raw pairwise cosine measures that mean.

| arm | centred eff rank | centred pairwise cos | spread | raw cos |
|---|---:|---:|---:|---:|
| `b10-slotembed` @2000 | 27.15 | −0.0147 | 1.678 | 0.2553 |
| `b10-slotembed` @4000 | 34.04 | −0.0152 | 1.460 | 0.3160 |
| `s0-slotembed` @2000 | **42.92** | −0.0155 | 1.640 | 0.2641 |
| `s0-slotembed` @4000 | 42.87 | −0.0155 | 1.783 | 0.2300 |
| `s0-stack` @2000 | 22.37 | −0.0156 | 1.868 | 0.2049 |
| `s0-stack` @4000 | 22.55 | −0.0156 | 1.882 | 0.1882 |

The centred pairwise cosine is −0.015 everywhere, which is exactly −1/(n−1) for 64 centred
vectors with no structure. The rows are uncorrelated in every arm at every step.

**P6 fails, and in the opposite direction.** The arm that took over at step 2225
(`s0-slotembed`) carries centred effective rank **42.92** at step 2000, against **27.15**
for the arm that held to 3625. More row diversity, not less. **P7 holds**: both fall from
the initial ~63 that isotropic jitter gives.

This closes next-step 2 of the parent document. Frozen orthogonal per-slot offsets would fix
a problem that does not exist — the rows never collapse back.

## Probe B — the slot path and the token path, same weights, same batch

| step | slot eff rank | token eff rank | ratio | slot k50 | token k50 | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1626 | 20.57 | 81.37 | 3.96 | 11 | 117 | 10.55 |
| 1651 | 19.57 | 84.80 | 4.33 | 10 | 125 | 11.98 |
| 1676 | 15.17 | 52.76 | 3.48 | 7 | 83 | 11.73 |
| 1701 | 14.39 | 51.60 | 3.59 | 7 | 78 | 11.62 |
| 1726 | 17.25 | 63.69 | 3.69 | 9 | 92 | 10.34 |
| 1751 | 21.43 | 68.83 | 3.21 | 12 | 98 | 8.39 |
| 1776 | 22.91 | 68.05 | 2.97 | 14 | 97 | 7.02 |
| 1801 | 27.63 | 75.16 | 2.72 | 17 | 105 | 6.10 |
| 1826 | 22.64 | 63.72 | 2.81 | 14 | 95 | 6.92 |
| 1851 | 11.22 | **25.10** | 2.24 | 6 | 26 | 4.55 |
| 1866 | 13.49 | **27.50** | 2.04 | 7 | 25 | 3.54 |

Healthy-rung means: slot 20.17, token 67.78, ratio **3.36**.

Energy curves on the slot path, averaged over the 12 core MLP linears:

| step | g@8 | u@8 | m2@8 | g@32 | u@32 | gap@32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1626 | 0.3889 | 0.2965 | 0.1963 | 0.5657 | 0.4616 | +0.1041 |
| 1701 | 0.3855 | 0.2962 | 0.2193 | 0.5864 | 0.4819 | +0.1045 |
| 1751 | 0.3304 | 0.2548 | 0.1758 | 0.5319 | 0.4329 | +0.0990 |
| 1801 | 0.3708 | 0.2752 | 0.1919 | 0.5696 | 0.4527 | +0.1169 |
| 1826 | 0.3463 | 0.2496 | 0.1989 | 0.5510 | 0.4321 | +0.1188 |
| 1851 | 0.5211 | 0.3678 | 0.3049 | 0.6960 | 0.5436 | +0.1524 |
| 1866 | 0.5974 | 0.4243 | 0.3694 | 0.7537 | 0.5891 | +0.1646 |

## Prediction scorecard

**Two of seven hold, and both are the weak ones.**

| # | prediction | result | measured |
|---|---|---|---|
| P1 | `frac_g >= 0.99` at every rung | **holds, uninformative** | 0.9927 — but through a 935-dimensional projector, so it could not have failed |
| P2 | slot input effective rank below 20 at every rung | **FAILS** | 11.22 to 27.63; 5 of 11 rungs are at or above 20 |
| P3 | `frac_u < 0.5` | **FAILS** as written | 0.9644 at the 99 % projector. At k=8 it is 0.30 to 0.42, but that is not what was written |
| P4 | `frac_u` falls at least 20 % across the onset | **FAILS, opposite direction** | u@8 RISES 0.2962 -> 0.4243, a factor of 1.43 |
| P5 | token path at least 10x the input rank and 2x `frac_u` | **FAILS** | rank ratio 2.04 to 4.33. u@8 is sometimes lower on the token path |
| P6 | the arm about to fail has less row diversity | **FAILS, opposite direction** | 42.92 against 27.15 |
| P7 | centred effective rank falls from its initial 64 | **HOLDS** | to 22.4 to 42.9 |

## Verdict: failure

**Under-determination as stated is refuted.** The mechanical version is false — the input
span is nearly full rank. The energetic version is real but small: the token path is
**3.4x** better constrained than the slot path at healthy rungs, not the 10x predicted.

**And "more positions" is directly undercut by the token-path pass.** Running the SAME sick
weights on 1024 token positions, the input effective rank collapses just as it does on 57
slots: **75.16 -> 27.50, a factor of 0.37**, between rungs 1801 and 1866. The input
concentration at takeover is a property of the WEIGHTS, not of how many positions are fed
to them. A fix that only adds positions has to survive that.

What the energy curves do show is that at takeover the gradient concentrates INTO the top
input directions (g@8 from 0.35 to 0.60) while the applied update lags behind it (u@8 from
0.25 to 0.42), so the whitening gap widens from +0.10 to +0.16. That is coincident with the
takeover, not ahead of it, and it re-describes the known forward state collapse from the
weight side rather than adding a cause.

## One usable lever came out of it

`tul.span_cap` 32 -> 12 raises the core input effective rank on the same healthy checkpoint
(`ROLL_step_1700`) from **14.39 to 25.68**, a factor of 1.78, closing about a quarter of the
slot-to-token gap. It is a config knob and needs no code change.

Two facts govern how it must be used. The packer **ends the row early** once `max_slots` is
spent, so finer spans bought that rank partly by padding tokens away — a confound in the
number above. And `max_slots` 128 with `span_cap` 12 covers the full 1024 tokens with no
early row end and **fits at batch 6, peak 14.49 GB measured**; the OOM recorded in the
parent document was only ever tested at batch 10 and 12.

`tul.boundary_chars` is NOT usable as a Hydra command-line override: the grammar needs the
comma escaped, and the escaped form arrives as the literal string `.;!?\,`, which would
silently make backslash a boundary character. A ladder that varies the boundary set needs a
config FILE, not an override.

## What this does NOT show

* **Nothing here is causal.** The token-path pass evaluates A1-trained weights in an A0
  configuration; it does not show what a model TRAINED with more positions would do. That
  is the granularity ladder, and it needs the GPU.
* **The `span_cap` 12 rank measurement is confounded by padding** and is one checkpoint.
* **One batch, one seeded set of Poisson depths, one run's ladder** for all of probe B.
* **Probe A is three arms**, and the two that matter differ in seed as well as in outcome.
* The energy curves mix the elementwise `1/sqrt(nu)` rescale with the accumulator's
  staleness. This method does not separate them.
