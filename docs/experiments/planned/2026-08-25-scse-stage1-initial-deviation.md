# Experiment: does SCSE Stage 1 (a non-zero initial deviation) stop the A1 takeover?

Status: planned

## Question

MORPH starts its core loop at `h_0 = e.clone()`. With the natural input-conditioned anchor
`h* = e` the initial deviation is exactly zero, and SCSE's Theorem 2 then makes the ENTIRE loop
trajectory the propagated forcing response `Delta_T = sum_k Phi_E(T, k+1) b_k(e)`, with
Corollary 5 showing that response can grow like `rho^T`. Nothing in MORPH bounds it.

[H21](../failures/2026-08-24-tul-forcing-bias-predicts-divergence.md) refuted the claim that the
forcing bias PREDICTS which seeds diverge: across three healthy seeds `b_t` at step 1000 spans
4.0 % and carries no ordering. What H21 did not test is whether the mechanism is CAUSAL, because
it never intervened. The one seed that did diverge carried `b = 13.1` at its first rung against
1.47-1.63 for the healthy seeds, and 82.5 at the abort.

Correlation studies on healthy seeds are exhausted. **This is the intervention.**

**Question.** With `Delta_0` moved off zero — SCSE Stage 1, `h_0 = e + 0.1 * H_0(e)` — does the
forcing bias stop growing, and does the divergence stop?

## Hypothesis

H23. `Delta_0 = 0` is load-bearing. Giving the loop a state of its own, so its trajectory is no
longer purely the propagated forcing response, slows the growth of `b_t` and removes the
takeover, at no cost in CE.

## Method

Four seeds (0, 1, 2, 3) of `tul_a1` with `model.core_init_scale=0.1`, 3500 steps, batch 6,
`ademamix_alpha_cap=3.5`, `model.use_kernels=false`, `eval_every=250`, `ckpt_every=500`.
Every setting is identical to H21's sweep except the one field.

**The control is H21's sweep, not a fresh run.** `core_init_scale=0.0` builds `_CloneInit`,
which holds no parameters and draws no RNG, and `tests/test_scse_core_init.py` pins that
bit-identically: same parameter set, same post-build RNG state, same logits. Re-running the
control would spend 2.2 GPU-hours to reproduce numbers already in hand.

**These are fresh samples, NOT paired trajectories.** `_SCSEInit` is built last, so a Stage 1
model shares byte-identical weights with its control everywhere except the new projection — but
that projection draws RNG, and the seed is set BEFORE the model is built
(`train.py:1273` then `:1335`), so every downstream stream shifts: data order, dropout masks,
depth draws. MORPH decorrelates within 11 steps of any perturbation regardless. Seed `s` here
and seed `s` in H21 are two draws from the same distribution, not the same run perturbed.

`G_theta(0) = 0` was verified numerically before this run, on the real 286.1M model with
retention live, in fp32 and under bf16 autocast: peak `|out| = 0.000e+00` in both. That closes
acceptance criterion 3 of the port plan. The zero-deviation mask (Stage 3) is still NOT enabled
and must not be until Stage 1 is measured — enabling it at `Delta_0 = 0` freezes the loop.

**Validity gate.** `R_0 = 1.000` and the probe's trajectory gate at `0.0` at every checkpoint,
as in H21.

**Power, stated in advance and learned from H21.** The divergence COUNT is the weakest endpoint
here and cannot decide anything alone: the baseline rate is 1 of 4, so under the null a clean
0-of-4 has probability `(3/4)^4 = 0.32`. It is reported, and it is reported as underpowered.
The continuous endpoints — `b_t` at matched rungs, and CE — carry the weight.

**The CE threshold is set from a MEASURED noise floor, which H21 failed to do.** This metric's
largest within-run rise that later recovered to a new minimum is 0.168 nats. Any CE threshold
below that is meaningless. P4 uses 0.17.

## Predictions

Written before the first Stage 1 step.

* **P1 (validity).** `||Delta_0|| / ||e|| >= 1e-3` at every checkpoint of every seed. If this
  fails the mechanism is not live and nothing below is readable.
* **P2.** The maximum `b_t` over rungs 500-3500 is LOWER than its control seed's, in at least 3
  of the 4 seeds. Control maxima: s0 = 82.519 (at the 2040 abort), s1 = 8.323, s2 = 4.184,
  s3 = 1.966.
* **P3 (low power, stated as such).** No seed triggers the divergence guard by step 3500. The
  control had 1 of 4.
* **P4.** Mean final validation CE over the seeds that stay healthy is not worse than the
  control's by more than 0.17 nats — one measured noise floor. Control healthy finals: 4.6863,
  4.5303, 4.5073 (mean 4.5746).

## What would refute H23

`b_t` maxima equal to or above the control in 2 or more seeds, AND a seed still diverging. That
would mean a non-zero initial deviation does not touch the mechanism, and the port should go
straight to Stage 2 (deviation coordinates, which moves the source injection out of the loop and
into the anchor) or be abandoned.

A partial result is also possible and must be reported as partial: `b_t` falling while CE gets
worse would make Stage 1 a trade, not a fix, and the decision would then turn on P4.

If P1 fails, the run is void — a mechanism that did not engage tests nothing.
