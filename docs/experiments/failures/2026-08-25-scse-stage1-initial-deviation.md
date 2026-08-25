# Experiment: does SCSE Stage 1 (a non-zero initial deviation) stop the A1 takeover?

Status: failure

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

## Results

Artifacts: [`../results/2026-08-25-scse-stage1-initial-deviation/`](../results/2026-08-25-scse-stage1-initial-deviation/).
Scored by `lab/divergence/score_h23.py`, committed at `76a7b3e` before any Stage 1
checkpoint was probed.

**Instrument regression gate passed first.** `forcing_bias` moved its anchor from
`points[0]["h"]` to `points[0]["e"]`; those are the same tensor on a baseline model, so
re-probing baseline seed 3 had to reproduce the stored `b_t` exactly. Worst relative change
across all seven rungs: **`0.000e+00`**, and baseline `delta0_rel` came back exactly `0.0`.

**Validity gate passed**: trajectory gates `0.0` everywhere; baseline holds `R_0 = 1.000` and
`Delta_0 = 0`.

### Verdicts

| prediction | verdict | number |
|---|---|---|
| P1 mechanism live | **HELD** | `\|Delta_0\|/\|h*\| = 0.0415-0.0432`, `R_0 = 0.996-0.999` (not 1) |
| P2 `b_t` max lower in >=3 of 4 | **FAILED** | 1 of 4 |
| P3 no divergence | **FAILED** | seed 0 diverged, same as control |
| P4 CE within 0.17 nats | **FAILED** | **+0.815 nats**, ~5x the noise floor |
| **REFUTER** | **FIRED** | 3 seeds not-lower AND a divergence |

### Forcing bias, control -> Stage 1

| seed | control max `b_t` | Stage 1 max `b_t` | ratio |
|---|---|---|---|
| 0 | 82.519 | 14.139 | **0.171** |
| 1 | 8.323 | 38.157 | 4.585 |
| 2 | 4.184 | 6.692 | 1.599 |
| 3 | 1.977 | 10.861 | 5.493 |

### Final validation CE

| seed | control | Stage 1 | delta |
|---|---|---|---|
| 1 | 4.6863 | 5.5751 | +0.889 |
| 2 | 4.5303 | 5.1077 | +0.577 |
| 3 | 4.5073 | 5.4870 | +0.980 |
| mean | 4.5746 | 5.3899 | **+0.815** |

## Verdict

**H23 is refuted.** The refuter fired on its pre-registered condition. A non-zero initial
deviation makes MORPH strictly worse: the forcing bias grows LARGER on three of four seeds,
final CE is 0.815 nats worse — about five times the measured noise floor, so this is not
sampling — and the seed that diverged under the control diverged again.

The single seed that improved on `b_t` (s0, 82.5 -> 14.1) is the seed whose control was
pathological from step 250 and never learned. Against a broken baseline, "better" is not
evidence.

Stage 1 did exactly what it was built to do — `Delta_0` is a real 4.2 % displacement and the
`R_0 = 1` identity is correctly broken — and the model got worse for it. The mechanism engaged
and the outcome went the wrong way. That is the cleanest kind of negative result.

**A partial effect worth recording:** on seed 0, Stage 1 trained for four times as long and
1.60 nats deeper before collapsing (min CE 5.0823 at step 1000, against the control's 6.6820
at step 250), then failed at the same guard step. The guard step being identical is NOT a
coincidence: the guard only arms at `step > 2000`, samples every 20 steps and needs two
consecutive strikes, so 2040 is the earliest abort it can emit. Both runs were already past
ppl 1000 when it armed.

## The R_t trap — and why this result had to be read on CE, not on the paper's diagnostic

SCSE's headline diagnostic is `R_t` (their eq. 9), and by that metric **Stage 1 looks like a
clear win**: at the last loop iteration `R` falls in almost every seed and rung, e.g. seed 3 at
step 3500 goes `0.520 -> 0.006`, and seed 1 at 3500 goes `0.185 -> 0.007`.

Reporting that as success would have been theater. `R_t = ||b_t||^2 / ||Delta_{t+1}-Delta_t||^2`
is a RATIO. SCSE drives it to zero by shrinking the numerator — killing the forcing bias.
Stage 1 drove it down by INFLATING THE DENOMINATOR: `b_t` went up 1.6x to 5.5x while the
realised per-step update grew faster still. Same direction on the diagnostic, opposite thing
happening in the model, and CE 0.815 nats worse.

**A second and larger caveat about applying this paper to MORPH.** The paper's looped baseline
sits at `R_0 = 1.000` rising to `R_47 = 4.35-5.44` — the anchor response GROWING to dominate
the realised update. MORPH's baseline does the opposite: `R_0 = 1.000` falling to
**0.056-1.906** at the last iteration, below 1 in most rungs. MORPH is not in the regime SCSE's
diagnostic was calibrated on, and the mechanism the paper removes may simply not be MORPH's
disease. That is measured here for the first time and it applies to the whole port, not just
to Stage 1.

## Updated hypothesis

Stage 1 is dead. Two things follow, and they point in opposite directions:

1. **Stage 1 is not SCSE.** The paper's mechanism is a zero-preserving core plus a
   zero-deviation mask plus a LEARNED anchor `h* = e + a_omega(e)`, acting together; their own
   Table 2 shows the anchor choice alone swings PPL from 155.14 to 294.37. Stage 1 changes
   `h_0` and leaves the recurrence untouched, so nothing in it reduces `b_t` — and `b_t` duly
   rose. A fair test of the paper is Stage 2, not this.
2. **The regime evidence argues against continuing.** MORPH's `R_t` falls with depth where the
   paper's rises. `G_theta(0) = 0` already holds exactly on the real model, so the
   reparameterisation the paper leans on is partly free here and still buys nothing. And H21
   already showed the forcing bias does not predict which seeds fail.

**Recommendation: do not run Stage 2 next.** The cost is a rewire of the carrier, the embedding
injection paths and every divergence instrument, and three independent measurements now point
away from the forcing bias being MORPH's mechanism: H21 (not predictive), H23 (intervening on
it makes things worse), and the `R_t` regime mismatch (MORPH is not the system the paper
describes). If SCSE is revisited it should be after a measurement that puts MORPH INSIDE the
paper's regime, not before.

## Amendment 2026-08-25 — the recommendation above is WITHDRAWN

Two claims in the sections above do not survive a re-read of the paper. The **verdict is
unchanged**: H23 is refuted, Stage 1 costs 0.815 nats, and every number in the tables
reproduces (independently recomputed from the raw JSONs and logs by an audit pass on
2026-08-25). What is withdrawn is the reasoning that went BEYOND the verdict.

**1. "MORPH is not in the regime SCSE's diagnostic was calibrated on" — unsupported.**
The comparison is MORPH's `R_t` at loop iteration <= 7 against the paper's `R_47`. The paper
reports baseline `R_t` at exactly two steps, t = 0 and t = 47 (their Table 3, Figure 5), and
`R_0 = 1.000` is an identity that holds in both systems by construction. There is therefore
no overlapping measurement, and the paper's baseline could equally sit below 1 at t = 7.
The defensible statement is narrower and is a within-MORPH one: MORPH never runs depth
extrapolation, and MORPH's own measurements (H19, H21) do not show the forcing response
accumulating in MORPH's operating range.

**2. "Recommendation: do not run Stage 2 next" — withdrawn.**
It rested on claim 1 plus the Stage 1 null. The paper's abstract states the ablations
"identify the learned anchor and the anchor-coordinate deviation recurrence as the primary
contributors to the gain". Stage 1 implements neither. Recommending against the only
configuration the authors credit with the improvement, on the strength of a configuration
they never report, is not supported by this experiment.

**3. A precision fix on `G_theta(0) = 0`.** `tests/test_scse_core_init.py` zeroes the
carrier, the source `e`, AND the injection terms. It therefore proves the core BLOCK STACK
is bias-free — which is `G_theta(0) = 0` for a **source-free** core, the one SCSE actually
uses. It does NOT show that MORPH's current core map, which takes `e` every iteration, is
zero-preserving. That map is not, and cannot be: `injection(0, e) != 0`.

Also corrected, both immaterial to any verdict: the Predictions section lists control seed 3's
`b_t` max as 1.966 where the scorer and the results table say 1.977, and the `delta0_rel`
range top is quoted as 0.0432 where seed 0 reaches 0.0439.

The full method is specified in [docs/scse-spec.md](../../scse-spec.md) and is being run.
