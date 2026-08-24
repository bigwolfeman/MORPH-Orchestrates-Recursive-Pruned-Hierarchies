# Experiment: which channel of the AdEMAMix update drives the core at onset

Status: failure

Written 2026-08-24, after the takeover-cure campaign
(`docs/experiments/failures/2026-08-24-tul-takeover-cure.md`) and BEFORE any number in
this file was computed. Nothing here is launched on the GPU as a training run: every
input already exists on disk.

## Question

Two questions, one instrument.

**Q1.** At the TUL core takeover onset, which half of the AdEMAMix update moves the core's
weights — the per-step normalised gradient `g / sqrt(nu/bc2 + eps)`, or the slow
accumulator `alpha * m2 / sqrt(nu/bc2 + eps)`?

**Q2.** The parent document names the severity measure it should have used and did not:
post-optimizer coherent core drift, `||dW_core||` times its directional autocorrelation.
Does that measure separate the arms in the order of their measured harm, where the
per-block backward gain demonstrably does not?

## Why this, and why now

Two structurally unrelated interventions both hold one seed and lose the other.
`per_slot_embed` changes the FORWARD input diversity. `bptt_depth 2` changes the BACKWARD
truncation. Neither touches the optimizer's memory. A slow accumulator with a half-life of
about 700 steps at beta3 = 0.999 is the one object in the system that could make two
unrelated fixes fail in the same seed-dependent way, and it is the one object nothing in
this campaign has measured.

The parent document also records a method defect: `bptt_depth 2` has a HIGHER per-block
backward gain than its control (2.784 against 2.445) with 64 % LESS harm. Gain is the map's
response per application. Harm is the integrated weight motion that survives truncation,
clipping and the optimizer. Q2 is the attempt to measure the second thing directly.

## Method

Inputs, all already on disk:

* `checkpoints/morph/onset-capture/` — 11 rungs, `ROLL_step_1625` .. `ROLL_step_1850` at
  25-step spacing plus `TAKEOVER_step_1866`. Each holds model AND optimizer state.
  Core share over these rungs: 0.016, 0.012, 0.021, 0.372, 0.890, 0.961 at
  1625/1700/1750/1800/1850/1866. The first three are healthy, the last three are sick.
* `checkpoints/morph/{b10-ctrl,b10-bptt2,b10-slotembed,s0-slotembed,s0-stack}/step_2000.pt`
  and `step_4000.pt` — the cure arms, two rungs each.

Instrument: a new offline reader, `lab/divergence/optstate_probe.py`. It builds the model
through the same path `jac_ladder.py` uses, so the optimizer index to parameter-name map is
the real one and not inferred from shapes. It dequantises the blockwise-8bit state through
the optimizer's OWN quantiser (`bitsandbytes.functional.dequantize_blockwise` with the
optimizer's code map and blocksize), never by a hand-written formula.

Derived quantities, per parameter and then aggregated per region using the SAME region rule
as `_preclip_probe` (first dotted component of the name, `_orig_mod.` stripped):

* `slow_rms = RMS( alpha_t * m2 / sqrt(nu/bc2 + eps) )`. The fast channel's per-coordinate
  RMS is approximately 1 by construction, because `nu` is the EMA of `g**2`. So `slow_rms`
  reads directly as "the slow channel in units of the fast channel". `alpha_t` and `bc2`
  come from the optimizer's own `_sched` at the checkpoint's own step.
* `coh = RMS( |m2| / sqrt(nu/bc2) )`, the same number with alpha removed. For a white
  gradient sequence an EMA at beta3 gives `E[m2**2]/E[g**2] = (1-b3)/(1+b3)`, which at
  beta3 = 0.999 is 5.0e-4, so `coh` has a white-noise floor near 0.022. Any value far above
  that is directional persistence in the gradient.
* `drift_k = ||dW_core||` between consecutive rungs, and
  `ac_k = cos(dW_core[k], dW_core[k+1])`, and their product. This is the parent document's
  named severity measure, at the 25-step scale rather than per step.

## Method amendments

**Amendment 1, 2026-08-24, after the first ladder pass and before any prediction was
scored.** `coh` was defined as an aggregate of per-coordinate ratios,
`RMS(|m2| / sqrt(nu/bc2))`. That estimator has an unbounded tail wherever `nu` is small but
non-zero, so its value is set by a few coordinates. Measured on the 11 rungs it returned
5.98, 2.88, 0.92, 28.4, 1.62, 2.83, 8.30, 18.4, 39.3, 0.087, 0.068 — three orders of
magnitude with no relation to the onset. It is replaced by the ratio of aggregates,

    coh = RMS(m2) / RMS_ema(g) = sqrt( sum(m2**2) / (sum(nu)/bc2) )

which is the same quantity at the tensor level and is not outlier-driven. The white-noise
floor is unchanged at 0.0224. **P4's thresholds were written against the rejected
definition and are scored against the new one; that substitution is recorded here rather
than hidden, and P4 is reported as scored-under-amendment whatever it does.** The rejected
per-coordinate number is still computed and stored as `coh_percoord`.

**Amendment 2, same pass, same reason.** `||dW||` is added to as a SHARE of the whole
model's displacement, not only as a magnitude. `clip_grad_norm_` rescales every gradient to
a fixed global norm each step, so `||dW_all||` is close to a constant of the schedule and
`||dW_core||` inherits that. The share is the part that can still move. This adds a derived
column; it changes no prediction.

## Predictions

Written before the instrument was run. Six of them can fail.

**P1.** At the healthy rungs 1625, 1700 and 1750, the core's `slow_rms` is BELOW 1.0: the
slow channel is smaller than the fast channel.

**P2.** At the sick rungs 1800, 1850 and 1866, the core's `slow_rms` is ABOVE 1.0, and it
rises monotonically across those three rungs.

**P3.** The rise is specific to the core. `slow_rms(core, 1866) / slow_rms(core, 1700)` is
at least 1.5 times `slow_rms(non-core, 1866) / slow_rms(non-core, 1700)`, where non-core is
every region except `core`.

**P4.** The alpha-free coherence `coh` for the core is below 0.15 at 1700 and above 0.25 at
1866.

**P5.** The coherent-drift measure has more dynamic range across the onset than the
per-block gain does. `drift * ac` at the last interval is at least 3 times its value at the
1700 -> 1725 interval, where the realized per-block gain moved only 1.057 -> 1.238, a factor
of 1.17.

**P6.** The directional autocorrelation `ac` is below 0.15 at the healthy intervals and
above 0.40 at the sick ones.

**P7.** On the arms at step 2000, the core `slow_rms` ranks in the order of measured harm:
`b10-slotembed` < `b10-bptt2` < `b10-ctrl`.

**P8.** The new measure does NOT invert on the `bptt_depth 2` arm the way the per-block gain
did. `b10-bptt2` scores BELOW `b10-ctrl`. **If P8 fails the new measure is no better than
the one it replaces, and this experiment is a failure whatever P1 to P7 do.**

## What this cannot decide

* `m2` and `nu` are blockwise-8bit for every parameter with 4096 or more elements, so every
  number carries the quantiser's error. The self-test measures that error rather than
  assuming it is small.
* `g` is not stored in a checkpoint. The fast channel's RMS is taken as approximately 1 from
  the definition of `nu`, not measured. If `nu` lags a fast-changing gradient the true fast
  channel is larger than 1 and `slow_rms` overstates the slow channel's share.
* `dW` between rungs is an integral over 25 steps, so `ac` is coherence at the 25-step
  scale. This is NOT the per-step measure the parent document names. The optimizer-state
  route (`coh`) is the per-step estimate, and P1 to P4 versus P5 to P6 agreeing is the
  cross-check between the two scales.
* Correlation only. Nothing here shows that suppressing the slow channel would prevent the
  takeover. That needs an arm, and an arm needs the GPU.

---

# Results

Ran 2026-08-24. No training run was launched: 34 checkpoints were read offline, 11 from
the onset ladder and 23 from ten arms. Instrument `lab/divergence/optstate_probe.py`,
scorer `lab/divergence/score_optstate.py`, tests `tests/test_optstate_probe.py` (19 pass,
4 sabotages caught).

## One defect in the instrument, found and fixed before any prediction was scored

The first version hardcoded the denominator as `sqrt(nu/bc2 + eps)`. MORPH runs
`ademamix_eps_inside: false`, which is `sqrt(nu/bc2) + eps`. At MORPH's gradient scale the
two differ by about a factor of 100, and the wrong one reported that 99.2 to 100 % of core
coordinates sat on the epsilon floor — which would have meant the optimizer was not
normalising at all. It was reading a denominator no run has ever used.

The mistake was caught by adding a floor-fraction diagnostic rather than by inspection, and
the fix is pinned by `test_eps_inside_changes_the_denominator`. Every number below comes
from the corrected reader. The numbers from the broken one are discarded, not adjusted.

The corrected reader also validates itself twice. `sqrt(nu/bc2)` on the largest core tensor
gives an implied per-coordinate gradient RMS of 5.045e-05, against 5.912e-05 predicted
independently from "a 286.1M-parameter gradient clipped to global norm 1.0" — 15 % apart,
from two unrelated routes. And the measured fast-channel RMS is 0.934 to 0.961 across the
ladder, so the pre-registration's "approximately 1" assumption holds.

## The onset ladder

`core share` is the run's own `preclip/core / preclip/total` at the nearest probed step.

| step | core share | slow/fast core | slow/fast noncore | coh core | coh noncore | coh ratio | dW share | ac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1626 | 0.0170 | 0.1751 | 0.2033 | 0.0426 | 0.0426 | 1.001 | — | — |
| 1651 | 0.0175 | 0.1812 | 0.2005 | 0.0572 | 0.0420 | 1.362 | 0.6700 | — |
| 1676 | 0.0212 | 0.2027 | 0.1989 | 0.0779 | 0.0415 | 1.879 | 0.8004 | +0.4934 |
| 1701 | 0.0423 | 0.2012 | 0.2014 | 0.0746 | 0.0434 | 1.719 | 0.6390 | +0.5738 |
| 1726 | 0.0169 | 0.2096 | 0.1988 | 0.0768 | 0.0422 | 1.818 | 0.7844 | +0.5494 |
| 1751 | 0.0505 | 0.2266 | 0.1964 | 0.0860 | 0.0412 | 2.090 | 0.8868 | +0.4341 |
| 1776 | 0.0707 | 0.2350 | 0.1943 | 0.0814 | 0.0403 | 2.018 | 0.8705 | +0.5107 |
| 1801 | 0.3883 | 0.2405 | 0.1900 | 0.0785 | 0.0393 | 1.997 | 0.9186 | +0.4886 |
| 1826 | 0.1810 | 0.2368 | 0.1876 | 0.0763 | 0.0385 | 1.981 | 0.8805 | +0.5237 |
| 1851 | 0.9289 | 0.2367 | 0.1860 | 0.0766 | 0.0379 | 2.020 | 0.8925 | +0.5301 |
| 1866 | 0.9606 | 0.2370 | 0.1811 | 0.0772 | 0.0367 | 2.103 | 0.9824 | +0.4668 |

## The arms

Comparison step 2000 for every arm. `harm` is the validation-CE rise above the arm's own
minimum, from the one table in `2026-08-24-tul-takeover-cure.md`. `gain` is the per-block
backward gain, the measure being replaced.

| arm | harm | gain | slow/fast | coh core | coh ratio | dW share |
|---|---:|---:|---:|---:|---:|---:|
| `a35-ctrl` | 1.186 | 2.772 | 0.1037 | 0.1038 | 54.12 | 0.9917 |
| `b10-ctrl` | 0.533 | 2.445 | 0.1190 | 0.0862 | 12.61 | 0.9633 |
| `b10-bptt2` | 0.192 | 2.784 | 0.1776 | 0.0814 | 7.25 | 0.9344 |
| `s0-stack` | 0.148 | 2.838 | 0.1952 | 0.0900 | 18.63 | 0.9919 |
| `s0-slotembed` | 0.119 | 2.501 | 0.3130 | 0.0503 | 0.76 | 0.9125 |
| `cure-a1r-ctrl` | 0.015 | 1.045 | 0.0778 | 0.0629 | 0.99 | 0.5549 |
| `b10-slotembed` | 0.000 | 1.052 | 0.2274 | 0.0175 | 0.27 | 0.6765 |

Rank correlation against harm, n = 7:

| candidate severity measure | spearman |
|---|---:|
| per-block gain (incumbent) | +0.536 |
| core slow/fast | **−0.393** |
| core coherence | **+0.857** |
| core/noncore coherence | **+0.857** |
| core share of dW | +0.750 |

Pooled over all 34 checkpoints, `spearman(core/noncore coherence, pre-clip core share) =
+0.861`.

The three spectral arms carry a core-local penalty, so their gradients — and therefore `m2`
and `nu` — contain the regulariser as well as the model. Scored separately, n = 3, which is
too small to conclude from and is reported because leaving it out would be selection:

| arm | harm | gain | slow/fast | coh core | coh ratio |
|---|---:|---:|---:|---:|---:|
| `a35-spec` | 2.737 | 2.038 | 0.1124 | 0.0893 | 10.62 |
| `a35-proj15` | 3.496 | 1.659 | 0.0949 | 0.0917 | 14.28 |
| `a35-proj15attn` | 1.108 | 1.828 | 0.2427 | 0.1008 | 2.80 |

Spearman there: gain −0.500, slow/fast −1.000, core coherence −0.500, **core/noncore
coherence +1.000**, dW share −0.500.

## Prediction scorecard

**One of eight holds.**

| # | prediction | result | measured |
|---|---|---|---|
| P1 | core slow channel below the fast one at healthy rungs | **HOLDS** | 0.175, 0.201, 0.227 |
| P2 | core slow channel above the fast one at sick rungs, rising monotonically | **FAILS** | 0.241, 0.237, 0.237 — never above 1, and not monotone |
| P3 | core rise at least 1.5x the non-core rise, based at 1700 | **FAILS** | 1.310. Based at 1625 it is 1.520, which is not what was written |
| P4 | core coherence below 0.15 at 1700 and above 0.25 at 1866 | **FAILS** | 0.0746 then 0.0772. Scored under Amendment 1 |
| P5 | coherent drift at least 3x its value at the 1700 interval | **FAILS** | 0.968x. Whole-ladder spread only 1.577x |
| P6 | autocorrelation below 0.15 healthy, above 0.40 sick | **FAILS** | +0.4341 to +0.5738 at EVERY rung |
| P7 | `b10-slotembed` < `b10-bptt2` < `b10-ctrl` on the slow channel | **FAILS** | 0.2274, 0.1776, 0.1190 — the exact reverse |
| P8 | the new measure does not invert on `bptt_depth 2` | **FAILS** | 0.1776 against the control's 0.1190. It inverts |

P8 was declared decisive before the run: "if P8 fails the new measure is no better than the
one it replaces, and this experiment is a failure whatever P1 to P7 do." P8 failed.

## Verdict: failure

**Q1 is answered, and the answer is no.** The slow accumulator is not the motor. Across the
whole onset it stays between 17 % and 24 % of the fast channel on the core, it never
approaches parity, and across the arms it is **anti-correlated** with harm (spearman
−0.393): the healthiest arm carries the LARGEST slow-channel share, because the arms that
took over have large core gradients and therefore a large `nu` to divide by. The hypothesis
that `alpha * m_slow` explains why two unrelated fixes are both seed-dependent is refuted,
and that line is closed.

**Q2 is answered, and the parent document's proposal does not work.** `||dW_core||` times
its directional autocorrelation fails on both factors. `clip_grad_norm_` rescales every
gradient to a fixed global norm, so `||dW_core||` is close to a constant of the schedule.
And the autocorrelation sits at +0.43 to +0.57 at every rung, healthy and sick alike,
because at 25-step spacing consecutive displacements are correlated in any run. The
composite spreads only 1.577x across an onset that moves the core share from 0.017 to 0.961.

## Updated hypothesis

What did separate is **gradient coherence**, `RMS(m2) / RMS_ema(g)`, and its core-to-non-core
ratio. On the ladder that ratio goes from 1.001 — the core and the rest of the model exactly
equally coherent — to 2.103, and it does so EARLY: it reaches 2.090 at step 1751, when the
core share is still 0.0505, roughly 90 steps before the share crosses 0.5. It then
plateaus. It ranks the seven clean arms at spearman +0.857 against the incumbent's +0.536,
it does not invert on `bptt_depth 2`, and it is the only candidate of five that also ranks
the three penalised arms in the right order.

New hypothesis, for the next experiment: **the takeover begins when the core's gradients
become directionally persistent relative to the rest of the model, and the core/non-core
coherence ratio detects this about 90 steps before the pre-clip core share does.** The
prediction that would falsify it: an arm whose coherence ratio crosses 2.0 and which does
NOT take over within the following 200 steps.

**This measure was NOT pre-registered.** It was chosen from five candidates after seeing the
data, the pre-registered candidate came out anti-correlated, and n = 7 arms with spearman
+0.857 sits just past the n = 7 one-tailed 5 % critical value of +0.714. It is a
hypothesis, not a result, until it is tested on a run it was not fitted to.

## What this does NOT show

* **Correlation only, on both answers.** Nothing here shows that suppressing coherence would
  prevent a takeover. That needs an arm, and an arm needs the GPU.
* **The coherence measure is post-hoc and untested out of sample.** Five candidates were
  compared on the same seven arms that suggested it.
* **The penalised-arm result rests on three points.** A spearman of +1.000 on n = 3 has a
  one-tailed p of 1/6. It is reported to avoid selection, not because it decides anything.
* **`m2` and `nu` are blockwise-8bit above 4096 elements.** The round trip through the
  optimizer's own quantiser has a relative residual of 3.3e-03, which is the precision floor
  under every number here.
* **The arms are compared at step 2000 only**, because that is the first checkpoint they
  all have. Three of the seven had already taken over by then, so part of the ranking is
  reading a state rather than predicting one. Only the ladder has resolution inside the
  onset, and the ladder is one run.
* **`ac` is spacing-dependent and the two spacings are not comparable.** 25-step spacing on
  the ladder gives +0.43 to +0.57 everywhere; 2000-step spacing on the arms gives +0.0036
  for the held arm against +0.083 and +0.113 for the arm that took over. That separation is
  real but it is measured at a spacing the ladder cannot match.

## Incidental verification

28 parameters carry no optimizer state at all after 1850 steps: `W_IQ`, `compressor.B_a`,
`compressor.W_aKV` and `compressor.W_aZ` in each of the 7 CSA blocks. All have
`requires_grad=True`. This is **by design and already documented** in
`morph/model/attention.py`: the indexer's scores feed only `topk` indices, the values are
discarded, and the projections are batched under `torch.no_grad()` so the parameters keep
grad=None rather than a zero tensor — a zero-grad parameter would still take weight decay.
The probe confirms the intent holds in a real 1850-step run. Not a defect.
