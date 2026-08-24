# Experiment: which channel of the AdEMAMix update drives the core at onset

Status: planned

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
