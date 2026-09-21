# Planned: LXTUL-P rung P1 — the K streams as K SAMPLES (`tul.fan_seed_noise`)

Status: planned

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
Part 2 change 2 and the ladder's rung P1. Partner: `slot-spandec-strict-fan4-all`
([`successes/2026-09-20-lxtul-fan4-all.md`](../successes/2026-09-20-lxtul-fan4-all.md)).
Queued behind `fan4-all-fp0` ([`2026-09-21-lxtul-fan4-all-fp0.md`](2026-09-21-lxtul-fan4-all-fp0.md))
and `fan4-all-trig` ([`2026-09-21-lxtul-fan4-all-trig.md`](2026-09-21-lxtul-fan4-all-trig.md)).

## Question

fan4-all keeps its four streams apart with a trained within-slot volume term (`epivol`,
lambda 0.1, passes 1-2). That term is the one diversity term measured NOT to be gamed
(centred rank 2.820 at pass 1, four live input-dependent streams) and the reader still
cashes almost nothing: oracle − mixed 0.018, deployed − oracle 0.040, and the mixture
probe closed the selector question — no branch-blind selector recovers the oracle's gap,
so the lever left is BETTER CANDIDATES. If the streams are instead K SAMPLES — an
independent N(0, 1) draw at each stream's seed, no diversity term at all — do they stay
apart, and does the reader cash more of the oracle's gap than it does under the term?

## Hypothesis

A term the streams are TRAINED to satisfy gives the K states a reason to be four
SEPARATED states and no reason to be four PLAUSIBLE continuations; noise has nothing to
game, so what survives the loop is whatever the map and the winner-takes-all loss find
useful. Against it, stated first: a contractive shared map decays the seed difference like
`gain^(2T)` (PLR Thm 4.4), the very fact the volume term was built for, and noise is
exactly "stream identity in the initial condition only" — the thing `fan_trigger_every_pass`
exists because the map can erase. So the honest prior is that the rank FALLS and the
question is whether the reader loses anything when it does.

## Method

`morph/configs/tul_slot_spandec_strict_fan4_all_noise.yaml`, one MECHANISM over
`tul_slot_spandec_strict_fan4_all` and therefore two keys that move together:
`tul.fan_seed_noise: 1.0` and `tul.fan_repel_lambda: 0.0`. Neither alone is the arm — the
noise REPLACES the term. `fan_repel_mode: epivol` is inherited unchanged so the
instruments (`fan/vol_t{t}`, `fan/epi_t{t}`, `fan/stream_cos_t{t}`) keep reporting on the
same axes; only the CHARGE is gone.

Build: `morph/model/tul.py` (`fan_seed_noise` + refusals), `morph/model/transformer.py`
(one `torch.randn` per forward, `[B, S·M, C]` fp32, times the std, masked by `slot_valid`,
added to `h = core_init(e)` through `_apply_injection` BEFORE the trajectory is seeded;
train AND eval; `e` untouched), `morph/training/tul_setup.py` (key, mapping, manifest,
banner), `tests/test_tul_fan_seed_noise.py` (12 tests: off consumes no RNG, per-channel
RMS 1.0 / pairwise sqrt(2) / deviation sqrt(3/4), linear in the std, the source `e` is
bit-identical, eval draws fresh, pads zero, refusals, compose, the instruments still
report at lambda 0). The scale rule: the seed lives in the `input_norm`'d field, so std
1.0 is unit per-channel RMS there.

Same seed, packer, rows, 5,000 steps, batch 6, the runner into
`/home/wolfe/morph-scratch/arc/results/2026-09-19-lxtul-fan4/`. Readouts as the fp0
prereg: `fan/*_final`, the 3070 stream probe at 5000 depth 6, the paired depth-6 read
against fan4-all's `tokens.npz`, and the mixture probe on the same 48 rows.

## Predictions (frozen)

fan4-all's finals for reference: `fan/oracle_ce` 4.3925, `fan/mixed_ce` 4.4348 (oracle −
mixed 0.018 on the probe's 48 rows: 4.2082 / 4.2485 deployed), `fan/stream_rank_t1` 2.820,
`_t6` 2.060, tokens K1−K6 +0.0049, RATE 7,096 tok/s, `loop/core_gain_t0` max 9.92. Seed
spread 0.024 nats. Probabilities are the builder's, reviewed by the orchestrator 2026-09-21 03:44 and adopted unchanged before any GPU step.

- **P-1 (stable).** 5,000 steps, tripwire HEALTHY (`preclip/total` under 1e4 at every step
  ≥ 200). **80 %.** A unit-RMS perturbation of the entry state is large — as large as the
  prelude's own output — and the gain hinge was tuned on prelude-shaped entries. The
  1000-step ramp and `slot_gain_lambda` 100 @ 0.9 are the holds.
- **P-2 (the streams stay apart at pass 1).** `fan/stream_rank_t1` above **2.30**.
  **60 %.** The seeds are drawn orthogonal-ish in 1024 dims, so pass 1 starts near rank 3;
  the risk is that pass 1 is where the contraction is strongest (the map's first
  application sets the scale, `morph-loop-is-a-power-iteration`).
- **P-3 (THE RUNG'S REASON: the reader cashes more).** Paired depth-6 reader CE,
  noise − fan4-all, **below −0.024** (better by more than the seed spread). **30 %.**
  This is the rung's falsifier and the prior is against it: nothing in P1 gives a pass a
  job, which is the note's own diagnosis of why every fan arm reads flat.
- **P-4 (the loop stays flat).** Tokens K1−K6 below **+0.010**. **80 %.** Sampling is not
  a job for a pass either. If this FAILS upward, read it against the note's theatre
  warning: a noisy entry makes early passes look productive because they are denoising
  their own entry, not because depth earns.
- **P-5 (the oracle gap does not shrink).** `fan/oracle_ce` − `fan/mixed_ce` within
  **±0.010** of 0.018. **55 %.** A wider candidate set would WIDEN this (a better oracle
  with the same reader), so a rise here is informative and not a failure.
- **P-6 (rank falls through the loop at least as fast as under the term).**
  `fan/stream_rank_t6` below **2.30**. **70 %.** With no term opposing the contraction
  the decay is the map's alone.
- **P-7 (cost).** RATE within **5 %** of 7,096 tok/s and peak memory within 0.2 GB.
  **90 %.** One `randn` of `[6, 256, 1024]` per forward, and the volume term's log-det is
  still computed (instrument), so the compute is essentially unchanged.

## Binding

If **P-3 holds**: noise beats the term at the reader, the diversity-term lane closes, and
rung P2 (a per-pass denoising target) is built on the noise seeds rather than on the term.

If **P-3 fails and P-2 holds**: the streams are apart and the reader still cashes nothing,
which is the note's own reading — the missing thing is a JOB for a pass, not distinctness.
Go straight to P2; do not try a third diversity term (`fan-diversity-terms-get-gamed`).

If **P-2 fails** (the samples collapse by pass 1): the volume term was load-bearing for
the rank after all. Then the honest base for P2 is fan4-all WITH the term, and the noise
is an addition rather than a replacement — a two-factor arm, to be split.

If **P-1 fails**: std 1.0 is too large for the hinge. Next rung is std 0.3 at the same
seed, NOT `slot_gain_lambda 0`.

## Not verified before launch

- No GPU step of this arm. Wall clock, memory and the tripwire are all unmeasured; the
  runner's smoke is the first.
- The claim that the seed field has unit per-channel RMS is from the RMSNorm's definition
  and its init scale of 1.0. The LEARNED scale drifts during training, so at step 5,000
  "std 1.0 = unit RMS in that field" is an init-time statement, not a run-time one. The
  knob is a fixed std by design; nothing renormalises it as the field moves.
- `torch.compile` not exercised on the new branch (the tests are eager; the branch is a
  build-time float).
- The interaction with the terminal fixed-point term (`core_fixed_point_lambda` 1.0, still
  on here) is unmeasured: that term asks the last pass to stop moving, which on a noisy
  entry may simply be asking the map to finish denoising. rung P0 is the arm that reads it.
