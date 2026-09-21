# Planned: LXTUL-P rung P1 — the K streams as K SAMPLES (`tul.fan_seed_noise`)

Status: failure

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

## Results

Run: 5,000 steps at `36a571d` on the 5090 through `run_recon.sh`, START 09:08 local
2026-09-21, DONE 10:35, tripwire HEALTHY (`preclip/total` max 55.7 at step 220), final
val_loss 4.4352, RATE 7,295 tok/s at step 200 (7,292 at 4800), peak 19.14 GB (fan4-all
19.14). wandb `rfysno84`. Artifacts in `../results/2026-09-19-lxtul-fan4/`:
`sweep_slot-spandec-strict-fan4-all-noise_{2500,5000}.json`, `worth_..._5000.json`,
`slot_state_..._5000.json`, `run_slot-spandec-strict-fan4-all-noise.txt`,
`wandb_series_noise.json`, `loop_norms_noise_vs_fan4-all.txt`, `fan_geom_noise_5000_d6.{json,txt}`
and `fan_mixture_noise_5000_d6.{json,txt}` (Spark, worktree `MORPH-0921` at `82868b4`,
48 rows = 2,573 slots, the same rows as fan4-all's probes), `paired_noise_5000.json`.

**Last-val keys (step 4,750; wandb).**

| key | noise | fan4-all |
|---|---|---|
| `fan/oracle_ce` | 4.3663 | 4.3925 |
| `fan/mixed_ce` | 4.4113 | 4.4348 |
| `fan/mixed_ce − fan/oracle_ce` | 0.045 | 0.042 |
| `fan/stream_rank_t1` / `_t6` | **1.542 / 1.323** (max over the run 2.873 / 2.776, both at step 500) | 2.820 / 2.060 |
| `val/fan_stream_cos_t1` / `_t6` | **+0.991 / +0.984** | −0.30 / +0.06 |
| `loop/core_gain_t0` max over the run | 1.74 (step 128); 1.08 at the end | 9.92 |

**The mechanism, read from the loop norms (`loop_norms_noise_vs_fan4-all.txt`).** The
entry state's norm (`loop/in_norm_t0`, the carrier the draw is added to) grows
**1,365 → 3,117 → 7,769 → 12,108 → 15,417** at steps 0 / 1250 / 2500 / 3750 / 4999 on this
arm, against 966 → 1,057 on fan4-all. The draw has a FIXED std (1.0 per channel in the
`input_norm`'d field, by design), so the model made it negligible by scaling the state
it is added to: an 11x larger carrier turns a unit-RMS sample into a 9 % perturbation.
`val/slot_norm_mean` (the read the coda sees, after the write) is unchanged (32 → 37 on
both arms). The prereg's own "Not verified" clause named this hole: "the LEARNED scale
drifts during training... nothing renormalises it."

**Stream probe at 5,000 (Spark, 2,573 slots).** Pass 0 (the seeded entry): cos **+0.994**,
rank 1.827, shared 0.998, norms **431 / 487 / 522 / 556** (fan4-all pass 0: cos +0.006,
rank 2.305, norms 18.9 / 12.3 / 10.7 / 13.0). Pass 1: cos +0.992, rank **1.456**, `++--`
74 %. Pass 6: cos +0.984, rank **1.263**, norms 481 / 560 / 606 / 646. The four streams
are one state at 15–20x fan4-all's scale, and the rank falls further through the loop.
In ABSOLUTE terms a 1 % deviation of a norm-500 state is as large as fan4-all's whole
pass-1 state; the reader normalises, so the relative reading is the one that counts.

**Mixture probe at 5,000 (Spark, same rows).** Oracle 4.2033, deployed 4.2450 (agreement
with the model's keys 4e-8 / 5e-8); deployed − prefix mixture **−0.0091 [−0.0112,
−0.0068]** (fan4-all −0.0098); deployed − oracle **+0.0417** (fan4-all +0.0404); the best
single stream 0.112 above the oracle (fan4-all 0.099); winner persistence 0.2708 [0.2486,
0.2944] vs chance 0.2525; shares 0.225 / 0.272 / 0.278 / 0.225. **Fan4-all's whole regret
structure is reproduced by four streams at cosine 0.99.** The oracle over K near-copies
reads 0.04 below the deployed and 0.11 below the best single, so at least that much of
every fan arm's "oracle gap" is a min-over-K on per-token noise, not candidate value.
Quote this floor beside every future `fan/oracle_ce` reading.

**Depth sweep (480 rows).** 5,000: d1 4.3118, d2 4.3081, d3 4.3074, d6 4.3073, d9 4.3075,
d12 4.3081, d16 4.3096; tokens K1−K6 **+0.0045 [+0.0040, +0.0050]**, K3−K6 +0.0001. 2,500:
K1−K6 +0.0024. fan4-all: +0.0049 / +0.0004.

**Paired against fan4-all (`paired_noise_5000.json`, 511,089 tokens, 500 blocks).** Depth
6: noise − fan4-all **+0.0017 [−0.0006, +0.0039]**; depth 1 +0.0062; depth 16 +0.0036.
Worth profile: zero 0.1922 (fan4-all 0.1955), shuffle 0.1950 (0.1879), wrong_seed 0.0874
(0.0949).

**Scoring.**

- **P-1: HOLDS.** 5,000 steps, HEALTHY.
- **P-2 (the streams stay apart): FAILS.** rank_t1 1.542 (wandb) / 1.456 (probe) against
  2.30. The samples were washed out by a scale escape, not by the map's contraction: the
  collapse is already there at pass 0 (cos 0.994).
- **P-3 (the rung's reason): FAILS.** +0.0017 [−0.0006, +0.0039] against −0.024. The reader
  cashes nothing more; the arm is fan4-all inside the seed spread.
- **P-4: HOLDS.** K1−K6 +0.0045.
- **P-5: FAILS as written, HOLDS against its own reference.** The clause says "within ±0.010
  of 0.018"; the prereg's reference line two paragraphs up gives fan4-all's regret as
  0.042 (4.4348 − 4.3925), and 0.018 was the epivol arm's number. The reading, 0.045, is
  within 0.003 of 0.042. The builder's clause carried the wrong constant; I score it
  against the reference the same file states, and say so here rather than silently.
- **P-6: HOLDS.** rank_t6 1.323 < 2.30.
- **P-7: HOLDS.** 7,295 tok/s within 5 % of 7,096; peak 19.14 GB, equal.

## Verdict

**Failure** (P-2 and P-3, the rung's reason, fail; 4 of 7 hold, one on its reference).
K independent unit-RMS draws at the entry do NOT give K distinct streams on this
architecture, because nothing charges the model for making the draw small relative to
the state and the carrier's scale is free: the entry norm grew 11x over training and the
four streams read as one (cos 0.99 at pass 0). Binding branch taken: "P-2 fails: the
volume term was load-bearing for the rank after all", so the honest base for rung P2 is
fan4-all WITH the term. Two readings survive the failure: (1) the fixed-std design is the
defect, and a draw scaled to the entry state's per-slot RMS (one scalar per slot shared
by the K draws, so their DIRECTIONS stay independent; the prereg's correlation worry was
about a per-pass rescale, not a one-time entry scale) is the next cut if a sampling rung
is wanted at all; (2) the oracle-over-streams instrument has a ~0.04 floor on near-copies,
which re-reads every fan arm's regret as mostly selection on noise. The lineage arm
composes this config and runs next by queue order; its P-2 base is gone, as its prereg
named, and it is read as "does the channel narrowing change anything on a collapsed
fan" only.

## Updated hypothesis

Distinctness at the exit needs a signal that (a) the model cannot remove by a rescale and
(b) something downstream is charged for losing. Noise at the entry has neither. The
denoiser rung (P2, `loop_denoise`) charges every pass against a target at ITS noise level
and builds the level into the loss, so the scale escape is closed there by construction
(the entry is a convex mix of a unit-RMS code and unit noise and the target is fixed); it
is the next rung to read, on the single-stream base.
