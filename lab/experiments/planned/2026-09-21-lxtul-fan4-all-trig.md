# Planned: LXTUL fan4-all with the per-stream trigger re-injected every pass (`tul.fan_trigger_every_pass`)

Status: planned

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
Alternatives considered item 2 (the outside agent: "a trigger supplied to the core every pass
changes the map to `f_k(z, C) = f(z, C, e_k)`, which keeps distinct fixed points per stream
without a volume penalty"). Partner: `slot-spandec-strict-fan4-all`
([`successes/2026-09-20-lxtul-fan4-all.md`](../successes/2026-09-20-lxtul-fan4-all.md)).
Queued behind `fan4-all-fp0` ([`2026-09-21-lxtul-fan4-all-fp0.md`](2026-09-21-lxtul-fan4-all-fp0.md)).

## Question

Under a shared contractive map the distance between two streams decays like gain^(2T) when
the stream identity lives only in the initial condition. fan4-all's centred stream rank falls
2.820 (pass 1) → 2.060 (pass 6). If the per-stream trigger (`TULSlotRegister`: `W_o(pooled) +
P_cell`, the term that makes the K seeds differ) is ADDED to each stream's state at the start
of every pass after the first, does the rank hold through the loop, and does anything the
reader sees change?

**Correction to the note, found at build (2026-09-21):** the seed, trigger included, ALREADY
reaches every pass through `DiagonalInjection` (`_e_arg` is bound once and handed to every
pass; `tul.reinject_seed_every_pass` refuses as a no-op for that reason). At the shipped
`model.injection_channels: ctx` that route writes `dt · e` into the context-channel slice only,
decayed by `A < 1`. The new knob is a full-width, undecayed, unscaled add of the trigger alone,
so on this config it is a new route. The note's mechanistic sentence was already true of the
context channels; the arm tests whether the full-width route does what the decayed slice did
not.

## Hypothesis

Re-supplying the identity every pass opposes the decay directly, so the pass-6 rank rises
toward the pass-1 rank at no CE cost. Weaker than the split-map construction (Jacobian 1 on
the deviations): the state Jacobian stays that of the shared map, and the separation is
re-injected, not preserved. The outside agent's own caveat, adopted: it "could maintain four
useful hypotheses — or four recognisable identity offsets."

## Method

One factor over `tul_slot_spandec_strict_fan4_all`:
`morph/configs/tul_slot_spandec_strict_fan4_all_trig.yaml` sets `tul.fan_trigger_every_pass:
true` and nothing else (the fixed-point term stays at 1.0; the fp0 arm is the other factor).
Build: `morph/model/tul.py` (config + refusal at `fan_k == 0`), `morph/model/transformer.py`
(`_tul_core` adds the register's `[B, S·M, C]` output to `_h_in` at every pass `t > 0`, outside
`_core_step` so it is a saved input of the checkpointed call; train and eval alike),
`morph/training/tul_setup.py` (key, manifest, banner), `tests/test_tul_fan_trigger.py` (12
tests: default bit-identical, zero term at step 0, `post == pre + trigger` at every captured
pass ≥ 2, gradient reaches `W_o`/`P_cell` through later passes under checkpointing, refusal,
Hydra compose). Same seed, packer, rows, 5,000 steps, batch 6, the runner into
`/home/wolfe/morph-scratch/arc/results/2026-09-19-lxtul-fan4/`. Readouts as the fp0 prereg:
`fan/*_final`, the 3070 stream probe at 5000 depth 6, the paired depth-6 read against
fan4-all's `tokens.npz`, and the mixture probe (`lab/divergence/fan_mixture_probe.py`) on the
same 48 rows.

## Predictions (frozen)

fan4-all's finals for reference: `fan/oracle_ce` 4.3925, `fan/mixed_ce` 4.4348, regret 0.042,
`fan/stream_rank_t1` 2.820, `_t6` 2.060, tokens K1−K6 +0.0049, RATE 7,096 tok/s, peak 19.14 GB,
`loop/core_gain_t0` max 9.92. Mixture probe on fan4-all: deployed − prefix mixture −0.0098.
Seed spread 0.024 nats. Probabilities are the builder's.

- **P-1 (stable).** 5,000 steps, tripwire HEALTHY. **85 %.** A full-width undecayed add each
  pass raises the state's scale; the gain hinge and the ramp are the holds.
- **P-2 (THE ARM'S REASON: the rank holds through the loop).** `fan/stream_rank_t6` at 5,000
  above **2.30** (fan4-all 2.060) AND the 3070 probe's pass-6 centred rank above 2.30.
  **55 %.** Against it: the fan4-all Correction's collapse mechanism is the shared core's
  contraction plus the winner-takes-all gradient share, and a re-supplied offset the map can
  learn to cancel is not a preserved deviation.
- **P-3 (pass 1 unchanged).** `fan/stream_rank_t1` within **±0.15** of 2.820. **70 %.** The
  trigger's parameters now also take gradient through later passes, so the trigger itself may
  change.
- **P-4 (the loop stays flat).** Tokens K1−K6 below **+0.010**. **75 %.** Re-injection is not
  a job for a pass.
- **P-5 (no CE cost).** Paired depth-6 CE, trig − fan4-all, inside **[−0.024, +0.024]**.
  **65 %.** Residual: 20 % better than −0.024, 15 % worse.
- **P-6 (the read is unchanged).** `fan/mixed_ce − fan/oracle_ce` within **±0.010** of 0.042,
  AND the mixture probe's deployed − prefix mixture within ±0.005 of −0.0098. **65 %.**
- **P-7 (scale stays bounded).** `loop/core_gain_t0` max over the run at or below **20**
  (twice fan4-all's 9.92). **70 %.**

## Binding

If **P-2 and P-5 hold**: re-injection keeps the streams apart through the loop for free. With
the fp0 result, the ladder's base config becomes fan4-all + fp0 + trig (both one-factor
readings composed), and the note's Alternatives item 2 moves to "adopted as a base".

If **P-2 fails**: re-supplying the initial condition does not hold the rank, so the pass-6
collapse is the shared map's action on the deviations, not a decay of the seed. The
anisotropic gain probe (Part 1 probe 2) says which directions, and the split-map ablation
(contract the mean, rotate the deviations by an orthogonal `O(C)`) is the next build.

If **P-1 fails**: the full-width add detonates the scale. Next: the same term scaled by 0.5, or
routed through the existing injection on all channels (`model.injection_channels: all`, where
the two routes overlap and the arm reads as a gain on the shipped path).

If **P-5 lands in its 20 % tail**: re-injection is worth CE. Read the offset profile before
believing it (a stream identity the coda can key on is a width-like gain, not a loop gain).

## Not verified before launch

- No GPU step; the runner's smoke is the first. Memory and wall clock unmeasured (one broadcast
  add per pass; "noise" is a guess).
- `torch.compile` not exercised on the new branch (tests ran eager; the branch is a build-time
  bool).
- Pre-existing hole, not this arm's: `_tul_core_db1` / `_tul_core_db1_ladder` ignore the
  register entirely and would ignore this knob; the strict spandec arms never take those paths.
- The claim that the shipped injection reaches the context slice only at
  `injection_channels: ctx` is from reading `DiagonalInjection`, not from a measurement of the
  two routes' relative magnitude.
