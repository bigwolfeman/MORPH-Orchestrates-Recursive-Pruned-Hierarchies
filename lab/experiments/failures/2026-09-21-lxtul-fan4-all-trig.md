# Planned: LXTUL fan4-all with the per-stream trigger re-injected every pass (`tul.fan_trigger_every_pass`)

Status: failure

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

## Results

Run: 5,000 steps at `5d8a599`'s tree on the 5090 through `run_recon.sh`, START 02:50 local
2026-09-21, DONE 04:16, tripwire HEALTHY (`preclip/total` max 37.1 at step 238; fan4-all
29.4), final val_loss 4.4427, RATE 7,284 tok/s at step 200 (7,301 at 4800; fan4-all 7,080),
peak 19.30 GB (fan4-all 19.14). Artifacts in `../results/2026-09-19-lxtul-fan4/`:
`sweep_slot-spandec-strict-fan4-all-trig_{2500,5000}.json`, `worth_..._5000.json`,
`slot_state_..._5000.json`, `run_slot-spandec-strict-fan4-all-trig.txt`,
`fan_geom_trig_5000_d6.{json,txt}` and `fan_mixture_trig_5000_d6.{json,txt}` (Spark, worktree
`MORPH-0921` at `82868b4`, 48 rows = 2,573 slots, the same rows as fan4-all's probes; the
3070 was encoding the SONAR cache), `paired_trig_5000.json`, `wandb_series_trig.json`
(wandb `a8lermim`).

**Last-val keys (step 4,750; wandb).**

| key | trig | fan4-all |
|---|---|---|
| `fan/oracle_ce` | 4.3611 | 4.3925 |
| `fan/mixed_ce` | 4.4023 | 4.4348 |
| `fan/mixed_ce − fan/oracle_ce` | 0.041 | 0.042 |
| `fan/stream_rank_t1` / `_t6` | 2.788 / **1.470** (min over the run 1.433; the max, 2.50, at step 1000) | 2.820 / 2.060 |
| `val/fan_stream_cos_t1` / `_t6` | −0.31 / +0.20 | −0.30 / +0.06 |
| `loop/core_gain_t0` max over the run | 10.31 (step 4799) | 9.92 |

**Stream probe at 5,000 (Spark, 2,573 slots).** Pass 1: rank 2.816, cos −0.31, `axis_cos`
0.166, shared 0.16, norms 14.7 / 13.3 / 11.3 / 11.7, sign family `++--` 60 %. Pass 3: rank
2.167, `+---` 84 %. Pass 6: rank **1.425**, cos +0.27, `axis_cos` 0.381, shared 0.78, norms
**63.7 / 21.6 / 15.3 / 15.5**, `+---` **95 %**. fan4-all at pass 6: rank 2.026, norms 56.0 /
26.2 / 18.1 / 24.7, `+---` 95 %. The same one-live-stream shape as fan4-all, reached two
passes earlier and further: with the trigger re-supplied every pass, stream 0 ends 4.1x its
siblings (fan4-all 2.1x). The pass-1 norms are half fan4-all's (the trigger's parameters
now take gradient through six passes and shrank).

**Mixture probe at 5,000 (Spark, same rows).** Oracle 4.1951, deployed 4.2352 (agreement
with the model's keys 3e-10 / 5e-8); deployed − prefix mixture **−0.0107 [−0.0132,
−0.0081]** (fan4-all −0.0098); deployed − oracle +0.0402 (fan4-all +0.0404); winner
persistence 0.3067 [0.2892, 0.3254] vs chance 0.2811 (fan4-all 0.3039 vs 0.2676); winner
shares 0.403 / 0.194 / 0.202 / 0.202 (fan4-all 0.354 / 0.259 / 0.203 / 0.183).

**Depth sweep (480 rows).** 5,000: d1 4.3094, d2 4.3043, d3 4.3020, d6 4.3005, d9 4.3009,
d12 4.3017, d16 4.3033; tokens K1−K6 **+0.0089 [+0.0082, +0.0096]**, K3−K6 +0.0016
[+0.0013, +0.0018]. 2,500: d1 4.6149, d6 4.6086 (K1−K6 +0.0063). fan4-all 5,000: K1−K6
+0.0049, K3−K6 +0.0004.

**Paired against fan4-all (`paired_trig_5000.json`, 511,089 tokens, 500 blocks).** Depth 6:
trig − fan4-all **−0.0051 [−0.0073, −0.0030]**; depth 1: +0.0038 [+0.0015, +0.0059];
depths 2–16 between −0.0013 and −0.0051. Worth profile at 5,000: zero 0.1978 (fan4-all
0.1955), shuffle 0.1798 (0.1879), wrong_seed 0.0764 (0.0949). Slot-state probe: the exit
moves 0.324 of its norm from depth 1 to 6 (fan4-all 0.328), cos 0.968 (0.970).

**Scoring.**

- **P-1: HOLDS.** 5,000 steps, HEALTHY, pre-clip max 37.1.
- **P-2 (the arm's reason): FAILS.** rank_t6 1.470 (wandb) and 1.425 (probe) against 2.30,
  BELOW fan4-all's 2.06. Re-supplying identity every pass collapsed the streams harder.
- **P-3: HOLDS.** rank_t1 2.788 (probe 2.816), within 0.15 of 2.820.
- **P-4: HOLDS.** K1−K6 +0.0089 [+0.0082, +0.0096] under +0.010; K3−K6 +0.0016.
- **P-5: HOLDS.** −0.0051 [−0.0073, −0.0030], inside ±0.024.
- **P-6: HOLDS.** Regret 0.041 within 0.010 of 0.042; deployed − prefix mixture −0.0107,
  within 0.005 of −0.0098.
- **P-7: HOLDS.** `loop/core_gain_t0` max 10.31 ≤ 20.

## Verdict

**Failure** (the arm's reason failed; 6 of 7 hold). A full-width, undecayed re-supply of
the per-stream trigger at the start of every pass does not preserve the K deviations
through the loop: the exit rank falls to 1.43 of 3 (fan4-all 2.03), one stream ends at
4.1x its siblings, and the `+---` family reaches 95 % by pass 6 as on fan4-all but from
pass 3. The prereg's counter-mechanism is the reading: an offset the shared map sees every
pass is one it can learn to cancel, and the winner-takes-all share then does the rest. The
read is unchanged (regret 0.041, mixture −0.011, persistence +0.026 over chance), the loop
is flat past pass 3, and the coda reads the exit 0.005 better, inside the seed spread.
Binding branch: the note's alternative 2 (re-supply identity to the map) closes. With fp0
also filed, both Part-1 levers that act on the STATE are closed; what is left is the
ladder's own claim, that a pass needs a JOB (P1 noise, then P2 denoise, queued).

## Updated hypothesis

Stream identity cannot be kept by supplying it, whether once (fan4-all) or every pass
(trig), because nothing after pass 2 is charged for losing it and the winner-takes-all
gradient rewards one stream. The rank at the exit will follow the OBJECTIVE, not the entry:
K exits stay distinct only if K distinct things are asked of them (a per-stream target, or
a per-pass job). The `+---` family's 95 % at pass 6 on three arms in a row (fan4-all, trig,
and fan4-epi's one-live-stream shape) is the winner-takes-all signature and the next
instrument to read on the noise arm is whether it appears there too with no diversity term
at all.
