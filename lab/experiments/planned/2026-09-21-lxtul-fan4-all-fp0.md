# Planned: LXTUL fan4-all with the terminal fixed-point term OFF (rung P0)

Status: planned

Date: 2026-09-21 (frozen before any GPU step of the arm). Arc: the LXTUL-P note
[`2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md),
Part 1 probe 3 / ladder rung P0. Partner: `slot-spandec-strict-fan4-all`
([`successes/2026-09-20-lxtul-fan4-all.md`](../successes/2026-09-20-lxtul-fan4-all.md)).

## Question

`model.core_fixed_point_lambda 1.0` (base.yaml, every slot-loop arm) charges the LAST pass
of the loop for not being a fixed point of the map on the WHOLE slot state. On a K-stream
arm that state includes the K−1 deviation directions between streams, so the term asks the
streams to stop moving relative to each other at the end of the loop. On fan4-all the
centred stream rank falls from 2.820 at pass 1 to 2.060 at pass 6 (trainer keys
`fan/stream_rank_t1` / `_t6`; the 3070 probe read 2.03 at pass 6). Is the term part of
that fall, and does removing it cost anything?

## Hypothesis

The term contracts the deviations for free: with it off, the pass-6 rank rises, the pass-1
geometry (which the term does not touch) is unchanged, the loop stays flat, and the CE and
the selector regret stay inside the seed spread. The outside agent's qualification (note,
Alternatives 2): "a full-state fixed-point penalty would oppose nontrivial recurrent
rotations; I would restrict that penalty to the mean." Under the 1000-step LR ramp the term
is measured free against detonation (CLAUDE.md), so the arm is safe to run without it.

## Method

One factor over `tul_slot_spandec_strict_fan4_all`: `morph/configs/tul_slot_spandec_strict_fan4_all_fp0.yaml`
sets `model.core_fixed_point_lambda: 0.0` and nothing else. Same seed, packer, rows, 5,000
steps, batch 6, the runner (`run_recon.sh`: smoke, train, SWEEP@2500/5000, PROFILE,
STATEPROBE, WORTH) into `/home/wolfe/morph-scratch/arc/results/2026-09-19-lxtul-fan4/`.
Readouts: the `fan/*_final` keys from wandb; the 3070 stream probe
(`lab/divergence/fan_stream_probe.py`, 48 rows, depth 6, same rows as fan4-all's
`fan_geom_fan4-all_5000_d6.json`); the paired depth-6 read (`paired_vs_ruler.py`, 1024-token
blocks) against fan4-all's own `sweep_slot-spandec-strict-fan4-all_5000...tokens.npz`.

## Predictions (frozen)

Probabilities are the builder's. fan4-all's finals for reference: `fan/oracle_ce` 4.3925,
`fan/mixed_ce` 4.4348, regret 0.042, `fan/stream_rank_t1` 2.820, `_t6` 2.060, tokens K1−K6
+0.0049, RATE 7,096 tok/s, peak 19.14 GB, `loop/core_gain_t0` max 9.92. Seed spread on this
tree: 0.024 nats.

- **P-1 (stable without the term).** 5,000 steps, tripwire HEALTHY, no detonation. **90 %.**
  The ramp is the measured cure; the term was the warmup-0 hold.
- **P-2 (THE ARM'S REASON: the term was contracting the deviations).** `fan/stream_rank_t6`
  at 5,000 above **2.30** (fan4-all 2.060), AND the 3070 probe's pass-6 centred rank above
  2.30. **45 %.** Against it: the pass-6 collapse mechanism in the fan4-all Correction is the
  shared core plus winner-takes-all gradient share (stream 0 norm 25 → 56), which the term
  does not create.
- **P-3 (pass 1 untouched).** `fan/stream_rank_t1` at 5,000 within **±0.15** of 2.820.
  **75 %.** The term acts on the last pass only.
- **P-4 (the loop stays flat).** Tokens K1−K6 below **+0.010**. **80 %.** Rank was never the
  depth limit (the vq panel); removing a terminal term does not give a pass a job.
- **P-5 (no CE cost).** Paired depth-6 CE, fp0 − fan4-all, inside **[−0.024, +0.024]**.
  **70 %.** Residual: 20 % better than −0.024 (the term was costing CE), 10 % worse.
- **P-6 (regret unchanged).** `fan/mixed_ce − fan/oracle_ce` within **±0.010** of 0.042.
  **65 %.**
- **P-7 (it is cheaper).** RATE at step 200 at or above **7,096** tok/s (the term computes one
  extra core pass on the exit state). **75 %.**

## Binding

If **P-2 and P-5 hold**: the term contracted the deviations at no CE cost. Every LXTUL-P
rung composes from `fan4_all_fp0`, and the note's Risks gain a line: a whole-state
fixed-point term is not for K-stream arms.

If **P-2 fails** (rank_t6 stays near 2.06): the pass-6 collapse is the shared map, not the
term; the anisotropic gain probe (Part 1 probe 2) says which directions, and the
trigger-every-pass arm (`fan4_all_trig`, being built) is next.

If **P-1 fails**: the term is load-bearing on the fan under the ramp; keep 1.0 on every rung
and build the mean-only form (the penalty on the stream mean, not the deviations) as the
next arm.

If **P-5 lands in its 20 % tail** (fp0 better by more than 0.024): the term was costing the
reader; report it beside the vq and fan filings as a shipped-default finding and re-read the
strict partner with the term off before believing it (one-vector arms have no deviations, so
the mechanism would be different there).

## Not verified before launch

- No GPU step of the arm; the runner's 12-step smoke is the first.
- The claim "the term computes one extra core pass" is from the CLAUDE.md description of
  `core_fixed_point_lambda`, not from reading the code path in this session.
- Whether fan4-all's `fan_geom` probe rows and this arm's probe rows coincide depends on the
  probe reading the validation stream from its start (memory: probe seed does not draw rows);
  the token counts will be checked.
