# Planned: LXTUL fan4-all with the terminal fixed-point term OFF (rung P0)

Status: failure

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

## Results

Run: 5,000 steps at `6d71e6c`'s tree (config commit) on the 5090 through `run_recon.sh`,
START 00:57 local 2026-09-21, DONE 02:33, tripwire HEALTHY (`preclip/total` max **522 at
step 239**; fan4-all's max was 29.4 at 219: without the term the early pre-clip is 18x
larger and still a factor 19 under the 1e4 tripwire), final val_loss 4.4368, RATE 5,675
tok/s at step 200 (7,287 at 3000, 7,469 at 4800; fan4-all 7,080 at 4800). Artifacts in
`../results/2026-09-19-lxtul-fan4/`: `sweep_slot-spandec-strict-fan4-all-fp0_{2500,5000}.json`,
`worth_..._5000.json`, `slot_state_..._5000.json`, `run_slot-spandec-strict-fan4-all-fp0.txt`,
`fan_geom_fp0_5000_d6.{json,txt}` (Spark stream probe from the `MORPH-0921` worktree at
`82868b4`, 48 rows = 2,573 slots, the SAME slot count as fan4-all's probe), `paired_fp0_5000.json`,
`wandb_series_fp0.json` (wandb `tg6n9rb1`'s series behind the finals below). The 3070 was
encoding the SONAR cache, so the probe ran on the Spark; the prereg named the 3070 and the
instrument, rows and depth are the same.

**Last-val keys (step 4,750; wandb).**

| key | fp0 | fan4-all |
|---|---|---|
| `fan/oracle_ce` | 4.3580 | 4.3925 |
| `fan/mixed_ce` | 4.4012 | 4.4348 |
| `fan/mixed_ce − fan/oracle_ce` | 0.043 | 0.042 |
| `fan/single_ce` | 4.4677 | (best single 4.4911) |
| `fan/stream_rank_t1` / `_t6` | 2.863 / 2.009 | 2.820 / 2.060 |
| `val/fan_stream_cos_t1` / `_t6` | −0.29 / +0.48 | −0.30 / +0.06 |
| `loop/core_gain_t0` max over the run | **2.81** (step-4999 value 2.61) | 9.92 |

**Stream probe at 5,000 (Spark, 2,573 slots).** Pass 1: rank 2.866, cos −0.30,
`axis_cos` 0.127, shared 0.20, norms 38.7 / 31.3 / 31.7 / 38.5, sign family `+--+` 54 %.
Pass 6: rank **1.999**, cos +0.58, `axis_cos` 0.409, shared 0.84, norms **91.4 / 111.6 /
132.1 / 86.9**, `+--+` 52 %. Against fan4-all at pass 6: rank 2.026, cos +0.09, shared
0.64, norms 56.0 / 26.2 / 18.1 / 24.7, `+---` 95 %. The rank lands in the same place by a
DIFFERENT shape: fan4-all collapses onto stream 0 (one live stream, the winner-takes-all
share), fp0 keeps four live streams that grow together (every norm 2.3–4.2x its pass-1
value) and pair up (`+--+`, cos +0.58, shared 0.84). The term was holding the exit's
SCALE, not the deviations' rank.

**Depth sweep (480 rows).** 5,000: d1 4.3100, d2 4.3029, d3 4.3006, d6 4.2998, d9 4.3007,
d12 4.3020, d16 4.3042; tokens K1−K6 **+0.0102 [+0.0096, +0.0109]**, K3−K6 +0.0008
[+0.0005, +0.0010]; span decoder K1−K6 +0.0330 [+0.0316, +0.0344]. fan4-all: K1−K6
+0.0049 [+0.0044, +0.0055], K3−K6 +0.0004. Twice fan4-all's pass-1 share, K3−K6 still
inside the slot-loop yardstick.

**Paired against fan4-all (`paired_fp0_5000.json`, 511,089 tokens, 500 blocks).** Depth 6:
fp0 − fan4-all **−0.0058 [−0.0080, −0.0035]**; depth 1: +0.0044 [+0.0022, +0.0068];
depths 2–16 between −0.0013 and −0.0058. Worth profile at 5,000: zero 0.1956 (fan4-all
0.1955), shuffle 0.1928 (0.1879), wrong_seed 0.0921 (0.0949).

**Scoring.**

- **P-1: HOLDS.** 5,000 steps, HEALTHY. The pre-clip max rose 29.4 → 522 without the term;
  the ramp held it.
- **P-2 (the arm's reason): FAILS.** rank_t6 2.009 (wandb) and 1.999 (probe) against 2.30.
  The term was not what contracts the deviations.
- **P-3: HOLDS.** rank_t1 2.863, within 0.15 of 2.820.
- **P-4: FAILS on its letter.** K1−K6 +0.0102 [+0.0096, +0.0109] against +0.010; the
  interval's lower end is 0.0096. K3−K6 +0.0008. The loop is as flat past pass 3 as every
  slot arm; the extra 0.005 is pass 1's.
- **P-5: HOLDS.** −0.0058 [−0.0080, −0.0035], inside ±0.024 and clear of zero on the better
  side by a fifth of the seed spread.
- **P-6: HOLDS.** Regret 0.043 against 0.042.
- **P-7: FAILS on its letter, by an artifact.** 5,675 at step 200 against 7,096. The
  builders' CPU test runs were on cores 0,1 at 01:00 (four agents); by step 3000 the arm
  ran 7,287 and by 4800 7,469, above fan4-all's 7,080 at the same step. The term's cost is
  real and small (about 5 %); the step-200 reading is contention.

## Verdict

**Failure** (the arm's reason, P-2, failed; 4 of 7 hold, one fails by artifact). The
terminal fixed-point term does NOT contract the K deviations: with it off the pass-6 rank
is unchanged (2.01 vs 2.06). What the term was doing on the fan is holding the exit's
SCALE: off, every stream's norm grows 2.3–4.2x through the loop and `loop/core_gain_t0`
falls from a 9.92 max to 2.81 (the reading is a ratio the growing state changes; it is
NOT a stabler loop), the early pre-clip is 18x larger, and the coda reads the larger,
paired exit 0.006 nats better with pass 1 doing twice the work. Binding branch taken:
"P-2 fails: the pass-6 collapse is the shared map, not the term". The gain probe
(`fan_gain_probe.py`, 354c639) then found the map contracts deviations and mean EQUALLY,
so the collapse is the DRIVE (what a pass adds), and the trigger-every-pass arm was next
(`2026-09-21-lxtul-fan4-all-trig.md`).

## Updated hypothesis

The K-stream rank at the exit is set by what the passes ADD (the winner-takes-all share
under fan4-all, a paired growth under fp0), not by a terminal penalty and not by the
Jacobian. A rung that wants K distinct exits has to give the passes K distinct jobs
(LXTUL-P P1–P2), not remove a hold. Every LXTUL-P rung still composes with the term ON
(the builders' configs keep 1.0 except the denoiser, which turns it off for its own
reason); this filing does not move that default, because the 0.006-nat CE gain is inside
the seed spread and the 18x pre-clip rise is a cost on the detonation margin.
