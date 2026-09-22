# Planned: LXTUL-R Step 1b — the same arm at the corrected one-slot-per-pass geometry

Status: failure

Date: 2026-09-22 (frozen before any GPU step; committed before the runner line is
appended). Re-run of Step 1
([`2026-09-22-lxtul-r-step1.md`](2026-09-22-lxtul-r-step1.md), filed as a method fault:
the register's in-loop reach was one slot per pass through attention and about one slot
per LAYER through the CCA conv and the value shift, which the branch left unsegmented).
The fix (a per-slot `tg_seg` beside the relation whenever `loop_reach > 0`, reach 0
untouched) is the commit right before this file; `tests/test_lxtul_r_composition.py`
(11) asserts exact zeros beyond one slot per pass with the shipped conv.

## Question

With the leak closed, does the fan's K streams on the chain geometry (fan4-all +
`loop_reach 1` + `tg_coda_prefix_reach prev` + fixed-point 0 + renorm) earn a token
K-curve of the size the far budget allows, and do its passes recover the depth-1 cost
the geometry imposes?

## Hypothesis

Unchanged from Step 1, with one number updated by the leaky run: the geometry's depth-1
cost against fan4-all-fp0 was 0.040 nats with several slots relayed per pass; with one
slot per pass it will be larger, and the passes have strictly more to fetch. The renorm
held carried content at 90 % under the leak; the same mechanism should hold it now.

## Method

One arm, config `tul_slot_spandec_strict_fan4_all_reach1` unchanged (the fix is in the
model code, gated on `loop_reach > 0`), runner kind `slot`, sweeps at 2500 and 5000, 480
rows, OUTDIR `/home/wolfe/morph-scratch/arc/results/2026-09-22-lxtul-r-step1b`, wandb name
`slot-spandec-strict-fan4-all-reach1` (a new run id; the prior run stays as the leaky
record). Readouts and references exactly as Step 1's Method, plus:

8. The leak check on the real model: the planted probe's g = 3 row at depth 1
   (P-9 below), which the leaky arm read at 0.048.

## Predictions (frozen)

- **P-1 (A1.1, compose).** As Step 1. **95 %.**
- **P-2 (A1.2, healthy).** As Step 1. **75 %.**
- **P-3 (A1.3, the headline).** `ce_tokens K1-K6` at 5000 at or above **+0.063**.
  **25 %.** Point estimate +0.03. The leaky arm read +0.020 with less to fetch per pass;
  closing the leak raises what a pass can earn and what the geometry costs. Below +0.02:
  35 %. In [+0.02, +0.063): 40 %.
- **P-4 (A1.4, the decay).** Planted g = 2 at depth 6 keeps at least **75 %** of its
  depth-1 benefit. **55 %.** The renorm held 90 % under the leak.
- **P-5 (A1.5, CE parity).** Paired depth-6 CE vs `slot-spandec-strict` at or above
  **−0.02**. **55 %.** The leaky arm sat at −0.033; a tighter reach costs more.
- **P-6 (A1.6, candidates).** `fan/stream_rank_t1` ≥ **2.5** of 4; oracle − mixed quoted
  beside the 0.04 floor. **65 %.**
- **P-7 (A1.7, rate).** ≥ **5,677** tok/s. **70 %.** The segment reset is a mask on
  existing ops; the leaky arm ran at 7,496.
- **P-8 (A1.8, the guard).** Fraction `(0.2021 − gap_arm) / 0.1259` at or above **0.25**:
  **25 %.** At or below 0: 45 %. Depth-1 gap vs fp0 above 0.040: 70 %.
- **P-9 (the leak is closed on the real model).** Planted g = 3 at depth 1 inside
  **[−0.01, +0.01]**, and g = 3 at depth 2 above +0.02 (arrival one pass later, the
  chain's staircase). **85 %.**

## Binding

Step 1's binding, applied to this run: pass = P-1 to P-7; P-3 holds and P-4 fails → Step
2; P-3 fails with P-4 holding → the reach frame is closed for this loop and the
two-channel design is the next note; both fail → `failures/`, Wolfe decides; P-8 is
reported and qualifies a P-3 pass; P-9 fails → the run is a method fault again and
nothing else is read from it.

## Not verified before launch

- No GPU step of the corrected composition; the smoke is the first.
- The segment reset's interaction with the fan's winner replay (the coda's
  `checkpoint_blocks` path) is outside the core and untouched, but not re-measured.
- Whether the value shift is the only remaining per-layer route besides the conv: the
  tiny-model probe reads exactly 0 beyond one slot with both in place, which is the
  claim, but the probe is one row of one fixture.

## Results

One arm on the runner at 0b338f4 (model code 9b430d3), 2026-09-22 05:00 to 06:38 (smoke
38 s, 5,000 steps at 1.03 steps/s), readouts to about 07:05. Artifacts:
`../results/2026-09-22-lxtul-r-step1b/`; wandb run `u8xh91vd`. The leaky Step 1 run's
numbers are in the right-hand column for the same reading. Intervals are the block
bootstrap.

| reading | Step 1b (one slot per pass) | Step 1 (leaky, about four slots per pass) | clause |
|---|---|---|---|
| startup config (wandb) | `warmup 1000`, `core_fixed_point_lambda 0`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`, `slot_state_renorm True`; smoke exit 0 (loss 33.67) | same | P-1 |
| `preclip/total` max at step ≥ 200 | **1,050 at step 4276**, 431 at 3991, 53 at 220; verdict AMBIGUOUS (runner label), under the 1e4 tripwire | 42 at 2675 | P-2 |
| `loop/in_norm_t0` step 200 → 4999 (max) | 940 → 1,208, **1.29x** (max 1,622 at 4441) | 1.27x | P-2 |
| token CE, forced depth 1 / 2 / 3 / 6 / 9 / 12 / 16 | 4.3525 / 4.3423 / 4.3320 / 4.3263 / 4.3258 / 4.3257 / 4.3258 | 4.3394 / … / 4.3192 / 4.3195 / … | P-3 |
| **token K1−K6 at 5000** | **+0.0261 [+0.0251, +0.0273]**; K3−K6 **+0.0057 [+0.0053, +0.0062]** | +0.0202; +0.0025 | P-3 |
| token K1−K6 at 2500 | +0.0168 | +0.0163 | — |
| paired d6 vs `slot-spandec-strict` (490 blocks) | **−0.0259 [−0.0290, −0.0223]**; d1: +0.0002 | −0.0329; −0.0128 | P-5 |
| paired d6 vs `slot-spandec-strict-fan4-all` (500) | +0.0207 [+0.0180, +0.0233]; d1: +0.0469 | +0.0136; +0.0338 | — |
| paired d6 vs `slot-spandec-strict-fan4-all-fp0` (500) | **+0.0265 [+0.0239, +0.0290]**; d1: **+0.0527 [+0.0496, +0.0555]** | +0.0194; +0.0396 | P-8 |
| paired d6 vs `budget-web-reachall` (481) | +0.2286 [+0.2184, +0.2397]; d1: +0.2548 | +0.2215 | P-8 |
| A1.8 fraction `(0.2021 − gap) / 0.1259` | **−0.21** | −0.15 | P-8 |
| `fan/stream_rank_t1` / `_t6` | **2.973** / 1.494 | 2.975 / 1.728 | P-6 |
| `fan/oracle_ce` / `fan/mixed_ce` / `fan/single_ce` | 4.3689 / 4.4202 / 4.4596; oracle − mixed **0.051** (floor 0.04) | 4.3641 / 4.4118 / 4.4566; 0.048 | P-6 |
| rate | **6,335 tok/s** at step 200 (RATE OK), 6,330 through the run; 0.85x the leaky run, 0.89x fan4-all | 7,496 | P-7 |
| final `val/loss` | 4.4246 | 4.4163 | — |
| channel worth (cells zeroed, 192 rows; fan4-all 0.196) | **0.153** (bins 0.643 / 0.327 / 0.260 / 0.213 / 0.162 / 0.106 / 0.068) | 0.162 | — |

Scoring (P-4 and P-9 below, from the Spark):

- **P-1 HOLDS.**
- **P-2 HOLDS on its letter, with a record.** Tripwire under 1e4 at every step, but two
  spikes (1,050 and 431) late in the run that no arm of this family has shown (fan4-all
  29, the leaky run 42, the noise arm's 11x entry-norm growth had none). The entry norm
  grew 1.29x with a transient 1.7x at step 4441, near the larger spike. The segment
  reset removed the conv's smoothing across slot boundaries inside the loop; the
  `loop/core_gain_t0` stays 1.000 under the renorm, so the spikes are not the scale mode.
  Not a detonation; noted as the first instability signal on the fan family.
- **P-3 FAILS.** +0.0261 against +0.063. Above the leaky run (+0.0202) by 0.006, with the
  late passes doing twice as much (K3−K6 +0.0057 against +0.0025): with one slot per pass,
  more of the reach is fetched by passes 3 to 6. Still a third of the bar. The prereg's
  middle band ([+0.02, +0.063), 40 %).
- **P-5 HOLDS.** 0.026 better than strict at depth 6 (the leaky run: 0.033); at depth 1 the
  arm sits exactly at strict (+0.0002).
- **P-6 HOLDS.** Rank 2.97 of 4; oracle − mixed 0.051, 0.011 above the floor.
- **P-7 HOLDS.** 6,335 tok/s against the 5,677 bar; the segment reset costs 15 % of the
  leaky run's rate.
- **P-8: the fraction is −0.21** ("at or below 0", the 45 % branch); the depth-1 gap
  against fp0 is **0.053** (the "above 0.040" clause, 70 %, holds). The corrected geometry
  costs the fan 0.053 nats at depth 1 and the passes recover 0.026 by depth 6: the K-curve
  is depth dependence again, larger on both sides than under the leak.

### The planted decay row (A1.4) and the leak check (P-9), DGX Spark, 07:51 to 09:16 (the valid run)

`hop_distance_probe.py` on the 5000 checkpoint from the `MORPH-0922` worktree at 83f1ef8
(model code 9b430d3), 480 rows, batch 3, `--planted --planted-len 2 --skip-localiser
--hops 6`, depths 1, 2, 3, 6; 2,842 sites. A first run of this probe (06:41 to 07:50) went
out on the worktree still at bcb0dc2, the pre-fix code, so it evaluated a model trained
with segmented convs using unsegmented ones; it read a weak one-hop residual (g = 3 at
depth 1: 0.020, g = 4 at depth 2: 0.017) that was the probe host's code, not the model. It
was killed, its log kept on the Spark as `INVALID_prefix_code_*`, and the worktree
advanced before the run below. Benefit = control − planted, controls 15.509 / 15.526 / 15.530 / 15.532 at depths
1 / 2 / 3 / 6. The leaky Step 1 row and the single-cell
chain's row beside it (the chain: fixed-point term on, no renorm):

| g | 1b d1 | d2 | d3 | d6 | kept | leaky d1 | d6 | kept | chain d1 | d6 | kept |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.673 | +0.651 | +0.643 | +0.638 | 95 % | +0.549 | +0.526 | 96 % | +0.566 | +0.558 | 99 % |
| 1 | +0.134 | +0.097 | +0.099 | +0.093 | 70 % | +0.115 | +0.106 | 92 % | +0.181 | +0.137 | 76 % |
| 2 | +0.078 | +0.073 | +0.056 | +0.048 | 62 % | +0.051 | +0.046 | 90 % | +0.148 | +0.071 | 48 % |
| 3 | -0.000 | +0.048 | +0.039 | +0.031 | — | +0.048 | +0.033 | 69 % | +0.000 | +0.062 | — |
| 4 | -0.000 | +0.000 | +0.036 | +0.023 | — | +0.023 | +0.022 | 96 % | +0.000 | +0.047 | — |
| 5 | -0.000 | +0.000 | -0.000 | +0.020 | — | +0.000 | +0.023 | — | +0.000 | +0.034 | — |
| 6 | -0.000 | +0.000 | -0.000 | +0.018 | — | +0.000 | +0.017 | — | +0.000 | +0.021 | — |

- **P-9 HOLDS.** g = 3 reads exactly the control at depth 1 (0.000) and arrives at depth 2 (0.048); g = 4 is exact at depths 1 and 2 and arrives at depth 3 (0.036): the chain's staircase, arrival at pass g − 1 and exact zeros before it. The leak is closed on the real
  model; the run counts.
- **P-4 FAILS.** g = 2 arrives with 0.078 at depth 1 (the leaky run 0.051, the chain 0.148)
  and keeps **62 %** at depth 6 (0.048), against the 75 % bar; the chain kept 48 % with the
  fixed-point term on and no renorm, the leaky run 89 % with the conv re-supplying the
  copy at every layer. With one hop per pass the renorm slows the decay but does not stop
  it: it fixes a cell's norm, not the direction of what it carries.
- **The instrument's caveat (Wolfe, 2026-09-22, during this readout).** The planted ids are arbitrary
  subwords absent from the batch, so the control CE is 15.5 nats, 4.7 above uniform
  (10.80), and the own-span copy (g = 0) lifts the token only 1.8x (15.5 → 14.8): the
  probe is an exact-induction test on out-of-context pairs that these 5k-step models
  barely pass, on every arm. Its rows are ratios of small effects on a floor the model
  does not operate at. The same checkpoint reads 4.267 nats (ppl 71, top-1 29.5 %) on
  ordinary tokens over 48 rows against fan4-all-fp0's 4.236, with K1−K6 +0.0255 there
  against the sweep's +0.0261 (`real_ce_check.py`, run on the Spark). The decay row is
  filed as the probe's reading, not as the loop's carrying capacity; the natural-token
  localiser (0.24 nats one span back, 0.008 two back on the chain) is the instrument for
  that question and was not run on this arm.

## Verdict

**Failure** (P-3 and P-4 fail; P-1, P-2, P-5, P-6, P-7, P-9 hold; P-8 reads −0.21). The
prereg's binding for this pair is "file under `failures/`, no Step 2; Wolfe decides". What
the corrected arm says, in the design note's own terms:

- The reach frame does make the passes matter: the token K-curve is **+0.026**, the
  largest on the slot-loop ledger, and the late passes (K3−K6 +0.0057) do twice the work
  they did under the leak. It is a third of the bar the far budget set (+0.063).
- The geometry that forces the relay costs **0.053 nats at depth 1** against
  fan4-all-fp0 and the passes recover **0.026** of it: depth dependence, the guard the
  note wrote for itself (A1.8), on both runs.
- Carried content decays under one hop per pass (62 % kept on the planted probe, with its caveat), so the note's Step 2 (a
  persistent component written once on arrival) is the lever the decay row points at; the
  binding does not run it after a double failure, and Wolfe decides whether the +0.026
  and the 0.053 make it worth a run.
- The channel carries less under the chain (0.153 against 0.196), the streams stay
  candidates (rank 2.97), the rate holds (6,335 tok/s), and the first tripwire spikes of
  the fan family appeared late in this run (1,050 and 431, under the 1e4 tripwire).

## Updated hypothesis

With the leak closed, the chain geometry on the fan is a controlled trade: it takes 0.053
nats of direct read away from the coda and gives the loop the job of fetching it back one
slot per pass; the loop fetches half, and what it fetches decays at about the chain's rate
despite the renorm. Two limits show in the same run: a pass-relayed copy is worth less than
a direct read of the same content (planted g = 2: 0.078 relayed at arrival against the chain's
0.148 and the plain stack's every-layer read), and it loses about a third of its worth
over the next five passes. A design that wants the loop to earn the far budget needs BOTH a persistent carry
(Step 2) and a channel that does not pay for the relay with the coda's direct read (the
two-channel design). Neither alone closes the 0.053 gap on this evidence. The decision
between "Step 2 on this chain" and "the two-channel note" is Wolfe's, with the numbers
above.
