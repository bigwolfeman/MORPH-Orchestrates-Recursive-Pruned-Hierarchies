# Planned: LXTUL-R Step 2 panel — the persistent carry (C1) and the history stream (C2) on the chain

Status: failure

Date: 2026-09-22, 10:31 (frozen before any GPU step of either arm; committed before the
runner lines are appended). Task list and rationale:
[`2026-09-22-lxtul-r-step2-speed-and-carry.md`](../../../.agents/notes/proposed/architecture/2026-09-22-lxtul-r-step2-speed-and-carry.md).
Baseline: Step 1b
([`failures/2026-09-22-lxtul-r-step1b.md`](../failures/2026-09-22-lxtul-r-step1b.md)):
token K1−K6 +0.0261 [+0.0251, +0.0273], K3−K6 +0.0057, paired depth-6 CE vs
`slot-spandec-strict-fan4-all-fp0` +0.0265 (depth 1 +0.0527), `fan/stream_rank_t1` 2.97,
6,335 tok/s at 1.03 steps/s.

## Question

On the one-slot-per-pass chain, does (C1) keeping what a cell read, once, at the loop exit
(no re-processing) or (C2) giving the relay its own stream (one history stream exempt
from the repulsion, three slot-local plan streams) raise the loop's token contribution
toward the far budget, without buying it with a larger depth-1 loss against fp0?

## Hypothesis

C1: the decay row (planted g = 2 kept 62 %) is in-cell decay after arrival; an exit-side
add of the accumulated read restores what arrived and gives every pass's read a direct
gradient from the coda, so the K-curve rises and the late passes keep their share. C2:
the relay and the diversity term fought inside the same four cells on Step 1b (rank 2.97
at pass 1, 1.49 at pass 6); a dedicated history stream carries without being repelled,
and the plan streams are then free to be candidates. Neither changes the reach, so the
depth-1 gap against fp0 stays near 0.053 on both.

## Method

Two arms, one factor each over `tul_slot_spandec_strict_fan4_all_reach1`, on the
S-merged code (S1–S3 merged with their numeric same-loss tests; S4 only if its GPU gate
passes; the exact commit is the one in the runner line and is printed by the readouts):

- C1 `tul_slot_spandec_strict_fan4_all_reach1_persist` (`tul.loop_carry: persist`), wandb
  `slot-spandec-strict-fan4-all-reach1-persist`.
- C2 `tul_slot_spandec_strict_fan4_all_reach1_hist1` (`tul.fan_history_streams: 1`), wandb
  `slot-spandec-strict-fan4-all-reach1-hist1`.

Runner kind `slot`, sweeps at 2500 and 5000, 480 rows, OUTDIR
`/home/wolfe/morph-scratch/arc/results/2026-09-22-lxtul-r-step2`. Readouts as Step 1b's
Method (K-curve, paired vs strict / fan4-all / fp0 / reachall at depths 6 and 1, fan
metrics with the raw-trajectory rank, worth, tripwire, rate), plus:

1. **S5 first.** A 48-step profile run of the UNCHANGED Step 1b config at the merged commit
   (`MORPH_PERF_REGIONS=1`), steps/s and region split beside the 971 ms baseline; the
   speed bar is in the task list (≥ 1.30 steps/s) and is not a prediction of this panel.
2. **The swap table** (I1, `hop_distance_probe.py --swap --swap-mode row`, 480 rows,
   depths 1, 2, 3, 6, hops 6) on the Spark at the training commit (commit and the
   `tg_seg` marker count printed first), for BOTH arms AND for the Step 1b checkpoint
   (its baseline row; the planted table is not run again).
3. C1 only: `carry/persist_ratio` and the raw `fan/stream_rank_t*` (the raw trajectory,
   the test contract) from wandb.
4. C2 only: `fan/plan_rank_t*` beside `fan/stream_rank_t*`.

## Predictions (frozen)

Shared bars. "K1−K6" is `ce_tokens` K1−K6 at 5000 on the 480-row sweep; "gap" is the
paired depth-6 CE minus fp0's on identical rows.

- **P-1 (compose, health).** Both arms start, `preclip/total` under 1e4 at every step ≥
  200, entry norm growth under 2x, no NaN. **80 %** each.
- **P-2 (the headline, C1).** C1 K1−K6 ≥ **+0.035** (above Step 1b's interval). **60 %.**
  ≥ +0.05: **35 %.** Point estimate +0.045.
- **P-3 (the headline, C2).** C2 K1−K6 ≥ **+0.035**. **45 %.** ≥ +0.05: **25 %.** Point
  estimate +0.035.
- **P-4 (shape, Wolfe's rule).** For each arm, K3−K6 / K1−K6 in **[0.06, 0.20]** (80–94 %
  of the gain by pass 3 at mean depth 6). C1 **60 %**, C2 **60 %** (Step 1b reads 0.22).
- **P-5 (the guard).** Gap vs fp0 at depth 6 at or below **+0.015** (closes at least 0.011
  of Step 1b's 0.0265): C1 **40 %**, C2 **35 %.** Depth-1 gap vs fp0 within
  [0.040, 0.065] on both (the reach is unchanged): **70 %** each.
- **P-6 (the swap table).** On Step 1b's checkpoint the swap benefit at g = 1, depth 6 is
  ≥ **0.10** nats with control CE within [4.2, 4.5] (the instrument has range at the
  operating point): **70 %.** C1's g = 2 kept fraction (benefit at depth 6 over benefit at
  depth 2) ≥ **0.9** where Step 1b's reads below 0.8: **50 %.** C2's g = 2 arrival
  benefit (depth 2) ≥ Step 1b's: **55 %.**
- **P-7 (candidates).** C1 raw `fan/stream_rank_t1` ≥ **2.5** of 4: **65 %.** C2
  `fan/plan_rank_t1` ≥ **2.0** of 3: **60 %.**
- **P-8 (rate).** Each arm within **3 %** of the S5-measured Step 1b rate at the same
  commit: **75 %** each.

## Binding

An arm PASSES when P-2 (or P-3) holds with P-4 and P-5's depth-1 clause; it is filed under
`successes/`. The passing arm with the larger K1−K6 (both passing: C3, the conjunction,
is queued) goes to C4 (20k, paired against fp0 at 20k). If neither passes but one reads
K1−K6 ≥ +0.035 with a worse gap, it is filed under `failures/` as depth dependence and
Wolfe decides. If both read below +0.035, both are filed under `failures/`, the
mechanism levers on this chain are closed, and the next note is the compute floor of
the 256-cell core (speed) plus the horizon (C4 on Step 1b itself). P-6's first clause
failing means the instrument is not read for P-6's other clauses and the planted table
stays the record.

## Not verified before launch

- No GPU step of either arm; the runner smoke is the first.
- S1–S4's speed effect is measured by S5 before the queue, not assumed.
- The swap instrument has run on the tiny fixture only; its dynamic range on a real
  checkpoint is P-6's first clause.
- C1's exit add and the fan's WTA winner replay compose in the tiny tests; the 5090
  memory of the composition is the smoke's reading.

## Results

Two arms on the runner, one after the other, each with the same smoke, sweeps at 2500 and
5000 (480 rows, forced depths 1, 2, 3, 6, 9, 12, 16), worth and state probes. Artifacts:
`../results/2026-09-22-lxtul-r-step2/` (sweeps, self-paired K-curves, pairings, wandb
summaries, worth, slot state, the swap tables). Intervals are the block bootstrap over
rows. The commits: C2 at 5b09669 (S1–S3 merged), C1 at 6bcbe89 (S1–S3 and C2 merged;
S4 landed after C1 was queued, so neither arm ran the fused conv). Wolfe's two calls during
the runs, recorded in the task-list note: the chain geometry is a measurement and not a
final design, and the swap instrument is a corruption whose numbers are read for
structure only.

### C1, `slot-spandec-strict-fan4-all-reach1-persist` (6bcbe89, run cekwk9q8, 12:55 to 14:37, 0.99 steps/s)

| reading | C1 persist | Step 1b | clause |
|---|---|---|---|
| startup config | `loop_carry persist` (the previous-slot-only capture through a second layer-0 attention call), `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`, fixed-point 0, renorm on; smoke exit 0 (loss 33.65) | — | P-1 |
| `preclip/total` max at step ≥ 200 | **39 at 746** (no late spikes) | 1,050 at 4276 | P-1 |
| `loop/in_norm_t0` 200 → 4999 (max) | 938 → 1,380, **1.47x** (max 1,943 at 3195, a 2.07x transient; the family's largest) | 1.29x | P-1 |
| `carry/persist_ratio` (RMS of the exit add over the raw exit, mean over valid cells) | 0.98 at step 0 and 0.98 at 4980: the add is one carrier-RMS on 98 % of valid cells, by construction | — | — |
| `carry/rms_t1` / `_t5` (the accumulator's own RMS) | 0.13 → 0.53 / 0.54 → 2.63 over the run | — | — |
| token CE, forced depth 1 / 2 / 3 / 6 | 4.3479 / 4.3366 / 4.3298 / 4.3245 | 4.3525 / 4.3423 / 4.3320 / 4.3263 | P-2 |
| **token K1−K6 at 5000** | **+0.0234 [+0.0222, +0.0246]**; K3−K6 +0.0052 [+0.0048, +0.0057]; ratio **0.222** | +0.0261; +0.0057; 0.22 | P-2, P-4 |
| token K1−K6 at 2500 | +0.0141 | +0.0168 | — |
| paired d6 vs fp0 (500 blocks) | **+0.0248 [+0.0220, +0.0274]**; d1 **+0.0481 [+0.0451, +0.0512]** | +0.0265; +0.0527 | P-5 |
| paired d6 vs strict (490) | −0.0278 [−0.0311, −0.0243]; d1 −0.0044 | −0.0259; +0.0002 | — |
| paired d6 vs fan4-all (500) | +0.0190; d1 +0.0424 | +0.0207; +0.0469 | — |
| paired d6 vs reachall (481) | +0.2270 [+0.2168, +0.2383] | +0.2286 | — |
| paired d6 vs Step 1b (500) | **−0.0018 [−0.0039, +0.0004]**; d1 −0.0045 | — | — |
| paired d6 vs C2 (500) | −0.0008 [−0.0032, +0.0015] | — | — |
| `fan/stream_rank_t1` / `_t6` (raw trajectory) | **2.95** / 1.73 | 2.97 / 1.49 | P-7 |
| `fan/oracle_ce` / `mixed` / `single` | 4.3722 / 4.4191 / 4.4700; oracle − mixed 0.047 (floor 0.04) | 4.3689 / 4.4202 / 4.4596 | — |
| rate | 6,106 tok/s at step 200 (0.96x Step 1b: the extra layer-0 attention call) | 6,335 | P-8 |
| final `val/loss` | 4.4232 (logged at 4750; runner's final 4.4646 at 4999) | 4.4246 | — |

Scoring: **P-1 holds** (tripwire 39, no spikes; entry-norm growth 1.47x at the end with a
2.07x transient at step 3195, named). **P-2 FAILS** (+0.023 against +0.035; inside Step
1b's band). **P-4 FAILS** (0.222). **P-5 FAILS** at depth 6 (+0.0248) and holds at depth 1
(0.048). **P-7 holds** (2.95 of 4; the raw trajectory is the `none` map's by the test
contract, and its rank is Step 1b's). **P-8 FAILS by 0.6 %** (6,106 against 6,335, the
second attention call). What the arm says: the persist term is live (0.98 of a carrier
RMS on every valid cell, the accumulator growing 5x over the run), it removes the late
spikes, and it leaves the depth-6 CE exactly where Step 1b had it (−0.002 paired) and the
K-curve 0.003 lower. Keeping what arrived, un-decayed, at the exit does not add to what
the coda can use: the reader already had it.

### C2, `slot-spandec-strict-fan4-all-reach1-hist1` (5b09669, run cn0l3vel, 10:56 to 12:35, 1.02 steps/s)

| reading | C2 hist1 | Step 1b | clause |
|---|---|---|---|
| startup config | `fan_history_streams 1`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`, `fan_repel_mode epivol`, fixed-point 0, renorm on; smoke exit 0 (loss 33.65) | — | P-1 |
| `preclip/total` max at step ≥ 200 | **48.7 at 220** (no late spikes) | 1,050 at 4276 | P-1 |
| `loop/in_norm_t0` 200 → 4999 (max) | 942 → 1,178, 1.25x (max 1,725 at 4441, the same step as Step 1b's max) | 1.29x | P-1 |
| token CE, forced depth 1 / 2 / 3 / 6 | 4.3457 / 4.3367 / 4.3299 / 4.3253 | 4.3525 / 4.3423 / 4.3320 / 4.3263 | P-3 |
| **token K1−K6 at 5000** | **+0.0204 [+0.0194, +0.0214]**; K3−K6 +0.0046 [+0.0042, +0.0050]; ratio **0.225** | +0.0261; +0.0057; 0.22 | P-3, P-4 |
| token K1−K6 at 2500 | +0.0139 | +0.0168 | — |
| paired d6 vs fp0 (500 blocks) | **+0.0255 [+0.0227, +0.0285]**; d1 **+0.0459 [+0.0430, +0.0490]** | +0.0265; +0.0527 | P-5 |
| paired d6 vs strict (490) | −0.0269; d1 −0.0066 | −0.0259; +0.0002 | — |
| paired d6 vs fan4-all (500) | +0.0197; d1 +0.0401 | +0.0207; +0.0469 | — |
| paired d6 vs reachall (481) | +0.2275 [+0.2173, +0.2384] | +0.2286 | — |
| paired d6 vs Step 1b (500) | **−0.0010 [−0.0032, +0.0011]**; d1 −0.0067 | — | — |
| `fan/stream_rank_t1` / `_t6` (all 4) | 2.65 / 1.04 | 2.97 / 1.49 | — |
| `fan/plan_rank_t1` / `_t6` (3 plan streams) | **1.99** / 1.17 | — | P-7 |
| `fan/oracle_ce` / `mixed` / `single` | 4.3705 / 4.4198 / 4.4549; oracle − mixed 0.049 (floor 0.04) | 4.3689 / 4.4202 / 4.4596 | — |
| rate | 6,294 tok/s at step 200 | 6,335 | P-8 |
| final `val/loss` | 4.4243 (logged at 4750; runner's final 4.4652 at 4999) | 4.4246 | — |
| channel worth (192 rows, cells zeroed) | 0.643 / 0.343 / 0.265 / 0.219 / 0.161 / 0.105 / 0.063 by offset bin | 0.643 / 0.327 / 0.260 / 0.213 / 0.162 / 0.106 / 0.068 | — |

Scoring: **P-1 holds** (and the family's first spike-free reach run). **P-3 FAILS** (+0.020 against
+0.035; below Step 1b's interval). **P-4 FAILS** (0.225). **P-5 FAILS** on the depth-6 clause
(+0.0255) and holds on the depth-1 clause (0.046). **P-7 reads 1.99 of 3** against 2.0
(fails by 0.005; the three plan streams sit at the rank the four streams had at pass 6 of
fan4-all). **P-8 holds** (6,294 against 6,335). What the arm says: giving the relay its own
stream and the diversity term the other three leaves the depth-6 CE exactly where Step 1b
had it (−0.001 paired) and moves 0.006 of the loop's gain from the passes to depth 1; the
repulsion and the relay were not fighting over the same capacity in a way that limited
either.

### The swap tables (I1, the Spark at 6bcbe89, `tg_seg` count 24), read for structure only

Benefit(g, d) = CE(span k's tokens | span k−g replaced by another row's text) − CE(original), 3,482
sites over 480 rows, no site skipped. Per Wolfe's call the swap is a CORRUPTION: its
benefits overstate content use and its kept fractions may be the later passes discounting
a mismatched span, so the rows are read for the exact zeros and the arrival pass, not as
delivered nats. Step 1b (`reach1`) and fp0:

| g | Step 1b d1 / d2 / d3 / d6 | fp0 d1 / d2 / d3 / d6 |
|---|---|---|
| 1 | 0.268 / 0.258 / 0.290 / 0.287 | 0.318 / 0.306 / 0.302 / 0.305 |
| 2 | 0.042 / 0.043 / 0.034 / 0.028 | 0.026 / 0.031 / 0.033 / 0.033 |
| 3 | **0.000** / 0.019 / 0.017 / 0.014 | 0.015 / 0.018 / 0.019 / 0.019 |
| 4 | **0.000 / 0.000** / 0.010 / 0.007 | 0.010 / 0.013 / 0.014 / 0.013 |
| 5 | **0.000 / 0.000 / 0.000** / 0.004 | 0.008 / 0.010 / 0.010 / 0.010 |
| 6 | **0.000 / 0.000 / 0.000** / 0.003 | 0.006 / 0.007 / 0.007 / 0.008 |

Controls: Step 1b 4.301 / 4.291 / 4.278 / 4.272; fp0 4.247 / 4.239 / 4.237 / 4.236. The
zeros are bit-exact and sit exactly where the chain forbids arrival (g spans back at pass
g − 1); fp0, whose coda reads every slot, has no staircase and no depth dependence. **P-6's
first clause holds** (0.287 at g = 1, depth 6, control inside [4.2, 4.5]); its other two
clauses are not scored, per the call above. The C2 and C1 tables run after this filing and
are appended as dated addenda when they land.

## Verdict

**Failure, both arms** (P-2 and P-3 fail; P-4 fails on both; P-5 fails at depth 6 on both;
P-1, P-7 and P-6's first clause hold; P-8 holds on C2 and fails by 0.6 % on C1). The
binding's branch is "both below +0.035: both filed under `failures/`, the mechanism levers
on this chain are closed". The reading in one line: Step 1b, C1 and C2 have the SAME
depth-6 CE (paired −0.002 and −0.001, inside their intervals) and K-curves inside 0.006
of each other. Neither keeping the arrived content un-decayed (C1) nor separating the
relay from the diversity term (C2) changes what the coda gets from the loop. The far
content the chain relays is worth what it is worth, and the decay the planted and swap
rows show is not the limit; the relay itself is.

Wolfe's reading, 2026-09-22 (13:40 to 14:00, recorded before either arm finished): the chain
geometry is a sliding window with a stride of one span per pass, intentional as a
measurement of whether forced passes earn, and not a final design. Reaching g spans back
costs g full passes of the core where one attention hop is free; every arm on the
geometry sits behind fp0 at every depth (the A1.8 guard). No C3, no C4 on this geometry.
The binding's "compute floor plus horizon on Step 1b" branch is therefore NOT taken; the
speed items of the task list stand on their own (S5 reading 2 follows this filing).

## Updated hypothesis

LXTUL-R answered its question and closes: passes earn when the geometry forces a relay
(+0.020 to +0.026, the largest slot-loop K-curves on the ledger, with 78 % of the gain by
pass 3), and the earning is a partial refund of a tax the geometry charges (0.05 nats at
depth 1 against a direct read), never a gain over it. Persistence and stream separation
do not move it, because the loop delivers a fixed amount per hop and the coda already
uses all of it. The open problem is unchanged and sharper: keep direct access (fp0's
coda) and give the passes a job that is not relay. The candidate on the table from the
JEPA-Anything summary (2026-09-22): a slot target split into K orthogonal factors with
private predictors, one factor per pass, under an EMA target encoder with per-coordinate
variance floors, which is also the standard fix for the code-target collapse LCTUL hit.

## Addendum 2026-09-22 (16:15): the C2 hist1 swap table, a leak check

The Spark at 6bcbe89 (`tg_seg` count 24), `hop_distance_probe.py --swap --swap-mode row`
on `hist1@5000`, 480 rows, 3,482 sites, 0 skipped
(`../results/2026-09-22-lxtul-r-step2/hop_swap_hist1_5000.{json,runlog.txt}`). Controls
4.2955 / 4.2864 / 4.2748 / 4.2708 at depths 1 / 2 / 3 / 6. Benefit = swapped − control;
Step 1b's row beside it:

| g | hist1 d1 / d2 / d3 / d6 | Step 1b d1 / d2 / d3 / d6 |
|---|---|---|
| 1 | 0.313 / 0.293 / 0.302 / 0.299 | 0.268 / 0.258 / 0.290 / 0.287 |
| 2 | 0.044 / 0.038 / 0.034 / 0.027 | 0.042 / 0.043 / 0.034 / 0.028 |
| 3 | **0.000** / 0.021 / 0.020 / 0.015 | **0.000** / 0.019 / 0.017 / 0.014 |
| 4 | **0.000 / 0.000** / 0.010 / 0.009 | **0.000 / 0.000** / 0.010 / 0.007 |
| 5 | **0.000 / 0.000 / 0.000** / 0.005 | **0.000 / 0.000 / 0.000** / 0.004 |
| 6 | **0.000 / 0.000 / 0.000** / 0.003 | **0.000 / 0.000 / 0.000** / 0.003 |

The zeros are bit-exact (the JSON's `benefit` fields read `0.0`) and sit where the chain
forbids arrival, so the history stream leaks nothing: one slot per pass, at core layer 0,
on the shipped model. Beyond the zeros the rows are read for nothing (a swapped span is a
corruption); noted only that the numbers match Step 1b's within 0.045 at g = 1 and 0.002
from g = 2 on. The C1 persist table follows as its own addendum.

## Addendum 2026-09-22 (17:40): the C1 persist swap table, a leak check

The Spark at 6bcbe89 (`tg_seg` count 24), `hop_distance_probe.py --swap --swap-mode row`
on `persist@5000`, 480 rows, 3,482 sites, 0 skipped
(`../results/2026-09-22-lxtul-r-step2/hop_swap_persist_5000.{json,runlog.txt}`).
Controls 4.3019 / 4.2875 / 4.2788 / 4.2727 at depths 1 / 2 / 3 / 6. Benefit = swapped −
control:

| g | persist d1 / d2 / d3 / d6 |
|---|---|
| 1 | 0.318 / 0.311 / 0.309 / 0.301 |
| 2 | 0.053 / 0.048 / 0.037 / 0.030 |
| 3 | **0.000** / 0.022 / 0.020 / 0.013 |
| 4 | **0.000 / 0.000** / 0.010 / 0.008 |
| 5 | **0.000 / 0.000 / 0.000** / 0.005 |
| 6 | **0.000 / 0.000 / 0.000** / 0.003 |

The zeros are bit-exact and sit where the chain forbids arrival: the persist accumulator
(the second layer-0 attention call on the register) leaks nothing across the one-slot-per-
pass boundary. The rows beyond the zeros are not read (a swapped span is a corruption);
they sit within 0.01 of Step 1b's and hist1's at every g from 2 on, which is consistent
with the three arms sharing one depth-6 CE. All three leak checks of the Step 2 panel are
now filed; the Spark's `SWAPRUN3 COMPLETE` closes the instrument's run.
