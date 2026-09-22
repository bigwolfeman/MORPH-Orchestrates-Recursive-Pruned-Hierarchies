# Planned: LXTUL-R Step 2 panel — the persistent carry (C1) and the history stream (C2) on the chain

Status: planned

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
