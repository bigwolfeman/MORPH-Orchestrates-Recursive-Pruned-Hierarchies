# Planned: LXTUL-R Step 1 — the fan's K streams on the chain geometry (fan4-all + loop reach 1 + coda prefix reach prev + fixed-point 0 + slot-state renorm)

Status: planned

Date: 2026-09-22 (frozen before any GPU step of the arm; committed before the runner line
is appended). Arc: Step 1 of the LXTUL-R design note
[`2026-09-21-lxtul-r-reach-composition.md`](../../../.agents/notes/proposed/architecture/2026-09-21-lxtul-r-reach-composition.md).
Step 0 ([`2026-09-21-span-reach-split.md`](2026-09-21-span-reach-split.md), to be filed
separately) fired the note's queue rule on the far budget's LOWER bound: CE(reach1) −
CE(reachall) = **+0.1259 [+0.1165, +0.1364]** at depth 6 on 481 paired blocks (threshold
0.10). The bound does not depend on the fourth arm (`budget-web-reach1-coda`), whose
ordering check failed and is under investigation in the Step 0 filing. Wolfe 2026-09-21:
"I think we do this ... use a subagent to build, verify its work, then run."

## Question

When the coda reads only the previous slot's cells (`tg_coda_prefix_reach: prev`) and each
cell in the loop reads only its own slot and the previous slot (`tul.loop_reach: 1`), the
loop is the ONLY route for content more than one span back, one slot per pass. On the
single-cell strict loop that geometry moved the token K-curve from +0.002 to +0.0163
(arm `slot-spandec-strict-prev-reach1`) and the carried content decayed about a third per
pass. Does the same geometry on the fan (K = 4 written streams, the write-all read), with
the fixed-point term off and the per-cell renorm on, earn a token K-curve of the size the
far budget allows, and does the renorm hold carried content across passes?

## Hypothesis

The far budget (at least 0.126 nats at depth 6) is a job only depth can do in this
geometry, so the loss pushes the passes to relay. The fan gives the relay four channels
instead of one and the write-all read lets the coda take the relayed content from any of
them. The renorm removes the free scale by which the single-cell chain let carried content
shrink against re-supplied own-span content. Against it: the chain's decay was measured
with the fixed-point term ON (a pull toward a fixed point is a pull toward forgetting), and
the renorm fixes the NORM of a cell, not the direction; the pass can still rotate carried
content out. The design's Step 2 (a persistent component written once on arrival) is the
fallback if the decay survives the renorm.

## Method

One arm, one config, no code change beyond commit 227b5fb (which added the config and
`tests/test_lxtul_r_composition.py`, 8 tests):

| arm | config | what it composes |
|---|---|---|
| `slot-spandec-strict-fan4-all-reach1` | `tul_slot_spandec_strict_fan4_all_reach1` | `tul_slot_spandec_strict_fan4_all` (fan_k 4, fan_mix all, epivol, prefix_k 4, slot_cells 4, spandec) + `tul.loop_reach: 1` + `tul.tg_coda_prefix_reach: prev` + `model.core_fixed_point_lambda: 0.0` + `model.slot_state_renorm: true`; `training.warmup` 1000, lr 1e-4 flat, 5,000 steps, batch 6, seq 1024, seed 1, `norm_match` |

Runner line kind `slot` (the same kind and readouts as the fan4-all arms), sweeps at 2500
and 5000, 480 rows, OUTDIR `/home/wolfe/morph-scratch/arc/results/2026-09-22-lxtul-r`, the
commit of this file. Readouts, in scoring order:

1. The trainer's compose print at startup (A1.1) and the runner's smoke (12 steps).
2. `preclip/total` (the divergence tripwire) and `loop/in_norm_t0` from wandb (A1.2).
3. The runner's forced-depth sweep at 5000: `ce_tokens K1-K6` with its interval (A1.3).
4. The planted decay row (A1.4): `lab/divergence/hop_distance_probe.py` on the 5000
   checkpoint, `--planted --planted-len 2 --skip-localiser --hops 6 --depths 1,2,3,6
   --rows 480 --batch 3`, on the DGX Spark from a checkout at this file's commit. The
   probe's `--cut-after` mode refuses multi-cell arms (it cuts `tg_allow`, and a fan arm
   delivers its relation as `tg_relation`), so the row is read WITHOUT the cut. Verified
   2026-09-22 01:31 on the Spark: the planted path runs on a 4-cell arm (fan4-all-fp0 at
   5k, 4 rows, 23 sites, `hop_smoke_fan4-all-fp0_5000.json`).
5. Paired depth-6 reads (`paired_vs_ruler.py`, 1024-token blocks on `tok_index`) against
   `slot-spandec-strict` (A1.5), against `slot-spandec-strict-fan4-all` and
   `slot-spandec-strict-fan4-all-fp0` (the arm's own bases), and against
   `budget-web-reachall` (A1.8).
6. `fan/stream_rank_t1`, `fan/oracle_ce`, `fan/mixed_ce` at the last val (A1.6).
7. The rate line of the run log (A1.7).

Reference numbers, all at 5k, depth 6, 480 rows (481 paired blocks), from the filed
sweeps and the paired reads run 2026-09-22 (`paired_slotarms_vs_reachall.json`):

| arm | CE d6 | token K1−K6 | gap to `budget-web-reachall` (paired) | `fan/stream_rank_t1` | tok/s |
|---|---|---|---|---|---|
| `slot-spandec-strict` | 4.3474 | +0.0016 | +0.2544 [+0.2451, +0.2648] | — | — |
| `slot-spandec-strict-fan4-all` | 4.3056 | +0.0049 | +0.2077 [+0.1981, +0.2180] | 2.820 of 4 | 7,096 |
| `slot-spandec-strict-fan4-all-fp0` | 4.2998 | +0.0102 | +0.2021 [+0.1924, +0.2126] | — | — |
| `slot-spandec-strict-prev-reach1` (single cell, the chain) | — | +0.0163 | — | — | — |
| `budget-web-reachall` (plain, every earlier span) | 4.0922 | +0.0378 (plain loop) | 0 | — | — |

The chain's planted decay row (`failures/2026-09-19-hop-distance-plateau-and-dilution.md`,
planted g = 2, benefit = control − planted, no cut): d1 +0.148, d2 +0.108, d3 +0.088,
d6 **+0.071**, so the chain kept 48 % of the arrival worth at depth 6 (the design note's
"24 %: 0.148 → 0.035" is the row under the reach cut, which this arm cannot run).
`loop/in_norm_t0` on fan4-all: 966 → 1,057 over the run (1.09x); on the noise arm 11x.

## Predictions (frozen)

- **P-1 (A1.1, compose).** The startup print shows `warmup 1000`,
  `core_fixed_point_lambda 0.0`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`,
  `slot_state_renorm true`, and the smoke passes. **95 %.** The composition tests assert
  it; the smoke is the first GPU step.
- **P-2 (A1.2, healthy).** `preclip/total` under 1e4 at every step ≥ 200 AND
  `loop/in_norm_t0` at step 4999 under 3x its value at step 200. **75 %.** The renorm pins
  each cell's norm through a pass, so the entry norm has one fewer route to grow; the
  fixed-point term is off, which removed a hold on the plain loop (0 of 6 vs 4 of 7
  detonations under warmup 0) but was free under the 1000-step ramp.
- **P-3 (A1.3, the headline).** `ce_tokens K1-K6` at 5000 at or above **+0.05** AND at or
  above **+0.063** (half the far budget's lower bound), so the bar is **+0.063**. **30 %.**
  Point estimate +0.03. The chain gave +0.0163 with one cell; four channels and the
  write-all read are a width and a read change, and width alone did not move the K-curve
  before (fan4-all +0.0049 vs strict +0.0016). Below +0.02: 40 %. In [+0.02, +0.063):
  30 %.
- **P-4 (A1.4, the decay).** In the planted table, g = 2 at depth 6 keeps at least
  **75 %** of its depth-1 (arrival) benefit. **35 %.** The renorm fixes the norm, not the
  direction; the chain kept 48 % with the fixed-point term on. Below 48 % (worse than the
  chain): 25 %.
- **P-5 (A1.5, CE parity).** Paired depth-6 CE vs `slot-spandec-strict` at or above
  **−0.02** (the arm no worse than strict by more than 0.02). **70 %.** fan4-all sits
  0.047 below strict paired to the same ruler; the chain sat at CE parity with strict; the
  coda under `prev` loses the old slots' cells, which the strict panel's prev-reach1 arm
  paid 0 for.
- **P-6 (A1.6, candidates).** `fan/stream_rank_t1` at or above **2.5** of 4 (fan4-all
  2.820); `fan/oracle_ce − fan/mixed_ce` is quoted beside the 0.04 near-copy floor of the
  oracle instrument. **65 %.** The epivol term is unchanged; the reach narrows what a cell
  reads, which gives the streams less to differ on.
- **P-7 (A1.7, rate).** tok/s at or above **5,677** (0.80 x 7,096). **60 %.** The reach
  relation is a mask (no extra pass); the renorm is one norm per cell per pass; the
  `prev` coda read is a narrower mask. The single-cell chain's rate was not filed.
- **P-8 (A1.8, the depth-dependence guard).** Paired depth-6 CE vs `budget-web-reachall`,
  reported as the fraction of the far budget's lower bound the arm recovers relative to
  fp0: `(0.2021 − gap_arm) / 0.1259`. Fraction at or above **0.25** (gap_arm ≤ 0.171):
  **30 %.** At or below 0 (the arm no closer to reachall than fp0): 35 %. A pass on P-3
  with a fraction near 0 is depth DEPENDENCE (the K-curve forced by the geometry), not a
  win, and is scored as such.

## Binding (the design note's rule, copied so it cannot drift)

- Pass = P-1 to P-7 all hold. Then the note's Step 1 is `implemented/` and Step 2 is
  scoped from the decay row.
- P-3 holds and P-4 fails: Step 2 (the decay fix) runs next with its own planned file.
- P-3 fails with P-4 holding: the reach frame is closed for this loop; the two-channel
  design is the next note.
- P-3 fails and P-4 fails: file under `failures/`, no Step 2; Wolfe decides.
- P-8 is reported in every case and qualifies a P-3 pass; it is not a pass/fail clause.
- P-2 fails (a detonation): the arm is re-queued ONCE with `model.core_fixed_point_lambda:
  1.0` (the note's risk row), amended here with the date before that launch.

## Not verified before launch

- No GPU step of the composed arm; the runner's smoke is the first. The composition tests
  are CPU, eager, small shapes.
- Whether the renorm changes the epivol term's reading (the term is charged after passes
  1..2 on the state the renorm just scaled).
- Whether the planted probe's site count on a fan arm matches the chain's (2,778 sites
  over 480 rows); the 4-row smoke gave 23.
- The 3070 could not host the smoke (another job holds 6.6 of 8 GB); the Spark did, from a
  checkout at 8b8f824 for the fan4-all-fp0 model. The Step 1 readout needs a Spark
  checkout at this file's commit (the bundle is being shipped).
- Wall clock: fan4-all took about 2.5 hours at 1.15 steps/s; this arm is expected the
  same or slower.
