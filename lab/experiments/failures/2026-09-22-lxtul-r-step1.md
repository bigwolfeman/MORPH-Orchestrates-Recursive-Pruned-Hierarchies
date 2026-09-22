# Planned: LXTUL-R Step 1 — the fan's K streams on the chain geometry (fan4-all + loop reach 1 + coda prefix reach prev + fixed-point 0 + slot-state renorm)

Status: failure

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

## Results

One arm on the runner at bcb0dc2, 2026-09-22 01:38 to 03:03 (smoke 35 s, 5,000 steps at
1.21 to 1.22 steps/s), readouts to 03:25. Artifacts: `../results/2026-09-22-lxtul-r-step1/`
(the runner's sweeps at 2500 and 5000, the four paired jsons, the planted probe's json
and its Spark run log); wandb run `dwx8hig1`. Intervals are the block bootstrap.

| reading | value | clause |
|---|---|---|
| startup config (wandb config; the log prints the reach, the fan mix and the coda reach) | `warmup 1000`, `core_fixed_point_lambda 0`, `loop_reach 1`, `tg_coda_prefix_reach prev`, `fan_mix all`, `slot_state_renorm True`; smoke exit 0 | P-1 |
| `preclip/total` max at step ≥ 200 | 42.2 at step 2675 (tripwire 1e4); verdict HEALTHY | P-2 |
| `loop/in_norm_t0` step 200 → 4999 | 943 → 1,196 (**1.27x**; bar 3x; noise arm 11x, fan4-all 1.09x) | P-2 |
| `loop/core_gain_t0` | 1.000 at every logged step (the renorm pins it) | — |
| token CE, forced depth 1 / 2 / 3 / 6 / 9 / 12 / 16 at 5000 | 4.3394 / 4.3295 / 4.3217 / 4.3192 / 4.3195 / 4.3198 / 4.3201 | P-3 |
| **token K1−K6 at 5000** | **+0.0202 [+0.0193, +0.0212]**; K3−K6 +0.0025 [+0.0022, +0.0028] | P-3 |
| token K1−K6 at 2500 | +0.0163 | — |
| paired depth 6 vs `slot-spandec-strict` (490 blocks) | **−0.0329 [−0.0362, −0.0296]** (the arm better); depth 1: −0.0128 | P-5 |
| paired depth 6 vs `slot-spandec-strict-fan4-all` (500 blocks) | +0.0136 [+0.0110, +0.0163]; depth 1: +0.0338 | — |
| paired depth 6 vs `slot-spandec-strict-fan4-all-fp0` (500 blocks) | **+0.0194 [+0.0168, +0.0222]**; depth 1: **+0.0396 [+0.0367, +0.0427]** | P-8 |
| paired depth 6 vs `budget-web-reachall` (481 blocks) | +0.2215 [+0.2113, +0.2328]; depth 1: +0.2416 | P-8 |
| A1.8 fraction `(0.2021 − 0.2215) / 0.1259` | **−0.15** | P-8 |
| `fan/stream_rank_t1` / `_t6` at the last val | **2.975** / 1.728 (fan4-all 2.820 / 2.060) | P-6 |
| `fan/oracle_ce` / `fan/mixed_ce` / `fan/single_ce` | 4.3641 / 4.4118 / 4.4566; oracle − mixed **0.048** beside the 0.04 near-copy floor (fan4-all 0.042) | P-6 |
| `val/fan_stream_cos_t1` | −0.327 | — |
| rate | 7,496 tok/s at step 200 (RATE OK), 7,454 to 7,525 through the run; fan4-all 7,096 | P-7 |
| final `val/loss` | 4.4163 (fan4-all 4.4348 as `fan/mixed_ce`) | — |

Scoring:

- **P-1 HOLDS.** All six values on the run's config; the smoke passed (loss 33.67 at step 0,
  the family's number: fan4-all's smoke read 33.675).
- **P-2 HOLDS.** Tripwire max 42 at step 2675; the entry norm grew 1.27x.
- **P-3 FAILS.** +0.0202 against the +0.063 bar (and against +0.05 alone). It is the largest
  token K-curve of any slot-loop arm on the ledger (fan4-all +0.0049, fp0 +0.0102, the
  single-cell chain +0.0163, twelve arms at about +0.002) and it is a third of the bar. The
  prereg's middle band ([+0.02, +0.063), 30 %) is where it fell, at the band's floor.
- **P-5 HOLDS.** 0.033 better than strict, paired; the fan's write-all read is worth 0.047
  against strict on the same rows, so the chain geometry gave back 0.014 of it.
- **P-6 HOLDS.** Rank 2.98 of 4 at pass 1 (above fan4-all's 2.82); the oracle-over-streams
  gap 0.048 sits 0.008 above the 0.04 floor, as it did on fan4-all (0.042).
- **P-7 HOLDS.** 7,496 tok/s, 1.06x fan4-all; the reach mask and the renorm cost nothing
  measurable.
- **P-8: the fraction is −0.15** (the prereg's "at or below 0" branch, 35 %). The arm is
  0.019 FARTHER from reachall than fp0 at depth 6 and 0.040 farther at depth 1. Read
  together with P-3: the chain geometry costs the fan 0.040 nats at depth 1 (the coda
  reads one slot instead of all, the loop reads one neighbour instead of all), and the
  passes claw back 0.020 of it by depth 6. That is depth DEPENDENCE in the design note's
  own words (A1.8): the K-curve is forced by the geometry and the arm never reaches its
  base, so the +0.020 is not a win over fan4-all-fp0.

### The planted decay row (A1.4), DGX Spark, 03:08 to 04:38, and what it exposed

`hop_distance_probe.py` on the 5000 checkpoint from the `MORPH-0922` worktree (bcb0dc2),
480 rows, batch 3, `--planted --planted-len 2 --skip-localiser --hops 6`, depths 1, 2, 3, 6;
2,842 sites (the chain's run: 2,778). Benefit = control − planted, controls 15.407 / 15.403 /
15.397 / 15.401 at depths 1 / 2 / 3 / 6. The chain's row
(`failures/2026-09-19-hop-distance-plateau-and-dilution.md`, single cell, fixed-point term
on, no renorm) beside it:

| g | this arm d1 | d2 | d3 | d6 | kept at d6 | chain d1 | d2 | d3 | d6 | chain kept |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | +0.549 | +0.539 | +0.534 | +0.526 | 96 % | +0.566 | +0.562 | +0.560 | +0.558 | 99 % |
| 1 | +0.115 | +0.089 | +0.105 | +0.106 | 92 % | +0.181 | +0.158 | +0.149 | +0.137 | 76 % |
| 2 | +0.051 | +0.046 | +0.048 | +0.046 | 89 % | +0.148 | +0.108 | +0.088 | +0.071 | 48 % |
| 3 | **+0.048** | +0.033 | +0.030 | +0.033 | 68 % | +0.000 | +0.083 | +0.076 | +0.062 | — |
| 4 | **+0.023** | +0.022 | +0.017 | +0.022 | 94 % | +0.000 | +0.000 | +0.046 | +0.047 | — |
| 5 | +0.000 | +0.028 | +0.020 | +0.023 | — | +0.000 | +0.000 | +0.000 | +0.034 | — |
| 6 | +0.000 | +0.022 | +0.017 | +0.017 | — | +0.000 | +0.000 | +0.000 | +0.021 | — |

- **P-4 HOLDS on its letter.** g = 2 keeps **89 %** of its depth-1 benefit at depth 6
  (0.051 → 0.046), against the 75 % bar and the chain's 48 %.
- **The row is not the design's row.** g = 3 reads **0.048 at depth 1** and g = 4
  0.023; g = 5 and g = 6 read exactly the control at depth 1 (0.000) and arrive at
  depth 2. The leaky reach inside one pass was FOUR slots. Under a one-slot
  reach it must read exactly 0 there (the chain did: 0.000 at every g ≥ 3, arrival at
  pass g − 1). Content three spans back reached the coda inside ONE pass, so the arm's
  in-loop reach was not one slot per pass.

**The leak, found and fixed the same night.** The register branch of `_tul_core` hands
the core layers `tg_relation` alone. The CCA conv (kernel 4) and the value shift are
position-local on the flattened cell axis and were "left alone" by that branch (its
comment says why: at reach 0 every earlier cell is allowed anyway). Under a reach budget
they read the previous slot's last cells at EVERY core layer, so the six core layers
relayed about one slot per LAYER. On the tiny model at forced depth 1 (a perturbation of
span j − g, max |Δlogits| on span j): j−2 1.1e−1, **j−3 1.9e−2, j−4 6.5e−4, j−5 1.3e−4**
with the conv, and with the conv cut to kernel 1 still **j−3 2.9e−2** (the value shift
alone relays) and j−5 exactly 0. The single-cell chain never had this: its reach kwargs
carry a per-cell `tg_seg`, which resets both ops. The fix (9b430d3, committed before this filing:
a per-SLOT `tg_seg` beside the relation whenever `loop_reach > 0`, reach 0 untouched so
every filed register arm keeps its forward) makes j−3 and beyond read **exactly 0** at
depth 1 and j−4 exactly 0 at depth 2 with the shipped conv;
`tests/test_lxtul_r_composition.py` (11) asserts it, and `test_tul_fan*` (30 + the rest)
still pass. The composition test 3a checked the attention relation and nothing else; the
prereg's "Not verified" list did not name the position-local ops. That is the method
fault.

## Verdict

**Failure, method fault** (P-3 fails as scored; P-1, P-2, P-4, P-5, P-6, P-7 hold; P-8
reads −0.15; but the arm did not run the design's geometry). The readings stand as a
record of what the LEAKY chain does: the largest token K-curve of any slot-loop arm
(+0.0202 [+0.0193, +0.0212]), a 0.040-nat depth-1 cost against fan4-all-fp0 of which the
passes recover half, carried content that no longer decays (89 % kept) but arrives at a
third of the chain's worth, channel worth 0.162 against 0.196. None of the binding's
three clauses is applied: the clause "P-3 fails with P-4 holding → the reach frame is
closed" needs a one-slot reach, and this arm relayed several slots per pass through the
conv and the value shift, so its passes had less to do than the design gives them. The
next planned experiment is the same arm at the corrected geometry
(`2026-09-22-lxtul-r-step1b.md`), queued behind the seed twins.

## Updated hypothesis

The reach dial on a register is a relation on attention AND a segment reset on the
position-local ops; the first without the second is a per-layer relay. With the leak,
forcing history through the loop still moved the passes (K1−K6 +0.005 → +0.020) and the
renorm held what they carried; with the leak closed, each pass has strictly more to fetch
(one slot, not several) and the depth-1 cost against fp0 will be larger than 0.040. The
design's question, whether the passes recover that cost and more, is open until Step 1b
reads. What is settled: a pass-relayed copy under this recipe is worth less than a direct
read of the same content (planted g = 2: 0.051 relayed against 0.148 on the chain), and
the coda's `prev` read costs the channel 0.034 nats of worth.
