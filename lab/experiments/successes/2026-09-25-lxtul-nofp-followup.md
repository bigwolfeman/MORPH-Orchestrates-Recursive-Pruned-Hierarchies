# Planned: does the fixed-point-off positive replicate, and can a weak term keep it stable

Status: success

Date: 2026-09-25 09:02 (frozen before either arm's first GPU step).
Parent: [`../successes/2026-09-24-lxtul-stage3-map.md`](2026-09-24-lxtul-stage3-map.md).
Wolfe's go: 2026-09-25 ("Yes queue both.").

## Question

With `core_fixed_point_lambda 0` the strict e4probe slot loop earned coda K1−K6 +0.0123
[+0.0115, +0.0130] (seed 1), the code rollouts separated 3.47 at depth 6 (0.27 with the
term at 1.0), and the run detonated once at step 4048 and recovered. (1) Does it replicate
at a second seed? (2) Does the term at 0.1 keep the stability hold without stopping the
motion that earns?

## Hypothesis

The term at 1.0 charges the pass-to-pass motion the loop needs; the positive is the loop
moving, so it replicates. A tenth of the term is small next to the coda's pull on the
motion and still large enough to damp the single-slot excursion.

## Arms

5000 steps from scratch, one trainer at a time.

| arm | config | changes from `lxtul-e4probe-nofp` |
|---|---|---|
| `lxtul-e4probe-nofp-s2` | `tul_slot_spandec_strict_e4probe_nofp_s2.yaml` | `training.seed` 1 → 2 |
| `lxtul-e4probe-fp01` | `tul_slot_spandec_strict_e4probe_fp01.yaml` | `model.core_fixed_point_lambda` 0 → 0.1 |

References on disk: `lxtul-e4probe-nofp` (seed 1) and `lxtul-e4probe-map` (arm C, term 1.0).

## Instruments

The Stage 2 scorer with `--ref nofp` and arm C beside it (480 rows, forced depths 1..6):
coda mix CE, K1−K6, exit separation, width gain, paired CIs. `core_map_fd.py` fp32.
Sweeps, notul pairing. The sustained tripwire on each probe file.

## Predictions

nofp-s2:

- **S-1 (replicates).** Coda K1−K6 >= 0.005, CI clear of zero: **65 %.**
- **S-2 (same size).** Coda K1−K6 in [0.0083, 0.0163] (seed 1 ± 0.004): **45 %.**
- **S-3 (the codes move).** Exit separation at depth 6 >= 1.0: **65 %.**
- **S-4 (stable).** Tripwire HEALTHY: **45 %.**

fp01:

- **F-1 (still earns).** Coda K1−K6 >= 0.005, CI clear of zero: **45 %.**
- **F-2 (stable).** Tripwire HEALTHY: **70 %.**
- **F-3 (the coda gains).** fp01 coda mix @6 − nofp (seed 1) <= 0: **45 %.**
- **F-4 (the codes move).** Exit separation at depth 6 >= 1.0: **45 %.**

## Verdict rules

The positive replicates if S-1 holds. The follow-up is a success if S-1 holds; the recipe
question is answered YES if F-1 and F-2 both hold (a loop that earns and does not
detonate). If S-1 fails, the Stage 3 positive is filed as seed-dependent in the ledger.

No clause passes on a cosine.

## Method

Chain in `/home/wolfe/morph-wt-map` at the commit that adds this file: 30-step smokes,
nofp-s2, fp01, then `core_map_fd`, the scorer, sweeps and the notul pairing. Artifacts:
JSON and `.runlog.txt` files in `../results/2026-09-25-lxtul-nofp-followup/`.

## Results (filed 2026-09-25 12:48)

Chain at `10baceb` in `/home/wolfe/morph-wt-map`, both arms 5000 steps, exit 0. Artifacts:
[`../results/2026-09-25-lxtul-nofp-followup/`](../results/2026-09-25-lxtul-nofp-followup/)
(`score.runlog.txt`, `followup_score.json`, `sweep_*`, `core_map_fd.*`, `tripwire_*`,
`notul_pair.runlog.txt`, `run_*`). Scorer: `lxtul_e_stage2_score.py --ref nofp`, 480 rows,
501,106 coda tokens, self-check max |dev| 1.1e-6. `core_depth_sweep.py` reads the same
K1−K6 on both arms to the fourth decimal.

| arm | fp term | final val | coda K1−K6 [95 % CI] | par K1−K6 | exit sep @6 | coda mix @6 | eval map fp32, passes 1-5 (pass 0) | tripwire |
|---|---|---|---|---|---|---|---|---|
| nofp (seed 1, Stage 3) | 0 | 4.4182 | +0.0123 [+0.0115, +0.0130] | +0.2186 | 3.47 | 4.3344 | 0.884 (0.826) | DETONATED @4048, recovered |
| **nofp_s2** (seed 2) | 0 | 4.4089 | **+0.0077 [+0.0071, +0.0082]** | +0.0528 | 3.48 | 4.3208 | 0.927–0.963 (1.187) | HEALTHY, max 38.8 @1310 |
| **fp01** (seed 1) | 0.1 | 4.4199 | **+0.0126 [+0.0119, +0.0134]** | +0.1190 | 5.61 | 4.3408 | 0.883–0.897 (0.758) | HEALTHY, max 475 @2777 |
| map (Stage 3) | 1.0 | | +0.0009 [+0.0007, +0.0011] | −0.0002 | 0.26 | 4.3254 | | HEALTHY |

Paired against nofp seed 1 on shared tokens (arm − ref; negative = the arm is better):

| arm | coda mix @6 | par mix @6 | par K1−K6 |
|---|---|---|---|
| nofp_s2 | −0.0137 [−0.0166, −0.0108] | −0.0387 | −0.1658 |
| fp01 | +0.0063 [+0.0039, +0.0087] | +0.0695 | −0.0996 |
| map | −0.0090 [−0.0114, −0.0066] | +0.0282 | −0.2188 |

Against the plain control (notul, coda tokens, `notul_pair.runlog.txt`): nofp_s2 +0.2451,
map +0.2496, nofp +0.2590, fp01 +0.2650 nats behind. The strict geometry's cost is
unchanged by any arm.

Anatomy of the two excursions, from the per-step probes:

- **Step 2912 is a data batch, not a model event.** `loss/total` jumps from about 12.0 to
  20.5–22.1 at step 2912 in EVERY arm of Stage 3 and of this follow-up (nofp 22.07, map
  21.05, floor 21.16, nofp_s2 20.50, fp01 21.47), against a median of 11.8–11.9 over steps
  2800–3000. The same step in five arms with two seeds and three objectives is the data
  order.
- **fp01 step 2777 is one hinge fire.** `gain_est` 1.163, every slot's gain above 1
  (`gain_slot_frac_gt1` 1.0), `gain_reg_weighted` 4.81, preclip ratio 475. It recovered
  with no second fire. Under the abort rule (1e4) it is healthy. nofp seed 1's event at
  4048 was 1.96e8.

Clause by clause:

- **S-1 HOLDS.** nofp_s2 coda K1−K6 +0.0077, CI [+0.0071, +0.0082], clear of zero and above
  0.005.
- **S-2 FAILS.** 0.0077 is below the band floor 0.0083 by 0.0006. The seed moved the size
  by 37 %.
- **S-3 HOLDS.** Exit separation at depth 6 is 3.48.
- **S-4 HOLDS.** Tripwire HEALTHY, max 38.8.
- **F-1 HOLDS.** fp01 coda K1−K6 +0.0126 [+0.0119, +0.0134].
- **F-2 HOLDS.** Tripwire HEALTHY, max 475 (the one hinge fire above).
- **F-3 FAILS.** fp01 coda mix @6 is 0.0063 WORSE than nofp seed 1. The seed spread
  between the two nofp seeds on the same statistic is 0.0137, so F-3's comparison cannot
  be read at n = 1 either way.
- **F-4 HOLDS.** Exit separation at depth 6 is 5.61, the largest of any e4probe arm.

## Verdict

**Success.** S-1 holds: the fixed-point-off positive replicates at a second seed, smaller
(+0.0077 vs +0.0123). The recipe question is YES: F-1 and F-2 hold, so the term at 0.1 keeps
the loop earning (+0.0126) with no detonation on this seed. Two predictions failed, S-2 (size
band, missed by 0.0006) and F-3 (the coda gain, within seed noise).

Three facts this run adds, measured:

1. **The fixed-point term's dose is the switch.** Coda K1−K6 at weight 1.0 is 0.0009; at 0.1
   it is 0.0126; at 0 it is 0.0077–0.0123 over two seeds. A tenth of the term does not
   stop the motion.
2. **The map does not predict the earning, again.** nofp_s2's eval map is the highest read
   in the strict family (0.93–0.96 on passes 1-5, pass 0 at 1.19) and it earns LESS than
   fp01, whose map (0.88–0.90) is at the injection floor's neighbourhood. Stage 3 and the
   map-cause filing said the same with other arms.
3. **Coda CE between these arms is seed noise at n = 1.** The two nofp seeds differ by
   0.0137 at depth 6. That is larger than map − nofp (−0.0090) and fp01 − nofp (+0.0063).
   Stage 3's "the depth-6 coda is 0.0046 worse than e4probe's" is inside this spread and is
   not a finding. K1−K6 is paired within one model and is readable. Cross-arm CE is not.

## Updated hypothesis

The term at 1.0 charges the pass-to-pass motion the loop needs; a tenth of it keeps the
motion and adds insurance against the scale-mode detonation the term was built for. fp01 is
the base for the next arm (span-level NextLat, note
[`../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md`](../../../.agents/notes/proposed/architecture/2026-09-25-span-level-nextlat.md)).
Any cross-arm CE claim on this family needs two seeds per arm; the per-model K1−K6 does not.
