# Planned: LXTUL Stage 3, what holds the slot map at 0.87, and does a map near 1 earn depth

Status: planned

Date: 2026-09-24 15:56 (frozen before any Stage 3 GPU step of arms A and B).
Parent: [`../failures/2026-09-24-lxtul-e-stage2.md`](../failures/2026-09-24-lxtul-e-stage2.md).
Wolfe's go: 2026-09-24 ("Both A and B"); framing, same day: "the map behaviors are an
effect, not a cause. we need to understand what cause yields the effect."

## Question

Stage 2 ended with the loop's depth value on the deployed coda at about zero (K1−K6
0.0011–0.0020) whoever shapes the loop. The map is the one reading that separates the
slot loop from the plain loop that earns. Measured with `lab/divergence/core_map_fd.py`
(fp32, 96 rows, depth 6):

| model | passes | typical gain (fp32) | positions above 1 | coda / token K1−K6 |
|---|---|---|---|---|
| notul (plain, earns) | 1–5 | 0.97 → 0.92 | 3–21 % | 0.053 |
| e4probe (slot loop) | 0–5 | 0.870 on every pass | 0 % | 0.0011 |

The slot map is a uniform contraction. It settles there by itself, below a 0.9 row hinge
that stops binding after about step 1000. Two questions:

1. **Cause (arm A).** The fixed-point term (`core_fixed_point_lambda 1.0`,
   `||h_T − h_{T−1}||² / ||h_T||²` at each slot's last pass) rewards settling. Is it the
   pull that holds the map at 0.87?
2. **Sufficiency (arm B).** If a floor pushes the map up into [0.95, 0.98], does depth
   value reach the deployed coda? If the map moves and the coda does not earn, a map near
   1 is not sufficient, and the map is an effect of something else. Wolfe's framing
   predicts this outcome is informative, not a dead end.

## Hypothesis

H-A: the fixed-point term is a direct pull toward a contraction. With it off, the map
drifts up toward the plain loop's 0.92–0.97. The E14 arc (hinge 1.02, map at 0.99) earned
+30–85 % more contribution before one excursion to 15.9 killed it; the tail hinge
(100 @ 1.1) now charges that excursion.

H-B: the loop's depth use needs a map near 1 so that a pass does not erase what the last
pass wrote. Forcing the typical gain to about 0.95 gives the passes room to integrate.
This is the weaker claim: arc E1 found that a looser hinge alone did not move the map, so
the map is chosen by the objective, and forcing it may buy a map without a use for it.

## Arms

All three compose `tul_slot_spandec_strict_e4probe` (strict geometry, K = 4 enumerated
codes, exact mixture, detached parallel head). They also carry the map guard of
`tul_slot_spandec_strict_e4probe_map` (row target 0.98, tail hinge 100 @ 1.1). 5000
steps from scratch, seed 1, one trainer at a time on the 5090.

| arm | config | changes from `e4probe_map` |
|---|---|---|
| A `lxtul-e4probe-nofp` | `tul_slot_spandec_strict_e4probe_nofp.yaml` | `model.core_fixed_point_lambda` 1.0 → 0.0 |
| B `lxtul-e4probe-floor` | `tul_slot_spandec_strict_e4probe_floor.yaml` | `model.slot_gain_floor_lambda` 0 → 100 at floor 0.95 |
| C `lxtul-e4probe-map` (control) | `tul_slot_spandec_strict_e4probe_map.yaml` | none: the guard alone |

C makes A and B one-factor. Reference on disk: `lxtul-e4probe` @5000 (Stage 2). Run
order: A, B, C. C is read against e4probe as the guard's own effect.

## Instruments

- `lab/divergence/core_map_fd.py` fp32 on the step-5000 checkpoints of A, B, C (96 rows,
  depth 6): typical gain per pass (`operator.rms_vjp`), per-position `g_pos` quantiles
  and `frac_gt_1` per pass.
- `lab/divergence/lxtul_e_stage2_score.py` with `--ref e4probe` (480 rows, forced depths
  1..6): coda and par mixture CE, width gain, exit separation, K1−K6, paired
  block-bootstrap CIs.
- `core_depth_sweep`, `worth_profile`, `lxtul_e_notul_pair.py` (coda against notul by
  span offset), generation samples.
- The logged training series: `loss/gain_est`, `loss/gain_slot_*`, `loss/gain_floor_pen`,
  `loss/fixed_point`, `loop/core_gain_t*`, `preclip/total`.
- Stability: `/home/wolfe/morph-scratch/arc/tripwire_sustained.py` on each probe file
  (DETONATED iff any row > 1e5 or two rows > 1e4 within 40 steps, step >= 200).

## Predictions

Map readings are fp32 `core_map_fd` at step 5000, the mean over passes 1–5. Coda readings
are the Stage 2 scorer at depth 6.

Arm A (fixed-point term off):

- **A-1 (the term holds the map).** Typical gain >= 0.92: **35 %.**
- **A-2 (depth on the deployed path).** Coda K1−K6 >= 0.005, CI clear of zero: **15 %.**
- **A-3 (the coda is not worse).** A coda mix − e4probe coda mix <= +0.005: **70 %.**
- **A-4 (stable).** Sustained tripwire reads HEALTHY: **80 %.**

Arm B (the floor):

- **B-1 (the floor moves the map).** Typical gain in [0.94, 0.99]: **80 %.**
- **B-2 (occasional steps over 1).** Positions above 1, pooled over passes 1–5, >= 1 %:
  **40 %.**
- **B-3 (depth on the deployed path).** Coda K1−K6 >= 0.005, CI clear of zero: **20 %.**
- **B-4 (the coda is not worse).** B coda mix − e4probe coda mix <= +0.005: **55 %.**
- **B-5 (the k/2 shape).** If B-3 holds: 80–95 % of B's coda K1−K6 is reached by pass 3
  (K1−K3 over K1−K6): **60 %.**
- **B-6 (stable).** Sustained tripwire reads HEALTHY: **65 %.**

Arm C (the guard alone):

- **C-1 (the guard is inert).** Typical gain within 0.02 of e4probe's 0.870, and coda mix
  within ±0.005 of e4probe: **75 %.**

## Verdict rules

Stage 3 is a success if A-2 or B-3 holds AND that arm's coda is not worse than e4probe
(A-3 or B-4). That is depth value on the deployed path, the positive this line exists for.

The map readings decide what the result means, whatever the verdict:

- B-1 holds and B-3 fails: the map near 1 is NOT sufficient. The contraction is an
  effect; the cause is upstream (the objective or the reader). Stage 4 goes to the cause,
  not to another map lever.
- A-1 holds: the fixed-point term is a cause of the contraction. If A-1 holds and A-2
  fails, it is a cause of the map and not of the depth deficit.
- A-1 fails and B-1 holds: the objective without the term still chooses 0.87; the choice
  comes from the coda's loss, and the Lean account (below) is the place to look.

No clause passes on a cosine.

## Method

The floor hinge (`model.slot_gain_floor_lambda`, `slot_gain_floor_target`) is built in
`morph/model/transformer.py::_slot_gain_penalty` and tested in
`tests/test_slot_gain_floor.py` (14 tests; six sabotages caught). Smoke A and B for 30
steps. Chain in `/home/wolfe/morph-to` at the merge commit: A, B, C (5000 steps each),
then `core_map_fd` on A, B, C, the Stage 2 scorer with `--ref e4probe`, sweeps, worth,
notul pairing, samples. Artifacts: JSON and `.runlog.txt` files in
`../results/2026-09-24-lxtul-stage3-map/`.

A Lean agent is working on the causal account of the contraction in parallel
(Wolfe, 2026-09-24). Its account is read beside these results. It does not change the
predictions above.

### Method amendment, 2026-09-24 17:04 (readings added; predictions unchanged)

The Lean account ([`../../theory/tul_exploration/MAP-CAUSE.md`](../../theory/tul_exploration/MAP-CAUSE.md))
arrived after the predictions were frozen and after the chain started. It finds that the
slot map's 0.87 is the `DiagonalInjection` floor, `sqrt((704 + sum A^2) / 1024)`: the
context channels are scaled by A, the other 704 pass through, and A has not moved from its
0.447 init in any model read (checked by me on e4probe, the ruler and notul: floors 0.864,
0.865, 0.867 against maps 0.870, 0.877-0.883, 0.92-0.97). A map at the floor means the
core's blocks barely respond to a change of the slot state. Two readings are added, both
from the step-5000 checkpoints, no forward pass needed for the first:

- **Injection decay A** per arm (`injection.log_A`), and the floor it implies.
- **The blocks' part of the gain**, map minus floor, per pass, beside the map itself.

These decide how B-1 reads. The floor hinge can be met by raising A alone (A about 0.83
gives a floor of 0.95) with the blocks as quiet as before. B-1 holding with the map within
0.03 of B's own floor means the gain moved through the injection, not through the blocks.
