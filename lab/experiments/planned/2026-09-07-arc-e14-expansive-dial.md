# Planned: ARC E14 — the expansive dial: the mean-12 mask arm with the gain hinge at 1.02

Status: planned
Date: 2026-09-07 (frozen before launch; queued behind E13. Wolfe: "It is worth testing
depth behavior if we let it get just a touch over 1.0 so it *can* be expansive.")
Arc: `2026-09-04-loop-contribution-arc.md`.

## Question

E13's mask arm at Poisson mean 12 survives, holds a typical gain of 0.85–0.92 under the
0.90 hinge, and finishes its work by depth 6 (forecast K6−K12 0.002 at 5000): a map held
contractive converges to a fixed point in a few iterations, and past that point depth is
free. Every run that earned past iteration 3 in this arc sat at the gain-1 edge and then
detonated. If the hinge lets the map's typical gain sit a touch ABOVE 1 (target 1.02),
does the loop keep working through 12 iterations, and does it survive? The two measured
deaths of an expansive slot map (E7 at mean 16, E12 at fixed 12) were the state's SCALE
growing ~100x at one iteration while the differential gain read ~1, so the second arm
pins the scale (`slot_state_renorm`) and leaves only the direction free.

## Method

Two arms, one seed each, 5000 steps, everything as E13's mask arm (`tul_m12_mnext_mask`:
Poisson mean 12, max 16, `bptt_depth 16`, ramp 1000, flat 1e-4, batch 6, seq 1024, clip
4.0, `core_fixed_point_lambda 1.0`, eager `tg_restrict`), sustained tripwire:

| arm | config | change |
|---|---|---|
| g102 | `tul_m12_mask_g102.yaml` | `slot_gain_target 1.02` |
| g102-rn | `tul_m12_mask_g102_rn.yaml` | `slot_gain_target 1.02` + `slot_state_renorm true` |

Readouts as E13: `core_depth_sweep.py` at forced depths 1, 2, 3, 6, 9, 12, 16, 24 on the
480 rows at 2500 and 5000, `worth_profile.py` at 5000, the `loss/gain_est`,
`loop/core_gain_t0`, `loop/out_norm_last`, `loss/fixed_point` and `preclip/total` traces.
Ruler: E13's mask arm at 5000 (`results/2026-09-07-arc-e13/sweep_m12-mnext-mask_5000.json`:
token CE 4.3347 at depth 12, forecast 6.7653 at 12, 6.7673 at 6, 6.7845 at 3) and E4.
A 2500 checkpoint of an arm that later trips is pre-onset and not cited for earning.
Runner `arc/run_e14.sh`, which waits for `E13 COMPLETE` in the queue log. Cost ~2 h per
arm plus readouts.

## Predictions (frozen)

- **P14a (does it climb).** On g102 the hinge reading `loss/gain_est` averages ≥ 1.00
  over steps 2000–5000 (or over the steps it lives, if it trips): **45 %** (E1: at
  targets 0.95/0.98 the map settled at 0.913/0.929, below the target).
- **P14b (survival).** g102 reaches 5000 with the tripwire silent: **35 %**; g102-rn:
  **60 %**.
- **P14c (the scale mode).** If g102 trips, `loop/core_gain_t0` exceeds 10 before
  `preclip/total` first exceeds 1e3: **75 %**. On g102-rn `loop/core_gain_t0` stays
  below 3 all run: **85 %** (the renorm pins the exit norm, so the ratio is bounded
  by one application).
- **P14d (depth, the point).** On the last non-pre-onset checkpoint of each arm,
  forecast `mux_local` K6−K12 > 0.01 with the paired CI above 0: g102 **30 %**,
  g102-rn **30 %**. K3−K6 > 0.03 (E13: 0.017): **40 %** each.
- **P14e (value).** Token CE at forced depth 12 beats E13's mask arm at depth 12
  (4.3347) on the 480 rows, paired CI below 0, at 5000: g102 **25 %**, g102-rn **25 %**.
- **P14f (the two terms fight).** On g102 `loss/fixed_point` averages above 0.01 over
  steps 2000–5000 (E13 mask: 0.0015–0.005): **55 %**. An expansive map cannot settle at
  its last iteration, so the fixed-point term pulls against the hinge.
- **P14g (trajectory).** `loop/delta_ratio_last` on g102 averages above 0.10 over its
  last 1000 steps (E13 mask: 0.025–0.036): **55 %**.

## Binding

- P14d TRUE on either arm ⇒ contraction was the cap: the depth question moves to the
  expansive regime, and the next run is the same arm at target 1.05 and at mean 12 with
  the fixed-point term OFF (the two terms fight, P14f); Wolfe's call.
- P14d FALSE on both with P14a TRUE ⇒ an expansive-in-typical-direction map still
  finishes by 6: the cap is not the gain constraint; E3 (staged targets) is next.
- P14a FALSE ⇒ the map does not climb to a looser hinge (E1 again); the hinge is not
  what holds it at 0.9 and the dial is closed as a lever; file and move to E3.
- P14b FALSE on g102 and TRUE on g102-rn ⇒ the renorm is the state bound the README's
  regime table needs for an expansive slot map; it becomes the default for any
  target ≥ 1.
- NO 20k run from this experiment; Wolfe's call.

## Not verified before launch

Hinge + renorm together in one build (each is measured alone; the base.yaml comment says
"either one, not both, is the tested configuration"; SCSE is off on this arm so the
build does not refuse); a target above 1 anywhere (E1 stopped at 0.98); the sweep at
forced depth 24 on a model trained at max 16.

## Run log (appended during the run; Predictions untouched)

- 16:35–18:00 `m12-mask-g102` (hinge target 1.02): DETONATED at 3877 by the single-row rule
  (`preclip/total` 1.41e6 at 3877; one earlier blip 5.7e3 at 3867; the probe read 2.2 again by
  3894 when the kill landed, so the run may have lived through it — unknowable from here).
  NOT the E7 scale mode: `loop/core_gain_t0` stayed 1.0–1.3 all run (never above 3); the
  hinge reading climbed 0.89 → 1.00 by step 1000 and averaged 0.990 over 2000–3877, at the
  edge and never above 1; the exit norm grew 4.5x over the 12 iterations (0.90 arm: 2.2x);
  the last-iteration relative change stayed 0.031 and the fixed-point term 0.0024, so a
  map held at gain 1 still settles by its last iteration. The spike was one sample's
  differential gain reading 15.9 at one iteration (`loss/gain_est_max`), the
  spike-at-the-crossing mode of 2026-09-04.
- **The effect to keep (Wolfe, 2026-09-07 18:10).** At 2500, pre-onset, paired over the 480
  rows against the 0.90 mask arm at the same step: tokens K1−K6 +0.0425 [+0.0408, +0.0440]
  vs +0.0326 [+0.0312, +0.0338], a third more loop contribution to the tokens from letting
  the map sit at gain 1; forecast K1−K6 +0.381 vs +0.458 (less); tokens K3−K6 +0.0007 vs
  +0.0005 (unchanged: nothing past iteration 3 on either); token CE at depth 6 4.665 vs
  4.657 (end point 0.008 worse, across runs). The gain target moves how much of the tokens'
  work the loop does, not how deep the loop works.
