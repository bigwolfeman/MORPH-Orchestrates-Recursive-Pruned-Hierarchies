# Planned: does the faster WTA (one posterior winner; MAP-only grad) keep quality and depth use?

Status: planned

Date: 2026-09-29 15:20 (frozen before either run trained).
Code: c2d77a3 (`tul.fan_all_wta_winner: map`, `tul.fan_all_wta_grad_rollouts: map`; note
[`2026-09-29-onewinner-perf-fused-ce-shared-posterior-grad-map.md`](../../../.agents/notes/proposed/architecture/2026-09-29-onewinner-perf-fused-ce-shared-posterior-grad-map.md)).
Wolfe (2026-09-29): "This is still quite slow but worth testing."

## Question

The ev01 profile went 2.278 -> 1.553 s GPU per step with two changes. (1) `map`: pick each slot's
winner once, under its MAP rollout from the exact mixture posterior (the objective is unchanged
except the winner choice; on (b) at 5k the MAP winner matches each rollout's own 0.911 of the
time). (2) `grad_rollouts: map`: charge the winner-alone CE on the MAP rollout only (an objective
change). Do they keep CE and loop depth use at 3000 steps?

## Hypothesis

H: (1) is quality-neutral; (2) costs a little CE (the WTA term sees a quarter of the rows) and
leaves depth use alone, since the codes, not WTA, bought the depth.

## Control band (ev01 = per_rollout, lambda 0.1, 3000 steps; 480 sweep rows)

| seed | K1-K6 | gap to plain 5k | tok/s |
| --- | --- | --- | --- |
| 1 | +0.0150 | +0.4497 | 2613 |
| 2 (s2 @ 3000) | +0.0082 | +0.4627 | (5k run) |
| 3 | +0.0072 | +0.4545 | 2624 |

## Predictions (mine, orchestrator, before either run)

- **P-1**: neither run detonates. 85 %.
- **P-2**: mapwin K1-K6 inside [+0.0072, +0.0150]. 65 %.
- **P-3**: mapwin gap to plain 5k inside [+0.445, +0.468] (the band widened by 0.005). 70 %.
- **P-4**: mapgrad K1-K6 inside [+0.0072, +0.0150]. 55 %.
- **P-5**: mapgrad gap to plain 5k inside [+0.445, +0.468]. 60 %.
- **P-6**: trainer tok/s: mapwin >= 3000 (1.15x ev01); mapgrad >= 3500 (1.34x). 75 %.
- **P-7**: pass-1 cell rank of both inside [2.55, 2.95] (ev01 seeds read 2.60, 2.92). A
  diagnostic. 60 %.

Pass for (1): P-1, P-2, P-3, P-6. Pass for (2): P-1, P-4, P-5, P-6. One seed each, read against a
three-seed band; an arm outside the band by less than the band's own width is not a verdict.

## Method

- Configs `tul_slot_spandec_strict_lxfan4_wta_fp01_mapwin` and `..._mapgrad`: ev01 plus the
  knob(s); composed, and `build_tul_runtime` shows (map, all) and (map, map). Seed 1, 3000 steps.
- Queue at c-SHA of this prereg, after the running plain-panel control (ctrl-s1r), with the
  latent-WTA arm when it is built; the queue pauses after these (Wolfe). Runner
  `runner_steps.sh`, guard `maptest_guard.sh`. Readouts as for the seed bands: depth sweep,
  gap to plain, worth; the stage-2 scorer exits 1 on fan arms by design.
