# Planned: does the faster WTA (one posterior winner; MAP-only grad) keep quality and depth use?

Status: failure

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

## Results

Filed 2026-09-29 18:25 CDT. mapwin ran at 2e19cb8 (runner `runner_steps.sh`, seed 1, 3000
steps); readouts as for the seed bands. **mapgrad did NOT run**: Wolfe paused the queue after
mapwin and it sits in `queue_hold.txt`, so P-4 and P-5 are untested. Artifacts:
[`../results/2026-09-29-wta-map-speed-quality/`](../results/2026-09-29-wta-map-speed-quality/).

| reading | mapwin | ev01 band | prediction |
| --- | --- | --- | --- |
| detonation | none (tripwire healthy, max 472 at 2443) | none | P-1 held |
| K1-K6, 480 rows | +0.0096 [+0.0087, +0.0105] | +0.0072 .. +0.0150 | P-2 held |
| gap to plain 5k | +0.4553 [+0.4381, +0.4739] | +0.4497 .. +0.4627 | P-3 held |
| final val loss | 4.5400 | 4.5363 (s1), 4.5444 (s3) | context |
| pass-1 stream rank | 2.87 (`stream_rank_t1`), cell eff rank 2.80 | 2.60, 2.92 | P-7 held |
| trainer tok/s, mean of the 14 log points at step >= 200 | 2739 (min 2362, max 3048) | 2605 (s1), 2613 (s3) | P-6 FAILED (>= 3000 predicted) |

The mapwin tok/s series (every 200 steps): 3048, 2852, 2777, 2576, 2402, 2362, 2412, 2415, 2925,
2977, 3037, 2955, 2728, 2880. The low stretch (steps 1000-1600) matches the window when CPU
test suites ran on this machine (orchestrator and builders, about 16:30-17:10 CDT); outside it
the arm reads 2880-3048. The ev01 seeds ran without that load and read flat (2446-2645). So the
unloaded speedup is about 1.10-1.16x, the loaded mean 1.05x. Both are below P-6's 1.15x mean,
and the dip says the eager TUL trainer is sensitive to CPU contention (launch-bound), which
the profiler's GPU-busy time (the source of the 1.18x estimate for this mode) cannot see.

## Verdict

FAILURE on P-6, the speed half of the question. The quality half held: one shared posterior
winner is quality-neutral (K1-K6, gap, val loss and cell rank all inside the three-seed ev01
band). But the wall-clock gain is 1.05-1.16x, not the 1.18x GPU-time estimate and far from
the 3.4x this arm trails plain by. One seed; the band's own width bounds what it can show.

## Updated hypothesis

The WTA winner pick is not where LX-Fan's cost lives. From the same runner at 5k: a2 (write-all
cells + coda WTA, no code rollouts) 7783 tok/s, the K=4 code arms 5538-5723, (b) 2306, plain
9355-10089. The K code rollouts, which multiply the coda and measured as an ensemble
(`2026-09-26-lx-selection-ceiling.md`), are the cost. Exploration must live on slot positions
and the coda must read the tokens once: arms A (`fan_all_wta_grader: head`) and B
(`code_policy_k`), being built 2026-09-29. mapgrad only trims the grad pass on the same
K-rollout arm; whether to run it or drop it from the hold is Wolfe's call.
