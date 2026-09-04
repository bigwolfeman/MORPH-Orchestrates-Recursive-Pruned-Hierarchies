# Planned: ARC E3 — staged targets: memory first, then forecast (two jobs on one loop)

Status: planned
Date: 2026-09-04 (frozen after the code passed its CPU tests; not launched; GPU time is
Wolfe's call)
Arc: `2026-09-04-loop-contribution-arc.md`, branch (b) TARGET.

## Question

Every stable slot loop does its work in three iterations and then holds still, on the
memory target (M-own: own-loss 3.179 → 3.143 → 3.141 over depths 1 / 3 / 4, K3−K6
0.001) and on the forecast (Y2: 6.763 → 6.750 → 6.749, K3−K6 0.000). Both are ONE job,
and a one-job loop that finishes in three passes has nothing to do with a fourth. The
Depth-Adaptive Transformer (Elbayad et al. 2020) measured the same thing from the other
side: supervising every iteration with the SAME target collapses the states into
decodable form early. E3 gives the loop two jobs in sequence on one live-carry
trajectory: the state after iteration k is supervised toward the span the slot
TERMINATES (memory, `target="own"`; the target the loop earned 0.037 nats on), and the
final state toward the NEXT span (forecast, `mux_target: next`) as before. The local
loss is the mean of the two. Does a loop with a second job past iteration k keep
earning past iteration 3, and does the forecast read the memory the early iterations
wrote?

Code: `tul.mux_stage_own_iters` (`morph/model/tul.py`; the trajectory in `_tul_core`
with the carry LIVE, unlike `db_loop`; `_tul_mux_loss(target=...)`; the staged branch in
`_forward_tul`; eval exposes `mux_local_own_final` / `mux_local_next_final` on the final
state; `core_depth_sweep.py` reports both by forced depth). Tests:
`tests/test_tul_stage_targets.py` (8 tests: trajectory kept and live, exact loss
decomposition, eval columns, gradient reach, refused combinations, the construction
guard, the shallow-batch clamp).

## Arms

| # | run | config | one line |
|---|---|---|---|
| E3-2 | `to-mnext-y2-stage2` | `tul_to_mnext_y2_stage2` | Y2 + memory at iteration 2, forecast at the end |
| E3-3 (second, if E3-2 survives) | `to-mnext-y2-stage3` | `tul_to_mnext_y2_stage3` | memory at iteration 3 |

Control: Y2 (forecast only) and M-own (memory only, `to-mown`, the panel's R3) on disk.
Same recipe, seed, rows and readouts as E1; 5000 steps; tripwire on.

## Readouts

As E1, plus: `mux_stage_own` / `mux_stage_next` from the probe file (the two training
terms over time); the sweep's `mux_local_own` and `mux_local_next` columns by forced
depth (both targets read from the final state), each with K1−K6, K3−K6 and, for the
own column, K2−K6 (does the memory DEGRADE after its stage?); the step-ratio profile
over t = 1..7 per 500-step window.

## Predictions (frozen)

- **P3a.** E3-2 reaches 5000 with the tripwire silent: **55%** (memory ran clean on the
  recipe; the forecast face runs clean under the hinge; the two together are new).
- **P3b (the bar).** E3-2 THINKS on the forecast column (`mux_local_next` K3−K6 > 0.01,
  CI above 0, 480 rows, step 5000): **25%**. K1−K6 on that column > Y2's 0.0135: **55%**.
- **P3c.** The memory stage works: `mux_stage_own` (training, iteration-2 state) median
  over 4500–4999 ≤ 3.30 (M-own's FINAL own loss is 3.141 at depth 4): **55%**.
- **P3d (the two-job signature).** Memory read from the FINAL state degrades after its
  stage: `mux_local_own` K2−K6 < 0 with the CI below 0 (the later iterations overwrite
  what iteration 2 wrote while they build the forecast): **50%**.
- **P3e.** The step-ratio profile on 4500–4999 is non-monotone in t, with a ratio at
  t = 3 or 4 above the ratio at t = 2 (a second job starting): **40%**.
- **P3f.** Last-four val CE within 0.03 of Y2's 4.2809: **45%**. Below Y2: **25%**.
- **P3g.** Plan worth at offset 0, zero-out, ≥ 0.10 (Y2 0.074, M-own 0.048): **35%**.

## Decision rule (binding)

- P3b TRUE ⇒ the loop earns past iteration 3 when it has a second job; E4 on E3-2 (the
  mask), then E5 (20k, matched wall clock). E3-3 runs to place the stage.
- P3c TRUE, P3d TRUE, P3b FALSE ⇒ the loop CAN do two jobs in sequence and the forecast
  still finishes in the iterations it has after the memory: the forecast target is
  finished, not the loop; (b) is closed for forecast targets at this scale and the arc's
  closing rule applies (data where depth pays, or the deep slot stack).
- P3c FALSE ⇒ the memory stage did not form under the hinge; E3-3 runs before any
  conclusion (the stage may sit too early), then the same rule.
- P3a FALSE ⇒ the trip step and `preclip/*` shape are filed; the memory gradient into
  the shared core at iteration 2 is the suspect, and E3-3 does not run.

## Not verified before launch

The staged loss under torch.compile at the real shape together with the hinge's two
extra core steps (CPU tests only; the 12-step compiled smoke is the pre-launch check, as
Method note); the interaction of the own term's gradient with the gain constraint's
random-iteration sample (the hinge samples one t; the own term pulls on iterations ≤ k).
