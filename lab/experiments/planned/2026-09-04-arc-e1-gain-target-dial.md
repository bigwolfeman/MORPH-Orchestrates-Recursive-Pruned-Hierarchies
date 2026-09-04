# Planned: ARC E1 — the gain-target dial (0.90 on disk, 0.95, 0.98)

Status: planned
Date: 2026-09-04 (frozen; not launched; GPU time is Wolfe's call)
Arc: `2026-09-04-loop-contribution-arc.md`, branch (a) MAP REGIME.

## Question

Y2 holds the slot map's typical gain at 0.894 and its trajectory still contracts at
0.45–0.54 per iteration at iteration 3 and 0.33–0.47 at iteration 7, and it earns
nothing past iteration 3 (forecast K3−K6 +0.0002). Two readings are possible. (i) The
typical-direction gain sets the trajectory's contraction, so moving the target toward 1
slows the contraction and the loop keeps earning past 3. (ii) The trajectory's
contraction is set by the target (a forecast the loop finishes in three passes) and the
typical gain is a bystander, so the dial changes spikes and nothing else. The literature
window for "as close to 1 as training allows" is 0.93–0.98 (Bai, Koltun, Kolter 2021,
Jacobian regularization; the report's synthesis). Two arms read it.

## Arms

| # | run | config | one line |
|---|---|---|---|
| E1-95 | `to-mnext-y2-g95` | `tul_to_mnext_y2_g95` | Y2 with `slot_gain_target: 0.95` |
| E1-98 | `to-mnext-y2-g98` | `tul_to_mnext_y2_g98` | Y2 with `slot_gain_target: 0.98` |

Control on disk: Y2 (`to-mnext-y2`, target 0.90; `successes/2026-09-04-tul-forward-levers.md`).
Same recipe, seed, rows and readouts as the forward-levers pair; 5000 steps; tripwire on.

## Readouts

`tul/gain_est` median and hinge-active share over 1000–5000; spike steps (`loop/cot_ratio`
> 30) after 1000; `loop/delta_ratio_t{3,7}` medians per 500-step window; `jac/rms_t3` at
5000 (fp32, fixed batch); `core_depth_sweep.py` depths 1..8, 480 rows, batch 3
(forecast `mux_local` K1−K6 and K3−K6 with CIs; token K1−K6); `worth_profile.py` 192
rows; last-four val CE; wall clock against Y2's 45.6 min.

## Predictions (frozen)

- **P1a.** E1-95 reaches 5000 with the tripwire silent: **70%**. E1-98: **45%**.
- **P1b.** The hinge binds where it is set: median `tul/gain_est` within 0.01 of the
  target on each arm that survives past 2000: **75%**.
- **P1c (the reading).** The trajectory follows the dial: median `loop/delta_ratio_t3`
  over 1000–5000 on E1-98 is ≥ 0.60 (Y2: 0.45–0.54): **35%**. The alternative, ratio
  stays ≤ 0.55 on both arms: **55%**.
- **P1d (the bar).** Any dial arm THINKS (forecast K3−K6 > 0.01, CI above 0, at its last
  checkpoint): **15%**.
- **P1e.** Spike steps after 1000 on E1-98 number ≥ 5: **55%**. On E1-95: **30%**.
- **P1f.** Last-four val CE on E1-95 within 0.02 of Y2's 4.2809: **60%**.

## Decision rule (binding)

- P1c TRUE and P1d TRUE ⇒ the regime was binding; E5 (20k) on the dial arm that THINKS,
  E4 on it first.
- P1c TRUE and P1d FALSE ⇒ the map can be slowed and the loss does not care: (a) is not
  binding on this target; E3 (target) becomes the arc's main line and E2 stays as the
  cheap symmetry check.
- P1c FALSE ⇒ the typical gain does not set the trajectory's contraction; the constraint
  stays at 0.90 for stability (cheapest of the three) and the dial is retired.
- E1-98 tripped before 2000 ⇒ 0.98 is over the edge for this recipe; its readings are
  taken at its last healthy checkpoint and P1c is scored on E1-95 alone.

## Not verified before launch

The hinge at 0.98 against the spike mechanism (the capture saw spikes at the crossing of
1.00 with a 2 % finite-difference gain estimate; 0.98 + eps 0.02 touches it); the wall
clock of the dial arms (same code path, expected 1.0x Y2).
