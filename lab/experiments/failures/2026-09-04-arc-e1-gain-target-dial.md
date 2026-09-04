# Failure: ARC E1 — the gain-target dial (0.90 on disk, 0.95, 0.98)

Status: failure
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

### Method note, 2026-09-04 14:52 (launched; Predictions untouched)

12-step compiled smokes on `/home/wolfe/morph-to` detached at 19c5dc3: g95 exit 0, 29 s,
no NaN, `loss/gain_est` 0.8731 at step 12; g98 exit 0, 28 s, no NaN, `gain_est` 0.8730.
Both print the same step-12 loss 22.2869 as each other (the hinge is inactive below its
target at init, as designed). Draws launched in the order g95, g98 by
`morph-scratch/arc/run_arc_a.sh` (E2's arm follows in the same queue); tripwire on;
one trainer.

## Results (2026-09-04 17:20; g95 14:51–15:37, g98 15:37–16:23; readouts 16:23–16:35; files in `results/2026-09-04-arc-a/`)

Both arms reached 4999. g95 with the tripwire silent (max `preclip/total` 134 at 279);
g98 reached 4999 with an AMBIGUOUS verdict (1003 at 1865, under the 1e4 trip) and 121
spike steps after 1000.

| arm | hinge | `gain_est` median 1000–4999 (p10 / p90 / max) | hinge active | step ratio t3 / t7, median 1000–4999 | spike steps > 1000 | last-four val CE | `jac/rms_t3` at 5000 |
|---|---|---|---|---|---|---|---|
| Y2 (control) | 0.90 | 0.894 (0.892 / 0.897 / 0.909) | 24.6 % | 0.505 / 0.418 | 0 | 4.2809 | — |
| g95 | 0.95 | 0.913 (0.903 / 0.930 / 0.976) | 4.8 % | 0.649 / 0.569 | 2 | 4.3291 | 0.895 |
| g98 | 0.98 | 0.929 (0.910 / 0.944 / 1.084) | 7.0 % | 0.632 / 0.612 | 121 | 4.4328 | 0.921 |

Depth sweeps at 5000 (480 rows, batch 3) and worth profiles (192 rows):

| arm | forecast K1−K6 | forecast K3−K6 | forecast loss d1 / d3 / d5 / d8 | token K1−K6 | plan worth offset 0, zero / shuffle |
|---|---|---|---|---|---|
| Y2 | +0.0135 [+0.0118, +0.0154] | +0.0002 [−0.0004, +0.0008] | 6.763 / 6.750 / 6.749 / 6.751 | +0.0003 | +0.074 / +0.034 |
| g95 | +0.0138 [+0.0120, +0.0158] | +0.0014 [+0.0007, +0.0021] | 6.777 / 6.764 / 6.763 / 6.764 | +0.0005 | +0.061 / +0.033 |
| g98 | +0.0180 [+0.0161, +0.0201] | +0.0005 [−0.0003, +0.0013] | (sweep JSON) | +0.0006 | (worth JSON) |

Scored:

| prediction | credence | verdict |
|---|---|---|
| P1a g95 silent to 5000 / g98 | 70% / 45% | TRUE / TRUE by the letter (no trip; AMBIGUOUS at 1e3) |
| P1b median `gain_est` within 0.01 of the target | 75% | FALSE on both (0.913 vs 0.95; 0.929 vs 0.98) |
| P1c g98 median ratio t3 ≥ 0.60 | 35% | TRUE (0.632; g95 0.649) |
| P1d any dial arm THINKS | 15% | FALSE (best K3−K6 +0.0014) |
| P1e spike steps ≥ 5: g98 / g95 | 55% / 30% | TRUE (121) / FALSE (2) |
| P1f g95 last-four val within 0.02 of Y2 | 60% | FALSE (+0.048) |

## Verdict

failure (P1b, P1d, P1f falsified; the decision rule's "P1c TRUE and P1d FALSE" branch).
The map does not climb to a looser hinge on its own in 5000 steps: the hinge at 0.95 and
0.98 is a ceiling touched on under a tenth of steps, and the map settles 0.02–0.035 above
Y2's 0.894. That small change in typical gain comes with a large change in the
trajectory's settling (ratio at t3 0.50 → 0.63–0.65) and with a val-CE price (+0.05,
+0.15) and, at 0.98, a spike train. The forecast column moves by a thousandth of a nat.
(a)'s regime half is not what binds this target: the map can be slowed and the loss
does not care.

## Updated hypothesis

The gain hinge is a stability dial, not an earning dial; it stays at 0.90 (cheapest of
the three) for every later arm. What DID move earning the same afternoon is the other
half of (a): E2's iteration conditioning read forecast K3−K6 +0.0077 at its pre-onset
2500 checkpoint (E2's Results) — the symmetry of the shared operator, not its
contraction rate, is the binding half.
