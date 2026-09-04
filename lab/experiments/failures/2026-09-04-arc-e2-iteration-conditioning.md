# Failure: ARC E2 — iteration conditioning on the constrained forecast arm

Status: failure
Date: 2026-09-04 (frozen; not launched; GPU time is Wolfe's call)
Arc: `2026-09-04-loop-contribution-arc.md`, branch (a) MAP REGIME (the symmetry half).

## Question

A weight-shared core applied to the same input every iteration is a fixed-point
iteration by construction: nothing in the map knows which pass it is on, so the only
thing it can do with a fourth pass is more of what it did on the third. Relaxed
Recursive Transformers (Bae et al. 2024) and Mixture-of-LoRAs (Nouriborji et al. 2025)
report that a per-iteration perturbation of the shared operator recovers most of the
unshared model's gap. The tree already has the cheapest form of that break:
`tul.core_stage_cond=iter`, an AdaLN-Zero modulation of every core layer keyed on the
iteration index (`morph/model/iter_cond.py`; zero-init, bit-identical to Y2 at step 0;
built for DiffusionBlocks and never run for depth earning). Does telling the loop which
pass it is on move earning past iteration 3?

## Arm

| # | run | config | one line |
|---|---|---|---|
| E2 | `to-mnext-y2-iter` | `tul_to_mnext_y2_iter` | Y2 + `core_stage_cond: iter` |

Control: Y2 on disk. Same recipe, seed, rows and readouts as E1; 5000 steps; tripwire on.

## Readouts

As E1, plus: the norm of each core layer's AdaLN gate output per iteration at 5000 (does
the modulation grow, and does it differ across t); the step-ratio PROFILE over t = 1..7
per 500-step window (a second bump past t = 3 is the signature of a pass doing new work).

## Predictions (frozen)

- **P2a.** E2 reaches 5000 with the tripwire silent: **60%** (the gates start at zero
  and the constraint measures the map with the same `stage_cond`; a per-iteration
  operator can also find an expansive pass the hinge, sampled at one random t, misses).
- **P2b.** The gates leave zero: mean AdaLN output norm at 5000 ≥ 1 % of the carrier
  norm on at least one core layer: **80%**.
- **P2c (the bar).** E2 THINKS (forecast K3−K6 > 0.01, CI above 0): **25%**.
  Forecast K1−K6 > 0.02 (Y2: 0.0135): **40%**.
- **P2d.** The step-ratio profile is non-monotone in t on the 4500–4999 window (some
  ratio at t > 3 exceeds the ratio at t = 3): **35%**.
- **P2e.** Last-four val CE within 0.02 of Y2's 4.2809: **60%**. Below it: **35%**.

## Decision rule (binding)

- P2c TRUE ⇒ symmetry was binding; E4 then E5 on E2. Per-iteration LoRA on the ternary
  core is the follow-up lever if the 20k reading is positive but small.
- P2b TRUE and P2c FALSE ⇒ the loop learns to tell its passes apart and still has
  nothing to do with the fourth; (a) is closed in both halves and the arc rests on E3.
- P2b FALSE ⇒ the zero-init gates never moved under the ramp's LR; the arm is rerun
  once with the gate LR scaled 10x before any conclusion (Method amendment, dated).
- P2a FALSE ⇒ the hinge's single-t sample is not a bound on a per-iteration map; the
  penalty is amended to sample every t (code, then a rerun) before E2 is read.

## Not verified before launch

`core_stage_cond=iter` together with `slot_gain_lambda` at the real shape under
torch.compile (the penalty takes `stage_cond` by signature; the CPU tests cover each
alone). A 12-step compiled smoke with no NaN and `tul/gain_est` present is the
pre-launch check, recorded as a Method note.

### Method note, 2026-09-04 14:52 (queued; Predictions untouched)

12-step compiled smoke of `tul_to_mnext_y2_iter` on `/home/wolfe/morph-to` at 19c5dc3:
exit 0, 27 s, no NaN, `loss/gain_est` 0.8751 and `gain_est_max` 0.8769 present at step
12 (the hinge and the iteration conditioning run together under torch.compile at the
real shape, the pre-launch check this file asked for). Step-12 loss 22.2869, the same
print as Y2's dial arms, so the zero-init gates have not moved the loss by step 12.
Queued third in `morph-scratch/arc/run_arc_a.sh`, after E1's two arms.

## Results, draw 1 (2026-09-04 17:00; `to-mnext-y2-iter`, 16:35–17:00, `arc/queue.log`)

DETONATED at step 2556 (`preclip/total` 1.17e5 at 2557; tripwire kill). **P2a FALSE.**
Val CE at 2500: 4.7751 (Y2 at 2500 was 4.62; not matched-eval). The probe file says why,
and it is the P2a-FALSE mechanism this file named:

| window | `preclip/total` | `gain_est` median | `gain_est_max` | hinge active | spike steps |
|---|---|---|---|---|---|
| 1500–1999 | 2.9 | 0.766 | 0.778 | 2.4 % | (132 after 1000 in all, from 1070) |
| 2000–2299 | 7.3 | 0.762 | 0.801 | 14.0 % | |
| 2300–2529 | 5.4–7.0 | 0.535 | 0.56 | 0.0 % | |
| 2530–2557 | 36.7 | 0.608 | 0.703 | 14.3 % | |

The single random-iteration sample read a typical gain of 0.53–0.61 for the last 250
steps (Y2: 0.894) while the run spiked from step 1070 and detonated: with a
per-iteration operator the sampled iteration is not the expansive one. The hinge bound
nothing (median penalty 0.000, active on 5 % of steps). P2b–P2e are not scored on a
detonated draw (no 5000-step checkpoint; the 2500 checkpoint's sweep is on disk for the
record only).

### Amendment 1 (2026-09-04 17:10; the P2a-FALSE rule of the Decision rule above)

The penalty is amended to hinge EVERY grad iteration (`model.slot_gain_all_iters`,
commit after this note; the per-iteration hinges SUM, `gain_est` / `gain_est_max` report
the mean / max over iterations; `tests/test_slot_gain_reg.py`, 3 new tests). The rerun
arm is `to-mnext-y2-iter-all` (`tul_to_mnext_y2_iter_all`). Predictions P2b–P2e stand
as written and are scored on the rerun; P2a is re-stated for the rerun at the same
credence (**60%**) with the added prediction **P2a'**: the all-iterations `gain_est_max`
median over 1000–5000 sits within 0.02 of the target 0.90 (the hinge now binds the
expansive iteration): **65%**. Cost: about 2x the penalty's extra core steps
(Y2's penalty cost 5 % wall clock; expected ≤ 1.15x Y2).

### Method note, 2026-09-04 17:14 (rerun launched; Predictions untouched)

12-step compiled smoke of `tul_to_mnext_y2_iter_all` at worktree 2f2ca98: exit 0, 31 s,
no NaN, step-12 loss 22.2869 (the same print as every Y2 arm), probe row carries
`loss/gain_est` 0.876 and `loss/gain_est_max` 0.889 (the max now ranges over every grad
iteration; the two differ, so more than one iteration is measured). `gain_n_iters` is
in the forward's output but NOT in the local probe row (train.py's probe row lists its
keys by name; added for later runs, this draw does without it). Draw
`to-mnext-y2-iter-all` started 17:13:35 by `arc/run_arc_a3.sh`; tripwire on; readout
and the E0 sweep follow in the same window.

## Results, draw 2 (2026-09-04 17:40; `to-mnext-y2-iter-all`, 17:13–17:38, worktree 2f2ca98)

The tripwire killed it at 1684 (`preclip/total` 17034). It was NOT a detonation. The
every-iteration hinge did what Amendment 1 asked: `gain_est_max` (max over the eight
iterations) median 0.887–0.897 from 200 to 1690, hinge active 2–10 % of steps, ZERO
spike steps (`loop/cot_ratio` > 30) against the first draw's 132 — the spike train is
gone. The kill was a single-step outlier: 1682 core 6.9 (gain max 0.94), 1683 1.1, 1684
core 17031 with the max gain 1.80 on that one step (penalty 13.5), then 1.3, 1.6, 1.8,
1.4, 1.8, 1.1, 0.3 — back at the 2.6 baseline by 1691, one blip of 31 at 1692 (gain 1.01),
and calm 2.4–3.1 through 1720 where the poll killed it. Val CE 5.29 / 5.12 / 5.12 at
1000 / 1250 / 1500 (Y2: 5.19 / 4.77 / 4.77 at 2000–2500 — different steps, no reading).
Pace 67 steps/min (0.61x Y2; the hinge at eight iterations).

Scored: **P2a FALSE by the letter** (killed) and **P2a' TRUE** (the all-iterations max
gain sat within 0.02 of 0.90). P2b–P2e unmeasured (no 5000 checkpoint; the 1500
checkpoint's sweep is on disk for the record).

### Amendment 2 (2026-09-04 17:45; Method only, Predictions untouched)

The single-step trip rule (`preclip/total` > 1e4 at step ≥ 200) was validated on
unconstrained runs (17/17 detonations, 0/44 false positives). Under the every-iteration
hinge a one-step outlier snaps back within seven steps, so that rule produced its first
false positive. The constrained arms use a SUSTAINED rule from here
(`lab/divergence/tripwire_sustained.py`): DETONATED iff any row > 1e5, or two rows > 1e4
within 40 steps; one recovered row > 1e4 is an EXCURSION, not a trip. Calibrated on
the real detonation (81617 at 2556 and 116930 at 2557: trips at 2557) and on this draw
(EXCURSION, no trip); every other arc-a probe file keeps its verdict.

The rerun is RESUMED from `to-mnext-y2-iter-all/step_1500.pt` as `to-mnext-y2-iter-all-r`
(`training.resume`, the full resume: model, step, RNG, scaler, optimizer state), queued
behind E0 in `arc/run_e2_resume.sh`. P2a is re-stated for the resumed run: reaches 5000
without a SUSTAINED trip: **65%**. Unverified: that the resume continues the loss trace
bit for bit (the 40k-continuation prereg named this check and it was never run; the
resumed run's first 50 steps against wandb's history for 1501–1550 of the killed draw
is the check, recorded in Results).

### Method note, 2026-09-04 17:52 (the resume check named in Amendment 2)

`to-mnext-y2-iter-all-r` resumed from step_1500 ("Resumed model+scaler+RNG from
checkpoint step 1500; next_step=1501", optimizer rebuild not needed). Its first probe row
(1501) equals the killed draw's bit for bit (`preclip/total` 3.4626, `gain_est` 0.81905 on
both); from 1502 the two drift apart at the 1e-3 relative level and decorrelate over
the next tens of steps (`memory: MORPH runs decorrelate in 11 steps at a fixed seed`).
The resume restores the state; the continuation is a repeat draw, not a replay. wandb
resumed the killed draw's run id, so its history for 1501–1705 keeps the killed draw's
rows and rejects the resumed ones as non-monotonic; the local `probe.jsonl` is the record.

## Results, draw 3 (2026-09-04 18:45; `to-mnext-y2-iter-all-r`, resumed from 1500, 17:48–18:39; readouts 18:39–18:45; files in `results/2026-09-04-arc-a/`)

Reached 4999 with the sustained tripwire silent (verdict AMBIGUOUS: thirteen isolated
rows over 100, the largest 2510 at 4667, every one back at baseline the next step; zero
spike steps). The every-iteration hinge was barely needed: the per-iteration mean gain
sat at 0.796 (p10 0.786, p90 0.814), the max over iterations at a median of 0.842
(p90 0.871, max 1.38), hinge active on 2.3 % of steps. Pace 67 steps/min.

| reading | iter-all-r @5000 | Y2 @5000 (control) |
|---|---|---|
| forecast `mux_local` K1−K6 | +0.0087 [+0.0073, +0.0102] | +0.0135 [+0.0118, +0.0154] |
| forecast K3−K6 | +0.0001 [−0.0003, +0.0006] | +0.0002 [−0.0004, +0.0008] |
| forecast loss d1 / d3 / d4 / d8 | 6.7610 / 6.7524 / 6.7520 / 6.7532 | 6.7632 / 6.7499 / 6.7492 / 6.7514 |
| token K1−K6 | +0.0004 | +0.0003 |
| step-ratio profile t1..t7 (4500–4999 medians) | 0.73 0.55 0.44 0.37 0.33 0.32 0.31 | 0.82 0.60 0.54 0.51 0.49 0.48 0.47 |
| last-four val CE | 4.3082 | 4.2809 |
| plan worth offset 0, zero / shuffle | +0.066 / +0.040 | +0.074 / +0.034 |
| AdaLN gates at 5000 | `to_mod` weights rms 0.0105 (zero-init), bias max 0.05, every core layer | — |

Scored (P2a on the resumed run per Amendment 2):

| prediction | credence | verdict |
|---|---|---|
| P2a reaches 5000 without a sustained trip | 65% | TRUE |
| P2a' all-iterations max gain within 0.02 of 0.90 | 65% | FALSE (0.842; the map sat BELOW the hinge on its own) |
| P2b the gates leave zero | 80% | TRUE on the weights (the 1 %-of-carrier clause was not measured) |
| P2c THINKS (K3−K6 > 0.01, CI > 0) | 25% | FALSE (+0.0001); K1−K6 > 0.02: FALSE (+0.0087, below Y2) |
| P2d ratio profile non-monotone | 35% | FALSE (monotone, and more contractive than Y2 at every t) |
| P2e val within 0.02 of Y2 | 60% | FALSE (+0.027); below Y2: FALSE |

## Verdict

failure (P2c, P2d, P2e and P2a' falsified; the "P2b TRUE and P2c FALSE" branch of the
Decision rule). The gates learn to tell the passes apart and the loop still has
nothing to do with the fourth: held stable, the per-iteration operator earns LESS than
the shared one (K1−K6 0.0087 vs 0.0135) at a val price of 0.027 and a 1.6x pace price.
The +0.0077 K3−K6 read at draw 1's pre-onset 2500 checkpoint was the expansive
iteration-0 map at work (fp32 gain 2.97 there), not depth earning: it vanished the moment
every iteration was held under 0.90 (−0.0009 at 1500, +0.0001 at 5000).

## Updated hypothesis

Branch (a) of the arc is closed in both halves. Neither slowing the map's contraction
(E1) nor breaking the weight-sharing symmetry (E2) makes a stable slot loop earn past
iteration 3 on the forecast target; every earning this campaign has seen past
iteration 3 came with an expansive map, and it left with it. The one remaining lever on
the loop itself is the TARGET (E3, staged memory-then-forecast), and E0 says the target
must be compute-limited in its whole loss, which the forecast is not. The every-iteration
hinge and the sustained tripwire stay in the tree as the instruments for any future
per-iteration arm.
