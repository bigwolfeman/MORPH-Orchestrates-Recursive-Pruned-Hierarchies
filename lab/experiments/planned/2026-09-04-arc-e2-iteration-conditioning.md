# Planned: ARC E2 — iteration conditioning on the constrained forecast arm

Status: planned
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
