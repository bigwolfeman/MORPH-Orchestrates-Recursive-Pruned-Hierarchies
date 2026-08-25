# Experiment: does starting the deviation at STATE size stop the SCSE failure?

Status: planned

Testbed: `docs/cookbook/replaying-the-core-takeover.md`, seed `ROLL_step_1750`.

## Question

The 2026-08-25 SCSE port stalled. `lab/divergence/scale_probe.py` and
`lab/divergence/delta_ladder.py` now say why, on trained weights:

- MORPH's blocks are **pre-norm**, so the core map's output size comes from the weights,
  not from its input. Shrink the input 1000x and the output moves 31 %;
  `||stack(D)||/||D||` goes 1.79 -> **1235**.
- SCSE feeds the core the deviation ALONE, and starts it small (`init_scale` 0.1).
  Measured core amplification at loop iteration 0: **13.28**. Every arm below that hands
  the core a full-size input measures **1.16 to 1.68**.

So the method was never given a fair test: its first loop iteration destroyed the anchor.

## Arms

All resume `ROLL_step_1750` and run to step 1900, `deterministic=true`,
`use_kernels=false`, batch 6, seed 0, `alpha_cap` 3.5 — the configuration in which two
identical replays were measured **bit-identical** today (0 of 59 steps differ across 87
probe series) and the control aborts at step **1809**.

| arm | `scse_enabled` | `scse_init_scale` | `scse_input_mode` |
|---|---|---|---|
| control | false | — | — |
| A | true | 1.787 | deviation |
| C | true | 0.1 | state |
| CA | true | 1.787 | state |

`init_scale` 1.787 is not a guess. `delta_ladder.py` solved it from the measured endpoints
(`||Delta_0||` = 0.0543 at `init_scale` 0, 0.5561 at 1.0) to target the control's state RMS
of 0.9513, and the arm reports its realised `||Delta_0||` = 0.992 so the choice is checkable.

## Predictions

Written before any arm ran.

- **V1 (validity).** Every arm builds and trains. The SCSE projections are new parameters
  that `ROLL_step_1750` does not contain, so they load cold. If any arm raises on load, the
  screen is void and nothing below is readable.
- **V2 (validity).** The control reproduces its abort at step 1809, as it did twice today.
- **P1.** Arm **A does not abort** by step 1900. The control aborts at 1809.
- **P2.** Arm A does **not stall**. Its training loss at step 1800 is within **0.5 nats** of
  the control's 5.1407. The full-method arm's failure mode was a stall — loss frozen near
  6.5 while the control reached 4.87 — so an arm that merely stops diverging by refusing to
  learn must not count as a pass.
- **P3.** Arm A's realised `||Delta_0||` is within 20 % of 0.9513, confirming the
  calibration transferred from the forward probe to the live forward.
- **REFUTER.** Arm A aborts at or before step **1809**. Then starting the deviation at state
  size does not help, and the scale diagnosis does not carry into training.

## What each outcome licenses

- **P1 and P2 both hold** — the scale fix is real on this testbed. It licenses a from-scratch
  arm (T3), not a claim about SCSE. A 60-step screen on a checkpoint whose weights never saw
  SCSE cannot say what SCSE does over a full run.
- **P1 holds, P2 fails** — the arm bought its survival by not learning. That is the stall
  again, and it is a failure, not a partial win.
- **REFUTER fires** — the scale diagnosis is forward-only. It explains the 13.28x
  amplification and explains nothing about training.

## Declared confounds

1. **The SCSE parameters start cold at step 1750.** The base weights trained 1750 steps
   without them. This screen therefore measures whether the loop SURVIVES the change of
   parameterisation, not whether SCSE is better. T3 is the real test.
2. **`use_kernels=false` is not the production path.** It runs eager attention at 2.28x
   lower throughput and about half the batch, which changes the gradient-noise scale.
   Conclusions transfer as mechanism, not as step numbers.
3. **Arm C's deviation reaches 1.70x the state size in the forward probe** while arm A's
   contracts to 0.87x. C is expected to be the weaker arm and is run to check that
   expectation, not because it is favoured.
