# Experiment: does starting the deviation at STATE size stop the SCSE failure?

Status: failure

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

---

# Results

Status: failure

Ran 2026-08-25, 18:53 to 19:04. Four runs, sequential, resumed from `ROLL_step_1750`.
Artifacts: [`../results/2026-08-25-scse-delta-scale-screen/`](../results/2026-08-25-scse-delta-scale-screen/).

| arm | ABORT | loss at 1800 | core share, steps 1755-1895 |
|---|---|---:|---|
| control | **1809** | 5.1407 | 0.028 0.103 0.175 0.079 **0.489** 0.161 |
| A — state-size Delta_0 | **1800** | — | **0.743 0.994 0.931 0.999 1.000** |
| C — state input | **none, reached 1900** | 5.1364 | 0.065 0.088 0.063 0.037 0.058 0.034 0.043 0.056 0.032 0.026 0.033 0.034 0.020 0.056 0.027 |
| CA — both | 1842 | 5.1420 | 0.075 0.279 0.491 0.461 0.367 0.465 0.652 0.245 0.343 |

"core share" is the core's share of the pre-clip gradient, the quantity the abort guard
reads. The control's own HEALTHY value at step 1755 is 0.028.

## Verdict: the predictions failed, and they failed backwards

**V1 and V2 pass.** All four arms built and trained; the two SCSE projections loaded cold
through the new name-based optimizer alignment. The control reproduced its abort at 1809.

**P1 FAILED and the REFUTER FIRED.** Arm A aborted at step **1800**, EARLIER than the
control's 1809. P2 and P3 are unreadable for A: it never reached step 1800's log line.

Arm A does not merely fail. It is catastrophic. Core share is **0.743 at the first probed
step and 1.000 by step 1795** — the core owns the entire gradient immediately. Starting the
deviation at state size does not stabilise the loop; it hands the loop everything at once.

**The arm this file predicted would be WEAKER is the one that worked.** Declared confound 3
reads: "Arm C's deviation reaches 1.70x the state size in the forward probe while arm A's
contracts to 0.87x. C is expected to be the weaker arm." Arm C survived to step 1900 with
core share never above 0.088, sitting in the control's own healthy band, while the control
died at 1809 with 0.489.

## The lesson that cost this experiment

**The forward-only deviation NORM does not predict training behaviour.**
`delta_ladder.py` ranked the arms by how big `Delta` grew over eight loop iterations at a
frozen checkpoint. That ranking is exactly inverted against the training outcome:

| arm | forward `D_T/state` | forward verdict | training outcome |
|---|---:|---|---|
| A | 0.87 | best, contracts | aborts at 1800, core share 1.000 |
| CA | 1.74 | worst | aborts at 1842 |
| C | 1.70 | second worst | **survives, core share 0.02-0.09** |

A frozen-weight forward probe measures the size of a quantity. The takeover is about which
parameters the GRADIENT flows to. Those are different questions, and this file assumed one
answered the other. Any future forward probe used to rank arms must state that it is a
screen for feasibility, never for ranking.

## What this does and does not license

**Does:** one more experiment. Arm C is the only intervention in this campaign that has held
the core share flat across the onset window on this testbed.

**Does not:** any claim that SCSE works, or that the takeover is solved. Specifically:

1. **n = 1, one seed, 150 steps.** C reached 1900 because that is where the run ended, not
   because it was shown to be stable. The next test is a long run and a second seed.
2. **The SCSE parameters started cold at step 1750**, on base weights that trained 1750 steps
   without them. This measures whether the loop survives the change of parameterisation.
3. **A frozen core would also show a low core share.** The evidence against that reading is
   that C's loss at step 1800 is 5.1364 against the control's 5.1407, and that C sits at the
   control's own healthy band (0.028) rather than at zero. Over 150 steps that is weak. A
   long run must check that the core is still contributing.
4. `use_kernels=false` is not the production path.

## Updated hypothesis

The failure is not that SCSE evolves a deviation. It is **what the core map is asked to
read**. Feeding the core the deviation alone puts a small tensor into a pre-norm stack whose
output size comes from its weights, and the measured amplification at loop iteration 0 is
**13.28** against 1.16-1.68 for every arm given a full-size input. Feeding it the full state
and accumulating the CHANGE keeps the core in the regime it was trained in.

Making the deviation bigger does not fix that, and arm A shows it makes it far worse.

Next: run arm C long, on two seeds, and measure whether the core is still doing work.
