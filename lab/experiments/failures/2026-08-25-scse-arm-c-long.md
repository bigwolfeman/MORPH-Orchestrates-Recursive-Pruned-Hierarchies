# Experiment: does arm C hold over a long run, and is the core still doing work?

Status: failure

Testbed: `docs/cookbook/replaying-the-core-takeover.md`, seed `ROLL_step_1750`.
Prior: [`../failures/2026-08-25-scse-delta-scale-screen.md`](../failures/2026-08-25-scse-delta-scale-screen.md).

## Question

The 150-step screen showed arm C — core input `h* + Delta`, update accumulating the CHANGE
the core made — holding the core's share of the pre-clip gradient at 0.02-0.09 for 150
steps, in the control's own healthy band of 0.028, while the control climbed to 0.489 and
aborted at step 1809.

That screen cannot distinguish two readings:

1. **C stabilises the loop.** The core keeps working and stops running away.
2. **C quiets the core.** A core that contributes nothing also shows a low gradient share,
   and over 150 steps a loss difference of 0.004 nats cannot tell the two apart.

This experiment separates them.

## Method

One arm, one seed. `training.steps=6000` from `ROLL_step_1750`, so the run passes step 1809
— where the control dies — and then continues 4200 further steps.

```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m morph.training.train --config-name tul_a1 \
  training.resume=checkpoints/morph/onset-capture/ROLL_step_1750.pt \
  training.steps=6000 training.ademamix_t_beta3=5000 training.seed=0 \
  training.ademamix_alpha_cap=3.5 training.deterministic=true model.use_kernels=false \
  training.batch_size=6 model.scse_enabled=true model.scse_init_scale=0.1 \
  model.scse_input_mode=state training.grad_probe_every=1 \
  training.abort_core_share=0.5 training.ckpt_every=500
```

`ademamix_t_beta3` is PINNED to 5000, the value the run that produced this checkpoint used.
`base.yaml` ships it as `null`, which falls back to `training.steps`, so leaving it unset
would silently set the beta3 horizon to 6000 and change the optimizer schedule with the run
length. The 150-step screen ran at 1900. This run therefore also RE-TESTS the screen's
result under the corrected schedule when it passes step 1809.

### The core-work test

SCSE makes the core's contribution an explicit tensor: the loop exits at `h = h* + Delta_T`.
Setting `Delta_T = 0` at the exit removes **everything the core loop did**, exactly, with no
architecture change and no change to depth. The gap between the two evaluation losses IS the
core's contribution in nats.

`lab/divergence/delta_ablation.py` measures that gap on each saved checkpoint. Reading the
TREND matters more than any single value: a core being progressively switched off shows a
gap that shrinks toward zero as training proceeds.

## Predictions

Written before the run.

- **V1 (validity).** The run reaches step 6000 or stops on its own guard. A crash is not a
  result.
- **V2 (validity).** The delta ablation is a real intervention: the gap at the FIRST saved
  checkpoint is above 0.05 nats. If zeroing `Delta_T` changes the loss by nothing at the
  start, the probe is not measuring what it claims and nothing below is readable.
- **P1.** The run does **not** abort. The control aborts at 1809.
- **P2.** Core share stays below **0.15** at every probed step from 1809 to 6000. The
  control was at 0.489 by 1795 and the healthy band is 0.028.
- **P3 (the point of this experiment).** The delta-ablation gap does **not** collapse: at
  step 6000 it is at least **50 %** of its value at the first saved checkpoint. A core being
  quieted shows a shrinking gap; a core still working keeps one.
- **P4.** Training loss at step 6000 is **below** its value at step 1800 (5.1364). A run that
  survives by not learning is a stall, not a cure.
- **REFUTER.** The gap at step 6000 is under 25 % of its first value, or under 0.05 nats
  absolute. Then C works by quieting the core, and it is a lobotomy dressed as a fix.

## What each outcome licenses

- **P1-P4 hold** — arm C survives 4200 steps past the control's death with the core still
  contributing. That licenses a second seed and a from-scratch arm. It does NOT license a
  claim that the takeover is solved: one seed, one resume, non-production kernels.
- **P1 holds, P3 fails or the REFUTER fires** — C is a lobotomy. Report it as that. It would
  join `bptt_depth 2` on the list of interventions that "helped" by removing the mechanism.
- **P1 fails** — the 150-step screen did not survive a longer horizon, and the result was a
  delay, not a cure.

## Declared confounds

1. **The SCSE parameters start cold at step 1750**, on base weights that trained 1750 steps
   without them. A from-scratch arm is still the real test.
2. **`use_kernels=false` is not the production path.** Eager attention, 2.28x lower
   throughput, about half the batch, so a different gradient-noise scale.
3. **One seed.** The campaign's own replication writeup says two RUNS at n=1 are unreadable.
   That does not apply to a resume from ONE checkpoint with one change, which is the design
   here, but it does apply to any comparison against a differently seeded run.
4. **No matched long control exists**, because the control dies at 1809. P4 therefore
   compares C against ITSELF over time, not against a healthy peer.

---

# Results

Status: failure

Ran 2026-08-25, 19:20 to 20:45. Artifacts:
[`../results/2026-08-25-scse-arm-c-long/`](../results/2026-08-25-scse-arm-c-long/).

## The runs

The first long run aborted at 1800, which sent this experiment somewhere it did not plan to
go. `ademamix_t_beta3` is the reason and it is worth stating plainly: the checkpoint's own
value is **5000**, `base.yaml` ships the key as `null`, and `null` falls back to
`training.steps`. So the 150-step screen — which set `training.steps=1900` — silently ran
every arm at `t_beta3=1900` and printed `re-applied config hyperparameters: t_beta3
5000→1900`. This experiment PINNED 5000 to be faithful to the checkpoint, and moved the arm
out of the regime where it worked. Two more runs were added to close the square.

| run | `t_beta3` | ABORT | core-share median | max |
|---|---:|---:|---:|---:|
| control | 1900 | **1809** | 0.189 | 0.947 |
| C (screen, 150 steps) | 1900 | none, ran out at 1900 | 0.039 | 0.869 |
| **C1900 (long)** | 1900 | **2506** | **0.024** | 0.924 |
| ctrl5000 | 5000 | **1800** | 0.188 | 0.982 |
| Clong | 5000 | **1800** | 0.440 | 0.897 |

`ctrl5000` and `Clong` differ at all 50 shared steps, so SCSE is changing the forward; the
shared abort step is the guard firing at the earliest moment its 50-step window allows.

**P1 FAILS.** The run aborts. At `t_beta3=1900` arm C reaches step 2506 against the
control's 1809 — **697 extra steps** — and its median core share is 0.024 against 0.189.
At `t_beta3=5000` it buys nothing: both arm and control die at 1800, and the arm's median
core share is WORSE (0.440 against 0.188).

**P2 FAILS.** C1900's core share reaches 0.924, far above the 0.15 bound.

Arm C is a **delay, not a cure**, and the delay is conditional on the optimizer's beta3
horizon. That puts it in the same class as H5 `per_slot_embed` and H16 finer spans.

## The core-work test, and what it actually found

**P3 and P4 are unreadable as written** — the run never reached step 6000. Over the range
that exists, the gap is 0.62 of its first value at step 2500 and the loss falls from 5.1364
to 4.5698, so both point the right way.

**V2 FAILS and the REFUTER FIRES on the pre-registered numbers.** V2 wanted the first gap
above 0.05 nats; it is 0.0260. The REFUTER fires below 0.05 absolute.

Both thresholds were wrong, and the reason matters more than the arm. They were set from an
ASSUMPTION about what MORPH's looped core is worth. That assumption had never been measured.
It has now:

| checkpoint | what was removed | cost in nats |
|---|---|---:|
| ROLL_step_1750 (healthy control) | **prelude**, 4 blocks | **+3.2205** |
| ROLL_step_1750 (healthy control) | **coda**, 4 blocks | **+3.1051** |
| ROLL_step_1750 (healthy control) | **the whole core loop**, 6 shared blocks x 6-8 iterations | **+0.0169** |

Same checkpoint, same 8 fixed validation batches, same ablation machinery. The prelude and
coda ablations are the PROBE'S OWN CONTROL: a core ablation that costs almost nothing is
either a real finding or a broken patch, and one number cannot tell them apart. They cost
over 3 nats each. The machinery works.

Across the healthy control's whole onset ladder the core is worth 0.013 to 0.018 nats, and
at `ROLL_step_1850` — after the takeover — it **collapses to 0.0037**.

Against that baseline, arm C's core is doing MORE work than the healthy control's, not less:

| checkpoint | gap |
|---|---:|
| C1900 step 2000 | **0.0260** |
| C1900 step 2500 | **0.0161** |
| healthy control, 1625-1825 | 0.0130 - 0.0178 |

**Arm C is not a lobotomy.** That was the question this experiment was asked to answer, and
the answer is no.

## Verdict

Failure by the pre-registration: P1 and P2 fail, and V2 and the REFUTER trip on thresholds
built from an unmeasured assumption.

The finding that survives is not about SCSE:

**MORPH's looped core contributes about 0.017 nats. The prelude contributes 3.22 and the
coda 3.11.** The component this entire campaign has been trying to stop from "taking over"
is worth about half a percent of what a four-block prelude is worth. And at takeover its
contribution FALLS to 0.0037 — it captures the gradient while doing less.

## Updated hypothesis

The takeover may not be a stability problem at all. A loop that contributes 0.017 nats has
no gradient signal worth speaking of pulling it toward a useful solution, so what the
optimizer does with those parameters is close to unconstrained. That reframes every cure
tried so far: they were all attempts to stabilise a component that has nearly nothing to
stabilise it.

The next question is not "how do we stop the takeover". It is **"why is the looped core
worth 0.017 nats, and is that true of the shipped recipe or only of this arm and this
checkpoint?"** That must be measured on a non-TUL checkpoint and at a longer-trained one
before any of the above is generalised.

## Declared confounds, restated against the result

1. The SCSE parameters started cold at 1750. Still true; a from-scratch arm is still the
   real test of arm C.
2. `use_kernels=false`, eager, batch 6.
3. **The 0.017 nats is measured at ONE checkpoint family**, `onset-capture`, which is a TUL
   arm at 1750 steps. It is NOT yet known for a plain MORPH run, for the shipped recipe, or
   for a well-trained model. Do not generalise it until it is.
