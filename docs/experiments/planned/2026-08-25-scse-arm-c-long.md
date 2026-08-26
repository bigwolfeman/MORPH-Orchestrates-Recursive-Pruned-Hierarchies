# Experiment: does arm C hold over a long run, and is the core still doing work?

Status: planned

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
