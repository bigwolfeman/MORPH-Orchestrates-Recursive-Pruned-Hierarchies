# Experiment: does bounding the core map's spectral norm cure the TUL core takeover?

Status: planned

## Question

The [RCA](../results/2026-08-24-tul-takeover-rca.md) established the mechanism — the
per-block BACKWARD gain of the looped core crosses 1 and compounds `n_core x bptt_depth`
= 24 blocks deep — and showed that no forward-side intervention reverses it once the
weights are in that state. What it did NOT establish is causality: an intervention that
holds the block gain below 1 and thereby PREVENTS the takeover.

Does a soft spectral-norm cap on the core MLP linears, applied from step 0, prevent the
takeover, and does it prevent the loss turnaround that is the actual harm?

## Background: what the harm is

The takeover is not a loss spike. It is a loss TURNAROUND. Pulled from wandb
`adew-me/morph-tul`, train loss at its minimum versus at the end of the run:

| run | config | min loss | end loss | outcome |
|---|---|---:|---:|---|
| `tul-a1` `82easori` | alpha_cap 3.5, b14 | 5.013 @ 900 | 6.814 @ 4500 | turned around |
| `tul-a1` `cyushbhr` | alpha_cap 3.5, b14 | 4.956 @ 1100 | 6.622 @ 5720 | turned around |
| `tul-a1r` `8e49z6u8` | alpha_cap 3.5, b14, seed 1 | 5.278 @ 600 | 6.863 @ 3120 | turned around |
| `tul-a1r` `0ujvtukf` | **alpha_cap 1.0**, b12, seed 1 | 4.477 @ 1600 | 6.857 @ 4000 | turned around |
| `tul-a1` `c23dwx4a` | **alpha_cap 1.0**, b12, seed 0 | — | 3.62 @ 19200 | healthy to 20k |
| `tul-a0` `l4apqgyo` | alpha_cap 3.5, b14, TUL off | — | 3.21 @ 19000 | healthy to 20k |
| `tul-a3` `4lb85o25` | alpha_cap 3.5, b14, no core | — | 3.17 @ 19000 | healthy to 20k |

Two facts fix the target. First, `ademamix_alpha_cap: 1.0` — the setting `tul_short.yaml`
calls "THE TUL DIVERGENCE FIX" — holds seed 0 for 20k steps and does NOT hold seed 1,
which turned around at 1600 and aborted at 4140. It is not a cure. Second, arms A0 (TUL
off) and A3 (no core at all) are healthy at the same optimizer settings, so the failure
needs the core to be looping over 64 SLOT positions rather than 1024 token positions.

## Hypothesis

The core map `f_theta` becomes expansive — `sigma_max(J_core) > 1` — and because the core
is weight-shared the loop applies that same expansive operator 24 times per backward.
Bounding `sigma_max(W)` of the core MLP linears bounds `sigma_max(J_core)` through the
submultiplicative inequality, holds the realized block gain below 1, and prevents the
takeover. It costs no CE, because the healthy operating point is already below the cap and
the penalty is exactly zero there.

## What is already in hand, and was NOT predicted here

Honesty about the order of events. The deterministic microcosm arm `spec-scratch` RAN
BEFORE this file was written — it was the pending arm of the RCA, and its verdict rule
("core share above 0.5 on more than 30 % of the last 50 probed steps") was fixed in
advance there, but no numeric prediction was. Its result is therefore the OBSERVATION that
motivates this experiment, not evidence for it:

* At the control's abort step 1866, the control sits at core share 0.602 with block gain
  1.303 (r2 0.898); `spec-scratch` sits at 0.011 with block gain 0.968 (r2 0.161) and does
  not abort. Train loss at step 1800: control 5.1289, `spec-scratch` 5.1348.

Everything below is predicted before its run.

## Predictions

The cure is the soft hinge already in `morph/training/spectral_penalty.py` at
`spectral_penalty_cap: 1.5`, `spectral_penalty_lambda: 10.0`, ON from step 0.

P4. **Second seed, real configuration.** At `tul_a1r` (seed 1, batch 12, kernels on,
    `alpha_cap` 1.0, `t_beta3` 20000) — the configuration in which `alpha_cap` 1.0 FAILED —
    the cure run's train loss at step 4000 is BELOW its own minimum-so-far plus 0.3 nats,
    where the control `0ujvtukf` was 2.38 nats above its minimum.

P5. **The control reproduces.** A same-code control at that configuration turns around:
    its loss at step 4000 is at least 1.0 nat above its own minimum.

P6. **Mechanism.** `sigma_max(J_core)` measured by `morph/training/core_jacobian.py` is
    above 1 on the taken-over checkpoints (`ROLL_step_1850`, `TAKEOVER_step_1866`) and
    below or near 1 on the healthy ones (`ROLL_step_1625`..`1700`), and the rise leads the
    core share.

P7. **The cap is derivable, not tuned.** Projecting the core linears of the SICK
    checkpoint onto `sigma_max(W) <= c` and re-measuring `sigma_max(J_core)` gives a
    monotone curve, and the largest `c` at which `sigma_max(J_core) <= 1` lands between
    1.0 and 2.0 — the region the chosen cap of 1.5 sits in.

P8. **Dose response, not generic regularisation.** A second arm at `cap` 3.0 with the
    same lambda — a cap that barely binds, since the control's worst core linear reaches
    3.41 — TAKES OVER by step 6000 under the same verdict rule. A strong regulariser that
    happened to move the trajectory would help at 3.0 too; only a spectral bound has to
    bind to work.

P9. **Can the spectral cap stand alone?** A third arm restores `ademamix_alpha_cap` to
    3.5 — the setting `tul_short.yaml` records as diverging 5/5, at abort steps 2080,
    3240, 4540, 5900 and 6200 — and keeps only the spectral cap. Expectation: it does NOT
    take over by 6000. This one is a genuine open question rather than a confident
    prediction; the two fixes act on different objects (the optimizer's slow-EMA weight
    versus the map's spectral norm) and nothing measured yet says the second subsumes the
    first.

Falsifiers. P4 failing makes the microcosm result seed 0 luck, exactly the way
`alpha_cap` 1.0 failed, and kills the cure. P5 failing makes the seed-1 pair unreadable
and the experiment must be redone. P6 failing means the block gain is not an operator-norm
phenomenon and the whole mechanism story is wrong, cure or no cure.

## Method

All runs on the 5090.

* Microcosm arms: `--config-name tul_a1`, `training.deterministic=true`,
  `model.use_kernels=false`, `training.batch_size=6`, `training.seed=0`,
  `training.steps=2100`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported before the process
  starts, `training.grad_probe_every=1`. Control is the existing `onset-capture` probe
  mirror. Cure adds `training.spectral_penalty_cap=1.5`,
  `training.spectral_penalty_lambda=10.0`.
* Seed-1 arms: `--config-name tul_a1r`, stock `tul_short` settings (batch 12, kernels on,
  `alpha_cap` 1.0), `training.steps=4200`, `training.ademamix_t_beta3=20000` so the
  optimizer schedule matches the 20000-step control `0ujvtukf`, `grad_probe_every=25`.
  Control and cure differ ONLY by the two penalty keys. Arms, in the order they run:
  `cure-a1r-ctrl` (6000), `cure-a1r-spec` (12000, cap 1.5), `cure-a1r-cap30` (6000, cap
  3.0), `cure-a35-spec` (6000, cap 1.5 with `alpha_cap` back at 3.5).
  `spectral_penalty_log_every` is 100 on EVERY arm including the control, so sigma_max of
  the core MLP linears is on the record whether or not the penalty is active.
* Mechanism: `lab/divergence/jac_ladder.py` over
  `checkpoints/morph/onset-capture/ROLL_step_*.pt`, one fixed batch and one fixed depth
  draw for every rung, 300 power iterations, convergence residual reported with every
  value.
* Cap derivation: the same script with `--sweep-caps` on `ROLL_step_1850.pt`, projecting
  each core 2-D weight onto the spectral ball before measuring.

The verdict rule for "took over" is the one already fixed in the RCA: core share above 0.5
on more than 30 % of the last 50 probed steps.

## Method amendment, 2026-08-24 10:40 — the seed-1 control is a coin flip

Reason, written while the control was still running and before any cure arm at this
configuration had started. `cure-a1r-ctrl` (seed 1, `alpha_cap` 1.0, no penalty) is at step
3500 with val CE falling monotonically — 5.42, 4.80, 4.62, 4.55, 4.16, 4.10, 4.05 — and
`spec/sigma_max` at 2.53, which tracks the HEALTHY seed-0 run (2.58 at step 3000) and not
the seed-1 run it was meant to reproduce (4.87 at step 3000). At `alpha_cap` 1.0 the
failure is roughly a coin flip: it held seed 0 for 20000 steps and killed seed 1 at 4140.
A single arm at a configuration that fails half the time cannot carry a comparison, whether
it fails or not.

So the deciding pair moves to `ademamix_alpha_cap` 3.5, which `tul_short.yaml` records as
diverging 5/5 at abort steps 2080, 3240, 4540, 5900 and 6200, with a matched control run
under the same code on the same day: `a35-ctrl` and `a35-spec`, 7000 steps each, differing
only by the two penalty keys. `cure-a1r-ctrl` still runs to 6000 and is reported whatever
it does — a control that failed to fail is a fact about the phenomenon's variability, not
an arm to discard. P5 is judged against `a35-ctrl`; P4 is judged against `a35-spec` at
`alpha_cap` 3.5, which is a HARDER test than the one predicted, since it removes the
incumbent fix as well. The `cure-a1r-spec` arm keeps its role as the CE-cost measurement,
at 10000 steps against the free uncapped control `c23dwx4a` (train loss 3.74 at 12000).

Nothing in Predictions is edited.

## Method amendment 2, 2026-08-24 13:58 — the lever is not in the weights

Four size-based interventions have now failed at `alpha_cap` 3.5, all against the same
control, all scored at a common step 2050 by the same rule. Written before the arm below
started.

| arm | share criterion fires | val CE rise |
|---|---:|---:|
| `a35-ctrl` no control | 1700 | +0.623 |
| `a35-spec` soft cap 1.5 | 1225 | +2.737 |
| `a35-cap30` soft cap 3.0 | 1225 | +2.176 |
| `a35-proj15` hard cap 1.5, MLP | 1625 | +3.496 |
| `a35-proj15attn` hard cap 1.5, MLP + attention | 1675 | +1.108 |

Two measurements say why, and both were taken before this amendment:

* **No core weight's spectral GAP grows.** `sigma_1 / sigma_2` per core linear, by deflated
  power iteration on the same ladder: median 1.069 -> 1.132 across the whole onset, and the
  WORST gap falls, 2.647 -> 2.421. Power iteration aligns at rate `(sigma_1/sigma_2)^k`, so
  if no matrix's gap is opening, no single matrix's spectrum is driving the alignment.
* **The concentration that IS measured is over POSITIONS.** The cotangent falls from 13
  effective slot positions to 2.5, while the same weights on the token path keep 26 to 59.

So the alignment is a property of the composition and of how few positions the cotangent is
spread over, not of any weight matrix's spectrum. Every intervention tried so far acts in
feature space. The next one acts in position space.

P10. **The slot budget.** `tul.max_slots` 64 -> 128 at `alpha_cap` 3.5, 7000 steps, against
     `a35-ctrl`. Prediction: the share criterion does NOT fire before step 3400, twice the
     control's 1700, and validation CE at 2000 is below its own minimum plus 0.3 nats where
     the control is +0.623. Confound declared in advance: more slots means ~12 % more tokens
     per row (1161 against 1033), so a CE comparison against the control is not
     token-matched and only the takeover verdict is clean.

Nothing in Predictions is edited.

## Risks

* n = 1 per arm. Bit-reproducibility makes the microcosm pair a controlled comparison, not
  a sample of the phenomenon's variability. The seed-1 arms run with kernels on and are
  NOT bit-reproducible; only a difference much larger than the 6.5 % run-to-run spread
  measured in [the failed replication gate](../failures/2026-08-23-tul-run-replication.md)
  is readable there, which is why P4/P5 are stated in nats and not in step numbers.
* The cap may bind harder later in training than it does at 2100 steps. 4200 steps is
  still far short of 20000, so "no CE cost" is a claim about the early schedule only.
* The penalty covers the core MLP linears and NOT the attention projections, which the RCA
  found were the largest movers. If it works anyway, that is worth explaining rather than
  assuming.
