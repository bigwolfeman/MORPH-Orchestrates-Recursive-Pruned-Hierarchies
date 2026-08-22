# Experiment: is `ademamix_alpha_cap=1.0` a cure or a delay?

Status: planned

Follows `../failures/2026-08-22-tul-divergence-cause.md`, whose own "Next planned
experiment" section names this measurement. Costs zero training: it reads checkpoints
that already exist.

## Question

`tul_short.yaml` now ships `training.ademamix_alpha_cap: 1.0`. That value carried two
TUL arms to 20000 steps where the uncapped recipe detonated 5/5. Tonight's campaign runs
at **batch 12**, not the batch 14 that produced that evidence. Batch size sets the
gradient noise scale, which is the drive feeding the slow EMA that the cap throttles.

So: does the cap **suppress the mechanism**, or does it only **slow the approach** to the
same cliff? A suppressed mechanism transfers to batch 12. A slowed approach does not.

## The quantity

Task #276 states the mechanism as subspace alignment, not magnitude. The six core blocks
each stay individually calm while their top singular subspaces rotate into alignment, so
the blocks chain multiplicatively instead of cancelling. The order parameter is

    ORDER = sigma_max(J of the whole core step) / max_i sigma_max(J of block i)

Reference values from Task #276: **~0.90 healthy** (blocks cancel), **~9.5 at the cliff**
(blocks aligned). Those come from a different model, so this run also measures A0 and A3
— arms that never diverged — to establish the healthy value **for this architecture**.

Every per-linear sigma number in the prior RCA is a different quantity. Alignment can
inflate ORDER while every per-linear sigma stays flat, which is precisely why the RCA
concluded "sigma is a correlate, not the trigger".

## Method

`ignore/perf/order_param.py`. Estimator and both validation gates ported verbatim from
the already-validated `00-MORPH-ademamix-b1zero/ignore/E2_sigma_on_ckpt.py`.

- sigma_max by power iteration on JtJ: forward-difference JVP, autograd VJP, k=60.
- fp32 throughout. The FD step is 1e-3 relative, which is fine in fp32 and noise in bf16.
- **Gate A** — the estimator must recover sigma_max=6.04 on a non-normal matrix whose
  spectral radius is 0.5. This separates sigma_max from rho; an estimator that returns
  rho would pass a normal-matrix test and fail here.
- **Gate B** — `_apply_core_step` must reproduce the real forward's iteration-0 core
  output to < 1e-4. Without this the probe measures a map the model does not run.
- Either gate failing aborts the script rather than printing a number.
- Fixed seed 1234, batch 2, seq 256, identical token ids for every checkpoint. Random
  ids, matching the reference instrument. Off-distribution for a language model, so the
  ranking gets a real-text spot check on two checkpoints afterward.
- `slot_layout=None` for every checkpoint. This measures the **weight-space** map at one
  common operating point. The A1 arms are therefore probed off their training
  distribution; the confound-free comparison is A1-vs-A1, and A0/A3 are context only.
- Checkpoints: `tul-a1/DIVERGED_step_4540`, `tul-a1r/DIVERGED_step_3240`,
  `tul-a1-acap1/step_{5000,10000,15000,20000}`, `tul-a0/step_{5000,20000}`,
  `tul-a3/step_{5000,20000}`.

## Already known before the predictions below

The smoke run measured `acap1_5k`: composition 76.18, worst block 16.93,
**ORDER = 4.500**, realized gain 1.270. Both gates passed, Gate B at exactly 0.000e+00.
That number is data, not a prediction, and is excluded from scoring.

## Predictions

1. Both diverged controls score **ORDER > 6**, above every surviving arm's value at a
   comparable step. If a diverged checkpoint scores below the cured arm, the order
   parameter does not separate divergence in this model and the whole framing is wrong.
2. The cured trajectory 5k -> 20k **rises by less than 1.0 absolute** (4.50 -> under
   5.5). A rise past 6 means the cap delays rather than cures.
3. A0 and A3 — the arms that never diverged — sit **below 4.0** at both 5k and 20k, and
   their 5k -> 20k drift is smaller than A1-acap1's.
4. Per-block sigma stays within a factor of 2 across every checkpoint, diverged included.
   This is the alignment claim: the runaway lives in the composition, not the blocks. If
   the diverged controls also show blown-up per-block sigma, then magnitude explains the
   divergence and alignment is not needed.

## Decision rule

- Predictions 1 and 2 both hold -> the cap suppresses the mechanism. Batch 12 is
  acceptable risk. Run the arms.
- Prediction 1 holds and 2 fails -> the cap only delays. Batch 12 changes the drive, so
  the arms need a live ORDER readout with an abort threshold before they run overnight.
- Prediction 1 fails -> the order parameter does not discriminate here. Report that
  plainly and do not use it as a gate.

## Risks

- n=1 per condition. This ranks checkpoints; it does not prove causation, exactly the
  error the parent experiment made. The claim stays at "the order parameter does / does
  not separate the diverged from the survivors".
- Random-token operating point may not reflect real-text Jacobians. Mitigated by the
  spot check, not eliminated.
- A0 is a no-TUL arm probed at its native operating point while A1 is probed off its own.
  Any A0-vs-A1 difference carries that confound. A1-vs-A1 does not.
