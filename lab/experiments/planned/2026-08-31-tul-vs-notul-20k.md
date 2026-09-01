# Planned: TUL vs no-TUL at 20k on the loop-killer winner recipe

Status: planned
Date: 2026-08-31 (frozen before launch; Wolfe: "20k run then... Lets get this
up and running". The honest re-run of the original 30k ask, shortened by
decision, on the campaign winner recipe.)

## Question

With the leak fixed and GLA+cap removed (the campaign winner: retention off,
spectral_project_cap 0, carry none), does TUL earn its keep at scale — on the
token axis, the wall-clock axis, or neither?

## Method

Two arms, sequential on the 5090, same panel flags (batch 6, seed 1,
alpha_cap 3.5, t_beta3 3500, eval_every 250, gen_every 0, grad_probe_every 1),
training.steps=20000, ckpt_every=2000 with post-run prune to step_20000.
- **A: tul-20k** — `tul_g0c0` (tul_l2 + retention=false + cap=0; TUL from
  step 0, TG-restrict, eager kernels by requirement).
- **B: notul-20k** — `notul_bg0c0` (fused kernels), wandb.name notul-20k.
Readouts: val CE trajectory (val/loss), wall-clock per arm from the queue-log
epochs, `core_depth_sweep.py` (A) / `token_depth_sweep.py` (B) at 1..8,
48 rows, gen samples with rep4/distinct-3. Wall-clock-matched comparison:
A's final CE vs B's CE at the step B reaches A's total wall time.
Stability risk accepted: the no-cap recipe is untested past 4500 steps;
div-guard is the instrument, a detonation is itself a result (contraction
prerequisite evidence).

## Predictions (frozen)

- **P1 (token axis).** B's final val CE < A's at matched 20k steps: 75%
  (B led by 0.096 at 4500; some chance TUL amortizes).
- **P2 (wall clock).** A's total wall time ≤ 0.85 × B's: 80%.
- **P3 (stability).** BOTH arms reach 20k without div-guard abort: 60%
  (two exposures of an uncapped core to a 4.4× longer horizon).
- **P4 (the value prop).** At matched WALL CLOCK, A's CE < B's: 45%.
- **Binding.** P4 TRUE ⇒ TUL ships as a supported first-class mode in the
  master merge and the cloud recipe gets a TUL variant. P4 FALSE ∧ P1 TRUE ⇒
  TUL stays merged but default-off (status quo), revisit only with R3-style
  span memory. Either arm detonates ⇒ contraction redesign becomes a merge
  BLOCKER for the no-cap recipe and the surviving comparisons are reported
  as-is.

## Not verified before run

No-cap stability beyond 4500 steps (either geometry); TUL never trained with
retention absent (smoke gates construction); disk: ~10 ckpts/arm × ~3.3 GB
transient before prune.
