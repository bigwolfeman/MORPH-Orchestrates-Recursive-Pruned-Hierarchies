# Experiment: the zero-deviation forcing bias in the TUL core loop (SCSE frame)

Status: planned

## Question

SCSE (arXiv:2607.27656, Fig. 1) separates two recurrences. The **baseline** re-injects the
source `e` at every step, `h_{t+1} = G_θ(h_t + e)`, which leaves a source-driven degree of
freedom `b_t(e) ≠ 0` at the anchor: the "do nothing" state is not a fixed point, so the
trajectory drifts every step and the drift compounds with depth. The **cure** uses `e` once
to set an anchor `h*(e)`, evolves the deviation `Δ_t = h_t − h*(e)` through a bias-free core
with `G_θ(0) = 0`, and masks the update so `Δ = 0` is an exact fixed point.

MORPH's core loop is the baseline form twice over. Read from
`morph/model/transformer.py:_apply_core_step`, one iteration is

```
h ← A ⊙ h_ctx + dt ⊙ e_ctx            # DiagonalInjection, 320 of 1024 channels, EVERY iteration
for i in 0..n_core-1:
    h ← h + inj_term_i                # loop-INVARIANT additive term, EVERY iteration
    h ← Block_i(h)
```

Both injections repeat at every iteration, and `inj_term_i` is a constant with respect to
`h`, so `G_θ(0) ≠ 0` by construction. The structural match is not in doubt. What is in doubt
is whether the resulting drift is what kills arm A1.

**Question.** Does the core loop carry a persistent, position-shared displacement that
grows with loop iteration and with training step, and does the repeated additive injection
supply it?

## Hypothesis

H19. The per-iteration displacement `d_t = f_θ(h_t) − h_t` contains a large component that
is common to all slot positions. Repeating it over `n_core × T ≈ 36` additions drives every
slot state along one shared direction, which is the measured forward rank collapse
(input effective rank ≈ 28 → post-loop 1.7–4.8). The repeated additive injection is the
source of that shared component.

This is a distinct claim from oversmoothing. Oversmoothing says the map is a contraction and
the states fall into a shared fixed point, so the displacement must SHRINK with `t`. The
forcing-bias claim says a persistent drive keeps the state moving, so the displacement must
NOT shrink. The two readings differ in the sign of `d‖d_t‖/dt`, which is the discriminator.

## Method

One model, one validation batch, one seeded Poisson depth draw, replayed at each of the 11
`ROLL_step_*` / `TAKEOVER_step_*` rungs in `checkpoints/morph/onset-capture` (they bracket
the onset at ~1866). `CoreJacobianProbe.capture()` already records `(h, e, inj, ret_state,
active, iter_idx)` per loop iteration, which is the exact training core map input; the probe
replays `_apply_core_step` off those points, so no model change is needed.

Measured on the ACTIVE positions only (a frozen or pad slot is not updated):

* `rel_t = rms_p‖d_t‖ / rms_p‖h_t‖` — relative displacement per iteration.
* `C_t = n_pos · ‖d̄‖² / mean_p‖d_p‖²` with `d̄ = mean_p d_t` — the **shared concentration**
  of the displacement. `C = 1` for displacements spread isotropically over positions;
  `C = n_pos` when every position moves in exactly the same direction. The `n_pos` factor
  makes the number comparable between the 57-slot path and the 1024-token path.
* `C_inj` — the same concentration computed on `inj_term_i` itself.
* Attribution: `d_t` recomputed at the same `h_t` with (a) `inj_terms` zeroed and (b) the
  DiagonalInjection replaced by the identity. The drop in `C_t` attributes the shared drive.

Probe correctness gate: the replayed `f_θ(h_t)` must reproduce the NEXT captured `h_{t+1}` on
the active positions to within bf16 round-off. A probe that cannot reproduce the trajectory
it claims to measure is not measuring the training path. The probe raises if it does not.

## Predictions

Written before any rung is read.

* **P1.** `C_t` rises across loop iterations at every rung: `C_last ≥ 3 × C_first`.
* **P2.** `C_last` rises across the ladder: the TAKEOVER rung is at least 1.5× the 1625 rung.
* **P3.** `rel_t` does NOT collapse: `rel_last ≥ 0.3 × rel_first`. If instead `rel_t` decays
  toward zero, the forcing-bias frame is wrong for MORPH and oversmoothing owns the result.
* **P4.** Zeroing `inj_terms` cuts `C_last` by at least 30 %.
* **P5 (control).** The same weights on the token path (`slot_layout=None`, arm A0, which is
  healthy) show a weaker rise: A0's `C_last / C_first` is below A1's. If A0 matches A1, the
  measure does not discriminate the failing arm from the healthy one and no verdict may be
  read from P1–P4.

## What would refute H19

P3 failing (displacement decays) refutes the frame outright. P5 failing voids the whole
panel. P1 or P2 failing leaves the structure present but not growing with the disease, which
would make SCSE a correct description of MORPH's form and a wrong explanation of its
failure.
