# Experiment: the zero-deviation forcing bias in the TUL core loop (SCSE frame)

Status: failure — H19 refuted. The shared component of the core loop's displacement DECAYS across iterations by ~100x instead of accumulating, on the failing arm and the healthy arm alike, and the repeated additive injections de-correlate the states rather than driving them together.

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

## Method amendments

**Amendment 1 (2026-08-24, before any verdict was read).** The first A1 pass returned
`C -> 1` across the loop. `C` alone cannot say which of two opposite pictures produced that:
uncorrelated displacements of similar size, or ONE position holding nearly all the energy —
a sink drives `C` to 1 just as hard as isotropy does. The participation ratio over positions
and the top position's energy share were added to separate them. Predictions untouched.

**Amendment 3 (2026-08-24, after the first writeup over-claimed from it).** The injection
ablation replaced the WHOLE `DiagonalInjection` with a pass-through, which drops the fresh
per-slot injection `dt * e_ctx` AND the decay `A * h_ctx` together — so the effect could not be
attributed to either, and the first writeup credited it to the injection anyway. The ablation is
now split: `dt = 0` with the decay intact, and `A = 1` with the injection intact. Both were
re-run on both paths at all 11 rungs. The conclusion holds and is now attributed; see below.

**Amendment 2 (2026-08-24, forced by the trajectory gate).** The first run failed the gate at
24 % relative error. Cause: `model.train()` runs `nn.Dropout(0.1)` inside every core block, so
a replayed step draws a different mask than the captured one and the replayed map is not a
function of `(h, e, inj)`. Every dropout rate is now zeroed for the capture and for every
replay, in train mode, so the Poisson depth draw is unchanged. Dropout is zero-mean and
independent across positions, so removing it can only ADD isotropic energy back into the
comparison — i.e. it can raise the measured `C`, never lower it, which is the conservative
direction for a hypothesis that needs `C` to be large. After the fix the gate reads exactly
`0.0e+00` at all 11 rungs on both paths: the probe reproduces the captured trajectory bit for
bit.

## Results

11 rungs, `checkpoints/morph/onset-capture` (ROLL 1625..1850 + TAKEOVER 1866), one fixed
validation batch, one seeded depth draw, `tul_a1` at batch 6 with `model.use_kernels=false`.
Artifacts: `../results/2026-08-24-tul-zero-deviation-forcing-bias/drift_a1.json` and
`drift_a0.json`. Probe: `lab/divergence/drift_probe.py` (`--self-test` →
`DRIFT_PROBE_SELF_TEST_PASS`, sabotage-checked five ways).

`C_first` is at loop iteration 0, `C_last` at iteration 7. `P` is the active position count:
342 → 96 on the slot path, 6912 → 1152 on the token path, because the Poisson depth draw
freezes positions as the loop runs. `C` is normalised by `P` at that iteration, so `C = 1` is
the isotropic reading at every iteration and on both paths.

| | A1 slots, 1625 | A1 slots, 1866 | A0 tokens, 1625 | A0 tokens, 1866 |
|---|---|---|---|---|
| `C_first / P` | 0.495 | 0.439 | 0.173 | 0.175 |
| `C_last` | 2.07 | 1.13 | 12.50 | 3.10 |
| `C_last / C_first` | 0.0122 | 0.0076 | 0.0105 | 0.0026 |
| `rel_first` | 1.788 | 2.169 | 1.444 | 1.740 |
| `rel_last` | 0.702 | **1.081** | 0.682 | **1.077** |
| `C_last`, `inj_terms` zeroed | 2.06 | 1.12 | 12.52 | 3.12 |
| `C_last`, fresh injection off (`dt = 0`) | 8.50 | 1.41 | 42.83 | 3.85 |
| `C_last`, decay off (`A = 1`) | 6.24 | 1.31 | 46.45 | 4.98 |
| `C_last`, whole DiagonalInjection off | 7.50 | 1.31 | 41.52 | 4.17 |
| top position's share of displacement | 0.024 | 0.035 | 0.004 | 0.004 |

### Verdict on each prediction

* **P1 — refuted, by 400x and in the opposite direction.** `C_last / C_first` is 0.0076 at
  the TAKEOVER rung where the prediction needed ≥ 3. The shared component of the
  displacement does not accumulate across the loop; it is largest at the first iteration and
  is gone by the last one. Every rung on both paths agrees.
* **P2 — refuted.** `C_last` FALLS across the ladder, 2.07 → 1.13, where the prediction
  needed a ≥ 1.5x rise.
* **P3 — held, and the control makes it uninformative about the arm.** `rel_last` never
  collapses; it rises from 0.702 to 1.081 as the run takes over. But A0 does the same thing,
  0.682 → 1.077. See "What this did establish".
* **P4 — refuted.** Zeroing `inj_terms` moves `C_last` from 1.13 to 1.12. The repeated
  additive injection contributes nothing measurable to the shared component of the
  displacement.
* **P5 — the control matches the failing arm on every trend.** Under the rule written before
  the run, no verdict about A1's FAILURE may be read from P1–P4. The refutation of H19 does
  not depend on that: P1, P2 and P4 fail on A1's own numbers, so the mechanism is absent from
  the failing arm whatever the healthy one does.

### The sink confound, closed

`C` near 1 would also be the reading if a single position held all the displacement. It does
not. The top position's share of the displacement energy is 0.024–0.28 on the slot path and
0.004–0.007 on the token path, and the effective number of positions carrying the
displacement tracks the active count (`eff/P` = 0.11–0.79). There is no forward sink in the
core loop's displacement.

## What this did establish

Three facts, none of them H19.

1. **MORPH's injections DE-correlate the states.** Turning the DiagonalInjection off RAISES
   the shared concentration at every rung on both paths — 8.40 → 15.44 at A1/1700, 12.50 →
   41.52 at A0/1625, 1.13 → 1.31 at A1/TAKEOVER. The re-injection is not a drift source; it is
   what keeps the positions apart. SCSE's cure deletes exactly this term and replaces it with
   a once-only anchor, so porting SCSE to MORPH unmodified would remove the only measured
   de-correlator in the loop. That is an argument against the port, and it is the opposite of
   what the structural reading suggested.

2. **The loop stops settling as the run approaches takeover, on BOTH arms.** `rel_last` — the
   size of the last iteration's displacement relative to the state — rises from ~0.68 to ~1.08
   at the onset on the slot path AND the token path, to three decimal places of agreement
   (1.081 vs 1.077). At takeover every core iteration moves the state by more than its own
   norm: the realized map is not a contraction at any iteration. This is the forward-trajectory
   form of the `rho(J_core) >= 1` reading in `CLAUDE.md`, measured on the state sequence
   instead of the operator. Because arm A0 is healthy and shows the identical number, it is a
   background condition of the recipe at this step count, NOT a sufficient cause of A1's
   divergence — and any candidate cure that only restores contractivity therefore has to
   explain why A0 never needed it.

3. **A1's first core iteration is ~3x more shared than A0's, and it is stable.**
   `C_first / P` is 0.44–0.58 on the slot path against 0.17–0.19 on the token path: about half
   of the first step's displacement energy at every slot lies in one direction common to all
   342 slots and all 6 rows. It does not move with the onset (0.495 at 1625, 0.439 at
   TAKEOVER), so it is a property of the slot construction, not of the failure. It is,
   however, the strongest arm-specific asymmetry the campaign has measured in the forward
   pass, and it lands on the same iteration as the untested
   `../planned/2026-08-23-tul-iteration0-mediation.md`.

## Updated hypothesis

The zero-deviation forcing bias is a correct description of MORPH's FORM and a wrong
explanation of its FAILURE. The looped core is not accumulating a source-driven drift; it is
losing its shared structure fast and moving by order-1 amounts forever. Attention should go to
what is arm-specific — the first iteration's shared displacement is the one candidate this run
produced — and away from any cure whose mechanism is equally present in arm A0.
