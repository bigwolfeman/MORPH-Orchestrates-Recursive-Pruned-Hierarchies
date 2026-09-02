# Planned: A2s — the restricted paid loop, and the frozen 20k selection

Status: planned
Date: 2026-09-02 (frozen before launch, ~03:00; Wolfe's directive: "we need
to do a2s then which ever had lower loss (a2 vs a2s) needs a 20k run")

## Question

Does restoring the TG restriction (same-span-or-slot window, slot-only
compressed branch) inside the paid core keep A2's loop earning — and which
of A2 / A2s carries the lower loss into a 20k confirmation?

## Hypothesis

The restriction is TUL's information discipline (slots as the one channel
to the past). On the paid axis it may cost short-horizon CE (restricted
arms trailed at 20k in the h2h) while making the loop's earning cleaner or
larger (the core must route through slots instead of re-reading tokens).
Counter-hypothesis: the restriction strangles the paid loop's token-axis
earning, which lived precisely in unrestricted token attention.

## Method

Config `tul_a2s` = `tul_a2` + exactly ONE delta: `tul.tg_restrict=true`
(forcing `model.use_kernels=false`; the runner adds
`+model.tg_scoped_kernels=true` like every TG arm). Mechanism: `_core_region`
now threads `tg_allow`/`tg_slot_mask` through the active-set sort (commit
this change). Verification before launch: 5 new CPU tests
(tests/test_a2s_restricted_core.py — construction, forward/backward, masks
reach the core, position-level causality, reorder-equivariance at
nonuniform depth); full suite 764 passed / 8 skipped / 1 xfailed; GPU
causality bitwise at 2 perturbed positions under scoped kernels on the
composed tul_a2s at real shape; forward deterministic.

Run: 5000 steps, panel flags, ckpt 2500+5000 both kept,
`a2_depth_sweep.py` on both (same lever — A2s uses the per-sample core).
Smoke gates: TUL ON + TG-RESTRICT ON + kernels=EAGER+TGSCOPED + scoped
print + layer_passes_per_token=4x + no acausal + 0 retention keys +
loss<14. Retry-once on detonation (the R1/A2 convention), attempt-1
forensics archived.

## Predictions (frozen)

- **P-S1 (survival).** A2s completes 5000 steps on the first or second
  draw with no unresolved div-guard abort: 85% (per-draw ~60%, two draws).
- **P-S2 (earning survives restriction).** A2s a2-sweep K1−K6 at 5000
  >= 0.05: 55%.
- **P-S3 (restriction is free).** A2s final val/ce_tokens at 5000 <= A2's
  4.2315: 35%.

## Binding (the 20k selection — frozen NOW, before A2s runs)

Winner = the arm with the LOWER `Final val_loss` (val/ce_tokens, full final
eval at step 5000) between tul-a2 (4.2315, already measured) and tul-a2s.
The winner runs 20k: winner config + panel flags, `training.steps=20000
ckpt_every=2000`, prune to step_20000, `a2_depth_sweep.py` 1..8, NO gen
samples (the eager generator's interaction with tokens_through_core is
unverified — run later, not silently). No auto-retry at 20k — a detonation
there is a finding to wake up for, not a retry to burn hours on.
References for the 20k readout: tul-20k last-5 3.8461 / notul-20k last-5
3.4894 / notul-20k token-axis K1−K6 0.2072.

## Not verified before run

A2s training dynamics beyond the smoke (first live gradient through the
masked paid core); a2_depth_sweep on an A2s checkpoint (same code path as
A2's, but the masked-core eval is new); the sps of the eager+scoped paid
core (estimated ~0.5-0.7 from R0's 2.4 at 4.3x fewer passes — the 20k
duration estimate hangs on it).
