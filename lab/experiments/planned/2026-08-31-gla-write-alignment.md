# Planned: GLA write alignment — same-step archive vs next-latent predictor

Status: planned
Date: 2026-08-31 (frozen before the run; trigger: Wolfe — FWA paper
arXiv:2608.27763, "we will likely try it either way").

## Question

MORPH's GLA writes the same-step pair (k_t, v_t): the state is trained as an
archive of the current token. FWA (Falcon) shows the read-after-write-correct
pair is (k_{t-1}, v_t): the state as next-latent predictor. Is the same-step
archive objective part of why GLA (a) hurts absolute CE in the uncapped
regime and (b) suppresses loop depth-earning (BC0 vs BG0C0)?

## Hypothesis

The archive objective is trivially learnable, wins the early gradient race,
and stores exploitable copies (the 30k transcription cheat lived on exactly
such a state). Aligning the write makes the branch's internal problem
predictive, reducing both the bypass pull and the wasted capacity.

## Method

Arm `notul_bws` = BC0's exact recipe (notul_l2, cap 0, GLA all three sites,
carry none, TUL never, kernels on, panel flags, 4500 steps, seed 1) with ONE
diff: `model.retention_write_shift=true` (k stream shifted right with zero
sentinel before the recurrence; reset positions zeroed; gate and reads
untouched — `morph/model/gla.py`, tests `tests/test_gla_write_shift.py`,
5 passed incl. hand-shifted-oracle parity and chunked==recurrent with shift).
Readout: `token_depth_sweep.py` 1..8, 48 rows, vs BC0 (K1−K6 0.142, K6 4.401)
and BG0C0 (0.220, 4.206). Runner queued behind BGpc (round 4).

## Predictions (frozen)

- **P-A1.** Trains clean, no stall/detonation (val < 6.0 by 1500): 90%.
- **P-A2.** Absolute K6 ≤ 4.35 (≥ 0.05 nats better than BC0 — the archive
  objective was wasting capacity): 50%.
- **P-A3.** Depth-earning K1−K6 ≥ 0.19 (alignment alone recovers
  BG0C0-level earning — the bypass pull was objective-driven): 30%.
- **Binding.** A3 TRUE ⇒ alignment is the GLA fix; keep the branch, Falcon-2
  normalized update is the next arm. A3 FALSE ∧ A2 TRUE ⇒ alignment helps but
  the bypass persists ⇒ compose with BGpc's placement verdict (shifted GLA at
  detached sites). A2 ∧ A3 FALSE ⇒ the write objective is not the axis; the
  branch decision falls entirely to BGpc/Raven reasoning.

## Not verified before run

Fused-kernel path with the shifted k stream on real shapes (contract tests
are eager CPU; the shift is a pre-transform so the kernel sees ordinary
inputs, but the live smoke is the check). Interaction of shift with
retention_gate_bias timing (gate stays same-step by design; not ablated).
