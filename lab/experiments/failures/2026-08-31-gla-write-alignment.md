# Planned: GLA write alignment — same-step archive vs next-latent predictor

Status: failure
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

## Results

Div-guard abort at step 2040 (`DIVERGED_step_2040.pt`). Loss thrashed 7.4–8.6
(above unigram) from ~step 500, never trained. Forensics: branch gates AT INIT
(sigmoid 0.0028; GLA gate_bias exactly 2.000) — the branch stayed inert at the
output. Grad probe: preclip/core median 2.57e11, first >1e6 excursion at step
303, 78% of steps exploded; BC0 and BG0C0 (same recipe, no shift) never once
exceed 1e6 in 4500 steps (median 0.57). Localization at steps 300–400: coda
O(1), core.0–5 UNIFORMLY 1e5–1e8, prelude and embed likewise — the signature
of backward amplification THROUGH the loop (ρ_backward > 1, iterative-map
note), originating in-core and flowing down, not a branch-parameter blowup.

## Verdict

**P-A1 FALSE** (the 10% branch; a 90% prediction badly missed — recorded).
P-A2/P-A3 unmeasurable (no completed checkpoint). Failure class: the naive
alignment retrofit (shifted write on raw gated GLA, no normalized update)
detonates the UNCAPPED looped core via backward amplification, with the
branch output-inert. What this method cannot distinguish: (a) alignment is
inherently destabilizing in-loop, (b) alignment requires Falcon's normalized
NLMS step + renormalization (the paper never runs raw shifted writes), (c)
alignment × uncapped-core interaction that a σ-cap or decay-parameterized
contraction would absorb. Discriminators, if ever needed: BWS+cap=1.5 (45
min), or Falcon-2 proper. Campaign-level: reinforces that the uncapped core
has THIN stability margin — the shipping config needs TitanMAC-style
decay-parameterized contraction rather than "no control at all".

## Updated hypothesis

GLA's harm in MORPH is not (shown to be) the same-step objective; the
write-objective axis is untested in a stable regime. The no-GLA verdict from
BGpc stands on its own evidence.

## Follow-up diagnostics — 2026-08-31 20:55 (frozen before run, Wolfe's go)

- **D1** `notul_bws` + `model.use_kernels=false`, 600 steps (onset was 303):
  discriminates fused-kernel numerics vs dynamics. **P-D1: 70%** eager ALSO
  explodes (preclip/core > 1e6 by step 600) — dynamics, not a kernel bug.
- **D2** `notul_bws` + `training.spectral_project_cap=1.5`, 1000 steps:
  tests the uncapped-margin story. **P-D2: 65%** trains clean (no >1e6
  excursion, val descending).
- Binding: D1-stable ⇒ fused_gla backward bug with zero-sentinel key row —
  fix the kernel path. D1-explodes ∧ D2-clean ⇒ margin story confirmed;
  write-alignment retest waits for the contraction redesign. Both explode ⇒
  the raw shifted write is inherently unstable in-loop ⇒ Falcon NLMS
  normalizer pre-transform is the next arm.

### Diagnostic results — 2026-08-31 22:45

- **P-D1 TRUE** (70%): eager explodes too (first >1e6 at step 405, max
  2.0e9). The fused kernel is exonerated; the instability is dynamics.
- **P-D2 FALSE**: capped run never explodes (max preclip 1.2e4) but STALLS at
  unigram (val 7.44/7.61/7.37 at 700–900) — the BG0 stall signature.

The 2×3 matrix this completes (rows: cap; cols: GLA variant):

|            | same-step GLA | no GLA        | shifted GLA |
|------------|---------------|---------------|-------------|
| cap 1.5    | trains (4.34) | stalls 7.4    | stalls 7.4  |
| cap off    | trains (4.40) | trains (4.21) | explodes    |

Reading: same-step GLA's ONE real contribution is the early escape from the
unigram basin under the cap — the archive objective is an easy bootstrap
signal. The shifted (predictive) write provides no such bootstrap (capped ⇒
stall, same as no GLA) AND is actively destabilizing uncapped (worse than no
GLA, which trains fine). The naive alignment retrofit is dead on both rows.
The Falcon NLMS-normalized version remains the only live form of the idea,
and only matters if a memory branch ever returns; with the no-GLA campaign
verdict, priority is low. Final: GLA = training-dynamics crutch whose job
disappears with the cap; not a capability.
