# GATES — cure the MORPH TUL core takeover

Task: find the mechanism end to end, build a cure, test it, explain it.
Working tree: /home/wolfe/morph-perf (branch perf/throughput-lever-stack).
Control: `onset-capture` (tul_a1, deterministic, batch 6, seed 0, 2100 steps) takes over at 1866.

## G1 — measurement: locate the amplifier
- [x] G1.1 A realized-Jacobian probe exists in the repo and is unit-tested against an
      exact reference on a map with a known spectral norm.
      CHECK: `.venv-py -m pytest tests/test_core_jacobian.py -q`
      EXPECT: `passed`, 0 failed
      EVIDENCE: `10 passed in 3.60s`. Sabotage-checked twice: returning a constant sigma
      fails 5 of 10; dropping the pad mask from the capture fails
      test_capture_collects_one_point_per_loop_iteration_and_excludes_pads.
      Module gate: `python -m morph.training.core_jacobian` -> CORE_JACOBIAN_GATE_PASS
      (4 cases with an independently known answer).
- [x] G1.2 sigma_max(J_core) measured on the ROLL_1625..1850 checkpoint ladder, healthy
      versus taken over, with the per-block decomposition.
      EVIDENCE: `cure/ladder.json`, six rungs, 300 power iterations, convergence residual
      <= 1e-7 everywhere. Isotropic per-block gain 1.0292 -> 1.0552; sigma_max per block
      2.35 -> 3.23; whole-step typical gain 1.19 -> 3.93. FIRST attempt was invalid — the
      script loaded no MLP weights (QAT parametrization + _orig_mod key mismatch under
      strict=False); caught because the sweep disagreed with the run's own log. Rerun with
      apply_quantization + the trainer's load_checkpoint.
- [x] G1.3 The magnitude-versus-alignment question is ANSWERED with a number.
      EVIDENCE: ALIGNMENT. The isotropic per-block gain moves +2.5 % across the onset while
      the alignment factor (whole-step gain / product of the six blocks' gains) moves 0.986
      -> 2.850, x2.9, tracking the core share step for step (1.136 at share 0.021, 1.469 at
      0.372, 1.919 at 0.890, 2.850 at 0.961). Plus the concentration measurement: the
      cotangent falls from ~13 effective slot positions to 2.5, while the SAME weights on
      the token path stay at 26-59.

## G2 — the cure

NOTE 2026-08-24 12:45: the cure changed on measurement. The soft spectral penalty
(cap 1.5, lambda 10) holds in the deterministic microcosm and LOSES at the real
configuration — it let sigma_max reach 4.26 by step 1800 and its arm took over 475 steps
EARLIER than its control, with validation CE 2.74 nats above its minimum against the
control's 1.19. The cure under test is now the HARD projection
`W <- W * min(1, cap/sigma)` applied after the optimizer step. Gates below apply to it.
- [x] G2.1 An intervention is implemented in the repo (not a scratch script), off by
      default, and inert when off.
      EVIDENCE: `CoreSpectralProjection`, `training.spectral_project_cap: 0.0` constructs
      nothing. `tests/test_spectral_penalty.py::test_projection_leaves_an_under_cap_model_
      untouched_bit_for_bit` asserts torch.equal on every parameter after a step() under a
      cap it cannot reach. 19 tests green.
- [x] G2.2 The intervention holds the block backward gain below 1 through the step at
      which the control took over. MICROCOSM ONLY.
      EVIDENCE: deterministic pair, seed 0, batch 6, probe cadence 1: control block gain
      1.303 (r2 0.898), criterion fires 1760, aborts 1866. Cure 0.968 (r2 0.161), never
      fires, runs to 2100. At batch 12 the same setting does NOT hold: sigma_max 4.26 by
      step 1800 against a cap of 1.5.
- [x] G2.3 FALSIFIED at the real configuration. Judged by the pre-registered rule.
      EVIDENCE: five arms at ademamix_alpha_cap 3.5 against `a35-ctrl`, all TOOK OVER —
      soft cap 1.5, soft cap 3.0, hard cap 1.5 on the MLP, hard cap 1.5 on MLP+attention,
      and (see G3) the position-space arm. End core share 0.91 to 0.998 on every one.
- [x] G2.4 The HARD interventions cost no CE; the SOFT ones cost a lot.
      EVIDENCE: validation CE minimum — control 4.7881, hard cap 1.5 4.8084, hard cap 1.5
      +attn 4.7418 (BETTER than the control), soft cap 1.5 5.4525, soft cap 3.0 5.1054.
      Throughput cost of the projection: 6 % (1.95 against 2.08 steps/s).
- [x] G2.5 The panel is calibrated: a control at the same settings takes over.
      EVIDENCE: `a35-ctrl` (alpha_cap 3.5, no penalty) val CE 4.7881 @1000 -> 5.4116 @2000,
      sigma_max 8.16 by step 3100. Plus four historical arms at the same setting, val CE
      rise +1.28 to +1.59. The RCA's from-checkpoint placebo (token dropout 0.15 -> 0.145)
      also took over.

## G3 — durability
- [x] G3.1 Superseded by the method amendment and then FALSIFIED.
      EVIDENCE: the seed-1 control at alpha_cap 1.0 did not fail (validation CE fell
      monotonically to 3.7732), so at that setting the failure is a coin flip and a single
      arm cannot carry the comparison. The pair moved to alpha_cap 3.5, which fails 4/4,
      and every cure arm there took over.
- [x] G3.2 The control's horizon is established and every arm was run past it.
      EVIDENCE: `a35-ctrl` runs 7000 steps, validation CE 4.7881 @1000 -> 5.9742 @6500,
      +1.186. Every cure arm was aborted by the divergence guard before 2250 with a larger
      rise, so none of them needed the longer horizon to be judged.

- [x] G3.3 The state-side intervention is run and scored. It HOLDS ONE SEED OF TWO.
      EVIDENCE: `b10-slotembed` (`tul.per_slot_embed`, pre-registered as P12 before it
      reported) against `b10-ctrl`, same optimizer, batch and schedule: end core share
      0.0223 against 0.9999, per-block backward gain 1.052 at r2 0.48 (the flat healthy
      signature) against 2.445 at r2 0.97, validation CE monotone for the whole run and
      finishing AT its own minimum, 4.0747, which is 0.78 nats below the control's
      best-ever 4.8528. `sigma_max` fell as a consequence, 2.88 at step 1500 against 4.86.
      The second seed (P13) then FALSIFIED it: `s0-slotembed` took over at step 2225 with
      end share 0.9998 and gain 2.501. What survives both seeds: time-to-failure 1150 ->
      2225, val CE damage +0.533 -> +0.119, and best CE 0.78 / 0.46 nats better than the
      control's best. Left off by default because of the second seed. P15 (stacking it with
      bptt_depth 2) ALSO took over, worse than either lever alone. P14 (no jitter) was
      launched and stopped to free the GPU for P15; not reported.
      `tul.max_slots` 64 -> 128 (P10) is NOT run: a measured OOM at batch 12 and 10, and a
      typical row uses 57 of 64 slots so the budget rarely binds.

## G4 — ship
- [x] G4.1 Tests added and green; whole suite green.
      CHECK: `.venv-py -m pytest tests/ -q -p no:randomly`
      EXPECT: `passed`, 0 failed
      EVIDENCE: `308 passed, 1 xfailed in 11.65s` with CUDA available, 2026-08-24 16:44
      (the 5 CPU-only skips run under CUDA). 33 of those are new:
      tests/test_core_jacobian.py (10), tests/test_spectral_penalty.py (21) and four
      per-slot-embedding tests in tests/test_tul_forward.py, 35 together.
      Module gates: `python -m morph.training.core_jacobian` -> CORE_JACOBIAN_GATE_PASS,
      `python -m morph.training.spectral_penalty` -> SPECTRAL_PENALTY_GATE_PASS.
- [x] G4.2 Pre-registered experiment file written and committed BEFORE the deciding runs.
      EVIDENCE: lab/experiments/planned/2026-08-24-tul-takeover-cure.md, commit c51a142,
      09:57. The seed-1 arms started after it. The file states explicitly that the
      microcosm arm ran BEFORE it and is reported as an observation, not a prediction.
- [x] G4.3 Results filed, agent note written, verify ok.
      EVIDENCE: lab/experiments/failures/2026-08-24-tul-takeover-cure.md (filed under
      FAILURES, because P4 and P9 are falsified),
      .agents/notes/implemented/architecture/2026-08-24-core-takeover-is-positional.md,
      docs/cookbook/measuring-the-core-map.md, both indexed in docs/MANIFEST.md.
      `python scripts/verify_template.py` -> ok.
- [x] G4.4 Committed on the working branch.
      EVIDENCE: `perf/throughput-lever-stack`, 41 commits this session, working tree clean.
      Branch is LOCAL and unpushed, as it was at the start of the session.
