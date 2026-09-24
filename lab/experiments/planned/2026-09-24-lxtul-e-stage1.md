# Planned: LXTUL-E Stage 1, a loop that carries an enumerated code, read in parallel

Status: planned

Date: 2026-09-24 (frozen before any Stage 1 GPU step beyond the builder's 30-step smoke).
Design and priors:
[`2026-09-23-provable-loop-contribution.md`](../../../.agents/notes/proposed/architecture/2026-09-23-provable-loop-contribution.md)
(its P-1..P-5 are carried here unchanged; P-1's threshold is now fixed by Stage 0). Gate:
[`2026-09-23-lxtul-e-stage0.md`](../successes/2026-09-23-lxtul-e-stage0.md), B0 = 0.1031.
Build: 017d472. Wolfe's go: 2026-09-23 ("lets try it").

## Question

If the slot loop carries one of K = 4 enumerated codes, re-added at a fixed size every pass,
and a committed parallel reader of the next span is trained on the exact mixture over the
four rollouts, does width survive training, and does the loop's depth get a job (integrating
the code into a separated per-code state) that the parallel read pays for?

## Hypothesis

From the Lean results: width pays only a committed reader (T3), a re-supplied code is
amplified by the passes as Σ λᵗ (T5, linear case), and a teacher-forced coda hedges (T3), so
its K-curve and its width gain stay flat.

## Arms

On the strict ruler recipe (`tul_slot_spandec_strict`: seq 1024, batch 6, seed 1,
norm_match, ramp 1000, fixed point 1.0), 5000 steps from scratch, at 017d472, one trainer at
a time on the 5090.

| arm | config | one factor |
|---|---|---|
| `lxtul-e1` | `tul_slot_spandec_strict_e1.yaml` | the ruler with the parallel head in place of the span decoder; K = 1, no code |
| `lxtul-e4` | `tul_slot_spandec_strict_e4.yaml` | e1 + K = 4 enumerated codes re-added every pass (r 0.1), 4 rollouts, exact mixture for the head and the coda |

Partner on disk: the ruler `slot-spandec-strict` @5000.

## Instrument

`lab/divergence/lxtul_e_stage1_score.py` on the 480 validation rows (terms in its
docstring): par CE and coda CE at forced depths 1..6; width gains (one code minus the
mixture); exit separation between codes and its depth-6 / depth-1 ratio; paired e4 vs e1 and
e4 vs ruler. Plus the runner's `core_depth_sweep` and `worth_profile` on both arms.

## Predictions (frozen; the note's, with P-1's threshold filled in)

- **P-1 (width survives).** e4's parallel mixture CE beats e1's parallel CE by at least
  max(0.01, 0.5 · B0) = **0.0515** nats per head token, paired CI clear of zero: **50 %.**
- **P-2 (the coda hedges).** e4's coda width gain (one code alone minus the 4-rollout read)
  is at most 0.005: **85 %.**
- **P-3 (width creates a depth job).** e4's parallel-read K1−K6 >= 0.02 with the CI clear
  of zero: **35 %.** e1's parallel-read K1−K6 <= 0.005: **60 %.**
- **P-4 (the teacher-forced bypass).** Coda token K1−K6 <= 0.005 on both arms: **80 %.**
- **P-5 (no price).** e4's coda CE under the 4-rollout read, paired against the ruler at
  depth 6, within +0.02: **55 %.**
- **Diagnostic, not scored:** e4's exit-separation ratio, depth 6 over depth 1 (linear
  theory 4.6 at gain 0.89).

## Verdict rules

From the note. What refutes the account: P-1 holds, P-3's first clause fails, and the
separation ratio is below 1.3 (the reader reads the one-pass separation; integration is not
a job). Or P-2 fails by more than 0.02 with P-1 failing (the teacher-forced coda is not the
hedging reader T3 treats it as). P-3's first clause holding is the loop-contribution
positive the design exists for; it is read on a TRAINING-ONLY head, and says nothing about
the deployed token path unless P-2 or P-4 also moves. No clause passes on a cosine.

## Method

e1 then e4, then the scorer and the sweeps. Artifacts: JSON and run logs in
`../results/2026-09-24-lxtul-e-stage1/`.
