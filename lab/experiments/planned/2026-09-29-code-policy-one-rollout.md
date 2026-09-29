# Planned: can ONE sampled code per slot (REINFORCE) replace LX's four code rollouts?

Status: planned

Date: 2026-09-29 18:31 CDT (frozen before either run trained).
Code: c5e9e36 (`tul.code_policy_k`; note
[`2026-09-29-code-policy-one-rollout.md`](../../../.agents/notes/proposed/architecture/2026-09-29-code-policy-one-rollout.md)).
Wolfe (2026-09-29): "We need to test both a and b"; "Exploring the latents shouldnt be more
expensive than the central loop."

## Question

LX (`code_enum_k: 4`) runs four code rollouts of the loop AND the coda per row and trains the
exact mixture: about 4x the coda, and its gain measured as an ensemble
(`failures/2026-09-26-lx-selection-ceiling.md`). Arm B keeps the codes, runs ONE rollout, and
lets a policy on each slot's own content pick one code per slot, trained by REINFORCE on the
span's own CE. Does the choice carry anything (the policy beats a random code), and what do
the one-rollout arms cost and lose against the K=4 parent?

## Hypothesis

H: dropping to one rollout loses LX's ensemble gain (about 0.025 nats at 5k) and runs about
1.7x faster. A per-slot policy on the prelude's summary has little to choose on: codes that
measured as span-blind rarely matter per span, and REINFORCE's per-span noise (about 1 nat of
span difficulty) swamps a code effect near 0.01. So the policy arm reads about the same as the
K=1 control.

## References (seed 1, 5000 steps, 480 sweep rows)

| arm | K1-K6 | gap to plain 5k | tok/s |
| --- | --- | --- | --- |
| e4probe_fp01 (LX, K=4, the parent) | +0.0126 [+0.0119, +0.0134] | +0.265 | not in the logs; K=4 siblings pk4 5723, hard 5538 |
| plain panel (nm ctrl) | - | 0 | 9355-10089 |

K1-K6 seed spread in this family is about 0.007 (`lxfan-wta-earns-depth`); the paired CE bar
used on this tree is 0.0137.

## Predictions (mine, orchestrator, before either run)

- **P-1**: neither run detonates. 85 %.
- **P-2**: the K=1 control's gap to plain 5k is WORSE than the parent's +0.265 by 0.010 to
  0.045 (the ensemble gain it gives up). 55 %.
- **P-3**: trainer tok/s (mean of the logged points at step >= 200, no CPU test load during the
  run) of the control >= 9000. 65 %.
- **P-4**: policy4 tok/s within 10 % of the control's. 80 %.
- **P-5**: `val/code_policy_vs_random` at step 5000 > +0.002 nats/token (the choice matters).
  35 %.
- **P-6**: policy4 `val/code_policy_entropy` at 5000 > 0.5 (no collapse onto one code). 60 %.
- **P-7**: policy4's gap to plain is not worse than the control's by more than 0.0137. 75 %.
  Better than the control by more than 0.0137: 20 %.
- **P-8**: policy4 K1-K6 within 0.007 of the control's. 60 %.

Pass for arm B: P-4, P-5 and P-7 (the choice matters, costs little, hurts nothing). Pass for
the one-rollout control as a cost result: P-3.

## Method

- Configs `tul_slot_spandec_strict_e1probe_fp01` (= `e4probe_fp01` with `code_enum_k: 1`) and
  `..._e1probe_fp01_policy4` (+ `code_policy_k: 4`, defaults lambda 1.0, entropy 0.01, value
  1.0, eval argmax). Seed 1, 5000 steps, runner `runner_steps.sh` at c-SHA of this prereg.
- Readouts as for every slot arm: depth sweep (K1-K6, 480 rows), gap to plain 5k, worth;
  the stage-2 scorer and pair read against fp01. tok/s from the trainer log.
- No CPU test suites run on this machine while either arm trains (mapwin's tok/s dipped 20 %
  under CPU load, `failures/2026-09-29-wta-map-speed-quality.md`).
