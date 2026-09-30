# Planned: can ONE sampled code per slot (REINFORCE) replace LX's four code rollouts?

Status: failure

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

## Results

Runs 2026-09-29 23:49 to 2026-09-30 01:59, seed 1, 5000 steps, both at 1679914. No CPU test
suite ran on this machine during either run (one 1.5 s, one-core pin test at 00:00, logged).
tok/s is the mean of the trainer's 24 logged points at step >= 200.

| arm | tok/s | K1-K6 | gap to plain 5k | val loss |
| --- | --- | --- | --- | --- |
| control (K=1) | 11574 (10795-12228) | +0.0047 [+0.0043, +0.0051] | +0.2830 [+0.2678, +0.3000] | 4.4389 |
| policy4 | 10919 (10511-11274) | +0.0038 [+0.0034, +0.0042] | +0.2777 [+0.2626, +0.2947] | 4.4397 |
| plain (nm ctrl s1r) | 9988 | - | 0 | - |
| parent e4probe_fp01 (K=4) | ~5600 (siblings) | +0.0126 | +0.265 | - |

policy4 minus control, paired on the same 501106 tokens: -0.0050 [-0.0075, -0.0024].
policy4 val at 5000: `code_policy_vs_random` -0.0003, entropy 0.999 nats (max ln 4 = 1.386),
argmax shares 0.06 / 0.10 / 0.71 / 0.14, advantage std 1.09 nats against mean 0.03.
Tripwire: control HEALTHY (max 83); policy4 one recovered excursion 1.94e4 at step 2082.

| prediction | reading | held |
| --- | --- | --- |
| P-1 no detonation | one recovered one-step excursion | yes |
| P-2 control worse than parent by 0.010-0.045 | +0.018 | yes |
| P-3 control >= 9000 tok/s | 11574 | yes |
| P-4 policy4 within 10 % of control | 0.94x | yes |
| P-5 vs_random > +0.002 | -0.0003 | NO |
| P-6 entropy > 0.5 | 0.999 | yes |
| P-7 not worse than control by > 0.0137 | -0.0050 (better, under the bar) | yes |
| P-8 K1-K6 within 0.007 | 0.0009 apart | yes |

## Verdict

Failure for arm B: P-5 failed, so the per-slot choice carries nothing (the argmax code reads
the same CE as a random one). The -0.0050 paired edge over the control is under the 0.0137
seed bar and cannot come from the choice. REINFORCE's advantage noise (std 1.09 nats per span)
is 36x its mean, as the hypothesis said.

Success for the control as a cost result (P-3): the one-rollout strict slot loop runs at
1.16x plain's tok/s, the first slot-loop arm faster than plain in wall clock. It pays for it
with LX's ensemble gain: its gap to plain is 0.018 worse than the K=4 parent.

## Updated hypothesis

The code axis has nothing per-slot to choose: codes that measured as span-blind stay
span-blind when a policy picks them. The cost of LX was the ensemble, and the ensemble is
worth ~0.018 nats at 5k. Drop the code axis; the one-rollout loop is the cost base
(11.6k tok/s) that further exploration must be priced against.
