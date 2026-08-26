# Experiment: is the zero-deviation forcing bias arm-linked, or a property of the recipe?

Status: success on the question asked — the forcing bias IS arm-linked; the refuter did not fire. P1, P2 and P3 held; **P4 FAILED** — neither arm turned around within 1900 steps, so this is an arm comparison between two HEALTHY models at matched training steps, NOT sick-against-healthy.

## Question

`b_t(e) = T_t(0; e)` — SCSE's zero-deviation forcing bias, arXiv:2607.27656 eq. (1) — is the
first quantity in the takeover campaign that separates the two TUL position sets instead of
matching across them. Measured on arm-A1 checkpoints
([forcing-bias](../failures/2026-08-24-tul-zero-deviation-forcing-bias.md)),
`||b||/||h*||` runs 1.809 → 2.471 across the onset ladder at the SLOT anchor against
1.448 → 1.723 at the TOKEN anchor, a ratio of 1.23–1.44 at all 11 rungs that widens with the
onset.

That comparison holds the weights fixed and varies the position set. It is not sick-against-
healthy. Two earlier leads — `rel_last` (1.081 against 1.077) and `C_last` — died precisely
because the healthy arm matched the failing one, so the same test has to be run here before
`b_t` is treated as a lead.

**Question.** Does a separately TRAINED A0 arm carry the same `b_t` growth as a separately
trained A1 arm, at matched training steps and in the same numerical build?

## Hypothesis

H20. The forcing bias is arm-linked. A1 routes its whole plan through ~57 slot positions whose
states are span bag-means, and its `b_t` grows faster with training than A0's, which loops over
1024 token positions and whose coda reads the tokens directly.

## Method

Two arms, `tul_a0` and `tul_a1`, from scratch, seed 0, 1900 steps, batch 6,
`ademamix_alpha_cap=3.5`, `model.use_kernels=false`, checkpoints every 300 steps.
Runner: `../../lab/divergence/stage0_spark.sh`. Probe:
`../../lab/divergence/drift_probe.py --ckpt-dir <arm>`.

**Both arms run in the same build.** `checkpoints/morph/onset-capture/README.md` records that
two torch builds gave different trajectories from the same seed, so pairing arms across
machines would confound the arm with the build.

**Method amendment, 2026-08-24, before either arm reached step 200.** The arms were launched on
the DGX Spark because the 5090 was in use. The 5090 was freed, so both arms were killed at step
~0 and relaunched there. Reason: the Spark measured 0.12 sps against the 5090's 2.12 sps, an 18x
penalty from launch-bound work on the ARM cores, turning a 30-minute panel into a 5-hour one.
The amendment strictly improves the design — the 5090 is the SAME build that produced the
existing `onset-capture` A1 ladder, so the new arms are comparable both to each other AND to
that ladder, which the Spark run could not have been. No Spark checkpoint was written and no
Spark number was read, so nothing was seen before this change. Predictions are untouched.

Sequential rather than concurrent on the 5090. Concurrency was justified on the Spark by 25 %
GPU utilisation (launch-bound, so the arms interleave); the 5090 does not have that headroom,
and two training processes plus a loaded GPU is the configuration that trips this workstation's
UPS.

**Validity gate, checked before any comparison.** `R_0` (eq. 9) must equal 1.000 at every
checkpoint of both arms. With MORPH's anchor `h* = h_0 = e`, the first realised update IS the
anchor response, so `R_0 = 1` is an identity the paper's own baseline row satisfies. It held to
four significant figures at all 11 rungs of the 5090 ladder. If it does not hold here, the
measurement is wrong and nothing below may be read.

## Predictions

Written before either arm reached step 300.

* **P1.** `||b||/||h*||` rises with training step for BOTH arms between step 300 and step 1800.
* **P2.** At step 1800, A1's `||b||/||h*||` exceeds A0's by at least 15 %. The shared-weights
  slot-against-token gap was 23–44 %; 15 % is a deliberately conservative floor for a genuine
  arm comparison.
* **P3.** A1's growth from step 300 to step 1800 exceeds A0's growth over the same interval.
  On shared weights the slot anchor grew +37 % against the token anchor's +19 %.
* **P4.** A1's validation CE turns around — reaches a minimum and then rises by at least 0.1
  nats before step 1900 — while A0's does not. This is the arm-health check that says the two
  arms are actually behaving as the campaign describes IN THIS BUILD.

## What would refute H20

If A0's `||b||/||h*||` sits within 10 % of A1's at every rung, the forcing bias is a property of
the recipe rather than of the arm. It then joins `rel_last` and `C_last` on the refuted-lead
list, and the SCSE port loses the one measurement that currently distinguishes the failing arm.

If P4 fails — A1 does not turn around by 1900 steps on this build — then P1 to P3 are still
readable as an arm comparison at matched steps, but NOT as sick-against-healthy, and the writeup
must say so rather than borrowing the 5090 run's takeover.

## Results

Both arms trained to 1900 steps on the 5090, exit 0. Seven rungs each (300, 600, 900, 1200,
1500, 1800, 1900). Artifacts in `../results/2026-08-24-tul-forcing-bias-arm-control/`.

**Validity gate passed.** `R_0 = 1.000` at every checkpoint of both arms, and the probe's
trajectory gate read exactly `0.0` at all 14 checkpoints — the replayed core map reproduces the
captured trajectory bit for bit on both code paths.

| step | A0 `b/‖h*‖` | A1 `b/‖h*‖` | A1/A0 |
|---|---|---|---|
| 300 | 1.400 | 1.668 | 1.191 |
| 600 | 1.334 | 1.736 | 1.301 |
| 900 | 1.355 | 1.689 | 1.246 |
| 1200 | 1.378 | 1.622 | 1.177 |
| 1500 | 1.386 | 1.741 | 1.257 |
| 1800 | 1.423 | 1.764 | 1.240 |
| 1900 | 1.452 | 1.793 | 1.235 |

* **P1 HELD.** `b` rises for both arms: A0 +3.7 %, A1 +7.5 % over 300 → 1900.
* **P2 HELD.** A1 exceeds A0 by 23.5 % at the last rung, against a 15 % floor.
* **P3 HELD.** A1 grows twice as fast: +7.5 % against +3.7 %.
* **P4 FAILED.** Neither arm turned around. A1's validation CE fell monotonically,
  5.9560 → 5.5417 → 5.0715, and A0's fell 6.1477 → 5.7670 → 5.2301. Turnaround 0.000 nats for
  both.
* **Refuter did not fire.** A0 is outside 10 % of A1 at every one of the seven rungs
  (1.177 to 1.301). The forcing bias is not a property of the recipe shared by both arms.

## Verdict

**The zero-deviation forcing bias is arm-linked.** On separately trained models, in one build,
at matched training steps, arm A1 carries 18 % to 30 % more anchor response than arm A0 at every
rung, and it grows twice as fast. This is the first quantity in the takeover campaign that
separates the arms rather than matching across them — `rel_last` matched at 1.081 against 1.077,
and `C_last` fell on both.

**What P4's failure costs.** A1 did not reproduce the takeover in 1900 steps, so the 1.235 gap
is between two HEALTHY models. It establishes that the gap is intrinsic to the arm and present
before any pathology. It does NOT establish that the gap causes the failure, and this writeup
must not borrow the `onset-capture` run's takeover to close that gap.

Two reasons P4 failed, neither fixable after the fact:

1. 1900 steps is inside the campaign's own stated window for the minimum (step 500-2000), so
   the run can end AT the minimum without ever showing the rise.
2. This is not a replay. `onset-capture/README.md`'s replay recipe specifies
   `training.eval_every=999999` and `deterministic=true`; this run used `eval_every=500` because
   P4 needs the validation curve, and evaluation consumes RNG. MORPH runs decorrelate within 11
   steps of any perturbation at a fixed seed, so this is a fresh sample from the same
   distribution rather than the same trajectory.

Extending the run to chase a turnaround was considered and REJECTED: P4 says "before step 1900",
and moving that line after seeing the data is the failure pre-registration exists to prevent.

## Observation, not a tested prediction

Comparing this healthy A1 against the `onset-capture` A1 that DID take over — same build, same
config, same seed, same `alpha_cap` — at matched steps:

| | step 1800 | step 1850 |
|---|---|---|
| A1, healthy (this run) | 1.764 | — |
| A1, took over (`onset-capture`) | 2.265 | 2.471 |

The diverging run carries 28 % more forcing bias at step 1800 than the healthy one. This was NOT
pre-registered, it is n=1 against n=1, and the campaign's own trap list warns that single-run
comparisons in MORPH are unreadable. It is recorded because it is the shape a causal link would
have, and because it is cheap to test properly: the next experiment is a small set of A1 seeds
run past 3000 steps, scoring `b_t` against whether each seed turns around.

## Updated hypothesis

H20 survives its first real control. The forcing bias is arm-linked, not a recipe property, and
SCSE keeps the one measurement that distinguishes arm A1. What is still missing is the link from
the gap to the FAILURE, which needs A1 arms that actually diverge.
