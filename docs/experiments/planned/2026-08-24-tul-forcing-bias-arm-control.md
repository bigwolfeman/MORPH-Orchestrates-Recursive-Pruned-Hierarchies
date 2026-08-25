# Experiment: is the zero-deviation forcing bias arm-linked, or a property of the recipe?

Status: planned

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

**Both arms run in the same build, on the DGX Spark** (torch 2.13.0+cu130, aarch64, GB10
sm_121). `checkpoints/morph/onset-capture/README.md` records that two torch builds gave
different trajectories from the same seed, so pairing a Spark A0 ladder against the 5090 A1
ladder would confound the arm with the build. The existing 5090 A1 ladder is therefore NOT the
control for this experiment; the Spark A1 arm is.

Concurrent rather than sequential: measured GPU utilisation is 25 % at 0.12 sps, so the
workload is launch-bound and the two processes interleave on an otherwise idle GPU.

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
