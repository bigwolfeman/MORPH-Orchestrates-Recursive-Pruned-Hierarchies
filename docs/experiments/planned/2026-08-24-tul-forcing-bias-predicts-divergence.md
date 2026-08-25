# Experiment: does the forcing bias predict WHICH A1 seeds diverge?

Status: planned

## Question

[Stage 0](../successes/2026-08-24-tul-forcing-bias-arm-control.md) established that the
zero-deviation forcing bias `b_t(e) = T_t(0; e)` is arm-linked: arm A1 carries 18-30 % more
anchor response than arm A0 at every one of seven matched rungs, on separately trained models
in one build, and grows twice as fast. The refuter did not fire.

It did NOT establish that the gap causes the failure, because neither Stage 0 arm diverged. The
gap was measured between two HEALTHY models, so it is a property of the arm and not yet a cause
of the bug. The only evidence pointing at causation is a single-run comparison — the
`onset-capture` A1 that took over carries `b = 2.265` at step 1800 where Stage 0's healthy A1
carries `1.764` — and this campaign's own trap list forbids reading single-run comparisons.

**Question.** Across A1 seeds that do and do not diverge, does `b_t` measured BEFORE any
turnaround predict which ones turn over?

## Hypothesis

H21. The forcing bias is not merely correlated with the arm; it is the quantity whose growth
drives the divergence. Seeds that later turn over carry a higher `b_t` early, before the
turnaround is visible in the loss.

## Method

Four seeds (0, 1, 2, 3) of `tul_a1`, from scratch, 3500 steps, batch 6,
`ademamix_alpha_cap=3.5`, `model.use_kernels=false`. `eval_every=250` for fine turnaround
resolution; `ckpt_every=500` giving seven aligned rungs per seed (500 … 3500).

3500 steps rather than 1900: the campaign's window for the CE minimum is step 500-2000, and
Stage 0's 1900-step arm ended inside it without ever showing the rise. 3500 clears it.

All four seeds run fresh and identically. Seed 0's Stage 0 run is NOT resumed: its `eval_every`
differed, evaluation consumes RNG, and MORPH decorrelates within 11 steps of any perturbation,
so a resumed seed 0 would not be comparable to the other three.

Probe: `../../lab/divergence/drift_probe.py --config-name tul_a1 --ckpt-dir <seed>`.

**Validity gate.** `R_0 = 1.000` at every checkpoint of every seed, and the probe's trajectory
gate at `0.0`. Both held at all 14 Stage 0 checkpoints. If either fails, nothing is readable.

**Power, stated in advance.** Four seeds is a screening experiment, not a powered test. A
perfect rank ordering of four items has probability 1/24 = 0.042 under the null, so a clean
ordering IS meaningful at this n; anything short of near-perfect is not. No correlation
coefficient will be quoted as if it were significant.

## Method amendment 1, 2026-08-24, recorded BEFORE any `b_t` was measured

Seed 0's validation minimum is at step **250** — 6.6820, then 6.7751 at 500 and 7.0968 at 750,
with train loss at 7.26 by step 1000. That is earlier than the campaign's stated 500-2000 window
for the minimum, and earlier than this experiment's first checkpoint at step 500.

**Consequence: P2's premise fails for seed 0.** P2 reads `b_t` at step 1000 and calls it "before
any turnaround". For seed 0 it is not; the turnaround began before the first rung exists.

**Predictions are NOT changed.** P2 will be scored at step 1000 exactly as written, and its
premise failure reported alongside the verdict. `b_t` at step 500 will be reported as an
EXPLORATORY secondary readout, labelled as such, and no verdict will rest on it.

This is recorded at the moment the val curve was seen and before the probe was run on any
checkpoint, so the amendment cannot have been fitted to the quantity under test. If a future
version of this experiment wants a genuinely pre-turnaround rung it needs `ckpt_every` at 100
or less over the first few hundred steps, which is a different run.

## Predictions

Written before any seed reached step 500.

* **P1.** At least 2 of the 4 seeds show a validation-CE turnaround of at least 0.1 nats by step
  3500. This is the contrast the rest depends on.
* **P2.** `b_t` at step 1000 — before any turnaround — rank-orders the seeds by turnaround step,
  earlier turnaround meaning higher `b_t`, with at most one adjacent swap.
* **P3.** Mean `b_t` at step 1000 for seeds that turn around exceeds the mean for seeds that do
  not, by at least 10 %.
* **P4.** Within each seed that turns around, `b_t` at the last rung before the turnaround
  exceeds `b_t` at step 500.

## What would refute H21

`b_t` ranges overlapping between the turned-over and not-turned-over groups with no ordering, or
an ordering in the wrong direction. That would leave the forcing bias arm-intrinsic but not
predictive of failure, and SCSE would lose its causal claim on MORPH — the port would then rest
only on the structural argument that `Delta_0 = 0` makes the whole trajectory a propagated
forcing response, with no measured link to the bug.

If P1 fails — fewer than 2 seeds turn around — the experiment is underpowered rather than
informative. That is a protocol failure, files under `failures/`, and the next planned run must
either extend the horizon or raise `ademamix_alpha_cap`.
