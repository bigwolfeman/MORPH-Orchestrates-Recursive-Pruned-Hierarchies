# Experiment: does the forcing bias predict WHICH A1 seeds diverge?

Status: failure

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

## Method amendment 2, 2026-08-24, recorded BEFORE any `b_t` was measured on seeds 1-3

Seeds 0 and 1 have finished. Seed 1's validation CE rose above its running minimum four times
while still trending down, and three of those rises RECOVERED to a new minimum afterwards:

| step | rise above running min | outcome |
|---|---|---|
| 1500 | +0.057 | recovered |
| 2000 | +0.129 | recovered |
| 2500 | +0.005 | recovered |
| 2750 | **+0.168** | recovered |
| 3250 | +0.121 | last eval, no chance to recover |

**Consequence: P1's threshold is below this metric's noise floor.** A rise of 0.168 nats is
demonstrably noise here, because the run set a new minimum after it. P1 calls anything at or
above 0.100 nats a turnaround. Seed 1 therefore scores as "turned around" on a final rise of
0.121 nats that is smaller than a rise the same seed already recovered from.

Seed 0 is a different phenomenon by the same measure: every eval after step 250 sat above the
running minimum, the rise grew monotonically to +0.797, no eval ever recovered, and the
divergence guard aborted the run at step 2040.

**Predictions are NOT changed.** P1 through P4 are scored exactly as written, and this defect is
reported alongside the verdict rather than repaired by moving a threshold after seeing the data.
Two strictly descriptive columns are added to the scorer — the largest rise that recovered, and
the number of evals since the last new minimum — and both are barred from every verdict.

The defect is mine: the 0.1-nat threshold was chosen without first measuring the within-run
spread of this validation metric. The next planned run in this line must set its threshold from
a measured noise floor and must require several consecutive evals with no new minimum, rather
than reading a single final point.

This is recorded before the drift probe has been run on any seed of this sweep, so it cannot
have been fitted to the quantity under test.

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

## Results

Artifacts: [`../results/2026-08-24-tul-forcing-bias-predicts-divergence/`](../results/2026-08-24-tul-forcing-bias-predicts-divergence/)
(per-seed `drift_s*.json`, `val_s*.txt`, and the full `scorecard.txt`).
Scored by `lab/divergence/score_h21.py`, committed at `b04beeb` before any seed of this sweep
was probed.

**Validity gate passed**: `R_0 = 1.000` and trajectory gate `0.0e+00` at every checkpoint of
every seed.

### Validation curves

| seed | min CE | @step | last CE | @step | rise (scored) | peak rise | largest rise that RECOVERED | P1 verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 6.6820 | 250 | 7.4790 | 2000 | **+0.797** | +0.797 | 0.000 | turned around |
| 1 | 4.5653 | 3000 | 4.6863 | 3250 | +0.121 | +0.168 | 0.168 | turned around |
| 2 | 4.4458 | 3000 | 4.5303 | 3250 | +0.085 | +0.151 | 0.151 | held |
| 3 | 4.4674 | 3000 | 4.5073 | 3250 | +0.040 | +0.156 | 0.156 | held |

Seed 0's divergence guard aborted it at step 2040.

### Forcing bias, mean over loop iterations

| step | s0 | s1 | s2 | s3 |
|---|---|---|---|---|
| 500 | **13.130** | 1.605 | 1.472 | 1.625 |
| 1000 | **15.924** | 1.701 | 1.664 | 1.731 |
| 1500 | **22.207** | 1.578 | 1.715 | 1.586 |
| 2000 | **69.631** | 1.667 | 1.916 | 1.698 |
| 2040 | **82.519** | — | — | — |
| 2500 | — | 2.213 | 2.050 | 1.977 |
| 3000 | — | 5.227 | 3.046 | 1.951 |
| 3500 | — | 8.323 | 4.184 | 1.966 |

### Verdicts as scored

| prediction | verdict | what actually carried it |
|---|---|---|
| P1 | HELD (2 of 4) | seed 0 genuinely; seed 1 by a threshold below the noise floor |
| P2 | HELD (1 discordant of 5 pairs) | seed 0 alone; the one discordant pair is (s1, s3) |
| P3 | HELD (5.193x) | seed 0 alone. **With seed 0 removed the ratio is 1.002x** |
| P4 | HELD for s1; NOT SCORABLE for s0 | s0's minimum precedes its first rung (amendment 1) |
| **REFUTER** | **FIRED** | `min(turned) = 1.701 < max(held) = 1.731` |

## Verdict

**H21 is refuted at the pre-registered rung.** The refuter condition was written in advance as
"`b_t` ranges overlapping between the turned-over and not-turned-over groups with no ordering",
and it fired: at step 1000 seed 1 (scored as turned around) carries `b = 1.701`, BELOW seed 3
(scored as held) at `1.731`.

All four predictions read HELD at the same time. That is not a contradiction to be explained
away — it is the signature of a panel dominated by one seed. Every HELD verdict is carried by
seed 0, whose forcing bias is 8x to 40x every other seed at every shared rung. Removing seed 0
collapses P3 from 5.193x to **1.002x**: among the three healthy seeds the forcing bias at step
1000 spans 4.0 % and carries no ordering at all.

**This experiment is also underpowered, which the pre-registration named in advance as a
protocol failure.** It asked for at least two diverging seeds and got one. P1 reads 2 of 4 only
because of the threshold defect recorded in Method amendment 2: seeds 1, 2 and 3 have peak rises
of +0.168, +0.151 and +0.156 — the same behaviour to within 10 % — and the scored rule split
them into different classes purely on where the final eval happened to land. One divergence
cannot support a rank test that was budgeted for four.

Filed under `failures/` for both reasons: the refuter fired, and the contrast the design needed
did not materialise.

## What was learned anyway

**1. The core map is NOT iteration-invariant, and the campaign assumed it was.**
`score_h21.py` shipped with a guard asserting `b_rel` is "constant by construction before
`route_start`". It raised on the real data. Measured on `seedsweep-s1/step_3500.pt`:

* The probe is bit-exact run to run — two independent runs agree to `0.00e+00` — so the
  variation is deterministic, not kernel nondeterminism.
* Pinning `ret_state` to iteration 0's collapses the across-iteration spread to **exactly
  `0.00e+00`**. Pinning `iter_idx` instead leaves it unchanged at `3.87e-03`.

So `T_t` depends on `t` entirely through the **GLA retention state carried across loop
iterations**. The effect is small — worst spread `4.25e-03` — and the readout was switched from
iteration 0 to the mean, a change 12x smaller than the tightest seed-to-seed gap it must
resolve, so it moves no verdict here. `score_stage0.py` carries the same false assertion in its
docstring and its numbers are unaffected for the same reason, but the claim is wrong there too.

This does not conflict with SCSE. Their Theorem 2 already indexes the forcing bias as `b_k`, so
a `t`-dependent bias is what the theory expects. It conflicts only with the campaign's
assumption.

**2. Seed 0 confirms the forcing bias tracks divergence when divergence is real.**
`b` at step 500 is 13.130 against 1.472-1.625 for the healthy seeds — an 8x separation at the
first rung — rising to 82.519 at the abort. This is one seed, and this campaign's own trap list
forbids reading a single run as a result. It is consistent with H21, not evidence for it.

**3. EXPLORATORY, not a result: the late rungs order the healthy seeds.**
At steps 2500, 3000 and 3500 the ordering is s1 > s2 > s3 every time, and the gap widens
(8.323 / 4.184 / 1.966 at 3500, a 4.2x spread from a 4.0 % spread at step 1000). That matches
the scored final-rise order s1 > s2 > s3 exactly.

Three caveats, all of which have to travel with that observation:
* It is post-hoc. The pre-registered rung was 1000, where the signal is absent.
* n = 3, so a correct ordering has probability 1/6 under the null. That is not significant.
* **The `peak_rise` ordering DISAGREES**: s1 > s3 > s2, not s1 > s2 > s3. The match therefore
  depends on which rise statistic is used, and one of the two does not match.

The robust part is qualitative: seed 3's `b` is flat across the whole run (1.625 -> 1.966,
+21 %) while seeds 1 and 2 climb steeply after step 2500 (+419 % and +184 %). Whether that
climb predicts a later collapse cannot be answered by a run in which none of them collapsed.

## Updated hypothesis

H21 does not survive as written. The replacement to pre-register next:

**H22. The forcing bias predicts divergence through its GROWTH RATE late in training, not
through its level at a fixed early step.**

The next run must fix four things this one got wrong:

1. **Get more than one divergence.** Raise `ademamix_alpha_cap` above 3.5 or extend the horizon.
   Three of four seeds stayed healthy for 3500 steps. A design needing a two-group contrast must
   first establish it produces two groups.
2. **Set the turnaround threshold from a measured noise floor**, not from a guess. This metric's
   within-run recovered rise reaches 0.168 nats, so a threshold of 0.1 is meaningless. Require
   several consecutive evals with no new minimum rather than reading a single final point.
3. **Checkpoint densely early** — `ckpt_every` at 100 or less over the first 500 steps — so a
   genuinely pre-turnaround rung exists. Seed 0's minimum at step 250 preceded its first rung in
   both this experiment and Stage 0.
4. **Pre-register the growth rate**, `d log b / d step` over a stated window, as the predictor,
   since that is where this run's signal appears and the level at step 1000 has now been
   measured not to carry one.
