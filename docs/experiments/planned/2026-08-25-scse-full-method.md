# Experiment: does the FULL SCSE method improve MORPH's CE?

Status: planned

Written and committed BEFORE any arm produced a checkpoint. Predictions are not editable
once the first run starts; if the method must change, Method gets a dated amendment.

Implementation: [docs/scse-spec.md](../../scse-spec.md), commit `7849da9`.
Decision record: [.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md](../../../.agents/notes/proposed/architecture/2026-08-24-scse-source-centered-core-loop.md).

## Question

Does source-centered state evolution — the learned anchor, the anchor-coordinate deviation
recurrence, and the zero-deviation mask, together — lower MORPH's validation cross-entropy at
a matched step count, against an otherwise identical `tul_a1` control?

This is the experiment [H23](../failures/2026-08-25-scse-stage1-initial-deviation.md) was
mistaken for. H23 tested a non-zero initial deviation ALONE, which the paper never reports as
a configuration and does not credit with any gain. Its result does not transfer to this.

## Hypothesis

SCSE improves CE. The paper's abstract credits the gain to "the learned anchor and the
anchor-coordinate deviation recurrence", and its Table 1 reports the improvement INSIDE the
training loop-depth range, not only under depth extrapolation: WikiText-103 at `T = 8`,
95.6M, 117.1 -> 96.9 PPL, and 50M, 151.1 -> 123.1 PPL. Converted to nats per token those are
**0.189** and **0.205**, so the paper's effect size at a comparable token budget is about
**0.19-0.21 nats**.

The mechanism, if it transfers, is that MORPH stops re-injecting the source on every core
iteration and instead carries it in a fixed anchor, so the loop's trajectory is no longer the
accumulated response to a per-iteration forcing term.

## Predictions

Thresholds are set from measured MORPH quantities, not from taste. The measured val-CE noise
floor is **0.168 nats** (the largest within-run rise that later set a new minimum), and the
across-seed spread of the control's final CE is **0.179 nats** (4.6863 / 4.5303 / 4.5073 on
seeds 1/2/3 in the H21 sweep), so the per-seed SD is about 0.10 nats and the standard error
of a mean over 4 paired seeds is about 0.05.

* **P1 — VALIDITY GATE, not a result.** On every SCSE checkpoint the forcing bias is exactly
  zero, `b_t = T_t(0; e) = 0.000e+00`, and the initial deviation is non-zero,
  `||Delta_0|| / ||h*|| > 1e-3`. On every control checkpoint the baseline anchor `h* = e`
  gives `R_0 = 1.000` and `||Delta_0|| = 0`. If this gate fails, nothing below is readable:
  the arms are not the two things they are supposed to be.
* **P2 — the headline.** Mean final validation CE over the completed seed PAIRS is LOWER for
  SCSE than for the control by at least **0.10 nats**. That is about half the paper's effect
  and about two standard errors, so it is a real claim and not a coin flip.
* **P3 — paired sign.** SCSE's final CE is lower than its own seed's control on at least
  **3 of 4** completed pairs (or all of them, if only 3 pairs finish).
* **P4 — no new instability.** No SCSE seed diverges whose paired control did not. The
  divergence guard arms at `step > 2000`, samples every 20 steps with a ppl ceiling of 1000
  and needs two consecutive strikes, so 2040 is the earliest abort it can emit.
* **P5 — the loop is not frozen.** SCSE's val CE must fall by at least 1.0 nat from its first
  eval to its last. The paper's `h* = H_0(e)` ablation collapses `Delta_0` to zero, the mask
  then holds `Delta_t = 0` forever, and PPL becomes independent of loop depth (their Table 2,
  294.37 against the learned anchor's 200.10). A frozen loop would still train the prelude and
  coda, so it can look like learning; this is the cheap field check that it did not happen.

**REFUTER.** If SCSE's mean final CE is HIGHER than the control's, or if it is higher on 3 or
more of the completed pairs, then the full method does not transfer to MORPH at this scale
and the port is refuted — not merely unproven. Stating this in advance is the point: the
previous round declared the port dead on a configuration the paper never reports, and this
one must be able to die honestly or not at all.

## Method

* Config `tul_a1` for both arms. The SCSE arm adds `model.scse_enabled=true` and NOTHING
  else; the control adds `model.scse_enabled=false`. Every other override is identical.
* `training.steps=3500`, `training.batch_size=6`, `training.ademamix_alpha_cap=3.5`,
  `model.use_kernels=false`, `training.eval_every=250`, `training.ckpt_every=1000`,
  `training.gen_every=0`. That is 3500 x 6 x 1024 = **21.5M training tokens**, against the
  paper's comparable 20.48M-token budget.
* Seeds 1, 2, 3, 4 for both arms, run as PAIRS in the order ctrl-1, scse-1, ctrl-2, scse-2,
  ... so that an interrupted campaign still yields complete pairs. Seed 0 is known to be
  pathological in the control (it diverged in both H21 and H23), so an SCSE seed-0 run is
  appended LAST as a divergence probe against that known control, and is excluded from P2/P3.
* **Controls are re-run, not reused.** TUL arms are not bit-reproducible — the `bag_mean`
  atomics give a ~4% per-step gradient error — so a stored control from a previous commit
  cannot be shown to be a valid control for this one. Re-running removes the confound.
* Runs are strictly SEQUENTIAL on the 5090. One trainer at a time.
* Scoring is by `lab/divergence/score_scse.py`, committed before the first run, which reads
  the training logs and the probe JSONs and prints a verdict per prediction with no judgement
  calls.

## Results

Not yet run.

## Verdict

Not yet run.

## Updated hypothesis

Not yet run.
