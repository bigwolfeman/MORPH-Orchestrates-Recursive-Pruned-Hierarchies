# Experiment: does the FULL SCSE method improve MORPH's CE?

Status: failure

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

## Method amendment 2026-08-25 — stopped at 3 pairs

The campaign was cut from 4 pairs + a seed-0 divergence probe to **3 pairs**
(seeds 1, 2, 3). Predictions are UNCHANGED; only the number of replicates is.

Reason, recorded before the remaining runs finished and with only pair 1 in hand: pair 1
returned control 4.6863 against SCSE 6.4277, a gap of 1.74 nats — about ten times the
0.168-nat noise floor — and, more importantly, the SCSE arm's TRAINING loss is flat from
step 200 (6.53 / 6.55 / 6.38 / 6.53 at steps 200 / 1000 / 3000 / 3400). That is a stall, not
an underperformance: a method that merely fails to help tracks below the control and keeps
descending. A live probe of the seed-1 checkpoints shows the deviation running 10-33x the
anchor norm and jumping 80x in a single loop iteration.

Three pairs is the minimum this file already specifies as scorable, so the verdict on P2-P5
is still issued under the original thresholds. The GPU hours freed go to diagnosing the
stall, which is worth more than a fourth replicate of a gap that is already ten times the
noise floor. Decision taken by the user after being shown pair 1.

Consequences for scoring, stated now rather than after the fact:
* P3 reads as "SCSE lower on all 3 of 3 pairs", which this file already provides for.
* The seed-0 divergence probe is NOT run, so P4 is scored on seeds 1-3 only.
* Nothing else moves. The refuter is unchanged and can still fire.

## Results

Scored by `lab/divergence/score_scse.py` (committed before the first run) against the probe
JSONs and training logs. Raw: `docs/experiments/results/2026-08-25-scse-full-method/`.

**P1 VALIDITY GATE — PASSED at every checkpoint of both arms.** The SCSE arm has
`b_t = 0.000` at all 8 loop iterations and `||Delta_0||/||h*||` = 0.12; the control has
`R_0 = 1.000` and `Delta_0 = 0`. The arms are the two things they claim to be, so what
follows is a result about SCSE and not about a broken arm.

| seed | control CE | SCSE CE | delta |
| --- | --- | --- | --- |
| 1 | 4.6863 | 6.4277 | **+1.7414** |
| 2 | 4.5303 | 6.1722 | **+1.6419** |
| 3 | 4.5073 | 6.1715 | **+1.6642** |
| mean | 4.5746 | 6.2571 | **+1.6825** |

| prediction | verdict | number |
| --- | --- | --- |
| P2 mean CE improves >= 0.10 nats | **FAILED** | -1.6825 (SCSE worse) |
| P3 SCSE lower on >= 3 of 3 pairs | **FAILED** | 0 of 3 |
| P4 no new divergence | HELD | no aborts in either arm |
| P5 val CE falls >= 1.0 nat | **FAILED** | 0.191 / 0.378 / 0.501 |
| **REFUTER** | **FIRED** | mean worse AND worse on 3 of 3 |

Three seeds agree to within 0.10 nats. This is not sampling.

## Verdict

**The full SCSE method does not transfer to MORPH at this scale. The refuter fired.**

This is a real negative result, not a broken arm: the pre-registered validity gate passed, the
implementation survived two adversarial audit rounds, and the recurrence is machine-checked in
Lean ([lab/scse-lean](../../../lab/scse-lean/README.md)) to recover the paper's stability
regime and to reduce exactly to the published algorithm when the carry is an identity.

**The failure mode is a STALL, not a slowdown.** SCSE tracks the control to step ~250 and then
stops. Training loss on seed 1: 6.53 / 6.55 / 6.38 / 6.53 at steps 200 / 1000 / 3000 / 3400,
against the control's 6.58 / 5.67 / 4.87 / 5.23. P5 was written as a frozen-loop guard and it
caught this.

### Mechanism — what was measured

1. **The deviation explodes.** `||Delta_t||/||h*||` across the loop, seed 1:
   step 1000 -> 0.10, 1.98, 9.27, 17.45, 25.40, 32.67; step 3000 -> 0.10, 8.30, 8.04, 8.48,
   9.17, 9.84. It jumps ~80x in ONE iteration. `h_T = h* + Delta_T` is then ~90 % `Delta_T`:
   the fixed reference SCSE exists to provide is swamped by the thing it is meant to anchor.
   The mask is fully active throughout, so this is NOT the paper's frozen-loop ablation.
2. **The core is driven expansive.** Core MLP `sigma_max`, control vs SCSE: 1.44/1.45 at step
   100, 1.87/2.83 at 400, 3.03/**6.04** at 2000. The spectral norms separate at steps 200-400;
   the loss curves separate at 250-500. The timing matches.
3. **The loop is numerically unstable in the precision it trains in.** Recomputing ONE core
   step in bf16 instead of fp32, per-iteration relative error:
   control **4.5e-2 to 7.7e-2**, SCSE **4.1e-1 to 7.1e-1** — about 10x worse. Training runs
   under `torch.autocast(bfloat16)`, so the SCSE loop computes something 40-70 % away from the
   exact map at every iteration, six to eight times per forward. Gradients through that are
   noise-dominated, which is a sufficient explanation for a stall at step 200.

### A sub-hypothesis that was tested and REFUTED

`G(D) = stack(D) - D` subtracts two vectors that are 88 % aligned with similar magnitude, so
catastrophic cancellation in bf16 looked like the obvious culprit, with an obvious one-line
cure: the algebraically identical convex form `(1-s)*D + s*stack(D)`, which contains no
subtraction. **It made no difference at all** — SUBTRACT and CONVEX agree to every digit
(6.38e-1 vs 6.38e-1, 6.74e-1 vs 6.74e-1, ...). The error is entirely inside `stack(D)` itself
(6.4e-1 to 9.3e-1 in bf16). The arithmetic is not the problem; evaluating the core stack AT
THE DEVIATION is.

## Updated hypothesis

The port is faithful and the method still loses, so the interesting question is why MORPH
specifically rejects it.

**Leading hypothesis: a scale/conditioning mismatch between SCSE's entry condition and a
pre-normalised block stack.** `Delta_0` enters at ~0.1x the anchor norm, but MORPH's core
blocks are RMSNorm-pre-normalised — their output scale is set by learned weights and is
independent of input scale. The first core application therefore rescales `Delta` by ~80x, the
optimizer chases it, `sigma_max` doubles against the control, the loop becomes expansive, and
in bf16 the resulting map is 40-70 % noise per step. The Lean result narrows where to look:
the carry provably cannot cause expansion, so it must come through the branch term `U`, and
`sigma_max` is a statement about `U`.

**This is a hypothesis, not a finding.** Nothing here isolates cause from effect: the
expansion could be driving the noise, or the noise could be driving the optimizer into the
expansive regime. Distinguishing them needs an intervention, and none was run.

**Next experiments, in value order, none run:**
1. Match `Delta_0` to the scale the core expects instead of `0.1x`. Directly tests the
   leading hypothesis and is a two-line change.
2. `s = 1.0`. The Lean result proves every `s` in (0,1] is non-expansive, so stability does
   not constrain the choice, and `s = 1` is exactly MORPH's own core map in deviation
   coordinates. It is the natural second point; a SMALLER `s` is contraindicated because `s`
   also damps the loop's Cayley stream mixing.
3. Run the SCSE arm in fp32 for a few hundred steps. If the stall lifts, the bf16 conditioning
   is causal rather than incidental. Expensive but decisive.

**What is NOT justified by this result:** concluding anything about SCSE as published. The
paper evaluates 22M-139M models on WikiText-2/103, OpenWebText and C4 with uniform depth
sampling and without truncated BPTT, stochastic depth, ternary QAT or structured sparsity.
MORPH has all of those, plus a 6-block core where the paper has one, plus a HyperConnection
carry the paper does not model. A null here is a statement about this port on this model.
