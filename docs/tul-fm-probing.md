# TUL-FM probing doctrine — how to measure a plan latent without fooling yourself

Companion to the arc proposal:
[.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md](../.agents/notes/proposed/architecture/2026-08-28-tul-fm-arc.md).
This file is the HOW; the note is the WHY. Every rule here was paid for by a specific
failure in the 2026-08 campaign; the failure is cited so the rule survives skepticism.

## 1. The validated instrument set (reuse these, do not rebuild)

| Instrument | File | What it measures |
|---|---|---|
| Plan worth (zero / shuffle / no-loop) | `lab/divergence/slot_path_worth.py` | ce_main cost of removing or corrupting z, through the SHIPPED coda |
| Whole-slot shuffle | `slot_path_worth.py::plan_shuffled` | span-specificity of z's worth (rank-agnostic; permutes whole slots within a row) |
| Token-tax (reader starvation) | `slot_path_worth.py::token_tax` | forces the coda's token dropout ON at eval by flipping the TRAINING branch of the shipped function — never reimplement the drop |
| Dose-response sweep | `lab/divergence/reader_or_target.py` | worth and specificity at p in {0, .5, .9, 1.0}; reseed torch AND cuda before EVERY condition so all conditions drop the same positions |
| Takeover score | `lab/divergence/score_arms.py` | core share > 0.5 on >30% of last 50 probed steps; refuses <20 samples |

## 2. The positive-control signature (ctrl-s3, tul_a1 aux ON, step 3000)

A healthy content-bearing plan produces THIS shape. A new arc's plan should be compared
against the shape, not just any single cell:

| | p=0.00 | p=0.50 | p=0.90 | p=1.00 |
|---|---|---|---|---|
| shuffle cost (nats, ce_main) | 0.0096 | 0.0409 | 0.0800 | 0.0867 |
| zero cost @p=1.0 | | | | 0.0863 |

Monotone rise ~9x, and at p=1.0 shuffle ~= zero (=> all worth is span-specific). The
aux-off arms sit at 0.0001–0.0104 across the whole sweep. ctrl-s3 is n=1 — the single
most valuable checkpoint on disk — and every claim of "healthy" traces to it.

## 3. The retrieval probe (replaces the blind decoder — build for P1)

The blind-decoder probe was retired after refusing twice (memorization at 1.6k examples,
underfitting at 41k; `lab/experiments/failures/2026-08-28-plan-content.md`). Its
replacement has no trained component to confound:

- Take z_i and the pooled target representations y_j of every span in the batch.
- Score cos(z_i, y_j); ask whether j = i+1 ranks first. Report top-1 and MRR against
  chance = 1/N. No decoder, no fit/eval split problem, minutes to run.
- Run it on the FM arm AND on ctrl-s3 (positive control) AND on an aux-off arm
  (negative control) in the same invocation, same batches.
- SIGReg's isotropy is what makes cosine ranking meaningful; also report the target
  effective rank so collapse can't masquerade as easy retrieval.

## 4. Rules written in blood (each cites its scar)

1. **Report the shuffle COST, not the specificity FRACTION, whenever the zero cost is
   not comfortably positive.** The fraction's denominator collapses through zero
   (tg3b at p=1.0 read −55.4%; R4 failure in `2026-08-28-reader-or-target.md`).
2. **`exit=0` is not "no takeover".** Only `score_arms.py` decides; both a1noaux seeds
   exited 0 and both had taken over (3556, 3505).
3. **Block backward gain is the MECHANISM (gain > 1, r² ≥ 0.5); core share is the
   SYMPTOM.** Gain separates arms at matched steps; share verdicts are run-length
   confounded (every "held" verdict at 3500 steps is untested past 3500).
4. **Worth metric is `ce_main`, never `loss`** — `loss` folds in the §5 half-weighted
   emit/plast positions. For cross-arm curves use `val/ce_tokens`.
5. **Pin `training.ademamix_t_beta3` explicitly on every run.** null falls back to
   `training.steps` and silently changes the optimizer between run lengths. Third
   experiment it cost, 2026-08-28. Verify from Hydra's frozen `config.yaml`, not the CLI.
6. **n=1 comparisons are unreadable** (11-step decorrelation, 6.5% median spread).
   Two seeds minimum; within-run measurements beat across-run wherever possible.
7. **Eval-step val jumps are correlated ACROSS arms and seeds** (all six panel curves
   step down together at eval 2250) — an eval-batch sampling artifact. Compare arms at
   the same step only; never read a within-run val jump as a model event.
8. **Forcing new inputs at eval measures OOD shock, not worth** (prediction B2 fell
   3.6–7.3x in the wrong direction). A condition is comparable only if the model
   TRAINED with that input distribution, or if a positive control absorbs the shock.
9. **A probe must run on the shipped path.** Every arm runs `_tul_core`; a token-path
   guard protects nothing (caught twice in one day, 2026-08-27).
10. **Never edit a running script; write a new one** (`slotpay.sh`, killed one arm
    later). Detached scripts need `setsid`, then verify with a later `pgrep` call.
11. **Generation claims need the diversity guard** — gen-PPL 1.46 on a repetition loop
    vs 32.44 on real text; always pair gen-PPL with rep4/distinct-3
    (`morph/inference/gen_metrics.py`), sampled decoding ranks, greedy is a diagnostic.
12. **Pre-register before the run** (`lab/experiments/planned/`), predictions frozen;
    a failed prediction is withdrawn, not defended (B2 precedent).

## 5. Phase gates for TUL-FM (bind these into each planned file)

- **P1 (planner on frozen backbone):** retrieval top-1 >> 1/N on held-out rows with the
  aux-off negative control at ~chance; target effective rank stable; if ctrl-s3's
  retrieval is ALSO at chance the probe is broken, not the arm.
- **P2 (coda wired, token scarcity):** dose-response reproduces the §2 signature shape;
  shuffle cost @p=0.9 ≥ 0.08 nats; block gain < 1.0 at every probed step; ce_main in the
  control band at step 3000.
- **P3 (from scratch):** beat A3 at matched WALL-CLOCK, or at 4096 beat A3 on CE
  stratified by distance-to-evidence, or win multi-span coherence at matched CE.

## 6. Model-level comparison against the non-TUL baseline (Wolfe directive, 2026-08-28)

Planner-level losses are incommensurable across objectives (an EDM weighted-denoising
loss, a CFM velocity loss, and a token CE cannot be ranked against each other), and the
DiffusionBlocks paper's own §5.4 admits it cannot compute true perplexity — its
MAUVE + teacher gen-PPL numbers are NOT 1:1 with held-out CE. Therefore:

- **At the planner level (P1x):** the ONLY cross-arm gate is the retrieval probe. It is
  objective-agnostic by construction — every arm answers the same lineup question — so
  EDM vs CFM vs any future objective compare 1:1. Never rank arms by their training
  losses, even rescaled.
- **At the model level (P2/P3):** the comparison against the non-TUL baseline (A3) is
  (a) **held-out token CE** — valid because TUL-FM leaves the token path on ordinary CE,
  so `val/ce_tokens` stays a real likelihood on both sides; and
  (b) **AR generation**: gen-PPL under a judge model PLUS the diversity guard
  (rep4 / distinct-3, sampled decoding ranks, greedy as diagnostic only —
  a repetition loop scores gen-PPL 1.46 vs real text's 32.44).
- **Never MAUVE.** And never invent a "planner perplexity" — the planner path has no
  likelihood, and dressing its loss up as one is theater.

## 7. True flow matching is a required arm, not a variant

The EDM denoiser is the DiffusionBlocks formulation. A genuine conditional
flow-matching arm (straight-line interpolation, velocity target, uniform t) must run
beside it under the SAME retrieval gate before any objective-level conclusion is
drawn — the two differ in weighting, source distribution, and conditioning variable,
and P1's sigma_data episode showed exactly how much those choices move the result.
CFM knob to respect: the SOURCE distribution's scale is load-bearing at low capacity
(DeepWeightFlow App. H: source std 0.001 vs 0.01 flipped a ViT result by 4.6 points
with 20x the variance).
