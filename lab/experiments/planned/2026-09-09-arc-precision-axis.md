# Planned: ARC precision axis — loop contribution with no ternary anywhere (the third point)

Status: planned
Date: 2026-09-09 (frozen before launch; Wolfe: "we needed to test ternary vs bf16. REMEMBER
WE ARE TRYING TO SOLVE LOOP CONTRIBUTION."). Arc: `2026-09-04-loop-contribution-arc.md`,
precision row. Chained behind the width re-sweep in the queue (`arc/run_precision.sh`).

## Question

Two points on the weight-precision axis already ran on the Parcae-entry recipe: ternary
everywhere (`parcae-entry`: K1−K6 0.033, K3−K6 0.0009, core MLP branch out/in 0.19–0.24 per
pass) and bf16 in the looped core only (`depthcand-dense-core`: K1−K6 0.168, K3−K6 0.012,
MLP branch 0.75–0.96). The third point is bf16 everywhere (`training.ternary: false`;
embeddings keep the recipe's int6 QAT). It says whether the loop's contribution depends on
the CORE's precision alone or on the prelude and coda too.

## Hypothesis

H-prec: the core's precision is what matters. bf16 everywhere lands within 0.03 of the
bf16-core arm on K1−K6 and within 0.005 on K3−K6. H-prec′: the prelude and coda matter too;
bf16 everywhere moves K1−K6 above 0.20 or K3−K6 above 0.02.

## Method

One arm, `precision-bf16-all` (`notul_precision_bf16`), the Parcae-entry recipe with
`training.ternary: false` and nothing else changed; 5,000 steps; the same readout as the
other panels (sweeps at 2,500 and 5,000 over 480 rows at depths 0,1,2,3,6,9,12,16 with
per-token files, `core_anatomy.py --rows 3 --depth 8`, `core_init_probe.py --rows 96`).
Commit pinned in `arc/PRECISION_COMMIT`. Scored by the depth-candidates scorer's readers
against `parcae-entry` and `depthcand-dense-core`. Loop CONTRIBUTION is the reading; the
5,000-step CE is reported as a horizon reading and ranks nothing.

## Predictions (frozen)

- **P-prec-a (survival).** HEALTHY to 5,000: **90 %**.
- **P-prec-b (contribution).** K1−K6 within 0.03 of the bf16-core arm's 0.168: **55 %**;
  above 0.20: **25 %**. K3−K6 within 0.005 of 0.012: **55 %**; above 0.02 with the CI above
  0: **20 %**. Converged by 6 (K6−K12 within ±0.005): **80 %**.
- **P-prec-c (mechanism).** Core MLP branch out/in at iteration 6 above 0.7: **70 %**.
  Consecutive state movement at iteration 6 above 5 %: **60 %**.
- **P-prec-d (horizon reading).** Token-paired CE at depth 6 under the bf16-core arm's: **60 %**.
- **P-prec-e (cost).** Wall within 0.95× of the bf16-core arm's 0.88 h: **60 %**; peak under 16 GB: **90 %**.

## Binding

- H-prec ⇒ the core's precision is the lever and the prelude/coda stay ternary in every
  under-ternary arm that follows (the proposed ternary-core note).
- H-prec′ ⇒ the under-ternary work covers the whole backbone, not the core alone.
- NO 20k run from this experiment.

## Not verified before launch

No GPU forward of the config before the queue's smoke. Whether `training.ternary: false`
leaves any other quantizer on (embed int6 stays by design; attention-projection quant is
off in the recipe).

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
