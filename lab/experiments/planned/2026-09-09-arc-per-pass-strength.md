# Planned: ARC per-pass-strength panel — can a ternary core be a strong per-pass map?

Status: planned
Date: 2026-09-09 (frozen before launch; Wolfe: "once the research comes back plan your next
arms for 5k runs"). Arc: `2026-09-04-loop-contribution-arc.md`, per-pass-strength row.
Chained behind the 25k horizon run in the queue (`arc/run_strength.sh` waits for `HORIZON
COMPLETE`).

## Question

The depth-candidates panel found the one factor that moves the loop's contribution: bf16
core weights (K1−K6 0.033 → 0.168, K3−K6 0.0009 → 0.012), with the mechanism in the
anatomy (core MLP branch out/in per pass 0.19–0.24 ternary against 0.75–0.96 bf16). On the
Parcae-entry checkpoint the core MLPs' latent weights are Gaussian-shaped (scale over std
0.80), 30.5 % of them sit in the dead zone, and the absmean rule shrinks each ternary
weight's norm to 0.668 of the latent's, so a gate-up-down MLP emits 0.44 of its bf16 twin
from the scale alone. The prior-art search (Awesome-Loop-Models, 2026-09-09) found no
looped model trained ternary from scratch; the field's fixes for a weak per-pass map are
all gains on the write path. Ternary stays (Wolfe's rule). The question: which
ternary-preserving change restores the per-pass strength, and does the loop's
contribution follow it?

## Hypothesis

H-str: per-pass strength is the lever and the scale rule is the cheapest handle. The
norm-matching scale lifts the core MLP branch ratio above 0.4 and K1−K6 above 0.08;
learnable scales do the same or better; the narrower dead zone helps less. H-str′: the
branch ratio moves but the K-curve does not (strength is necessary, not sufficient).
H-str″: nothing moves; the shrink is compensated elsewhere and the limiter is the codes,
not the scale.

## Method

Base: `notul_parcae_entry` (ternary backbone, noise state init, all-dim carry with learned
B, no fixed-point term, seq 1024, batch 6, 5,000 steps, ramp 1000 then flat 1e-4). Arms,
one factor each:

| arm | config | the one change |
| --- | --- | --- |
| `scale-norm-match` | `notul_scale_norm_match` | `ternary_scale_mode: norm_match` (new): symmetric codes, scale = ‖W‖/√nnz per tensor so the ternary norm equals the latent norm; no parameters |
| `scale-ttq` | `notul_scale_ttq` | `ternary_scale_mode: ttq`: learnable γ₊, γ₋ per tensor (in the no-decay group) |
| `threshold-0.3` | `notul_threshold_03` | `ternary_threshold: 0.3` (dead zone ~18 % instead of 31 %) |
| `precision-bf16-all` | `notul_precision_bf16` | `training.ternary: false` (the precision axis's third point; the ceiling) |

Order as listed. Runner `arc/run_strength.sh`: per arm a 12-step smoke, the draw with the
sustained tripwire, then `core_depth_sweep.py` at 2,500 and 5,000 (480 rows, depths
0,1,2,3,6,9,12,16, per-token files), `core_anatomy.py --rows 3 --depth 8` and
`core_init_probe.py --rows 96` at 5,000. Commit pinned in `arc/STRENGTH_COMMIT`. Results
to `lab/experiments/results/2026-09-09-per-pass-strength/`; scored with the shared readers
against `parcae-entry` (ternary base) and `depthcand-dense-core` (bf16 core). Loop
CONTRIBUTION is the reading (K-curve, branch out/in, state movement); CE at 5,000 is a
horizon reading and ranks nothing.

## Predictions (frozen)

- **P-str-a (survival).** HEALTHY to 5,000: norm-match **80 %** (a 1.5× gain on every
  ternary layer at step 0 under the ramp), ttq **80 %**, threshold-0.3 **85 %**, bf16-all
  **90 %**.
- **P-str-b (mechanism: core MLP branch out/in at iteration 6, anatomy).** Above 0.4:
  norm-match **60 %**, ttq **50 %**, threshold-0.3 **50 %**. bf16-all above 0.7: **70 %**.
- **P-str-c (contribution).** K1−K6 above 0.08: norm-match **45 %**, ttq **40 %**,
  threshold-0.3 **35 %**; bf16-all within 0.03 of the bf16-core arm's 0.168: **55 %**. K3−K6
  above 0.005 with the CI above 0: norm-match **40 %**, ttq **35 %**, threshold-0.3 **30 %**;
  above 0.02 (the panel bar): any ternary arm **15 %**. Every arm converges by 6 (K6−K12
  within ±0.005): **80 %**.
- **P-str-d (ttq learns a gain).** ttq's mean |γ₊| over the core MLPs at 5,000 exceeds 1.3×
  its init: **50 %**.
- **P-str-e (horizon readings).** Each ternary arm's token-paired CE at depth 6 against
  `parcae-entry`: norm-match within ±0.03: **55 %**; threshold-0.3 within ±0.03: **60 %**.
  bf16-all under the bf16-core arm's 3.9275: **60 %**.
- **P-str-f (cost).** Ternary arms within 1.1× of 1.00 h: **80 %**; peaks under 16 GB: **90 %**.

## Binding

- Any ternary arm with P-str-b TRUE and K1−K6 > 0.08 ⇒ H-str: the scale rule was the
  limiter; that mode becomes the recipe's ternary rule in `base.yaml` (an Agent Note in the
  same change) and the paid TUL loop is re-read on it.
- P-str-b TRUE and P-str-c FALSE on every ternary arm ⇒ H-str′: strength alone is not
  sufficient; the next lever is the codes (dead-zone reactivation, Tequila / SZT).
- Nothing moves ⇒ H-str″; the ternary lane on loop contribution closes at 5k and the
  under-ternary work moves to the 25k horizon.
- P-str-a FALSE on an arm ⇒ its trip step and probe go to the divergence README; norm-match
  re-runs ONCE with the gain ramped in over the first 1,000 steps if it is the one that trips.
- NO 20k run from this experiment.

## Not verified before launch

No GPU forward of `norm_match` or `ttq` on the full model before the queue's smoke (the
mode is tested on CPU: codes equal the symmetric codes, norms match to fp16 precision,
gradient is the identity, per-group and ragged tails, refuses the scale cap and the EMA).
Whether the deploy quantizer (`morph/posttrain`) reads a norm-matched or TTQ scale
correctly at export (the training arm does not need it; noted for the Binding's recipe
change). The ttq mode on a compiled model.

## Results

(after the run)

## Verdict

(after the run)

## Updated hypothesis

(after the run)
