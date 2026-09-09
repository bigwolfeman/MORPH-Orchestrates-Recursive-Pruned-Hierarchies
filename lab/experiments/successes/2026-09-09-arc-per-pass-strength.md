# ARC per-pass-strength panel — can a ternary core be a strong per-pass map?

Status: success
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
| `threshold-03` | `notul_threshold_03` | `ternary_threshold: 0.3` (dead zone ~18 % instead of 31 %) |
| `precision-bf16-all` | `notul_precision_bf16` | `training.ternary: false` (the precision axis's third point; the ceiling) |

Order as listed. Runner `arc/run_strength.sh`: per arm a 12-step smoke, the draw with the
sustained tripwire, then `core_depth_sweep.py` at 2,500 and 5,000 (480 rows, depths
0,1,2,3,6,9,12,16, per-token files), `core_anatomy.py --rows 3 --depth 8` and
`core_init_probe.py --rows 96` at 5,000. Commit pinned in `arc/STRENGTH_COMMIT`. Results
to `lab/experiments/results/2026-09-09-per-pass-strength/`; scored with the shared readers
against `parcae-entry` (ternary base) and `depthcand-dense-core` (bf16 core). Loop
CONTRIBUTION is the reading (K-curve, branch out/in, state movement); CE at 5,000 is a
horizon reading and ranks nothing.

**Amendment 2026-09-09 (after the three ternary arms, before the fourth).** The horizon run
was stopped at step 10,200 on Wolfe's decision, so the runner launched immediately
(`NOWAIT=1`, commit `d2670e2`) with the three ternary arms. The arm `threshold-0.3` was
renamed `threshold-03` before launch (a dotted arm name breaks the shared readers' regex).
`precision-bf16-all` is deferred to the GPU window after Wolfe's 17:00 CST use of the
machine; the panel is filed on the three ternary arms (the Binding needs only those) and the
ceiling reading is appended to Results when it runs. The Binding's recipe change went in the
same day (commit on master: `base.yaml` `ternary_scale_mode: norm_match`, the rule carried
through the carve and the deploy packer, the Agent Note moved to `implemented/`).

## Predictions (frozen)

- **P-str-a (survival).** HEALTHY to 5,000: norm-match **80 %** (a 1.5× gain on every
  ternary layer at step 0 under the ramp), ttq **80 %**, threshold-03 **85 %**, bf16-all
  **90 %**.
- **P-str-b (mechanism: core MLP branch out/in at iteration 6, anatomy).** Above 0.4:
  norm-match **60 %**, ttq **50 %**, threshold-03 **50 %**. bf16-all above 0.7: **70 %**.
- **P-str-c (contribution).** K1−K6 above 0.08: norm-match **45 %**, ttq **40 %**,
  threshold-03 **35 %**; bf16-all within 0.03 of the bf16-core arm's 0.168: **55 %**. K3−K6
  above 0.005 with the CI above 0: norm-match **40 %**, ttq **35 %**, threshold-03 **30 %**;
  above 0.02 (the panel bar): any ternary arm **15 %**. Every arm converges by 6 (K6−K12
  within ±0.005): **80 %**.
- **P-str-d (ttq learns a gain).** ttq's mean |γ₊| over the core MLPs at 5,000 exceeds 1.3×
  its init: **50 %**.
- **P-str-e (horizon readings).** Each ternary arm's token-paired CE at depth 6 against
  `parcae-entry`: norm-match within ±0.03: **55 %**; threshold-03 within ±0.03: **60 %**.
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

Scored by `results/2026-09-09-per-pass-strength/score_strength.py` (`score_strength.txt`);
token-paired on 491,520 tokens / 481 blocks; every sweep-vs-trainer gap −0.020 to −0.030 (OK).

| arm | verdict | K1−K6 [CI] | K3−K6 [CI] | K6−K12 | core MLP out/in at pass 6 | attn out/in at 6 | movement per pass (%) | carrier rank t8 | depth-6 CE at 5k | vs base (paired) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `scale-norm-match` | HEALTHY (22@340) | **+0.1849 [+0.1816, +0.1883]** | **+0.0139 [+0.0131, +0.0146]** | −0.0024 | **0.86–1.10** | 0.15–0.56 | 61/28/16/9/6/4/3 | 55.3 | 4.0391 | +0.0817 [+0.0780, +0.0856] |
| `scale-ttq` | HEALTHY (109@3364) | +0.0410 [+0.0398, +0.0422] | +0.0018 [+0.0015, +0.0021] | −0.0001 | 0.08–0.14 | 0.04–0.31 | 42/17/9/5/3/2/2 | 62.3 | 3.9839 | +0.0266 [+0.0241, +0.0291] |
| `threshold-03` | HEALTHY (47@226) | +0.0261 [+0.0251, +0.0271] | +0.0004 [+0.0002, +0.0007] | −0.0001 | 0.19–0.24 | 0.06–0.23 | 47/18/8/4/3/2/2 | 86.1 | 4.0185 | +0.0611 [+0.0584, +0.0640] |
| `parcae-entry` (base) | E19 | +0.0332 | +0.0009 | −0.0001 | 0.19–0.24 | 0.08–0.21 | 37/23/7/5/3/2/2 | 73.0 | 3.9573 | — |
| `depthcand-dense-core` (bf16 core) | E20 | +0.1682 | +0.0119 | −0.0018 | 0.75–0.96 | 0.14–0.43 | 63/38/20/12/9/6/5 | 87.7 | 3.9275 | −0.0298 |
| `precision-bf16-all` | NOT RUN (deferred, see Method) | | | | | | | | | |

TTQ learned scales at 5,000 (`step_5000.pt`, γ over the latent's mean|W|): core mean 0.874
over 12 tensors (gate-up 0.50–0.68, down 0.79–1.40); coda 0.71–1.31; the `x0_injects`
projections wander to −8 … +21 (no positivity constraint on γ). Wall 0.88–0.89 h (0.89× base),
peak 10.2 GB, init-probe spread ≤ 0.0002 on every arm (start-independent fixed point). The
trainer's own held-out curve: norm-match +0.21 at 1k, +0.13 at 2k, +0.11 at 4k, +0.08 at the
end against the base.

Predictions: P-str-a TRUE on all three (predicted 80–85 %). P-str-b: norm-match TRUE (60 %),
ttq FALSE (50 %), threshold-03 FALSE (50 %). P-str-c: K1−K6 > 0.08 norm-match TRUE (45 %), ttq
FALSE, threshold-03 FALSE; K3−K6 > 0.005 with CI > 0 norm-match TRUE (40 %), the others FALSE;
> 0.02 FALSE on every arm (15 %); every arm converges by 6 (80 %) TRUE. P-str-d FALSE (50 %):
γ shrank instead. P-str-e: norm-match within ±0.03 of the base FALSE (55 %), threshold-03
FALSE (60 %). P-str-f TRUE on all three.

## Verdict

H-str holds on the norm-matched arm and only there: the scale rule was the limiter. The
absmean scale is what makes the ternary core a weak per-pass map; matching the latent norm
puts a ternary core ABOVE the bf16-core diagnostic on every contribution instrument (K1−K6
0.185 vs 0.168, K3−K6 0.0139 vs 0.0119, MLP branch 0.86–1.10 vs 0.75–0.96). The two controls
say what the lever is not: the dead zone is not it (threshold 0.3 reads as the base), and a
free scale is not it (the optimizer shrinks the learnable scales to 0.87× absmean and the
branch to 0.08–0.14, below the base). The scale has to be imposed.

Binding applied: `norm_match` is the recipe's ternary rule
(`.agents/notes/implemented/architecture/2026-09-09-ternary-core-is-a-weak-per-pass-map.md`);
the paid-TUL re-read is open. The 0.08 higher CE at 5k is a horizon reading, and the ttq
control explains it: at this horizon the loss gradient prefers the weaker map. Ranking the
rule needs the 20k horizon (Wolfe's call).

## Updated hypothesis

The loop's contribution is bounded by the per-pass branch strength, and under ternary the
branch strength is set by the scale rule, not by the code count and not by anything the
optimizer will do on its own. With the branch restored the loop still converges by pass 6
(K3−K6 0.014 against Parcae's K3−K8 0.040); the remaining gap is not a ternary question
(block and stream count, the optimizer, tokens). Next reads, in order: the paid TUL loop
under norm_match; the bf16-everywhere ceiling; the 20k horizon of norm_match against the
existing 20k control.
