# Agent Note: staged latent pretraining of the slot loop (stage 1 latents, stage 2 coda)

Status: proposed

## Problem

Every latent-selected arm (2026-09-30 to 2026-10-02) trains the slot loop toward the next
span's latent beside the coda's token CE. The latent head never learned its target: val
R^2 at or below 0.01 in all six arms, at weight 10 and at weight 1. Two explanations fit:

1. **Competition.** The token CE starves the latent task. Lowering the latent weight from
   10 to 1 improved CE by 0.042 and tripled the loop's K1-K6 (+0.0095 -> +0.0333), so the
   two objectives do interfere.
2. **Blurred means.** The next span on web text is mostly unpredictable from the past, so a
   regression target only admits its conditional mean. The 2026-09-17 code-only arm
   ([`lctul-code-only-ref`](../../../../lab/experiments/failures/2026-09-17-lctul-code-only-ref.md))
   already ran a version of stage 1 below: no coda at train, a training prelude, a frozen
   reference twin, L2 to a frozen VAE code of the next span. After 20k steps: cosine
   0.167, prediction rank 17, the passes added nothing (l6 - l1 +0.003). It found the
   conditional mean in one pass.

Wolfe (2026-10-02): "we have essentially ground truth latents. Why can we just train the
loop and prelude on the slot latents and then, once it isn't terrible, add a coda for
decode? Similar to I-JEPA?" and "Right, we may get blurred means. We should try those
variants you mentioned."

## Proposal

Two stages, with a gate between them. Stage 1 is a measurement first: it tells us whether
any latent target is learnable from the past, and whether the loop's passes add to it,
before any coda is involved.

### Stage 1: latents only (no coda, no token CE)

The prelude and the slot loop train on a latent objective alone. A predictor head reads
the loop's exit (and, for the readings, every pass). What is new against 2026-09-17 is the
target, the loss, the hypothesis count and the verdict instrument.

Ablation grid, one factor at a time from S1-A:

| arm | target | loss | cells | why |
| --- | --- | --- | --- | --- |
| S1-A | frozen plain-5k prelude, pooled over the next span, LayerNormed | L2 | 1 | the blurred-mean baseline on the new target |
| S1-B | as S1-A | InfoNCE over in-batch spans | 1 | ranking, not regression: no reward for hedging to the mean |
| S1-C | as S1-A | relaxed WTA over M cells (multiple-choice learning) | 4 | M hypotheses can each take a mode instead of averaging them |
| S1-D | frozen plain-5k model's FINAL hidden state at the next span's last token | best of A-C | best of A-C | a deeper, more semantic target |
| S1-E | the live EMA prelude twin (today's target) | best of A-C | best of A-C | the collapse control: the EMA twin lost half its rank by step 1500 on 09-22 |

Targets are frozen copies of the plain 5k checkpoint
(`plain-panel-nm-ctrl-s1r/step_5000.pt`), never a live or EMA copy of the model being
trained (S1-E is the control that measures why). S1-D's target encodes the whole past, so
cross-row negatives are trivial to reject from context: S1-D's retrieval readings use
same-row negatives only.

Readings, at every pass t = 1..6 (the loop's per-pass exit through the predictor):

- **Retrieval top-1** of the true next span among negatives, same-row and cross-row. Chance
  is 1 / (number of candidates), a fixed anchor. This is the verdict instrument.
- R^2 against the target (diagnostic; no arm passes or fails on a cosine or an R^2 alone).
- Prediction effective rank (participation ratio) and the target's own rank.
- For S1-C: the best-of-M retrieval and the spread of the M cells.
- tok/s (no coda, so faster: the 09-17 code-only arm ran 1.11x its coda arm).

Length: 3000 steps per arm, seed 1, on the strict slot loop's data and geometry.

### Gate

Stage 2 runs only on a stage-1 arm where BOTH hold:

1. Same-row retrieval top-1 at the exit >= 2x chance, CI above it.
2. Retrieval at pass 6 minus pass 1 > 0, CI above 0 (the loop's passes do latent work).

If no arm passes 1, the targets on offer are not learnable from the past at this scale,
and the staged plan stops there. If an arm passes 1 but not 2, the latent is learnable in
one pass, and stage 2 would not add depth: record it and stop.

### Stage 2: add the coda

Initialise from the gated stage-1 checkpoint (prelude, loop, predictor). Add the coda and
the token CE; the latent term continues as a rank-only head (`tul.fan_lsel_head_input:
detached`, if the 2026-10-02 pair supports it) or at weight 1 (if it does not).

| arm | init | steps | why |
| --- | --- | --- | --- |
| S2-A | stage-1 winner | 5000 - stage-1 steps | the staged recipe at matched total steps |
| S2-B | scratch, same final config | 5000 | the matched-compute control |
| S2-C | stage-1 winner, prelude frozen for the first 1000 steps | as S2-A | the handoff risk: a latent-only prelude is not a token prelude |

Verdict on stage 2: ledger CE on the 480 sweep rows against S2-B and the ungraded fan,
K1-K6, the exploration ledger's selection and search readings, the repetition eval.

## Alternatives considered

- **Keep training jointly and tune the latent weight.** Two points exist (10 and 1). It
  cannot tell competition from blur, because R^2 stays at 0 in both.
- **The self-EMA target (I-JEPA's own recipe).** I-JEPA works on images, where a masked
  patch shares a lot with its context. Here the 09-22 EMA arm lost half the target's rank
  by step 1500 with no token loss to anchor the prelude. Kept only as control S1-E.
- **The VAE code target of 2026-09-17.** Already measured: cosine 0.167 at 20k, one pass.
  Not repeated.
- **NextLat-style transition on the exit.** The 09-25 arm made the exit a constant: the
  stop-grad on the target is no guard when every exit is its successor's source. Not
  used.

## Acceptance criteria

- Stage 1 is built as training keys on `train.py` (Hydra, wandb, the full config logged),
  reusing `tul.code_target_skip_coda` and `tul.code_target_ref` where they fit, with a
  frozen plain-5k target loader that refuses a live or missing checkpoint.
- Every stage-1 arm has a prereg with predictions before it trains, and logs the per-pass
  retrieval readings at val.
- Tests: the target is frozen (no grad, no drift across steps); the predictor sees no
  token of the next span (strict geometry); retrieval chance is computed from the actual
  candidate count; InfoNCE and WTA reduce to hand computations on a tiny batch.
- The gate is applied as written above, and its outcome is filed either way.

Amended 2026-10-02 at the stage-1 build (three design decisions changed; the build is
`tul.latent_pre_*`, `morph/model/tul_latent_pre.py`, `morph/training/latent_pre_ref.py`,
tests `tests/test_tul_latent_pre.py`):

- **New keys, not `code_target_skip_coda` / `code_target_ref`.** Both are wired to the
  VAE-stage encoder E and its projection (`TULCodeProj`, M = prefix_k cells, a frozen
  deep copy of the LIVE model), and their eval forward still runs the coda. Stage 1 needs
  a different target model (the plain run, its own config), a different head, and no coda
  at val either, so it is its own forward (`_forward_latent_pre`) behind
  `tul.latent_pre_target`.
- **The plain-prelude target (S1-A) is span-local.** The frozen plain model's front runs
  on each next span as its own sequence, so slot s's target reads span s+1 and nothing
  else. A causal prelude over the whole token row would carry the prefix the loop already
  reads, and same-row retrieval could then be won by matching that prefix instead of
  predicting span s+1 (this note's S1-D risk, applied to S1-A). S1-D's final-hidden target
  stays causal over the plain token row, as written above.
- **Both plain targets right-pad**, which is exact only while CSA selects every block
  (top_k >= n_blocks; true at the run's shapes: 256 >= 1152 / 8 or 1280 / 8). The code
  refuses any other shape (`assert_padding_invariant`): measured on the tiny fixture at
  top_k 8, eight pad tokens moved earlier positions by 0.44.
- **The two frozen targets are standardised with FIXED statistics** (decided 2026-10-02
  after the first smokes; `tul.latent_pre_target_norm: standard`, the default). Why: the
  LayerNormed plain prelude is about 98 % one common direction. On two val batches its
  mean pairwise cosine is 0.98, its centred participation ratio 3.5 to 5.3, its
  per-coordinate variance about 0.02, with four channels at |mean| about 11. An L2 on it
  mostly rewards predicting a constant. The target is now `(z_ln - mu) / sigma` per
  coordinate. `mu` and `sigma` are computed ONCE at build from the frozen model, over the
  real span slots of 64 TRAINING batches from document 40 000. That is past the run's own
  documents (refused if the run's estimated document count reaches it) and before val's
  50 000 (refused if the draw reads into it). The batches come from a separate loader in a
  separate process, so the run's training stream is untouched (pinned by a test). `sigma`
  is floored at 1e-3 x its median. The statistics are not saved; a resume recomputes them
  bit-identically. Standardised, the mean predictor's L2 is about 1.0, so L2 is about
  1 - R^2. InfoNCE takes cosines of the standardised vectors.
- **The live EMA control keeps the LayerNorm only** (`latent_pre_target_norm: ln`;
  `standard` is refused with it). Its target drifts with the live model, so statistics
  fixed at build go stale, and per-batch statistics would standardise away the very rank
  collapse that arm exists to measure.

## Risks

- **The target is unpredictable.** If S1-A to S1-D all sit near chance, the "ground-truth
  latent" carries little that the past determines at 300M / 3k steps. That is a real
  answer, not a failure of the build.
- **Retrieval by shortcut.** Cross-row negatives can be rejected from topic alone. The
  same-row reading is the one that counts.
- **S1-D leaks the past into the target.** Its target state has read the whole prefix, so
  it shares content with the loop's input. Same-row negatives share the prefix too, which
  limits the shortcut, but its numbers are not comparable to S1-A's.
- **The handoff.** A prelude trained only on latents may not serve tokens. S2-C measures it;
  the 2026-09-13 bootstrap arm found reader maturity was not the confounder in the other
  direction.
- **Training-time savings are small.** No coda saves the coda's share of a step in stage 1
  only. Stage 2 costs the same as any coda arm.
