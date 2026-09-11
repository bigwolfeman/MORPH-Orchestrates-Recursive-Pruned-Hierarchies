# Planned: the cross-span information budget — how many nats live across a span boundary?

Status: failure

Date: 2026-09-11 (frozen before launch). Arc: `2026-09-04-loop-contribution-arc.md`.
Design note:
[`2026-09-11-cross-span-budget.md`](../../../.agents/notes/proposed/architecture/2026-09-11-cross-span-budget.md).
Evidence it is built on: the nine slot-loop arms filed on 2026-09-10/11 —
[`slot-mnext-parcae-core`](../successes/2026-09-10-arc-slot-mnext-parcae-core.md),
[`slot-mux-mask-norm-match`](../failures/2026-09-10-arc-slot-mux-mask-norm-match.md),
[`slot-loop-mask-norm-match`](../failures/2026-09-10-arc-slot-loop-mask-norm-match.md),
[`slot-mux-prefix4-norm-match`](../successes/2026-09-10-arc-slot-mux-prefix4-norm-match.md),
[`slot-mnext-staged-20k`](../successes/2026-09-10-arc-slot-mnext-staged-20k.md) —
and the [slot-geometry audit](../results/2026-09-10-slot-geometry-audit/README.md).

## Question

Nine arms have asked whether the slot loop can be made to earn. Not one has asked how
much there is to earn. Token K1−K6 stayed inside [−0.0001, +0.0033] across the loss, the
reader, the geometry, the input, the write width, the core and a stability term. If web
text only puts a few hundredths of a nat across a span boundary, every one of those flat
readings is the correct answer to the question the arm asked, and no core, seed, width or
attention work can move it.

The budget is a property of the DATA and the JOB. Measure the budget.

**How many nats of next-token prediction depend on tokens in an earlier span, and how are
those nats distributed by offset after a span start?**

## Hypothesis

Two plain models that differ in exactly one thing — whether any information can cross a
span boundary — bound the quantity. Their paired CE gap IS the budget, and its profile by
offset is the shape a slot could ever fill.

Two independent estimates of the size, and they disagree, which is why the run is worth
its 2 hours.

**Route 1, this tree's own ablations, says SMALL (0.10 to 0.18 nats).**
`slot-mux-mask-norm-match` is a model trained with `tg_restrict` (same-span-or-slot), so
its slot cells are the ONLY cross-span channel it has, and its prelude is masked too so a
cell sees only its own span. Two measured numbers bracket the whole cross-span value of
that model: it sits **0.090** nats behind the unmasked ruler on the 480-row sweep, and
zeroing its slot states at eval costs a further **0.093** (its `worth_profile` zero mode,
token-weighted over the seven distance bins: 0.554 / 0.182 / 0.128 / 0.109 / 0.083 /
0.057 / 0.042 at offsets 0 / 1 / 2 / 3 / 4-7 / 8-15 / 16+). So an unmasked model beats a
no-cross-span model by about **0.183** nats *at eval, on weights trained with the routes*.
A model TRAINED without them adapts, so the honest budget is below that, and above the
mask's own 0.090 price.

**Route 2, ordinary context scaling, says LARGE (0.3 to 0.7 nats).** The restricted model
carries a mean of about 10 tokens of context (mean span ~20 on OpenWebText under this
rule, 51 spans per 1024-token row). The unrestricted one carries ~512. Position-in-context
loss curves for small LMs put that distance at several tenths of a nat.

The gap between the two routes is the interesting part: Route 1 says a 128-cell slot
channel already carried nearly all of it, Route 2 says the routes carry far more than any
ablation on this tree has priced. **Both cannot be right, and the pair decides it.**

Secondary question with no prior at all: does a model that cannot see past its span use
its looped core MORE? A plain MORPH core under the prelude entry reads token K1−K6 0.033
under `norm_match`. A plain PARCAE core's token loop has never been measured at any depth
draw — the only Parcae-core number in the tree is the SLOT loop's +0.0001, which is a
different mechanism. Both arms carry a K-curve and it is recorded as a reading, not a bar.

## Method

Two arms, `budget-web-full` and `budget-web-span`, composing `budget_root.yaml`, which
composes `notul_panel_norm_match` — the plain ruler the nine slot arms are read against
(seq 1024, batch 6, seed 1, 5,000 steps, ramp 1,000, retention off, cap 0, alpha_cap 3.5,
`ademamix_t_beta3` 3500, `norm_match` ternary, `core_fixed_point_lambda` 1.0, prune /
carve / route off). A Hydra compose of the two arms differs in `model.span_mask` and
`wandb.name` and **nothing else** (printed below).

### The one knob

`model.span_mask` — `"row"` (arm `budget-web-full`) gives every row ONE span, so every
relation degenerates to the unrestricted one; `"span"` (arm `budget-web-span`) cuts the
row with the ONE TUL `BoundaryRule` (`.;!?` + newline + dashes, no comma, min_span 4,
span_cap 32 — `morph/model/tul_layout.py`, the same rule the loader and the generator
use, resolved from the same tokenizer: |B| = 1,101 of 49,169 ids). The two values build
the IDENTICAL model and run the IDENTICAL code; only the span-id tensor differs, so the
gap cannot carry an operator change.

### Every cross-position route, and its verdict

| route | verdict |
|---|---|
| window attention branch (prelude, coda) | **masked** — `tg_allow` = `span_allow_mask`, causal AND same-span |
| compressed attention branch (prelude, coda) | **masked** — the branch becomes the DENSE per-position form (the build `tul.tg_restrict` already had: no pooled compressor, no indexer) and takes the same relation through the new `tg_comp_allow` |
| core attention | **masked** — `ParcaeDenseAttention` takes the relation as its `attn_mask` in place of `is_causal` |
| CCA causal conv, both stages | **masked** — `segment_causal_conv(seg=span_id)`, already in the tree for the coda under `tg_restrict_scope: coda` |
| CCA value shift (`W_v_prev`) | **masked** — the same `seg` zeroes the shift at a span's first position |
| hash-bigram embedding | **masked** — a span's first position takes the "no previous token" key (prev = 0), the key the row's first position already takes. This route is the one a slot's `boundary` seed carries, and it is invisible to an embedding Jacobian |
| GLA retention, `retention_carry` | **off** — `retention: false` composes, and the build RAISES on `retention: true` |
| TST input bagging | **off** — `tst_bag_size 0` composes, and the forward RAISES on `bag_size > 0` |
| MTP heads | **off** — `mtp_heads 1`, and the build RAISES otherwise |
| `DiagonalInjection` (the widened carry) | **per-position** — elementwise in `h` and `e`; the learned `B` is a channel map |
| `x0_injects`, value embeds, `_build_injection_term` | **per-position** — a Linear on that position's own signal |
| Hyper-Connection residual (Cayley, n=4) | **per-position** — mixes the 4 streams at one position; `hyper_connections.py` carries no sequence op |
| CCA channel compression, RMSNorm, SwiGLU/MORTAR, the router, the LM head, the CE | **per-position** |
| CoPE / RoPE | **positional, not content** — both arms see the same absolute index |
| static CUDA graphs | **refused** — a captured FRONT region would replay every batch under the capture batch's mask; the build RAISES |

### The gate, run before the arms were queued

`tests/test_span_mask_leak.py`, two-sided everywhere: the restricted model must show
EXACTLY zero cross-span influence and the `"row"` control, run through the same code, must
show a nonzero one. Three instruments — an end-to-end id perturbation (edit one token of
span 0; every logit in a later span must be bit-identical), a per-region autograd Jacobian
through the embedding read at the prelude exit, the core exit, the coda exit and the
logits, and an id perturbation on the bigram alone. Removing each of the five span routes
in turn makes it fail: 5 of 5 sabotages caught, all by the end-to-end instrument.

### Deviations from the design, recorded

1. **Both arms run the dense per-position compressed branch.** A pooled block of 16 or 256
   positions straddles several spans at a mean span of ~20, so there is no block-level
   same-span restriction that leaves the branch alive. Consequence: neither arm's absolute
   CE is comparable to `plain-panel-norm-match`, and the pair's only product is the paired
   gap.
2. **`core_impl: parcae` is required, not chosen.** Same reason, for the core. The build
   RAISES on `core_impl: morph` rather than masking part of the model. The arms carry
   Parcae's own A/B (`injection_channels: all`, `injection_all_decay` 0.447,
   `injection_all_dt` 0.8, `injection_B: true`), the four lines
   `slot-mnext-parcae-core` carries and which are measured INERT on loop depth by
   themselves.
3. **`use_kernels: false`** (the build RAISES otherwise), so the CE is the full-logits
   path, not the chunked one, in both arms.
4. **The span ids are computed in the model, not in the loader** (`_span_context`, one
   `BoundaryRule.cut` per forward on the host). Every entry point — training, eval,
   `core_depth_sweep.py`, any probe — is therefore masked by construction.
5. **No CODE pair is built.** The corpus is present (`data/pretok/code`, 366,943,467
   tokens of `code_search_net`, same `bigcode/starcoder2-7b` tokenizer, `role:
   pretrain_bulk`, reachable through `curriculum.blend: {code: 1.0}`), but the READOUT is
   not: `core_depth_sweep.py` builds its val rows from `data.dataset` (the OpenWebText
   arrow glob) and knows nothing about pretok shards, and no code `eval_holdout` shard
   exists, so a code pair would train on code and be scored on web text. What it needs:
   a ~30-line holdout shard (a tail-doc slice of `doc_offsets`/`doc_lens` with `split:
   eval_holdout`, symlinking `tokens.u16.bin`) plus a `core_depth_sweep.py` that can read
   a pretok shard as its val stream. No download. Wolfe's call.

### Cost, measured before launch, not predicted

A 30-step GPU probe of both configs at this change's working tree (2026-09-11 13:05-13:07,
exit 0 both):
260.5 M params, 64.50 M of them the Parcae core; **tok/s at step 20 = 10,862 (full) /
11,644 (span)**; peak 19.06 GB. 5,000 steps x 6,144 tokens at ~11,000 tok/s is ~47 min per
arm, plus two 480-row sweeps and the `plain` kind's `core_anatomy` / `core_init_probe`.
A direct check at init on 3 val batches, both configs at seed 1: CE 11.164159 (full)
against 11.198573 (span), so the mask is live at EVAL and not only in training.

### Readout

Runner `arc/run_slotloop3.sh`, KIND `plain` (depth sweep at 0,1,2,3,6,9,12,16 on 480 rows
at checkpoints 2,500 and 5,000; `core_anatomy.py` and `core_init_probe.py` at 5,000), a
12-step smoke first, the draw under the sustained tripwire.

Then `lab/divergence/span_budget_profile.py --full <full.tokens.npz> --span
<span.tokens.npz> --config budget_web_full --depths 6,1`: pairs the two arms on
`tok_index` with `numpy.intersect1d`, recovers each token's offset inside its span by
rebuilding the sweep's own val rows and running the same rule, and prints the gap overall
and by offset 0..7 and 8+ with a 400-draw block bootstrap over 1024-token blocks. **The
offset is that of the PREDICTED token inside its span**, so offset 0 is a span's first
token, predicted from the previous span's last token — the position the worth profile
reads and where a slot would earn its living. Self-checked before launch: an arm against
itself reads +0.0000 at every offset, and `norm-match-20k` against `slot-mnext-staged-20k`
at step 5,000 reads **+0.1919** overall at depth 6 [+0.1859, +0.1983] and +0.0574 at depth
1, reproducing the recorded number.

## Predictions (frozen)

All at step 5,000, depth 6, paired on `tok_index` over the sweep's 480 rows, gap =
CE(`budget-web-span`) − CE(`budget-web-full`).

- **P-a (survival).** Both arms HEALTHY to 5,000, no sustained tripwire (`preclip/total >
  1e4` at step >= 200): **85 %**. Reasoning: the panel recipe carries the 1,000-step ramp
  and the fixed-point term, which have held every arm drawn since 2026-09-04, and the
  Parcae core read gain 0.22 with the hinge never binding on the slot loop. The 15 % is
  that a plain Parcae core over 1,024 positions has never been trained on this tree, and
  the widened carry detonated 2 of 2 draws under `warmup: 0` in E9.
- **P-b (the budget exists).** Overall gap above **0.05** nats, CI excluding 0: **95 %**.
  Reasoning: the tg_restrict mask alone already costs 0.090 on the slot family, and this
  mask is strictly tighter — it removes the slot channel as well.
- **P-c (the budget's size — the headline).** Overall gap inside **[0.08, 0.30]**: **70 %**.
  Point estimate **0.15**. Above **0.30** (Route 2 is right and this tree's ablations
  under-price the routes): **18 %**. Below **0.08** (the restricted model adapts almost
  completely): **12 %**. Reasoning: Route 1's 0.183 is an eval-time ablation of weights
  trained WITH the routes, which is an upper bound on what retraining loses, and the
  mask's own 0.090 is a lower bound on this stricter cut. The 18 % on Route 2 is real
  weight: every number in Route 1 comes from a model carrying 128 slot cells at 1,152
  positions, and a plain 1,024-position model has more context to lose.
- **P-d (the shape — front-loaded).** The offset-0 gap is at least **3x** the offset-8+
  gap: **75 %**. Reasoning: the worth profile is front-loaded on every arm measured
  (0.554 / 0.042 masked, 0.093 / 0.013 unmasked — 13x and 7x), and offset 0 is the only
  position with literally no same-span history. The 25 % is that a slot ablation and a
  full cut are different operations: a full cut also removes the long-range topical
  context that helps at EVERY offset, which flattens the curve.
- **P-e (offset 0).** The offset-0 gap above **0.35** nats: **60 %**. Reasoning: the masked
  arm's own slot ablation reads 0.554 there and the unmasked arm's 0.093; a full cut sits
  between them at worst and above the higher at best, because the mask arm's first token
  still sees its own span's... nothing (it IS the span's first token) but does see every
  earlier slot.
- **P-f (offset 8+).** The offset-8+ gap below **0.12** nats: **70 %**. Reasoning: a token
  eight or more into its span has 8-31 tokens of same-span context, and the masked arm's
  slot is worth only 0.042-0.057 there.
- **P-g (depth 1 vs depth 6).** The gap at depth 1 is SMALLER than at depth 6: **55 %**.
  Reasoning: a shallower model extracts less from any context, so it has less to lose —
  that is what `norm-match-20k` vs `slot-mnext-staged-20k` shows (+0.0574 at depth 1
  against +0.1919 at depth 6). Only 55 % because that pair differs by a mechanism, not by
  context.
- **P-h (the loop, `budget-web-full`).** Token K1−K6 inside **[0.00, 0.15]**: **80 %**.
  **No prior exists**: 0.033 is the plain MORPH core under the prelude entry, 0.185 is the
  same under a noise entry, and the only Parcae-core reading in the tree (+0.0001) is the
  slot loop, a different mechanism. Recorded as a reading, not a bar.
- **P-i (the loop, does the cut make the loop matter?).** `budget-web-span`'s token K1−K6
  strictly above `budget-web-full`'s, CIs disjoint: **35 %**. Reasoning: two opposite
  arguments of similar weight. For: with no long-range attention to lean on, iterative
  refinement of a short span is the only depth left. Against: there is less to compute
  about 10 tokens than about 512, and E6 already showed K-diffs measure depth DEPENDENCE
  rather than depth value. 35 % rather than 50 % because nothing on this tree has moved a
  K-curve except the core's per-pass strength.
- **P-j (cost).** Both arms finish 5,000 steps inside **55 min** of wall clock each: **80 %**.
  Reasoning: measured 10,862 / 11,644 tok/s at step 20 of a 30-step probe; 5,000 x 6,144 /
  11,000 = 47 min. The 20 % is the caching allocator at 19.06 GB peak and the eval cadence.

## Binding

- **P-c TRUE (gap 0.08-0.30).** The budget is small and front-loaded, and the arc's nine
  flat K-curves are the correct answer for web text at this scale. The slot's job on this
  data is worth about a tenth of a nat, which is the same order as the CE noise between
  seeds, so no slot mechanism on web text can be scored at 5k. The next move is NOT
  another slot arm: it is a corpus whose spans depend on each other, and the code pair in
  "not built" is the cheapest first test of that (the boundary rule cuts code twice as
  often: 107 spans per 1024 tokens against 51). Report the offset profile as the ceiling
  any future slot arm is measured against.
- **P-c FALSE upward (gap above 0.30).** The information IS there and nine arms failed to
  route it, and this tree's slot ablations under-price cross-span routes by 2x or more.
  The lane becomes the reader and the target, not the core: `slot_z_optimize` already says
  a gradient-fitted z is worth 0.9-2.6 nats to the frozen coda while the loop's exit
  equals its entry to 0.0015. The first follow-up is the budget profile against the worth
  profile, offset by offset, on the same rows — the difference between the two curves is
  exactly what the slot channel fails to carry.
- **P-c FALSE downward (gap below 0.08).** Web text at 1,024 tokens, cut this way, is
  nearly span-local. That closes the slot loop on this corpus outright and makes every
  cross-span mechanism in the arc unmeasurable here. Same next move as the first branch,
  with more urgency, and the boundary RULE goes on the table too (a rule that cuts at
  ~20 tokens may be cutting inside the unit the text actually depends on).
- **P-i TRUE.** A context-starved model uses its loop more. That is the first positive
  depth signal in the arc and it says the loop's job is local computation, not memory —
  which is a different research program from the slot loop and is followed by a matched-
  compute shallow control at the restricted geometry before anything else.
- **P-a FALSE (detonation).** Read `loop/core_gain_t0` and the fixed-point term first. A
  detonation on the `row` arm is a plain-Parcae-core-at-1024 fact and belongs in
  `lab/divergence/DIVERGENCE-README.md`, not in this arc; a detonation on the `span` arm
  alone is a statement about the restricted geometry's conditioning and the pair is
  re-drawn with both arms at a lower `ademamix_alpha_cap`.

### Method amendment 1 (2026-09-11 13:20 CDT; two checks closed AFTER launch, no prediction touched)

Both were on the "not verified" list below and are now measured. Neither changes the
arms, which are pinned at `cc4e034` and were already running.

1. **The depth sweep's build.** `lab/divergence/_build.py::build_model` +
   `DepthLever` on both configs, on CPU, at depths 0, 1 and 6: both build (`tul_rt is
   None`, so `core_depth_sweep.py` takes its `plain` path) and run a forward at every
   depth. The readout can therefore reach the checkpoints.
2. **The TRAINING path's mask.** `tests/test_span_mask_leak.py` in `cc4e034` runs three
   instruments, all in `eval()`, where the core draws one uniform depth and never
   reorders the batch. Training sorts by per-sample depth and slices the active prefix,
   so the span relation must follow the same permutation. Covered in `b258d1f` by a
   fourth instrument in `train()` mode: it holds, and it exposed two ways the older
   assertions could have passed vacuously (a global no-grad prefix at `bptt_depth <
   max_depth` — the E7 defect, which the budget arms do not meet at 8 and 8 — and a
   tensor view the cross-row check zeroed before the cross-span check read it). No model
   code changed.

## Not verified before launch

- The 5,000-step conjunction itself. The longest either config has run is 30 steps.
- That the two arms' val streams are identical token-for-token. They share a config
  lineage, a seed and a loader, and the 30-step probe's `[VAL 25]` line printed the same
  value for both, but no token-level comparison was made.
- The `plain` KIND's `core_anatomy.py` / `core_init_probe.py` on a `span_mask` model. Both
  call `model(...)` and neither touches `compressor` / `indexer` / `prelude_states` (read,
  not run), so neither is expected to fail, but neither has been run on this build. A
  failure there costs nothing: it runs after the sweep.
- The depth sweep on a `span_mask` model. `core_depth_sweep.py` builds the model through
  `build_cfg` + `build_morph_config`, which resolves `span_rule`, and forces depth through
  `DepthLever` — read, not run. If the sweep cannot build, there is no gap and the arms
  are re-scored by hand from the checkpoints.
- Any number on CODE. See deviation 5.
- Whether the offset profile is stable across the 2,500 and 5,000 checkpoints. Only the
  5,000 pair is predicted.

## Results

Filed 2026-09-11 14:53 CDT. Both arms at `cc4e034`, seed 1, 5,000 steps, Parcae core,
`span_mask: row` (full) against `span_mask: span`. Raw files in
`lab/experiments/results/2026-09-11-span-budget/` (sweep, anatomy and init-probe JSON, the
profile output at 2,500 and 5,000); npz in `ignored/experiment-artifacts/2026-09-11-span-budget/`.

**Survival and cost.** Both HEALTHY to 4,999: full pre-clip max 21.3 at step 261, span 14.6
at step 613 (bar 1e4). Wall 43m32s (full) and 43m02s (span) from START to DONE. Runner
final val: full 4.0685, span 4.4630.

**The budget, paired on `tok_index` over 491,520 tokens, gap = span − full, step 5,000.**

| depth | CE full | CE span | gap | 95 % CI |
|---|---|---|---|---|
| 6 | 4.0394 | 4.4389 | **+0.3994** | [+0.3838, +0.4162] |
| 1 | 4.0673 | 4.4689 | +0.4016 | [+0.3863, +0.4187] |

By offset of the PREDICTED token in its span (offset 0 = the span's first token, predicted
from the previous span's last token; offset 1 = predicted from the span's first token):

| offset | n | gap @6 | 95 % CI | gap @1 |
|---|---|---|---|---|
| 0 | 24,338 | +0.2826 | [+0.2622, +0.3025] | +0.2886 |
| 1 | 24,584 | +0.9580 | [+0.9257, +0.9921] | +0.9357 |
| 2 | 24,546 | +0.6899 | [+0.6437, +0.7476] | +0.6741 |
| 3 | 24,504 | +0.5910 | [+0.5581, +0.6278] | +0.5815 |
| 4 | 23,343 | +0.5082 | [+0.4834, +0.5364] | +0.5094 |
| 5 | 22,646 | +0.4479 | [+0.4254, +0.4728] | +0.4466 |
| 6 | 21,799 | +0.4021 | [+0.3772, +0.4263] | +0.4048 |
| 7 | 21,034 | +0.3672 | [+0.3479, +0.3890] | +0.3663 |
| 8+ | 304,246 | +0.3149 | [+0.3025, +0.3314] | +0.3217 |

**The budget grows with training.** The same pair at step 2,500: gap +0.1633 [+0.1562,
+0.1702]; offsets 0..7 = 0.156 / 0.667 / 0.373 / 0.272 / 0.219 / 0.182 / 0.149 / 0.143, offset
8+ = 0.094 [0.088, 0.100]. From 2,500 to 5,000 the offset-8+ gap tripled (0.094 → 0.315)
while the offset-1 gap rose 0.667 → 0.958. The unrestricted model learns to use far context
faster than the restricted model learns from its span.

**The loop (token K-curves, 480 rows, 2,000-draw bootstrap).**

| arm | d0 | d1 | d2 | d3 | d6 | d9 | d12 | d16 | K1−K6 | K3−K6 |
|---|---|---|---|---|---|---|---|---|---|---|
| full | 4.9388 | 4.0673 | 4.0445 | 4.0396 | 4.0394 | 4.0406 | 4.0412 | 4.0414 | +0.0279 [+0.0269, +0.0290] | +0.0002 [−0.0001, +0.0005] |
| span | 5.7699 | 4.4689 | 4.4450 | 4.4395 | 4.4389 | 4.4405 | 4.4414 | 4.4419 | +0.0301 [+0.0288, +0.0313] | +0.0007 [+0.0002, +0.0010] |

Anatomy at 5,000, movement per pass: full [0.256, 0.091, 0.044, 0.025, 0.016, 0.010, 0.007],
span [0.206, 0.071, 0.036, 0.022, 0.014, 0.010, 0.007]. Init probe: on both arms the prelude,
RMS-noise, zero and small-noise entries agree to 0.0002 nats by depth 4 (full 3.9646 /
3.9656 / 3.9655; span 4.3510 / 4.3523 / 4.3522): a start-independent fixed point on a plain
token loop.

**Against the slot family's worth profile (cross-model, offsets aligned).** The worth
profile bins by the offset of the PREDICTING position; this profile by the predicted
token, so worth bin k lines up with budget offset k+1. `slot-mux-mask-norm-match` at 5,000,
zero-ablation of the prefix write, worth bins 0 / 1 / 2 / 3 / 4-7 / 8-15 / 16+ =
0.554 / 0.182 / 0.128 / 0.109 / 0.083 / 0.057 / 0.042 (192 rows), token-weighted total
0.093. The budget at the same positions is 0.958 / 0.690 / 0.591 / 0.508 / ~0.42 / ~0.31 /
~0.31, total 0.399. The prefix write is worth about a quarter of the budget at every
position, a little more than half at the span's first position. This is a comparison
between different models on different rows and is a reading, not a controlled result.

**Predictions.** P-a TRUE (both healthy). P-b TRUE (0.399, CI excludes 0). **P-c FALSE
upward** (0.399 > 0.30; the 18 % branch). P-d FALSE (offset 0 is 0.9x offset 8+, not 3x).
P-e FALSE (offset 0 reads 0.283 < 0.35). P-f FALSE (offset 8+ reads 0.315 > 0.12). P-g
FALSE (depth 1 gap 0.4016 is not smaller than depth 6's 0.3994). P-h TRUE (full K1−K6
0.0279 in [0, 0.15]). P-i FALSE (span 0.0301 [0.0288, 0.0313] against full 0.0279 [0.0269,
0.0290]: the intervals overlap by 0.0002). P-j TRUE (43 min each). 4 of 10 held; the
headline and every shape prediction missed.

## Verdict

Failure by the protocol: the headline P-c missed upward and all four shape predictions
(P-d to P-g) missed. The measurement itself is clean and is the arc's first ceiling.

What the numbers say, in the prereg's own binding (P-c FALSE upward): the information IS
there, 0.40 nats of it at 5k on web text at 1,024 tokens, and this tree's slot ablations
price it at 0.09 because they ablate one route (the prefix write) of at least two. The
shape prediction was wrong in an instructive way. The budget is NOT front-loaded at the
span's first token. It has two parts: a long-range part worth 0.31 nats at EVERY position
eight or more tokens into a span, and a short-context spike (0.96 at the span's first
position, decaying over about seven tokens) that is the cost of predicting from one to
seven tokens of same-span history. The worth profile's front-loaded decay is the shape of
what the prefix write CARRIES, not the shape of what is there to carry.

The loop is unchanged by the cut: K1−K6 0.028 → 0.030, K3−K6 0.0002 → 0.0007, the same
contraction series, the same fixed point from every entry. A context-starved model does
not loop more. The loop's job on tokens is local computation done by pass 3, on a
restricted geometry as much as on a full one.

Two readings NOT to make: (1) that 0.40 is the corpus's budget. It is the 5k budget on
this recipe and it is still rising (0.16 at 2,500); the 20k number is unmeasured. (2) that
the dense per-position compressed branch is free in general: the full arm's 4.0394 sits
0.0004 from the MORPH plain ruler's 4.0390 at the same step, but those arms differ in
recipe (Parcae core and entry, seed) as well as in the branch.

## Updated hypothesis

The slot mechanism's ceiling on web text is 0.40 nats at 5k (rising with the horizon),
with a flat 0.31-nat long-range component and a 0.96-nat spike at the span's first
position. The slot loop's prefix write recovers about a quarter of it. The remaining
routes under `tg_restrict` ("same span OR any slot position") are the slot CELL positions
themselves, whose coda states are made by the prelude over their own span and are not
zeroed by the worth profile's ablation. Next, in order:

1. Split the budget between routes on the mask arm: a worth profile that zeroes the slot
   cells' coda states as well as the prefix write, on THESE 480 rows, so the three curves
   (budget, prefix-write worth, all-slot worth) sit on the same tokens. The difference
   between the budget and the all-slot worth is what the mask arm fails to route at all;
   the difference between all-slot and prefix-write is what the prelude carries through
   the cell without the loop.
2. Re-run the pair at 20k (Wolfe's word needed) only if the route split says the loop's
   share matters. The budget's growth from 2,500 to 5,000 says the 20k ceiling is higher,
   which raises the stakes of the mechanism and does not change what it must do.
3. Step 2 of the 2026-09-11 proposal (decode-cheaply at matched decode cost) now has a
   ceiling to be scored against: a slot that carries the 0.31-nat long-range part at every
   position is worth more than one that only covers the spike.
