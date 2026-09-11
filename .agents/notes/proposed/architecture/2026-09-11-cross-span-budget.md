# Agent Note: measure the cross-span information budget with no slot cells

Status: proposed

## Problem

Every arm of the loop-contribution arc has asked the same question from the supply side:
can the slot loop be made to earn depth? Nine arms on 2026-09-10/11 varied the loss
(progressive, per-pass LoRA, MUX-every-pass, staged, staged-all), the reader
(`mux_readout full`), the geometry (`tg_restrict`), the input (`bag_mean`), the write
width (`prefix_k 4`), the core (`core_impl: parcae`) and a stability term
(`core_fixed_point_lambda` off). Token K1−K6 stayed inside [−0.0001, +0.0033] on every
one of them. The slot's measured worth is small and front-loaded: zeroing the slot state
costs 0.09 nats at the first token after a boundary unmasked and 0.17 masked, decaying to
about 0.01 by offset 6.

None of that says how much there is to earn. The arc has been optimising a channel
without ever measuring the quantity the channel could carry. If web text only puts a few
hundredths of a nat across a span boundary, then every flat K-curve in the table is the
correct answer to the question the arm asked, and no amount of core, seed, width,
attention or geometry work can move it.

The quantity is a property of the DATA and the JOB, not of MORPH: how many nats of
next-token prediction depend on tokens in an earlier span?

## Proposal

Measure it directly, with no slot cells at all. Two plain models that differ in one thing
— whether any information can cross a span boundary — trained on the same recipe, scored
on the same val tokens. Their paired CE gap IS the budget; its profile by offset after a
span start is the shape a slot could ever fill. Arms `budget-web-full` /
`budget-web-span`, prereg
[`lab/experiments/planned/2026-09-11-arc-span-budget.md`](../../../../lab/experiments/planned/2026-09-11-arc-span-budget.md).

**The code is one knob, `model.span_mask`, with three values.** `"off"` is the tree as it
was. `"row"` and `"span"` build the IDENTICAL model and differ only in the span ids
computed per batch: `"row"` gives every row one span, so every relation below degenerates
to the unrestricted one; `"span"` cuts the row with the ONE TUL `BoundaryRule`
(`morph/model/tul_layout.py` — `.;!?` + newline + dashes, no comma, min_span 4, span_cap
32), the same rule the loader and the generator use.

The span ids are computed inside `MORPHTransformer._span_context`, once per forward, from
the token ids, and threaded into the prelude, the core and the coda the way the TG masks
already are. In the model and not in the loader, so that every entry point — training,
eval, `lab/divergence/core_depth_sweep.py`, any future probe — is masked by construction
rather than by remembering to pass a mask.

Cross-position routes, each named with what closes it:

| route | verdict |
|---|---|
| window attention branch (prelude, coda) | masked — `tg_allow` = `span_allow_mask` |
| compressed attention branch (prelude, coda) | masked — the branch becomes the DENSE per-position form (the build `tul.tg_restrict` already had) and takes the same relation through the new `tg_comp_allow` |
| core attention | masked — `ParcaeDenseAttention` takes the relation as its `attn_mask` in place of `is_causal` |
| CCA causal conv, both stages | reset — `segment_causal_conv(seg=span_id)`, already in the tree |
| CCA value shift (`W_v_prev`) | reset — the same `seg` |
| hash-bigram embedding | cut — a span's first position takes the "no previous token" key (prev = 0), the same key the row's first position already takes |
| GLA retention / `retention_carry` | off — the build RAISES on `retention: true` |
| TST input bagging | off — the forward RAISES on `bag_size > 0` |
| MTP heads | off — the build RAISES on `mtp_heads > 1` |
| `DiagonalInjection` (the carry) | per-position — elementwise in `h` and `e` |
| `x0_injects` / value embeds / `_build_injection_term` | per-position — a Linear on the position's own signal |
| Hyper-Connection residual (Cayley, n=4) | per-position — mixes streams at one position; no sequence op in `hyper_connections.py` |
| CCA channel compression, RMSNorm, SwiGLU, MORTAR, the router, the LM head, CE | per-position |
| CoPE / RoPE | positional, not content — both arms see the same absolute index, and an index is not an input the Jacobian can carry |
| static CUDA graphs | refused — a captured FRONT region would replay every batch under the capture batch's mask |

**Both arms run the restricted CODE.** The compressed branch becomes dense per-position
in both, so the pair differs by the mask CONTENT and not by an operator. The price is
that neither arm's absolute CE is comparable to `plain-panel-norm-match`; the only number
the pair exists to produce is the paired gap.

**The Parcae core is required**, not chosen for speed. A MORPH core's compressed branch
attends POOLED BLOCKS of positions, and a 16- or 256-position block straddles several
spans at a mean span of ~20 tokens, so there is no block-level same-span restriction that
leaves the branch alive — masking it would re-create finding F1 (a dead branch the gate
still spends half its mixture on). The build RAISES on `core_impl: morph` rather than
masking part of the model.

**The gate is `tests/test_span_mask_leak.py`**, and it is two-sided everywhere: the
restricted model must show EXACTLY zero cross-span influence and the `"row"` control, run
through the same code, must show a nonzero one. Its strongest instrument is an id
perturbation — edit one token of span 0, and every logit in a later span must be
bit-identical — which covers every route at once. Verified by removing each of the five
span routes in turn: 5 of 5 sabotages are caught by that one test.

The readout is `lab/divergence/span_budget_profile.py`: it pairs the two arms' per-token
sweep files on `tok_index`, recovers each token's offset inside its span by rebuilding
the sweep's val rows and running the same rule, and prints the gap overall and by offset
0..7 and 8+ with a 400-draw block bootstrap.

## Alternatives considered

**Drive `tul.tg_restrict` with zero slot cells instead of a new knob.** This was the first
choice and it does not reach. `tg_restrict`'s relation is "same span OR any slot", which
is a cross-span channel by design — it is the arc's mask arm, not a cut. Its masks are
built from a `SlotLayout`; a `tg_restrict` model RAISES in `forward()` without one, and
RAISES at build under `core_impl: parcae`. A layout with zero slots would also change the
row packing and the token count per row. `span_mask` reuses everything `tg_restrict`
built — the dense compressed branch, `tg_allow`, `tg_seg`, the active-set permutation in
`_core_region` — and supplies a different relation from a different source. The two are
mutually exclusive at build.

**Cut the rows instead of masking them.** Training on rows of one span each is a
perfectly airtight restriction and needs no mask at all. Rejected: ≤32-token rows change
the batch shape, the position statistics, the padding and the optimiser's token count all
at once, so the pair would differ by much more than the cut.

**Leave the hash bigram alone.** It reads one token back, which crosses a boundary only at
a span's first position. Rejected: that ONE position is precisely where the budget is
concentrated (the worth profile's offset 0), and the token it reads is the previous span's
boundary token — exactly the datum `slot_seed: boundary` feeds a slot. Leaving it would
measure a leak and call it a budget. The end-to-end leak test at the `boundary_token` edit
site exists for this route alone.

**Mask a MORPH core's pooled blocks instead of swapping the core.** A query could be
allowed a pooled block only when every position in the block shares its span. At block 16
or 256 against a mean span of 20 that is almost never true, so the compressed branch would
be near-dead and the gate would spend half its mixture on a zero tensor — finding F1,
re-created deliberately. Rejected; the build refuses it.

**Keep the arms on the MORPH core and accept a partly masked model.** Rejected outright:
a budget measured through a leak is worth nothing, and "the compressed branch still sees
other spans" is a leak.

**Use the `"off"` model as the unrestricted arm.** It is the real plain model and its CE
is comparable to every other arm in the arc. Rejected: it runs a pooled compressed branch
where the restricted arm runs a dense per-position one, so the gap would carry an operator
change. The pair's internal validity beats the external comparison, and the arc already
has `plain-panel-norm-match` for the external one.

**Measure the budget on CODE as well as web text** (Wolfe asked for the pair on code). The
corpus is here — `data/pretok/code`, 367 M tokens of `code_search_net`, the same
`bigcode/starcoder2-7b` tokenizer, reachable through the curriculum loader. The READOUT is
not: `core_depth_sweep.py` builds its val rows with `create_dataloader(cfg.data.tokenizer,
cfg.data.dataset, ...)`, which reads `data.dataset` (the OpenWebText arrow glob) and knows
nothing about pretok shards, and there is no code `eval_holdout` shard, so a code pair
would train on code and be scored on web text. Deferred with the requirement written down
rather than bodged; see the prereg's "not built".

## Acceptance criteria

1. `tests/test_span_mask_leak.py` passes, and every one of the five span routes, removed
   in turn, makes it fail. Both are verified in this change.
2. `model.span_mask: "off"` is bit-identical to the tree before this change: the full CPU
   suite passes, and `base` / `tul_slot_mux_norm_match` / `notul_panel_norm_match` compose
   with `span_mask: off` and an unchanged `core_impl`.
3. A Hydra compose of `budget_web_full` against `budget_web_span` differs in
   `model.span_mask` and `wandb.name` and nothing else.
4. `lab/divergence/span_budget_profile.py` reads 0.0000 at every offset for an arm against
   itself, and reproduces the recorded +0.1919 overall gap between `norm-match-20k` and
   `slot-mnext-staged-20k` at step 5000, depth 6.
5. The pair runs to 5,000 steps and the profile script produces the gap and its offset
   curve. (Not met yet — the arms are queued.)

## Risks

- **The measurement is a lower bound on the budget, not the budget.** Both models are
  trained for 5,000 steps at 6 M parameters' worth of tokens per step. A bigger or
  longer-trained unrestricted model would extract more from its context, so the gap would
  grow. The number is "what THIS recipe can extract across a boundary", which is the right
  bar for a slot in THIS tree and not a universal constant.
- **The restricted arm has a different effective task.** 51 span starts per row is 51
  cold starts; the model spends capacity on them that the unrestricted model spends
  elsewhere. That is what "nothing crosses" costs, and it is part of the budget by
  construction, but it means the gap is not purely "missing information".
- **The restricted arm is slower and heavier.** The dense compressed branch materialises a
  `[B, H, S, S]` fp32 score tensor in each of the 6 prelude/coda blocks, and
  `use_kernels: false` puts the CE back on full logits. Measured before launch, not
  assumed; the prereg carries the number.
- **A budget that comes back small does not exonerate the slot.** It would say the SLOT
  LOOP's flat K-curves are the correct answer for web text at this scale, and move the
  question to data where spans depend on each other. A budget that comes back large says
  the opposite: the information is there and nine arms failed to route it, and the reader
  and the target are the lane.
- **The offset axis depends on a definition** (the offset of the PREDICTED token inside
  its span, not of the query). The script says so in its docstring and prints it; a reader
  who assumes the other convention is off by one span position at the boundary.

## Outcome

Filed 2026-09-11 14:53 under `lab/experiments/failures/2026-09-11-arc-span-budget.md`
(failure by the protocol: the headline prediction missed upward). The budget on web text at
5k is **0.3994 nats [0.3838, 0.4162]** paired on 491,520 tokens, and it grew from 0.163 at
2,500. Its shape is a flat long-range part of 0.31 nats at every position eight or more
tokens into a span plus a short-context spike of 0.96 at the span's first position, decaying
over seven tokens; it is not front-loaded at offset 0 (0.28). The slot family's prefix-write
ablation (0.093 total) prices about a quarter of it. The loop does not change under the cut
(K1−K6 0.028 → 0.030). Next: split the budget between the prefix write and the slot cells'
coda states on the same rows (an all-slot ablation on the mask arm).
