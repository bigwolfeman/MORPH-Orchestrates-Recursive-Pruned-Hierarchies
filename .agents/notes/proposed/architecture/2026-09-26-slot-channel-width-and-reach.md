# Agent Note: slot channel width and coda reach under strict geometry

Status: proposed

## Problem

fp01 (`morph/configs/tul_slot_spandec_strict_e4probe_fp01.yaml`) trails the plain model
(no slots, full causal attention) on the same tokens by 0.265 nats at 5k and 0.333 at 10k,
and the gap widened with training
([lab/experiments/failures/2026-09-26-lxtul-fp01-vs-plain-10k.md](../../../../lab/experiments/failures/2026-09-26-lxtul-fp01-vs-plain-10k.md)).
Under `tul.tg_geometry: strict` the slot loop is the only cross-span channel. Each slot
writes `prefix_k: 2` cells, and each cell is `W_prefix[k] @ z`, a projection of the one
exit state `z`. So every fact a token needs from an earlier span, the previous span
included, must pass through two projections of one vector per span. The plain model reads
the previous span token by token. The gap can come from two places:

1. the channel is too narrow (too few cells, or cells with no distinct content);
2. the geometry makes near context (the previous span) and far context share that one
   narrow channel, when near context is cheap to read directly.

Loop depth is not the question here. fp01's loop earns (coda K1-K6 +0.0126) and the gap
still grows.

## Proposal

Three arms, built 2026-09-26 on branch `tul-arms-ab`, not yet run. The prereg comes before
the runs.

**(a1) a wider channel.** `tul_slot_spandec_strict_e4probe_fp01_pk4.yaml` = fp01 +
`tul.prefix_k: 4`, nothing else. It composes with `code_enum_k: 4`, the parallel span head
and strict geometry (no refusal fires; the tiny build trains one step). The row is
`L_total = 1024 + 4 * 64 = 1280` positions (fp01: 1152); `max_slots` and `seq_len` do not
change. The four cells are four projections of one `z`: this widens what the coda can READ
per slot (four keys and values), not the state the loop carries.

**(a2) distinct content per cell.** The ideal arm, fp01 + fan4 write-all, cannot be built:
`TULConfig._check_code_enum` refuses `fan_k > 0` ("the fan's K streams are a second width
axis"), and `_check_spandec_parallel` refuses it too. Those refusals stay.
`tul_slot_spandec_strict_fan4_all_fp01.yaml` = the filed fan4-all recipe +
`model.core_fixed_point_lambda: 0.1` (fp01's term). Against fp01 it lacks `code_enum_k 4`
and the parallel head, and it inherits `spandec: true`, `slot_gain_target 0.9` and
`slot_gain_tail_lambda 0` from the fan chain. Its one-factor partner is
`tul_slot_spandec_strict_fan4_all`, not fp01.

**(b) a looser coda.** New key `tul.tg_coda_token_reach: r` (default 0), valid only under
`tg_geometry: strict`. At `r >= 1` a coda TOKEN of span `s` also reads the TOKENS of spans
`s-r .. s-1` directly. The prelude stays same-span only, a prefix cell still reads itself
alone, and `tg_coda_prefix_reach` still picks the cells a token reads. The relation lives in
the one place strict relations are built (`tul_layout.tg_strict_allow`, called from
`MORPHTransformer._tul_tg_kwargs`). At `r = 0` the call is the tree's call and no new term
is built. `tul_slot_spandec_strict_e4probe_fp01_reach1.yaml` = fp01 + `r = 1`.

Three decisions inside (b):

- **The conv, the `W_v_prev` value shift and the retention carry keep the
  `tg_segment_ids` partition.** They compute one feature per KEY position, and every query
  reads that feature. A conv at the head of span `s-1` that read the tail of span `s-2`
  would hand `s-2` to span `s` in one layer. Keeping the partition makes the attention
  relation the only new route.
- **The open tail reads the last real span.** The tokens after a row's last boundary carry
  the dump-bin id `max_slots`. `strict_span_ordinal` maps them to `n_slots`, so the tail is
  one more span and its previous span is the last real one.
- **The receptive field is stated, not hidden.** One coda layer reaches `r` spans back.
  Layers compose: layer `l` reads span `s-1`'s layer `l-1` states, which already read
  `s-2`. Measured on the tiny model with the loop's write zeroed: an edit `m` spans back
  moves span `s` if and only if `m <= r * n_coda`, at `(r, n_coda)` = (1,1), (1,2), (1,3),
  (1,4), (2,1), (2,2). At init the influence falls about tenfold per extra span. fp01 has
  `n_coda 4`, so arm (b) reads 4 spans back through tokens and relies on the loop beyond.

Tests: `tests/test_tul_arms_ab.py`. Brief: [morph/model/CLAUDE.md](../../../../morph/model/CLAUDE.md).

## Alternatives considered

- **Let the coda conv cross into span `s-1` under reach 1 (a query-relative conv
  relation).** Rejected. The conv output at a key position is computed once and shared by
  every query. It cannot honour "allowed for a query in span `s`, not for one in `s-1`", so
  any conv that reads across the `s-1`/`s-2` boundary leaks `s-2` into span `s` in one
  layer. The conv sabotage in the test file shows the leak: max |dlogit| 0.030 at one
  coda layer, where the correct build reads exactly 0.
- **Gate the open tail out, the `tg_coda_prefix_reach: prev` precedent.** Rejected for this
  knob. `prev` gates the dump bin because `bag_id - 1` is not the tail's previous span
  (`tg_strict_allow` docstring), and it chose to gate rather than remap. Here the tail is
  real tokens the coda decodes. Gating it would give the
  last span of every row less context than every other span, an effect of row position.
- **Widen only one coda layer (the `model.span_mask` `span_reach_layer` precedent).** It
  would bound the token path to exactly `r` spans. Not built: the arm asks whether near
  context read directly closes the gap, and a single widened layer mixes that question with
  where in the coda the read happens. It is a one-knob follow-up if (b) moves CE and the
  loop's share collapses.
- **fp01 + fan4 write-all by removing the refusals.** Rejected. The refusals give their own
  reasons: under code_enum "the fan's K streams are a second width axis" beside the K
  rollouts, and under the parallel head "the fan's K streams are Phase B's to define". No
  per-span mixture over rollouts times streams exists in the tree.
- **`prefix_k: 8` for (a1).** The plain strict ruler measured pk8 at 0.035 nats better than
  pk2 at 5k ([lab/experiments/successes/2026-09-21-strict-pk8-ruler.md](../../../../lab/experiments/successes/2026-09-21-strict-pk8-ruler.md)).
  4 was chosen as the first step: it matches a2's width (fan4 writes 4 cells), so a1 and a2
  read at the same cell count. They still differ in five other keys (a2 below).
- **Go back to `tg_geometry: restrict`.** Rejected. Its "OR any slot cell" relation lets the
  seed bypass the loop (the whole slot channel 0.182 nats, the loop's write 0.078). Arm (b)
  opens a named, bounded token route and keeps the cells strict.

## Acceptance criteria

- Build gates (met 2026-09-26, commands in the build report): reach 0 is `torch.equal` to
  7cb6783 on loss, eval logits and every gradient; the perturbation leak test and the
  receptive-field law in `tests/test_tul_arms_ab.py` pass, and five source sabotages (mask
  term removed, coda conv reset dropped, tail ordinal dropped, ordinal built at reach 0,
  prelude widened) each fail it; each arm composes, differs from its parent by the stated keys
  and trains one step; a 30-step GPU smoke of each arm exits 0.
- Read each arm against fp01 and against the plain model on the same tokens, token-paired,
  at 5k and 10k, as the prereg sets.
- For (b), read the loop's contribution as well as CE: coda K1-K6 and the whole slot
  channel's worth (`plan_mode: zero`). A CE gain with the slot channel's worth near zero
  means the tokens replaced the loop; it does not mean the loop got better.

## Risks

- **(b) can close the gap by bypassing the loop.** The TUL goal is loop contribution, not
  matched-compute nats. With 4 spans of direct token reach, the loop may carry only what is
  older than that. The slot channel's worth must be reported beside CE.
- **(a1) adds read bandwidth, not state.** If the gap is state capacity, a1 cannot close
  it. a2 carries distinct states at the same cell count, but it is a different recipe, so
  a1 vs a2 separates the two only approximately.
- **(a2) is not one-factor against fp01.** Five inherited differences (no code_enum, no
  parallel head, `spandec: true`, two gain keys). Its reading against fp01 is a recipe
  comparison.
- **Memory and speed are measured on 30-step smokes only.** Peak allocated at step 20 (after
  the step-5 reset), steady median step time over steps 6-29, batch 6: fp01 15.86 GB, 0.965 s;
  reach1 15.86 GB, 0.980 s; pk4 17.46 GB, 1.090 s; fan4-all-fp01 17.22 GB, 0.770 s (no
  K = 4 rollout expansion). Longer runs can differ.
- **The window must cover the reach.** fp01's `window_size 256` covers one previous span
  (cap 32 tokens plus its cells) with room to spare. At large `r` the window, not the
  relation, would bound the read.
