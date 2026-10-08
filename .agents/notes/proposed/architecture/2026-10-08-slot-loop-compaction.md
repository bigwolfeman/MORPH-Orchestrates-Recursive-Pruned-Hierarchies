# Agent Note: slot-loop compaction (run only the live cells of each pass)

Status: proposed

## Problem

The LXTUL slot loop (`MORPHTransformer._tul_core`, config `lxtul_pointer`) runs every pass
over all 64 slots x 4 cells of a row. Under `model.graph_safe` it runs a fixed 8 passes and
keeps a finished cell with `torch.where(active, h_new, h)`. The audit of 2026-10-08 puts the
discarded slot-pass work at 40.7 % of the core's passes
([speed plan](2026-10-08-lxtul-training-speed-plan.md), item C). The finished cells are not
dead: a later slot's attention reads every cell of every earlier slot at every pass. So a
compacted pass must still give the finished cells' keys and values to the live queries.

A graph-captured step needs fixed shapes. With independent Poisson depths, the number of
live slots at pass `t` changes from row to row and from step to step.

## Proposal

Two keys under `model.`, both off by default. With both off, the step is byte-identical to
the tree before them. Code: `morph/model/slot_compact.py` (the proof is in its docstring),
the `slot_compact` hooks in `morph/model/transformer.py`, `morph/model/mhc.py` and
`morph/model/attention.py`, and the `qpos` input of `morph/kernels/triton/tg_strict_attention.py`.

1. `model.slot_depth_stratified: true` changes the joint law of the depths and keeps each
   slot's marginal law. Per row, the 64 slot depths are a systematic sample of
   `clamp(Poisson(6), 1, 8)`: one `U ~ U[0,1)` per row, sorted slot `k` gets the quantile at
   `(k + U) / 64`, then a random permutation per row. Every slot's marginal is exact. The
   number of slots of a row that are deeper than `t` is at most
   `C_t = 64 - floor(64 * P(D <= t))` = 64, 63, 61, 55, 46, 36, 26, 17. The bound holds for
   any float rounding of `k + U`, and pad slots (depth 1) only lower the count.
2. `model.slot_compact: off | full | gather` (needs the stratified draw).
   - `full` is the reference form of the new function. Every cell is still computed. A
     frozen cell gives each core layer's attention its CACHED attention input
     (`norm(x_bar)`), written at the cell's last live pass. The old loop instead recomputed
     a frozen cell's K/V from its frozen carrier and the CURRENT state of the slots before
     it. So `full` is a function change against `off`.
   - `gather` computes the same function on `C_t * 4` rows per pass. A stable sort moves the
     live cells to the front of the row. Row-wise ops run on the first `C_t * 4` rows. K/V
     come from the cache over all cells, with this pass's rows written in. The strict
     kernel (`MODE_CELLS`) reads each query's original position from `qpos`, so the
     cell-causal mask and the window are those of the full row. Padding rows are real
     inactive cells with finite states, and nothing writes their outputs back.
   - All row moves use `_RowPermute`, a gather by a full permutation whose backward is the
     gather by the inverse. There is no scatter and no atomic add.
   - The gain hinge runs the same compacted step on detached caches.
   - A `torch._assert_async` per pass stops the run if more cells are live than `C_t * 4`.
   - `validate_slot_compact` refuses every config the compacted pass does not carry, by key.
     Examples: checkpointed passes, `loop_reach`, `loop_carry`, `reread`, `xhc_streams`,
     code rollouts, `slot_gain_reuse_f0`, one cell per slot, eager attention, and
     `slot_depth_fixed` with the stratified draw.
3. `morph/training/train.py`: with `slot_compact: gather` the dynamo cache limit is 256, not
   64. The HC residual's resume frame guards on each block's CMS type, so the 8 pass shapes
   need about 120 entries. At 64 the excess ran eager (`recompile_limit (64)` in the first
   bench).

Measured on 2026-10-08 (agent `compaction`):

- Tiny fixture, `lxtul_pointer_ditto`, `gather` against `full`: the loss is bit-identical.
  The worst gradient is 1.1e-5 relative in fp32 and 1.5e-6 in bf16. The positive control,
  `off` against `full`, differs by a median of 8 % per gradient.
- Real shapes, `lxtul-pointer-rerun/step_5000.pt`, 2 batches, fp32, fp32 pick, no TF32:
  `gather` against `full` gives a loss within 3.1e-7 relative, 0 flipped winner picks of
  1211 live slot-passes, and a median gradient difference of 7.2e-5. In bf16 at batch 6, a few
  winner picks flip (0 to 8 per pass of 86 to 328 live slots), so the gradient median is
  2.7 %. With every row computed (`C_t = 64`, the same row moves) the loss is bit-identical
  to `full`. So the bf16 gap comes from GEMM shape rounding through the discrete pick. It
  is not a dropped row.
- The function change, `off` against `full` at real shapes (fp32): the loss differs by
  2.6e-4 and 1.9e-5 relative, the gradient median by 3.0 %, and up to 8 of 80 picks flip.
- `core_depth_sweep.py --depths 1,6` is the same with the keys on or off (K1-K6 +0.0095
  both). The sweep runs at one forced depth, so no cell is frozen and the draw is not used.
  It cannot see either key.
- Speed: see the bench and profile in the agent's report of 2026-10-08. FAST2 runs
  343.3 ms per step and FAST2 + keys 328.3 ms (two runs each, alternating). Reserved memory
  drops 21.18 to 19.19 GB.

## Alternatives considered

- **Cache K/V per layer instead of the attention input.** The CCA attention makes K/V from
  the input through a causal depthwise conv and a value shift. A K/V cache would need the
  conv state and the value-shift state of each frozen cell's neighbours as well. The input
  cache gives all of them for one `[B, N, C]` tensor per layer. It costs one recompute of
  the K/V projections over all cells per pass, which is cheap next to the query rows.
- **Per-batch capacity (one count for the whole batch).** Shapes stay fixed only if the
  batch maximum is bounded, and the per-row draw already bounds each row. Per-row capacity
  keeps the kernel's row layout simple (`qpos` is `[B, C_t * M]`).
- **Stratify over the valid slots only.** Pads would then not take part in the draw. That
  makes the capacity depend on the row's valid count, which is not a fixed shape. Pads get
  depth 1 after the draw, which only lowers the live count, so the bound holds.
- **Keep the independent draw and pad to the batch's live maximum.** Shapes change every
  step. That breaks the graph-captured step and makes a dynamo entry per count.
- **Recompute frozen cells' K/V as today (`off`) but on fewer query rows.** The keys of a
  frozen cell depend on the current state of the earlier slots, so every cell must run the
  row-wise ops at every pass. That removes the saving.

## Acceptance criteria

- Keys off: byte-identical step. Gate (20 steps, deterministic, FAST2): the HEAD export and
  the tree with the keys gave the same trace (`cmp` IDENTICAL, 2026-10-08).
- `gather` equals `full` at fixed weights to kernel rounding, in loss and every gradient
  (`tests/test_slot_compact.py`). The test must fail under stale pre-freeze K/V and under
  wrong query positions. Both sabotages were run and both failed the test.
- The graph-step replay with the keys on equals its eager step bit for bit
  (`tests/test_graph_step.py`, id `slot_compact`).
- Wolfe's acceptance for near-equivalent cuts (speed plan): the paired continuation from
  step 2500 to 3500 must keep the loop contribution within 5 % of the baseline. Not run yet.
  Run both decomposition arms: stratified draw only, then both keys.

## Risks

- Two function changes, not one. The stratified draw changes the joint law of the depths
  (a row can no longer draw many deep slots at once). The cache changes what a frozen cell
  shows to later slots. Neither is measured on training yet.
- The saving is 4 % of the step, not 40 %. The slot loop's core is one part of the step,
  and the pass still pays the K/V projections and the attention over all cells.
- Every feature the guard refuses (checkpointed passes, loop reach, rollouts) stays on the
  uncompacted path. A future arm that turns one on must extend `_slot_compact_step` first.
- Eval and decode do not use the keys. A model trained with them is evaluated on the old
  path. At one forced depth no cell is frozen and the two paths agree. With per-slot depths
  they differ by the cache semantics (training mode, fixed weights: 2.6e-4 and 1.9e-5
  relative in loss).
