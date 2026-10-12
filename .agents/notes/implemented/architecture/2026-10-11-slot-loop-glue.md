# Agent Note: Slot-loop glue

Status: implemented

## Problem

The labelled graph-replay ledger of `nowta_pointer_fast_spdcells` (253 ms busy step,
`/home/wolfe/morph-scratch/perf/flame2/LEDGER.md`) charged 16.7 ms to "core glue" and 75 ms to
the six-to-eight core passes. The glue is not matrix work. It is row moves, selects, norms and
the fixed-point term. Source lines, by cost:

- `slot_compact_step` gather, scatter and pad (`take_rows`, `put_rows`): about 4 ms. Each
  pass gathered and padded the FULL row, then used `n` positions.
- The freeze `torch.where(active, h_new, h)` and the four fixed-point accumulators: 2.5 ms.
- `input_norm` (eager `RMSNorm` on `[6, 1280, 4, 1024]` fp32): 1.6 ms.
- The slot-cell pass norm (`_cell_norm_apply`): 2.5 ms.
- `_fp_term_exact`: 2.5 ms.
- `inj_terms[i]` (a `SelectBackward` per layer, zero-fill plus add): about 1.5 ms across the
  passes. The eager injection add per layer: 0.8 ms per layer per pass.

## Decision

Exact changes, no key (byte gate: `FINAL_SHA256` 4a73ebf5... before and after, 45 steps):

- `take_rows` gathers only the pass's `n` positions and `put_rows` scatters `n` rows into
  the carrier. `compact_order` returns `perm` only (`inv` is gone).
- `_apply_core_step` unbinds `inj_terms` once and, under `model.inject_fold`, hands layer
  i+1's term to layer i's Hyper-Connection exit kernel, as the prelude and coda do.
- `DiagonalInjection.source_view(e)` cuts `e` to the context channels once, before the loop.
- The first fixed-point accumulator write uses a scalar `0.0`, not `zeros_like`.

Keyed changes, default off, not bit-identical:

- `model.loop_glue_compile`: `input_norm` and the cell norm run through
  `torch.compile(rms_norm_fp32)`.

## Alternatives considered

- Run the cell norm, the fixed-point term and the freeze on the compact rows
  (`slot_compact_epilogue`). Built and measured, then cut. The cotangent clip
  (`slot_cot_clip`) hooks the full carrier, so the term's gradient bypassed it: at
  `core_fixed_point_lambda` 1000 the gradient was 60x larger, and the 45-step gate showed
  grad-norm spikes of 1e4 to 1e5. The norm alone (term and freeze left on the full carrier)
  matched the eager trace bit for bit on the gate config but gained nothing: 230.2 ms per
  step with the key and 230.1 without, two runs each.
- Compile the fixed-point accumulators. They are pure memory traffic with no arithmetic to
  fuse. Not tried.
- A hand-written exact RMSNorm backward. Fragile against ATen order. Not built.
- One fused `take_rows` for q, x, q_lat and gate_pre. Needs a `cat` copy that costs what the
  gather saves.

## Consequences

- Wall clock on `nowta_pointer_fast_spdcells`, 300 steps, graph step, ms per step between
  logged steps 100 and 280: pristine 234.3 (three runs); exact changes 230.1 (three runs);
  with `loop_glue_compile` 226.2 (two runs). That is 1.8 % and 3.5 % faster.
- Graph-replay ledger of the exact changes: core glue 16.7 to 12.9 ms, gain hinge 17.5 to
  16.5 ms, GPU span 261.4 to 255.9 ms (about 2 %). The wall-clock bench agrees (above).
- `loop_glue_compile` moves step-0 loss by 3e-5 relative and grad norm by 6e-4. The 45-step
  trajectory decorrelates in the early, chaotic phase (max loss deviation 0.058 nats at loss
  about 21). Any non-identical change does this.
- The remaining glue (12.9 ms) is the freeze and accumulator selects, the fixed-point term
  and the row moves. The fixed-point term and the freeze cannot move to the compact rows
  without a compact-space version of the cotangent clip.
- Not verified: the eval path with either key, memory with either key, a full-length run.

Related: [slot-loop compaction](../../proposed/architecture/2026-10-08-slot-loop-compaction.md)
(the `RowPermute` it named is replaced by `_RowTake` and `_RowPut`).
