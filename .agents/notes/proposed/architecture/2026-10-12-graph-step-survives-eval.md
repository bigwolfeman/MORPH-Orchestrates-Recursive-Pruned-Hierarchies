# Agent Note: The CUDA graph step after an eval costs a re-record, and eval memory forces it

Status: proposed

## Problem

After every eval, generation or checkpoint the trainer calls `_gs.mark_dirty()`, and the 20-step
window after an eval reads about half the steady tok/s (13.3k against 28.6k on the speed recipe).
The question was whether the graphs have to be rebuilt.

## What the code does (read, then measured 2026-10-12)

- `GraphStep.mark_dirty()` only sets `_dirty`. The next replay then compares the `data_ptr` of every
  parameter, buffer and optimizer-state tensor with the recorded ones (`_check_addresses`). It
  records nothing and runs nothing eager. Its cost is microseconds.
- The re-record comes from `GraphStep.released()`, which the trainer wraps around eval and
  generation (`train.py`, `_gs_release`). It frees all graphs and their pool, and the next
  `step()` records every (kind, variant) again on that step's batch, then replays it. `_warm`
  persists, so no step runs eager. A checkpoint does not release anything.
- "About 20 steps" is the rate window. The trainer's tok/s counter restarts after the eval block
  and spans the next 20 steps, so one stall shows as a half-speed window.
- The stall is the re-record. Trainer line `[graph-step] recorded 16 graph(s) ... in 6.0 s`
  (speed recipe, `nowta_pointer_fast_spdcells_speed_bf16c`, 5.5 s on the second record), 13.2 s
  with `training.deterministic=true` (kernels off, more launches). Per (kind, variant): trace of
  the forward 0.10 s, backward body A 0.09 s, body B 0.08 s, post 0.01 s, instantiate under
  0.015 s. It is Python and launch-recording time on the host, 16 x 0.30 s plus about 1 s of
  bookkeeping. The step after an eval takes 8.6 s wall (record plus one replay), steady steps 0.2 s.

## Why eval cannot keep the graphs

- Eval peaks at 18.0 GB allocated on the speed recipe (21.5 GB in the deterministic run) with
  `n_eval_batches=1`. The graph pool holds 19.0 GiB (22.35 GiB deterministic). The card has
  31.4 GiB and the desktop holds a few.
- Tried: eval inside the graphs' own pool (`torch.cuda.use_mem_pool(self.mempool)`), graphs kept.
  It raised `OutOfMemoryError: Tried to allocate 1.41 GiB` at the first eval, the logits-sized
  tensor. The pool's free blocks are the activation-sized blocks of the capture and cannot serve it,
  so the pool tries to grow. This repeats the 2026-10-07 measurement in the `released()` docstring.
  Code removed; no key was added.
- The eval needs about 12 logits-sized tensors alive at once (the ablation, oracle and
  instrument forwards each keep an `out` dict). Eval would have to fit in about 2 GB to sit beside
  the pool.

## Proposal

1. Make eval fit in a few GB: chunk the val batch over rows and reduce the logits-sized
   instruments in chunks. This changes every logged eval value by float reduction order, and the
   fan, pointer and oracle instruments read the whole batch (rank, cosine), so each one needs its own
   chunking rule. Gain: the 5.5 s re-record per eval disappears and `released()` is not needed for
   eval.
2. Record lazily after a release: only the (kind, variant) a step draws. Over 250 steps about 14.5
   of 16 are drawn, so the saving is about 0.5 s per eval. Not worth the pool-growth risk
   of recording beside live graphs.
3. Leave it and raise `training.eval_every`: one eval per 250 steps costs 5.5 s against 50 s of
   steps (about 10 %, not the 1.7 % first estimated; re-measure with the recipe's own `eval_every`).

## Alternatives considered

- Hold the graphs through eval with no pool change: eval does not fit (18 GB beside 19 GiB).
- Eval in the graph pool: OOM at 1.41 GiB, above.
- Unmap the pool's physical pages during eval and map them back (expandable segments): PyTorch has
  no API for a private pool; not tried.
- Record the 16 graphs on threads: capture is host-bound Python under the GIL; not tried.

## Acceptance criteria

- With eval fitting beside the pool, `training.deterministic=true training.graph_step=true
  training.eval_every=40` for 120 steps gives a `MORPH_EXACT_TRACE` identical to the current code
  (current: `FINAL_SHA256 c83688e6005803bb28a16c12881d8337e0b35bb29b6b915a2d4f14b5ba896a5f`).
- The first step after an eval takes steady-state time (no `[graph-step] recorded` line).

## Risks

- Chunked eval changes logged eval numbers, so a 5k run's val curve shifts by float noise and
  cannot be compared bit for bit with older runs.
- An eval tensor that outlives the eval inside the graphs' pool would be overwritten by the next
  replay without an error. Any in-pool design needs a live-bytes check (before: after) on the pool.
