# Agent Note: The CUDA graph step survives eval, and eval costs what its forwards cost

Status: implemented

## Problem

A 5k run of `nowta_pointer_fast_spdcells_speed_bf16c` spends about 385 s of 1470 s wall in its 20
evals (`eval_every 250`, `n_eval_batches 20`). Measured 2026-10-12 with `training.step_time_print`
over a 600-step run (`eval_every 100`): 23.2 s per eval, counted as the eval plus the three steps
after it, minus the median step (191 ms). Two parts:

- `GraphStep.released()` frees the 16 graphs and the 19 GiB pool around eval; the next step records
  them again (5.3 to 5.8 s, `[graph-step] recorded 16 graph(s)`). The graph machinery after an eval
  is cheap: `mark_dirty()` only sets a flag and the next replay compares tensor addresses.
- The eval itself, about 16 s. The attention-lift probe also peaked at 8.9 GB (below), which is
  why eval did not fit beside the pool.

## Where the eval time goes

One eval is 20 batches of 6 rows (L 1280). The recipe sets `tul.eval_ablations`, so each batch runs
six forwards. Seconds per eval (20 batches), peak GB above the resident base (2.7 GB), from a timed
wrapper around each model entry point (`nowta_pointer_fast_spdcells_speed_bf16c`, 80 calls each,
graphs kept, command in the Verification section):

| Part | s per eval | peak GB | kernels per call |
|---|---|---|---|
| main forward: CE, pointer, fan oracle, stream CE, span stats, first-token readings | 4.0 | 1.96 | 14.1k |
| plan_worth zero | 2.1 | 1.47 | 10.5k |
| plan_worth shuffle | 2.1 | 1.47 | 10.5k |
| plan_worth wrong_seed (full forward, other slot seed) | 4.0 | 1.96 | 14.2k |
| slot-state probe (front + loop, rank and cosine on the CPU) | 1.9 | 0.53 | 5.7k |
| attention-lift probe | 1.9 lean, 2.8 full | 8.87 full | 9.7k |
| total | about 16 | | |

Every forward is launch-bound: 10k to 14k kernels in 0.1 to 0.2 s is 12 to 14 us per launch. The
ablation and probe block (zero, shuffle, wrong_seed, slot probe, lift) is 75 % of the eval. The
attention-lift probe counts 0 calls on this recipe (`val/attn_lift_n_calls` is 0.0 in every eval; I did not trace why the hooked window branch
never runs here), so its only logged value is that zero. The full probe built `[B, L, V]` logits and the pointer mixture over the vocabulary
(`mixed_logprobs`), 8.87 GB, for a result nothing reads.

## Decision

Two keys, both default off and byte-identical off (`morph/configs/base.yaml`, `training` block).

- `training.eval_keep_graphs: true` runs the periodic eval beside the held pool: no `released()`, no
  re-record. The attention-lift probe runs with `lean=True` (`tul_attn_lift_probe`): the forward stops
  at the coda readout (`_coda_state_only`), so it builds no logits. The attention layers run the
  same calls, so the counted statistics are the same numbers (bit-equal in
  `tests/test_eval_cost.py`). Eval peak is 5.75 GiB allocated beside the pool (speed recipe), with
  6.3 GiB of the 31.4 GiB card free after it; the trainer prints both numbers after every eval
  (`[eval-keep]`). The final eval and generation still release the graphs.
- `training.eval_diag_every: k` runs the ablation and probe block on every k-th periodic eval, the
  first included, and the final eval always. Every other reading (val loss, the fan and pointer
  readings, span stats) runs on every eval. Default 1 is the shipped behaviour. The shuffle probe
  draws its permutation from the global CUDA stream, so k > 1 shifts the training RNG after each
  skipped block: same distribution, not the same trajectory as k = 1.

Measured 600 steps, `eval_every 100`, same machine, separate processes, each of shipped and
`eval_keep_graphs` run twice (excess wall per eval, mean of five evals): shipped 23.2 s and 23.2 s;
`eval_keep_graphs` 16.3 s and 16.3 s; plus `eval_diag_every 4` (one run) 5.0 s on the skipped
evals, 16.4 s on the diag evals. Loop wall from step 100 to 600: 210.9 s, 175.1 s, 141.4 s. Median
step time is 191 ms in all arms, so the training rate is unchanged.

## Why eval fits only just

The pool is 19.0 GiB on the speed recipe and 22.35 GiB under `training.deterministic=true`. The card
has 31.4 GiB and the desktop holds 2 to 5 GiB. Eval needs the resident base (2.7 GiB) and 2.3 GiB for
the biggest forward. In the deterministic recipe at batch 6 the sum is within 1 GiB of the card: one
gate run (`training.deterministic=true`, batch 6) hit `OutOfMemoryError` inside a Triton launch at
the first eval while a second identical run passed, so the margin there depends on what the
desktop holds. The `[eval-keep]` line shows the margin each eval.

The 2.3 GiB forward peak is not one big tensor. At the peak the live set is the span CE
accumulators (0.7 GB), the quantised head and its row quantiser (0.5 GB), the loop and HC buffers,
each under 0.25 GB. Row-chunking the batch would shave it, but the fan, rank and cosine readings
and the span statistics read the whole batch, so a chunked eval changes what they mean. Not done.

## Alternatives considered

- Chunk the batch over rows so eval fits in about 2 GB (the earlier proposal): the instruments that
  read the whole batch change meaning; and with the lift probe lean, eval already fits. Not built.
- Share the front and loop between the main, zero, shuffle and slot-state forwards. Estimated saving
  up to 0.28 s per batch (3 reuses of a 0.095 s front and loop), 5 s per eval. Not built:
  `_forward_tul` is a 2000-line function with stashes (`_lsel_out`, `_core_aux`, fan statistics) that
  a cache hit would not replay, and a wrong stash changes logged values silently.
- Eval inside the graphs' pool: OOM at 1.41 GiB (earlier builder, 2026-10-12).
- Lazy re-record of only the drawn (kind, variant): saves about 0.5 s per eval. Moot with the key.
- Drop the attention-lift probe on this recipe because it counts 0 calls: not done, diagnostics are
  not thinned without the owner's decision. `eval_diag_every` covers the cost.

## Consequences

- With `eval_keep_graphs` the 120-step deterministic run with `eval_every 40` gives a trace
  byte-identical to the shipped code (below). The step after an eval records nothing.
- Final eval and generation still release the graphs and pay the re-record once.
- A run that adds memory next to the pool (a larger eval batch, a bigger probe) can run out of
  memory at eval. It fails loud at the first eval, not silently.

## Verification (2026-10-12, worktree `perf/eval-cost`, base 24b5077c)

- Trace gate A, `training.deterministic=true training.graph_step=true training.eval_every=40
  training.n_eval_batches=2`, 120 steps, `nowta_pointer_fast_spdcells`, batch 6: shipped tree
  (`git archive`), this tree key off, and this tree `eval_keep_graphs=true` all end in
  `FINAL_SHA256 9d2fc83abf0c98699461fd7d5560d91dcea70a93d9bf8fac99afbf0641769a6e`; `cmp` of the 121-line
  traces says identical.
- Trace gate B, the same at `training.batch_size=3` (the first keep run at batch 6 hit the OOM above):
  all three end in `4c0659953e13e6a58756b9c81ed3bb950fbe54a816ea695cf6b439f21887163a`.
- Eval values, key off against `eval_keep_graphs=true` on the same weights (gate B, evals at steps 40
  and 80, 87 `val/` keys, includes the plan-worth, slot-rank and `attn_lift_n_calls` keys): max abs
  diff 0.0 for every key. The lift statistics themselves are 0 calls on the recipe, so their equality
  is pinned by the unit test, 8 statistics with 4 calls on a CPU fixture.
- Standard byte gate (45 steps, `eval_every 100000`): `4a73ebf5426b9dfa889c81a8608a23d23a6785bb225ff32e2d5dd03476bd70d7`.
- `tests/test_eval_cost.py` 4 passed; sabotage (lean off, `diag` ignored) fails the matching tests.
