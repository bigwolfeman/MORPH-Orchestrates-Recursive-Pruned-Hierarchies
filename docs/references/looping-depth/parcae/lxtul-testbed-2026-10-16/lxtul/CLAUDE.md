# lxtul/ — MORPH's LXTUL slot loop on a Parcae backbone (testbed)

Branch `lxtul-testbed`. A clean backbone to test whether the slot loop contributes,
without MORPH's Hyper-Connections, ternary weights, CCA/CSA/HCA attention or AdEMAMix.

- `SPEC.md` — the LXTUL forward and losses read off MORPH master 3f9aae2, with file:line
  citations. `model.py` cites its section numbers. Read it before changing the model.
- `model.py` — `LXTULParcae(Parcae)`: Parcae's own blocks, injection, value embeddings and
  init, plus the LXTUL mechanism. The module docstring lists every backbone deviation
  from MORPH (one stream, full-width injection with floor gain 0.447 instead of 0.865,
  no x0/bigram re-injection, no dropout, kv heads = heads).
- `attention.py` — `FlexSelfAttention` (class-swapped onto Parcae blocks, same params) and
  the three strict masks. Parcae's stock attention IGNORES its mask argument.
- `ce.py` — chunked linear CE, slot-id logit masked, grads computed in the forward. Also the
  static row packer (`static_pack`, `take_rows`, `put_rows`, `linear_ce_rows`) behind `ce_rows`
  and `spandec_packed`: labelled rows go to a buffer fixed by the shapes, no `nonzero()` sync.
- `data.py` — MORPH's tokenizer, OWT stream, boundary rule and packer, unchanged (so CE is
  comparable to MORPH ledger CE). Needs MORPH_ROOT on sys.path (set inside).
- `data.py` also holds the eval-document guard: `training.exclude_eval_docs` (default true) makes the
  training stream drop the documents the 480 eval rows came from (OWT docs [50000, 50000 + ~406)).
  The stream before doc 50k is unchanged, so runs under ~10k steps are identical. `tests/test_data_hygiene.py`.
- Hygiene keys, all default off: `lxtul.static_index` (static-shape epivol and encoder floor, rounding-only),
  `lxtul.mask_cache` (reuse masks for one layout, exact, helps the K-sweep only), `plain.sync_free`
  (no host syncs in plain Parcae's depth bookkeeping, exact). Tests `test_static_index.py`, `test_sync_free.py`.
- `build.py` / `bench.py` / `configs/bench.yaml` — plain Parcae vs Parcae-LXTUL throughput.
- Speed keys (`LXTULConfig`, all default off; `plain.row_compact` for the plain loop):
  `compact_slots` (valid slots of the batch as one sequence; pad slots dropped), `static_depth`
  (no `int(dslot.max())` sync), `cot_clip_fn` (the clip as autograd Functions, no dict hooks),
  `ce_rows`, `spandec_packed`, `plain.row_compact`. Each is the same function as off, to
  rounding. A FINISHED slot cannot be dropped from a pass: later slots read its K/V and those
  depend on the earlier slots' current state (SPEC 2.2, 3.5; `test_a_finished_slot_cannot_be_skipped`).
  Pad slots must sit last (the packer guarantees it; `test_batch_has_pads_last_and_caps_hold`).
- `slot_depth_stratified` + `slot_compact` ("full" | "gather"; `slot_compact.py`, ported from MORPH's
  fast recipe): training draws per-row systematic depths (same marginal per slot) and a FINISHED cell
  serves each core layer's attention with the K/V of its last ACTIVE pass. A different function from off
  (tests: `test_full_differs_from_off_when_an_earlier_slot_stays_active`); "gather" is "full" computed on
  C_t * M cells per pass with static shapes, equal to rounding. Supported beside static_depth, cot_clip_fn,
  ce_rows, spandec_packed, any ckpt_passes and the gain hinge; refused with compact_slots and without the
  stratified draw. Training only: eval is unchanged. `tests/test_slot_compact.py`.
- Lean recipe keys (`configs/arm/lean.yaml`, all default off, tests in `tests/test_lean.py`; plan Step 3):
  `cell_count: poisson` (+ `cell_count_mean/min/max`, `cell_count_eval`): each train batch has M =
  clamp(Poisson(mean), min, max or cells) cells per slot; `data.cell_law` + `data.relay_cells` re-lay the loader's
  rows with MORPH's `relayout_batch` on the loader thread (counter-seeded, CPU, no sync), the model reads
  M = `layout.prefix_k` and uses roles 0..M-1 (`reg_q_embed[:M]`, `reg_p_cell_embed[:M]`, `W_prefix[:M]`), so the
  loop runs on S*M cells and the coda row is `cell_l_total(M)` long (computed, not masked). Needs
  `select: none`; M = 1 has no epivol term; eval rows stay at `cells` unless `cell_count_eval` says otherwise.
  Not the same function as fixed. `spandec_reads_cells`: `SetCross` after each span-decoder block reads the slot's
  M final cells as an unordered set (zero-init output, MORPH `_SpanDecCross`). The one-block decoder is the
  existing `spandec_layers: 1`. One compiled shape per M (row length and S*M both change).
- Minimal arm (`configs/arm/minimal.yaml`, `minimal_noptr.yaml`; Wolfe 2026-10-14, tests `tests/test_minimal.py`): the
  strict slot loop with ONE cell per slot, no span decoder, no gain hinge, no fixed-point term, no cotangent clip, no
  epivol, no latent selection; Parcae's diagonal injection, the pointer head (not in `_noptr`) and token-state dropout
  stay. Each removal skips its compute and builds none of its parameters (`spandec_weight 0`, `gain_lambda 0`,
  `fixed_point_lambda 0`, `cot_clip 0`, `repel_lambda 0`, `select none`); `model.py` raises on `cells=1` with
  `select=latent` or `repel_lambda > 0`. The layout follows `lxtul.cells`: `data.layout_for_cells` sets the runtime's
  `prefix_k` (static, not the per-batch relayout) so a one-cell row is `seq_len + max_slots` = 1088 positions, and
  `build.py` sets the model's `block_size` to the packed length. Init draws differ from the default arm (fewer modules
  built); the default arm's parameter set is pinned by `test_default_arm_parameter_set_is_pinned`.
- `data.make_loader` / `evaluate.eval_rows` take both the hygiene keys (`doc_stats`, `exclude_docs`) and the lean keys
  (`cell_law`, `eval_cells`): the cell re-layout wraps the document-filtered stream. `LXTULParcae._masks` keys its
  mask cache by the cell count M as well as the layout tensors; `_epivol` and `_epivol_static` take M.
  `plain.sync_free` and `plain.row_compact` are exclusive (`build.py` raises: row_compact runs its own loop).
- Flat references and the bench4 harness (2026-10-14): `configs/arm/gpt8.yaml` (8 layers), `flat14.yaml` (`layers: 14` = prelude 4 +
  core 6 + coda 4 unrolled once; `build.py` reads `arm.layers`), and `gpt8_pointer.yaml` / `flat14_pointer.yaml` (`pointer_heads: 4`:
  the SAME `PlainPointer` wrapper plain+pointer uses, hooked on the GPT's `transformer.ln_f`; with 0 heads `build()` returns the bare GPT).
  `evaluate.depth_sweep` scores a flat model once and reuses it for every depth; `mixture_off_eval` takes `kind: gpt`.
  `plain.gradient_checkpointing` (default false) turns on Parcae's own checkpointing of the plain loop. `bench_all.py` times every arm
  the same way (synced, compiled, median of 200 after 100 warm-up, cached real batches, batch 6 or 12; train and eval-forward;
  FlopCounterMode FLOPs in a subprocess via `flop_tally.py`); `summarize_bench4.py` prints the tables. Tests `test_flat_pointer.py`,
  `test_bench_all.py`.
- Spiral arms (2026-10-15, prereg `docs/experiments/planned/2026-10-15-parcae-spiral-5k.md`; `spiral.py`, tests `tests/test_spiral.py`):
  `spiral_published` (chunks 16/8/4/2), `spiral_span` (one level = the TUL boundary-rule spans, Poisson depth, written to the NEXT
  span's tokens only), `spiral_span_pair` (spans, then chunks 4 and 2) and `spiral_span_pair_mesh` (the same with the MeSH buffer
  topology: n_levels + 3 slots, per-level softmax write/read routers). Kind `spiral` = `SpiralParcae(Parcae)`: the plain arm's blocks,
  init, value embeddings and tied head; the core runs on pooled group sequences (learned softmax pool per level, `pool_w`), the update
  is `C(zh)` broadcast with gain sqrt(group size) and causally shifted, Anchor `h = h0 + u` unless `topology: mesh`. Plain rows (like
  plain and gpt); the span layout is cut on the host (`SpanLayout`, passed as `layout=`). `depth=` sets the SPAN level's passes only; a
  model without a span level raises on it and `depth_sweep` scores it once. Spans past `max_spans` (96) form an uncounted-as-source tail
  (logged: `spiral/rows_over_max_spans_per_step`, `summary.json["spans"]`). `build.py` raises Dynamo's recompile limits for this kind.
- Spiral cycle (2026-10-16, `spiral.cycle`, default absent = byte-identical; `CycleSpec` in `spiral.py`, tests `tests/test_spiral_cycle.py`):
  the whole coarse-to-fine descent repeats as an outer cycle (Anchor topology only; no span or poisson level inside). Count per row
  = Parcae's law (Poisson `mean`, floor 1, cap `max`, n no-grad then k grad), each cycle starts at a sampled level (`start`:
  uniform | coarsest) and skips later levels with `skip_p` (training only; eval = coarsest start, no skips, `round(mean)` cycles,
  `depth=` = cycles). `credit: end` = one CE; `deep` = a CE after every cycle on a state detached between cycles (`head_fn` from
  `PlainPointer`, logged as `cycle/ce_c{i}`). The schedule is CPU data and runs in eager Python: no new compiled shapes. Arms
  `spiral_cycle_{end,deep,fixedsched,deep_fixedsched}`. `plain.deep_supervision` (`plain_deep.py`, default off) is the same deep
  recipe on the plain Parcae loop, one CE per core pass.
- `tests/test_compact.py` — CPU tests of the speed keys (float64 reference; FlexAttention is
  replaced by SDPA under its mask). `tests/keys_plugin.py` runs any test file with keys on:
  `LXTUL_KEYS='{"ce_rows": true}' pytest -p lxtul.tests.keys_plugin lxtul/tests/test_geometry.py`.
- `tests/test_geometry.py` — causality and "loop is the only cross-span channel" on the
  forward (GPU). Run it after ANY change to masks, the write or the layout.

Run with the MORPH venv: `PYTHONPATH=/home/wolfe/parcae:$MORPH_ROOT
/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python -m lxtul.bench`, under
`flock /home/wolfe/morph-scratch/gpu.lock`, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Gotcha: the bench cycles 16 batches, so its CE is memorization (trained batch 0.92 vs
unseen 6.51 after 40 steps, 2026-10-06). Read only tok/s and memory from it.
