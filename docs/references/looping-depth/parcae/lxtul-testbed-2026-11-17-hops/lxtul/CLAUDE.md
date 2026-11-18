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
- `tests/test_compact.py` — CPU tests of the speed keys (float64 reference; FlexAttention is
  replaced by SDPA under its mask). `tests/keys_plugin.py` runs any test file with keys on:
  `LXTUL_KEYS='{"ce_rows": true}' pytest -p lxtul.tests.keys_plugin lxtul/tests/test_geometry.py`.
- `hops/` (2026-11-17, branch `lxtul/hops`; tests `tests/test_hops_{chain,graph,rows,eval,eval_tul}.py`) - hop-demanding
  documents mixed into the OWT stream, and exact-match accuracy at the marked answers by hops and loop depth. The loss
  stays plain next-token CE on every token; no extra target.
  `base.py` (Document, HopSource, `build_row` whole documents under the token AND span caps, `pack_row` = MORPH's packer on a
  spec that fits the row, then the packer's tail pad, `plain_row`), `chain.py` (hnet chain task: `Y_l is Y_(l-1).` statements,
  `Find Y now. It is V.` questions), `graph.py` (GraphQA friendship graphs: a forest of 2 trees; reachability, shortest path,
  edge existence; `solve` is a reference reader that parses the token ids), `mix.py` (`HopTask`, `build_task`,
  `mix_hop_rows`), `eval.py` (`hop_accuracy`, `HopEval` = the trainer hook, the CLI). A source's per-token `hops` (0 = not an
  answer, h = hops) and `family` mark the TARGET token; packed rows carry them aligned to the labels.
  Config: `task=hops` (`configs/task/hops.yaml`; default `task=none`, `kind: none`, builds nothing and changes no batch).
  `task.max_slots: 256` is the slot capacity of the slot arms (model `max_slots`, block_size = 1024 + cells * 256: 1280 at one
  cell) and of the layout hop rows are packed with. A 4-token statement is one span, so a 1024-token row holds 256 spans.
  The MORPH config stays STOCK (`tul.max_slots` 64): `pack_tul_row` fills a row until tokens + cells * slots = L_total, so
  raising `tul.max_slots` there would put more OWT tokens in every row (it did, and overflowed the flat arms' block_size).
  OWT rows (train stream and `eval_rows(widen_to=)`) are packed as without the task and, for slot arms, WIDENED by
  `base.widen_batch` (same tokens, slots, labels; extra tail pads). `tests/test_hops_widen.py` pins it, incl. equal hidden states.
  Training documents mix sizes and are dense (2026-11-17): chain `len_caps` (a cap per document, lengths ~ U(1, cap)) and
  `ask: nodes_all_but_one` (K - 1 chains, each once, at a random node; hop label = lookups from the value; eval keeps
  `heads_half`); graph `n_choices` (n per document), clip(n // 3, 2, `questions_max`) questions. Eval rows are pinned by a hash
  in `tests/test_hops_rows.py`.
  `row_share` per source
  (batch 6: a seeded draw per row; B * share whole: exact). Hop row r of source s is built from `(task.seed, s, r)` only, so a
  flat, a token-loop and a slot arm see the same hop documents; the OWT rows still advance the stream, so the other rows of
  a batch are the rows of the arm without the task. Eval rows are fixed (`task.eval_seed`, never the training seed). Graph eval rows are MIXED documents (buckets are per
  answer; a document whose questions share one answer is solved by repeating the previous answer: the first design read 0.94
  untrained). `tests/test_hops_leaks.py` pins repeat-previous / majority-so-far baselines at their permutation-null chance.
  `hop_accuracy` also returns `balanced` (`reach_balanced`, `reach_balanced/h<d>`, `edge_balanced`: always-Yes reads 0.5).
  The minimal arm needs `arm.lxtul.compact_bucket_slots=512` with mixed rows: its packed length is a multiple of the bucket
  (in slots) and the row mix varies the valid-slot count (measured 347 to 1022 per batch, mean 658), so 128 gives up to 12
  compiled shapes, 512 gives 3.
  The depth-1 twin is `+lxtul.mean_depth=1 +lxtul.max_depth=1 +lxtul.eval_depth=1` (the `+` adds keys to the empty `lxtul: {}`).
  Eval: the trainer logs `val/hop/<family>/acc_h<h>` and `.../acc_mean` at each `val_every` (the arm's own eval depth) and writes
  `hops.json` + a wandb table at the end (all `task.eval.depths`, which may exceed the trained `max_depth`).
  Standalone: `python -m lxtul.hops.eval --run <run dir> --depths 1,2,3,4,6,8,12,16` (needs `final.pt`).
  `PointerHead.mixed_many` is the new many-query form of `mixed_at`; the eval reads the argmax of the same mixture the NLL uses.
  `LXTUL_MAX_SLOTS=256` runs `tests/test_geometry.py` at the hop shapes. `cell_law` with a hop task raises (re-lay assumes stock).
- `tests/test_geometry.py` — causality and "loop is the only cross-span channel" on the
  forward (GPU). Run it after ANY change to masks, the write or the layout.

Run with the MORPH venv: `PYTHONPATH=/home/wolfe/parcae:$MORPH_ROOT
/home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python -m lxtul.bench`, under
`flock /home/wolfe/morph-scratch/gpu.lock`, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Gotcha: the bench cycles 16 batches, so its CE is memorization (trained batch 0.92 vs
unseen 6.51 after 40 steps, 2026-10-06). Read only tok/s and memory from it.
