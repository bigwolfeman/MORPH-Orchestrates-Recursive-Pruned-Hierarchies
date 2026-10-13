# Agent Note: Variable cell count on the ungraded fan

Status: proposed

Owner note for tracker item B. It supersedes the design half of
[`2026-10-05-lxtul-variable-cell-count.md`](2026-10-05-lxtul-variable-cell-count.md), which was
written for LXTUL with a router. That note keeps Wolfe's words and the CLAUDE.md reminder.

## Problem

Wolfe, 2026-10-05: "For lxtul cell count, we should try applying poisson sampling to the number
of cells to see if it generalizes. ... can I train at 4 and inference at 128 fine? a variable
cell count at inference would be huge."

There is now a second reason. The fan recipe step is 15.25 TFLOP (2026-10-11 ledger). Without
the span decoder it is about 11.1 against a 9.1 target. The loop (3.65), its glue (0.14) and the
gain hinge (0.91) scale with the cell count M. A training mean below 4 cuts them.

### What a cell is, to each part of the model

```
cell (s, i) of the fan, M = 4 today
|
+-- WRITER side (where the cell gets its identity)        INDEX-TIED
|   +-- row position slot_first[s] + i in the packed row   (prelude output e_i differs by offset)
|   +-- register query Q[i] and offset P_cell[i]           (TULSlotRegister)
|   +-- core position s*M + i on the compact axis          (RoPE, CCA conv k=4, window 256)
|
+-- LOOP (the map)
|   +-- relation: own slot all-to-all, earlier slots all  (slot_cell_relation, kernel CELLS=M)
|   +-- depth: one per slot, repeated over its M cells
|   +-- slot_compact capacity C_t * M, one plan per pass
|   +-- hinge, fixed-point term, cell pass norm           (per cell, M-agnostic)
|
+-- READERS
    +-- coda: token queries attend over earlier slots'    (set read, but softmax MASS grows with M)
    |   prefix positions, after W_prefix[i]               (W_prefix[i] INDEX-TIED)
    +-- span decoder: cross-attention over the M cells    (set read, normalised over M: invariant)
    +-- z = mean over cells                               (invariant)
    +-- epivol: log-det of the M x M Gram / (M - 1)       (set function, undefined at M = 1)
```

Three facts from this tree drive the design:

1. Every reader except W_prefix is already a set read. The index ties are on the WRITER side
   (offset, Q, P_cell, core position) and in W_prefix.
2. The cells are not exchangeable in the trained model. Measured on the fp32-carrier control
   (`ab-ctl-s1/step_5000.pt`, CPU read, 2026-10-12, `probe_cells.py` in this folder): the four
   `W_prefix[i]` sit 0.25 to 0.39 (relative Frobenius) from identity and 0.38 to 0.49 from each
   other; `P_cell` rows have norms 0.49 to 0.70 and pairwise cosines -0.44 to +0.18; the `Q`
   rows moved toward each other (cosines 0.57 to 0.78). The cell lens agrees: cell 0 is the copy
   cell (70 % of copied worth), cells 2 and 3 alone are worse than no cell
   (`lab/experiments/successes/2026-10-11-set-reader-cell-lens.md`). Cells have roles.
3. The coda and the core read cells through softmax attention over row or compact positions.
   M changes three things at once there: the number of cell keys (attention mass), the
   positional distances (the coda trained only up to L = 1280, the core only up to 256 cells),
   and the window coverage (the core window 256 covers all 64 slots at M = 4, 32 at M = 8).

### Where M is fixed today (master 9ae478bb)

| Tie | File:line | What is fixed |
|---|---|---|
| Config equality | `morph/model/tul.py:3140-3146` | `slot_cells == fan_k` |
| Write width | `morph/model/tul.py:3013`, `3148` | `prefix_k == fan_k` under `fan_mix: all` |
| Packed row | `morph/model/tul_layout.py:336-337`, `381`, `459-470` | `L_total = seq_len + prefix_k * max_slots`; each slot owns `prefix_k` positions; emit label at the last one |
| Data spec | `morph/training/tul_setup.py:284`, `329`; `morph/training/data.py:131`, `183` | one `prefix_k` for the run |
| Layout check | `morph/model/transformer.py:14525`, `5629`, `10736` | raise if `layout.prefix_k != tul.prefix_k` |
| Register seeds | `morph/model/tul.py:6438-6459` (`Q [M, d]`, `P_cell [M, d]`), `6478`, `6506` | one learned query and offset per cell index; build refuses `m < 2` (`6422`) |
| Per-cell write | `morph/model/tul.py:6843` (`W_prefix [prefix_k, d, d]`), `7059` and `7126` (`K = self.tul.prefix_k`) | one projection per index; positions from the CONFIG, not the layout |
| Cell layout view | `morph/model/transformer.py:6669-6692` | `_m_cells = cfg.tul.slot_cells`; `slot_index + arange(M)` |
| Register term | `morph/model/transformer.py:6718-6736` | `[B, S*M, C]` seed add |
| Depth per slot | `morph/model/transformer.py:7095-7118`; `_sample_slot_depths` `6367-6378` | draw repeated over M cells |
| Loop relation | `morph/model/transformer.py:7268-7330`, `slot_cell_relation` `1827-1950`; kernel `morph/kernels/triton/tg_strict_attention.py:584-590` (`CELLS` constexpr, `110`) | slot id = index // M |
| Compact plans | `morph/model/transformer.py:7608-7619`, `8432-8455` (`n = caps[t] * m_cells`); guard `8425` refuses `m_cells == 1` | capacity per pass |
| Hinge | `morph/model/transformer.py:8634` (`_slot_gain_penalty`) | runs on the pass's C_t * M rows; M-agnostic per graph |
| Fan exit seam | `morph/model/transformer.py:14922-14950` | `_m = tc.slot_cells`; `_reg_cells` reshape |
| Diversity term | `morph/model/transformer.py:15051-15162`; `morph/model/tul_fan.py:553` (epi, per-cell mean), `665` (vol `/ (m - 1)`) | needs M >= 2 |
| Span decoder | `morph/model/transformer.py:15331-15339`; `morph/model/tul_spandec.py:323-372` | shape-agnostic, no per-index weight, no mask |
| 1:1 write call | `morph/model/transformer.py:15465-15490` | `prefix_project(cells=...)` |
| Row caps | `morph/model/transformer.py:12345-12410` (`_spandec_row_cap`, `_lmhead_row_cap`) | read `layout.prefix_k`, so already per forward |
| `> 1` branches | `morph/model/transformer.py:6680`, `7103`, `7268`, `7400`, `8425`, `14944`, `17074`, `17364` | key the register path on the forward's M; at M = 1 they fall to the single-cell path |
| Instruments | `morph/model/transformer.py:9724` (`_tul_fan_oracle`), `16547` (`_tul_prefix_cells`), `17034-17074` (`tul_slot_state_probe`), `17398` (`_tul_layer_passes`) | config M |
| Graph step | `morph/training/graph_step.py:205-245` (`StaticBatch` shape check, `prefix_k` host field), variants doc `17-27` | one graph per (kind, hinge pass); fixed shapes |
| Compile | `morph/training/train.py:2801-2805` (`cache_size_limit` 256), `2145-2230` (`warmup_compile_all_shapes`) | one shape set |
| Positions | `morph/model/attention.py:171-208` (`CoPEEmbedding`, cache of `max_seq_len` 4096), `997` | coda `L(M) = 1024 + 64 M` must stay <= 4096, so M <= 48 |
| Generators | `morph/inference/tul_generate.py:93`, `199`, `228`; `tul_generate_cached.py:156-160` | `spec.prefix_k` per run; cached path needs `fan_k == slot_cells == prefix_k` and refuses `hc_carrier_bf16` |

Not tied: `TULFanMix` at `fan_mix: all` (no parameter), `FanReservoir` (`tul_fan.py:441`), the
cell pass norm, the fixed-point term, the pointer head (token order only).

## Proposal

### 1. M is a per-forward host integer: `layout.prefix_k`

`SlotLayout` already carries `prefix_k` as a host field, and `StaticBatch` already keys on it.
Under the new key the model reads the forward's cell count from the layout and the config
`fan_k` becomes the TABLE SIZE (4). `1 <= M <= fan_k` at train. `M > fan_k` is eval only.

Every `_m_cells > 1` / `_m > 1` branch in the table above keys on the BUILD instead
(`self.tul_register is not None`), so an M = 1 forward stays on the register path with a cell
axis of size 1: `slot_cell_relation(S, 1)` is plain causal, the kernel runs `CELLS = 1`,
`slot_compact` keeps working, and the diversity term is skipped (a Python branch per graph).

### 2. The row layout follows M: pack once at K = 4, re-lay out at M

`relayout_cells(row, m, spec_ref)` in `tul_layout.py`: a pure host function from a row packed at
the reference `prefix_k = 4` to the same tokens, spans, labels and boundaries with `m` positions
per slot. At `m = 4` it returns the input arrays unchanged. Both arms and every eval M therefore
read the SAME tokens: the training stream is paired across arms, and the eval CE(M) curve is
paired row by row.

Row length, a fixed shape per M:

- `M >= 4`: `L(M) = L_ref + (M - 4) * max_slots` (1280, 1344, ..., M = 8: 1536, M = 32: 3072).
- `M < 4`: `L(M) = L_ref - (4 - M) * S_lo`, where `S_lo` is the fewest slots a reference row
  can hold. Proof sketch for the builder: a row that did not hit `max_slots` is full to within
  `prefix_k` positions, so `n_tok >= L_ref - 4 - 4 n`; every span and the open tail hold at most
  `span_cap = 32` tokens, so `n_tok <= 32 (n + 1)`. Then `36 n >= 1244`, `n >= 35`. So
  `S_lo = 35` at these shapes (gate truncation off), and `L(3) = 1245`, `L(2) = 1210`,
  `L(1) = 1175`. The builder computes `S_lo` from the spec and the relayout ASSERTS the fit per
  row (a row that does not fit raises; it never drops a token).

The extra positions at `M < 4` are tail pads. The row caps (`_spandec_row_cap`,
`_lmhead_row_cap`) already read `layout.prefix_k` and stay valid proofs at `L(M)` (they need
`L(M) - M n >= L_ref - 4 n`, which holds for `n >= S_lo`).

### 3. Seeding: the trained table, first M rows, and an extension rule for M > 4

No new parameter. Cell `i < M <= 4` uses `Q[i]`, `P_cell[i]`, `W_prefix[i]` and its own row
offset, exactly as today. This reproduces master's forward at `M = 4` by construction, so the new
code can be byte-gated against master.

For `M > 4` (eval only) a cell `i >= 4` needs an identity. A model attribute set by probes
(`model.cell_extend`, refused in training mode, default `refuse`) picks the rule:

- `cyclic`: role `i mod 4` (its `Q`, `P_cell`, `W_prefix`) at its own row offset and core
  position. Every parameter it reads is trained. Recommended default for the sweep.
- `fresh`: `Q` drawn from a private generator at the init law, `P_cell` 0, `W_prefix` identity.
  The cell at init. Read at M = 8 only, as the contrast.

Why not more rows in the table: extra rows of `W_prefix` / `Q` / `P_cell` that a fixed-4 run
never uses still get weight decay and change the per-tensor optimizer state of the trained rows,
so the byte gate would fail; and at train M <= 4 they would never train anyway (section 4).

### 4. Train-time M: per batch, one graph per (kind, hinge pass, M)

M is drawn per BATCH on the host, counter-based so a resume replays it:
`M_b = law(np.random.default_rng([seed, 0xCE11, batch_index]))`. Law: `Poisson(mean)` clamped to
`[min, max]`.

Recommended law for the first arm: **Poisson(3.5) clamped to [1, 4]**. Masses 0.136 / 0.185 /
0.216 / 0.463 for M = 1..4, mean 3.006. Why max 4: the graph pool is sized by the largest graph,
and today's pool leaves about 1 GiB free after eval with kept graphs (0.34 GiB with the S1/S2
stream keys on; tracker C entries). Loop activations scale with M, so M = 5 or 6 very likely
does not fit beside the kept-graph eval on the 5090. With max 4 the largest graph IS today's
graph. Training above 4 belongs on the DGX Spark or after a memory cut (Wolfe's call, below).

FLOPs per step, analytic from the 2026-10-11 ledger with the T1/T2 cuts applied (decoder
3.69 -> 2.69, head -0.47; total 13.79 at M = 4). Per cell: loop + glue + hinge 1.175; prelude +
coda 0.122 per cell below 4 (35 slot positions per row), 0.223 per cell above 4 (64 per row).
The hinge figure is the active-step value, an upper bound.

| M | loop + glue + hinge | prelude + coda change | step TFLOP | non-decoder TFLOP |
|---|---|---|---|---|
| 1 | 1.18 | -0.37 | 9.89 | 7.20 |
| 2 | 2.35 | -0.24 | 11.19 | 8.50 |
| 3 | 3.53 | -0.12 | 12.49 | 9.80 |
| 4 | 4.70 | 0 | 13.79 | 11.10 |
| 8 (eval) | 9.40 | +0.89 | 19.38 | n/a |

Expected training step under the recommended law: **12.50 TFLOP** (-9.4 %), non-decoder
**9.81** (target 9.1). These are estimates.

Measured (G5, 2026-10-13, builder cellcount): flame_flops.py, eager, one counted step, fixed
slot depth 6, `slot_compact` off, `split_wgrad` off (the custom op is not counted). Step TFLOP
at M = 1, 2, 3, 4: **10.20, 11.57, 13.01, 14.38**; ratio to M = 4: 0.710, 0.805, 0.905 (the
estimate gave 0.717, 0.811, 0.906). Each point is +3.2 % to +4.3 % over the table, inside the
10 % gate. M = 4 under the variable-count build equals the fixed fan to the digit (14.377).
The table's absolute level is for a different depth and compaction setting. The ratios hold.

Options weighed (details in Alternatives):

| Scheme | Loop FLOPs fall | Prelude/coda fall | Graphs | Kernel change | Memory |
|---|---|---|---|---|---|
| Per batch, relayout (chosen) | yes, exact | yes | 16 x |M set| = 64 | none | today's (max graph = M 4) |
| Fixed M_max layout, masked cells | only with a new capacity proof | no, rises at M_max > 4 | 16 | key mask in MODE_CELLS | at M_max |
| Per slot, stratified count | yes, bound looser than the mean | only with a ragged layout | 16 | slot id is not index // M | at M_max |

Graph count and compile: 64 recordings (2 kinds x 8 hinge passes x 4 M) in one pool; the
graphs never run together, so they share blocks. Every shape-dependent compiled frame exists
per M: `cache_size_limit` 256 -> 1024 for `slot_compact: gather`, and the warm-up runs every
(M, pass) pair. Startup recording and compile time grow about 4x; gate G4 measures them.

### 5. The readers under a variable set

- **Coda:** no mask. The layout has exactly M positions per slot, so every key is a real cell.
  The coda's position layout changes with M (token positions shift by the inserted cells). At
  M <= 4 every distance is inside the trained range. At M > 4 the row grows past 1280 and the
  compact core past 256 cells. v1 does not separate that stretch from the new cells' content
  (see Risks); the M <= 4 range is clean.
- **Span decoder:** `_SpanDecCross` takes `mem [B, S, M, C]`; its softmax is normalised over the
  M cells, so it is a set read at any M. `z` is the mean over the M present cells. The packed
  decoder's cap reads `layout.prefix_k` and stays a proof.
- **Diversity term:** epi averages per cell, vol divides by `M - 1`; both are scale-normalised
  over M >= 2. At M = 1 the term is skipped and its stats are absent from that step's log.
- **Instruments:** the oracle loops `range(M)`; `fan/stream_ce_k{i}` keys exist only for
  `i < M`. Every step logs `tul/cell_count` so per-M series can be split.

### 6. Out of v1, on purpose

- **M above 48.** `L(M) = 1024 + 64 M` passes the CoPE cache (4096) at M = 49. "Train at 4,
  infer at 128" needs a read whose layout does not grow with M: a fixed number P of coda
  positions per slot, each a learned set-attention pool over the M cells (Perceiver style),
  and slot-level core positions. Both are new modules and break the byte gate. Build them only
  if v1 shows CE falling from M = 4 to 8 or 16.
- **Log-count key bias** (`-log(M / 4)` on cell keys in the coda): keeps a slot's attention
  mass constant as M grows and is exactly 0 at M = 4. It needs a per-key bias in the strict
  kernels and has no clean meaning in the compressed branches, where a block mixes tokens and
  cells.

### 7. Build brief (for a sonnet builder)

**Keys** (all default off; off is byte-identical; every key logged to wandb through the `tul:`
manifest):

| Key | Default | Meaning |
|---|---|---|
| `tul.cell_count` | `fixed` | `fixed`: M = `fan_k` on every batch, today's path. `poisson`: per-batch draw. `categorical` (Wolfe, 2026-10-12): per-batch draw from `cell_count_probs`. |
| `tul.cell_count_mean` | `3.5` | Poisson mean (read only under `poisson`) |
| `tul.cell_count_min` / `_max` | `1` / `0` (= `fan_k`) | clamp; `_max <= fan_k` enforced |
| `tul.cell_count_probs` | none | `categorical` only: P(M = m) for m = 1..`fan_k`, summing to 1 within 1e-6 or it raises |
| `tul.cell_count_subset` | `prefix` | `prefix`: the cells are roles 0..M-1. `random`: a uniformly random ascending subset of M roles per batch (every role trains), carried as `layout.cell_roles`, a device tensor gathered in the graph |
| `tul.cell_count_eval` | `0` (= `fan_k`) | M of the trainer's own val pass |

Refusals (`TULConfig`, near the top of `__post_init__`): `cell_count != fixed` with `fan_k == 0`,
`fan_mix != all`, `fan_loop_select != off`, `fan_route != none`, `fan_opf`, `prefix_per_cell > 1`,
`prefix_source != exit`, `code_enum_k > 1`, `fan_lineage != off`, `fan_history_streams > 0`,
`fan_trigger_every_pass`, `xhc_streams > 0`, `loop_denoise`, `latent_pre_target != off`,
`vq_codes > 0`, `center_exit`, `cond_layers > 0`, `ditto_rows > 0` (untested pairing), and
`model.slot_gain_rows > 0` (until checked). Each refusal names why (most are per-index
modules: `FanRouter` bias `[M]`, `FanOPF` factors, history split).

**Files:**

1. `morph/model/tul_layout.py`: `relayout_cells(arrays, m, spec_ref)`, `cell_l_total(spec_ref, m)`,
   `cell_slots_lower_bound(spec_ref, span_cap)`. Pure numpy.
2. `morph/training/data.py`: after `pack_tul_batch`, draw `M_b` (counter-based) and relayout;
   the val iterator uses `cell_count_eval`. `morph/training/tul_setup.py`: `KNOWN_TUL_KEYS`, the
   law in the manifest and in `ckpt["tul"]`.
3. `morph/model/tul.py`: `TULSlotRegister.forward(xn, layout, m)` slices or cyclic-indexes `Q`,
   `P_cell`; `TULSlots.prefix_project` / `prefix_positions` take `K = layout.prefix_k` and index
   `W_prefix` by role; the `fresh` rule's tensors are built on the fly under `no_grad` from a
   private generator, never as parameters.
4. `morph/model/transformer.py`: `_m_cells = layout.prefix_k` under the key; the eight `> 1`
   branches key on `self.tul_register is not None`; the layout check (14525, 5629, 10736) accepts
   `1 <= layout.prefix_k <= fan_k` in training and any M <= 48 at eval with `cell_extend !=
   refuse`; `_slot_compact_guard` accepts M = 1 on a register model; diversity term skipped at
   M = 1; instruments read M from the layout.
5. `morph/training/graph_step.py` + `train.py`: variant = (hinge pass, M); one `StaticBatch` per
   M; `warmup_compile_all_shapes` covers every (M, pass); `cache_size_limit` scaled by the M-set
   size; the step reads M from the batch before `set_variant`.
6. `lab/divergence/core_depth_sweep.py`: `--cells 1,2,3,4,8,16,32`, `--cell-extend
   cyclic|fresh`; rows packed once at K = 4 and relaid per M; per-row sums kept so CE(M) - CE(4)
   gets a paired bootstrap CI; at M = 4 it must reproduce the filed sweep's CE@6 to 1e-4.
7. Config `morph/configs/nowta_pointer_fast_spdcells_speed_bf16c_cellp3.yaml`: speed base +
   `tul.cell_count: poisson`, mean 3.5, clamp [1, 4]; wandb name `fan-cellp3`.

**Tests** (each with its sabotage, run with `OMP_NUM_THREADS=4` and capped torch threads):

| Test | Asserts | Sabotage that must fail it |
|---|---|---|
| `test_relayout_identity` | `relayout_cells(row, 4)` equals the packer's arrays (`np.array_equal`, every field) on 2,000 real OWT rows | shift `slot_first` by 1 |
| `test_relayout_content` | for every M in 1..8 and 16, 32: token ids, labels and `bag_id` at token positions equal the M = 4 row's; each slot's cells are at `slot_first' .. + M - 1`; the emit label sits on the last cell; adversarial rows (all spans 32, all spans 4, 64 slots) fit `L(M)` | `S_lo = 36`: the adversarial row overflows and the test sees the raise |
| `test_cellcount_off_bytes` | key off: loss, logits and every grad `torch.equal` to master on the tiny fan fixture | read `_m_cells` from the layout with a layout at M = 3 |
| `test_cellcount_fixed4` | key on, every batch M = 4: forward and backward `torch.equal` to key off | register uses `Q[1:5]` |
| `test_cellcount_prefix_model` | M = 2 on the 4-cell model equals a `fan_k = 2` model whose `Q`, `P_cell`, `W_prefix` are the first 2 rows (loss `torch.equal`) | cyclic indexing at train |
| `test_cellcount_m1` | M = 1 runs the register path (relation, compact, kernel CELLS 1), no diversity term, finite loss and grads | compute vol at M = 1: NaN caught |
| `test_cell_extend` | M = 8 `cyclic` and `fresh` run at eval; training mode refuses M > `fan_k`; default `refuse` raises | drop the training-mode refusal |
| `test_cellcount_draw` | counter-based draw reproduces after a simulated resume; 100k-draw frequencies inside 3 sigma of the clamped law | seed the draw from the global numpy stream |
| `test_graph_step` (extend) | for each M in {1, 2, 3, 4}: replay equals eager on that batch; the variant sequence equals eager's | one `StaticBatch` for all M |

**Gates, in order:**

- G1: the tests above, green, with the command and counts in the PR.
- G2: 45-step graph-step byte gate on `nowta_pointer_fast_spdcells_speed_bf16c`: key off
  `FINAL_SHA256` equals master's (the slot-loop glue precedent). This is what lets the existing
  bf16c seed 1 and 2 checkpoints stand as the control.
- G3: memory: graph pool after a kept-graph eval at the M set {1, 2, 3, 4} within +0.5 GiB of
  the control pool; report free memory after eval.
- G4: startup (compile + 64 recordings) seconds, reported; no recompile in 200 steady steps
  (`fail_on_recompile` stance).
- G5: flame2 FLOP ledger (`/home/wolfe/morph-scratch/perf/flame2/flops_only.sh`) per M:
  each M within 10 % of the table in section 4, or the table is corrected in this note.
- G6: `lab/perf/bench_step.py` steady-state ms per M on a quiet desktop.

## Alternatives considered

- **Fixed M_max layout with masked cells (cell dropout).** One graph per hinge pass. The coda
  side is cheap (an absent cell's positions get the dump-bin `bag_id`, so the strict coda
  relation already hides them), but the core needs a key mask in `MODE_CELLS`, the prelude and
  coda pay for M_max positions on every step, and loop FLOPs fall only with a new capacity
  proof for active cells per pass. Rejected: it costs a kernel change and gives up the
  prelude/coda saving.
- **Per-slot M (stratified per row, like the depth draw).** One graph, and per-span M is the
  natural test-time knob ("more cells for a hard span"). The sum of a systematic sample is
  nearly fixed per row, but the active-cells bound at pass t is the sum of the C_t largest
  counts, much looser than the mean. A ragged layout breaks the kernel's `slot = index // M`.
  Rejected for v1; the right v2 if v1 shows the knob works.
- **More table rows (M_tab > 4) trained at M up to 6.** Trains rows 4 and 5, so M = 6 at eval is
  in distribution. Rejected for the 5090: memory (section 4) and the byte gate (extra rows move
  the optimizer state). Viable on the Spark.
- **Random role subsets at train** (M sorted roles drawn from {0..3}, placed at offsets
  0..M-1). Trains every role equally and decouples role from offset. First not chosen (it adds
  a draw, and at M = 1 it would often drop cell 0, the copy cell). Built as
  `tul.cell_count_subset: random` on Wolfe's 2026-10-12 call that every cell role must train
  equally; `prefix` stays the default. The role indices ride in the batch as a device tensor, so
  one recording per (kind, pass, M) still serves every subset.
- **Index-free exchangeable cells** (shared query and write, identity from a frozen random code
  `r_i ~ N(0, I)` through a learned map, fresh codes per forward). In distribution at any M,
  because training already sees new codes. Rejected for v1: no byte gate, it discards the roles
  the trained model uses (fact 2 above), and the nearest precedent (`fan_seed_noise`,
  `lab/experiments/failures/2026-09-21-lxtul-fan4-all-noise.md`) collapsed the streams to cos
  0.99. The v2 seeding if the extension rules fail.
- **Pooled coda read** (P fixed positions, set attention over M cells). The only route to
  M = 128. Section 6.
- **Learned or halting cell count.** Refused by Wolfe's standing rules (no learned-sigma arms,
  no depth curricula). A sampled count is the analogue of the sampled depth that already works.

## Acceptance criteria

- The keys, refusals, files and tests of section 7 are built; G1 to G6 pass and are reported
  with commands and numbers.
- Key off and key on at fixed M = 4 are byte-identical to master (G2 and `test_cellcount_fixed4`).
- The measured FLOP per M replaces the estimates in section 4.
- The prereg (`lab/experiments/planned/2026-10-12-fan-variable-cell-count.md`) is committed
  before the first training step of the Poisson arm, and the M sweep is filed against it.
- The CLAUDE.md reminder stays until this note and the 2026-10-05 note leave `proposed/`.

## Risks

- **Roles do not extend.** The trained cells have roles (fact 2). A cell at M > 4 is a copy of a
  role or an untrained cell. CE above M = 4 may be flat or worse for that reason alone, and that
  would say nothing about an exchangeable design.
- **The coda's read is not M-invariant.** More cell keys pull softmax mass away from a token's
  own span. A dilution loss at large M is likely; the log-count bias is the measured fix, not in
  v1.
- **Positions confound M > 4.** The coda trained to L = 1280 and the core to 256 cells; at M = 8
  both stretch, and the cell keys double. v1 cannot tell "longer layout" or "diluted attention"
  from "the new cells carry nothing". A clean separation needs positions decoupled from the
  row index (an explicit position id in the strict kernels), which is v2 work. Below M = 4 every
  distance is in range, so the downward curve is clean.
- **Train at M < 4 costs CE at M = 4.** Rows 2 and 3 train on 68 % and 46 % of steps. If the
  M = 4 cost is above the 0.003 line, the FLOP lever fails even if the knob works downward.
- **Graph and compile count.** 64 recordings and about 4x the compiled frames. If startup or the
  dynamo cache becomes the limit, a smaller M set ({2, 3, 4}) is the fallback, at a weaker M = 1
  reading.
- **Memory is the binding limit for training M > 4** on the 5090. The note does not solve it.
- **Decode.** The cached generator reads `spec.prefix_k` per run and refuses
  `hc_carrier_bf16`; a per-call M there is follow-up work, so v1 measures eval-forward tok/s, not
  decode tok/s.

## Status 2026-10-13: v1 measured at 5k

v1 is merged (33113fb6) and run, 2 seeds per draw: `lab/experiments/successes/2026-10-13-fan-variable-cell-count-v1.md`.
The Poisson draw holds CE@6 (+0.0002) and K1-K6 (95.5 %) at M = 4 for 0.921x the step wall, and its
model loses only 0.005 nats at M = 1 (the fixed-four model loses 0.135). The gentle draw holds too,
and runs free at M = 3. No arm gains above M = 4, cyclic or fresh. So v1 is a FLOP lever and a
downward inference knob; the upward question needs v2 (exchangeable cells, pooled coda read).

