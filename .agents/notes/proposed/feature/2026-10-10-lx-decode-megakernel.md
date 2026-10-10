# Agent Note: LXTUL decode, row-batched fused layer kernels and fused glue

Status: proposed

## Problem

`LxFusedDecoder` (`morph/inference/lx_fused_decode.py`, note
[2026-10-10-lxtul-fused-decode.md](2026-10-10-lxtul-fused-decode.md)) decoded LXTUL + pointer
at 3.10x / 2.34x the plain engine (exact weights, B = 1 / 8). The B = 8 margin over the
2.3x target was thin. Wolfe's call: make decode much faster with kernel fusion, up to a
per-layer megakernel, same function (fp32 reduction order only).

Measured before this work (`/home/wolfe/morph-scratch/megadec/probe.py`, the step
functions under `torch.profiler`, graph replay timed with CUDA events,
`lxtul-pointer-rerun/step_5000.pt`, exact weights):

| event | replay | kernels | where the time goes |
|---|---|---|---|
| token, B = 1 | 0.581 ms | 194 | 8 layers x 12 kernels = 317 us (39.6 us per layer); LM head 121 us; glue 140 us (45 before the layers, 95 after: readout, slot mask, log-softmax, pointer, argmax) |
| token, B = 8 | 0.876 ms | 195 | ternary MLP GEMVs 200 us (12.5 us per call vs 4.7 at B = 1), head 160 us, `front_gemm` 98 us, HC-projection GEMVs 97 us |
| slot, bucket 1 (4 query rows) | 2.389 ms | 690 | 44 layers x 48 us = 2.12 ms; glue 332 us (seed and register 106, per pass 31 x 6, prefix write 25) |
| slot, buckets 2 / 4 / 8 | 3.24 / 5.06 / 8.40 ms | 695-745 | bucket 8: ternary GEMVs 3.46 ms (40 us per call), `front_gemm` 1.61, `front_post` 1.33 |

Floors (weights as the exact mode stores them: per layer 11.89 MB = `Wqkv` 5.24 fp32,
`W_up` 2.10, two HC projections 1.57, MLP 2-bit codes 2.16, conv 0.80; LM head 201.4 MB fp32;
5090 HBM 1.79 TB/s; a graph-replayed kernel costs at least about 1 us):

| event | weight-read floor | launch floor | measured | dominated by |
|---|---|---|---|---|
| token, B = 1 | 300 MB -> 168 us (head alone 112 us) | 194 -> 0.19 ms | 0.581 ms | a chain of small kernels at 1.3-8 us each (latency) |
| token, B = 8 | 168 us | 0.19 ms | 0.876 ms | per-row weight re-reads (every GEMV kernel reads its weight once PER ROW) |
| slot, bucket 1 | 93 us (14 distinct layers) .. 292 us (44 reads) | 0.69 ms | 2.39 ms | launch count and per-kernel latency |
| slot, bucket 8 | same + ~11 GMAC of fp32 FMA | 0.69 ms | 8.40 ms | per-row weight re-reads (x32) and the 7x conv-history recompute |

## Proposal

Built and measured on branch `perf/lx-decode-megakernel` (2026-10-10).

`layers="fused"` (the default) runs every block on new kernels; `layers="engine"` keeps the
plain engine's kernels as the reference for the gates and the A/B arm (`lx_fused_eng` in
the bench). The plain engine is not changed.

**The layer: 12 kernels become 8** (`morph/kernels/triton/lx_layer_decode.py`):

| # | kernel | does | replaces |
|---|---|---|---|
| 1 | `hc_pre` | sums the HC-projection partials, the HC mapping (softmaxes, Cayley), `x_bar`, the sublayer RMSNorm -> `xn` | `_gemv_row_kernel` + `_hc_premap_fwd_kernel` |
| 2 | `lat_gemm` | the CCA projection of the CURRENT row only (K-split partials) | `_front_gemm_kernel` (7 rows per query row) |
| 3 | `lat_post` | partial sums, the convs over 6 cached latent rows + the current one, prologue, K/V store; writes the current row's latents into a per-site LATENT cache | `_front_post_kernel` |
| 4 | `lx_attn` | unchanged | |
| 5 | `up_post` | `W_up`, HC post, and the next HC projection's partial sums over the CTA's channel tile | `_gemv_row_kernel` (W_up) + `_hc_post_fwd_kernel` + `_gemv_row_kernel` (HC proj) |
| 6 | `hc_pre` | as 1, MLP side | |
| 7 | `gu_swiglu` | ternary gate_up, gate row o and up row o in one CTA -> `silu(g) * u` | `_ternary_gemv_kernel` (mode 1) |
| 8 | `dn_post` | ternary down, HC post with the next layer's injection term, the next layer's attention HC-projection partials | `_ternary_gemv_kernel` (mode 2) + `_hc_post_fwd_kernel` + `_gemv_row_kernel` |

Details that the measurements decided:

- **Latent cache.** A position's CCA projection depends only on its own normed input, so
  it is computed once and kept: token sites `[B, P, LQ+LK+Vh]`, loop sites a ring of
  `RW = M + 6` cells per cache row (`M + 6` so that an event never overwrites a cell it
  reads; the history commit copies go away). History index: `>= 0` a table row, `-1` the
  segment's zero row (no projection bias, so exactly the zero latent), `<= -2` row
  `-2 - idx` of this event.
- **Tap-major conv weight** (`conv_taps_major`): the engine's `[C, D, KC]` layout strides
  the input channel by KC; that gather was 6.5 of the site front end's 8.5 us
  (`lat_post` 9.7 -> 3.5 us at one row).
- **Rows per CTA** (`_rb`). `lat_gemm` and `gu_swiglu` from 4 rows, `up_post` from 16,
  take 4 rows per CTA with one explicit 2D accumulator per row: each weight tile loads (and
  decodes) once for 4 rows, and every row's sum is the one-row sum bit for bit. In-graph at
  the model's widths, one row per CTA -> four: `lat_gemm` 4.00 -> 2.27 us at 4 rows, 7.76
  -> 3.28 at 8, 16.0 -> 5.13 at 16; `gu_swiglu` 4.98 -> 3.69, 8.12 -> 6.57, 17.5 -> 8.91;
  `up_post` 10.9 -> 8.41 at 16 (slower below 16: its tile is 8 channels x 512 inputs).
  `dn_post` stays one row per CTA (four rows: 4.96 -> 12.5 us at 4 rows).
- `hc_pre` runs one warp (2.27 vs 2.98 us at 4 warps, in-graph).

**The LM head as int codes** (`morph/kernels/triton/lm_head_codes.py`, `head="auto"`). The
tied head's euclidean block is `IntNRowSTE` int6 codes x row scale; `head_codes_pack`
recomputes the codes with IntNRowSTE's own ops and uses them only when `code * scale`
equals the fp32 table bit for bit (0 of 37,761,792 entries differ on the LXTUL-pointer
and plain step-5000 checkpoints). 88.3 MB read instead of 201.4 MB; 55 / 68 / 135 us at
1 / 8 / 16 rows against 124 us (B = 1) and about 134 us of cuBLAS (B = 8).

**The glue** (`morph/kernels/triton/lx_glue_decode.py`):

| where | kernels | replaces |
|---|---|---|
| token entry | `tok_entry` (embedding row from the head table, hashed bigram row, span buffers, non-ctx terms, first carrier, RoPE rows), `tok_x0` (stacked ternary x0 GEMV, ctx-slice terms) | about 38 PyTorch kernels |
| prelude -> coda | `tok_mid` (input_norm per stream, the register's stream mean, coda carrier) | 4 |
| token exit | `readout_gemv`, `rmsnorm_rows`, head, `ptr_proj`, `ptr_attn`, `row_lse`, `ptr_scatter`, `mix_logp`, `argmax_rows` | 59 |
| each loop pass | `loop_entry` (winner reset, injection, first core layer's HC partials), `cell_norm` (rms cell norm + the router's layer-normed cell), `route` (partial scores on a (rank tile, cell) grid + the pick) | about 15 per pass |
| loop exit | `prefix_write` (only the winner's `W_prefix[k]` block read, zeros to the other cells) | 8 |

The entry reads the input embedding as the head table's rows. `m.embed(ids)` itself is
not batch-shape stable: against `lm_weight()[ids]` it differs on 3061 of 49169 Lorentz
rows at B = 1 (worst 2.98e-8), 2147 at B = 8, 0 at B = 16
(`/home/wolfe/morph-scratch/megadec/glue/check_embed_rows.py`), so the table row is one of
the values the reference itself produces.

The slot-loop glue covers the shipped geometry (rms cell norm, `M == Kp`, power-of-two
cell count and router rank: `slot_glue`); any other geometry keeps the PyTorch glue around
the same fused layers.

**Result** (`lab/perf/bench_lxtul_decode.py --step_profile --check_fused --tf_time 384
--gen_reps 3`, `lxtul-pointer-rerun/step_5000.pt` vs `plain-panel-norm-match/step_5000.pt`,
exact weights, 256 new tokens per row, tok/s of the median of 3 generation runs;
`/home/wolfe/morph-scratch/megadec/bench_s8.json`):

| B | lx_fused tok/s (3 runs) | lx_fused, engine layers | plain_graphed (3 runs) | lx_fused / plain |
|---|---|---|---|---|
| 1 | 1338 (1333, 1338, 1377) | 872 | 268 (268, 297, 265) | 4.99x |
| 8 | 4357 (4357, 4407, 4276) | 2414 | 999 (981, 999, 1001) | 4.36x |
| 16 | 5251 (5323, 5251, 4989) | 2896 | 1212 (1212, 1215, 1210) | 4.33x |

After the NaN fix below, the same bench without the engine-layer arm (`bench_s9.json`)
gave lx_fused 1580 / 4842 / 5988 tok/s against plain 309 / 1149 / 1387: 5.12x / 4.22x /
4.32x. Both arms ran about 15 % faster than in s8 (machine state); the ratio stayed in
4.2-5.1x across s6-s9.

Two earlier single-run benches of the same code gave 5.06x / 4.71x / 4.38x (s6) and
5.06x / 4.43x / 4.53x (s7); in s6 one engine-layer run lost half its speed to a stall (the
GPU also drives the desktop), which is why the bench now takes a median.

Graph replay per step (CUDA events, 50 replays), this work vs the engine layers:

| event | fused | kernels | engine layers | kernels |
|---|---|---|---|---|
| token, B = 1 | 0.369 ms | 74 | 0.665 ms | 190 |
| token, B = 8 | 0.469 ms | 74 | 1.003 ms | 191 |
| token, B = 16 | 0.689 ms | 74 | 1.521 ms | 191 |
| slot, bucket 1 / 2 / 4 | 1.56 / 1.93 / 2.63 ms | 438-439 | 2.71 / 3.69 / 5.66 ms | 686-691 |
| slot, bucket 8 / 16 | 3.89 / 6.80 ms | 440 | 9.68 / 17.52 ms | 741 |

`busy_frac` (device kernel time / wall) is now 0.86 / 0.91 / 0.86 at B = 1 / 8 / 16
against 0.97-0.98 for the engine layers: about a tenth of the wall is now host time
between graph replays, the next lever after the kernels.

**NaN rows.** The Z3 bounds work found that an all-NaN row gave the fused argmax no
index (next token out of range) and all-NaN router scores gave `PICK = M` (another slot's
cell, or past the buffer). `mix_logp`, `argmax_rows` and the router pick now follow
`torch.argmax` (the first NaN), and the router's relu keeps a NaN as `torch.relu` does
(`tl.maximum` turned it into 0); `tests/test_lx_glue_decode.py::check_nan` and four
mutants pin it.

## Alternatives considered

- **A persistent per-layer (or per-step) megakernel with grid barriers.** One launch per
  layer, phases separated by an atomic-counter barrier. Not built: the layer has 7 data
  dependencies across rows or channels (HC-projection reduction, the RMSNorm of `x_bar`,
  the K-split, the attention reading K/V written by other CTAs, the `W_up` input needing
  every head, the MLP input needing every gate_up output, the down input needing every
  hidden unit). Each needs a grid-wide barrier, which costs about what a graph kernel
  boundary costs (~1 us), the phases need different tilings, and spin barriers deadlock
  unless the grid stays co-resident. The measured waste was per-row re-reads, recompute,
  phase count and glue, which the 8-kernel layer and the glue kernels remove. After this
  work the bucket-1 slot step replays in 1.51 ms with 438 kernels; its layer kernels sum
  to 1249 us (`probe2_x5`). A barrier megakernel could save at most the ~1 us boundaries
  (about 0.3-0.4 ms per slot step); it is the next lever if one is wanted.
- **All rows inside one CTA** (3D `[rows, out, k]` accumulators, each weight tile read
  once per event). Built first (v1) and measured 3-5x SLOWER at 4-32 rows (layout
  conversions through shared memory, far fewer CTAs). The 4-rows-per-CTA form with
  explicit 2D accumulators is what works.
- **Several channel tiles per CTA in `dn_post`** (the input chunk loaded once for 2 or 4
  tiles of 8 channels). Measured: 5.23 -> 5.03 us at 4 rows, 14.45 -> 12.17 at 16, 3.80 ->
  4.62 at 1 row (`/home/wolfe/morph-scratch/megadec/rb/mb_dn_tpc.py`). Not taken.
- **2 rows per CTA** at 2 rows: slower on `gu_swiglu` (3.49 -> 3.91 us).
- **`tl.dot` for the multi-row GEMMs.** fp32 `tl.dot` runs at TF32 by default;
  `input_precision="ieee"` keeps fp32 but needs at least 16 rows (most events have 1-8).
- **One CTA per slot for the router** (the first `route`): 25 us per pass, reading all of
  `W_c` serially; the (rank tile, cell) grid takes 2.2-2.5 us.
- **bf16 weight copies.** A different function (the exact mode is the gate); the
  `lx_fused_bf16` arm already exists for that trade.
- **The same int-code head on the plain engine.** It would speed the plain engine
  equally (same table). Not built; the reported ratio is against the unchanged plain
  engine.

## Acceptance criteria

- Greedy tokens of the fused decoder equal the engine-layer decoder for 256 tokens on
  prompts 0 and 1 (exact mode); every row at B = 8 and 16 equals its B = 1 run.
- Teacher-forced log-prob rows against the engine-layer decoder on real text: worst
  difference reported.
- Tolerance argued from fp64 on the test fixture: the fused decoder's distance to a
  float64 eager forward is no larger than twice the engine-layer decoder's.
- New kernels: Z3 proofs of in-bounds access and coalescing
  (`/home/wolfe/morph-scratch/megadec/proofs/`); unit tests against fp64 references and
  sabotage tests (each fails on a broken copy).
- `tests/test_decode_engine_widths.py`, `tests/test_lx_decode_attn.py`,
  `tests/test_decode_front_kvrow.py`, `tests/test_lx_fused_decode.py` pass.
- Bench (`lab/perf/bench_lxtul_decode.py`, one invocation): lx_fused vs plain_graphed exact
  at B = 1, 8, 16, and the teacher-forced per-step timing.

## Risks

- Reduction order changes everywhere (GEMV trees, partial sums). The router's argmax over
  4 cells is a discrete pick: a near tie can flip and move a whole span. The gates compare
  greedy tokens over 256 tokens; a flip in deployment shows as a different span, not as a
  crash.
- The latent ring: a ring shorter than `M + 6` would overwrite a history cell in the same
  event. The Z3 ring proof (`/home/wolfe/morph-scratch/megadec/proofs/`) is the guard; no
  test sabotages the ring length. At `RW = M + 5` the only collision is cell M-1's write
  over the cell s*M-6 that cell 0 reads, inside the same `lat_post` launch: a race between
  CTAs, so a test could pass on a real break depending on the CTA schedule.
- The eager reference conv (`segment_causal_conv_reference`, `morph/model/attention.py`)
  fails with a shape error on a row shorter than the conv width (`F.pad(x[:, :, :-j],
  (j, 0))` returns j columns when j > S). Found while building the fp64 reference; not
  changed here (outside this work's files).
