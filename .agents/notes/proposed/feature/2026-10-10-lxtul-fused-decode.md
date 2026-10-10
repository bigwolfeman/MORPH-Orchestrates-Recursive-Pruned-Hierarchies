# Agent Note: LXTUL fused, batched decoder on the plain engine's kernels

Status: proposed

## Problem

LXTUL's point is "think once per span, decode cheap" (phase-2 tracker items C1, C1b,
`.agents/notes/proposed/process/2026-10-10-phase-2-tracker.md`). A token pays the prelude
and the coda (4 + 4 block applications); the looped core (6 layers x 6 passes, 4 cells)
runs once per span. Plain MORPH pays 4 + 6 x 6 + 4 = 44 block applications per token. The
target is LXTUL decode at least 2.3x plain MORPH.

Measured 2026-10-10 (`lab/perf/bench_lxtul_decode.py`, 128-token prompts, 256 greedy
tokens; `/home/wolfe/morph-scratch/plaindecode/bench_plaingraphed.json`):

| decoder | batch 1 tok/s | batch 8 tok/s | per event |
|---|---|---|---|
| plain `StaticDecodeEngine`, exact weights | 302.6 | 1137.5 | 617 kernels, about 2.2 ms per token |
| LXTUL graphed (`tul_generate_graphed.py`) | 295.3 | 260.6 (rows one after another) | token 1.94 ms / 835 kernels; slot 11.7 ms / 4056 kernels |

The LXTUL token step runs 8 block applications in 1.94 ms (about 240 us each); the plain
engine runs one in about 50 us. The saving is lost in kernel efficiency (generic PyTorch
ops under CUDA graphs) and in batching (the graphed decoder holds one row), not in the
architecture.

## Proposal

A new decoder, `morph/inference/lx_fused_decode.py` (`LxFusedDecoder`), that computes the
same function as `tul_generate_graphed.TulGraphDecoder` (same refusals:
`tul_generate_cached._check_supported`) from fixed-capacity buffers for B rows, with two
kinds of CUDA graph: one TOKEN graph over all B rows, and one SLOT graph per bucket size
(1, 2, 4, 8, 16) over the rows that closed a span at this step.

### Op mapping

(a) = an engine kernel as is; (b) = an engine kernel with a new shape, a new index or a
new mask; (c) = a new kernel; (t) = the model's own PyTorch module or ops inside the graph.

Token step, every row, every step:

| op | reference (graphed decoder) | fused path |
|---|---|---|
| embedding row, bigram hash, all 14 layers' injection terms | `m.embed`, `get_bigram`, `_build_injection_term` per layer | (a) `W_lm` row gather, the engine's hash, one stacked x0 GEMV (`ternary_gemv_rs` or `small_gemv`) and the engine's term arithmetic |
| HC pre-map + `norm_attn` | `HyperConnectionResidual` | (a) `small_gemv` + `_hc_premap_fwd_kernel` with the norm folded (`HAS_NOUT`), output into the site's conv-history staging row |
| CCA projection, segment conv, value shift, prologue, K/V store | `_cca_rows` (`segment_causal_conv`, `fused_cca_prologue`) | (a) `decode_front`. The conv history is the engine's x-history: a span's earlier rows, and a ZERO row for every position before the span's start. A zero row gives exactly the zero latent that the segment cut gives (the projections and convs have no bias). K/V go to entry j (span offset) of the row's prelude cache, entry `Cc + j` of its coda cache. |
| window branch + slot branch + sink + gate + alpha | `_branches` (SDPA + fp32 einsum softmax) and `_gate_combine_up` | (c) `lx_attn`: one CTA per (row, head), two online softmaxes over the row's key buffer, the sink as the slot softmax's initial state, the gate2 GEMV and the combine of `decode_attn` |
| `W_up` | `cca.W_up` | (a) `small_gemv` |
| HC post | `_hc_post_fwd_kernel` | (a), with the next layer's term folded (as the engine) |
| MLP | `MortarLinear` dense ternary | (a) `ternary_gemv` x 2 (norm and SwiGLU folded) |
| `input_norm`, readout (stream mean, `lm_mixer`, `final_norm`), LM head | modules | (a) `rmsnorm_rows`, the engine's mixer GEMV, `bf16_gemv` (fp32 weights in the exact mode) |
| slot-masked log-softmax, pointer mixture | `_logp`, `pointer_keys`, `pointer_mix` | (t) batched over rows: the same formulas on `[B, ...]` tensors |
| history commit | (the graphed decoder keeps raw latents) | (a) `ring_commit` with "position" = j + 1, so the staging row lands at row j |

Slot step, bucket rows only (each bucket row carries its row id):

| op | reference | fused path |
|---|---|---|
| seed, bigram mean, register term, router, reset, cell norm, winner write | model modules (`W_sent`, `_e_slot_term`, `TULSlotRegister`, `FanRouter`, `reset_to_winner`, `prefix_write`) | (t) the same modules on `[Bs, ...]` tensors, rows gathered by id; the cell norm through (a) `rmsnorm_rows` |
| prelude of the M cells | `_site_rows(fresh=True)` | the token kernels at Bs x M rows. History: the cells only (their own segment). K/V go to entry `P + i` of the row's prelude cache. |
| 36 loop sites | `_site_rows(cells=M)` | the same kernels at Bs x M rows. Conv history on the compact cell axis crosses slots: per site, a 2-slot history ring per row plus a staging block for the current slot, committed at the end of the step. K/V at entry `s*M + i` of the row's (pass, layer) cache. `lx_attn` loop mode: keys `[0, (s+1)M)`, the window by compact index (later siblings included). |
| coda of the Kp cells | `_cell_rows` (self only) | the same kernels; `lx_attn` self-only mode. The last coda layer's attention and MLP are skipped when nothing reads the cells' output (emit source "token"): only that layer's K/V are read later. |

(b), one engine kernel change: `decode_front` gets an optional per-row K/V row index
(`kv_row`), so a bucket row writes into its own row's cache. Default `None` is the old
kernel unchanged (a constexpr branch).

### Strict-geometry attention in `lx_attn`

Per query row the kernel reads `[row, lo1, hi1, lo2, hi2, qidx1, qidx2, qpos]`:

* range 1 `[lo1, hi1)` = slot columns (cells): every key enters the slot branch; a key
  enters the window branch iff `0 < |dist| < window_size`, `dist = qidx1 - k` (index) or
  `qpos - kpos[row, k]` (the coda token reading earlier cells by absolute position).
* range 2 `[lo2, hi2)` = token columns: window branch only, `dist = qidx2 - k`.

| site | range 1 | range 2 |
|---|---|---|
| prelude token | empty (slot branch = sink only = exactly 0) | the span's earlier tokens |
| prelude cell i | cells 0..i | the span's tokens |
| loop cell (s, i) | every cell of slots 0..s | empty |
| coda token | every written cell (window by absolute position) | the span's earlier tokens |
| coda cell | itself | empty |

### Batching

Rows are independent and feed one token per step. A row whose token closes its span
also runs the slot step in the same step; the host gathers those rows into the smallest
bucket that holds them and pads it with a dummy row (row B, never read). Expected cost at
B rows: one token graph per step plus a bucket graph on about `1 - (1 - 1/18.4)^B` of the
steps. Rows with different prompt lengths are allowed: every row runs the same number of
steps and a row's extra tokens past its budget are discarded.

### Capacity

Span `P = rule.span_cap`, slots `S = spec.max_slots`, cells `Cc = prefix_k * S`, pointer
entries = tokens per row. The loop caches dominate: 36 sites x (B+1) x 8 heads x S*M x 64
x 2 x 4 bytes (976 MB at B = 16, S = 97).

### Buffer and batching layout (the contract between the kernels and the decoder)

Shapes at `lxtul-pointer-rerun/step_5000.pt`: C = 1024, H = 8 query heads, Hkv = 4,
D = 64, LQ = H*D = 512, LK = Hkv*D = 256, Vh = 128, GH = 256 (gate hidden), gate2 out
2H = 16, window W = 256, n streams N = 4, P = 32, M = Kp = 4, depth 6, 4 + 6 + 4 layers.
Test fixture (kernel-legal widths): C = 256, H = 4, Hkv = 2, D = 32, GH = 64, W = 16.

* Cache rows: R = B + 1. Row B is the dummy row that pads a slot bucket; nothing reads
  it. Every K/V cache is `[R, H, N_site, D]` fp32, K after RoPE, both GQA-expanded to H
  heads (what `decode_front` stores).
* Query rows: token step B' = B (query row b = cache row b). Slot step with bucket nb:
  B' = nb * M, query row `q*M + i` = cell i of bucket row q, whose cache row is
  `r_q` (`kv_row`, or `meta[:, 0]` in `lx_attn`).
* `N_site`: prelude `P + M` (tokens at entry j = span offset, cells at `P + i`); coda
  `Cc + P` (cells at `s*Kp + k`, tokens at `Cc + j`); loop `S * M` (cell at `s*M + i`).
* Conv history for `decode_front` (`x_hist`, `x_off`): token sites `[B, P + 2, C]` per
  site (rows 0..P-1 the span's normed inputs, row P always zero, row P+1 the staging row
  the HC pre-map writes); `x_off[b] = [j-6 .. j-1 -> row or P if negative, P+1]`;
  after the step `ring_commit` with "position" j + 1 moves the staging row to row j.
  Cell sites: one flat table `[nb*M + 1, C]` (last row zero), passed EXPANDED to
  `[B', rows, C]` (stride 0), `x_off` absolute rows. Loop sites: flat table per site
  `[R*RW + Bmax*M + 1, C]`, RW = max(8, M) history cells per cache row (cell c at
  `r*RW + c % RW`), then the staging block, then a zero row.

### Kernel specs for the kernel agent

**K1 (new, class c): `lx_attn`**, file `morph/kernels/triton/lx_decode_attn.py`.

```
lx_attn(q, k_cache, v_cache, meta, kpos, gateh, w_gate2, xres, sink, alpha,
        window: int, scale: float) -> out
  q        [B', H, D]  fp32, last dim contiguous (decode_front's q_out)
  k_cache  [R, H, N, D] fp32, last dim contiguous; v_cache the same shape
  meta     [B', 8]  int32: row, lo1, hi1, lo2, hi2, qidx1, qidx2, qpos
  kpos     [R, N] int32 or None (None = HAS_KPOS False, a constexpr)
  gateh    [B', GH] fp32: gate[0](x), before SiLU (decode_front's gateh)
  w_gate2  [2H, GH] fp32 or bf16: cca.gate[2].weight
  xres     [B', H*D] fp32: the raw q_lat row (decode_front's xres)
  sink     [H] fp32: cca.sink_logits;  alpha [H] fp32: cca.alpha flattened
  window   int (constexpr): cca.window_size;  scale = D ** -0.5
  out      [B', H*D] fp32, head h at columns h*D .. h*D + D - 1 (W_up-ready)
```

Per (query row b', head h), with `row = meta[b', 0]` and `s_k = scale * <q[b',h], K[row,h,k]>`:

* slot branch: softmax over `{s_k : lo1 <= k < hi1}` plus the logit `sink[h]`, whose
  value is the zero vector; `oc = sum_k p_k V[row,h,k]`. Empty range: `oc = 0` exactly.
* window branch: keys `k in [lo1, hi1)` with `0 < |d1| < window`, `d1 = qidx1 - k`
  (HAS_KPOS False) or `qpos - kpos[row, k]` (True); and keys `k in [lo2, hi2)` with
  `0 < |qidx2 - k| < window`. `ow` = softmax-weighted sum of V over that set; an empty
  set gives `ow = 0` (no NaN).
* `g0 = sigmoid(w_gate2[2h] . silu(gateh[b']))`, `g1` the same with row `2h + 1`;
  `out = g0 * oc + g1 * ow + alpha[h] * xres[b', h*D:(h+1)*D]`.

Reference it must match (up to fp32 reduction order): `tul_generate_cached._branches`
(window = SDPA over a 0 / -inf bias, slot = fp32 einsum + softmax with the sink column)
followed by `attention._CCABase._gate_combine_up` up to, not including, `W_up`. The gate
fold is `fused_decode_step._decode_attn_kernel`'s `GH > 0` path. Numerics: fp32 FMA, no
`tl.dot` (Triton runs fp32 `tl.dot` at TF32 by default). Graph-capturable: every range is
read from `meta` on the device; no host read, no data-dependent shape. Sizes: `hi1 - lo1`
up to `S*M` (about 400 at the bench budget), N up to about 450, ranges can be empty.

`meta` per call site (computed by the decoder on the device each step):

| site | lo1, hi1 | lo2, hi2 | qidx1 | qidx2 | qpos | HAS_KPOS |
|---|---|---|---|---|---|---|
| prelude token (span offset j) | 0, 0 | 0, j | 0 | j | 0 | no |
| prelude cell i (span length n) | P, P+i+1 | 0, n | P+i | n+i | 0 | no |
| loop cell (s, i) | 0, (s+1)M | 0, 0 | s*M+i | 0 | 0 | no |
| coda token (abs pos p, cells written c) | 0, c | Cc, Cc+j | 0 | Cc+j | p | yes (`kpos` = cell abs positions) |
| coda cell k of slot s | s*Kp+k, s*Kp+k+1 | 0, 0 | s*Kp+k | 0 | 0 | no |

The coda-cell row is "itself alone": the window set is empty (d1 = 0) and the slot
softmax is over itself and the sink.

Kernel tests the kernel agent owns (`tests/test_lx_decode_attn.py`, CUDA): random tensors
against a PyTorch reference built from `_branches` + the gate combine, for all five site
rows above at both widths; empty ranges; a window shorter than range 1 (loop, coda);
later siblings in the loop row; `kpos` mode; R > 1 with `meta` rows pointing at
different cache rows; a CUDA-graph replay with `meta` changed between replays. Tolerance
argued from an fp64 run of the reference. One sabotage per rule (drop the sink, keep the
self key in the window, ignore `kpos`, read cache row b instead of `meta[b, 0]`) must fail.

**K2 (changed, class b): `decode_front(..., kv_row=None)`**, file
`morph/kernels/triton/fused_decode_step.py`. New optional `kv_row [B'] int`: query row b
stores its K/V at `win_k[kv_row[b], h, pos_dev[b] % win_k.shape[2]]` (same for `win_v`)
instead of row b. A constexpr `HAS_KVROW` in `_front_post_kernel`; `None` must leave the
engine's kernel unchanged (`tests/test_decode_engine_widths.py` passes, same numbers). Also
to be pinned by a test: a stride-0 expanded `x_hist` with absolute `x_off` rows is legal
(the kernels index `X + b * x_hist.stride(0) + x_off[b, r] * x_hist.stride(1)`). A
reference version of this change (not run): `/home/wolfe/morph-scratch/fdec/stage2-start-reference.patch`.

No other new kernel is needed for a first build. The pointer mixture, the loop-history
commit, the router, the register and the winner write stay PyTorch ops inside the
graphs (class t). Candidates for a later kernel, after a profile says so: the batched
pointer mixture (about 20 small ops per token step) and the router (about 10 per pass).

### File ownership

| owner | files |
|---|---|
| kernel agent | `morph/kernels/triton/lx_decode_attn.py` (new), `morph/kernels/triton/fused_decode_step.py` (K2 only), `tests/test_lx_decode_attn.py` (new), a K2 test file |
| decoder agent | `morph/inference/lx_fused_decode.py` (new), `morph/inference/engine.py` (move `StaticDecodeEngine._hc` to a module function `hc_apply`, pure code motion, in the reference patch), `tests/test_lx_fused_decode.py` (new), `lab/perf/bench_lxtul_decode.py` (`lx_fused` arm) |

The decoder agent can start against a PyTorch stand-in of `lx_attn` with the K1
signature and swap the kernel in when K1's tests pass.

## Alternatives considered

- **Keep the generic decoder and only batch it.** Rejected: its token step is 4.9x the
  engine's per-layer cost at batch 1; batching does not touch that.
- **A slot graph over all B rows with a row mask** (rows that did not cut compute and
  discard). Simpler (no row ids in the slot kernels). Rejected for the bench shapes: with
  1/18.4 cuts per row per step, a B = 8 step that runs a slot graph has about 1.2 cutting
  rows, so the masked graph does about 7x the needed slot work.
- **A new front kernel** for the strict geometry instead of `decode_front`. Rejected: the
  projection, conv, prologue and K/V store are the engine's math; a segment reset is a
  zero history row, and the only missing piece is a per-row K/V row index.
- **`decode_attn` for the slot branch.** Not possible: its compressed branch reads one
  shared `[B, NB, D]` block cache (CSA/HCA pooled blocks); the strict slot branch reads
  the per-head K/V of the slot columns, the same tensors as the window branch.
- **Stride-0 tricks to make `decode_front` write a bucket row's K/V without a kernel
  change.** Rejected: the view needs an out-of-bounds `as_strided`.

## Acceptance criteria

- Greedy tokens equal the graphed decoder and the eager `generate_tul` for 256 tokens on
  at least 2 prompts on `lxtul-pointer-rerun/step_5000.pt`; worst logit difference
  reported against both references (they differ from each other by 4.1e-3).
- B = 1, 4, 8, 16: every row's tokens equal its own B = 1 run.
- CUDA tests on a tiny LXTUL + pointer fixture: teacher-forced log-probs against the
  graphed decoder's step functions, B > 1 rows against B = 1; one sabotage per new
  kernel and per new index makes its test fail.
- Bench arm `lx_fused` at B = 1 and 8 next to `plain_graphed` (exact weights) in one
  invocation: tok/s, ms per token, kernels per event, GPU busy fraction, top kernels, the
  ratio against 2.3x.

## Risks

- Reduction order: `lx_attn` is an online softmax, not SDPA; the fused GEMVs are not
  cuBLAS. Greedy near-ties can flip a token; the gate compares worst row differences.
- The reference's loop conv runs through `fused_cca_conv`'s `tl.dot`, which takes fp32
  operands at TF32 by default; `decode_front`'s conv is fp32 FMA. The fused path can sit
  closer to the exact fp32 function than the reference does. To be measured, not assumed.
- The graphed decoder's fixed 4.1e-3 against eager is not explained. Already ruled out
  by earlier runs (`/home/wolfe/morph-scratch/decode/decode/bench_notf32.json`,
  `bench_det.json`, `bench3_forceeager.json`): cuDNN/matmul TF32 off, deterministic
  algorithms on, and `MORPH_FORCE_EAGER=1` all leave it at 4.1e-3. The `fused_cca_conv`
  TF32 dot runs in both eager and cached, so it is not a likely cause either. The
  fused decoder will report its distance to both references.
- The router's argmax is a discrete pick: a near-tie between two cells can flip under a
  reduction-order change and move the whole span.
- Memory grows with B x S; the bench's tight slot budget keeps it near 1 GB at B = 16.
