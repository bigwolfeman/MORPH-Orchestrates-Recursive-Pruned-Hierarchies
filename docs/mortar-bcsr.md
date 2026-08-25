# MORTAR BCSR

**Status:** implemented (MORPH-original; no paper)  
**Code:** `morph/model/layers/block_sparse.py`, `morph/model/sparsity.py`, `morph/sparse/stk/`  
**Schedule:** [CMS lifecycle figure](figures/morph_cms_lifecycle.png) · `morph/training/pruning.py`  
**Not this doc:** ReMoE / `TileRouter` internals, ternary QAT details

MORTAR (**M**acro-**O**rchestrated **R**outing and **T**ile-**A**ligned **R**ecompaction) is
MORPH’s post-carve sparse weight format: a **128×128 block-CSR (BCSR)** packing of an MLP
linear, executed by the vendored MegaBlocks STK Triton kernels. It is the **only** sparse
backend in tree. Provides a surface for routing ReMoE per tile.

CMS picks which blocks survive. MORTAR packs them.

---

## CMS (how blocks die)

CMS is the **pre-carve** topology path on the same `CMSBlockLinear`: an EMA of per-tile
saliency, then scheduled drops of whole **128×128** blocks under a dense mask.

**Score.** Default `cms_score_mode: taylor`. After every `backward` (before
`zero_grad`), `accumulate_scores()` updates `block_score_ema` on **16×16** tiles:

```
tile score  ←  ‖ W_tile ⊙ ∇W_tile ‖_F     # Taylor; alt: grad-only or magnitude
block_score_ema ← EMA(block_score_ema, tile score)
```

Under ternary QAT, grads come from the smooth shadow leaf (STE), not the discrete forward weight.

**Prune.** From `prune_start`, every `prune_interval` steps, `prune_step_blocks`:

```
pool 8×8 tile EMAs → one score per 128×128 block
drop ~prune_rate of lowest-score ALIVE blocks   # global top-k
constraints: ≥1 block kept per block-row; optional protection never drops
expand deaths → 16×16 tile mask (exactly block-aligned)
apply_prune_mask() every step so dead tiles (+ grads) stay zero
```

Still **dense** `F.linear` until carve — only the mask is sparse. Stops when density ≤
`target_density`. Orchestration: `PruningSchedule.step` in `pruning.py`.

**Hand-off.** `carve()` at `compact_step` reads that block-aligned mask and packs survivors
into MORTAR BCSR (next sections). Recipe integers stay in `morph/configs/base.yaml`.

---

## What it is

A `nn.Linear`-shaped layer (`CMSBlockLinear`, wrapped as `MortarLinear`) has two lives:

| Phase | Storage | Matmul |
|---|---|---|
| Pre-carve | dense `weight [out, in]` + CMS prune mask | `F.linear` (cuBLAS) |
| Post-carve | `mortar_data [nnz, 128, 128]` + BCSR index buffers | `stk_dds` (Triton) |

CMS zeroes **128×128** blocks in the dense matrix while training.
`carve()` runs once at `compact_step`: it **packs the surviving blocks** into BCSR and
**deletes** the dense `weight`. Topology is frozen after that.

```
dense [out × in]                 after carve
┌──┬──┬──┬──┬──┬──┐              mortar_data[nnz,128,128]
│██│  │██│  │  │██│              ██  ██  ██  ██   ← packed blocks only
├──┼──┼──┼──┼──┼──┤              + row_indices / column_indices / offsets
│  │██│  │██│  │  │              + transpose trio for bwd
├──┼──┼──┼──┼──┼──┤
│██│  │  │██│██│  │              density = nnz / (Rb · Cb)
└──┴──┴──┴──┴──┴──┘
 each cell = one 128×128 block
 Rb = out/128,  Cb = in/128
```

---

## Geometry

- **Block grain:** `blocking = 128`. Both `out_features` and `in_features` must be divisible by 128.
- **Block grid:** `Rb = out/128`, `Cb = in/128`.
- **`nnz`:** number of kept blocks = `mortar_data.shape[0]`.
- **Density (post-carve):** `nnz / (Rb · Cb)`.
- **CMS saliency tiles:** 16×16 — eight-by-eight of them make one 128×128 block. Carve
  consumes the **block** mask CMS produced (§ CMS).

---

## On-disk / in-memory layout

After `carve()`:

| Name | Role |
|---|---|
| `mortar_data` | `Parameter [nnz, 128, 128]` — the nonzero blocks |
| `mortar_row_indices` | block-row of each nonzero (BCOO-style companion) |
| `mortar_column_indices` | block-column of each nonzero |
| `mortar_offsets` | CSR row pointers over blocks |
| `mortar_column_indices_t` | transpose topology |
| `mortar_offsets_t` | transpose row pointers |
| `mortar_block_offsets_t` | maps forward nnz order → transpose order |

Forward topology drives `y = x @ Wᵀ` via `stk_dds`. The `_t` trio is for efficient
transposed iteration in the STK autograd path (same idea as MegaBlocks BCSR + transpose
metadata — see [references § MegaBlocks / STK](references.md)).

---

## Carve (what actually happens)

Pseudocode of `CMSBlockLinear.carve(blocking=128)`:

```
assert still dense, dims % 128 == 0
W ← live weight  (smooth shadow if ternary-parametrized)
apply current prune mask to W

tile_mask [R, C]     ← CMS 16×16 alive bits  (or all-True if never pruned)
block_mask [Rb, Cb]  ← any(tile alive) inside each 128×128 cell
require ≥ 1 kept block per block-row

(row_idx, col_idx) ← nonzero(block_mask)
offsets            ← exclusive_cumsum(nnz_per_block_row)
data [nnz,128,128] ← gather W’s 128×128 tiles at those coords

(col_t, offsets_t, block_offsets_t) ← stk_transpose(row_idx, col_idx, offsets)

delete Parameter "weight"
register mortar_data ← data
register the six index buffers
_mortar ← True;  _dense_mode ← False
density ← nnz / (Rb * Cb)
return nnz
```

**Lossless** when pruning used `prune_step_blocks` (already 128-aligned). A legacy
tile-only mask keeps any partially alive 128×128 whole (dead tiles ride as zeros) —
numerically exact, worse compression; code warns.

**Optimizer rebuild** is required after carve: the dense `weight` is gone;
`mortar_data` is the new trainable. This causes a brief loss spike that settles out.
It is like resuming a checkpoint with out optimizer state.

---

## Forward / backward (post-carve)

```
# forward  (training and deploy)
x2 ← reshape(x, [-1, in])
pad M to multiple of 128 if needed   # decode shapes
y  ← stk_dds(x2, mortar_data,
             row_indices, column_indices, offsets,
             column_indices_t, offsets_t, block_offsets_t,
             transpose_b=True)         # y = x @ Wᵀ
y  ← y[:M] + bias?
return reshape(y, *lead, out)
```

Autograd is the STK custom op (`morph.sparse.stk.backend.custom_ops`): DDS for the
forward product, SDD / DSD pieces for `dW` / `dx` as in the MegaBlocks toolkit.
MORPH does not reimplement those kernels; it feeds them this BCSR layout.

Under autocast, lhs + `mortar_data` are cast to the autocast dtype before `stk_dds`
(byte-compatible with the old stk `custom_fwd` path).

---

## Where it sits in the model

Every SwiGLU MLP in **prelude, core, and coda** uses `MortarLinear` for
`gate_up` and `down` (`_SwiGLUMortar` in `transformer.py`). The looped core
**reuses the same carved weights** every iteration — sparsity is on the weight
topology, not on loop depth.

Attention, embeddings, and norms are not MORTAR.

---

## What MORTAR is not

| Thing | Relation |
|---|---|
| **ReMoE / TileRouter** | Activation MoE over `d_ff` **clusters** after SiLU·up. Orthogonal to which weight blocks exist. Arms after `route_start`. MORTAR supplies the tile map the router selects over. |
| **Block-ELL** | Removed legacy sparse backend. MORTAR BCSR + STK only. |

CMS is in this doc (§ CMS): it chooses *which* blocks die; MORTAR packs what’s left.

---

## Operator gotchas

1. **Carve without prune** (or `prune_start` past the run) → `block_mask` all true →
   `density ≈ 1.0`. You still have “BCSR,” but it’s a dense matrix in sparse clothing.
   Trust `[prune] density=` logs before claiming 0.25.
2. **`training.sparse_backend`** must be MORTAR (or unset). Anything else raises.
3. Miss **`accumulate_scores` between `backward` and `zero_grad`** and CMS saliency is a
   silent no-op — prune never learns; carve often freezes a still-dense model.

Recipe step numbers (`prune_start`, `compact_step`, `target_density`) live in
`morph/configs/base.yaml` — not restated here.

---

## Code map

| Path | Role |
|---|---|
| `morph/model/layers/block_sparse.py` | `CMSBlockLinear` (score, prune, carve, `_forward_mortar`) |
| `morph/model/sparsity.py` | `MortarLinear` wrapper |
| `morph/model/transformer.py` | `_SwiGLUMortar` wiring |
| `morph/training/pruning.py` | CMS schedule: score / prune / carve / route gates |
| `morph/sparse/stk/` | vendored STK Matrix, `stk_dds`, Triton kernels |

Ledger acceptance of the 0.25 deploy path: [ablation-ledger.md](ablation-ledger.md) row **MORTAR-0.25**.
