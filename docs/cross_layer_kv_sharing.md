# Cross-Layer / Cross-Iteration KV Sharing (CLA) — Design Doc

**Status:** PROPOSED (not implemented). Deferred follow-up after the ternary/8-bit
quant factorial.
**Author:** Claude (with Wolfe), 2026-06-02
**Scope:** KV reuse to cut attention compute and KV-cache memory in MORPH's looped core.

---

## 1. Motivation

KV sharing / Cross-Layer Attention (CLA, popularized by Gemma-style models): later
layers reuse the **K** and **V** projections computed by an earlier layer; **queries
stay layer-specific** so attention patterns still differ. Reported gains: ~50% KV-cache
reduction with minimal quality loss; longer contexts / higher throughput on
memory-constrained hardware.

**MORPH's twist — the loop makes this a bigger win than vanilla CLA.** MORPH is
`prelude(3) → core(6)×T → coda(3)` with per-sequence Poisson depth (mean 6, max 8). In a
*stacked* model, CLA shares across N distinct layers (one-time ~2× cut). In MORPH's
*looped* core, the high-value move is sharing KV **across the T loop iterations**:
compute K,V once on the first core iteration, then for iterations `1…T` recompute only Q.
With mean depth 6 that reuses KV ~6×, not ~2× — far more compute and cache saved.

**Why the approximation should be mild here (architecture-specific evidence):** the
looped-blockell lineage measured **T=4 ≈ 98% of T=8** quality. If the loop's
later-iteration refinements are already near-negligible, then early-iteration keys/values
are a good stand-in for later iterations — the architecture's own depth-scaling property
predicts cross-iteration KV sharing is cheap. This is the central bet; the ablation in §6
tests it directly.

---

## 2. Prior art (validated in the sibling `subq-attention` project)

`subq-attention` (the JEPA/TPU line) is *also* a looped model and already implemented
**both** axes — this is a working reference to port, not a green-field build:

- **Cross-layer within a pass** — `subq-attention/attention_cla.py`: odd layers (1,3,5)
  reuse KV from the preceding even layer; `share_interval=2`.
- **Cross-loop-iteration** — `interop/pt_model.py:~70,1460`: *"cache KV on loop iter 0,
  reuse for 1+ … only recompute Q."* This is the MORPH-relevant variant.
- **Result:** `cla_integration_5k` val ppl **65.3 → _v2 62.9** at 5k steps, smooth
  convergence, no instability (wandb `adew-me/subq-attention-ablation`,
  runs `pscmvhwg`, `fb06hkrx`). Proves it *trains stably* in a looped model.
  ⚠️ Not transferable as a number — different model/loss/data; MORPH's baseline is 39.46.

**Do NOT copy the reference's cache mechanism.** subq's `attention_cla.py` uses a module
-global dict `_CLA_KV_CACHE` and its CLAUDE.md warns *"layers MUST execute in order
(layer 0 clears cache)."* Global mutable state is **torch.compile-hostile** (dynamo guard
churn) and adds exactly the kind of thread/stream-state surface where MORPH's
**startup timing-race** lives (confirmed 2026-06-02 to be thread/stream-scheduling
sensitive). MORPH must use **explicit loop-carry** instead (§4).

---

## 3. Where KV lives in MORPH today (the integration surface)

`morph/model/attention.py`, `_CCABase` (read before implementing):
- `W_down_q: d_model → n_heads·d_head`, `W_down_k: d_model → n_kv_heads·d_head`,
  `d_head = d_model/(compression·n_heads)` (compression=2 → half-width latent).
- Causal convs `conv_q_{dw,gp}`, `conv_k_{dw,gp}` (depthwise + head-grouped).
- Value-shift `W_v_curr`, `W_v_prev` (half current token, half t-1).
- `fused_cca_prologue(...)` fuses: QK-mean → RMSNorm(q,k) → learnable temp → CoPE-RoPE →
  GQA repeat → value-shift assembly → returns `q,k,v [B,H,S,D]` (K/V GQA-expanded).
- Downstream of `q,k,v`: CSA (even layers) / HCA (odd layers) compressed branch **+**
  windowed local branch (XSA, self excluded) → sigmoid `gate` blend → residual-`alpha` →
  `W_up: latent_q_dim → d_model`.

**KV is already heavily compressed** — GQA (n_kv_heads=4 → 3×) × CCA channel (compression=2
→ half) × sliding window × CSA/HCA sequence pooling. So the *byte* savings from CLA are
modest on top of this; **the real prize is the cross-iteration COMPUTE saved** (skip
`W_down_k`, `conv_k_{dw,gp}`, `W_v_*`, and the K/V portion of the prologue for iters 1+).

---

## 4. Proposed design — cross-iteration KV sharing via loop-carry

The core loop in `morph/model/transformer.py` already threads the hidden state `h` as
explicit loop state. Thread the cached KV alongside it — no globals.

1. **Iteration 0 (or first share iteration):** run the full `_cca_project` → get
   `k, v` (and the CSA/HCA-pooled/selected derivatives, if we cache post-pool). Store
   `kv_cache = (k, v)` in the loop carry.
2. **Iterations `t ≥ cla_share_start`:** compute **only** `q` (a new Q-only prologue
   variant, §5), attend Q against the carried `k, v`. The window branch, CSA/HCA branch,
   gate, residual-α, and up-projection all run as normal but consume the shared K/V.
3. Queries remain per-iteration ⇒ attention patterns still evolve across the loop
   (the property that makes CLA safe).

Cache the KV at the **CCA latent level** (`k, v` post-prologue, GQA-expanded). CSA/HCA
then derive their pooled/selected blocks from the shared `k,v` — consistent and correct.

### Config knobs (add to `morph/configs/base.yaml`, default OFF)
```yaml
  use_cla: false          # cross-iteration KV sharing in the core loop
  cla_share_start: 1      # first loop iter that REUSES cached KV (0 computes it)
  cla_mode: cross_iter    # cross_iter | cross_layer | both (cross_layer = within the 6 core layers)
```
Log full config to wandb (mandate). `use_cla=false` must be bit-identical to today.

---

## 4b. Implementation reality — discovered 2026-06-02 reading the code (CORRECTS §4/§5)

Three findings from `attention.py` + `mhc.py` + the `transformer.py` loop that make this
bigger and more delicate than §4 assumed. **Read before implementing.**

1. **The shared "KV" is THREE things, not just k/v.** `_CCACSAAttention.forward` /
   `_CCAHCAAttention.forward` compute, per call: (a) `q,k,v = cca._cca_project(x)` — but
   **k,v feed only the WINDOW branch** (`_window_attn`); (b) `C_comp =
   comp_norm(compressor(x))` — the compressed-sequence representation the CSA/HCA branch
   actually attends to; (c) for CSA only, the LightningIndexer `scores → top_idx,
   invalid_mask`. Only `q` and the sigmoid `gate` (in `_gate_combine_up`) are query-side.
   So cross-iteration "KV sharing" must cache+reuse **{k, v, C_comp, (CSA: top_idx,
   invalid_mask)}** and recompute only q + gate. The compressor + indexer are the *bulk*
   of the key-side compute, so caching them is where the speedup actually comes from
   (caching only k,v would save almost nothing).

2. **The MRR residual wrapper hides the attention's multi-output.** `MORPHBlock.forward`
   runs attention through `self.mrr_attn(h, _attn_fn)` where `_attn_fn(x) -> Tensor`
   returns a single tensor. There is **no return path** for the attention to hand its
   computed KV back to the loop. `attn_kwargs` is a clean *input* channel (already
   plumbed) for passing the cache + mode IN, but getting the freshly-computed KV OUT of a
   compute-iteration needs a side channel.

3. **`checkpoint()` forbids the obvious side channel.** Grad iterations run
   `checkpoint(_core_step, …, use_reentrant=False)`, which **re-executes `_core_step` in
   the backward pass**. If a compute-iteration mutated a cache dict during its forward, the
   backward recompute would mutate it again, and the cached tensors would already be wired
   into *other* iterations' graphs → double-use / wrong-graph hazards. Mutable-cache-inside-
   checkpoint is unsafe (this is the same class of bug the global-dict reference has, §2).

**Corrected design (the only checkpoint-safe shape):** compute the shared KV in a
**dedicated KV-pass OUTSIDE the per-iteration checkpoint**, at the `h` state of the
chosen share iteration, materialize {k,v,C_comp,top_idx,invalid_mask} per core layer as
explicit detach-aware tensors, then pass them **read-only** via `attn_kwargs` into the
reuse iterations (which recompute only q+gate). For trap §7.2 (BPTT), run that KV-pass at
the first in-grad iteration (`cla_share_start ≈ n_nograd = total_iters − bptt_depth`) so
W_down_k / compressor / indexer receive gradient; the read-only reuse in later grad
iterations is a clean `autograd` fan-out from that single KV-pass. Reuse iterations slice
the cached batch dim to the current `n_active` prefix (trap §7.1).

**Effort:** this is a real multi-hour change touching `_cca_project` (a q-only variant),
both CSA/HCA forwards (compute-vs-reuse branches returning/consuming the KV bundle), a
KV-pass helper in the loop, and the loop-carry plumbing — plus a hard verify gate
(use_cla=false bit-identical; grad reaches K/V projections; active-prefix slice correct;
finite loss under compile+checkpoint). NOT a quick wire-up. Implement deliberately.

## 5. Kernel work

`fused_cca_prologue` currently produces q **and** k,v together. CLA needs a **Q-only**
path for the reuse iterations: QK-mean coupling currently couples q and k (the "QK-mean"
step reads both) — so a Q-only variant must either (a) also carry the cached `k_lat`
needed for the mean, or (b) cache the *post-mean* q-side statistics. Decide during
implementation; (a) is simpler (carry `k_lat` too, cheap). Then Q-only does:
`q_lat = W_down_q(x)` → conv_q → (QK-mean using cached k_lat) → RMSNorm(q) → temp → RoPE.
Keep it branchless / `tl.constexpr`-gated to stay compile-clean. Verify fwd max-err and
grad-cosine against an eager Q-only reference, per the repo kernel convention.

---

## 6. Ablation / test plan

Baseline = current full-KV-recompute (val ppl 39.46 @ 15k, kernels b4). Arms:
- `cla_iter1` — `use_cla=true, cla_share_start=1` (share from iteration 1).
- `cla_iter2` — `cla_share_start=2` (compute fresh KV for 2 iters, then share) — buys
  back quality if iter1 hurts.
- (optional) `cla_xlayer` — `cla_mode=cross_layer` (within the 6 core layers).

**Measure both ppl AND step-time/peak-mem** — the win is compute, so a ppl-neutral arm
that isn't faster is pointless. Prediction (from T=4≈T=8): ppl cost small,
step-time down (fewer K/V projections + convs per iteration).
Gate: accept if ΔPPL vs 39.46 is within ~1–2% AND step-time improves.

---

## 7. ⚠️ MORPH-specific traps (these are NOT in the sibling reference)

1. **Active-set shrinking + KV indexing.** The core loop sorts sequences by sampled depth
   and processes a shrinking prefix `h_s[:n_active]` (A1 optimization). KV cached at the
   first iteration covers all `B` sequences; later iterations process only the alive
   prefix. The cached `k,v` must be **sliced to the active prefix** each iteration
   (`kv[:n_active]`), and this is only correct if the depth-sort order is **stable** across
   iterations. Confirm the sort is done once and the prefix is contiguous — otherwise a
   gather is needed. This is the most likely correctness bug.

2. **Truncated BPTT × where K/V gradients come from.** `bptt_depth=4`: only the last ~4
   loop iterations carry gradient (truncated). If KV is computed at iteration 0 (outside
   the grad window) and reused, then `W_down_k`, `conv_k_*`, `W_v_*` receive gradient
   **only** from iteration 0's forward — which may be in the no-grad region ⇒ those
   projections barely train. Fix options: (a) compute/cache KV at the **first in-window
   iteration** (`cla_share_start ≈ max_depth − bptt_depth`), so KV projections sit inside
   the gradient window; or (b) accept iter-0 KV training and verify the projections still
   learn. Decide and document; (a) is safer.

3. **torch.compile / timing-race.** Use loop-carry, no global cache (§2). The share
   decision keys on the **static** unroll index `t` (Python loop var), not data — so no
   data-dependent branch, no recompile. Re-run the startup-wedge sensitivity check after
   integration (the race is thread/stream-scheduling sensitive; new state in the loop
   carry is exactly the kind of change that perturbs it — verify with eval+gen ON, the
   config that the 2026-06-02 bs=2 hunt showed is needed to surface the race).

4. **Composition with the gate + residual-α + sinks.** These are per-head, per-iteration
   and consume the (shared) attention outputs — they should compose unchanged, but the
   gate's input is `x` (full d_model), so it still varies per iteration. No change needed,
   but confirm no module caches a per-iteration KV-derived statistic.

---

## 8. Files to touch (when implementing)
- `morph/model/attention.py` — `_CCABase`: add `cca_project_q_only(x, cached_k_lat)`;
  thread `(k, v)` out of / into `forward` via an optional `cached_kv` arg.
- `morph/model/transformer.py` — core loop: carry `kv_cache`; gate compute-vs-reuse on the
  static iter index; slice to active prefix (trap #1).
- `morph/kernels/triton/fused_cca_prologue.py` — Q-only variant (§5) + parity test.
- `morph/configs/base.yaml` — `use_cla`, `cla_share_start`, `cla_mode`.
- `ignore/verify_cla.py` — gate: `use_cla=false` bit-identical to baseline; finite
  loss + grad flow to K/V projections (trap #2) under compile; active-prefix slice
  correctness (trap #1).

---

## 9. Sequencing
Implement after the ternary/8-bit quant factorial completes (don't perturb the running
GPU). The two are orthogonal and can eventually compose (8-bit AdamW + CLA + ternary).
