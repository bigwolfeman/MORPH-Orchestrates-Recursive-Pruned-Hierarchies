# `morph/model/` — notes for whoever edits this next

The forward is the hot path for every MORPH run. Two rules dominate everything here:
**no runtime feature flags inside the forward** (features are baked in at construction,
so `torch.compile` sees a branch-free graph), and **any new mechanism must be
bit-identical to the previous behaviour when it is off** — proven by running the old
and new code on the same seed and comparing `torch.equal` on loss, params and grads,
not by reading the diff.

## Layout

| File | Role |
| --- | --- |
| `transformer.py` | `MORPHConfig`, `MORPHTransformer`. The forward is split into REGIONS — `_front_region` (embed+prelude), `_core_region` (the Poisson-depth loop), `_back_region` (coda+head) — which are the single source of truth for both the eager path and the CUDA-graph capture. Add code to a region, never a second copy of one. |
| `attention.py` | CCA + CSA (even layers) / CCA + HCA (odd), `RMSNorm`, CoPE. Positions come from the tensor's own sequence axis, so a gathered sub-sequence is automatically re-coordinated. `n_blocks = S // compress_ratio` can be **0**, and the paths do not crash — but "does not crash" is not "is fine": the compressed branch then contributes NOTHING for the whole run while the gate keeps spending half its mixture on a zero tensor. Read the section below before choosing any ratio. |
| `mhc.py` / `hyper_connections.py` | The `[B, S, n, C]` Hyper-Connection carrier. Every injection is a single-stream `[B, S, C]` term broadcast into the streams. |
| `fused_ce.py` | Chunked weight-tied cross-entropy. Never materialises `[N, V]`. Each call allocates and SAVES a `[V, d]` fp32 `grad_w` (201 MB at V=49169, d=1024) — so prefer one call with `weights=` over one call per label group. `mask_token_id` forces a vocab row's logit to −inf (probability and gradient exactly 0). |
| `tul_layout.py`, `tul.py` | TUL (`docs/tul-spec.md`). See the table in the root `CLAUDE.md`. |

## Things that look like bugs and are not

* `RMSNorm` returns **fp32** even under autocast: its final `* self.weight` promotes.
  Anything that scatters into or concatenates with a normed carrier must cast at the
  boundary — that is why `scatter_positions` does `values.to(pad.dtype)`.
* The TUL core deliberately does **not** use the token path's active-set shrinking. The
  per-slot depth is a masked update over the whole compact sequence, because MORPH
  recomputes K/V from the current carrier every iteration and a frozen slot must keep
  serving the same keys (`runtime-invariants.md` §6b).
* TUL parameters are constructed even when the layout is off. They stay inert (grad
  `None`, so the optimizer skips them), which is what lets a mid-run activation happen
  with no optimizer rebuild. All three inits are deterministic, so they draw no RNG and
  the baseline weights are unmoved.
* `x0_injects` / the bigram term are added at EVERY layer including the coda. Any
  mechanism that claims to hide a token's identity from the coda (TUL's token-state
  dropout) must zero those too, or the token walks straight back in.

## What the core's attention ACTUALLY is at the shapes it runs

Measured 2026-08-25 with `lab/divergence/attn_sink_probe.py --geometry --token-path`, on
`tul_a1` (`d_model 1024`, `n_heads 8`, `compression 2` → `d_head 64`, `window_size 256`,
`csa_compress_ratio 8`, `hca_compress_ratio 256`, `top_k 256`). **Print this before
reasoning about core attention. Three of the four rows were a surprise.**

The looped core does NOT run at the stack's sequence length. Under TUL it loops over SLOT
positions — 64 with `tul.max_slots: 64` — while prelude and coda run on all 1152.

| core layer | branch | slot path, S = 64 | token path, S = 1152 |
|---|---|---|---|
| CSA (even, 3 of 6) | window | dense causal over all 64 | window 256 of 1152 |
| CSA | compressed | `n_blocks` 8, `tk` 8 → **every block selected** | `n_blocks` 144, `tk` 144 → **every block selected** |
| HCA (odd, 3 of 6) | window | dense causal over all 64 | window 256 of 1152 |
| HCA | compressed | `n_blocks` **0**, `\|out_comp\|` **0.0000** | `n_blocks` 4, `\|out_comp\|` ~1030 |

1. **The HCA compressed branch is dead on the slot path.** 256 does not divide into 64, so
   `GatedPoolCompressor` returns `[B, 0, c]`, `fused_hca_attention` has nothing to attend
   to, and `_gate_combine_up` blends `g_comp ~ 0.50` into a zero tensor. Three of six core
   blocks deliver about half the attention output they were built for, silently, for a whole
   run. `model.core_hca_compress_ratio` exists to fix exactly this; `16` gives the slot core
   4 blocks, the same number the token path gets. Deploy (`seq_len 4096`, 512 slots) is not
   affected — the defect needs a slot budget below the ratio.
2. **CSA's sparse selection never fires on the short schedule.** `top_k: 256` exceeds
   `n_blocks` at `seq_len 1024` (144), so `tk = min(top_k, n_blocks) = n_blocks` and CSA is
   dense pooled attention, not sparse. At the deploy `seq_len 4096` there are 512 blocks and
   selection does fire. Every TUL arm measured to date ran a dense CSA.
3. **The window branch covers everything on the slot path**, `window_size` 256 > 64, and XSA
   excludes the self token — so query `i` attends keys `0..i-1` and query 0 attends NOTHING
   (its softmax row is all `-inf`; `F.scaled_dot_product_attention` returns 0 there, an
   explicit softmax returns NaN). Early slots therefore absorb mass by CONSTRUCTION: any
   claim about a "sink" on slot 0 must be a comparison across steps or iterations, never an
   absolute reading.
4. **The forward attention is diffuse, and stays diffuse.** Across the whole `onset-capture`
   ladder the window participation ratio is 0.59–0.68 of 57 valid slots and the top key holds
   6.4–8.4 % of the mass. There is no attention sink in the core — the cotangent sink that
   the backward shows is a consequence of the slot-state rank collapse, not its cause
   (`docs/experiments/failures/2026-08-25-h18-positional-attention-sink.md`).

**The general lesson, which is why this section exists.** A module correct at its design
shape can be degenerate at the shape a new subtree runs it at, and integer floor division to
zero is silent. Before trusting any reasoning about attention here, run the geometry probe
and compare the sick path against the healthy one at the same weights.

**Not a shipped bug, recorded so it is not rediscovered.** `_window_fallback` builds its
mask with `torch.where(mask, 0.0, -inf)`, which is fp32, and hands it to
`F.scaled_dot_product_attention` whatever the dtype of `q`. At fp64 `q` that mismatch makes
SDPA return a silently WRONG result (measured error 3.03 against the explicit softmax, where
a matched fp64 mask is exact to 4e-16). At fp32, bf16 and fp16 — every dtype MORPH runs — the
matched and mismatched masks agree to the last bit. Do not "fix" the hot path for a case that
cannot occur; do not write an fp64 test against this function either
(`tests/test_attn_sink_probe.py` says so in place).
