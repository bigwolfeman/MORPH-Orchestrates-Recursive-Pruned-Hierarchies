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
| `attention.py` | CCA + CSA (even layers) / CCA + HCA (odd), `RMSNorm`, CoPE. Positions come from the tensor's own sequence axis, so a gathered sub-sequence is automatically re-coordinated. `n_blocks = S // compress_ratio` can be 0 for a short sequence; both the eager and fused paths handle it (the windowed branch then carries the layer). |
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
