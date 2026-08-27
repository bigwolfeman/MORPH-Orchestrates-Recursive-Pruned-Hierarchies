# TUL-TG: Thought-Gestalt restriction for TUL

Status: SPEC — approved for build 2026-08-27. Source of truth for the TG arms.
Pre-registration: `lab/experiments/planned/2026-08-27-tg-restriction.md` (committed before build).
Paper: "Modeling Language as a Sequence of Thoughts" (arXiv 2512.25026, Thought Gestalt / TG).

## 0. Why

TG shows a load-bearing sentence latent with a SINGLE next-token CE loss, from scratch,
at 85M non-embedding params — below our scale, on plain LM data. Its one trick: token
self-attention is restricted to the current sentence; prior context is reachable ONLY
through cross-attention to a memory of sentence vectors. No shortcut ⇒ the latent is
load-bearing by construction. TUL has the latent, the memory, and extra think compute,
but leaves the token shortcut open — so the plan is optional (plan-off worth ~0.015
nats, loop worth ~0.009). This spec closes the shortcut.

Under the restriction the existing module geometry maps onto TG directly:
- window branch  → within-span token attention (TG's within-sentence attention)
- compressed branch → attention over prior SLOT positions (TG's per-layer
  cross-attention to sentence memory, with per-layer projections for free)
- core loop on slots → extra compute on the only channel to the past

## 1. The mask (single source of truth)

Derived per forward from `SlotLayout` (a DATA argument, like `slot_layout` itself —
no runtime feature flag in the hot loop). For query position i and key position j:

    allow(i, j) = (j <= i)                                  # causal
                  AND ( bag_id[i] == bag_id[j]              # same span (tokens+own slot)
                        OR slot_mask[j] )                   # or j is any slot position

Notes:
- `bag_id` at a slot position is its own slot index, same as its span's tokens, so a
  span's tokens and its slot form ONE segment. Tail-pad / post-last-slot tokens share
  the dump bin `max_slots` and form one trailing segment.
- Build once per forward: `[B, 1, L, L]` bool, from `layout.bag_id` and
  `layout.slot_mask`. Helper lives in `morph/model/tul_layout.py`
  (`tg_allow_mask(layout) -> Tensor`) so tests and both branches share ONE builder.

## 2. Window branch (within-span attention)

`_window_fallback` gains an optional `extra_mask` (`[B,1,S,S]` bool) that is ANDed
into its existing mask. XSA self-exclusion (`j != i`), the window-size bound, and the
`n_skip_rope` suffix rules keep their exact current semantics. The fused window kernel
is NOT touched: `tg_restrict` + `use_kernels=true` RAISES at construction (arms run
eager; a silent unmasked kernel path is the theater we refuse).

Window size 128 ≥ max span extent (32 tokens + slot positions), so no within-span
visibility is clipped. Distant slots are the compressed branch's job.

## 3. Compressed branch → slot attention

Under `tg_restrict`, BOTH `_CCACSAAttention` and `_CCAHCAAttention` replace the
pooled-block compressed stream with direct attention over slot positions:

    out_comp = SDPA(q, k, v, mask = causal(i,j) AND slot_mask[j]) with sink

- q, k, v are the ALREADY-COMPUTED per-position CCA tensors from `_cca_project`
  (no new projections; per-layer W_down_q/k, W_v_* play TG's per-layer
  cross-attention projections).
- Sink: append the existing per-head `sink_logits` as an extra logit with a ZERO
  value vector, so a query with no visible slot (first span) gets a well-defined
  softmax and a zero-ish output — same contract as the fused kernels' sink.
- Construction: when `tg_restrict=true`, the `compressor`, `comp_norm`, and (CSA)
  `indexer` modules are NOT built. Dead params under weight decay silently change
  trajectories; we do not build what we do not use. `_fuse_mods` shrinks to match.
  Param-count delta vs control is reported in the build report and wandb config.
- Gate MLP, residual-alpha, W_up, RMSNorm/temp/CoPE machinery: unchanged.

## 4. GLA segment reset (prelude + coda only)

Retention attaches at layer index 1 of prelude, core, and coda. The prelude/coda
branches scan the FULL token sequence and would carry past-span state around the
mask. Fix, zero new scan code:

    reset[i] = (i == 0) OR (bag_id[i] != bag_id[i-1])       # segment starts

AMENDED 2026-08-27 (same day, during build): the spec first prescribed flooring
`log_alpha` to −30 at reset positions. The build measured that this collides with
`_chunked`'s PRE-EXISTING −30 overflow clamp on the chunk-global cumsum: with
several resets per chunk (the tg_restrict regime) the cumsum dives past −30, the
clamp pins later positions to one value, and the relative decay is destroyed
(up to ~780% rel err at span-density resets). The shipped mechanism is
STRUCTURAL instead: `_recurrent` zeroes the state entering a reset position
(exact); `_chunked` computes the cumulative log-gate RELATIVE TO EACH RESET
SEGMENT, masks intra-chunk pairs that cross a reset, feeds the carried state
only to pre-first-reset positions, and carries only the last segment out of the
chunk — each segment is then exactly the single-segment case the −30 clamp was
designed for. `GatedLinearAttention.forward` gains an optional `reset_mask`
argument (kernel mode raises); `MORPHBlock` threads it via a new
`ret_reset_mask` forward arg. The CORE loop's GLA (slot sequence — the allowed
memory channel) gets NO reset and keeps `retention_carry`.
Equivalence gate: recurrent-with-reset EXACTLY equals, and chunked-with-reset
matches to rel err ≤ 1e-5 (fp32), the per-segment recurrent oracle run from true
zero state — including multi-reset-per-chunk and misaligned boundaries (test T2).

## 5. Accepted local leaks (documented, bounded, adjacent-only)

| Leak | Width | Why accepted |
|---|---|---|
| CCA causal conv (kernel 4) | ≤ 3 positions across a boundary | local only; cannot carry long-range context |
| Value-shift `W_v_prev` | 1 position | same |
| Hash-bigram embedding at span-first token | 1 token | same |
| `retention_carry` cross-chunk non-causality (known, pre-existing) | — | unchanged; hits control and TG arms equally |

Everything else that crossed spans (full-attention prelude, compressed-block pooling,
prelude/coda GLA) is closed by §§1–4.

## 6. Configuration

`TULConfig.tg_restrict: bool = False` — construction-time, mirrors the Hydra
`tul.tg_restrict` key (resolved in `morph/training/tul_setup.py`, logged in the wandb
manifest). `False` builds nothing and is bit-identical to master (test T4).
`tg_restrict=true` requires `use_kernels=false` and `tul` active; violations RAISE at
construction. Arm configs: `tul_tg1.yaml`, `tul_tg2.yaml`, `tul_tg3.yaml`
(TG3 = soft restriction: same-span OR previous-span OR slot — one extra allow term,
`tg_soft_prev_span: true`).

## 7. Tests (each one fails when the code is broken)

- **T1 mask**: `tg_allow_mask` equals a brute-force triple-loop reference built from
  `bag_id`/`slot_mask` on a hand-written layout AND on a random packed batch.
- **T2 GLA reset**: chunked path with `reset_mask` matches the recurrent oracle run
  segment-by-segment from true zero state, rel err ≤ 1e-5 (fp32).
- **T3 severed-channel leak probe (the falsifier)**: eager fp32, kernels off. Zero the
  slot channel (mask slot keys out of BOTH branches via a test-only hook), then
  `∂ logits[t] / ∂ embed[u]` for u a token ≥ 4 positions before the end of span 0 and
  t a token in span 2 must be EXACTLY 0. The same probe on a `tg_restrict=false`
  model must be NONZERO (proves the probe can detect the leak it claims to rule out).
- **T4 off-path identity**: a `tg_restrict=false` model is bit-identical to master
  (existing TUL tests keep passing; state dict keys unchanged).
- **T5 smoke**: 12-step train run of `tul_tg1`, loss decreases, no NaN/Inf, and the
  step logs show the TG construction banner.

## 8. Arms

See the pre-registration for predictions and decision rules. TG1 = restriction only
(losses untouched). TG2 = restriction + `plast_weight=0, emit_weight=0` (TG's own
single-objective recipe; also removes the takeover's fuel, per the O5 result).
TG3 = soft restriction, run only if TG1's CE craters past the pre-registered line.
