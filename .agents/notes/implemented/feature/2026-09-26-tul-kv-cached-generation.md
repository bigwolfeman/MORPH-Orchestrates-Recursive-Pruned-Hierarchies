# Agent Note: KV-cached generation for the strict TUL slot loop

Status: implemented

## Problem

TUL's purpose is "think once, decode cheap", but the only generator, `generate_tul`,
recomputes prelude -> slot loop -> coda over the whole grown row at every token (O(n^2),
no cache). Decode cost had never been measured. On the fp01@10k strict arm
(`tul_slot_spandec_strict_e4probe_fp01`, `code_enum_k` 4) it runs at 11.4 tok/s (256
tokens) and 7.4 tok/s (512) on the 5090, batch 1, fp32.

## Decision

Two generators that emit the same tokens as `generate_tul`:

- `morph/inference/tul_generate_cached.py` (`generate_tul_cached`): each position is
  computed once. The strict geometry makes the dependency graph span-local. The prelude
  reads its own span, so its K/V and conv history are dropped at each boundary. The slot
  loop is causal on the compact slot axis, so each (pass, layer) keeps every earlier
  slot's K/V. A coda token reads its own span plus every earlier prefix cell, so the
  coda keeps the cells for good and the span tokens until the boundary. The K rollouts
  are the batch axis, and the label-free read is the per-span Bayes mixture, kept as a
  running sum. Every per-position op reuses the model's own module. Only the attention
  read is written again, against the cache. A model with any mechanism outside this set
  raises (`_check_supported`).
- `morph/inference/tul_generate_graphed.py` (`generate_tul_graphed`, `TulGraphDecoder`):
  the same math from fixed-capacity buffers. Capacities are span `rule.span_cap`, slots
  `spec.max_slots`, and coda cells `prefix_k * max_slots`. The masks read position and
  validity rows on the device. Each token is one CUDA graph replay, and so is each slot.
  The decoder holds `parametrize.cached()` open for its whole life, because the graphs
  read the cached QAT weights at fixed addresses.

Measured on fp01@10k, 5090, fp32, greedy, prompt 0. Eager is timed inside
`parametrize.cached()` here. Without that context it runs 11.4 and 7.4 tok/s.

| new tokens | eager | cached | graphed |
|---|---|---|---|
| 256 | 16.6 tok/s | 89.0 tok/s | 280 tok/s |
| 512 | 9.3 tok/s | 71.6 tok/s | 235 tok/s |

Parity with `generate_tul`: tokens are identical on 6 of 6 prompts x 256 greedy tokens,
for both generators. The sampled log-prob rows differ by at most 8.6e-3 (median ~6e-4).
With every fused kernel forced to its reference (`set_force_eager(True)`), the cached
generator differs by at most 1.6e-4, so the rest comes from the reference forward's
fused kernels. One example is the core window kernel, which runs TF32 `tl.dot`. On the
tiny CPU fp32 model the difference is at most 2.9e-6 (`tests/test_tul_generate_cached.py`).

## Alternatives considered

- **A Triton single-query attention kernel first** (the original Stage 2 plan). The
  torch profile of the cached generator says it is launch-bound. The GPU was busy
  169 ms of ~750 ms for 64 tokens, over 57.5k launches (~900 per token). Attention was
  under 10 % of GPU time. A faster attention kernel cannot move a CPU-bound step, so
  CUDA graphs came first.
- **`torch.compile(mode="reduce-overhead")`.** The cache shapes grow every token, which
  forces recompiles or dynamic shapes. Fixed-capacity buffers with explicit graphs give
  one capture per decoder (0.2-0.8 s) and no recompiles.
- **Recomputing each span's prelude and coda instead of caching them.** This is simpler,
  but the cost per token grows with the span and with the number of cells, and it still
  pays the launch cost for every recomputed layer.

## Consequences

- The graphed generator is now GPU-bound, at ~2.9 ms of GPU time per token. The biggest
  costs are skinny fp32 GEMMs (25 %, weight bandwidth) and the segment-conv grouped kernel
  on 7-row windows (9 %). The next steps are bf16 weights, which would change numerics
  against the fp32 reference, and a decode-shaped conv kernel.
- Anything `_check_supported` refuses needs its own port before these generators can run
  it: the paid loop, the register, TUL-Code, restrict geometry, the gate/halt, gram, fans,
  and others. There is no halt mode.
- Tests: `tests/test_tul_generate_cached.py`. It covers the cached generator and the
  graphed step functions (run eagerly on CPU) against `generate_tul`, plus two sabotage
  cases that must fail.
