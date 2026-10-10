# Agent Note: StaticDecodeEngine runs on the current plain-model geometry

Status: implemented

## Problem

The phase-2 target "LXTUL decode at least 2.3x plain MORPH" needs plain MORPH on its best
decoder, the fused CUDA-graphed `StaticDecodeEngine` (`morph/inference/engine.py`). The
engine was built for a 276M stack and refused the 265M plain panel checkpoint
(`plain-panel-norm-match/step_5000.pt`) at its first width assert. That model differs from
the 276M stack in every width the engine had fixed:

| Assumption in the engine | 276M stack | 265M plain model |
|---|---|---|
| CSA indexer width (q_I, K_I) equals d_head | 32 = 32 | 32 vs 64 |
| CSA block length `csa_m` (emit kernel fixed M = 4, x-offset rows fixed at 8) | 4 | 8 |
| CSA compressor width (emit kernel fixed 32) | 32 | 64 |
| vocab a multiple of 16 (LM-head GEMV asserted it) | yes | 49169 |
| window 128, HCA block 128 | 128 / 128 | 256 / 256 |

`d_head` 64 also made the one-warp front-end post kernel slow (11.1 us per site).

## Decision

The kernels read their widths from their operands instead of assuming them, and the engine
asserts every geometry fact it still relies on:

- `decode_front`: the indexer query width is `qi.shape[1]` (`DI`), and the Wqkv row count
  is checked against `LQ + LK + 2 VH + GH + DI`. Post kernel at `num_warps` 2 when
  d_head > 32 (7.7 vs 11.1 us per call, microbenched); 1 warp at d_head 32 as before.
- `csa_emit`: block length M, compressor width C and indexer width CI come from `B_a`
  `[M, C]` and `B_ia` `[M, CI]`; the scratch is `[B, KS, M, 4C + 2CI]`; the pooling divides
  before it sums, the eager order. `ring_meta` writes `2 M` emit-window rows.
- `bf16_gemv` masks the last output tile; `csa_select` uses `BN = min(128, NB)`.
- The engine asserts: compressor width = d_head, indexer compressor width = d_indexer (one
  per model), every site's compress ratio = its kind's global ratio (a
  `core_hca_compress_ratio` model would read the wrong ring), conv k = 4 (the front unrolls
  7 rows), `2 VH = Hkv d_head`, and an `h_0 = e` loop entry (`_CloneInit`).

Two additions make the engine a complete decoder and its error measurable:

- `reset()`: position 0 with the empty state, so a prompt goes through the graphs with no
  eager prefill. This needed a non-negative ring modulo in `ring_meta` / `ring_commit`
  (Triton's `%` truncates), an empty-window guard in `decode_attn`, and the first CSA
  block's B stream gated to -inf in `csa_emit`, as the eager `F.pad(value=-inf)` does.
- `exact_weights=True`: fp32 copies of every weight the engine otherwise rounds to bf16.
  A parity mode. On the 265M model it puts the engine 1.0e-5 nats (worst log-prob, 256
  steps) from the eager kv_cache path, against 2.9e-2 to 3.4e-2 with the bf16 copies; so the engine's
  arithmetic is exact up to reduction order and the shipped deviation is the rounding.

## Alternatives considered

- Pad q_I to d_head with zero rows of `W_IQ` and zero columns of `K_I`. Rejected: it reads
  and scores twice the indexer width every token, and it leaves the 4-token, 32-wide
  `csa_emit` wrong for `csa_m` 8 anyway.
- Keep the 276M `csa_emit` and add a second kernel for the new widths. Rejected: two
  kernels for one formula drift; the general kernel covers both shapes. The 276M emit is
  no longer byte-identical (reduction order and divide-before-sum); no test or checkpoint
  pinned it.
- Time the engine after an eager kv_cache prefill (its original entry). Rejected for the
  bench: the eager prefill is about 60 ms per token at batch 1 and would dominate a
  graphed decode; the LXTUL graphed arm also feeds its prompt through its graphs.

## Consequences

- `lab/perf/bench_lxtul_decode.py` `plain_graphed` builds and captures one engine per
  batch size outside the timing, then `reset()`s and feeds prompt and new tokens through
  graph replay. `--check_plain` gates it against `plain_cached`; `--plain_exact_weights`
  times the parity engines.
- `tests/test_decode_engine_widths.py` (CUDA only) builds a tiny d_indexer 32 / d_head 64 /
  csa_m 8 model and checks teacher-forced logits from both entries against
  `kv_cache.decode_step`. It fails on the unfixed engine and on three single-line
  sabotages of the new code (numbers in the file).
- With bf16 weight copies the engine can flip a greedy token at a near tie: in the
  2026-10-10 gate, validation prompt 1 after a reset prefill, step 0 (eager top-2 margin
  0.0062 nats, engine deviation 0.018). `exact_weights=True` gives equal tokens on all ten
  gated streams (worst 1.3e-5 nats) at 302.6 vs 319.7 tok/s at batch 1 and 1137.5 vs
  1318.5 tok/s at batch 8.
- Not covered: the mixed-length entry (`load_from_eager_mixed`), retention sites, routed
  MORTAR MLPs and the int8 deploy stack were not run on the new kernels.
