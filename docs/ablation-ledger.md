# Ablation Ledger

Accepted / rejected / deferred components in the default stack. Rows point at
config keys and the engineering logs under gitignored `ignore/` (and `Ai-notes/`)
that backed the call. Metrics are approximate and hardware-bound; see
[known-good-runs.md](known-good-runs.md) for the default recipe.

Confidence: **high** = repeated runs or a gate with a pass marker; **medium** =
single campaign or partial stack; **low** = directional / incomplete.

## Accepted (in the default stack)

| ID | Decision | Config / mechanism | Notes | Confidence | Logs / scripts |
| --- | --- | --- | --- | --- | --- |
| HC-Cayley-n4 | Accept Cayley Hyper-Connections (n=4) as sole residual | `hc_streams: 4`, `hc_use_kernel: true` | Fused HC vs eager: similar loss trajectory; isolated HC speedup ~1.45× on local shape | high | `ignore/ab_hc_kernel_result.json`, `ignore/verify_hyper_connections.py` |
| MORTAR-0.25 | Accept CMS prune → MORTAR BCSR carve at 0.25 density | `target_density: 0.25`, `compact_step`, MORTAR only | Carve is lossless when prune is 128-block-aligned; sparse path is the deploy stack | high | `ignore/verify_compaction.py`, phase_c mortar campaign logs |
| ReMoE-whole-body | Accept whole-body tile routing after carve | `routing.route_scope: all`, `route_start` | Routing engages at TST recovery boundary; aux input detached for BPTT memory | high | `ignore/gate_fused_router_parity.py`, route campaign logs |
| TST-0.3 | Accept Token Superposition Training for first 30% of steps | `tst_bag_size: 6`, `tst_ratio: 0.3` | Multi-hot CE wired (loss ~log(V), not log(V)/s); ~−4% ppl on quant-only d=768/15k stack | medium | `ignore/gate_tst_mce_wiring.py`, task #231 notes |
| Ternary-backbone | Accept forward-STE ternary on MLP backbone | `ternary: true`, `ternary_scope: backbone` | Training ppl is deploy-ternary ppl; symmetric scale won A-series | high | `ignore/verify_ternary_qat.py`, quant A/B logs |
| Embed-int6 | Accept int6 QAT on euclidean + bigram embeddings | `embed_quant: "int6"` | Lorentz stays bf16 (geometry-sensitive) | high | embed quant campaign notes |
| AdEMAMix-b1zero | Accept β1=0 AdEMAMix (2-buffer, 8-bit state) | `optimizer: ademamix_b1zero`, `adam8bit: true` | Memory parity with AdamW8bit; divergence cures (SNR gate, stale-push cap, update clip) validated | high | `ignore/gate_ademamix_*`, `Ai-notes/06-21-2026/AdEMAMix-b1zero-Divergence-Cure/` |
| GLA-retention | Accept GLA branch on section-local layers (not full-attn interleave) | `retention: true`, `retention_layers: [1]` | Parallel branch with small init gate; carry across core iters | medium | retention / context-coverage notes |
| CCA-CSA-HCA | Accept compressed attention stack | baked into `MORPHAttention` | Sub-quadratic long-context path; kernels have reference fallbacks | high | `ignore/parity_*`, fused attn gates |
| Flat-LR | Accept flat 1e-4 (no warmup, min_lr==lr) | `lr`, `warmup: 0`, `min_lr: 1e-4` | Validated mortar winner recipe at local scale | medium | phase_c mortar scratch logs |

## Rejected or deferred

| ID | Decision | What was tried | Why not default | Confidence | Logs / scripts |
| --- | --- | --- | --- | --- | --- |
| MRR-residual | Reject MultiRateResidual as live residual | Legacy `mrr_*` attribute names remain for checkpoints | HC-Cayley won; names kept for compatibility only | high | CLAUDE.md naming gotcha, HC deploy logs |
| Block-ELL | Reject Block-ELL sparse backend | Legacy compact path | Slower than dense at target density; MORTAR only | high | stk / removal verifies |
| FP8-default | Reject FP8 as default at d=768 | `fp8: false` | Dead in small-GEMM regime; mutually exclusive with ternary per-layer | medium | `ignore/verify_fp8*.py`, `ab_fp8_*` |
| Attn-proj-QAT | Reject attn-proj int8 as default | `attn_proj_quant: "off"` | Needs own long validation; winner=off at validated ppl | medium | quant A/B notes |
| STP | Reject STP as default training objective | STP-on vs STP-off campaigns | Not part of survivor recipe | medium | `ignore/gate_stp_*`, `tst_stp_*` runlogs |
| Stock-AdEMAMix-3buf | Reject stock 3-buffer AdEMAMix path | Removed from code | β1=0 fork is the deploy optimizer | high | optimizer module history |
| Static-graphs-default | Defer static/opt CUDA graphs as default | `MORPH_STATIC_GRAPHS`, `MORPH_OPT_CUDA_GRAPH` | Bit-exact when on; memory pool cost OOMs local deploy shape without allocator tweaks | medium | `ignore/perf/*graph*` |
| Zyphra-RSA | Deferred | Outer inference harness | Requires RL; not in training path | low | CLAUDE.md / architecture notes |
| JAX-parity | Deferred | `morph/jax/` | Mirror lags (still MRR residual); PT is source of truth | high | `morph/jax/`, interop converter |

## How to extend

When a decision lands in or leaves `base.yaml`: add or update a row, cite the
config keys, and point at a small log or gate script under `ignore/` (not a
multi-GB trace).
