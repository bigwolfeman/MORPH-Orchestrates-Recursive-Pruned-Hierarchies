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
| STP | Reject STP as default training objective | STP-on vs STP-off campaigns | Not part of survivor recipe, mixed results in pretraining and SFT punc STP strictly better in SFT. | low | `ignore/gate_stp_*`, `tst_stp_*` runlogs |
| Stock-AdEMAMix-3buf | Reject stock 3-buffer AdEMAMix path | Removed from code | β1=0 fork is the deploy optimizer | high | optimizer module history |
| Static-graphs-default | Defer static/opt CUDA graphs as default | `MORPH_STATIC_GRAPHS`, `MORPH_OPT_CUDA_GRAPH` | Bit-exact when on; memory pool cost OOMs local deploy shape without allocator tweaks | medium | `ignore/perf/*graph*` |
| Zyphra-RSA | Deferred | Outer inference harness | Requires RL; not in training path | low | CLAUDE.md / architecture notes |
| JAX-parity | Deferred | `morph/jax/` | Mirror lags (still MRR residual); PT is source of truth | high | `morph/jax/`, interop converter |

## Planned — TUL (`experiments/tul`; short schedule `morph/configs/tul_short.yaml`: seq 1024 × batch 14 × 20k steps = 287 M tokens, TST off, prune/carve/route off (dense), TUL from step 0; first pass = A0, A1, A1r, A3)

Arms from [tul-spec.md](tul-spec.md) §7. Every row is PLANNED; confidence is
blank until a gate script exists under `ignore/`. Do not cite these as results.

| ID | Arm | Config / mechanism | Isolates | Status |
| --- | --- | --- | --- | --- |
| TUL-A0 | MORPH baseline | `tul.activate_at: never` (plain schedule) | reference | planned |
| TUL-A1 | TUL | `tul:` block defaults (slots looped, tokens skip core, coda sees slots, per-slot Poisson) | the method | planned |
| TUL-A1r | TUL repeat | as A1, second seed | retrain noise floor — read BEFORE any cell | planned |
| TUL-A2 | slots-as-memory | `tul.tokens_through_core: true` | C2 alone (plan readable, uniform depth) | planned |
| TUL-A4 | depth-only | `tul.coda_sees_slots: false` | C1 alone (depth per idea, plan unreadable) | planned |
| TUL-A3 | shallow control | no slots, `n_core` bypassed for tokens (seed path) | compute floor | planned |
| TUL-p | token-state dropout sweep | `tul.token_state_dropout ∈ {0, 0.15, 0.3}` | the collapse tax | planned |
| TUL-act0 | activate at step 0 | `tul.activate_at: 0.0`, TST off | isolates the 3-transitions-at-30k risk | planned |
| TUL-stp | punc-STP on slot trajectory | `tul.stp_lambda > 0` | slot warm-up (Wolfe's punc-STP finding) | planned |
| TUL-set | slot-set MCE warm-up | `tul.set_lambda > 0` | slot warm-up (TST MCE); Block Transformer §4.2 says aux on the latent hurt | planned |
| TUL-prefix1 | prefix length 1 | `tul.prefix_k: 1` (default is 2, projection prefixes, Block Transformer App. F.2 / Fig 3f) | plan and first-token label forced onto one coda position | planned |
| TUL-A1+ | TUL reinvest | `n_coda: 8`, `tul.slot_mean_depth: 12` (≤ A0 layer-passes/token) | the fair-compute cell | planned |
| TUL-xattn | cross-attn branch | `tul.xattn: true` (attach like retention) | BLT T7 vs Block Transformer Fig 3f | planned |
| TUL-carry | explicit `W·h_{i-1}` | `tul.carry: true` | Coconut feedback vs attention-only memory | planned |
| TUL-A5 | fixed stride | `tul.fixed_stride: 19` (mean span matched, boundary rule off) | alignment vs depth (SpaceByte T1 +0.10 bpb, HAT T1 −2.7 HS); A1 − A5 is what the boundary rule buys | planned |
| TUL-bcast | broadcast-add | `tul.bcast: true` (offset-indexed linears, init 0, `h_i` added to span i+1's coda token input) | AU-Net T4: tie at 2 levels, +5.4 at 3; expected null at one level | planned |

Metrics per arm: `val/ppl_tokens`, `val/first_tok_ce`, `val/plan_nats` (slots
masked at eval minus unmasked), `val/first_tok_counterfactual`, rep4@512,
span-length distribution of generations, layer-passes/token, tokens/s.
All of them are logged by `morph/training/train.py` as of the implementation
(2026-08-16); none has been RUN, so every row above stays `planned`.

Measured SHAPE facts for the arms (5090, `tul_short.yaml`, 13-25 steps,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) — engineering numbers, not
results. Reviewer-measured 2026-08-16 at the batch every arm actually runs:

| arm | config | batch | peak alloc | s/step | tok/step | layer-passes/token | 20k steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | `tul_a0` | 14 | 20.32 GB | 0.947 | 14336 | 44 | 5.3 h |
| A1 | `tul_a1` (`max_slots 64`) | 14 | 24.06 GB | 0.544 | ~14462 | 10.68 | 3.0 h |
| A1r | `tul_a1r` | 14 | as A1 | as A1 | as A1 | as A1 | 3.0 h |
| A3 | `tul_a3` (`n_core 0`) | 14 | ~17.7 GB | ~0.30 | 14336 | 8 | ~1.7 h |
| — | A0 at batch 16 (superseded) | 16 | 22.92 GB | 0.99 | 16384 | 44 | 5.5 h |
| — | A1 at batch 16 | 16 | **OOM** | — | — | — | — |

A1's prelude and coda run on `L_total` = 1152 positions where A0's run on 1024, and
those 8 layers are not checkpointed, so A1 costs MORE activation memory while running
1.7× faster per step. A1 cannot fit batch 16, so EVERY arm was moved to 14 rather than
letting the batch size vary across a paired comparison — at 14 the arms match on
tokens/step to 0.9 % (`tul/tokens_per_batch` is logged every 20 steps).

## Planned — DiffusionBlocks (arXiv:2506.14202; pre-registration in [diffusionblocks-experiment-sheet.md](diffusionblocks-experiment-sheet.md))

Every row is PLANNED. **Nothing has run.** Expected numbers, metric contract, kill criteria and the
per-block schedule-stretch mitigations live in the sheet — do not restate them here, and do not cite
any row below as a result. Design rationale:
[diffusionblocks-morph-assessment.md](diffusionblocks-morph-assessment.md).

| ID | Arm | Config / mechanism | Isolates | Status |
| --- | --- | --- | --- | --- |
| DB-0 | instrumentation gates | `perf/{layer_passes_per_token, positions_per_token, flop_proxy, model_tflops, mfu}` + measured GEMM ceiling | FLOP efficiency is unmeasurable today; `perf/mfu` does not exist | planned |
| DB-1 | recurrent-depth, `B=1` | whole 4:6:4 net as one denoiser, single-pass training, T-step Euler inference | does the diffusion objective work on MORPH at all (the paper's literal Huginn setting) | planned |
| DB-2 | block-wise, `B=3` | prelude \| core \| coda on MORPH's own seams; σ cut into T+2=8 equi-probability Euler steps; core conditioned on `x0` only | block independence, and whether the 320-dim ctx slice can carry the core's whole conditioning | planned |
| DB-3 | `B=3` + TUL | DB-2 with `tul:` on; slot denoising target = next span's first token embedding | the TUL cross Wolfe asked for; the slot-target fork | planned |
| DB-4 | `B=1` + TUL | DB-1 with `tul:` on | TUL cross on the cheaper conversion | planned |
| DB-5 | `B=3`, prelude conditioning | core conditioned on a no-grad prelude forward | the 320-channel pipe (DB-2 − DB-5 measures how thin it is) | planned |
| DB-6 | batch sweep | 14 → 20 → 24 → 28 on the best Phase-1 arm | turns the memory win into tok/s; A1 currently OOMs at 16 | planned |
| DB-7 | overlap `γ` | `γ ∈ {0.0, 0.05, 0.1}` (their text default is 0.1) | block-boundary smoothing | planned |
| DB-8 | `B=2` | prelude+core \| coda | Table 8's best cell was B=2 (FID 9.90 vs B=1's 12.09) | planned |
| DB-9 | σ partition | equi-probability vs uniform | Table 7: 38.03 vs 43.53 FID — confirm it holds for text | planned |
| DB-10 | T-c fallback | core not independent, `B=2` for TUL arms | rescues DB-3 if the slot target fails | planned |
| DB-11 | σ-blend contraction ONLY | `h_k ← α h_{k-1} + (1−α) f(h_{k-1})`, `α<1`, **DB objective off** | the `ρ(J_core) ≤ 1` handle on its own, as a Task #276 cure. Keeps ordinary CE, so it lands on THIS ledger | planned |

**Metric warning.** DB arms cannot be compared to A0/A1 on `val/ppl_tokens` — DiffusionBlocks is
not ELBO-derived, so its CE is σ-conditioned reconstruction, not a likelihood (paper App. E.4). The
only legal cross-family metrics are gen-PPL(GPT2-XL), MAUVE and rep4@512, computed post-hoc. Sheet
§1.3.

## How to extend

When a decision lands in or leaves `base.yaml`: add or update a row, cite the
config keys, and point at a small log or gate script under `ignore/` (not a
multi-GB trace).
