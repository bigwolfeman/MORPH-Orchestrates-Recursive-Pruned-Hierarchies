# Ablation Ledger

Accepted / rejected / deferred components in the default stack. Rows point at
config keys and the engineering logs under gitignored `ignore/` (and `Ai-notes/`)
that backed the call. Metrics are approximate and hardware-bound; see
[known-good-runs](../.agents/notes/implemented/process/2026-07-03-known-good-runs.md)
for environment assumptions. The default recipe itself lives in the heavily commented
[`morph/configs/base.yaml`](../morph/configs/base.yaml) — do not restate its numbers here.

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
| TUL-A0 | MORPH baseline | `tul.activate_at: never` (plain schedule) | reference | **RUN** 2026-08-18 (seed 0, b14): val CE 3.2805 |
| TUL-A1 | TUL | `tul:` block defaults (slots looped, tokens skip core, coda sees slots, per-slot Poisson) | the method | **RUN** 2026-08-23 (seed 0, b12): val CE 3.4175. Also 2026-08-18 (seed 0, b14): 3.2243. Repetition vs A0 at matched b14: NO detectable effect (−0.026 ± 0.097 at top-k, n=12, MDE 0.272) | 
| TUL-A1r | TUL repeat | as A1, second seed | retrain noise floor — read BEFORE any cell | **DIVERGED 2/2** (step 3240 uncapped, step 4160 CAPPED at b12). NO NOISE FLOOR EXISTS; every TUL cell below is one seed. [bake-off](experiments/results/2026-08-23-tul-gate-bakeoff.md) |
| TUL-A2 | slots-as-memory | `tul.tokens_through_core: true` | C2 alone (plan readable, uniform depth) | planned |
| TUL-TG1 | TG restriction | `tul.tg_restrict: true` (docs/tul-tg-spec.md; Thought Gestalt 2512.25026) | close the token shortcut → plan load-bearing | **RUN** 2026-08-27 (2 seeds, b6, 3500 steps): plan worth 0.087 ce_main, loop worth 0.006; s2 TOOK OVER @1258. The control's 0.012-0.016 is NOT a fair baseline for plan worth — an unrestricted arm recovers cross-span information through causal attention, so its plan worth is low by construction (confound correction 2026-08-28). [experiment](../lab/experiments/failures/2026-08-27-tg-restriction.md) |
| TUL-TG2 | TG single-objective | TG1 + `plast_weight/emit_weight: 0` | TG's own recipe; removes takeover fuel (O5) | **RUN** 2026-08-27 (2 seeds): 0/2 takeovers, END CORE SHARES 0.0020/0.0035 (campaign lowest), loop worth 0.024-0.036 ce_main (campaign largest, sub-0.05), ce within 0.045 of TG1. The plan-worth-vs-control comparison is confounded (see TUL-TG1); the plan-content probe decides it. |
| TUL-TG3 | soft restriction | TG1 + `tul.tg_soft_prev_span: true` | is the hard restriction too tight? | **DIVERGED 2/2** 2026-08-28 (abort @2040, @2120; end shares 0.9986/0.9927, block gain 1.95/1.75 at r2 0.97/0.98). NO step-3000 checkpoint, so the arm CANNOT answer its own question — it is tul_tg1-based and dies of the aux-loss takeover. Re-asked as TUL-TG3b. [experiment](../lab/experiments/planned/2026-08-28-tg-round2-seed-and-softness.md) |
| TUL-TG4a | delete the bag-mean seed | TG2 + `tul.slot_seed: e_slot` | pooling law says the bag-mean dilutes | **RUN 2/2** 2026-08-28: ce_main@3000 4.8094/4.7735 (mean 4.791 vs TG2's 4.794 — **the seed moves CE by 0.003 nats**). Takeover held 0/2, core shares 0.0011/0.0010 = campaign lowest. Loop worth is NOT comparable across slot_seed modes (prediction B2 falsified 3.6-7.3x). |
| TUL-TG3b | soft restriction, no aux | TG2 + `tul.tg_soft_prev_span: true` | TG3's question from a base that survives | planned, pre-registered, chained |
| TUL-TG4b | boundary slot seed | TG2 + `tul.slot_seed: boundary` (`E_slot + W_sent·embed(t_last)`) | TG's own sentence-head shape | **RUN 2/2** 2026-08-28: ce_main@3000 4.9477/4.7497. **s1 TOOK OVER @1951** (end share 1.0000, block gain 2.608 at r2 0.97) — the FIRST tul_tg2-based seed to do so, which breaks the clean "aux losses are the only fuel" account. Mechanism uninvestigated. |
| TUL-cap64 | raise span_cap 32→64 | `tul.span_cap: 64`, `gate_k_max: 80`, TG2 base | 26.9% of spans were cut by the BUDGET, not a boundary | **RUN 2/2** 2026-08-28: ce_main@3000 4.7299/4.7200 (mean 4.7250 vs TG2's 4.794, **−0.069**), 0/2 takeovers. FREE: +2.00% token positions, +0.0% core compute, KV 9.85×→12.47×, AT-CAP 26.9%→4.5%. [experiment](../lab/experiments/planned/2026-08-28-span-cap-64.md) |
| TUL-spec | plan span-specificity | `plan_shuffled` vs `plan_off` (no training) | is the plan CONTENT read, or just its positions? | **RUN** 2026-08-28 on 7 checkpoints. **CORRECTED 2026-08-28: the AUX LOSSES write the plan's content; the restriction degrades it ~20x.** Full 2x2: aux ON 65.1% (no restr) / 3.0% (restr); aux OFF 0.4% / 0.3%. Without emit/plast there is no content at either mask setting — so TG2 and every arm built on it had an empty plan by construction. Every restricted arm 0.1–0.6%. The coda gains 0.037–0.123 nats from the slot path and ~none of it depends on which span wrote the plan. [experiment](../lab/experiments/planned/2026-08-28-plan-span-specificity.md) |
| TUL-A4 | depth-only | `tul.coda_sees_slots: false` | C1 alone (depth per idea, plan unreadable) | planned |
| TUL-A3 | shallow control | no slots, `n_core` bypassed for tokens (seed path) | compute floor | **RUN** 2026-08-18 (seed 0, b14): val CE 3.2407 at 2.76x A0 throughput — beats A0 |
| TUL-p | token-state dropout sweep | `tul.token_state_dropout ∈ {0, 0.15, 0.3}` | the collapse tax | planned |
| TUL-act0 | activate at step 0 | `tul.activate_at: 0.0`, TST off | isolates the 3-transitions-at-30k risk | planned |
| TUL-stp | punc-STP on slot trajectory | `tul.stp_lambda > 0` | slot warm-up (Wolfe's punc-STP finding) | planned |
| TUL-set | slot-set MCE warm-up | `tul.set_lambda > 0` | slot warm-up (TST MCE); Block Transformer §4.2 says aux on the latent hurt | planned |
| TUL-prefix1 | prefix length 1 | `tul.prefix_k: 1` (default is 2, projection prefixes, Block Transformer App. F.2 / Fig 3f) | plan and first-token label forced onto one coda position | planned |
| TUL-gate | span-length gate | `--config-name tul_gate` (`tul.gate: true`, `gate_lambda: 1.0`, `gate_budget_cond: true`, `gate_truncate_p: 0.15`) | does a model-chosen span length pay? ([tul-gate-spec.md](tul-gate-spec.md)) | **RUN 2026-08-23: YES, −0.105 nats vs A1 at identical layer-passes/token, `plan_nats` 42x. NO ERROR BAR (A1r died).** [results](experiments/results/2026-08-23-tul-gate-bakeoff.md). ALSO repeats far less: −0.251 rep4 at top-k (t=−3.27, 10/12 prompts) and −0.177 at greedy (12/12), with the better CE on the same arm, so it is not the diversity trap. [repetition](experiments/results/2026-08-23-tul-repetition-sampled-decoding.md) |
| TUL-halt | halting gate | the SAME run: `gate_drives_depth: true` scores every eval a second time with the gate choosing each slot's depth (`val/halt_*`) | does variable depth pay on top? | **RUN 2026-08-23: NO — prediction held. Worse at 39/40 evals, and `depth_mean` COLLAPSED to 1.00 at every eval, so the near-tie is degeneration not merit.** [results](experiments/results/2026-08-23-tul-gate-bakeoff.md) |
| TUL-A1+ | TUL reinvest | `n_coda: 8`, `tul.slot_mean_depth: 12` (≤ A0 layer-passes/token) | the fair-compute cell | planned |
| TUL-xattn | cross-attn branch | `tul.xattn: true` (attach like retention) | BLT T7 vs Block Transformer Fig 3f | planned |
| TUL-carry | explicit `W·h_{i-1}` | `tul.carry: true` | Coconut feedback vs attention-only memory | planned |
| TUL-A5 | fixed stride | `tul.fixed_stride: 19` (mean span matched, boundary rule off) | alignment vs depth (SpaceByte T1 +0.10 bpb, HAT T1 −2.7 HS); A1 − A5 is what the boundary rule buys | planned |
| TUL-bcast | broadcast-add | `tul.bcast: true` (offset-indexed linears, init 0, `h_i` added to span i+1's coda token input) | AU-Net T4: tie at 2 levels, +5.4 at 3; expected null at one level | planned |

Metrics per arm: `val/ppl_tokens`, `val/first_tok_ce`, `val/plan_nats` (slots
masked at eval minus unmasked), `val/first_tok_counterfactual`, rep4@512,
span-length distribution of generations, layer-passes/token, tokens/s.
All of them are logged by `morph/training/train.py` as of the implementation
(2026-08-16). Rows marked **RUN** have results; the rest stay `planned`.

> **Read this before citing any TUL number.** `TUL-A1r` is the retrain noise floor and it
> has diverged on both attempts — uncapped at step 3240, and CAPPED at batch 12 at step
> 4160. **No TUL cell has an error bar.** The −0.105 nat gate result is a single seed-0
> pair. Separately, every capped run that has ever survived is seed 0 (4/4) and seed 1 has
> been tested once and failed, so the `ademamix_alpha_cap: 1.0` stability claim is
> seed-0 evidence only. Both facts and the divergence signature are in
> [the bake-off results](experiments/results/2026-08-23-tul-gate-bakeoff.md).
>
> **Two corrections, 2026-08-23.** (1) The `spec/sigma_max` precursor claim is withdrawn:
> the gate arm survives 20k with a core linear at 5.618, ABOVE the diverged arm's 5.508
> at its abort, and `rho_eff` is 1.9–3.2 on every run that finishes. Contractivity is not
> the discriminator; the live precursor that does work is `gradnorm/core`, which ratchets
> 0.009 → 0.043 → 0.108 → 0.90 about 140 steps before the norm explodes.
> [mechanism](experiments/failures/2026-08-23-tul-forward-backward-asymmetry.md)
> (2) **Every ABSOLUTE CE in this table is inflated** by the causality defect in
> [`retention-carry-breaks-causality`](../.agents/notes/proposed/bug-fix/2026-08-23-retention-carry-breaks-causality.md):
> `retention_carry: true` lets every position read the whole sequence from core iteration
> 2 onward, measured at **+0.1433 nats** — larger than the −0.105 gate result. Arm-minus-arm
> differences survive because every arm carries it. Absolute numbers do not.

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


**Pre-registered 2026-08-21, before any arm is run [W].** `TUL-halt` does NOT beat
`TUL-gate` on val CE: fixed depth wins or ties, and we ship fixed depth because it keeps
inference shapes static. **Falsifier:** `TUL-halt` beats `TUL-gate` on val CE by more than
the `A1`/`A1r` retrain noise floor AND does not lose on the generation metrics (rep4@512,
distinct-3, mean span length, fraction of spans ending on a boundary). A results note is
written whether these arms win or lose — the predecessor missed both of its pre-registered
numbers and never wrote one ([tul-gate-spec.md](tul-gate-spec.md) §2, §11).

**Amendment 2026-08-22, still before any arm was scored.** Building the arms found the
halting prediction is true *by construction*, and the reason is worth writing down rather
than claiming as a result. The per-slot depth is a Poisson draw independent of the input,
so a head trained to emit 0 until its last iteration converges to the HAZARD, and the
length is scaled away — measured `k = 5.00` against gold `18.98`, matching the hazard
table to the integer ([tul-gate-spec.md](tul-gate-spec.md) §6). With that half of the
target removed (`gate_train_zeros: false`), `g` sits near the mean length, `k ≥ 1` fires
on the first iteration, and `TUL-halt` halts at depth 1 everywhere. So `TUL-halt` is no
longer a test of "is variable depth better"; it is the measured cost of running the loop
ONCE, and the honest answer to "can one scalar carry both stop and length" — it cannot.
The pre-registration stands, but the credit goes to the encoding argument, not to the run.

**Bake-off, 2026-08-22:** `ignore/perf/gate_bakeoff.sh` — `tul_gate`, then `tul_a1`, then
`tul_a1r`, sequentially on one 5090, all three with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (measured necessary: without it
`tul_gate` OOMs on the first backward with 8.17 GB reserved-but-unallocated).

## Rejected — DiffusionBlocks (arXiv:2506.14202)

**Not planned, not running, not on this branch.** Thirteen arms were pre-registered here; the
method was then tested to a verdict in a clean-room reference implementation and rejected. The
MORPH implementation was removed from `master` on 2026-08-21 and parked on
`park/db-master-line` (the wired-into-MORPH line) and `feat/db-objective-l2` (the later
concat/L2/CE line). Verdict, numbers and what stays unverified:
[`.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md`](../.agents/notes/rejected/feature/2026-08-21-diffusionblocks-verdict.md).

One line of it: at a matched 143.4M-token budget on a clean 124M Llama, plain next-token
training reached held-out CE **4.0010**; the best DiffusionBlocks arm reached **5.0801** at
sigma_max, and only **4.6740** after 4x the tokens. Generation, scored at matched output
diversity, was 162 gen-PPL for AR against 326 for the best DB arm.

**Metric warning, kept because it outlived the arms.** A DB arm cannot be compared to A0/A1 on
`val/ppl_tokens`: DiffusionBlocks is not ELBO-derived, so its CE is sigma-conditioned
reconstruction, not a likelihood (paper App. E.4). And generative PPL alone is not a quality
score in either family -- a repetition loop scores better than real English, so it only means
something next to a diversity measure. Both traps cost real analysis time.

## How to extend

When a decision lands in or leaves `base.yaml`: add or update a row, cite the
config keys, and point at a small log or gate script under `ignore/` (not a
multi-GB trace).
