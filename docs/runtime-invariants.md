# Runtime Invariants

MORPH is a looped / recurrent model trained with truncated BPTT. Several
process-global and import-time choices look like ordinary global state, but they
exist to keep **autograd, `torch.compile`, CUDA graphs, and the recurrent core**
coherent. Treat them as invariants, not cleanup targets.

This document is the public map. Implementation comments in
`morph/training/train.py` and `morph/model/transformer.py` remain the detailed
source of truth. It exists to stop LLMs from breakings things, and people from complaining.

## 1. Process-global kernel mode

| Switch | Location | Default |
| --- | --- | --- |
| `MORPH_FORCE_EAGER` / `set_force_eager` | `morph/kernels/triton/_eager_flag.py` | off (`0`) |
| `MORPH_HC_FORCE_EAGER` / `set_hc_force_eager` | same | off |
| `MORPH_DYNAMO_FENCE` / `kernel_fence` | same | on (`1`) |
| `model.use_kernels` | Hydra / `MORPHConfig` | `true` |

At `MORPHTransformer` construction, `use_kernels=False` calls
`set_force_eager(True)` so every fused Triton entry point falls back to its
pure-PyTorch reference. That flag is **process-global**: the last model built in
the process wins. Do not build two models with different `use_kernels` values in
one process and expect both to stay correct.

**Why it exists:** same architecture and weights, kernel-ON vs kernel-OFF A/B,
without threading a flag through every call site. Reference paths stay Dynamo-
traceable; fused dispatchers are fenced via `@kernel_fence` (default =
`torch.compiler.disable`, the historical behavior).

**Dynamo fence (2026-07-03):** the fences are now conditional. `MORPH_DYNAMO_FENCE=0`
(read once at import) removes them so the fused kernels stay **inside** the compiled
graph — modern dynamo (verified torch 2.11) traces the autograd Functions and Triton
launches natively. Measured on the d512 seed (B32/S64): fenced kernels+compile ==
kernels+eager (~21 ms/step, every frame falls back); unfenced 19.6 ms; pure
reference+compile 14.9 ms. Opaque kernels are fusion barriers — at small shapes
prefer `use_kernels=false` + compile; unfence for cloud-scale shapes where the
kernel islands win. Known issue: this torch nightly's inductor mis-generates the
`_hc_premap` launcher (grid args) — compose with `MORPH_HC_FORCE_EAGER=1` until fixed
upstream. Correctness of the compiled reference path is gated by fp32 no-autocast
parity: loss Δ = 0.0, worst grad rel 5.6e-5 (bf16 shows ~2 % — reduction-order
rounding, not error).

**Do not:** flip `set_force_eager` mid-training under a live compile stance, or
assume import-time `DISABLE_FUSED_KERNELS` is the primary switch (runtime path is
`force_eager()`).

## 2. BPTT, checkpointing, and compile

| Invariant | Where |
| --- | --- |
| `torch._functorch.config.donated_buffer = False` | `train.py` import-time |
| Compile **MLP submodules only** (core MLPs `dynamic=True`) | `train.py` |
| Warmup every active-set size (incl. `n_active==1`) **before** wandb / dataloader threads | `warmup_compile_all_shapes` |
| After warmup: `torch.compiler.set_stance("eager_on_recompile")` | `train.py` |
| Generation: `@torch.compiler.set_stance("force_eager")` | `run_generation_test` |
| Truncated BPTT depth / selective activation checkpointing | `model.bptt_depth`, `model.ckpt_grad_iters` |

**Why:** the looped core uses non-reentrant gradient checkpointing. Donated-buffer
reuse aliases inputs under compile. Mid-loop Inductor/Triton recompiles fork
workers while wandb/httpx threads hold glibc arena locks → intermittent hangs.
Warmup in a thread-free window plus `eager_on_recompile` is the mitigation, not
optional polish.

`MORPH_COMPILE_CARVED` (default off at d=768) is an opt-in recompile window at
carve/route boundaries; measured net-negative locally, kept for cloud-scale
revisit.

## 3. Optional CUDA graphs (default off)

| Switch | Role |
| --- | --- |
| `MORPH_STATIC_GRAPHS` | Capture fixed-shape embed/prelude and coda/head regions |
| `MORPH_OPT_CUDA_GRAPH` | Capture fused AdEMAMix step |

The variable-depth core loop stays eager. Fused CE stays eager (host `.item()` is
illegal during capture). Failed capture must abort — never silently fall back
with a poisoned CUDA RNG. See comments on `MORPHTransformer.build_static_graphs`.

Graphs require a large memory footprint, but are faster. Some configurations get stuck looping graph compiles during training because of poisson depth sampling.


## 4. Phase schedule (TST → prune → carve → route)

Canonical local recipe: `morph/configs/base.yaml`.

| Phase | Keys | Notes |
| --- | --- | --- |
| CMS prune | `prune_start`, `prune_interval`, `target_density` | Density must reach target **before** prune step |
| MORTAR carve | `compact_step` | Freezes topology into BCSR; rebuild optimizer |
| ReMoE route | `routing.route_start` | Requires compact (unless carve explicitly disabled) |
| TST superposition | `tst_bag_size`, `tst_ratio` | Training-only; eval/gen always `bag_size=0` |

`PruningSchedule.step` must run **after** `loss.backward()` and **before**
`optimizer.zero_grad()` so saliency sees live grads.

Routing aux uses `aux_detach_input: true` so load-balance grads do not extend
BPTT depth into OOM.

## 5. Causality contract (2026-07-03)

**No module may pool statistics across the sequence axis.** Every position's
output must depend only on positions ≤ t. This sounds obvious; AI constantly makes this mistake
(humans too), it is answer leakage. This one was well hidden so it is documented to prevent regression.

| Invariant | Where |
| --- | --- |
| GLA readout norm is **per-token** (S folded into batch) | `morph/model/gla.py` `_readout` |
| Trailing right-pad must be inert at real positions | gated in Olympiad `tests/models/test_morph_seed.py`, and **here** in `tests/test_causality_contract.py` |
| Kernel dtype pins `(q * scale).to(tl.float32)` stay | `fused_csa_attention.py`, `fused_hca_attention.py` |
| ⚠️ **BROKEN 2026-08-23: `retention_carry: true` violates this section.** The GLA final state (a whole-sequence summary) is carried into core iteration t+1, so from iteration 2 every position sees the future. Corrupting tokens after k moves logits at ≤ k by up to **4.08** (mean \|logit\| 2.41); with the carry off, by exactly 0.000. Worth **+0.1433 nats** of val CE, which is larger than the TUL-gate result it was used to measure. | [`.agents/notes/proposed/bug-fix/2026-08-23-retention-carry-breaks-causality.md`](../.agents/notes/proposed/bug-fix/2026-08-23-retention-carry-breaks-causality.md) |


**Gate:** the future-corruption probe (corrupt tokens after position k, assert
logits at ≤ k unchanged) is the cheap decisive test for any new branch — norms,
pooling, attention variants — before it trains. This catches all answer leakage problems reliably.
It now lives in this repository as `tests/test_causality_contract.py`; for a real
checkpoint use `ignore/perf/future_corruption_probe.py`, and
`ignore/perf/causality_bisect.py` hooks every submodule to name the first one that
diverges. Running the gate ONLY in the Olympiad tree is how the `retention_carry`
violation above survived from 2026-07-03 to 2026-08-23.

The dtype pins exist because dynamo's kernel re-trace promotes python-float
kernel args to fp64, poisoning `tl.dot` operands and loop-carried accumulators.
They are numerical no-ops in eager. Removing them re-breaks compile.

## 6. Diagnostic env knobs (default off)

These are observation-only when unset: `MORPH_EXACT_TRACE`, `MORPH_MEM_PROBE`,
`MORPH_DIAG_*`, `MORPH_PERF_REGIONS`, `MORPH_PROF_WINDOW`, `MORPH_NSYS_WINDOW`,
`MORPH_DEBUG_STEP`, `MORPH_FAULT_TIMEOUT`, `MORPH_DIV_PPL`.

`MORPH_EXACT_TRACE=<path>` appends per-step loss hex for bit-identical A/B gates.
Use only on gate runs (adds a host sync per step).

## 6b. TUL invariants (`experiments/tul`, LIVE — see `tul-spec.md` §9)

These are runtime invariants now, not aspirations: each row names the test that
fails when it is broken (`tests/test_tul_layout.py`, `tests/test_tul_forward.py`;
116 tests pass at implementation time). Each was also mutation-checked — the code
was deliberately broken and the named test observed to go red.

| Invariant | Why |
| --- | --- |
| The boundary rule (`.;!?` + newline + dashes, `min_span`, `span_cap`, EOS) is ONE function used by the loader and the generator, parity-tested. | The slot layout is structural; a train/generation mismatch silently decodes without the plan (the coconut `assert_layout_parity` lesson). `test_incremental_parity`, `test_generator_row_builder_matches_the_packer`. |
| **Run collapse is CAUSAL: the boundary lands after the FIRST token of a run of boundary tokens, and `min_span` absorbs the rest.** | `tul-spec.md` §3.1 rule 2 places it after the LAST token of the run, which cannot be decided without reading the NEXT token — so it is not implementable at generation, where the rule must be causal (§6, and invariant 1 above). A `.`+`\n` run still yields exactly ONE boundary, which is what rule 2 was for. `test_run_of_boundary_tokens_yields_exactly_one_boundary`. |
| The packer fills a row to exactly `L_total`; when the next unit does not fit, the leftover ≤ `prefix_k` positions become TAIL PADS (input `slot_id`, label −100, in `slot_mask`, absent from `slot_index`). | Fixed shapes without ever dropping a boundary inside the row. Measured cost on OWT at `max_slots 64`: 1.18 % of positions. `test_l_total_is_fixed_and_token_count_varies`. |
| Per-slot depth is a masked update over the full compact slot sequence, never a per-position gather. The token path's active-set shrinking is deliberately NOT used on it. | Frozen slots must still serve same-iteration K/V; a gather changes what they attend to. `test_per_slot_depth_is_a_masked_update_not_a_gather`, `test_batch_of_mixed_depths_matches_separate_single_row_runs`. |
| Slot core states have no loss; a slot's only label is the first token of the next span; pad slots are `-100`. | Loss-free latent (MegaByte, H-Net, LD4LG, Pred-Sent); the LTD think-position failure. A pad slot's `slot_index` is 0, so a missing validity mask would silently train on the PREVIOUS row's last token. `test_pad_slots_are_excluded_from_every_loss_group`. |
| `slot_id` is masked from the LM head in the fused CE (`mask_token_id`) and at generation (`index_fill`). Its logit is −inf, so its probability and its gradient are exactly 0. | The model must never emit a slot; slots are inserted by the rule. Two masking sites, one test each: `test_masked_vocab_row_receives_zero_gradient`, `test_slot_id_logit_is_minus_inf_and_never_top_1`. |
| `L_total = tokens + prefix_k · slots` is fixed per curriculum stage; token count varies per row and is logged. | Fixed shapes for kernels/graphs; BLT's tokens-per-batch control held in expectation. |
| Val/gen run with the TUL layout ON and `bag_size 0`; passing both raises. | Val PPL over token positions stays comparable to the baseline. `test_bag_size_and_slot_layout_are_mutually_exclusive`. |
| The §5 half-weight double label is ONE weighted CE call, not one call per label group. | Each `fused_linear_cross_entropy` call allocates and saves a `[V, d]` fp32 `grad_w` (201 MB at V=49169, d=1024). The per-group CEs are §7.2 METRICS and run at eval only. `test_weighted_ce_equals_the_explicit_half_weight_combination`. |
| The static-region CUDA graphs are never captured while the TUL layout is on, and are invalidated at a mid-run activation. | They capture the PLAIN front/back at the plain shape; `L_total ≠ seq_len` so they could never replay, and their private pool permanently reserves ~9 GB. |
| `slot_layout=None` is bit-identical to today's forward, and building the TUL parameters does not perturb the baseline. | The TST phase and every pre-TUL checkpoint must reproduce. VERIFIED against `b268ba3`: loss, every parameter and every gradient `torch.equal`, with and without `MORPHConfig(tul=...)` (all three TUL inits are deterministic and constructed last, so zero RNG draws move). `test_tul_params_do_not_perturb_the_plain_path`. |
| A config key for an unimplemented arm RAISES; it is never silently ignored. | `tul.stp_lambda`, `tul.set_lambda`, `tul.carry`, `tul.xattn`, `tul.bcast` are specified (§3.5) but not built. `test_unimplemented_arm_keys_raise`. |
| **NOT HELD TODAY.** A TUL arm must not run the looped core with a DEAD compressed branch: the core's `hca_compress_ratio` must divide into the slot budget at least once. | `GatedPoolCompressor` computes `n_blocks = S // m`. The core loops over SLOT positions (64 at `tul.max_slots: 64`) while `base.yaml` ships `hca_compress_ratio: 256`, so `n_blocks = 0`, `fused_hca_attention` has nothing to attend to, and `_gate_combine_up` blends `g_comp ~ 0.50` into a tensor that is exactly 0.0000 — three of six core blocks at half their attention output, silently, for the whole run. The token path at the same weights has 4 blocks and `\|out_comp\| ~ 1030`. Measured 2026-08-25 with `lab/divergence/attn_sink_probe.py --geometry --token-path`. The fix is `model.core_hca_compress_ratio` (default `null` = today's behaviour, scoped to the core); whether it ships depends on the H24 arm. Guarded by a STRICT xfail so the day it starts holding, the suite says so: `test_core_hca_branch_gets_at_least_one_block_at_the_tul_slot_budget`. The knob's own wiring is covered by `tests/test_core_hca_ratio.py`. Deploy (`seq_len 4096`, 512 slots) is unaffected. |

## 6c. TUL gate invariants (LIVE — see `tul-gate-spec.md` §9)

Built 2026-08-22. `pytest tests/` → 189 pass; the named tests are in
`tests/test_tul_gate.py`, one per row. Every row was mutation-checked: the code was
deliberately broken and the named test observed to go red (15 mutations, 15 caught).

| Invariant | Why / the test that fails |
| --- | --- |
| **`tul.gate: false` builds NOTHING** — no parameter, no layout field, no random draw — and `gate_lambda = 0` with `gate_budget_cond = false` is bit-identical to arm A1: same loss, same gradients. | `nn.Linear`/`nn.Embedding` draw from the global RNG in `reset_parameters`; a draw would advance the stream and change every later Poisson depth and dropout mask. The head is therefore a rank-1 linear written as two zero parameters and the budget table a zero `Parameter`. `test_building_the_gate_draws_no_random_number`, `test_lambda_zero_and_no_budget_cond_is_bit_identical_to_a1`, `test_gate_off_leaves_the_packer_byte_identical`. |
| **A loss now touches the slot core state, and §6b's "slot core states have no loss" is narrowed on purpose.** A scalar/discrete READOUT is allowed; a vector regression onto a target representation stays forbidden. | The §6b citation set (MegaByte, H-Net, LD4LG, Pred-Sent, the LTD think-position failure, Block Transformer §4.2) is about reconstructing the latent, not reading a scalar off it. Tested structurally, not asserted: `test_the_gate_reads_a_scalar_and_regresses_nothing_vector_valued`. |
| The length label is the **NEXT** span's length, not the slot's own. | Slot i sits after span i, so causal attention lets it condition only span i+1 — the span generation asks it about. Grading span i would read out the past. `test_the_label_is_the_NEXT_span_length`. |
| `span_len` is 0 and `len_supervised` is False at every pad slot, and pad slots contribute exactly zero to the gate loss. | A pad slot's `slot_index` is 0, so a missing validity mask trains on row 0's first span — the existing §6b pad-slot trap. `test_pad_slots_carry_no_label`, `test_pad_slots_contribute_exactly_zero_to_the_gate_loss`. |
| An RNG-truncated slot, and the row's last slot, are conditioned on their realised length but never GRADED on it. | The truncation point and the row boundary are ours, not the data's. Grading a length head on them is label noise. `test_an_rng_truncated_span_is_not_graded`, `test_an_unsupervised_slot_supervises_the_zeros_but_not_the_value`. |
| No label lands on the sigmoid's asymptote: `gate_k_max > span_cap`, and the decoded `k` is clamped to `span_cap`. | Measured: at `k_max = span_cap` **24.5 %** of real labels are exactly 1.0, where the gradient vanishes. A decoded `k` above `span_cap` would index a budget row no example ever trains and hand the coda a zero vector. `test_no_label_lands_on_the_sigmoid_asymptote`, `test_choose_k_never_exceeds_the_rule_s_span_cap`, `test_the_open_tail_label_is_clamped_to_span_cap_not_to_k_max`. |
| The coda gets the **realised** length in training and the **predicted** length at generation, never a mixture. The layout carrying a label IS the switch. | Mixing makes the LM loss chase the gate's error. Scheduled sampling is §12's unbuilt key and RAISES. `test_training_uses_the_realised_length_and_generation_the_predicted_one`. |
| `g_traj` leaves the core region as a RETURN VALUE, and the readout runs OUTSIDE the checkpoint. | The `ret_capture` lesson: a side channel is not checkpoint-safe. Outside the checkpoint the gate shapes the core on exactly the truncated-BPTT window the token loss uses, and the head is still supervised on the frozen iterations — otherwise the ~28 % of slots whose depth falls inside that window would have no gradient at all. `test_the_gate_gradient_reaches_the_core_inside_the_bptt_window`. |
| Truncation is a pure INSERTION: both halves clear `min_span`, and restarting the rule at an inserted cut still yields the same next DATA boundary. | This is what makes a wrong `k` at generation cost one span's quality instead of desynchronising every later span in the row. `test_truncation_is_consistent_with_the_rule`, `test_both_halves_of_a_truncation_clear_min_span`, `test_a_forced_cut_leaves_the_rule_in_the_same_state_as_a_real_one`. |
| Every gate parameter is in the optimizer's **no-decay** group, and the run REFUSES to start if any gate parameter is missing from the optimizer, sits at `lr = 0`, or has a smaller travel budget than its targets need. | Weight decay on a zero-init readout direction pulls against the only gradient it has. The predecessor lost a whole ladder to a bias that moved 0.04 % of what it needed, with nothing in the run saying so. `test_the_gate_lands_in_the_no_decay_optimizer_group`, `test_the_audit_refuses_a_gate_that_is_not_in_the_optimizer`, `test_the_audit_refuses_a_step_budget_too_small_to_move_the_gate`, `test_a_frozen_gate_direction_fails_the_alive_check`. |
| Val runs the DATA's segmentation: `gate_truncate_p` is 0 on the val loader whatever the train loader uses. | A val CE that moves with our augmentation seed is not comparable to the reference arm's. |
| `gate_drives_depth` is EVAL ONLY and caps at `slot_max_depth`; every slot emits `k ≥ 1`. | Training teacher-forces the depth (§4), so the two arms are one checkpoint scored twice. A slot that never halts must not hang the generator. `test_halt_is_eval_only`, `test_halt_depths_stay_inside_one_and_max_depth`, `test_halt_and_fixed_depth_share_every_weight`. |
| A config key for an unimplemented gate arm RAISES. | `gate_scheduled_sampling`, `gate_stop_head`, `gate_ponder_lambda` are specified (§12) and not built. `test_unimplemented_gate_keys_raise`. |

## 7. What not to “fix”

- Removing process-global `force_eager` without a per-module replacement that
  preserves reference A/B and Dynamo fences.
- Enabling `compile_mode=reduce-overhead` when on constrained hardware (CUDA graphs + eval OOMs.
- Setting `fullgraph=True` on the looped core.
- Calling `carve()` while density is still ~1.0 (produces a “sparse” model with
  K/C=1.0).
- Silent fallbacks when a kernel, dataset path, or checkpoint topology fails.
- Turning `tul.gate_train_zeros` back on because "the predecessor did it that way": the
  per-slot depth is a Poisson draw the head cannot observe, so that target's optimum is
  the hazard and the length is scaled away (measured `k = 5.00` against gold `18.98`;
  `tul-gate-spec.md` §6).
- Reverting `@kernel_fence` to hard `@torch.compiler.disable` (kills graph
  composition), or flipping `MORPH_DYNAMO_FENCE=0` into the default without an
  fp32 parity gate on the target torch version.
- “Simplifying” `gla.py` `_readout` back to `gn(o.transpose(1,2))` — it looks
  more idiomatic and it is a causality leak (§5).
- Removing the `(… * scale).to(tl.float32)` dtype pins in CSA/HCA kernels as
  “redundant casts” (§5).

Public contract tests under `tests/test_lifecycle_*.py` cover a minimal subset.
Longer campaign logs and gate scripts live under gitignored `ignore/`.
