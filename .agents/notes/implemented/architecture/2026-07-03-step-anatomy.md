# Agent Note: Step Anatomy

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Step-Anatomy.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Phase 0: Step Anatomy (2026-07-03)

**Constraints :** HARD — bit-exact: same config+seed ⇒ identical loss/param
trajectory vs current main. Numerics-changing speedups are surfaced with evidence for
his opt-in testing (z3/tile-prover proofs where applicable), never shipped as default.
SOFT — checkpoint interop with past checkpoints. Anything needing an *ablation* is
parked (list at bottom).

**Target workload:** local `base.yaml` — 276M-class d768, 3:6:3 Parcae loop
(Poisson T mean 6 max 8, BPTT 4, all grad-iters checkpointed), seq 4096, batch 4,
quant stack ON (ternary backbone STE + int6 embeds), TST bag 6,
`use_kernels=true`, compile=MLPs-only. ~1.2 sps today.
**Optimizer under test: `ademamix_b1zero` FUSED (dynamic qmap)** — 2026-07-03:
"we are fully onboard the ademamix_b1zero fused"; base.yaml's `optimizer: adamw +
adam8bit` default is NOT the deploy truth. All bench runs pass
`training.optimizer=ademamix_b1zero` (fused=true, fused_dynamic_qmap=true defaults).
The opt.step is therefore a custom Triton kernel → direct ncu target.
**Cloud target:** dual RTX Pro 6000 Blackwell = GB202/SM120 (same SM arch as the 5090
→ kernel tuning transfers; deltas are 96 GB VRAM ×2 and future DDP).

## The four regimes of a base.yaml run (profile each; weight by wall-clock)

| # | steps | share | what's different in the hot path |
|---|---|---|---|
| 1 | 0–3k (dense+TST) | 3% | dense cuBLAS MLPs, ternary STE parametrization, TST bag-6 multi-hot CE, `accumulate_scores` every step |
| 2 | 3k–29k (prune ramp, still TST till 30k) | 26% | + `apply_prune_mask` does real weight/grad masking (2 muls × 24 layers), prune events every 167 steps |
| 3 | 29k–30k (carved, pre-route) | 1% | MORTAR BCSR stk dds kernel replaces dense GEMMs; scoring/masking OFF |
| 4 | 30k–100k (routed ReMoE recovery) | **70%** | + TileRouter on all 12 MLPs, every core loop iteration (incl. checkpoint recompute); standard NTP (bag 0) |

Regime-4 resume seed for benching: `checkpoints/morph/tst_stp_on_50k/step_36000.pt`
(verified: `pruning_compact=True, pruning_routed=True`, 24 mortar layers, 12 routers).
Use `training.resume=<path> training.resume_fresh_optimizer=true` (fork-continue —
skips the deterministic data replay, rebuilds topology, fresh optimizer).

## Per-step anatomy (train.py:1495–1700)

```
zero_grad(set_to_none) → next(train_loader) → x,y.to(device) [DATA]
→ autocast bf16 forward (fwd) [FWD]
→ [routed] collect_routing_aux_losses (model.modules walk)
→ scaler.scale(loss).backward [BWD]
→ pruning.step(model, step) [PRUNE-BOOKKEEP]
→ scaler.unscale_ → clip_grad_norm_ (all ~450 tensors, foreach)
→ scaler.step (found_inf .item SYNC) → scaler.update [OPT]
→ every 20: loss.item; every 500: eval (20 batches); every 2500: sync torch.save ~2GB
```

Forward internals (transformer.py `_forward_single`):
- embed (hybrid euc+Lorentz; `lm_weight` rebuilds the Lorentz tangent [V,192] per fwd)
- HC expand [B,S,4,768] carrier → prelude ×3 → core loop (sort by depth, active-set
 prefix; `depths.max.item` = 1 SYNC; `active_counts .tolist` = 1 SYNC)
 - per iteration: DiagonalInjection + 6 core blocks; first (T−4) iters no_grad,
 last 4 checkpointed (recomputed in backward ⇒ ~2× fwd compute on grad iters)
 - each block: HC pre-map (fused) → attention (CCA conv + fused prologue + fused
 CSA/HCA + fused window + gate/up) [+ GLA retention on layer 1, fused DH=64]
 → HC post → MLP (MortarLinear dense/BCSR [+ router in regime 4]) → HC post
- coda ×3 → HC mean-reduce → lm_mixer → final_norm → fused chunked CE
 (16 chunks @ ce_chunk_size=1024; `int(valid.sum.item)` = 1 SYNC)

## GPU→CPU sync inventory (hot path, per step)

| site | count | fix class |
|---|---|---|
| `block_sparse.apply_prune_mask`: `bool(self._prune_mask.all)` early-exit | **24/step in regime 1** (pre-prune: elem mask None, mask all-alive) | A — cache all-alive as a CPU bool, invalidate on prune_step_blocks/resume |
| `fused_ce`: `int(valid.sum.item)` | 1/fwd | A — keep n_valid on-device |
| `transformer`: `depths.max.item` | 1/fwd | needed for loop control; alternatives change RNG → parked |
| `transformer`: `active_counts...tolist` | 1/fwd | already batched (OPT1); OK |
| `scaler.step` found_inf check | 1/step | A-ish — see GradScaler item below |
| `pruning.log_stats` / prune events | every 167 steps only | ignore |

## Candidate list (pre-measurement; ranked after Phase 1 numbers)

### Class A — bit-exact by construction (the Phase-2 menu)
1. **apply_prune_mask sync storm** (24 syncs/step, regimes 1–2): CPU-cached
 "nothing pruned yet" flag. Same pattern as the Olympiad AnswerTokenWeighter fix.
2. **Data pipeline**: base OWT loader is synchronous in-process HF tokenization with a
 Python-list buffer (O(n) `buf = buf[chunk_len:]` slice) and blocking unpinned
 `.to(device)`. Background prefetch thread (same pattern as
 `data_placement.Prefetcher` — bit-identical stream while one generator lives) +
 pinned staging + `non_blocking=True`. All regimes.
3. **pruning.step bookkeeping**: `_find_cms_layers` walks `named_modules` EVERY
 step (all regimes) — cache the layer list (invalidate at carve). Pre-carve:
 `accumulate_scores` builds a `TopologyScorer` object per layer per step and
 launches per-tile reduction kernels ×24 layers; batchable / stream-able. In taylor
 mode `self.weight` access re-runs the ternary STE parametrization per layer per
 step (extra quantize kernels) — reuse the shadow leaf + realized ternary.
4. **Async checkpoint save**: `torch.save` ~2 GB synchronous every 2500 steps
 (30–60 s). Snapshot state_dict to pinned CPU, write on a thread.
5. **Ternary STE recompute**: parametrization re-quantizes on every `.weight`
 access — core-layer weights are read ~(T + n_ckpt_recompute) ≈ 10–12×/step.
 `parametrize.cached` around fwd+bwd computes once. Grad-accumulation-order
 caveat: cached = one graph node fanned out vs N independent recomputes — MUST be
 verified bit-exact by the harness before shipping (may land in class B).
6. **Triton launch-config retuning** (`num_warps`/`num_stages`/grid swizzle) on the
 fused attention/HC/GLA kernels: does NOT change FP accumulation order → bit-exact.
 BLOCK-size changes are case-by-case (can split reductions) — classify per kernel,
 prove scheduling properties with tile-prover.
7. **fused_ce `n_valid` on-device** (kills the 1 sync; division by identical value).
8. **`ckpt_grad_iters` tune-down** (config-documented "exact — never changes
 gradients"): 5090 has headroom at mb4? Measure VRAM; un-checkpointing the last
 grad iters kills recompute. NOTE must VERIFY bit-exactness claim in the harness
 (dropout RNG consumption inside checkpoint recompute!). If dropout>0 makes
 recompute draw RNG, changing n_ckpt changes RNG stream → would be config-knob-only.
9. **collect_routing_aux_losses / retention-gate logging walks**: modules scans per
 step (regime 4) / per log step; cache module lists.
10. **Eval loop `losses.append(out["loss"].item)`** — per-eval-batch sync; batch to
 one stack+item. Eval-only, minor.

### Class B — numerics-changing (surface with evidence; opt-in flag + parity gate)
- **GradScaler removal for bf16**: scale factors are powers of 2 (65536, ×2/×0.5) so
 scale/unscale is exponent-only and *likely* bit-exact absent overflow — but inf
 detection/step-skip semantics differ. Verify empirically; if the harness shows
 bit-identical 200-step trajectory, promote to class A. Kills 1 sync + full-grad
 unscale pass + scale kernel per step.
- **Liger-style one-pass fused linear-CE Triton kernel** (~20 ms/step estimate from
 the seed-run profile): different reduction order vs the chunked eager path.
- **Router in bf16** (regime 4): fp32 768×768 GEMM × ~66 calls/step ≈ 1.3 TFLOP fp32.
 bf16 would halve-plus it, but changes routing numerics.
- **ce_chunk_size default change** (1024 → 8192 measured 59→51 ms at 16k tokens on
 the seed): changes fp32 loss/grad_w accumulation order across chunks. It's a config
 knob — per-run opt-in is already legitimate; just not a silent default flip.
- **CPU-side Poisson depth sampling** (removes `depths.max.item` sync): different
 RNG stream → trajectory change.

### Parked — needs ablation (off-limits this arc)
CLA re-ablation (plumbing already live in attention.py: `cla_capture`/`cla_kv` +
`_cca_q_only`; cross-loop-iteration KV reuse; prior arm `checkpoints/morph/cla_iter1_b4`),
retention_chunk / retention layer placement, csa_compress_ratio / top_k / window_size,
hca_compress_ratio, tst_bag_size, hc_cayley_iters / hc_streams, seq-length warmup.
(Optimizer is NOT an open question — ademamix_b1zero fused is the settled deploy choice.)

## Open measurement questions for Phase 1
- Where do the ms actually go at seq4k? (fwd/bwd/opt/data/bookkeep split via CUDA events)
- Regime 4: router cost + stk dds BCSR efficiency (ncu: SOL, occupancy, smem) vs the
 3.09× claim at Gate G1 (that was MLP-isolated, not in-model).
- Regime 1–2: accumulate_scores + apply_prune_mask measured cost; TST bag-6 CE (MCE) cost.
- Checkpoint recompute share: grad-iters recompute ≈ how many ms (bounds what
 ckpt_grad_iters can buy)?
- torch.compile MLP-only: how much is dynamo guard-eval overhead per call at 4k?
- b1zero fused opt.step: per-tensor kernel launches (~450 tensors) → ms? kernel SOL?
- fused window/CSA/HCA/HC/GLA kernels: ncu SOL on SM120 + num_warps/num_stages headroom.
