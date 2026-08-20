# Agent Note: Perf Pass Focus4 Remaining Cuts

Status: proposed

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Focus4-Remaining-Cuts.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Focus 4: Remaining Class-A Launch Cuts (2026-07-03)

CPU analysis + worktree prototype ONLY (single 5090 + UPS → orchestrator
serializes GPU gates; the GPU was NOT used for gating — see honest edges for
one 10s pytest caveat). Everything labeled MEASURED was run; everything
labeled ESTIMATE needs the GPU gate.

**Worktree:** `.claude/worktrees/focus4-remaining-cuts` (branch
`perf/focus4-remaining-cuts`, off HEAD `4048497`, NOT committed).
Files changed: `morph/model/gla.py` (+58/−10), `morph/model/routing.py`
(+83/−15), `morph/model/attention.py` (+51/−7), NEW
`morph/kernels/triton/fused_router_tail.py` (~250 lines).
**Parity scripts + archived results** in the worktree's `scratchpad/`:
`parity_gla_fused_proj.py`, `parity_router_tail.py`, `parity_rope_cache.py`,
`parity_full_model.py`, `op_census_focus4.py`, `gpu_probe_focus4.py`
(+ `*_results.txt`).

## Ranked summary

| # | target | mechanism | Δ (per step) | class | CPU parity (autocast-tested) | GPU-blind edges |
|---|---|---|---|---|---|---|
| 1 | **GLA q/k/v/g/r superblock** (gla.py) | concat-N GEMM + split views, the landed attention-superblock template — batched **5** projections, not 3 (g_proj and r_proj share the same x) | **MEASURED −4 mm/fwd-exec, −12 mm/grad-exec** (18→6 fwd+bwd); ≈ **−90 to −120 aten::mm/step** at 13.6 fwd execs (above the mandate's −50–80 because g/r joined) | **A\*** | fwd **bitwise in every arm** incl. autocast; all param grads bitwise; only x.grad reassoc ≤3.15·eps fp32 / **≤0.79·bf16-eps under autocast** | cuBLAS algo bitwise-ness at [16384×768]·[768×3840] |
| 2 | **Router elementwise tail** (routing.py + fused_router_tail.py) | 3 elementwise Triton fusions (K1 pk-add+bias, K2 sub+relu, K3 normalize+mask) around UNTOUCHED aten reductions; analytic aten backward replicating eager op-for-op | ESTIMATE **−4 kernel launches/router fwd pass** (7 aten → 3 Triton in the 1:1 branch) ≈ **−450/step** at ~113 calls; bwd exactly neutral (MEASURED Δ0 on the fallback census) | **A** (elementwise = provably bitwise; reductions + scalar-div untouched) | gates/aux/EMA/**all grads bitwise** in 32 arms × mixed-dtype autocast; caught+fixed a real cast-order bug in K1 bwd via the autocast arm | **Triton kernels never compiled/ran** (no GPU) — first-run compile + fwd bitwise-ness need the probe |
| 3 | **RoPE cos/sin cast cache** (attention.py CoPEEmbedding) | per-(dtype,device,data_ptr) cache of the full-buffer cast; eager sites only (`CoPEEmbedding.forward`, `_cca_q_only`); the fused-prologue KERNEL path deliberately untouched (its bwd upcasts fp32 loads — pre-cast bf16 would change bits) | MEASURED −2 `_to_copy` + −2 `copy_` per call (warm) ≈ **−120 to −240 launches/step** at the cited ~120 casts | **A** | bitwise everywhere incl. `set_context` invalidation-vs-fresh-module and autocast `_cca_q_only` fwd+grads | none beyond the loss-trace formality |

Recommended landing order = table order. #1 is the only one with likely
visible GPU-region ms; #2/#3 are launch-stack contributions (Command-Buffer-
Full relief), invisible individually at ±14 ms wall noise — same story as the
injection hoist.

## Per-mechanism kill-switches (all default ON, all with in-process setters)

- `MORPH_FUSED_GLA_PROJ=0` / `gla.set_fused_gla_proj(False)`
- `MORPH_FUSED_ROUTER_TAIL=0` / `routing.set_fused_router_tail(False)`
- `MORPH_ROPE_CAST_CACHE=0` / `attention.set_rope_cast_cache(False)`

Patch-off arms are **proven bitwise no-ops** (fwd + all grads torch.equal vs
git-HEAD copies) in all three module parities → A/B isolation is clean.

## 1. GLA superblock — details

`_project` now does ONE `F.linear(x, cat(q,k,v,g,r weights).to(x.dtype))` +
`split` views; `r_pre` threads to `_readout` (o_proj stays separate —
dep-chained). `w.to(x.dtype)` targets the GEMM-consumer dtype exactly like the
landed attention template; under autocast F.linear casts both operands either
way (the q‖k-conv lesson does not bite here — no custom kernel consumes the
projections directly; `fused_gla` takes the post-view q/k/v/log_alpha and
`.contiguous()`es internally, dtypes unchanged). No new params; state_dict
keys unchanged. Covers training (chunked/kernel), the recurrent oracle, and
the kv_cache inference path (all call the same `forward`).

Parity (`parity_gla_fused_proj.py`, single-threaded CPU, base-rerun noise
floor 0): d768/H12/C256 + d64/H4/C16 × {chunked, recurrent} × {S%C==0,
S%C≠0} × {fp32, autocast+fp32-x, autocast+bf16-x} × {S0=None, random}:
**forward + final_state bitwise in EVERY fused arm; 10/11 grads bitwise; the
one non-bitwise grad is x.grad** — one dX reduction over 5·d instead of an
autograd sum of five, ≤0.79 bf16-ulp@scale under autocast (fp32-eps
normalization misleads here: 4.1e4·fp32-eps == 0.63·bf16-eps). Same class as
the landed superblock's dX.

## 2. Router tail — details

Deploy branch confirmed from base.yaml: n_clusters=16, n_sub_keys=4 →
n_products==G (1:1), activation_k=8. Trace evidence the tail runs EAGER aten
in the routed regime (r4: aten::topk 1134 = 113/step router calls; zero
inductor `triton_poi_*` rows) — so fusing it is real, not double-fusing.

Fusion boundary discipline (why this is class A and not a numerics gamble):
- K1/K2/K3 are pure elementwise → per-element op sequences identical to eager
  (each op correctly rounded) → bitwise by construction. K1 replicates the
  autocast promotion exactly (bf16 add via exact fp32 sum + round, then fp32
  bias add).
- topk, gate_sum, batch-load mean, var: NEVER fused, fwd or bwd.
- `activation_k / gate_sum` scalar-div: NOT fused (its aten lowering is
  version-dependent); computed eager, passed into K3; d(t) flows back through
  the eager clamp/div autograd nodes.
- Backward = analytic aten replicating eager op-for-op (incl. the exact
  `aten.threshold_backward` op). The autocast parity arm caught a REAL bug
  here: the promotion boundary casts the grad to bf16 BEFORE the broadcast
  reductions (cast-then-sum), not after — fixed and now bitwise.
- `mark_non_differentiable(mask)` + `set_materialize_grads(False)` keeps the
  grad graph and grad-None pattern identical (hard-checked in parity).
- Entry points are `@torch.compiler.disable`-fenced (precedent:
  `_cca_prologue_dispatch`).

Parity (`parity_router_tail.py`): 4 branch-configs (1:1 deploy, topk-branch,
wrap-branch, k==G edge) × {2D, 3D} × {fp32, autocast-cpu bf16 (fp32 weights +
bf16 GEMM activations — the REAL mixed-dtype flow)} × iter_idx × 3 sequential
calls: **gates, aux_loss, group_load_ema, every param grad, x.grad — ALL
torch.equal in every arm** (0 FAIL); base-rerun determinism arm exactly 0.

Census (MEASURED, CPU fallback path): fwd Δ0, fwd+bwd Δ0 — the fallback is
op-neutral, so CPU/force_eager regimes lose nothing. The −4/fwd-pass launch
cut exists ONLY via the Triton path (GPU): eager 1:1 tail = bcast-add,
bias-add, sub, relu, mul, gt, float-cast (7 launches) → K1+K2+K3 (3). At
~113 router fwd passes/step (incl. ckpt recompute) ≈ −450/step, plus the
skipped intermediates' alloc/free traffic. ESTIMATE — kernels never ran.

## 3. RoPE cast cache — details

Cache keyed `(dtype, device, cos_cached.data_ptr())`, invalidated by
`_build_cache` (init + `set_context`) and by module moves (data_ptr change);
same-dtype requests bypass (`.to` is already a no-op). Consumers: the
`CoPEEmbedding.forward` eager path and `_cca_q_only` (CLA reuse — the actual
~120 casts/step site; the kernel prologue never casts eagerly).
`parity_rope_cache.py`: bitwise across dtypes/S/cache-hit/toggle-off;
post-`set_context` == fresh module == HEAD; `_cca_q_only` autocast fwd+grads
bitwise on repeat calls (warm cache).

## Full-model integration (MEASURED)

`parity_full_model.py`: 0.6M MORPHTransformer (embed → prelude → Poisson core
loop with no_grad + checkpointed grad iters + recompute → coda → CE), GLA
retention attached, **iteration-aware ReMoE routers enabled on every MLP**
(1:1 branch), loss = CE + collected routing aux. dropout 0.0 AND 0.1:
**loss BITWISE identical** all-three-on vs all-off (dropout=0.1 passing
proves the RNG stream through checkpoint recompute is untouched — the fused
tail adds/removes zero RNG ops); grad-None pattern unchanged (292/292);
whole-model grad max|Δ| ≤ 1.19e-7 fp32 (the GLA dX reassociation propagating).
`pytest tests/ -x -q` on the worktree: **48 passed, 0 failed** (10.45s).

## GPU gate (for the orchestrator, serialized — run in this order)

1. **Module probe (~2 min, run FIRST):**
   `python scratchpad/gpu_probe_focus4.py` (from the worktree, venv python).
   First Triton compile of the 3 router kernels + torch.equal fwd/grads at
   B4/S4096 bf16-autocast; GLA fwd torch.equal both chunked and kernel modes;
   x.grad ≤ 2 bf16-ulp; rope bitwise. Router/rope: ANY fwd diff is a bug
   (elementwise) → kill that switch. GLA fwd not bitwise → record ULP, expect
   ≤ few eps (cuBLAS split-K on the fatter N=3840).
2. **Loss-trace noise-floor gate (the landing test):**
   ```
   MORPH_EXACT_TRACE=ignore/perf/focus4.trace MORPH_PERF_REGIONS=1 \
   python -m morph.training.train \
     training.resume=checkpoints/morph/tst_stp_on_50k/step_36000.pt \
     training.resume_fresh_optimizer=true training.optimizer=ademamix_b1zero \
     +training.seed=1234
   ```
   (ckpt_grad_iters=-1 is the config default; use absolute paths if launching
   from the worktree.) Assert loss trace ≤ 6e-4 by step 11 vs
   `ignore/perf/ckpt_base.trace`.
3. **A/B isolation if the gate fails:** flip `MORPH_FUSED_ROUTER_TAIL=0`
   first (only GPU-unexercised code), then `MORPH_FUSED_GLA_PROJ=0`, then
   `MORPH_ROPE_CAST_CACHE=0`.
4. **Perf readout:** MORPH_PERF_REGIONS + 10-step profiler capture; expect
   aten::mm ~1493 → ~1400/step, `aten::_to_copy` −100-250, cudaLaunchKernel
   −500 to −700 total; do NOT expect wall-step movement (±14 ms noise) —
   attribute via GPU-region time / op census, per the Overnight methodology.

## Honest edges (NOT verified)

- The three `fused_router_tail` Triton kernels have **never been compiled or
  launched** (no GPU here). Elementwise bitwise-ness is argued, not GPU-
  measured; a compile-time typo would surface at probe step 1, not silently.
- GLA fused-GEMM GPU forward bitwise-ness (cuBLAS shape-dependent algos) —
  CPU-proven only; probe decides.
- All launch/step deltas are trace-arithmetic ESTIMATES; no ms is claimed.
- `mode="kernel"` GLA path exercised on CPU only via its eager fallback
  branch (the probe covers the real kernel at DH=64).
- torch.compile: the router sits inside the compiled-MLP wrap
  (`layer.mlp = torch.compile(...)`, train.py:1164). Trace evidence says the
  tail runs eager aten in the routed regime, and the new entry points are
  compiler.disable-fenced, but a compile-fence regression (extra graph break
  → recompile-guard trip) is possible in principle — the fence tests pass
  (in the 48) and the gate run would surface it as a startup recompile.
- One caveat on GPU non-use: `pytest tests/` ran 48/48 (the 2 previously-
  skipped CUDA-gated fence tests executed, ~10s) — I checked nvidia-smi
  first: GPU was idle (1% util, desktop-only VRAM), no training job was
  touched.
