# MORPH — Resume + Overnight Monitoring (2026-06-01, post-compaction)

REPO: `/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`
PY: `/home/wolfe/.venv/bin/python`  (PYTHONPATH=. from repo root). GPU: RTX 5090 sm_120, bf16 autocast (model is NOT .to(bf16); params/buffers stay fp32).
Branch `remove-jepa-memory`, pushed through commit ~`a64f48c`+ (toggle/fixes/JAX). 5 commits on origin.

## ⚡ IMMEDIATE TASK: monitor the dense A/B campaign overnight (Wolfe is asleep)
A 3-run chain is RUNNING, harness-tracked as background task **bl3wmwvjf** (it notifies me when run_ab.sh EXITS = all 3 done, ~4-6h). BUT run_ab.sh has no `set -e`, so a single run crashing does NOT stop the chain — it advances. So I MUST POLL to catch mid-campaign failure early; the harness notification alone only fires at the very end.

**Launch an overnight watcher** (harness-tracked, run_in_background, NO trailing `&`) that polls every ~90s and EXITS EARLY (to notify me) on trouble, else on done:
- exit if `grep -q "AB CAMPAIGN DONE" ignore/ab_campaign.log`
- exit if any of `ignore/ab_*.log` contains `Traceback|OutOfMemoryError|CUDA error|out of memory|nan` (recent)
- exit if NO `train.py` proc alive AND campaign not DONE (crash between runs is normal-brief, so require ~3 consecutive misses)
- cap ~8h
On exit: read the relevant log tail, diagnose, and react (see failure modes). Then report to Wolfe in the morning.

### The runs (chained, in `ignore/run_ab.sh`), flat LR (lr=min_lr=1e-4, warmup 1500), pruning DISABLED (prune_start=999999), dense:
1. `dense15k_eager_b4`  → log `ignore/ab_eager_b4.log`   (use_kernels=false, B4/S4096) — BASELINE. **~31 GiB, OOM risk** (esp. eval@1500: full [4,4096,V] logits).
2. `dense15k_kernels_b4`→ log `ignore/ab_kernels_b4.log` (use_kernels=true,  B4/S4096) — matched A/B (~12.5 GiB).
3. `dense15k_kernels_b8`→ log `ignore/ab_kernels_b8.log` (use_kernels=true,  B8/S4096) — headroom (~25 GiB).
Campaign echo log: `ignore/ab_campaign.log`. wandb project `morph` (entity adew-me), online. Metrics logged every 20 steps: `perf/steps_per_sec, perf/tokens_per_sec, perf/peak_mem_alloc_mib, perf/peak_mem_reserved_mib, train/loss, train/ppl`.

### Failure modes + responses
- **eager_b4 OOM** (most likely, at ~31 GiB / eval): EXPECTED-ish finding ("eager can't fit B4/S4096 + eval on 5090"). Don't panic — the chain advances to kernel runs. Note it as a result (kernels enable what eager can't). If it OOMs at step-0/compile, consider relaunching eager at B2 to still get a baseline curve — but ASK Wolfe first / note it.
- **kernels_b8 OOM** (~25 GiB + desktop ~1.4 GiB): possible. If so, that's the headroom ceiling; note it, not a bug.
- **NaN loss**: Lorentz `acosh` backward can NaN — but only under model.to(bf16), which we DON'T do (autocast keeps embed fp32). If NaN appears in the autocast path it's a real new bug → stop, investigate. (See arcsinh rewrite in queue.)
- **Data stall**: OpenWebText is STREAMING from HF (only partly cached) → early throughput is network-paced; not a crash.
- **Desktop apps** hold ~1.4 GiB GPU (vesktop/godot/dotnet) — factor into OOM math.

### Expected healthy signal
loss starts ~11.6, should fall steadily (flat LR 1e-4). sps: probe showed ~1.36 warmup-incl at B4; steady likely ~2. tokens/sec ~22k at B4. peak_mem_alloc: eager ~30 GiB, kernels ~12.5 GiB (THE headline A/B number Wolfe wants).

## ✅ DONE this session (all committed + pushed)
- Removed JEPA/neural-memory (PT+JAX); fused CCA attention (4 Triton kernels, eager-ref fallback); Z3 proofs in tile-prover/.
- **Efficiency levers (verified, gated):** fused chunked CE (`morph/model/fused_ce.py`, 19× CE mem, training-only, logits=None in train); x0-projection hoist (bit-exact); active-set loop shrinking (`transformer.py`, ~25% fwd FLOPs, fp32-dense bit-exact logic proof; reorders dropout RNG); 8-bit AdamW (`optimizer.py`, `training.adam8bit`, sm_120-verified, embed state fp32).
- **`use_kernels` master toggle** (`MORPHConfig`+base.yaml+`morph/kernels/triton/_eager_flag.py`): false→eager refs+full-logits CE. x0-hoist+active-set stay both arms.
- **Fixes:** core MLPs `torch.compile(dynamic=True)` (active-set variable batch was recompiling); `block_sparse.py topology_step` scatter dtype = buffer dtype (bf16-robust; only mattered under model.to(bf16)).
- **Compaction verified** (subagent): prune→compact→sparse-Triton works; active-set variable-batch × sparse BlockELL OK at B=4..1.
- **JAX/TPU mirror:** x0-hoist (bit-exact) + chunked CE (lax.scan; XLA does NOT auto-avoid [N,V]); active-set NOT ported (static-shape; where-mask is TPU idiom). Verified JAX-CPU.
- Gates (all green, in `ignore/`): verify_removal, verify_active_set, verify_integrated_step, verify_adam8bit, bench_efficiency parity, fused_ce self-test.

## 📋 QUEUE (after A/B frees the GPU — one model per GPU)
Per Wolfe's sequence: optimizer/ternary tests → pruning. End goal: ternary shadow + 8-bit AdamW together.
1. **`ignore/verify_ternary_8bit.py`** — STAGED, run it: TernaryShadowOptimizer wrapping AdamW8bit (composition + uint8 state + shadows update + finite). (#189)
2. **Pruning run** — real training with pruning enabled (compaction now verified). Watch topology_step (fp32-safe in autocast path).
3. **Lorentz dormancy experiments** (Wolfe flagged: embeddings are FLAT, radius 0.069 ≈ init, geometrically ~Euclidean):
   - (a) measure `‖lor_embed.space_embed.weight‖` (geodesic radius) on the dense 15k checkpoint — did it grow off ~0.07? Script pattern in `ignore/lorentz_bag_tradeoff.py` / the inline norm check.
   - (b) ABLATION: `model.lorentz_fraction=0` (all-euclidean) vs current, equal steps, val PPL — does the Lorentz channel earn its keep at all? It's doubly suppressed: tiny radius (flat) + 14× smaller init norm than euclidean (drowned post-RMSNorm).
   - Diagnosis: likely dormant near-Euclidean channel (no learnable curvature, no hierarchical loss pressure). Gate any "fix" on the ablation.
4. **arcsinh log-map rewrite** (`embeddings.py _log_map_origin`): replace `acosh(x0)/sqrt(x0²-1)` with the equal-value `arcsinh(‖xs‖)·x̂s` — numerically robust at any radius, bf16-NaN-safe. Do regardless (strictly better); unblocks larger Lorentz radius if we pursue (3).
5. **TST** (Wolfe asked: NO dedicated kernel). = (i) mean-bag embedding OUTPUTS (uniform euc+lor; tangent-mean is cheapest AND closest-to-Fréchet — tradeoff negligible, measured 0.04% at current radius), (ii) multi-hot CE = ~10-line extension of fused_ce (gather s targets, mean: `logsumexp - (1/s)Σ logits[bag_i]`), (iii) phase scheduler. Open modeling Q: bigram-in-phase-1 (superposed token has no single id to hash).
6. Optional: B4 selective-activation-checkpointing (#184, regime-dependent memory↔speed knob), gate-combine fusion (#179).

## Landmines / key facts
- Triton sm_120: num_stages=1, num_warps=8, no TMA; interior-axis 3-D `tl.sum` doubles output → use `tl.dot`.
- bf16 fused-kernel gate: fwd-cos>0.9999 + grad-cos>0.995 (2e-2 max-err is below bf16 floor).
- fused-CE: chunk over ROWS (token dim) = near-exact (each logit independent); only loss/grad_w reduce reorders. chunk grad matmuls in bf16 (match autocast eager), grad_w accumulate fp32. `model.ce_chunk_size` config knob (default 1024).
- active-set: dropout RNG differs from old where-loop (different valid realization) — NOT a bug; eval (equal depths) is bit-exact.
- Commit/push ONLY when Wolfe asks. wandb FULL config (Hydra handles). Don't run 2 models on one GPU.
- Memory file: `[memory]/project_morph_efficiency.md`. Plan: `Ai-notes/05-31-2026/MORPH-Full-Kernel-Plan/plan.md`.
