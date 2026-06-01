# MORPH Full Forward+Backward Kernel / Efficiency Plan (2026-05-31)

> Durable plan — written to survive context compaction. Captures status, the
> hardware/regime decision, the verified wins, and the remaining kernel+efficiency
> work for the looped-core hot path. Branch: `remove-jepa-memory` (UNCOMMITTED).

## 0. Hardware targets & the regime decision (drives everything)

Real targets (NOT the 5090 we prototype on):
- **RTX Pro 6000 96 GB** (CUDA) — fill VRAM as much as possible.
- **TPU v6e, more likely** — 2–4 chips (maybe 6→3/2); may scale up for very long context.
- Goal: **2B+ params, long context, healthy batch sizes.**

**Regime = compute-bound, param ≫ cache.** At 2B+ with healthy batch, B·S ≫ the
roofline ridge → matmuls are compute-bound; the core (≥625 MiB at d=2048, more at
2B) does NOT fit any L2/VMEM. Therefore:
- **Foundation = hardware-invariant FLOP cuts + memory cuts** (these transfer to any
  device and any param count). 
- **Cache-residency / CUDA-graph / launch-overhead tricks are 5090-local artifacts**
  — they're autotuned *conditionals* the per-target autotuner may enable, NOT things
  to architect around. (Measured: 5090 core layer is launch-overhead-bound below
  M=B·S≈4096, compute-bound above; L2 96 MiB vs core 66.8 MiB local / 625 MiB cloud.)
- The autotune philosophy (tile size × layer × loop-count, per hardware) stays — it
  picks residency/tiling per target via a ridge probe (`ignore/ridge_probe.py`,
  portable; run on each target incl. `ssh dgx-spark`).

## 1. STATUS — done & verified this session

- **JEPA + neural-memory REMOVED** (torch + JAX), −1179 lines, both frameworks green
  (fwd/bwd finite, 262.4M params lockstep). STP (Semantic Tube Predictor) kept.
- **Attention fully fused** (5 kernels), each verified grad-cosine 1.0, end-to-end vs
  eager 0.99999, Z3-proven, integrated into `morph/model/attention.py`:
  - `fused_cca_prologue.py` — qk-mean/RMSNorm/temp/RoPE/GQA/v-shift (1.88× micro)
  - `fused_cca_conv.py` — depthwise+grouped causal conv (conv-bwd 9.6%→3.2% of step)
  - `fused_hca_attention.py` — HCA dense compressed attn (3.86× mem @ B32/S8k)
  - `fused_csa_attention.py` — CSA top-k gather attn, gather-on-the-fly
    (**11.1× / 9.7 GiB-per-layer mem saved @ B32/S8k** — never materializes C_sel)
  - `fused_window_attention.py` — pre-existing
- Full model (B2/S2048 dense): **238.8→215.7 ms (−10%), 6643→6151 MiB (−7%)**;
  far larger memory wins at scale.
- Dead **MoSA** `fused_gate_combine.py` deleted; `memory_mlp.py` + `kernels/cuda/` gone.
- **PT↔JAX parity established** (subagent): d=768 cosine 0.992 (bf16 rounding only),
  embedding 1.0, architecture matches. Subagent fixed 3 bugs: PT `attn_kw` missing
  `d_indexer`/`conv_kernel`/`init_alpha` (now config-connected); JAX inference loop
  used `max_depth` not `mean_depth`; checkpoint converter fully rewritten for the
  post-removal namespace. (Touched `morph/model/transformer.py` attn_kw,
  `morph/jax/model/transformer.py`, `morph/interop/checkpoint.py`.)

## 1b. LANDED + VERIFIED THIS SESSION (2026-05-31, efficiency round)

Branch `remove-jepa-memory`, all UNCOMMITTED. Four levers, each gated end-to-end
on the 5090 (gitignored harnesses in `ignore/`):

- **A2 x0-projection hoist** (`mhc.py` ChannelInject.precompute/apply_precomputed;
  `transformer.py` precomputes x0 terms once, passes stacked into `_core_step`).
  BIT-EXACT (eval rel=0.0). ~48 redundant matmuls/launches/fwd removed.
  Cost: +~48 MiB saved-activation (hoisted out of checkpoint). Gate: `bench_efficiency.py`.
- **B5 fused chunked cross-entropy** (`morph/model/fused_ce.py`,
  `embeddings.lm_weight()` single-source tied weight, wired into `transformer`
  training path; `out["logits"]=None` in training, eager full logits in eval/gen).
  Standalone gate: fp32 grad cos 1.0, bf16 cos 0.99999; **CE peak 18.5 GiB→1 GiB
  (19×) at N=B8·S4096**. Model B2/S2048: -814 MiB (-13%). chunk_size is a config
  knob (`model.ce_chunk_size`, base.yaml). Chunk over ROWS = numerically near-exact.
- **#171 8-bit AdamW** (bitsandbytes 0.49.1, RUNS on sm_120 — verified, uint8 state
  on non-embeds, fp32 override on the 6 embedding tables per bnb stability rec).
  `optimizer.py` create_optimizer + `training.adam8bit` flag. State -20.8% (embeds
  in fp32 dilute the headline 75%). Also fixed a pre-existing YAML bug (wd/beta were
  mis-indented under `routing:`). Gate: `ignore/verify_adam8bit.py`.
- **A1 active-set shrinking** (`transformer.py` core loop: sort by depth desc,
  process shrinking active prefix, cat frozen suffix, inv-perm restore — replaces
  the full-batch + `torch.where` discard). LOGIC PROVEN bit-exact: fp32 DENSE
  rel=0.0/cos=1.0; varying-depth drift (fp32 1.3e-4 / bf16 4e-3, cos>0.99999) is
  batch-size cuBLAS reduction-order rounding compounded over the recurrence, NOT
  logic. ~25% fwd FLOP cut at depth spread. Gate: `ignore/verify_active_set.py`.
  NOTE: changes dropout RNG consumption order (permuted sub-batches) → a different
  but equally-valid dropout realization in training; eval (equal depths) bit-exact.

Combined (all 4) vs original baseline, B2/S2048: eval bit-identical, train_loss
rel 1.4e-4, **peak -12.3%**, time +1.0% (neutral at this tiny size; the FLOP/mem
wins show at scale, not in micro-bench — by design). Integrated gate
(`ignore/verify_integrated_step.py`) green for eager + compiled + compiled-8bit.
`ignore/verify_removal.py` updated to the new train/eval logits contract, green.

REMAINING (deliberately scoped, not dodged):
- **B4 selective-activation checkpointing**: a memory↔speed TRADEOFF knob. At the
  memory-bound 2B/long-ctx target, current full checkpointing is already the
  min-memory choice; SAC only helps when there's memory headroom to trade for
  backward speed → decide per-target with the ridge probe. Lower priority than the
  4 landed wins; implementing it as a "win" would trade away our scarce resource.
- **C7 gate-combine fusion (#179) / C8 MORPHBlock-glue fusion**: small launch-count
  polish. **C9 per-target autotune**: needs the real hardware to ridge-probe.
- **TST**: separate training-recipe track (§3).

## 2. THE FULL KERNEL — remaining forward+backward work (scale-robust, prioritized)

The looped-core hot path is: `prelude → [core×T] → coda`, where each core-step =
`injection → 6× (x0/ve/bigram inject → MORPHBlock(attn + SwiGLU MLP + MRR + norms))`.
Attention is now fused. Remaining levers, ordered by scale-robust value:

### A. FLOP cuts (win on every compute-bound target)
1. **Active-set shrinking in the loop.** Today `total_iters = depths.max()` (≈8) and
   every iteration computes the FULL batch, then `torch.where(active, h_new, h)`
   discards Poisson-frozen samples → ~25% forward FLOPs wasted on frozen samples +
   8 `where` copies. Fix: sort batch by depth desc, process the shrinking active
   prefix each iteration (contiguous → clean batched matmul), scatter back at the end.
   ~25% real forward FLOP cut + kills the where-copies. (`transformer.py` `_forward_single`.)
2. **Hoist the constant x0 projection out of the loop.** `_apply_x0` →
   `ChannelInject.forward` does `F.linear(x0, proj.weight)` every iteration, but `x0`
   is loop-constant → 42 redundant matmuls + launches per forward (6 core layers ×
   ~7 redundant iters). Precompute `project(x0)` once per core-layer index before the
   loop; inside, only the cheap `h[...,slice] += precomputed`. (value-embeds are a
   no-op in the loop — ve_layer_map=[0,1,2], core uses gi=3..8; bigram inject is just
   `x + λ·const`, already cheap.)
3. **Block-ELL pruning** (existing pipeline) — the biggest FLOP lever: dense→prune→
   compact to 25% MLP density → ~75% MLP FLOP cut post-compact, AND shrinks the param
   footprint as we scale to 2B+. Already built (`sparsity.py`, `SparsitySchedule`).

### B. Memory cuts (these are what let 2B+ fit at long context + healthy batch)
4. **Selective-activation checkpointing of the BPTT window.** Today each of the 4
   `bptt_depth` iterations is `checkpoint(_core_step, use_reentrant=False)` →
   recomputes the ENTIRE 6-layer core in backward (full recompute even of cheap
   pointwise). Move to SAC (save matmul/attn outputs, recompute only norms/activations)
   → cuts backward recompute time AND tunes the activation-memory/recompute trade,
   which is what caps batch size at scale. (PyTorch `torch.utils.checkpoint` SAC policy.)
5. **LM-head logits** — the measured scale OOM driver: `[B,S,vocab]` fp32 for CE ≈
   6 GiB at B8/S4096, far more at 2B/long-ctx. Use a **chunked / fused cross-entropy**
   (compute logits+CE in vocab tiles, never materialize the full logits tensor; or a
   fused-linear-CE kernel). Big, hardware-invariant batch-headroom win.
6. **8-bit AdamW** (task #171, bitsandbytes `Adam8bit`) — optimizer state −75%; at 2B+
   that's GBs of headroom. Mind the STE ternary shadow-weight interaction (shadow IS
   the bf16 param the optimizer updates). Wire a config flag, log full config to wandb.

### C. The "full kernel" fusion vision (the remaining launch/intermediate squeeze)
7. **Fuse the 2-way gate-combine** (#179) — last eager attention intermediate
   (`combined`, `x_res` [B,H,S,D]); fold into the compressed-attn epilogue. Small.
8. **Fuse the MORPHBlock glue** (MRR per-channel γ scale + RMSNorm + residual) — these
   are many small launches (`fused_ops.py` already has `_FusedRMSNorm`,
   `_FusedResidualRMSNorm`; wire them in). Reduces the per-layer launch count.
9. **Per-target autotune layer** (the user's design): ridge-probe each target → if
   bandwidth/launch-bound (Spark, decode), enable CUDA-graph-the-loop-body and/or L2
   persistence (contiguous core-weight arena + `accessPolicyWindow`); if compute-bound
   (Pro 6000 / TPU at healthy batch), skip them. NOT a foundation — a conditional.

### Backward note
All of A/B apply to backward symmetrically (BPTT reuses the same weights; #4 is the
backward lever; #1 shrinks the checkpointed activation set; pruning halves bwd matmul
FLOPs). On TPU the residency mechanism differs (VMEM/Pallas, not GPU-L2) — the JAX
path is parity-aligned and ready; kernel-level TPU fusion is a separate Pallas effort.

## 3. TST — the training-throughput card (orthogonal, future)

Token Superposition Training (Shao et al. / the pasted doc): two-phase. Phase 1
(ratio r∈[0.2,0.4] of steps): average embeddings of s∈[4,8] contiguous tokens →
"s-tokens" (input superposition), predict next bag-of-s-tokens via **multi-hot CE**
(mean of s one-hot CEs; drop the log|y| const) (output superposition). Phase 2: revert
to standard token-level, **same architecture / embedding / LM head unmodified**
(representation alignment is load-bearing — re-init kills the gains). Net: s× token
throughput at constant per-step FLOPs → ~0.5× training cost for equal loss, robust
s∈[4,8] r∈[0.2,0.4]; ALSO longer effective context during phase 1 (good for our
long-context goal). **Implementation = embedding-bagging + multi-hot-CE in the training
loop; touches embedding + loss, NOT the core kernels.** Constraint for MORPH: embed is
hybrid(eucl+lorentz+bigram), LM head is weight-tied (`embed.attend`) — keep both
unmodified across phases. Low kernel risk, high throughput payoff. Plan as a separate
training-recipe task after the kernel/efficiency work.

## 4. Key facts to survive compaction
- Repo: `/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies`
- venv python: `/home/wolfe/.venv/bin/python`; run with `PYTHONPATH=.`. RTX 5090 sm_120,
  flash-attn unsupported; num_stages=1/num_warps=8; bf16 in/fp32 accum.
- Triton-3.6/sm_120 footgun: interior-axis 3-D reduce (`tl.sum(a[:,:,None]*b[None,:,:],axis=1)`)
  silently doubles output — use `tl.dot`, pad dims ≥16. (Hit + fixed in HCA kernel.)
- Verify scripts (gitignored `ignore/`): `verify_removal.py`, `verify_removal_jax.py`,
  `bench_attention_baseline.py` (+ `baseline_attention.json`), `profile_step.py`,
  `ridge_probe.py`, `p1_prologue_harness.py`, the parity audit `parity_audit_v2.py`.
- Z3 proofs: `tile-prover/proofs/{fused_cca_prologue,fused_cca_conv,fused_hca_attention,fused_csa_attention}/`.
- Tasks #170 (CCA eff) effectively done bar #179 gate-combine; #171 (8-bit AdamW) pending.
- bf16 max-err gate caveat: 2e-2 is below the bf16 floor for these ops — gate on
  grad-cosine>0.995 + fwd-cosine>0.9999 + (fused-vs-fp32-truth ≤ eager-vs-fp32-truth).

## 5. Recommended next sequence (post-compaction)
1. **Memory levers first** (B4/B5/B6) — they're what let 2B+ fit on the Pro 6000 / TPU
   at long context; chunked-CE logits + selective-checkpoint + 8-bit AdamW.
2. **FLOP levers** (A1 active-set shrink, A2 x0 hoist) — clean, hardware-invariant.
3. **Then** the per-target autotune layer (C9) once we're on the real hardware and can
   ridge-probe it. Commit the verified work before starting (currently uncommitted).
4. **TST** as a parallel training-recipe track (low kernel risk, ~2× throughput).
