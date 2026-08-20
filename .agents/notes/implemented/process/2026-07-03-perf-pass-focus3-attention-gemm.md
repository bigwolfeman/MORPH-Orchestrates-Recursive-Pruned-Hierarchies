# Agent Note: Perf Pass Focus3 Attention Gemm

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Focus3-Attention-GEMM.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Focus 3: Attention Small-GEMM Batching (2026-07-03)

CPU analysis + worktree prototype ONLY (single 5090 + UPS → another agent owns the
GPU; orchestrator serializes GPU gates). Everything labeled MEASURED below was run;
everything labeled ESTIMATE needs the GPU gate.

**Worktree:** `.claude/worktrees/attn-gemm-batch` (branch `perf/attn-gemm-batch`, NOT
committed). One file changed: `morph/model/attention.py` (+199/−33).
**Parity scripts:** `scratchpad/parity_attn_fused_proj.py` (module-level, vs git-HEAD
copy), `scratchpad/parity_full_model.py` (end-to-end), `scratchpad/op_census_attn_fused.py`
(aten-op census). Results archived: `scratchpad/parity_results.txt`, `scratchpad/op_census_results.txt`.

## 1. The aten::mm map (r4 routed trace, ÷10 = per step; MEASURED trace + code anchors)

`aten::mm` = 2,518 calls/step, 121.9 ms/step CUDA-total (923.9 ms self /10 = 92.4 ms
kernel-self). Exec-count anchors from the same trace: `_FusedCCAPrologue` 75.6/step
= attention forwards (37.8 CSA + 37.8 HCA, incl. checkpoint recompute);
`_FusedCCAPrologueBackward` 30/step = grad-bearing attention execs (15+15).

Per CSA forward (attention.py, all bias-free `nn.Linear` on the SAME x [B,S,768]):

| mm | N (out) | K | grad? |
|---|---|---|---|
| `W_down_q` | 384 | 768 | yes |
| `W_down_k` | 128 | 768 | yes |
| `W_v_curr` / `W_v_prev` | 64 / 64 | 768 | yes |
| `gate[0]` | 192 | 768 | yes |
| CSA compressor `W_aKV/W_aZ/W_bKV/W_bZ` | 4×32 | 768 | yes |
| indexer `W_IQ` + its compressor `W_aKV/W_aZ` | 3×32 | 768 | **NO — grad-dead** (scores→topk indices only) |
| `gate[2]` | 24 | 192 | yes (dep-chained, not batchable) |
| `W_up` | 768 | 384 | yes (dep-chained) |

= 14 mm/CSA-fwd (+1 bmm indexer scores, already batched). HCA = 9 mm/fwd (no indexer,
2-weight compressor). Every batchable one is SKINNY (N ≤ 384, K = 768, M = B·S = 16,384
at mb4/seq4k) → launch-bound, low SOL, and reads the same 25 MB x up to 9–12×.

**Attention share (from census × exec counts): ~1,652 mm/step ≈ 66% of all aten::mm.**
Residual ~866/step: TileRouter fp32 `query_proj` (routing.py:97, ~113 router calls/step
+ bwd — queue item #5), GLA retention q/k/v/o (gla.py:57-62, ~13.6 execs/step —
same-input q/k/v batchable, follow-up F1), ChannelInject projections (covered by the
#1 injection-hoist), lm_mixer `mix` (transformer.py:357), Lorentz `lm_weight()` rebuild,
fused-CE chunk GEMMs (large, healthy).

## 2. What was prototyped (all default-ON in the worktree, env kill-switches)

**Mechanism A — fused input-projection superblock** (`MORPH_FUSED_ATTN_PROJ=0` to kill):
one GEMM `Y = x @ [W_down_q; W_down_k; W_v_curr; W_v_prev; gate.0; compressor W_*]^T`,
split into views. Concat along N never touches the K-reduction → same dot products.
- `W_v_prev`'s input shift moved to the OUTPUT: `pad(W(x)[:, :-1]) == W(pad(x[:, :-1]))`
  exactly for a bias-free Linear (W·0 = 0). Shares the GEMM.
- Compressor block-truncation moved to the OUTPUT: `W(x[:, :n]) == W(x)[:, :n]` (row-wise).
- **Indexer trio in a SEPARATE no_grad GEMM.** Critical catch: indexer scores feed ONLY
  `topk` indices, so eager grads are **None**. Batching them into the grad-bearing cat
  gives zero-TENSOR grads → optimizer weight-decays never-trained params → silent
  trajectory change. `torch.no_grad()` reproduces the None semantics exactly. (The
  inference engine already batches "v_curr + gate-hidden + CSA indexer q" into one GEMV
  — engine.py:21 — so this mirrors the deploy path.)
- Weight cat = 1 cheap kernel/fwd; cat-backward = narrow views (no copy kernels);
  weight grads are exact slices of dW_cat. NO new parameters → checkpoint-compatible
  (state_dict keys unchanged; verified by strict load in the parity script).

**Mechanism B — fused q‖k causal-conv pair** (`MORPH_FUSED_ATTN_QKCONV=0`): the weight
order makes `Y[..., :512]` the contiguous q_lat‖k_lat pair → ONE `fused_cca_conv` call
(512 ch, groups 16) instead of two. Depthwise = per-channel; grouped conv keeps Cg=32
group membership under concat (q = groups 0..11, k = 12..15) → identical reductions.

Not batchable (dep-chained): `gate[2]`, `W_up`. Untouched: CLA paths (`cla_kv`/
`_cca_q_only`), kv_cache/inference engine, eager fallback (proven no-op below).

## 3. CPU verification (MEASURED — all commands run, exit code 0)

`parity_attn_fused_proj.py` — module-level vs the **committed HEAD copy** of
attention.py, production dims (d768/H12/Hkv4/C2/m4/m128/topk128/dI32), single-threaded
CPU (multithreaded CPU backward is nondeterministic — a base-rerun control arm proves
the noise floor is exactly 0 under `torch.set_num_threads(1)`). Matrix: {CSA,HCA} ×
{fp32,bf16} × {B2/S256, B1/S302 (S%m≠0)} × {n_skip_rope 0,4} × {patch-off, proj,
proj+conv}. Result: **OVERALL PASS**:
- **Forward bitwise (`torch.equal`) in EVERY fused arm** — fp32 and bf16.
- patch-off arm: bitwise fwd AND grads → the refactor is a provable no-op when disabled.
- Grads (fused): max deviation **≤ 3.84·eps at tensor scale** (fp32; abs ≤ 7.6e-6) and
  **≤ 1.6·eps** (bf16) — pure reassociation (dX = one reduction over ΣN vs autograd sum
  of 9; dW via dW_cat slices), same class as the landed x0-hoist / Focus-2's
  injection-hoist (1.6e-8). No grad-None-pattern change (hard-gated in the script).
- proj+conv arm == proj arm to the bit → the conv pairing added ZERO extra deviation.

`parity_full_model.py` — full MORPHTransformer (0.6M toy: embed → prelude → Poisson
core loop with no_grad iters + checkpointed grad iters + recompute → coda → CE),
dropout 0.0 AND 0.1: **loss BITWISE identical** fused-on vs off in both regimes
(dropout=0.1 passing proves the RNG stream through checkpoint recompute is untouched
— no RNG ops added/removed). Whole-model grad max|Δ| ≤ 1.04e-7 fp32. PASS.

`pytest tests/ -x -q` on the worktree: **46 passed, 2 skipped** (skips are pre-existing
CUDA-gated tests).

## 4. Launch / time impact

Op census (MEASURED, CPU aten dispatches == CUDA launch sites for these ops), per
attention exec: CSA fwd mm 14→4, bwd 22→6; HCA fwd 9→3, bwd 18→6; conv wrapper calls
2→1 (fwd) and 2→1 (bwd); +4–5 small `cat`/exec; `narrow`/`split` growth is autograd
views (no kernels).

Scaled by trace exec counts (75.6 fwd, 30 bwd/step) — arithmetic on measured counts:
- **aten::mm: −1,025/step (2,518 → ~1,493, −41%).** fwd −604.8, bwd −420.
- `_FusedCCAConv` wrappers: −105.6 fwd + −30 bwd/step ≈ −250 to −330 Triton launches.
- +~300 small cat launches/step. Net kernel-launch delta ≈ **−1,000 to −1,300/step**
  (~4–5% of 27.2k) — plus relief on the 225 ms/step Command-Buffer-Full stall.
- ESTIMATE (autocast, not in CPU census): −8 to −11 weight casts/exec ≈ −600 to −800
  of the 5,091 `aten::_to_copy`/step, since the cat is cast once vs per-weight.
- ESTIMATE wall-time: attention's mm ≈ 55–75 of the 121.9 ms/step; fused replacement
  (75.6 fat fwd GEMMs M16384·K768·N896–1024 + 60 bwd) ≈ 25–35 ms → **−20 to −45 ms/step
  mm time** + stall relief. At ~1.2 sps (~833 ms/step) that is 2.5–5.5% direct before
  stall relief. GPU-gated; do not quote as measured.
- Memory: Y retained instead of 9 pieces (same bytes); kernel-wrapper `.contiguous()`
  on slice inputs adds transient copies ≈ +36 MB per stored attention exec → order
  +0.5–1 GB transient at mb4/seq4k vs 10.5 GB routed headroom. Watch on GPU.

## 5. Ranked batchings

| # | win | mechanism | Δ (per step) | class | CPU parity | status |
|---|---|---|---|---|---|---|
| 1 | Input-proj superblock (CSA 9→1, HCA 7→1 grad GEMMs) | concat-N weights → 1 GEMM → split views | −937 mm (fwd −529, bwd −408); ESTIMATE −600–800 casts | **A\*** | fwd bitwise; grads ≤3.8·eps reassoc; full-model loss bitwise | **PROTOTYPED + CPU-verified** |
| 2 | Indexer trio no_grad GEMM | separate cat under no_grad (grad-None preserved) | −88 mm fwd | **A** (grad-dead path, values bitwise) | fwd bitwise, None-pattern hard-gated | **PROTOTYPED + CPU-verified** |
| 3 | q‖k conv pairing | one fused_cca_conv, groups 16, zero-copy prefix slice | −105 fwd + −30 bwd wrapper calls (≈−300 Triton launches) | **A\*** | bit-identical to arm #1 (zero added deviation) | **PROTOTYPED + CPU-verified** |
| F1 | GLA retention q,k,v batch (gla.py:57-59, same input) | same concat-N mechanism | ≈−50–80 mm | A\* expected | not built | follow-up spec |
| F2 | Hoist weight cats out of the core loop (cat once/step, pass as checkpoint input — same trick as the injection-hoist) | plumbing through transformer.py | −~300 cat | A (narrow-exact) | not built | follow-up spec, only if cat shows up on GPU |
| — | Router fp32 GEMM | queue item #5 (bf16 = class B) | — | B | — | out of scope |

A\* = forward bitwise on CPU; grad deltas are fp reassociation ≪ the 6e-4 noise floor,
same class as the already-landed x0-hoist. GPU bitwise-ness of the forward is NOT
guaranteed (cuBLAS algo selection is shape-dependent, split-K on skinny shapes) — gate
below decides.

## 6. GPU validation recipe (for the orchestrator, serialized with other gates)

1. **Module probe** (5 min, cheap, run FIRST): on the 5090, run
   `scratchpad/parity_attn_fused_proj.py` WITHOUT the force_eager override and with
   tensors on cuda (edit the two `set_force_eager(True)` lines or parametrize) at
   B4/S4096 bf16 — checks `torch.equal` forward under real cuBLAS algos + Triton
   kernels (kernels' own contiguous() handles the strided slices; wrappers were built
   for that). If fwd not bitwise, record max ULP — expect ≤ few eps; the amplification
   risk to watch is **CSA topk tie-flips** if indexer-score bits move.
2. **Noise-floor gate** (the landing test): resume
   `checkpoints/morph/tst_stp_on_50k/step_36000.pt`, `training.resume_fresh_optimizer=true
   training.optimizer=ademamix_b1zero +training.seed=1234`, `MORPH_EXACT_TRACE=<path>`,
   `MORPH_PERF_REGIONS=1`, ckpt_grad_iters=-1; assert loss trace ≤ 6e-4 by step 11 vs
   `ignore/perf/ckpt_base.trace` (two identical baseline runs diverge to 6e-4 by step 11
   from atomics — the fused arm must stay within that envelope).
3. **Perf readout**: ms/step via MORPH_PERF_REGIONS + a 10-step profiler capture; verify
   aten::mm ≈ 1,49x/step and Command-Buffer-Full self-time drop. ncu the fused GEMM
   (M16384 K768 N1024) vs a skinny baseline GEMM for the SOL claim.
4. **A/B isolation** if the gate fails: `MORPH_FUSED_ATTN_QKCONV=0` first (conv arm),
   then `MORPH_FUSED_ATTN_PROJ=0`; if topk tie-flips are the culprit, the fallback is
   keeping the indexer trio eager (3 small mms back) while retaining the main superblock.

## 7. Honest edges (NOT verified)

- GPU forward bitwise-ness (cuBLAS shape-dependent algos) — CPU-only proven; gate item 1.
- The −20 to −45 ms/step and cast-count deltas are estimates from trace arithmetic, not
  measurements.
- bf16-autocast + GradScaler interaction untested on GPU (no numerics reason to differ —
  cast-then-cat == cat-then-cast elementwise — but unverified).
- `attn_proj_quant != "off"` (non-deploy config): cat reads parametrized `.weight`
  (one access per weight per fwd, same as eager) — should compose, untested.
- Transient-copy VRAM (+0.5–1 GB est.) unmeasured.
- torch.compile: attention is eager in the MLP-only compile regime (fence tests pass);
  full-graph compile mode untested.
