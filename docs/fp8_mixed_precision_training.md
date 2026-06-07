# FP8 Mixed-Precision Training — Design Doc

**Status:** PROPOSED (not implemented). Config-gated ablation, default OFF; the bf16
path is bit-identical when `fp8: false`. Planning only — no code changed, no GPU training
run for this doc.
**Author:** Claude (with Wolfe), 2026-06-02
**Scope:** FP8 the large linear GEMMs (MLP + attention projections + LM head) for
throughput + memory; keep attention math, norms, softmax, the loop's diagonal injection,
and the fused-CE loss in bf16/fp32 — the DeepSeek-V3 / Ling-1T "mixed" recipe.

---

## 1. Motivation

FP8 (8-bit float) tensor-core GEMMs roughly halve the bytes per matmul operand and roughly
double tensor-core throughput vs bf16 on hardware with FP8 tensor cores. The canonical
large-scale results:

- **DeepSeek-V3** (Dec 2024): trained 671B params in FP8. Only the *compute-dense GEMMs*
  run FP8; embedding, LM head, MoE gate, normalization, and **attention** stay bf16/fp32.
  Fine-grained scaling: activations per-`1×128` tile, weights per-`128×128` block,
  fp32 accumulation. Uses E4M3 throughout (fwd + grad).
  ([report](https://arxiv.org/pdf/2412.19437),
  [Colfax analysis](https://research.colfax-intl.com/deepseek-r1-and-fp8-mixed-precision-training/))
- **Ling-1T** (Ant Group, Oct 2025): largest FP8-trained model to date. **15%+ end-to-end
  speedup, ≤0.1% loss deviation vs bf16 across 1T tokens**, via tile/block-wise scaling +
  FP8 AdamW moments; attention kept bf16 (standard).
  ([Ling-1T-FP8](https://huggingface.co/inclusionAI/Ling-1T-FP8),
  [Ant blog](https://huggingface.co/blog/im0qianqian/ling-mini-2-fp8-mixed-precision-training-solution),
  [tweet](https://x.com/AntLingAGI/status/1975943121650524479))

**Why MORPH wants this.** MORPH is GEMM-heavy in exactly the places FP8 helps: the SwiGLU
`gate_up`/`down` (d_model→2·d_ff and d_ff→d_model), the CCA projections (`W_down_q/k`,
`W_v_*`, `W_up`), and the tied LM head (49152-row). The looped core (`core×T`, mean depth
6) reuses those weights ~6× per step, so the per-step GEMM volume in the core is ~6× a
stacked model of the same param count — FP8 compounds in the loop.

---

## 2. Hardware reality (DO THIS FIRST — it gates everything)

The task framing hedged that consumer Blackwell may not accelerate FP8. **That hedge is
falsified by live measurement on the actual rig.** Verdict below is from running on the
5090, not inferred.

### RTX 5090 (sm_120, consumer Blackwell), torch 2.11.0+cu130 — VERIFIED, real speedup

- `torch._scaled_mm` with `float8_e4m3fn` inputs **runs natively** on sm_120 (no error,
  `out_dtype=bf16`). Confirmed live: `torch.cuda.get_device_capability() == (12, 0)`,
  `NVIDIA GeForce RTX 5090`, torch `2.11.0+cu130`, cuda `13.0`.
- **Measured FP8 vs bf16 GEMM speedup on this exact GPU** (`torch.mm` bf16 vs
  `torch._scaled_mm` E4M3, per-tensor scale, bf16 out):

  | shape (M×N×K)        | bf16        | fp8 E4M3    | speedup |
  |----------------------|-------------|-------------|---------|
  | 4096³                | 1.47 ms (93 TF/s)  | 0.62 ms (221 TF/s)  | **2.36×** |
  | 8192³                | 9.65 ms (114 TF/s) | 2.24 ms (492 TF/s)  | **4.32×** |
  | 16384×4096×4096      | 3.01 ms (182 TF/s) | 1.14 ms (483 TF/s)  | **2.65×** |

  This is genuine 5th-gen-tensor-core acceleration, not emulation — the FP8 path hits
  ~2.4–4.3× bf16, consistent with the FP8:bf16 peak-flops ratio. (5th-gen Blackwell tensor
  cores add native FP8/FP6/FP4;
  [NVIDIA Blackwell arch PDF](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
  [Spheron specs](https://www.spheron.network/blog/nvidia-rtx-5090-specs/).)
- **torchao `float8` training works end-to-end on the 5090** (verified live):
  `from torchao.float8 import convert_to_float8_training, Float8LinearConfig`,
  `convert_to_float8_training(model)` rewrites `nn.Linear → Float8Linear`, and a
  forward+backward on a 2-layer bf16 MLP both succeed on cuda.
- **The scary warning is harmless.** torchao 0.15.0 prints
  *"Skipping import of cpp extensions due to incompatible torch version 2.11.0+cu130"*
  ([ao#2919](https://github.com/pytorch/ao/issues/2919)). That only drops the optional
  fbgemm / CPU-low-bit C++ kernels. The **float8 *training* path is pure-Python +
  `torch._scaled_mm`**, which we proved runs and accelerates. FP8 training does **not**
  depend on those cpp extensions.

> **Bottom line for the 5090:** FP8 GEMM training is *available and genuinely accelerated*
> today (torch 2.11+cu130 + torchao float8 dynamic scaling). The 5090 gets **both** the
> speedup and the memory saving — it is **not** a memory-only or emulated path.

### NVIDIA TransformerEngine on sm_120 — NOT a path here

Parent-session install attempts in this venv left `transformer_engine` either missing or
import-failing (`import transformer_engine.common` raised). TE's C++/CUDA build chain is
Hopper/datacenter-Blackwell-centric and brittle on cu13/sm_120. **Use torchao float8, not
TE.** torchao's dynamic-scaling float8 needs no TE and is torch.compile-native.

### TPU v6e (JAX mirror) — NO FP8 HARDWARE

**FP8 is not available on TPU v6e** (Google: v6e lacks FP8 MXU support; v7p is expected to
add it). ([XLA discussion](https://github.com/openxla/xla/discussions/23124),
[Flax FP8 guide](https://flax-linen.readthedocs.io/en/latest/guides/quantization/fp8_basics.html)).
JAX *supports* the dtypes (`jnp.float8_e4m3fn`, `jnp.float8_e5m2`) and `dot_general`
`preferred_element_type` fp32 accumulation, so the **code** ports, but on v6e it would be
emulated/no-speedup. → The TPU mirror is a **parallel future track gated on v7p**; do not
plan FP8 throughput wins on v6e. Keep TPU on bf16 (AQT) for now.


---

## 3. Prior art / recipe to port

The mixed recipe is well-established; we copy it, we do not invent it.

- **Scope (which ops are FP8):** *only the big linear GEMMs.* DeepSeek-V3 keeps embedding,
  LM head, gating, **all norms, all softmax, and attention** in bf16/fp32. Ling-1T does the
  same and reports ≤0.1% loss gap. **MORPH mirrors this: FP8 the projection Linears, keep
  the hand-written Triton attention kernels and fused-CE in bf16.**
- **Format:** E4M3 for the forward operands (more mantissa, weights/activations need
  precision); E5M2 *can* be used for gradients (more range), but DeepSeek-V3 and Ling found
  fine-grained scaling lets them keep **E4M3 for the grad GEMMs too**, reducing precision
  loss. torchao's default dynamic recipe uses E4M3 for fwd and E5M2 for the grad-output
  cast — start with the torchao default, it matches the proven configs.
- **Scaling:** two families —
  - *Per-tensor* with **dynamic** scaling (amax computed per call, stateless) — simplest,
    torch.compile-friendly, no buffers. **Recommended for MORPH** (see traps §7).
  - *Per-tensor delayed* (amax history buffers, one-step-stale scale) — slightly faster
    casts but **stateful**, which is dangerous in a reused-weight loop (trap §7.2).
  - *Fine-grained per-block* (`1×128` act / `128×128` weight, DeepSeek-V3) — best accuracy,
    most kernel work. Defer; torchao's rowwise/blockwise recipes are newer and the
    per-tensor dynamic path already hits the 2.4–4.3× we measured.
- **torchao float8 API:** `convert_to_float8_training(model, config=Float8LinearConfig(...),
  module_filter_fn=...)`. The `module_filter_fn` is how we restrict the conversion to the
  exact Linears we want FP8 and exclude everything else (norms aren't Linear; we still must
  exclude the gate MLP, embeddings, etc. by name — §4).

---

## 4. Proposed design — scope + config

### What goes FP8 (the recommended mixed recipe)

**FP8 (convert to `Float8Linear`):**
- `_SwiGLU.gate_up`, `_SwiGLU.down` (the dominant GEMMs — d_model↔2·d_ff↔d_model).
- CCA projections in `_CCABase`: `W_down_q`, `W_down_k`, `W_v_curr`, `W_v_prev`, `W_up`.
- LM head matmul — **but this lives inside `fused_linear_cross_entropy`** (a hand-written
  fused-CE Triton kernel, `morph/model/fused_ce.py`), **not** an `nn.Linear`. So the LM head
  is *not* auto-convertible by torchao; FP8'ing it needs a fused-CE FP8 variant → **defer**
  (kept bf16 in phase 1, exactly as DeepSeek keeps the output head bf16 anyway).

**STAY bf16/fp32 (do NOT convert):**
- All hand-written Triton attention kernels (`fused_cca_prologue`, `fused_cca_conv`,
  `fused_hca_attention`, `fused_csa_attention`, `fused_window_attention`) — these are not
  `nn.Linear`, and the attention scores/softmax are exactly what the recipes keep
  high-precision.
- `fused_linear_cross_entropy` (the loss).
- All `RMSNorm` (already fp32-internally — see `attention.py:RMSNorm.forward`).
- `DiagonalInjection`, the diagonal SSM injection in the loop (tiny, stateful, range-
  sensitive — keep fp32).
- The gate MLP (`_CCABase.gate`), `LightningIndexer`, `GatedPoolCompressor` (small Linears
  feeding softmax/relu selection logic — low FLOP, high sensitivity; keep bf16).
- Embeddings (`MORPHEmbedding`, value-embed tables, bigram), `LMHeadMixer.mix`
  (eye-initialized residual mixer — sensitive, low FLOP).
- `ChannelInject` projections (injection terms; small, additive, x0-hoisted).

This is the Ling-1T/DeepSeek-V3 split: **dense GEMMs FP8, everything fiddly bf16.**

### Quantify the Triton-kernel-rewrite cost (and why we don't do it now)

FP8'ing the *attention* would mean writing FP8 variants of 5 hand-tuned sm_120 Triton
kernels (each already has a bf16 fwd + backward + parity test; the repo convention is
fwd-max-err + grad-cosine vs an eager reference). Each is a multi-day effort with new
per-tensor or per-block descale plumbing inside the flash online-softmax, and attention is
the *one place every public recipe keeps bf16* — so the expected upside is near zero and the
risk (causal-mask + sink + early-query-guard correctness under FP8 descale) is high.
**Recommendation: never FP8 the attention kernels; mirror Ling-1T.** The GEMM-only scope
captures essentially all the available win.

### Config knobs (add to `morph/configs/base.yaml`, default OFF)

```yaml
training:
  fp8: false                 # master switch; false → bit-identical to today (bf16)
  fp8_recipe: dynamic        # dynamic | delayed | rowwise  (dynamic = recommended, stateless)
  fp8_scope: mlp             # mlp | mlp_attn_proj | all_gemm
                             #   mlp           = SwiGLU gate_up/down only (safest)
                             #   mlp_attn_proj = + CCA W_down_q/k, W_v_*, W_up
                             #   all_gemm      = mlp_attn_proj (LM head stays bf16 — fused-CE)
  fp8_filter_min_dim: 256    # skip Linears with any dim < this (FP8 only helps big GEMMs)
```

Log the full resolved config to wandb (mandate) — including `fp8_recipe`, `fp8_scope`,
and the list of converted module names, so a run is reproducible/greppable from its config.

**Bit-identical-when-off guarantee:** `fp8=false` skips `convert_to_float8_training`
entirely → the model is the unmodified bf16-autocast model → numerically identical to the
current 39.46 baseline. (Verify with the bit-exact gate in §8.)

---

## 5. Kernel work

- **None required for phase 1.** torchao's `Float8Linear.forward` does the FP8 cast +
  `torch._scaled_mm` + descale internally. The dense MLP GEMM is a plain `nn.Linear`, so it
  converts cleanly.
- **`_SwiGLUBlockELL` / `BlockELLLinear` is the wrinkle** (`morph/model/sparsity.py`).
  Pre-compact it's dense (`initial_density=1.0`) and *could* be FP8'd as a dense GEMM; but
  it is **not** an `nn.Linear` (custom module with a Triton Block-ELL forward post-compact),
  so torchao's converter won't touch it. FP8 for the *sparse* post-compact path would need
  an FP8 Block-ELL Triton kernel (real work). **Phase 1: leave the core MLP (Block-ELL) in
  bf16; FP8 only the prelude/coda `_SwiGLU` dense MLPs + (optionally) CCA projections.** This
  is a clean, honest scoping line, not a cop-out — see §7.6.
- **LM head FP8** would need a fused-CE FP8 variant (`fused_ce.py`) — deferred (§4).

---

## 6. Composition with existing low-precision work (orthogonality map)

| feature | what it touches | FP8 interaction |
|---|---|---|
| **ternary QAT** (`ternary_qat.py`) | a *forward weight view* (`{-1,0,+1}×scale` STE) on selected Linears via `parametrize`; fp32/bf16 shadow is the live param | **Mutually exclusive per layer.** A ternarized Linear's forward weight is already ternary — there is no bf16 weight to cast to FP8. Ternary and FP8 must not target the *same* Linear. They can target *different* Linears (e.g. ternary backbone MLP + FP8 attention projections), but the common case is **FP8 XOR ternary per layer**. |
| **8-bit AdamW** (bitsandbytes, optimizer state) | optimizer *moments* only, not compute | **Fully orthogonal.** Stacks freely with FP8. (FP8 + 8-bit AdamW = compute-FP8 + state-8bit, like Ling-1T's FP8 moments but via bnb.) |
| **autocast bf16** | compute dtype for non-FP8 ops | FP8 layers consume bf16 activations and emit bf16; autocast stays on for everything else. Master weights remain **fp32** (train.py does `model.to(device)` with no dtype cast — confirmed; bf16 is autocast-only). torchao keeps an fp32/bf16 master and casts to FP8 per forward. |

**Defined arms:**
- *FP8-only*: `fp8=true, ternary=false` — FP8 the dense MLP/proj GEMMs.
- *FP8 + 8-bit*: add bnb AdamW8bit — orthogonal, expected additive memory win.
- *FP8 ⊕ ternary (disjoint scopes)*: ternary on `backbone` (core MLP), FP8 on attention
  projections — the two never overlap a Linear. Document the scope split explicitly; a
  guard must assert no Linear is in both sets.
- **Forbidden:** ternary scope and fp8 scope overlapping the same module (assert + raise).

---

## 7. ⚠️ MORPH-specific traps (the valuable part — not in any FP8 tutorial)

1. **Looped core × FP8 weight cast — recast-per-iteration waste + amax conflation.**
   The 6 core weights are reused across T (mean 6) loop iterations *within a single forward*.
   torchao's `Float8Linear.forward` recasts the weight to FP8 on **every call** — so a core
   Linear gets cast `T × (#core layers calls) ≈ 6×` per step instead of once. With
   **dynamic** scaling this is merely *redundant* (correct, just extra cast FLOPs — cheap vs
   the GEMM). With **delayed/amax-history** scaling it is **a correctness bug**: each
   intra-step reuse pushes a fresh weight-amax into the history buffer, so the "history"
   conflates *iterations of the same step* with *across-step history*, corrupting the
   one-step-stale scale. → **Use `fp8_recipe: dynamic` (stateless) for any FP8 in the
   looped core.** If we ever want delayed scaling for speed, the FP8 weight must be cast
   **once per optimizer step and cached across the T iterations** (a custom wrapper, not
   stock torchao). This is the #1 looped-architecture trap.

2. **Truncated BPTT × where the FP8 amax comes from.** `bptt_depth=4`: the first
   `total_iters − 4` core iterations run under `torch.no_grad()` (transformer.py:381). A
   `Float8Linear` under `no_grad` still does its FP8 cast + `scaled_mm` forward (fine —
   forward-only). With **dynamic** scaling the amax computed in those no-grad iterations is
   *used for that forward and discarded* → no leakage, correct. With **delayed** scaling, the
   no-grad iterations would push amax into the persistent buffer and pollute the scale used
   by the in-grad iterations. → **Second independent reason dynamic scaling is the safe
   default for MORPH.** Also note: the activations entering the FP8 GEMM in the no-grad
   region are themselves the loop-carried `h`; their amax is representative, so dynamic
   per-call scaling is well-behaved across the freeze boundary.

3. **Active-set shrinking → variable-M GEMM + recompile/timing-race surface.** The core loop
   sorts by depth and processes a *shrinking* contiguous prefix `h_s[:n_active]`
   (transformer.py:373–391). The Linear's input row count is `n_active·S` and **changes every
   iteration**. (a) *Numerically fine*: per-tensor activation amax is over all elements, so a
   smaller M just has its own amax; `torch._scaled_mm` handles variable M. No correctness
   issue. (b) *The real hazard*: variable M is already a known dynamic-shape source that the
   compiled MLPs recompile on (the fused CCA kernels JIT-specialize `n_active==1` separately
   — train.py:283-286, the documented startup wedge surface). **FP8 adds cast + descale ops
   to each recompiled graph**, enlarging exactly the recompile/fork surface where MORPH's
   intermittent **startup timing-race** lives (thread/stream-scheduling sensitive; six
   hypotheses falsified, root cause OPEN). → Mitigation: torchao float8 dynamic scaling
   composes with the existing `eager_on_recompile` stance (FP8 ops run eager on a guard
   miss, no recompile storm). After integration, **re-run the startup-wedge sensitivity
   check with eval+gen ON** (the config the 06-01/06-02 hunt showed is needed to surface
   the race). If FP8 worsens it, scope FP8 to the *prelude/coda* dense MLPs only (these are
   fixed-shape, not in the variable-M loop) — a clean fallback that keeps most of the win.

4. **torch.compile + FP8 + force-eager generation.** Phase-1 FP8 targets the prelude/coda
   `_SwiGLU` (and optionally CCA proj). The compiled MLPs (train.py compiles MLPs only) + FP8
   is a *supported* torchao path (torchao float8 is explicitly torch.compile-designed). But
   `run_generation_test` runs under `@torch.compiler.set_stance("force_eager")` — FP8
   `Float8Linear` must produce identical-enough output in eager as in compiled (it does; the
   cast is dtype-only). Verify gen stays coherent with FP8 on. Do **not** use
   `fullgraph=True` (the loop's `autograd.grad`/checkpoint already forbid it).

5. **STP geometric regularizer + fp32 boundaries.** `stp_loss` is computed on
   `final_norm(x).float()` (transformer.py:404) and the fused-CE runs in bf16 with fp32
   accumulation. FP8 must not creep into these: ensure the LM-head/mixer/`final_norm`/STP
   chain is excluded from conversion (it is, per §4). The `module_filter_fn` must exclude by
   name, not just by `isinstance(nn.Linear)`, because `LMHeadMixer.mix` *is* an `nn.Linear`
   we want to keep bf16.

6. **Block-ELL pruning phase boundary.** The core MLP is `_SwiGLUBlockELL`/`BlockELLLinear`,
   which switches from dense to Triton Block-ELL at `compact_step` (66000). FP8 on the core
   MLP would have to span a *dense→sparse representation change mid-run* — the dense FP8
   path and a (nonexistent) sparse-FP8 path are different kernels. → **Do not FP8 the core
   Block-ELL MLP.** FP8 the prelude/coda dense `_SwiGLU` and (phase 2) the CCA projections,
   which are plain `nn.Linear` and never change representation. This keeps FP8 orthogonal to
   the pruning schedule entirely.

7. **Master-weight dtype assumption.** train.py does `MORPHTransformer(cfg).to(device)` with
   **no dtype cast** — params are fp32 master, bf16 is autocast-only (confirmed). torchao
   float8 expects exactly this (fp32/bf16 master, per-forward FP8 cast). If a future change
   ever stores bf16 master weights, revisit — FP8 dynamic scaling from a bf16 master is fine,
   but the amax/precision budget shifts.

---

## 8. Ablation / test plan

**Baseline:** current bf16, val ppl **39.46** @ 15k steps (kernels, b4). Arms (all 15k, same
data/seed/schedule, eval+gen ON):

- `fp8_mlp` — `fp8=true, fp8_scope=mlp, fp8_recipe=dynamic` (prelude/coda dense SwiGLU only).
- `fp8_mlp_attn` — `fp8_scope=mlp_attn_proj` (+ CCA projections).
- (optional) `fp8_mlp_rowwise` — `fp8_recipe=rowwise` (finer scaling, accuracy check).
- (optional) `fp8_mlp_8bit` — FP8 + bnb AdamW8bit (orthogonality + combined memory).

**Measure (all three, not just ppl):**
1. **Loss deviation vs bf16** — target **≤0.1–0.5%** rel error on val loss, matching Ling-1T
   (≤0.1%). This is the *accept gate*: FP8 is only worth it if quality is within budget.
2. **Throughput / MFU** — steps/sec and tok/sec. Expect a *partial* win even though we
   measured 2.4–4.3× on isolated GEMMs: MORPH is also bound by the bf16 attention kernels +
   fused-CE, so end-to-end speedup will be < the GEMM speedup (Amdahl). Honest expectation:
   single-digit-to-low-double-digit % step-time improvement, in line with Ling-1T's 15%
   *whole-model* number (their attention is bf16 too).
3. **Peak memory** — FP8 weights/activations for the converted GEMMs ≈ half those bytes;
   modest at MORPH's 262M scale but real. Compare nvidia-smi peak + torch reserved/alloc.

**Gate:** accept an arm iff Δval-ppl vs 39.46 is within ~0.5% **AND** step-time improves.

**Bit-exact OFF gate (mandatory first check):** with `fp8=false`, assert the model state
and a fixed-seed forward loss are **bit-identical** to the current baseline (no torchao
import side effects, no graph change). Put this in `ignore/verify_fp8.py` alongside: finite
loss + grad flow under compile with FP8 on; the no-Linear-in-both-ternary-and-fp8 assert
(§6); and a coherent generation sample with FP8 on (§7.4).

**Where to run:** the 5090 is a *valid* FP8 training target (we measured the speedup), so
the ablation runs locally — **this does NOT have to move to Spark/TPU/cloud.** (The earlier
"5090 = memory-only" worry is retired.) If a larger-scale confirmation is wanted, the same
torchao path should run on DGX Spark (verify GEMM speedup there first) and on cloud H100.
TPU v6e is excluded (no FP8 HW).

---

## 9. Files to touch (when implementing)

- `morph/training/train.py` — after `model = MORPHTransformer(cfg).to(device)` and
  **before** `torch.compile` (so FP8 modules are compiled), gate on `cfg.training.fp8`:
  call `convert_to_float8_training(model, config=..., module_filter_fn=<scope filter>)`.
  The filter selects by scope (§4) and excludes Block-ELL core MLP, attention kernels' inner
  Linears we keep bf16, gate/indexer/compressor, embeddings, LMHeadMixer, fused-CE LM head.
  Add the ternary∩fp8 overlap assert. Log resolved fp8 config + converted module list to
  wandb. **Preserve the load-bearing build-order** (build → convert → compile → warmup,
  before wandb/dataloader threads — the timing-race fork window).
- `morph/configs/base.yaml` — `fp8`, `fp8_recipe`, `fp8_scope`, `fp8_filter_min_dim`
  (default OFF, §4).
- `morph/model/fp8_scope.py` *(new, small)* — the `module_filter_fn` builder + the
  ternary/fp8 disjointness check (mirrors `ternary_qat.py`'s scope taxonomy so the two
  scope systems agree on names).
- `ignore/verify_fp8.py` *(new)* — the §8 gates: bit-exact OFF, finite-loss + grad-flow
  under compile ON, disjoint-scope assert, coherent-gen-with-FP8.
- *(phase 2, deferred)* `morph/model/fused_ce.py` — FP8 LM-head variant; FP8 Block-ELL
  kernel for the post-compact sparse path. **Not** in phase 1.

---

## 10. Risks (ranked)

1. **sm_120 tooling drift (was risk #1 "does FP8 even work").** *Downgraded by live
   verification* — it works and accelerates today on torch 2.11+cu130 + torchao 0.15
   float8. Residual risk: a torch/torchao upgrade could regress the float8 path on sm_120
   (it's still a less-trodden arch). **Pin torch + torchao versions for FP8 runs**; re-run
   the §2 GEMM benchmark after any upgrade. The cpp-extension-skip warning is expected and
   benign.
2. **Looped-core scaling-state corruption** (trap §7.1/§7.2) — mitigated by mandating
   `dynamic` scaling. If delayed scaling is ever wanted, it needs a custom once-per-step
   weight-cast cache.
3. **Worsening the startup timing-race** (trap §7.3) — FP8 enlarges the recompile surface.
   Mitigated by `eager_on_recompile` + the wedge re-check; fallback = FP8 prelude/coda only.
4. **Smaller-than-hoped end-to-end speedup** (Amdahl: attention/CE stay bf16). Honest
   expectation ~Ling-1T's 15%-class whole-model number, not the 2.4–4.3× isolated-GEMM
   number. The ablation measures the real number; if it's <~5% with a ppl cost, FP8 isn't
   worth turning on at this scale.
5. **Block-ELL / pruning interaction** (trap §7.6) — avoided by not FP8'ing the core MLP.

---

## 11. Sequencing

Implement after the running ternary/8-bit quant factorial completes (no GPU contention).
Phased rollout:
1. **GEMM-only FP8, dynamic scaling, prelude/coda dense MLP** (`fp8_scope=mlp`) — smallest
   surface, no loop/pruning interaction, fixed shapes. Validate the §8 gates + the wedge
   re-check.
2. **Add CCA projections** (`fp8_scope=mlp_attn_proj`) — these are in the variable-M loop;
   this is where the active-set/recompile interaction (§7.3) gets exercised for real.
3. **(Optional, large effort)** FP8 LM-head fused-CE + FP8 Block-ELL sparse kernel. Only if
   phases 1–2 show a worthwhile speedup and we want more.
Never FP8 the attention math kernels (mirror Ling-1T/DeepSeek-V3 — attention stays bf16).
