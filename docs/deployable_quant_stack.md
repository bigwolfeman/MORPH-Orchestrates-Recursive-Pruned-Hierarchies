# Deployable Quant Stack — folding the validated quant work into MORPH (#199 "Ablation F")

**Status:** SPEC (build-out pending). **Branch:** `ablate/quant-buildout`.
**Goal:** assemble the *validated* quantization components into one coherent, branch-free
deployable MORPH configuration — bit-identical to bf16 when every knob is off, no runtime `if`
in the forward, `torch.compile`-clean, full state_dict round-trip — and land it back in the main
MORPH repo.

This is the convergence point of the quant campaign. Each component below earned its place with a
measured result; the components that *didn't* are listed under "Explicitly excluded" with the
evidence that killed them.

---

## 1. The stack (what folds in)

| component | where | config | evidence | cost |
|---|---|---|---|---|
| **Backbone ternary** (forward-STE QAT) | training | `ternary=true ternary_scope=backbone` (symmetric, per-tensor) | D arm = **42.07** vs bf16 39.46; A-series proved asym scales don't help | ~+6.6% ppl for 22.9% of params → {-1,0,1} |
| **int6 embeddings** (per-row STE QAT) | training | `embed_quant=int6` | E2 arm (euc+bigram int6, Lorentz bf16) | embeds are ~72% of params; int6 ≈ near-free at this scale |
| **bf16 attention** (NOT quantized) | — | `attn_proj_quant=off` | #205: attn is ~4.8% params + windowed/bandwidth-bound → no speedup; Efull (all-ternary attn) = **44.92** ≫ D | leaving bf16 is the win |
| **int4 KV cache** (inference PTQ, no QAT) | inference | `MORPHKVCache(kv_quant="int4")` | #201: Δppl int4 **−0.20 ≈ 0** on trained E2_int6; **3.5×** memory on the growing CSA term @128k | ≈ free quality |
| **8-bit AdamW** | training | `adam8bit=true` | #189 factorial: interaction −0.36 → composes free with ternary | optimizer-state memory only |

**Net deployable model:** ternary backbone + int6 embeds (bf16 Lorentz) + bf16 attention, trained
with 8-bit AdamW, served with an int4 KV cache. Weight footprint shrinks on the backbone+embeds;
the O(T) inference state (CSA `C_comp`+`K_I`) shrinks ~3.5× at int4 for long context.

---

## 2. Integration requirements (repo law)

1. **No runtime feature flags in the forward.** Every component is resolved at construction
   (`parametrize` registration) or lives in the inference-only decode path (`kv_cache.py`). The
   training forward graph has no `if quant:` branch.
2. **Bit-identical when off.** All five knobs default off; `ternary=false embed_quant=off
   attn_proj_quant=off adam8bit=false` + bf16 cache must reproduce the exact bf16 baseline.
3. **Disjoint, order-independent application** (train.py order is load-bearing, already wired):
   `apply_ternary_qat(scope=backbone)` → `apply_embed_quant(int6)` → `apply_attn_proj_quant(off)`
   → `create_optimizer(adam8bit)` → `torch.compile`. Ternary targets backbone Linears/CMSBlockLinear;
   embed targets nn.Embedding; attn-proj targets attention Linears — pairwise disjoint by
   construction, guarded by `skipped_already_parametrized==0`.
4. **`torch.compile`-clean:** `mode="default"`, no `fullgraph`, `eager_on_recompile`; recompiles
   under variable Poisson shapes bounded (warmup forces all active-set sizes incl. n_active==1).
5. **state_dict round-trips** all parametrizations (`parametrizations.weight.original`,
   ttq/dual scales if used) after stripping `._orig_mod.` (train.py:238).
6. **KV cache is a separate inference path** — never imported by the training forward.

---

## 3. Verification plan (the F gate)

Build `ignore/verify_deployable_stack.py` asserting, on the real GPU path unless noted:

1. **All-off ≡ bf16 baseline** — bit-identical loss + logits on a fixed seed/input.
2. **Full stack trains** — finite loss over N warmed steps, recompile count bounded (0 runtime
   triton/inductor recompiles after warmup), peak memory within budget at b4.
3. **Full-stack val ppl** — train (or load) the folded checkpoint to 15k; target ≈ D + the int6
   embed cost (expected low-40s). RANK via `fixed_eval.py` (n=200 fixed slice), NOT noisy
   in-train finals; gate the harness on the bf16 baseline reproducing ~39.5.
4. **int4 KV-cache ppl-delta on the FOLDED checkpoint** — extend `measure_kv_quant_ppl.py` to the
   ternary+int6 model; confirm Δppl(int4) stays ≈0 *on top of* the quantized weights (the new
   risk: quantized-weight activations may be less robust to KV-quant than bf16-attn E2 was).
5. **GPU Gate-2 + resident memory** — eager-decode vs fused-forward parity (~2e-2 tol), then
   measure *resident* cache bytes (bf16 vs int4) at 4k/16k/64k via the real decode, confirming the
   analytical table in `kv_quant.py`'s report.
6. **state_dict round-trip** — save/reload the folded model, all parametrizations intact.

Each check names what it does NOT cover (No-Theater): e.g. the ppl gate is a single seed/slice;
long-ctx generation quality under int4 KV is a separate eval.

---

## 4. Explicitly excluded (and why)

- **Attention weight quant** (int8/6/4 or ternary): attn ≈ 4.8% of params and MORPH attention is
  windowed → bandwidth-bound (dead ablation D: low-bit attn 2.3–2.8× *slower*), so it's ppl cost
  for ~zero memory/speed gain. The #205 sweep is retained only as a *precision-tolerance probe*
  (informs FP8-at-scale + KV-quant headroom), not as a deployable component.
- **FP8 attention:** scale-gated — loses on the current windowed/small-GEMM regime; revisit at the
  4B/cloud target on the CSA/HCA dense-compressed branch only.
- **Full ternary (incl. embeddings/attention):** Efull = 44.92, token-starved at 15k (BitNet needs
  ~trillions of tokens); parked for the 4B long-run, still excluding Lorentz.
- **KV-QAT:** unnecessary — int4 KV-cache PTQ is ≈free (#201), so no training-time activation QAT.
- **Gradient accumulation:** out of scope for throughput (trades step latency for effective batch;
  does not raise tok/s) — see the separate throughput-optimization track.

---

## 5. Sequencing / dependencies

1. **#205 sweep result** (running, cron collects ~09:06 CDT) confirms bf16-attn is the right call
   (expect int8 near-D, int4 worse → all justify leaving attn bf16). Until then, bf16-attn is the
   working assumption.
2. **fixed_eval baseline** (bf16 ≈ 39.5) gates every ppl claim.
3. **GPU free** (after #205) for Gate-2 + resident-memory + the folded-checkpoint train.
4. Land the clean diff back in the main MORPH repo (`attn_proj_quant.py`, `kv_quant.py`,
   `kv_cache.py` quant hooks, the five config knobs) once the F gate is green.

---

## 6. Code already in place (this branch)

- `morph/model/ternary_qat.py` — backbone ternary QAT (committed earlier).
- `morph/model/embed_quant.py` — int6/int8 embedding QAT (committed earlier).
- `morph/model/attn_proj_quant.py` — attn-proj int-N knob (default off; gated `ignore/verify_attn_proj_quant.py`).
- `morph/model/kv_quant.py` + `kv_cache.py` hooks — int4/6/8 KV-cache PTQ (gated `ignore/verify_kv_quant.py`).
- `morph/training/train.py` + `morph/configs/base.yaml` — the five knobs wired, manifests → wandb config.

The F build-out = the verification harness in §3 + the folded 15k training run + the main-repo
landing. After it is built and tested, the next track is **throughput (tok/s) optimization** with
no KV reduction, no grad accum, and no ppl regression.
