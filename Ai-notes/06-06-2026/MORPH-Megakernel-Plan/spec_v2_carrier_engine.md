# MORPH Fused Carrier-Engine — Megakernel Spec v2 (2026-06-06)

Supersedes the v1 carrier-fusion spec. v1 was scoped DOWN ("no loop-spanning megakernel,
backward impossible"). v2 raises the ambition after two corrections from tonight.

## 0. What changed tonight (the corrections that drive v2)

1. **"No training-megakernel backward" was overstated → RETRACTED.** FlashAttention is a
   fused kernel whose *backward* is fused-kernels-with-recompute — the existence proof. The
   absence of an *open* multi-layer training megakernel is **proprietary + hard, not
   forbidden** (Wolfe's hypothesis, correct). And tonight I hand-derived a fused glue-backward
   for the injection (fp32 grad cos = **1.000000**) — the carrier-glue gradient math is known
   and writable. So an ambitious fused training kernel WITH a hand-written backward is ON.

2. **MORPH is BANDWIDTH-bound, not compute-bound (measured).** Whole-step aten-op profile
   (B4·S4096, bf16): `aten::mm` (GEMMs) = **133 ms / 20% of busy**; the other **~80%** is
   memory-bound carrier traffic — `copy_` 91, `mul`+`add`+`sum` ~160, HC kernels (elementwise
   + 4×4 FMA, *not* tensor cores) 88, etc. 99% "busy" = mostly moving bytes, not matmul. On
   the **quant stack** (ternary GEMMs faster) it's EVEN more bandwidth-bound (~12% GEMM). The
   prize is the 80%, and it must be attacked HOLISTICALLY — tonight's per-op injection-fold
   netted only +0.76% wall (and was reverted: +2.9% peak mem, +564 launches) because the glue
   is hundreds of scattered small ops; you can't win a bandwidth war one `copy_` at a time.

## 1. North star

A fused **carrier-engine**: do the entire per-layer carrier transformation in MINIMAL HBM
passes with ONE unified hand-written backward —
  IN side:  inject(term)  →  RMSNorm(n·C)  →  HC-pre (proj GEMV + softmax² + cayley + x_bar)
  [sublayer attention/MLP GEMMs — cuBLAS/Triton library calls, the 20% real compute, UNTOUCHED]
  OUT side: HC-post (Hres mix + Hpost scatter)  →  residual add  →  cast
The engine OWNS the glue (the 80%); GEMMs stay library calls. **Stretch goal:** cross-LAYER
fusion — POST(i) + inject(i+1) + PRE(i+1)-prologue in one kernel, carrier resident across the
GEMM boundary (what L2-persistence partially gives for free; CUDA C++ if it provably beats it).

## 2. Carrier — LOCK FIRST ("last chopper out of Nam")

Each kernel change forces a re-baseline AND kernels tuned for one carrier shape/precision need
retuning for another. So lock the carrier BEFORE writing kernels:
- **n (streams):** 2 if the 7k ppl A/B holds (running: `n2_quant_7k` vs `n4_quant_7k`, full
  quant stack), else 4. **Kernel is n-GENERIC** (constexpr N) so it serves either — but we
  baseline + tune on the locked n. (Today's fused HC kernel hard-asserts N==4; v2 must support
  N∈{2,4}, which is also why n=2's TRUE perf needs this kernel — n=2 currently falls back to
  eager HC.)
- **Precision:** bf16 carrier default; **fp8/int8 carrier storage** as a gated, **ppl-tested**
  option (the bandwidth halver — residual stream is the most regression-prone place, so it is
  strictly ppl-gated, never assumed).
- **Quant-aware:** coexist with the deploy stack — ternary backbone (proj/MLP weights ternary,
  GEMMs take the ternary path), int6 embeds, 8-bit AdamW.

## 3. The two hard lessons from the reverted fold (NON-NEGOTIABLE gates for v2)

- **MEMORY ≤ baseline.** The inject-fold added +0.42 GiB by saving `term` for backward (the
  old external add never retained it). v2 MUST NOT inflate retained activations. Design choice:
  fuse inject into the **PREVIOUS** layer's POST *store* (post writes h_inj directly → saves
  h_inj like baseline, no extra `term` retention), OR recompute term in backward. Gate on peak.
- **LAUNCHES ≤ baseline.** The inject-fold added +564 launches via the algebraic proj-split's
  small GEMV/cast/reduce ops. v2 fuses INTO existing kernels (no new small kernels) and caches
  loop-invariants (e.g. proj_w_sum, version-keyed like the FP8 weight cache).

## 4. Backward strategy

- GEMMs' backward: cuBLAS/Triton library (standard).
- Glue backward: ONE hand-written fused backward = (the inject-grad derived tonight) ∘ (the
  existing analytic premap/post backward, already in the repo) ∘ (residual/cast). RECOMPUTE
  cheap glue in backward (FlashAttention-style) rather than store. Must compose with
  truncated-BPTT (only last `bptt_depth` iters get grad), gradient checkpointing
  (use_reentrant=False), and active-set shrinking — all already in the loop.

## 5. Portability (deploy: B200 / H100 / RTX PRO 6000 / multi-GPU)

- ONE portable Triton path (auto-ports sm_90/100/120, auto-tunes per arch). n-generic,
  quant-aware. No tcgen05/TMA-multicast/big-SMEM dependence (we're elementwise + 4×4 FMA).
- CUDA C++ cross-boundary residency = stretch, **A/B-gated**: must PROVE it beats the
  Triton + L2-persistence baseline (don't assume).
- Multi-GPU: `getCurrentCUDAStream()`, `CUDAGuard`, `.contiguous()`, prefer
  `torch.library.custom_op` + `register_autograd` + `register_fake`.

## 6. Sequencing

1. **Lock carrier:** n=2 ppl A/B (RUNNING, 7k, quant) + quant stack.  ← in progress
2. **Re-baseline** on n=2+quant (the n=4-quant arm IS the deploy baseline; the engine is
   measured against THIS, not the bf16 numbers).
3. **Build the carrier-engine** against the final carrier, n-generic, quant-aware.
4. **Gate:** bit-exact fwd (fp32 grad cos > 0.9999) → model parity vs eager (Δloss=0,
   `parity_model_fold.py` pattern) → **peak mem ≤ baseline** → **launches ≤ baseline** →
   re-profile (the bandwidth-bound 80% actually dropped) → 7k/15k ppl parity.
5. **15k validation** on the deploy config (wandb FULL config, Hydra).

## 7. Risks / unverified

- n=2 may regress ppl (the A/B decides; do NOT build the n=2 kernel as if it's settled —
  but the n-generic design means the kernel serves n=4 too if n=2 loses).
- fp8/int8 carrier precision: ppl risk, gated.
- The holistic engine must MEASURABLY beat the sum-of-per-op-folds (tonight proved per-op is
  sub-1%; the engine's thesis is that fusing the whole IN/OUT pass + cross-layer residency
  clears the relocation overhead that killed the per-op fold — UNPROVEN until built+measured).
- Cross-layer residency win unproven (L2-persistence may already capture most of it).

## 8. Tooling already built tonight (reuse)
- `ignore/verify_inject_fold.py` — kernel bit-exact gate (fp32 cos 1.0 pattern).
- `ignore/parity_model_fold.py` — full-model fp32/bf16 parity vs git-stashed baseline.
- `ignore/profile_step.py`, `attribute_elementwise.py`, `profile_hc_idle.py` — the
  roofline/attribution harnesses (busy/idle, aten-op breakdown, per-op timing).
- The reverted inject-fold's backward derivation is in `memory/project_morph_carrier_fusion.md`
  (the glue-backward math is correct and reusable in the engine).
