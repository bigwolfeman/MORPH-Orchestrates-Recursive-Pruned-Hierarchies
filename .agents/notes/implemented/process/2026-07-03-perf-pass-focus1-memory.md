# Agent Note: Perf Pass Focus1 Memory

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Focus1-Memory-Findings.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# Focus 1 — Post-carve memory root-cause (measured 2026-07-03)

## Headline correction to Phase-1 framing

Phase-1 reported routed +1.5–2GB over dense (16.2→19.3GB). That was **confounded**:
the numbers mixed measurement modes. The honest **fused-path** delta at mb4/seq4k is
**+0.66GB allocated, +0.82GB reserved** — under 1GB, and dominated by the router's
fp32 intermediates, NOT fragmentation.

## Authoritative fused-path peaks (MORPH_MEM_PROBE peak-only, no hooks, real kernels)

| Config (fused, mb4, seq4k, ademamix_b1zero) | peak alloc | reserved | gap |
|---|---|---|---|
| Dense (TST superposition, bag=6, pre-prune) | 18.22 GB | 18.86 GB | 0.64 GB |
| Routed (resume step_36000, post-carve, bag=0, steady-state 36015+) | 19.26 GB | 20.06 GB | 0.80 GB |
| **Routed − Dense** | **+1.04 GB** | **+1.20 GB** | +0.16 GB |

(First 3 post-carve steps read lower, ~18.88/19.68; reserved climbs to 20.06 and
settles by step ~36015 as the router gate/load-EMA buffers stabilize. Steady-state
is the honest number.)

- Desktop apps (electron/vesktop) hold ~0.88GB; total GPU 31.4GB.
- Routed reserved 20.06 + desktop 0.88 = ~20.9GB used ⇒ **~10.5GB headroom at routed mb4.**

## Allocation attribution (allocator snapshots, mb2, eager-HC forced)

CAVEAT: allocator history hooks NULL the fused HC autograd Function (SystemError:
apply returns NULL). Snapshots REQUIRE MORPH_HC_FORCE_EAGER=1. Eager HC materializes
huge carrier intermediates the fused kernel avoids → HC blocks in the snapshot are
**eager-inflated artifacts, discard them**. HC is identical in both dense+routed eager
snapshots, so it CANCELS in the dense-vs-routed diff. Valid, non-HC, routed-only:

- `routing.py:163 forward` (router fp32 q_proj + x_flat.to(fp32) upcast): **324MB @ mb2 → ~650MB @ mb4**. This IS the +0.66GB alloc delta. Router params default fp32; x arrives bf16 → real upcast + fp32 GEMM [16384,768]×[768,768].
- `stk_dds` carved MLP output 576MB @ mb2 ≈ dense `block_sparse.py:431` F.linear 564MB → **same output shape, NOT a real increase.**

## Fragmentation: real but small in the fused path

- Eager-HC mb2 showed reserved 18.98 vs alloc 14.0 = 5GB gap → **that gap is eager-HC's
 own variable-size intermediate churn, not the deployment path.**
- Fused mb4: gap is 0.64GB dense / 0.80GB routed → routed-specific fragmentation only
 +0.16GB. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (bit-exact, alloc-only)
 would reclaim ≤0.8GB total — LOW value, deprioritized.
- The carve-rebuild path already frees correctly (train.py:1700-1718:
 optimizer.state.clear + gc.collect + empty_cache before rebuilding).

## Router fp32 reclaim options

- No free bit-exact hoist: `iter_embed.to(proj_dtype)` is already a no-op (fp32 Param);
 `x_flat.to(fp32)` is a genuine required upcast because router params are fp32.
- bf16 router = **class B** (numerics change): halves the ~650MB AND speeds the fp32
 query_proj GEMM. opt-in. Parked with the class-B menu.

## Reframe of ckpt_grad_iters (the actual speed knob)

the premise "memory blocks ckpt_grad_iters" is NOT confirmed for local d768 mb4:
there's ~11GB headroom at routed peak. ckpt_grad_iters=-1 (checkpoint ALL core iters)
is current. Reducing it trades memory for recompute-saved speed. **Next: measure
routed mb4 with ckpt_grad_iters=2 for BOTH peak mem and ms/step** — if it fits in
headroom and the UPS/desktop margin holds, it's a bit-exact* speed win available NOW.
(*must gate RNG-consumption-under-checkpointing per plan; changing WHICH iters
checkpoint can shift the dropout RNG draw sequence.)

The memory premise may hold on the CLOUD target (dual RTX Pro 6000, larger d, DDP,
higher mb) — flag for cloud but do not assume locally.

## ckpt_grad_iters MEASURED — hard-blocked locally (2026-07-03)

bptt_depth=4 ⇒ 4 grad-iters, ckpt_grad_iters=-1 checkpoints all 4 (current default).
Tested routed mb4 resume, +training.seed=1234, fused path (no hooks, real signal):

| ckpt_grad_iters | eager core-iters | result |
|---|---|---|
| -1 (all 4 ckpt) | 0 | 19.26GB alloc / 19.88 reserved — baseline |
| 2 (last 2 eager) | 2 | **OOM at 24.66GB alloc** (fwd hc_post empty alloc) |

transformer.py:281 comment names this exactly: retaining cross-iteration activations
= "+7 GB/step at deploy shape; the post-compact OOM". Each eager core-iter retains
full [B,S,n=4,C] carrier + attn + MLP ⇒ multiple GB. So local headroom (~10.5GB at
ckpt=-1) evaporates after 2 eager iters.

**CONCLUSION: on the local 5090, ckpt_grad_iters cannot be reduced below -1.**
The bit-exact reclaim available (router bf16 ≈0.65GB [class B] + expandable_segments
≈0.8GB [class A]) totals ~1.5GB — far short of the ~7GB one eager-iter step needs.
So memory reclaim does NOT unlock the local speed knob. the premise ("memory
blocks ckpt_grad_iters") is CONFIRMED, and stronger than expected: it's blocked by
a wide margin locally, unblockable by the cheap fixes.

**Where ckpt_grad_iters DOES pay:** the cloud target (dual RTX Pro 6000, 96GB each).
There the +7GB fits trivially ⇒ reducing checkpointing is a large FREE speed win
(saves core-iter recompute: ~2×35ms + ~24 router/attn recompute calls). BLOCKER before
using it anywhere: gate the RNG-consumption-under-checkpointing claim. use_reentrant=
False (transformer.py:809) stashes+restores RNG for the recompute, and each core-iter's
forward runs once regardless of n_ckpt, so in THEORY changing n_ckpt shifts neither
forward RNG nor gradients. Could not complete the empirical gate (ckpt=2 OOMs before
producing a trace); on cloud, run the ckpt=-1 vs ckpt=2 loss-trace noise-floor gate to
confirm bit-exactness before landing.

## Bit-exact gate infra validated

Two identical ckpt=-1 routed runs (same seed, resume) diverge 0 at step 1 → up to
~6e-4 by step 11 (pure Embedding+Triton-bwd atomics). That IS the noise floor for any
routed-regime change-vs-baseline comparison. MORPH_EXACT_TRACE traces:
ignore/perf/ckpt_base.trace, ckpt_base2.trace.
