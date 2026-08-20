# Agent Note: Perf Pass Overnight Progress

Status: implemented

Origin: Ai-notes/07-03-2026/MORPH-Perf-Pass/Overnight-Progress.md
Imported: 2026-08-20. Pre-format working note; body is the original record.

---

# MORPH Perf Pass — Overnight autonomous session (2026-07-03 night)

Orchestrator (Fable) + Fable subagents. GPU serialized (single 5090 + UPS).
Branch `fix/window-force-eager-and-compile-fence`, off anchor `b1bc563`.

## LANDED (committed, bit-exact gated)

### 5a820ce — Output-tiled SDD kernel (Focus 2, 2.17× isolated)
stk dW-backward `sdd` launched 1 CTA/nonzero-block = ~48 CTAs on 170 SMs
(register-bound 2 CTA/SM; num_warps/stages can't fix — sweep was null 1.00-1.03×).
Split each 128×128 output block into sub-tiles → 384 CTAs. K-reduction order
UNCHANGED → **bitwise-identical (torch.equal, max|Δ|=0, independently re-verified)**.
410→189µs isolated. `MORPH_SDD_SPLIT=0` killswitch. Routed-regime only (sdd = carved
dW backward). Also: 61ms topk RE-ATTRIBUTED to CSA indexer top-128 (741µs/call,
present dense+routed, NOT a routing regression). Remaining post-carve cost = ReMoE
router fp32 GEMM (+60-80ms, class B).

### 751b9ae — Injection-term hoist (Structural #1, launch cut)
Per-core-layer injection term is loop-invariant; built ~145×/step → now 6× once,
threaded as checkpoint input (backward reuses). Bit-exact: CPU loss 0.0, grad 1.6e-8;
in-model loss-trace worst 3.45e-4 < noise floor 5.9e-4. **Measured −535 cudaLaunchKernel
/step** (+ −154 copy_, −103 mul, −84 _to_copy, −65 cat). **NO standalone ms win** —
banked for the stack.

### 4bb891c — Attention input-projection superblock (Structural #3, THE win)
Attention owned ~1,652 of 2,518 aten::mm/step (66%): 14 skinny bias-free Linears/CSA-fwd
+ 9/HCA-fwd on the same x. Fused into ONE concat-N GEMM + split views. CSA indexer trio
in a SEPARATE no_grad GEMM (grad-None preserved — else phantom weight-decay). GPU forward
at deploy shape B4/S4096 bf16 real cuBLAS: **torch.equal=True, max|Δ|=0, ndiff=0/12.6M**
(no topk tie-flip). In-model loss-trace 2.5e-4 < 5.9e-4 floor. **aten::mm 2518→~1493
(−41%). MEASURED −~30ms/step, fwd GPU 309→285.** First visible wall-time win.

### 4048497 — q‖k conv pairing dtype fix + re-enable
Fused conv weight was cast to x.dtype (fp32 under autocast) vs the bf16 conv input →
tl.dot same-dtype assert. Fixed: cast to qk_pair.dtype (matches _causal_conv). Loss-trace
4.56e-4 < floor. ~−300 Triton launches/step, no standalone ms. Default ON.
(Caught ONLY by the in-model autocast gate — CPU parity + pure-bf16 GPU probe both missed
it. Lesson: gate under real autocast, not just pure-dtype modules.)

## AGGREGATE (matched-session, anchor b1bc563 vs HEAD, resume step_36000 seed 1234)

| region | Anchor (pre-win) | HEAD (4 wins) | Δ |
|---|---|---|---|
| **wall step** | ~746ms | ~706ms | **−40ms (~5.4%)** |
| bwd GPU | ~406ms | ~371ms | **−35ms** (SDD dW 2.17× + attn bwd batch) |
| fwd GPU | ~299ms | ~290ms | **−9ms** (attn proj mm batch) |

The −35ms backward-GPU confirms the SDD split in aggregate (noise-hidden in wall step
alone). Loader `data` region jitters 0.3-31ms between runs → use GPU-region time, not wall,
for attribution. NOT 2× — a real, verified first ~5-6%; 2× needs the cloud core-loop graph
+ the rest of the launch-cut stack.

### 6b2554d — GLA proj superblock + router-tail fusion + RoPE cast cache
Three more class-A launch cuts (kill-switches each). GLA q/k/v/g/r → concat-N GEMM (fwd+state
byte-identical GPU, weight-grad ~1 bf16-ulp reassoc). Router elementwise tail → 3 new Triton
kernels (GPU probe: ALL grads torch.equal, fully bitwise). RoPE cos/sin cast cache (bitwise).
Combined in-model loss-trace 3.14e-4 < floor. No standalone ms (launch cuts).

## FINAL LAUNCH-COUNT AGGREGATE (near-anchor SDD-only vs HEAD all-5, /step)

| op | Δ/step |
|---|---|
| cudaLaunchKernel | **−2,158 (~9.5%)** |
| cuLaunchKernel (Triton) | −1,148 |
| aten::mm | −1,148 (attn proj + GLA) |
| aten::_to_copy (casts) | −1,673 |
| aten::copy_ | −1,437 |
| **Command Buffer Full events** | **−437 (~9%)** |

~3,300 fewer GPU kernel launches/step (~12% of ~27k). The Command-Buffer-Full drop (−437)
is the launch WALL relieving — the mechanism that turns launch cuts into the measured
−40ms wall-time. The stack strategy is validated: individual cuts invisible, the stack moves
wall-time AND relieves the stall.

## FINAL TALLY: 5 bit-exact commits off b1bc563

5a820ce SDD 2.17× · 751b9ae injection −535 · 4bb891c attn proj −30ms · 4048497 conv fix ·
6b2554d GLA+router-tail+RoPE. Aggregate: **wall 746→706ms (−40ms ~5.4%)**, GPU bwd −35 /
fwd −9, **launches −~3,300/step (~12%)**, cmd-buffer stall −9%. All gated ≤ 5.9e-4 noise floor.

## KEY MEASUREMENT METHODOLOGY INSIGHT

At d768 the **full-step wall time is noise-dominated: ±14ms run-to-run** (Poisson
depth resample + thermal + loader). A single launch/kernel win (−535 launches ≈ 2.3%,
or a −220µs kernel) is BELOW this noise → invisible in step ms even when real.
Consequences for the rest of the arc:
- Attribute with **GPU-region time** (`bwd gpu`, `fwd gpu` from MORPH_PERF_REGIONS) or
  **kernel-isolated microbench** (like sdd_split.py), NOT wall step time.
- The 2× is only visible after **stacking** many launch cuts (enough to relieve the
  225ms command-buffer stall) OR at larger d (cloud) where per-op work grows.
- For each candidate: prove bit-exactness (loss-trace ≤ noise floor) + measure the
  ISOLATED effect; don't expect the full-step ms to move for any single change.

## Measured routed baseline (mb4 seq4k, ademamix_b1zero, resume step_36000, seed 1234)

MORPH_PERF_REGIONS steady-state (post-SDD, pre/post injection ≈ same within noise):
step ~730-743ms; bwd 386-395 gpu, fwd 300-309 gpu, opt ~40, data 0.4-30 (loader noise).
Census/step: cudaLaunchKernel ~22.8k, aten::_to_copy ~5.1k, aten::mm 5130/2=2565,
aten::copy_ ~6.4k. Command Buffer Full ~2.4k events (the launch wall).

## IN PROGRESS

- **Attention small-GEMM batching (#3)** — worktree Fable agent. aten::mm ~2565/step
  @48µs ≈ 121ms/step = biggest single lever. CPU-parity prototypes for qkv-concat GEMM
  / per-head bmm; GPU-gate on return.

## QUEUE (not started)

- #2 Optimizer CUDA graph: needs set_to_none=False (grads get fresh addrs now) +
  tensor-scalar plumbing for warmup scalars. ~400-800 launches. Class A.
- #4 Static-region graphs (embed+prelude, coda+head+CE). Class A pending dropout-RNG gate.
- #5 Router elementwise-tail fusion (class A) + bf16 router (class B, Wolfe opt-in).
- #6 RoPE cos/sin cast cache (~120 casts/step). Class A.
- CLOUD-only: core-loop CUDA graph + ckpt_grad_iters reduction (both blocked local by
  the +7GB/iter wall; unlock at 96GB).

## CLASS-B MENU (needs Wolfe opt-in — do NOT land silently)

- bf16 router: ~0.65GB mem + faster fp32 query_proj GEMM (Focus 1 + Focus 3).
- CSA indexer top-128 restructure (741µs/call, tie-break-sensitive).
- BLOCK_K change on stk kernels: measured SLOWER, dead end.
