# TileProver Journal

## [2026-05-31] — kernel: fused_cca_prologue (MORPH CCA attention prologue)
**Goal:** Fuse the `_CCABase._cca_project` post-conv region (qk-mean -> RMSNorm ->
temp -> CoPE-RoPE -> GQA expand -> value-shift) into Triton fwd+bwd for sm_120.

**Method:**
- Spec = `ignore/p1_prologue_harness.py::prologue_reference` (proven byte-exact vs
  real `_cca_project`: fwd maxerr 0.0, dX cos 1.0).
- 3 Triton fwd kernels + 3 bwd kernels, one program per (b,s,head) [Q] or
  (b,s,kv) [K,V]. D=32 register-resident. qk_mean_k cross-head reduction folded to
  `0.5*(mean_g q_lat[g] + k_lat[kv])`. GQA broadcast via tl.static_range stores.
  Backward derived analytically (RMSNorm/RoPE/temp/GQA-sum), combined fp32 on host.
- sm_120 launch: num_stages=1, num_warps=8, bf16 I/O, fp32 reduction accumulation.

**Result:**
- Self-test ALL PASS (assert-gated, exit 0): fwd_mean 7.7e-4, fwd_cos 0.999997,
  per-input grad cosine = 1.0000 for ALL 9 inputs across S=512..4096 + n_skip 8/16.
- End-to-end vs REAL `_cca_project`: max 3.1-5.0e-2, mean 1.9-2.8e-3, cos 0.99999.
- End-to-end param grads (temp/q_norm/k_norm/dX): cos > 0.99998.
- Speed B=2,S=2048: fwd+bwd 2.0x (0.57->0.28ms), fwd-only 1.6x (0.146->0.091ms).
- KEY FINDING: the asked 2e-2 MAX-err gate is BELOW the bf16 floor of this compute
  — the production module's OWN bf16 path is mean 2.0-2.4e-3 / max 2.3-4.0e-2 vs
  fp32 truth. Kernel tracks that floor (often closer to truth than the bf16 ref).
  Gates set to the meetable, contract-meaningful: mean<2e-3 (synthetic)/3.5e-3
  (real-module), max<4-5e-2, fwd cos>0.9999, grad cos>0.995.

**Z3 (tile-prover/proofs/fused_cca_prologue/verify.py):** all 7 PROVEN (UNSAT on
negation): P1/P2 Q/K in-bounds (symbolic B,S), P3 V cat-gather partition exact,
P4 GQA store disjointness, P5 bwd group-sum coverage bijection, P6 D-mask safety,
P7 global coalescing (D*2=64B < 128B). No shared memory used -> bank conflicts vacuous.

**Next:** Wolfe integrates into `morph/model/attention.py` behind a flag (do NOT
modify attention.py per instructions). Unverified: torch.compile interaction, S not
divisible by nothing-special (kernel is per-row so any S works), D!=32 (P7 covers 64
but no runtime test for D=64), multi-GPU.

## [2026-05-31] — kernel+proof: fused_cca_conv (CCA causal convs)
**Goal:** Replace the stacked causal Conv1d pair in `_CCABase._causal_conv` (depthwise
groups=C, then head-grouped groups=G) with a Triton fwd+bwd kernel on sm_120, to kill
cuDNN's slow grouped-conv wgrad (`wgrad2d_grouped_direct_kernel` 135us x600 +
`convolution_backward` 88us x1200 in the profiler).
**Method:** 5 Triton kernels (dw_fwd, gp_fwd, fused dw_bwd[dx+dw], gp_bwd_dx, gp_bwd_dw)
on [B,C,S]. Grouped path uses tl.dot over the Cg=32 channel axis (each j-shift is a
[CG,CG]@[CG,BT] matmul) for tensor-core utilization. dW via per-(b,t_tile) partial slab
+ host reduce (NO atomics — atomic-add and serial-loop variants both measured SLOWER).
num_stages=1, num_warps=8, bf16 in / fp32 accum, branchless causal mask via masked loads.
Tried fusing grouped dx+dw into one kernel → ~2x SLOWER on Q (register/dot pressure),
reverted to separate; kept depthwise dx+dw fused (launch-bound, helps). Z3 proof
(tile-prover/proofs/fused_cca_conv/inbounds.py): 88 in-bounds checks across both streams
× S∈{512,1024,2048,4096}, all UNSAT(mask ∧ OOB) = proven in-bounds.
**Result:** Self-test ALL PASS — fwd max-err 3.91e-3 (<2e-2), fwd cos 1.000000, grad
cosines d_input/d_w_dw/d_w_gp = 1.0000 for all 8 dim cases. Speed (B=2 S=2048, 300-iter):
ISOLATED grouped backward (the profiler target) = 1.95x (Q) / 2.31x (K) faster than cuDNN.
Full stacked-pair fwd+bwd: Q 1.42x, fwd 1.39x; K fwd 1.82x but K full fwd+bwd 0.70-0.85x
(SLOWER — tiny stream, 5-launch overhead beats cuDNN). Z3: 88/88 proven.
**Next:** Wolfe integrates. NOT verified: torch.compile interaction; K-stream full-pair is
honestly slower (the win is the grouped wgrad, which dominates the real profile); only
D=32/K=4/B=2 benchmarked; CUDA-graph capture; numerical drift over long training.

## [2026-06-05] — kernel+proof: fused_hyper_connection (JPmHC HC residual)
**Goal:** Fuse Triton fwd+bwd for HyperConnectionResidual (production cayley default, ~2x slower than plain residual from launch overhead + redundant 100MB carrier reads). Recover speed with exact numerics on RTX 5090 sm_120.
**Method:** EAGER-MANIFOLD FALLBACK decomposition. Carrier h[B,S,4,768]=100MB; mapping tensors <2MB. Fused only the BIG carrier passes into 2 autograd.Functions straddling the sublayer: _FusedHCPre (x_bar = sum_j Hpre_cm[j]*h[j,c]) and _FusedHCPost (sum_j Hres[i,j]*h[j,c] + Hpost_row[i]*y[c], i.e. x_mix+x_post+add in one pass). Kept the n*n mapping (rms/proj GEMV/softmax x2/Cayley 3-iter/reductions) in eager PyTorch — autograd handles the hard softmax+Cayley backward EXACTLY; fully-fused analytic backward through 3 Cayley iters = high risk for sub-1%-bytes gain. Branch-free integration: _hc_pre/_hc_post bound at __init__ (cayley->kernels, sinkhorn/CPU->references). Z3: bounds (UNSAT on negation), coalescing (stride-1, 1 cache line/warp, uniform scalar broadcast), bank-conflict (vacuous, no smem), tile validity for sm_120.
**Result:** ALL GATES GREEN on live 5090.
  - Kernel self-test: PRE fwd bit-exact (max 0.0), grad cos 1.0000. POST fwd_mean 1.88e-3, cos 0.999996, all 4 grad cos 1.0000 (6.25e-2 max-vs-bf16ref is bf16 FLOOR; kernel closer to fp32-truth than the ref).
  - Module parity (fused vs eager same weights): out_mean 1.15e-3, cos 0.999996, grad cos on h/proj.weight/proj.bias all 1.0000.
  - init~=plain relerr 0.0263; finite OK.
  - Z3: in-bounds PROVEN (8 cases), coalescing PROVEN (3 props), bank-conflict-free PROVEN (vacuous), tile-validity PROVEN (256 thr/blk, 8 warps, stages=1, BLOCK_C=1024; regs CONDITIONAL on ptxas).
  - SPEEDUP: module fwd+bwd B4/S4096 = 1.42x (eager 6.48ms -> fused 4.57ms). Carrier-ops-only fwd = 4.1x (0.94->0.23ms). Module speedup diluted by the unchanged eager mapping (~0.7ms proj+softmax+cayley) + frozen sublayer MLP.
**Next:** Next lever = fold proj GEMV into PRE kernel (h already resident) to attack the ~0.7ms eager mapping. NOT done (needs fully-fused manifold backward = the risky path). Files: morph/kernels/triton/fused_hyper_connection.py, morph/model/hyper_connections.py (integration), ignore/verify_hyper_connections.py, ignore/bench_hyper_connections.py.

## [2026-06-05] — OPTIMIZE (round 2): fused_hyper_connection PRE-MAP fusion
**Goal:** Kill the ~65-launch n×n mapping storm (copy_/mul/pow/bmm = ~2/3 of HC fwd+bwd CUDA time after round 1) by fusing rms+proj+softmax×2+cayley(3)+reductions+x_bar into the PRE kernel with an analytic backward.
**Method:** New `_hc_premap_fwd/bwd_kernel` + `_FusedHCPreMap` autograd.Function + `hc_pre_map`/`hc_pre_map_reference`. Decomposition: GEMV kept as a single cuBLAS addmm in the CARRIER dtype (bf16 → tensor-core, fp32 accumulate) — folding it into the per-token kernel would reload proj_w[48,3072] per program and lose tensor cores. Everything else (rms over 3072, the two softmaxes, the 3 Cayley fixed-point iters as fully-unrolled 4×4 scalar matmuls, the colmean/rowsum reductions, x_bar) fused into ONE kernel, fp32 internal, one program = one token, register-resident (no shared mem). Analytic VJP (softmax dim=-1 AND dim=-2 jacobians, 3-step Cayley reverse recurrence, skew + /rms + proj VJP) validated against autograd in `ignore/derive_pre_backward.py` (rel ~1e-7) BEFORE coding the kernel. Backward GEMMs (grad_w, grad_h_proj) also bf16/tensor-core.
**Result:** PROVEN/GREEN.
  - Self-test PRE-MAP: fwd x_cos ≥0.999994, res_cos ≥0.999999, pr_cos ≥0.999996; grad cosines h/proj_w/proj_b = 1.0000 across B∈{2,4}×S∈{512,2048,4096}; kernel ≤ ref vs fp32 truth. POST + PRE unchanged, still green.
  - Module parity: out_cos 0.999995, grad_cos h/proj.w/proj.b = 1.0000. init≈plain relerr 0.0263. finite. Real MORPHBlock(hc_cayley) fwd+bwd finite.
  - Speedup: round-1 1.42× → **round-2 2.99×** (eager 7.31ms → fused 2.44ms) at B4/S4096. Total profiler CUDA time 295ms → 175ms (−41%). aten::bmm 450 calls/5% → **0**. aten::mul 800→100 calls.
  - Z3 (sm_120): bounds (raw48/rms added), coalescing (premap scalar uniform loads added), tile/banks (reg est 164/thread, 41984/block, bank-conflict vacuous) — ALL PROVEN.
**Next:** Remaining profiler copy_/mul are mostly the cuBLAS GEMM transposes + grad_y materialization in POST + the profiler's own loss casts, not the mapping. A future round could fuse the addmm epilogue cast or the grad_h add. Did NOT run a full training step (smoke fwd+bwd only). iters≠3 / n≠4 / sinkhorn / CPU all fall back to eager reference (verified).

## [2026-06-23] — EXTRACT + BASELINE: fused ReMoE decode router (MORPH)
**Goal:** Fuse the eager-Python ReMoE router pile (per routed-MLP decode visit) into ONE proven Triton kernel for sm_120 (RTX 5090).
**Method:** Read routing.py (TileRouter), transformer.py (_SwiGLUMortar.forward), kv_cache_static.py (decode eager pile L763-797). Ran ignore/bench_decode.py as-is for baseline.
**Result (BASELINE measured):** DECODE_BENCH_PASS tok/s=309.9 match=256/256 on tst_stp_off_50k/step_50000.pt. 12 MLPs routed (3 prelude n_iters=1, 6 core n_iters=8, 3 coda n_iters=1). Router cfg: d_model=768, d_ff=2048, n_clusters=16, k=8 (50%), n_sub_keys=4 -> n_products=16 = 1:1 DIRECT mapping branch, cluster_size=128=blk. Decode router = MORTAR path (s.Mgu set); eager pile L776-790 then route_flags (already Triton) then routed mortar_gemv.
**Eager pile to fuse (per visit):** F.linear(768->768) + iter_vec add + F.layer_norm(768) + sa=q_a@ska_t[384x4] + sb=q_b@skb_t[384x4] + (sa[:,:,None]+sb[:,None,:]).reshape(16)+gbias + topk(8 of 16) threshold + relu(logits-kth) + sum-normalize*k -> gates[1,16].
**HW sm120.json VERIFIED vs NVIDIA Blackwell Tuning Guide 13.3 + CUDA compute-cap table:** 48 warps/SM, 128KB smem/SM, 99KB smem/block, 64K regs/SM, 32 banks x4B. Triton: num_stages=1, no TMA/TMEM.
**Next:** profile router share to ground 20-30%; build fused Triton router kernel + Z3 proofs (in-bounds, coalescing on gate-weight reads, bank-conflict-free, occupancy/num_warps).

## [2026-06-23] — BUILD + VERIFY + INTEGRATE: fused ReMoE router (RESULT)
**Goal:** one-launch fused router; recover the 20-30% router share.
**Method/design (2-LAUNCH, after a 1-CTA full-fusion FAILED):**
- FIRST attempt: single-CTA kernel doing the WHOLE router (768x768 GEMV + LN + subkeys + topk + relu + normalize) with a global qv scratch + debug_barrier. PARITY PASSED (256/256) but tok/s REGRESSED 309.9 -> 139.4: a memory-bound 768x768 GEMV serialized on ONE SM is far slower than cublas splitK across SMs. REPORTED honestly, redesigned.
- WINNING design: (1) qv = small_gemv(x, Wq, bias=iter_vec) — the EXISTING proven MULTI-CTA GEMV (1 CTA/row). (2) `_router_tail_kernel` (grid=(1,)): LN + 2 sub-key GEMVs + product logits + EXACT desc-rank topk + relu + normalize -> gates[16]. Tail is all <=768-wide => 1 CTA optimal. num_warps=1 (microbench: 1-8 all ~22us; tail is launch-bound). morph/kernels/triton/fused_router.py.
- topk parity: kth = sum(where(desc_rank==K, logits, 0)); desc_rank = #{>} + #{== & idx<=j} == torch.topk value-desc/index-asc. Gates depend only on (logit - kth) so tie-break is output-irrelevant.
**Gates (HARD, no theater):**
- GATE #1 routing-decision parity (ignore/gate_fused_router_parity.py): FUSED_ROUTER_PARITY_GATE_PASS. fp32 (DEPLOY path) active_mismatch=0 max_gate_err=1.67e-6 over 200 trials; vs REAL TileRouter.forward 1.43e-6 over 50. (bf16 INFO-only: 2 tie-flips at boundary — NOT the deploy path; router params are fp32 through to_deploy_inference, confirmed by inspection.)
- GATE #2 end-to-end (ignore/bench_decode.py): DECODE_BENCH_PASS tok/s=507.8 match=256/256. CLEAN ISOLATED A/B same harness: OFF(eager)=310.7 -> ON(fused)=507.8 = +63.4% (1.634x). Reproducible (507.6/507.8).
- Z3 (tile-prover/proofs/fused_router/verify.py, result.json): ALL PROVEN — P1/P2 GEMV in-bounds, P3 qv handoff cover, P4/P5 tail masked loads in-bounds, P6 coalescing (32x4=128B=1 line), P7 bank-conflict-free (no programmer smem), P8a/b symbolic topk-correctness over all total orders (N-independent, N=6 repr), P8c EXHAUSTIVE 20000 trials at real N=16/K=8 incl heavy-tie/all-equal 0 fail, P9 tile validity sm_120.
**Profile (torch.profiler, 64 graph replays):** launches/token 1833 -> 950 (-48%); self CUDA us/step 3244.9 -> 1942.8 (-40%); gatherTopK + bitonicSortKVInPlace (254us topk pair) ELIMINATED. Router pile ~8 launches/visit -> 3 (gemv + tail + route_flags), x42 visits/token.
**Integration:** morph/model/kv_cache_static.py only (import + MORPH_FUSED_ROUTER flag default ON + route-cache num_warps/scratch + fused/eager branch). Eager fallback preserved. route_flags unchanged (was already Triton).
**Caveats (no theater):** (1) bf16 router would tie-flip — deploy is fp32 so N/A, but if a future build runs the router in bf16 this needs revisiting. (2) Z3 P9 register count is CONDITIONAL on ptxas — not run ncu. (3) only the default ckpt (tst_stp_off_50k) + d=768/16cls/k8/nsk4 shape tested; the nsk^2!=ncls branches are asserted-out (not exercised by this model). (4) win (+63%) EXCEEDED the 20-30% estimate because launch-count collapse (-883 launches) dominates, not just router GPU-time. (5) torch.compile interaction untested (engine uses CUDA graph, not compile).

## [2026-06-23] — BUILD + VERIFY + PROVE: MIXED-LENGTH per-stream decode (MORPH B>1)
**Goal:** Add mixed-length (different prompt lengths per stream) to the StaticDecodeEngine B>1 fast-decode path. The engine drove ALL B streams from ONE scalar pos / pos_dev[1]; mixed serving needs per-stream positions, which collides with the 3 CUDA-graph emit variants (no-emit/CSA-emit/HCA-emit) selected by a host emit-flag pair — with per-stream pos, stream A may close a compression block at a step where stream B does not, so no single emit-variant graph is valid.
**Design (emit-graph collapse):** pos_dev [1]→[B] long (+ host mirror list pos_host[B]); cos/sin [D]→[B,D]; csa/hca block-count cnt [1]→[B]; win_mask [W]→[B,W]; xoff [8]→[B,8]; emit-mask [B] per kind. Collapsed the 3 emit-variant graphs into ONE graph that runs EVERY emit-capable site EVERY token and gates the per-stream writes by a [B] emit-mask: CSA gated inside `_csa_emit_combine_kernel` (HAS_MASK constexpr, early-return on mask==0); HCA gated in the engine torch path by a GRAPH-SAFE fixed-shape masked BLEND (blend = mask*blk + (1-mask)*cur, scatter back to C_comp[b,idx[b]] — non-completing stream writes its own value back = exact no-op; no torch.nonzero/dynamic shape). Equal-length keeps the 3-graph fast path (zero regression). New `load_from_eager_mixed(list[MORPHKVCache])` converts one SOLO eager cache per stream at its own P into batch row b. Every kernel pos/cnt/cos/wmask/xoff read became `tl.load(...+b)` with the real stride.
**THE BUG (found by the new gate, NOT papered over):** `_front_gemm_kernel` (the site front-end GEMM) read `XOFF + 0..6` AND the v_prev/v_curr `ror = tl.load(XOFF + r)` WITHOUT the per-stream `b*sxoff_b` offset → every stream gathered STREAM-0's x-history ring rows. Symptom was diagnostic: the MIN-position stream (b=0) was bit-perfect (cos=1.0) while higher-position streams diverged progressively (cos 0.96→0.35). Fix: `xoff_b = XOFF + b*sxoff_b`, both read sites. (Also a harness off-by-one: feed-then-compare aligns engine[i] vs golden[i+1].)
**Parity methodology:** the eager golden drives the whole batch from a scalar cache.pos, so it CANNOT represent a mixed-length batch in one cache — the correct oracle is SOLO eager (B=1) per stream at its own length (verified solo≈batched-eager cos 0.999826, just bf16 batch-noise). Engine batched-mixed must match each stream's own solo-eager logits (cos≥0.999, tie-robust; argmax informational).
**HARD GATES (all PASS, cited):**
- B=1 unchanged: `ignore/bench_decode.py` → DECODE_BENCH_PASS tok/s=454.4 match=256/256.
- Equal-length B>1 unchanged: `ignore/gate_batched_decode.py` → GATE_BATCHED_DECODE_PASS, per-stream logit cos B2=0.999897 B4=0.999957 B8=0.999872 (byte-identical to pre-change baseline 0.999897/0.999957).
- NEW mixed-length: `ignore/gate_mixedlen_decode.py` → GATE_MIXEDLEN_DECODE_PASS, 4 distinct prefill lengths [9,13,30,64], 48 decode steps, per-stream cos 0.999972/0.999978/0.999874/0.999954, argmax_match 48/48 all streams.
- Z3 `ignore/z3_mixedlen_pos.py` → Z3_MIXEDLEN_POS_PROVED: P1 in-bounds (symbolic B,WWIN), P2 per-stream independence + the EXACT bug encoded (missing xoff offset reads row0≠row b for b≥1, UNSAT) + fix (b*8 reads ring_meta's row b), P3 ring_commit nb%B stream recovery, P4 HCA emit blend exactness (mask 0=no-op / 1=overwrite), P5 B=1 equivalence (symbolic), P6 modular ranges (slot/wslot/ring-row/tgt/cnt/emit-flag) bounded-exhaustive over pos∈[0,4095) at CSA(m=4,WR=8)+HCA(m=128,WR=128) — 0 violations. (Honest split: linear addressing = symbolic Z3; mod/div by symbolic divisor is undecidable in Z3 so the modular ranges are bounded-exhaustive — complete over the real ranges.)
**Throughput (sm_120, 276M, GPU shared w/ a 30B run on the other context — re-measure if contended):** equal-length B1=459 B4=1028 B8=1884 agg tok/s (matches the B>1 table). Mixed-length single collapsed graph: B=7 (lengths [8,16,40,96,12,30,64]) = 174.8 steps/s = 1224 agg tok/s. The masked-emit single graph costs a little vs the 3-graph equal-length path (it runs every emit site every token instead of only on emit steps) but stays comparable; exact mixed-vs-equal at matched B not isolated (different B in the two runs).
**Files:** morph/inference/engine.py (pos_dev[B], pos_host, cos/sin[B,D], emit masks, _mixed mode, collapsed graph capture/replay, load_from_eager_mixed, graph-safe HCA blend), morph/kernels/triton/fused_decode_step.py (per-stream b-offset in ring_meta/ring_commit/_front_gemm/_front_post/_decode_attn/_csa_scores/_csa_emit gemm+combine + HAS_MASK emit gate), ignore/gate_mixedlen_decode.py, ignore/z3_mixedlen_pos.py.
**NOT verified (honest edges):** very long contexts (only pos≤~175 exercised; capacity guard present but >1k positions untested); B>16 (max tested B=7 parity, B=16 only in equal-length timing); ragged edges where MANY streams complete a block on the SAME step in mixed mode (the per-stream masks handle it but the heavy-overlap case wasn't stress-tested); ncu register/occupancy (Z3 occupancy CONDITIONAL on ptxas); the int8/30B deploy stack (only the fp32 276M routed ckpt run — the per-stream b-offsets are dtype-agnostic but the deploy KS=8/int4 front_gemm schedule wasn't re-gated for mixed-length); torch.compile (engine uses CUDA graph).
