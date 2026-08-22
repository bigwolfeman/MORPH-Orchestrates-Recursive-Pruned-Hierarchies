# Agent Note: Throughput budget and cloud plan

Status: proposed

## Problem

The deploy-scale run is not tractable at today's throughput. Wolfe's 2026-08-21 target:
train on ~8 rented RTX 5090s, order $400 budget, and reach as many tokens as possible on
the cloud model (d2048, 4:8:4, ~850M physical params, ~3.5B effective params per token
before pruning, ~1B after the MORTAR prune to 0.25 density). The earlier perf pass
(2026-07-03, [structural-2x-hunt](2026-07-03-structural-2x-hunt.md)) was boxed to bit-exact
launch cuts and produced ~5%. This note records what is measured, where the remaining
uplift is, what the cloud costs, and what has to be developed on a cheap dev box first.

## Measured baseline (2026-08-21, 5090, `tul_short.yaml` A0: d1024, bs14, seq1024)

| Quantity | Value | Source |
|---|---|---|
| Step rate | 1.235 sps, **17.7k tok/s**, 284.0M params | wandb `morph-tul/tul-a0-acap1`, step 20k |
| Forward FLOPs per token | 1793 MFLOP; core loop 5.69 × 254 = **1444 (80%)**; prelude+coda 347; attention score/value 1.8% | `morph/training/flops.py` on the built model |
| Executed FLOPs per token | ~5.84 GFLOP = fwd + bwd + **4-iteration checkpoint recompute (1016 MFLOP, 17%)** + LM head (not counted by `flops.py`) | same |
| Realized compute | **103 TFLOPS** | 5.84 GFLOP × 17.7k tok/s |
| 5090 cuBLAS bf16 ceiling | **219–223 TFLOPS** at 16k×4k×4k and at the gate_up shape 14336×1024×5632; 150 at a skinny N=512 projection | measured on the local card |
| MFU | **~47% executed, ~40% useful** | |
| Core-loop region | ~57% MFU (A0 step minus A3 step) | A3 = `n_core 0` arm, 49.2k tok/s |
| Non-core region | **~30% MFU**: 0.29 s/step, of which ~75 ms is the 8 layers' GEMMs; ~165 ms is embed, fused CE, HC expand/reduce, optimizer, launches | A3 arm |

Conclusion: there is no 4× in kernels. Kernel-perfect is 2.1× theoretical; ~1.3× is the
practical kernel headroom. The cost is FLOPs per token: the loop and the recompute.

## Proposal

### A. Levers that do not change the loop budget (Wolfe: loop budget is out of scope for now)

| # | Lever | Expected | Class | Status |
|---|---|---|---|---|
| 1 | `ckpt_grad_iters=0` (no backward recompute of the 4 grad iterations) | ×1.2 (−17% executed FLOPs) | exact by config doc; the dropout-RNG-in-recompute caveat from the 07-03 notes must be harness-checked | gate running 2026-08-21 (`ignore/ckpt_gate*_2026-08-21.sh`); **OOM at batch 14 on the local 5090** (26 GB process + ~5 GB desktop), so locally it needs a smaller batch; see Measured follow-up below |
| 2 | Non-core region cut: profile the A3 arm, not A0. Targets: fused CE (a 201 MB fp32 `grad_w` per call), hybrid-embedding + Lorentz `lm_weight()` rebuild, int6 embed STE, HC expand/reduce, the 284M-param optimizer step | ×1.1–1.15 on the full step | mostly exact | not profiled |
| 3 | TUL on (`tul_a1`) | **×1.6 measured** (28.1k tok/s, slightly better CE, `docs/tul-arms-result.md`) | validated on the 5090 arms; not validated with prune/carve/route | measured |
| 4 | FP8 GEMMs on the ternary backbone. Ternary {−1,0,+1}×γ is exact in e4m3, so only activations are newly quantized. 5090 FP8 tensor rate is 2× bf16 | ≤ ×1.3 | B (numerics); `base.yaml` says ternary and FP8 are mutually exclusive per layer — a code constraint from the d768 era, not a math one | not built |
| 6 | Dead saliency scoring: `accumulate_scores()` ran on all 14 MortarLinears every step even when `prune_start` can never fire (the dense TUL arms). Measured **50 ms wall / step** of a 950 ms step (MORPH_PERF_REGIONS `prune` region, A0 batch 14, 2026-08-21). `PruningSchedule.scoring_live` now skips it when `prune_start >= total_steps` | ×1.05 on dense arms; 0 on the deploy recipe (which prunes) | exact (touches only the EMA buffers, never params) | built; `tests/test_lifecycle_phase_transition.py::test_scoring_is_skipped_when_prune_can_never_fire` |
| 5 | Lorentz/int6/QAT fusion and the `_to_copy` cast storm (5,091/step at d768) | part of #2 | exact | see structural-2x-hunt |

Stacked without the loop budget: **×2.2** (1+2+3), **×2.8** with 4. Numbers 1, 2, 4 are
estimates; 3 is measured.

### B. The pruning claim, stated honestly

Pruning to 0.25 density cuts MLP GEMM FLOPs 4× on paper, and MLP is ~85% of per-layer
FLOPs at d2048 (3·d·d_ff = 34.6M vs ~6M for CCA attention). ReMoE at `activation_ratio`
0.5 halves it again. So the routed regime (70% of the run) could cost ~0.3× the dense
per-token FLOPs. **That saving is not measured anywhere.** At d768 the carve netted only
−14 ms of a ~790 ms step (dense MLP −62 ms, BCSR kernels +48 ms), and the router added
+60–80 ms ([perf-pass-phase1](../../implemented/process/2026-07-03-perf-pass-phase1.md)).
"3.5B effective → 1B effective" is true for parameter count; whether the stk `dds`/`sdd`
kernels turn that into wall-clock at d2048 block counts (16×44 = 704 blocks per gate_up
vs 352 at d1024) is the single most important unknown for the cloud budget. It decides
whether $400 buys ~30B or ~70B tokens.

Gate: microbench `morph/sparse/stk` dds/sdd at d2048 shapes, density 0.25, and the
router, against cuBLAS dense at the same shape, before any cloud money is spent.

### C. Multi-GPU

`train.py` has no DDP today; this is greenfield. Data-parallel is linear at best:
~3.5× on 4 GPUs, ~7× on 8. It raises per-GPU efficiency only indirectly, by giving the
VRAM for lever 1.

Communication estimate (not measured): d1024 fp32 grads 1.1 GB → ring all-reduce ~1.7 GB
per GPU per step, ~85 ms at ~20 GB/s PCIe P2P, overlapped with a ~0.5 s backward → near
zero visible. d2048 ~850M params → ~6 GB per GPU per step, ~150 ms with a bf16 comm
hook, mostly overlapped → ~5–10% visible. NVLink does not pay for itself (table below).

MORPH-specific items a generic DDP wrap gets wrong:

1. **Poisson-depth straggler.** `_sample_depths` (`transformer.py:616`) draws per rank.
   The sum of depths over 14 sequences has ~11% std, so the slowest of 8 ranks runs
   ~15% longer than the mean every step and every all-reduce waits for it. Draw depths
   from a rank-shared generator so every rank sees the same depth multiset. Free, exact.
2. **Prune-saliency divergence.** `block_score_ema` (`block_sparse.py:453`) accumulates
   from local grads. Without an all-reduce before each prune event, ranks prune different
   blocks and the replicas desync. Carve and route topology: compute on rank 0, broadcast.
3. Loader rank-sharding on the RAM/mmap placement path; rank-0-only wandb, eval, ckpt.
4. `torch.compile` + DDP + the optgraph static-graph path; the GradScaler `found_inf`
   becomes a collective.
5. Checkpoints pushed off-box every `ckpt_every`; rented boxes die (97–99.5% reliability).

Develop and measure this on a rented **4×3090 box at $0.41/hr** (under $10/day), not on
the train box.

### D. Cost comparison (Vast.ai listings, 2026-08-21, "plus bandwidth")

Relative speed = bf16 dense fp32-accumulate tensor peak relative to the 5090 (the 5090
figure is measured at 219–223 TFLOPS; the others are vendor peaks, with a haircut for
the bandwidth-bound half of the step on 24 GB / ~880 GB/s cards).

| Box | $/hr | $/GPU-hr | rel. speed | speed per $/hr | Notes |
|---|---|---|---|---|---|
| 4×RTX 3090 (US, PCIe 3) | 0.412 | 0.10 | 0.32 | **3.2** | best per $; 3× wall-clock; no FP8; dev box |
| 1×RTX 5090 (KR) | 0.349 | 0.35 | 1.0 | 2.9 | single GPU; 8-day cap |
| 1×RTX 5090 (US, PCIe x8) | 0.361 | 0.36 | 1.0 | 2.8 | |
| **4×RTX 5090 (SI, PCIe 5 x16, 54 GB/s)** | 1.769 | 0.44 | 1.0 | **2.3** | 11-day cap, 99.5% |
| 8×A100 SXM4 40 GB (CA) | 4.273 | 0.53 | ~1.15 | 2.2 | no FP8; 95.7% reliability |
| 8×RTX 4090 | 2.946 | 0.37 | ~0.7 | 1.9 | 24 GB, 883 GB/s |
| 4×RTX 4090 (ES) | 1.343 | 0.34 | ~0.7 | 2.1 | |
| 8×A100 SXM4 40 GB (US-GA) | 6.741 | 0.84 | ~1.15 | 1.4 | |
| 8×RTX 6000 Ada (TR) | 4.809 | 0.60 | ~0.8 | 1.3 | |
| 4×B200 (US-VA) | 24.259 | 6.06 | ~8 | 1.3 | |
| 8×H100 SXM (US) | 23.476 | 2.93 | ~3.2 | 1.1 | NVLink; 2× worse per $ |
| 1×B200 (US-OR) | 5.321 | 5.32 | ~8 | 1.5 | |
| 8×RTX PRO 6000 Blackwell (CA) | 9.622 | 1.20 | ~1.1 | 0.9 | only if 96 GB is required |

8×5090 blades were all rented at the time of the survey; expect ~$3.5/hr by scaling the
4× price.

### E. What $400 buys

4×5090 at $1.769/hr → 226 h ≈ 9.4 days (inside the 11-day cap). Reserve ~15% for setup,
eval, checkpoint I/O → ~770 productive GPU-hours. Bandwidth is billed: ship a
pre-tokenized corpus, not raw text.

| Model | tok/s per GPU | Tokens for $400 |
|---|---|---|
| d1024 (284M, ~0.9B effective/token), today | 17.7k | ~49B |
| d1024 after the ×2.2 stack | ~39k | ~108B |
| d2048 cloud (~3.5B effective/token), today, dense | ~5k | ~14B |
| d2048 after ×2.2, dense | ~11k | ~30B |
| d2048 after ×2.2, **if** the routed regime realizes ~0.3× FLOPs for 70% of the run | ~25k blended | **~70B** |

200B tokens is not inside $400 on any hardware (~$1,500 at d1024 with the stack; ~$5,500
at d2048 dense). 70B at d2048 is reachable only if section B's gate passes.

### F. Research-grade levers (must be tested; not in the ×2.2)

These change numerics or the training recipe. Each needs an ablation arm. Ranked by
expected size.

1. **FP8 with ternary weights** (lever 4). The weight side is exact; the open question
   is activation scaling (per-tensor delayed vs per-block) through the looped core,
   where errors compound `ρ^T`.
2. **Sparse-kernel efficiency at 0.25 density** (section B). If stk cannot reach >2×
   over cuBLAS at d2048, alternatives: 2:4 structured sparsity on the tensor cores
   (hardware 2× on SM120, but 0.5 density not 0.25), or block size 256 for fewer,
   fatter BCSR tiles.
3. **Core-loop CUDA graph** — blocked locally by dynamic `[n_active, S, n, C]` shapes
   and VRAM ([structural-2x-hunt](2026-07-03-structural-2x-hunt.md)); on multi-GPU with
   smaller per-GPU batch and lever 1 the VRAM blocker lifts. Pad `n_active` to a fixed
   bucket to lift the shape blocker. Exact.
4. **Cross-iteration KV sharing (CLA)** — plumbing exists (`cla_capture`/`cla_kv`);
   prior arm `cla_iter1_b4`. Removes K/V projection + CCA conv from T−1 of T iterations.
   Parked pending ablation.
5. **Router cost** (regime 4, 70% of the run): route once per loop-iteration group or
   share gates across iterations; bf16 router. Class B.
6. **HC carrier width** n=4 → n=2: halves the residual's memory traffic (the
   bandwidth-bound half of the step). The n2-vs-n4 quant A/B in remaining-kernel-work
   decides; re-run at d1024+.
7. **Sequence-length curriculum** (short seq early): attention is 1.8% of FLOPs, so
   this buys little here; skip unless the data side wants it.
8. **Fewer gradient iterations (`bptt_depth` 4 → 3)** cuts backward FLOPs of the loop by
   25% — this is a loop-budget change and is explicitly out of scope for now; listed so
   nobody re-derives it.

### G. Iso-depth scaling law (verified 2026-08-21)

`arXiv:2604.21106`, Schwethelm, Rueckert, Kaissis, *How Much Is One Recurrence Worth? Iso-Depth
Scaling Laws for Looped Language Models* (Apr 2026): `L = E + A(N_once + r^phi N_rec)^-alpha + B D^-beta`
with **phi = 0.46** (full BPTT), **0.38** (truncated BPTT), **0.65** (hyper-connections). With HC,
r = 8 iterations of the 8-layer core is worth `8^0.65 = 3.9` cores of capacity, not 8: the cloud
model executes 72 layer-passes per token for ~39 layer-equivalents of capacity. This is the
quantitative form of "the loop is the cost" and the quality-side number for any future
`bptt_depth` or `mean_depth` decision (out of scope now). Goes in `docs/references.md` section 1.

Also verified: `arXiv:2607.14427` (Logan, Jul 2026) per-token fixed-point exit is an
**inference** result (4.94 mean loops for depth-8 quality, 135M on one 4090), not a training
lever. `arXiv:2606.31796` (CHERRY, Kwon and Park, Jun 2026) reports that selected-token
supervision "collapses free generation" — evidence against Rho-1-style selective loss here.

## Alternatives considered

- **Keep the bit-exact launch-cut path from 2026-07-03.** It produced 5% and the
  command-buffer stall analysis says the stack is worth ≤ 2× only at the cloud shape.
  Lost because the measured MFU is already 47%: launch cuts cannot give the asked 1.5–4×.
- **Rent H100/B200 for NVLink and FP8 transformer engine.** 2–2.5× worse per FLOP-dollar
  than 5090s; this model's all-reduce fits PCIe. Lost on price.
- **Rent 3090s for the train run.** Best per dollar (3.2) but 3× the wall-clock, which
  busts the duration caps and the 2–3 day goal. Kept as the dev box.
- **Go straight to 8×5090 without a dev run.** Lost because DDP is greenfield and the two
  MORPH-specific desync bugs (depth straggler, prune saliency) would burn paid hours.
- **Attention-kernel work first** (Wolfe's initial guess). Attention score/value FLOPs
  are 1.8% of the step at seq 1024 and the skinny projections were already fused in
  July. Lost on the numbers.

## Acceptance criteria

1. Lever 1 gated: loss trace within the 5.9e-4 noise floor over 200 steps vs
   `ckpt_grad_iters=-1`, and tok/s reported with `perf/flop_proxy` and peak alloc.
2. A3-region profile exists with a ms breakdown of the ~165 ms non-layer time.
3. stk dds/sdd microbench at d2048 / density 0.25 vs cuBLAS dense, same shapes, reported
   as a ratio. This decides the 30B-vs-70B row.
4. DDP on 4×3090: scaling efficiency measured; depth multiset identical across ranks
   (assert); prune masks identical across ranks after a prune event (assert).
5. $25 pilot on 1×5090: image boots, corpus ingests, checkpoint leaves the box.
6. Only then the $400 run, with the model size chosen from section E.

## Risks

- The relative-speed column is vendor peaks scaled by one measured card; the 3090 and
  4090 rows could be off by ±20%.
- Comm overlap assumes PCIe P2P works on the rented GeForce box; if P2P is disabled the
  all-reduce goes through host memory at ~10 GB/s and d2048 sees ~20% visible.
- TUL is validated dense only; its interaction with prune/carve/route is untested.
- Rented boxes die; an unpushed checkpoint is lost money.
- The pruning FLOP saving (section B) may not materialize; plan against the 30B row
  until the gate passes.
