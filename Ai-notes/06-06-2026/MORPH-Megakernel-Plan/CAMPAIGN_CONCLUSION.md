# Carrier-Engine / Copy-Fusion Campaign — Evidence-Based Conclusion (2026-06-06)

**Verdict: STOP the fusion campaign. The carrier-engine captured its one real slice
(inject-fold, ~1.2% wall). The rest of the apparent "bandwidth glue" is backward
gradient math (quality-coupled) + already-fused/compiled code.** Committed gated, default-off.

## How we got here
After locking n=4 (n=2 was a serious PPL regression and not faster), the megakernel plan
pivoted to the bandwidth-bound HC carrier glue. The whole-step profile showed ~32% of the
step in generic `copy_`/elementwise kernels, which I (wrongly) pitched as ~178 ms of
**forward** carrier round-trips fusable on-chip.

## The measurement that killed the premise
`ignore/profile_copy_stack.py` — `record_function` regions on each forward carrier site
(reliable where `with_stack` is blind: compiled + backward kernels carry no python frame).

**Forward carrier glue = ~21 ms total** (B4/S4096, deploy stack):
| site | CUDA ms |
|---|---|
| inject_add (78 broadcast-adds) | 18.4 |
| loop_cat | 1.1 |
| perm_gather (all 5 gathers) | 0.68 |
| h_clone | 0.26 |
| inv_perm_gather | 0.25 |
| expand_contig | 0.18 |

The gathers/clone are already sub-1ms. **There was no 178 ms forward prize.** The
`copy_` 97ms + `add` 96ms + `mul` 59ms (~255 ms, 34% of step) is dominated by **backward**.

## Localizing the backward glue (ablation diff)
PHASE split: fwd CUDA ~345 ms, optim ~22 ms, backward ~364 ms (record_function reads 0 for
bwd — autograd runs on a thread the CPU region doesn't parent; bwd = busy − fwd − optim).

| config | copy_ | add | mul | fwd |
|---|---|---|---|---|
| A base (engine off, ckpt all, bptt 4) | 98.86 | 96.51 | 60.02 | 352 |
| B inject-fold ON | 97.81 | **83.43** | 58.99 | 345 |
| C no-checkpoint | — | — | — | **OOM** (96 MiB fail, 27 GB used) |
| D bptt_depth 4→1 | **76.31** | **59.24** | **40.94** | 337 |

- **Inject-fold (A→B): add −13 ms** → the injection broadcast-add backward = the carrier-engine slice (built).
- **BPTT 4→1 (A→D): −79 ms of glue** → truncated-BPTT gradient accumulation + checkpoint
  recompute through the weight-shared loop. **Real gradient math, quality-coupled** (depth-4
  is the deliberate learning-signal choice; depth-1 changes the gradient).
- **No-checkpoint OOMs** → checkpointing is mandatory at deploy shape; recompute copies are
  not config-removable.

## Decomposition of the 255 ms backward glue
- ~79 ms BPTT-accum / checkpoint-recompute — irreducible without a quality cost.
- ~13 ms inject — carrier-engine inject-fold (BUILT, ~1.2% wall, −39 launches, mem-neutral).
- remainder (~76 ms copy_ even at bptt=1 + rest) — per-grad-iter glue around already-fused
  kernels + compiled-MLP epilogue (Inductor already optimizes).

Named backward kernels are ALL already fused: hc_post_bwd 21.4, hc_premap_bwd 17,
CCA_conv_bwd 15.4, CSA_bwd 12.9, CCA_prologue_bwd 11.6, gp_dw_bwd 10.6, window_bwd 9.1,
cca_q_bwd 7.25, + compiled MLP FxGraphs ~82 ms.

## Dead levers (measured, not assumed)
- **L2 residency (cheap per-iteration window): NET-NEGATIVE +5.3 %** (740.9→780.3 ms). MORPH's
  functional carrier hops addresses (window points at a stale buffer) AND pinning 62.9/100.7 MB
  of L2 as persisting steals capacity from streaming GEMMs (the isolated −19.6 % microbench had
  no concurrent GEMMs to reveal this). Ping-pong would fix the stale-address half but NOT the
  L2-theft half → no-go.
- **Forward carrier copy fusion: ~21 ms ceiling**, gathers already optimal → not worth a kernel.

## Committed (gated, default-OFF, bit-identical when off)
- `carrier_engine` (inject-into-POST fold) — fp32 parity Δloss=0, grad cos 1.0; the ~1.2 % win.
- `l2_persist` — verified dormant infra (mechanism proven, net-negative in-model; off by default).
- n-generic fused HC mapping kernel — two-oracle gate; n=4 keeps the tuned scalar path untouched.
- `MORPH_PROFILE_REGIONS` env-guarded `record_function` regions in transformer.py (nullcontext
  singleton when off → zero hot-path cost) — reusable copy-attribution infra.

## If revisited later
The only remaining lever touches **locked architecture** (n=4 stream width, BPTT depth) — both
quality knobs, not free perf. A genuine win would need a co-design (e.g. cheaper-gradient stream
readout) validated on PPL, not a kernel fusion. The launch axis is dead (99 % busy).
