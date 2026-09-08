# Olympiad panel throughput audit (2026-09-08)

The pit stop Wolfe asked for before E16 ("take a quick pit stop and optimize this"). A
perf-optimizer subagent measured `tul_oly_mask` (the E15/E16 mask arm: slot loop, TG
restriction, Poisson slot draw mean 12, seq 512, micro 12 × accum 2) at commit `ea57f69`
in the `/home/wolfe/morph-to` worktree, 60-step runs, one process at a time,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Every number is `[perf] step N mean
over 20` from `MORPH_PERF_REGIONS=1`, averaged over the step-39 and step-59 reports (40
steady steps). Scripts and per-run summaries are in `scripts/` and `summ/`; the run logs
were scratch and are gone. Run-to-run spread on one config, five repeats: mean 1359 ms,
sd 33 ms (2.5 %); anything under ~7 % is unresolved.

## Where the step goes (micro 12, eager, one micro-batch, 16 steady steps)

| phase | ms | share of forward |
|---|---|---|
| `_tul_front` (embed + 4 prelude layers, 640 positions) | 32.4 | 9 % |
| `_tul_core` (slot loop: 6 core layers × Poisson 12..16 iterations, 64 slots) | 237.0 | 66 % |
| of which `_slot_gain_penalty` (two extra core steps) | 21.1 | 6 % |
| `_back_region` (4 coda layers) | 27.6 | 8 % |
| `_tul_group_losses` (chunked CE + MUX head) | 35.8 | 10 % |
| untracked (TG mask build, scatter, input norm, prefix path) | 24.7 | 7 % |
| forward | 357.5 | |
| backward (with the slot loop's checkpoint recompute) | 664.4 | 1.86× forward |
| optimizer (AdEMAMix) | 21.4 | |

`torch.profiler` over 3 eager steps: ~59,000 kernel launches per step (`aten::copy_`
85k calls, `aten::mul` 65k, `aten::bmm` 16k, `aten::mm` 9.6k), self-CUDA ~604 ms of a
1050 ms step: on the EAGER path ~45 % of the step is launch gap.

## Ranked, all at micro 12 × accum 2 (effective 24)

| # | change | override | step ms | tok/s | peak GB | speedup |
|---|---|---|---|---|---|---|
| — | shipped `tul_oly_mask` (eager) | | 2143 | 5,735 | 21.81 | 1.00× |
| 1 | scoped fused kernels | `model.tg_scoped_kernels=true` | 1317 | 9,332 | 16.39 | 1.63×, −5.4 GB |
| 2 | 1 + checkpoint 8 grad iterations | `model.ckpt_grad_iters=8` | 1136 | 10,821 | 20.61 | 1.89× |
| 3 | 1 + checkpoint 4 | `model.ckpt_grad_iters=4` | **1055** | **11,642** | **22.72** | **2.03×** |
| 4 | 1 + checkpoint none | `model.ckpt_grad_iters=0` | 1022 | 12,018 | 24.81 | 2.09× |

Lever 1 first: on the eager path `ckpt_grad_iters=8` OOMs in the compile warmup and `=4`
at step 0. Levers 2–4 are exact: `loss/total` at steps 0 and 1 is bit-identical across
−1/8/4/0 (`22.283517837524414`, `22.328134536743164`).

Why lever 1 is big: `tg_restrict` refuses `use_kernels true`, and `use_kernels false` calls
`set_force_eager(True)` process-wide (CCA prologue and conv, the HC-Cayley residual, the
core window all go eager). `tg_scoped_kernels` keeps only the TG-restricted branches eager
(`tests/test_tg_scoped_kernels.py`; `tul_a2s.yaml` and every earlier TG arm ran with it).
The TG attention itself is 1.2 % of the step (both branches, 8 layers, 2 micro-batches:
15.8 ms of 1320); `flex_attention` is 2.9× faster on fwd+bwd but its per-row
`create_block_mask` costs 21 ms per call, more than it saves. Do not touch the TG attention.

Null and negative levers (vs the 1359 ms baseline): `training.compile=false` −3.5 % (inside
1.5 sd; compile only wraps the MLPs and `eager_on_recompile` sends them eager);
`compile_attention=true` +8.5 %; `compile_blocks=true` +7.8 % (base.yaml's "1.84x" was
another geometry); `grad_probe_every=0` −3.7 % (the probe is 2.4 ms per step; keep it, it
feeds the tripwire); `loop_cot_probe=false` −1.4 %; `core_fixed_point_lambda=0` 0 % and
−0.33 GB (keep it). Price of the gain hinge, reported not proposed: `slot_gain_lambda=0`
is −7.9 % and −1.08 GB (it already runs at one iteration per step).

## Batch

Micro 24 × 1 OOMs on every arm (mask 27.0 GB at step 45 in `fused_linear_cross_entropy`;
micro 20 with ckpt 8 27.5 GB). Holding effective 24, micro 12 × 2 with levers 1+3 is the
frontier: 11.6k tok/s at 22.7 GB peak (~23.6 GB process), 7.8 GB left for the desktop.
Changing the effective batch buys nothing: micro 16 × 1 with ckpt 8 is 12.4k tok/s at
24.6 GB, micro 20 × 1 12.4k at 23.1 GB.

## The four arms with `ckpt_grad_iters=4` (exact on each: step-0/1 loss bit-identical)

| arm | as configured | with ckpt 4 (+scoped on mask) |
|---|---|---|
| `tul_oly_mask` | 2143 ms / 21.81 GB | 1055 ms / 22.72 GB |
| `tul_oly_mnext` | 1408 ms / 16.00 GB | 1074 ms / 22.13 GB |
| `tul_oly_a1` | 1370 ms / 15.71 GB | 1058 ms / 21.84 GB |
| `notul_oly` | 1174 ms / 11.07 GB | 1081 ms / 20.35 GB |

With scoped kernels the MASKED arm is slightly faster than the unmasked `mnext`: the TG
slot-gather attention is cheaper than the pooled compressor + LightningIndexer + top-k it
replaces. The mask was never the cost.

## Findings beyond throughput

- **The gain hinge reads noise on the eager path.** `scripts/gain_eps.py`: at
  `slot_gain_eps 0.02` the eager finite difference reads 0.94 ± 0.014 on a map whose gain is
  0.870 (both paths converge on 0.870 as eps grows; fused reads 0.87 at every eps). With
  `tg_scoped_kernels` on, the hinge at target 0.9 stops firing. Note:
  `.agents/notes/proposed/bug-fix/2026-09-08-slot-gain-eps-noise-bias.md`.
- **`tg_scoped_kernels` is not bit-identical**: `mux_ce` 11.184428 → 11.184101 on one
  batch, CE grad median 0.20 % relative, `preclip/total` 46,192 vs 46,362; a few core HC
  projection weights differ up to 65 % relative on grads of norm ~1 against a global 4.6e4
  (cancellation-amplified rounding). `scripts/numerics_ab.py`.
- **`perf/flop_proxy` was wrong under grad accumulation**: `train.py` divided the last
  micro-batch's `n_tokens` by the EFFECTIVE batch (16.36 at micro 12 × 2 vs 12.15 at 16 × 1
  for the same model). Fixed in the E16 change; the E16 smoke reads 12.14 at micro 12 × 2.
- **Eval costs 28.5 s** (three timed runs at eval_every 250/20/12 solve to 28.5 s in-loop,
  26 s final). At `eval_every 250` over 6k steps that is 11 min per arm. It was an
  OpenWebText number; from E16 the val stream is the Olympiad held-out shard
  (`curriculum.val_source`), so the 11 min buy a math curve.
- **The carrier is fp32** (`_tul_front` returns fp32 under the RMSNorm autocast promotion,
  `morph/model/CLAUDE.md`); a bf16 carrier was measured on the 2026-08 geometry (−1.42 GB
  allocated, resident unchanged) and not repeated.

## Addendum (2026-09-08, main session): prefetch and the "data" region

The audit left `data_runtime.prefetch_batches` unmeasured (the data region is 43 ms of a
1055 ms step at accum 2). Measured on the recommended geometry (levers 1+3), 60 steps each:

| prefetch | step ms | data wall ms | data GPU ms |
|---|---|---|---|
| 4 | 1065 | 43.0 | 0.3 |
| 8 | 1101 | 42.5 | 0.3 |
| 16 | 1083 | 43.8 | 0.3 |

Depth does nothing, and the packer is not slow: `MultiSourceCurriculumLoader` yields a
12-row TUL batch in 6.9 ms synchronously on the main thread (plain 4.9 ms; profile: 60 %
`doc_tokens` numpy slicing, 35 % `pack_tul_batch`). The 43 ms is the host blocking on the
GPU queue at the pageable host-to-device copy, and the per-region GPU times (bwd 554 + fwd
479 + opt 23 + clip 7) sum to the step wall: **the step is GPU-time-bound under the
recommended levers**. A pinned non-blocking copy would move that wait, not remove it.
Prefetch stays at 4.

## Not measured

Anything past 60 steps (the Poisson draw and slot counts drift; watch
`perf/peak_mem_alloc_mib` for the first 500 steps under ckpt 4); `tul_oly_a1` at any
micro-batch but 12; training under a hinge that no longer fires (E16 P16i is the first
reading); `compile=false` as a real 3.5 % win.
