# Coda K=4 cost benchmark

**NOT pre-registered — cost benchmark.** This is not a science experiment; it measures
step-time and memory to size the planned arm in
`lab/experiments/planned/2026-08-25-gradient-flow-soft-min-arm.md` (K=4 candidate
latents, coda + loss run K times per step, prelude + core run once).

## Question

At TUL A1 (batch 6, seq 1024, L_total 1152): what fraction `f` of a fwd+bwd step is
coda + loss, what is the arm/baseline step-time ratio at K=4, and does it fit in 32 GB?

## Setup

- Script: `lab/divergence/bench_coda_k.py` (this repo, branch `perf/throughput-lever-stack`).
- Model: `tul_a1` config, **randomly initialized** (no checkpoint — weight values do not
  change kernel cost), built exactly as `lab/divergence/_build.py` builds the probe
  models (quantization applied). The build prints `loop 4:6×6:4` — the config is
  4 prelude / 6 core / 4 coda blocks, not the 3:6:3 in older notes. 286.1 M params.
- Data: real validation batches from the probe data pipeline
  (`create_dataloader(..., tul=tul_rt.val_data_cfg)`), 8 fixed batches cycled.
- **Loop depth pinned to 8** (`--depth 8`, `_sample_slot_depths` monkeypatched) for ALL
  variants. Rationale: the core loop is a masked update over the full compact slot
  sequence, so its cost is `total_iters = max` over B×64 Poisson(6) draws clamped at 8 —
  which is 8 with probability ≈ 1 at ≥ 384 slots. Pinning 8 reproduces the realized
  training iteration count, deterministically.
- Timing: CUDA events around fwd (bf16 autocast) + `loss.backward()`,
  `torch.cuda.synchronize()` before/after; warmup ≥ 10, then 30 timed iters per variant.
  **No optimizer step is included** (identical for arm and baseline, so it cancels in
  the ratio; absolute step times in a real run are slightly larger).
- Region decomposition: identity-ablation deltas, the exact `region_shapley.py`
  contextmanager (patched `block.forward` / `_apply_core_step`). Each delta therefore
  includes that region's backward cost.
- `loss_only` = `_tul_group_losses` (head + chunked fused CE) fwd+bwd on a random bf16
  carrier of the real `[B, 1152, 768]` shape — CE cost is content-independent.
- Kernels regime mirrors `morph/training/train.py`: `use_kernels=true`,
  `torch.compile(mode="default")` on the MLP submodules (core group dynamic-batch),
  `torch._functorch.config.donated_buffer=False`. Compile succeeded; no eager fallback
  was needed.
- Derived numbers below use **medians** (a handful of ~2x outlier iterations — one
  674 ms recompile at kernels-b6, periodic ~575 ms iterations in eager — inflate the
  means; all per-iteration times are in `results.json`).

## Commands (all exit 0 unless noted)

```
PYTHONPATH=/home/wolfe/morph-perf /home/wolfe/11-DiffusionBlocks-Testing/.venv/bin/python \
  lab/divergence/bench_coda_k.py --regime eager   --batch 6              --out ... --label eager_b6
  lab/divergence/bench_coda_k.py --regime kernels --batch 6  --warmup 15 --out ... --label kernels_b6
  lab/divergence/bench_coda_k.py --regime kernels --batch 24 --warmup 15 --variants full   # exit 42 OOM
  ... --batch 20   # exit 42 OOM     ... --batch 18   # exit 42 OOM
  ... --batch 16   # exit 42 OOM (fragmentation)      ... --batch 14   # exit 42 OOM (fragmentation)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ...
  ... --batch 24   # exit 42 OOM (true: 24.49 GiB allocated at death)
  ... --batch 18   # exit 42 OOM     ... --batch 17   # exit 42 OOM
  ... --batch 16 --label kernels_b16_expseg   # exit 0, 571.4 ms median, 24.30 GB peak
```

## Region breakdown (fwd+bwd, batch 6, median ms)

| Region                     | eager (kernels off) | kernels + compile |
|----------------------------|--------------------:|------------------:|
| prelude (4 blocks)         |               95.0  |             46.4  |
| core loop (6 blocks × 8)   |              228.0  |            132.1  |
| coda (4 blocks)            |               94.3  |             46.2  |
| loss (head + chunked CE)   |               29.2  |             29.3  |
| front (embed + TUL glue)   |               25.8  |             25.8  |
| **full step**              |          **472.4**  |        **279.9**  |
| `f` = (coda+loss)/full     |              0.262  |         **0.270** |
| `f_upper` (incl. front)    |              0.316  |             0.362 |

## K=4 estimate vs measured bound (kernels regime, batch 6, 6144 data tokens/step)

| Quantity                                  |    step ms | tokens/s |
|-------------------------------------------|-----------:|---------:|
| Baseline (measured, median)               |      279.9 |   21,954 |
| Arm est. `base × (1 + 3f)`, f = 0.270     |      506.4 |   12,134 |
| Arm est. with `f_upper` = 0.362           |      583.8 |   10,523 |
| Batch-16 full-model proxy (measured)      |      571.4 |    (n/a) |
| Batch-24 bound, linear extrap 571.4×24/16 |      857.1 |    7,168 |

Estimated **arm/baseline ratio at K=4: ~1.81** (2.09 with the conservative `f_upper`).
The batch-24 full-model proxy — which would run prelude + core 4x too — **does not
fit**: it OOMs (24.5 GiB allocated at death against ~25.6 GiB usable beside the ~5.8 GiB
desktop). Largest full-model batch that fits is 16 (571.4 ms, 24.30 GB peak alloc,
under `expandable_segments`), so the 857 ms row is a linear extrapolation, not a
measurement. The real arm sits well below that bound because only the coda quadruples.

## Peak memory (torch.cuda.max_memory_allocated)

| Config                              | peak alloc |
|-------------------------------------|-----------:|
| kernels, batch 6 (baseline)         |   10.75 GB |
| kernels, batch 16 (largest fitting) |   24.30 GB |
| kernels, batch 17 / 18 / 24         | OOM (24.8 / 24.8 / 24.5 GiB allocated at death) |

Rough arm-memory estimate at batch 6: the coda's activation contribution is
≈ peak(core+prelude off) − peak(all off) = 7.67 − 4.99 ≈ 2.7 GB, so K=4 adds
≈ 3 × 2.7 ≈ 8 GB → ~19 GB peak. **Estimate only, not measured** — the arm code does
not exist yet. It suggests batch 6 fits with margin and batch drop to 4 is likely
unnecessary, but the planned fallback (6 → 4) stays sensible.

## Reading

The coda + loss is 27 % of a compiled fwd+bwd step (46 ms coda blocks + 29 ms head/CE
of a 280 ms step), so folding K=4 candidates into the coda batch costs an estimated
1.81x step time — 280 → ~506 ms, 21.9k → ~12.1k tokens/s — with a hard measured
ceiling well above the estimate only because the ceiling run (full model at 4x batch)
does not even fit on the card. Memory is not the constraint for the arm itself: the
baseline peaks at 10.75 GB and the coda-only quadrupling adds roughly 8 GB, far from
the ~25.5 GiB working limit, though the real trainer also carries optimizer state
(~3+ GB of AdEMAMix slots) that this fwd+bwd-only benchmark excludes — batch-16
fitting HERE does not mean batch 16 fits in the trainer (A1 at 16 is a measured
trainer OOM per `tul_short.yaml`). The eager and compiled `f` agree (0.262 vs 0.270),
so the estimate is not an artifact of the compile regime.

## Not verified

- The arm itself (does not exist). `f` assumes the K-fold coda costs exactly K× the
  single coda; batching efficiency could make it cheaper per candidate, and soft-min
  loss overhead (K CE calls + logsumexp) is not in the estimate beyond the 29 ms × K
  CE term.
- Optimizer step time and optimizer-state memory (excluded; identical across arms).
- Default-allocator fragmentation (batch 14/16 OOM'd on 3.5–4 GiB contiguous requests
  in THIS harness while the real trainer runs batch 14; large-batch numbers here use
  `expandable_segments:True` — a deviation from the trainer's allocator config).
- The `exit=` capture for the first two runs went through a pipe under fish and
  reported the pipe tail; their success is evidenced by their result blocks and JSON
  writes, not a captured exit code. All OOM/bisect runs used direct capture (42/0 as
  listed).
