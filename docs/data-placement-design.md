# Hardware-Adaptive Data Placement — Design Spec

Status: **approved for build** (2026-07-02, Wolfe + Fable). Not yet implemented.
Motivating incident: the seed→MORPH forgetting run starved the GPU because the 21 GB OWT
pretok shard was `np.memmap`'d from a spinning HDD — main thread D-blocked in
`folio_wait_bit` at ~68 major faults/s (~12 ms seek each ≈ 80% of wall stalled). Invisible
to `read_bytes` (mmap page-ins are majflt, not read()). Runs started fast on warm page
cache and decayed (5.3 → 2.2 sps by step 1600) as the shuffled doc order went cold.
Demand-paging 21 GB at 4 KB/fault ≈ 21 h — it never recovers. The operator fix was a
manual copy to NVMe; **open-source MORPH must instead detect the hardware it is on and
place data accordingly.**

## Philosophy

Measure, decide, **report** — never assume, never silently degrade.
One new module: `morph/training/data_placement.py`. Loaders consume it.

## 1. Detection (loader init, cheap, portable)

**RAM budget**
- `MemAvailable` from `/proc/meminfo`, clamped by cgroup limits (v2 `memory.max`,
  v1 `memory.limit_in_bytes`) so containers behave.
- `budget = min(ram_budget_frac × available, available − ram_reserve_gb)`.
- Defaults: `ram_budget_frac: 0.5`, `ram_reserve_gb: 16`.

**Storage speed — microbench the actual file, don't trust flags**
- ~200 random 4 KB reads at offsets spread across the WHOLE file; take p50 latency.
  Expected: HDD ≈ 5–15 ms, SATA SSD ≈ 100–300 µs, NVMe ≈ 20–100 µs, cache hit < 5 µs.
- Costs < 3 s on HDD, < 50 ms on NVMe. Works on any OS/fs/NFS.
- Spreading offsets over the full file defeats the partially-warm-cache illusion (the warm
  region is a small fraction; p50 still exposes cold latency).
- Also one 64 MB sequential read → sequential MB/s (prices the preload option).
- Linux corroboration only (not decision-grade): `/sys/.../queue/rotational` via the
  file's st_dev.

## 2. Placement tiers (chosen per source shard)

| Tier | When | Mechanism |
|------|------|-----------|
| **A: RAM preload** | fits budget AND storage slow (p50 > ~1 ms) | `np.fromfile` — one sequential read (21 GB @ HDD ≈ 2.5 min, RAM-speed forever). Sequential preload beats ~21 h of demand paging. |
| **B: mmap + MADV_RANDOM** | storage fast (p50 ≤ ~200 µs) | today's memmap + `madvise(MADV_RANDOM)` so the kernel stops useless readahead on random draws. |
| **C: shuffle-window streaming** | slow storage AND > RAM (the 100 B-token cloud corpus) | read sequentially in large blocks; shuffle within a K-doc RAM window. Converts random access → sequential + bounded shuffle (tf.data / HF-datasets practice). |
| **D: prefetch queue** | always available, orthogonal | ONE producer thread filling a bounded queue (depth ~4) of assembled batches — hides residual IO latency and the ~4 ms Python batch fill. |

Gray zone (200 µs < p50 ≤ 1 ms): prefer preload if it fits, else B.

## 3. Determinism guarantee

The prefetch thread is a SINGLE producer running the SAME RNG sequence — batch order is
bit-identical to synchronous mode, just produced ahead.
`placement: mmap, prefetch_batches: 0` reproduces current behavior exactly (escape hatch,
and the bit-repro arm for the parity gate).

## 4. Config surface (Hydra; env overrides for non-Hydra contexts)

```yaml
data_runtime:
  placement: auto          # auto | ram | mmap | stream
  ram_budget_frac: 0.5
  ram_reserve_gb: 16
  probe: true              # microbench at init
  prefetch_batches: 4      # 0 = synchronous (bit-exact repro)
  shuffle_window_docs: 100000   # tier C
```
Env: `MORPH_DATA_PLACEMENT`, `MORPH_DATA_RAM_FRAC`, `MORPH_DATA_PREFETCH`, …

## 5. No silent behavior

One log line per source at startup, e.g.:
```
[data] owt: 21.1GB on /dev/sdb2, rand-4K p50=11.2ms (HDD-class), seq=152MB/s,
       RAM budget 54GB → Tier A PRELOAD (~2.4 min)
```
A laptop user understands why startup paused; the HDD bug would have announced itself in
line one.

## 6. Integration points

- `TokenStore` class in `data_placement.py`: unified `.doc(i) -> np.ndarray` backed by RAM
  array / memmap(+madvise) / streaming window. Handles the probe + tier decision.
- `morph/training/curriculum_data.py` `_Source`: swap raw `np.memmap` for `TokenStore`
  (few lines).
- Olympiad-AI `src/olympiad_data/datasets/nlp_source.py`: same swap. The module is
  duplicated across repos with a sync-note header (both repos publish independently).
- Prefetcher wraps `MultiSourceCurriculumLoader.batches()` (or in train.py).
- Checkpoint placement is a SEPARATE small item: async save thread, or warn when
  `ckpt_dir` resolves to slow storage (a 2 GB torch.save on HDD stalls 13–20 s).
- Out of scope phase 1: the HF-arrow val stream in `morph/training/data.py`.

## 7. Testing gates

- Unit: policy function with faked probes (HDD/NVMe × fits/doesn't-fit × cgroup cap).
- Determinism: prefetch on/off → identical batch sequence for the same seed (hash the
  first N batches).
- Integration smoke: real shards, assert the placement report line + stable sps.
- Never-OOM: preload refused when shard > budget (assert Tier C/B fallback).

## 8. Phasing

- **Phase 1** (~1 day): probe + budget + tiers A/B + prefetch D + policy tests.
  Covers every failure mode hit on 2026-07-02.
- **Phase 2** (~1–2 days): tier C shuffle-window streaming. Required before the 100 B-token
  cloud corpus; not before.
- Open question for phase 2 (verify, don't assert): shuffle-window vs global-shuffle SNR
  A/B at 100 k-doc windows — expected negligible, must be measured.

## Related fixes landed the same day (context)

- `fused_cca_conv` grouped kernels: fp32-SIMT+register-spill → tensor-core bf16 dot
  (32× at CG=64; bit-exact parity). CG=64 applies to the d2048 cloud config too.
- `fused_ce`: odd vocab (49169) halved head GEMMs → `_pad_vocab` to 128-multiple,
  pad logits −inf (exact; parity-gated incl. MCE).
- `load_weights_only`: `._orig_mod.` canonicalized on BOTH sides (uncompiled seed →
  compiled model no longer drops tensors).
- TST phase switch now preserves the curriculum loader (was silently reverting to the
  base OWT stream in curriculum mode).
- `gla.py`: fused-GLA smem guard (7·DH² > device optin → eager fallback; DH=128 on 5090).
- Step/VAL prints `flush=True` (stdout block-buffering made healthy runs look dead).
