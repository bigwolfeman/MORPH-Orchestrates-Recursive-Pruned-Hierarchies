"""Gates for docs/data-placement-design.md phase 1 (§7).

- policy unit tests with faked probes (HDD/NVMe × fits/doesn't-fit × forced placements)
- never-OOM: preload refused when the shard exceeds the RAM budget
- prefetch determinism: prefetch on/off produce an identical batch stream (same seed)
- TokenStore tier A and tier B expose byte-identical arrays
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from morph.training.data_placement import (
    DataRuntimeConfig,
    Prefetcher,
    TokenStore,
    compute_budget_bytes,
    decide_tier,
    detect_available_bytes,
    probe_storage,
)

GB = 1 << 30


# ── policy (pure function, faked probes) ────────────────────────────────────
@pytest.mark.parametrize(
    "size,budget,p50,placement,want",
    [
        (21 * GB, 54 * GB, 11_200.0, "auto", "ram"),    # the 2026-07-02 incident: HDD, fits → preload
        (200 * GB, 54 * GB, 11_200.0, "auto", "mmap"),  # HDD, too big → mmap (+ warning; tier C phase 2)
        (21 * GB, 54 * GB, 60.0, "auto", "mmap"),       # NVMe → mmap is fine
        (21 * GB, 54 * GB, 500.0, "auto", "ram"),       # gray zone (SATA), fits → prefer preload
        (200 * GB, 54 * GB, 500.0, "auto", "mmap"),     # gray zone, too big → mmap
        (21 * GB, 54 * GB, None, "auto", "ram"),        # probe disabled, fits → never-starve default
        (200 * GB, 54 * GB, None, "auto", "mmap"),      # probe disabled, too big → mmap
        (21 * GB, 54 * GB, 11_200.0, "mmap", "mmap"),   # forced mmap wins over slow storage
        (21 * GB, 54 * GB, 60.0, "ram", "ram"),         # forced ram wins over fast storage
    ],
)
def test_decide_tier(size, budget, p50, placement, want):
    tier, reason = decide_tier(size, budget, p50, placement)
    assert tier == want, reason


def test_never_oom_forced_ram_refused():
    tier, reason = decide_tier(200 * GB, 54 * GB, None, "ram")
    assert tier == "mmap"
    assert "REFUSED" in reason


def test_stream_is_phase2():
    with pytest.raises(NotImplementedError):
        decide_tier(GB, GB, None, "stream")


def test_unknown_placement_rejected():
    with pytest.raises(ValueError):
        decide_tier(GB, GB, None, "florp")


def test_compute_budget():
    # frac binds when available is large; reserve binds when it's small; never negative.
    assert compute_budget_bytes(100 * GB, 0.5, 16.0) == 50 * GB
    assert compute_budget_bytes(20 * GB, 0.9, 16.0) == 4 * GB
    assert compute_budget_bytes(8 * GB, 0.9, 16.0) == 0


def test_detect_available_cgroup_clamp(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 131072000 kB\nMemAvailable: 65536000 kB\n")
    # cgroup v2: limit 8 GB, current 2 GB → headroom 6 GB clamps the 62.5 GB MemAvailable
    cgdir = tmp_path / "cg" / "mygroup"
    cgdir.mkdir(parents=True)
    (cgdir / "memory.max").write_text(str(8 * GB))
    (cgdir / "memory.current").write_text(str(2 * GB))
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/mygroup\n")
    got = detect_available_bytes(str(meminfo), str(proc_cgroup), str(tmp_path / "cg"))
    assert got == 6 * GB
    # "max" (unlimited) → MemAvailable passes through
    (cgdir / "memory.max").write_text("max")
    got = detect_available_bytes(str(meminfo), str(proc_cgroup), str(tmp_path / "cg"))
    assert got == 65_536_000 * 1024


# ── TokenStore tiers ────────────────────────────────────────────────────────
def _blob(tmp_path, n_tokens=50_000, seed=7):
    rng = np.random.default_rng(seed)
    toks = rng.integers(1, 60_000, size=n_tokens, dtype=np.uint16)
    p = tmp_path / "tokens.u16.bin"
    toks.tofile(p)
    return str(p), toks


def test_tokenstore_tiers_identical(tmp_path, capsys):
    path, ref = _blob(tmp_path)
    a = TokenStore(path, runtime=DataRuntimeConfig(placement="ram", probe=False))
    b = TokenStore(path, runtime=DataRuntimeConfig(placement="mmap", probe=False))
    assert a.tier == "ram" and b.tier == "mmap"
    assert np.array_equal(a.array, ref) and np.array_equal(b.array, ref)
    out = capsys.readouterr().out
    assert out.count("[data]") >= 2          # design §5: one report line per store
    b.close()


def test_tokenstore_never_oom(tmp_path, capsys):
    path, _ = _blob(tmp_path)
    # budget smaller than the blob → preload refused even when forced
    st = TokenStore(path, runtime=DataRuntimeConfig(placement="ram", probe=False),
                    budget_bytes=1024)
    assert st.tier == "mmap"
    assert "REFUSED" in capsys.readouterr().out
    st.close()


def test_probe_runs_on_real_file(tmp_path):
    path, _ = _blob(tmp_path, n_tokens=2_000_000)
    r = probe_storage(path, n_reads=50)
    assert r.rand_p50_us > 0 and r.seq_mbps > 0


# ── Prefetcher ──────────────────────────────────────────────────────────────
def test_prefetcher_order_and_exhaustion():
    pf = Prefetcher(iter(range(100)), depth=4)
    assert list(pf) == list(range(100))
    with pytest.raises(RuntimeError):
        next(pf)                              # closed after exhaustion


def test_prefetcher_close_midstream():
    pf = Prefetcher(iter(range(1_000_000)), depth=4)
    got = [next(pf) for _ in range(10)]
    assert got == list(range(10))
    pf.close()
    assert not pf._thread.is_alive()
    with pytest.raises(RuntimeError):
        next(pf)


def test_prefetcher_propagates_producer_exception():
    def gen():
        yield 1
        raise RuntimeError("boom in producer")
    pf = Prefetcher(gen(), depth=2)
    assert next(pf) == 1
    with pytest.raises(RuntimeError, match="boom in producer"):
        next(pf)


# ── loader integration: prefetch determinism gate ───────────────────────────
def _make_shard(root, name, n_docs, seed, role="pretrain_bulk"):
    d = root / name
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    lens = rng.integers(5, 90, size=n_docs).astype(np.int32)
    toks = rng.integers(1, 60_000, size=int(lens.sum()), dtype=np.uint16)
    offs = np.zeros(n_docs + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    toks[offs[1:] - 1] = 0                    # trailing EOS per doc
    toks.tofile(d / "tokens.u16.bin")
    np.save(d / "doc_offsets.i64.npy", offs)
    np.save(d / "doc_lens.i32.npy", lens)
    (d / "meta.json").write_text(json.dumps({"eos_id": 0, "role": role}))


def _loader(tmp_path, prefetch):
    from morph.training.curriculum_data import MultiSourceCurriculumLoader
    rt = DataRuntimeConfig(placement="mmap", probe=False, prefetch_batches=prefetch)
    return MultiSourceCurriculumLoader(
        str(tmp_path), {"alpha": 0.5, "beta": 0.5}, [64], seed=0, data_runtime=rt)


@pytest.fixture()
def shard_root(tmp_path):
    _make_shard(tmp_path, "alpha", 400, seed=1)
    _make_shard(tmp_path, "beta", 400, seed=2, role="reasoning_midtrain")
    return tmp_path


@pytest.mark.parametrize("bag", [0, 2])
def test_prefetch_determinism(shard_root, bag):
    """Design §3: prefetch on/off → bit-identical batch stream for the same seed."""
    sync = _loader(shard_root, prefetch=0).batches(4, bag_size=bag)
    pre_loader = _loader(shard_root, prefetch=4)
    pre = pre_loader.batches(4, bag_size=bag)
    for _ in range(40):
        xs, ys = next(sync)
        xp, yp = next(pre)
        assert torch.equal(xs, xp) and torch.equal(ys, yp)
    pre_loader._close_prefetch()


def test_batch_shapes_tst_bag(shard_root):
    loader = _loader(shard_root, prefetch=2)
    x, y = next(loader.batches(3, bag_size=2))
    assert x.shape == (3, 2 * 64) and y.shape == (3, 64, 2)
    loader._close_prefetch()


def test_loader_rebuild_closes_producer(shard_root, capsys):
    """batches()/set_stage must stop the old producer before state is reused."""
    loader = _loader(shard_root, prefetch=4)
    it1 = loader.batches(4, bag_size=2)
    next(it1)
    it2 = loader.batches(4, bag_size=0)       # e.g. the TST phase switch
    assert not it1._thread.is_alive()
    x, y = next(it2)
    assert x.shape == (4, 64) and y.shape == (4, 64)
    loader.set_stage(0)                       # also closes; then a fresh iterator works
    assert not it2._thread.is_alive()
    x, y = next(loader.batches(4))
    assert x.shape == (4, 64)
    loader._close_prefetch()
