"""Hardware-adaptive placement for pre-tokenized training shards.

Design: docs/data-placement-design.md (Phase 1: probe + budget + tiers A/B + prefetch).
Motivating incident (2026-07-02): a 21 GB shard np.memmap'd from a spinning HDD starved
the GPU — the main thread demand-paged 4 KB per fault at ~12 ms seek (majflt, invisible
to read_bytes) and the run decayed 5.3 → 2.2 sps as the shuffled doc order went cold.
The policy here is: measure the actual storage, decide a tier, and REPORT the decision —
never assume, never silently degrade.

Tiers (per source shard):
  A "ram"  — np.fromfile preload (one sequential read; RAM-speed forever). Chosen when
             storage is slow (rand-4K p50 > ~1 ms) and the shard fits the RAM budget.
  B "mmap" — mmap + MADV_RANDOM (kernel readahead off for random draws). Chosen when
             storage is fast (p50 ≤ ~200 µs), or as the fallback when preload can't fit.
  C "stream" — shuffle-window streaming for bigger-than-RAM corpora on slow storage.
             Phase 2; requesting it raises.
  D Prefetcher — orthogonal: ONE producer thread filling a bounded queue of assembled
             batches. Same RNG, same order → bit-identical to synchronous consumption
             while a single generator is live (see Prefetcher docstring for the switch
             caveat). `prefetch_batches: 0` reproduces synchronous behavior exactly.

Config surface (Hydra `data_runtime:` section; env vars override for non-Hydra contexts):
  placement: auto|ram|mmap|stream   MORPH_DATA_PLACEMENT
  ram_budget_frac: 0.5              MORPH_DATA_RAM_FRAC
  ram_reserve_gb: 16                MORPH_DATA_RAM_RESERVE_GB
  probe: true                       MORPH_DATA_PROBE (0/1)
  prefetch_batches: 4               MORPH_DATA_PREFETCH
  shuffle_window_docs: 100000       (tier C, phase 2)

SYNC NOTE: this module is duplicated in Olympiad-AI at
`src/olympiad_data/datasets/data_placement.py` (both repos publish independently and
must stay self-contained). Keep the two copies identical below this docstring.
"""
from __future__ import annotations

import dataclasses
import mmap
import os
import queue
import threading
import time
from typing import Generator, Iterator, Optional

import numpy as np

__all__ = [
    "DataRuntimeConfig", "ProbeResult", "TokenStore", "Prefetcher",
    "probe_storage", "detect_available_bytes", "compute_budget_bytes", "decide_tier",
]

GB = float(1 << 30)

# Tier thresholds (rand-4K p50). HDD ≈ 5–15 ms, SATA SSD ≈ 100–300 µs, NVMe ≈ 20–100 µs.
SLOW_P50_US = 1000.0   # above → HDD-class: demand paging will starve the GPU
FAST_P50_US = 200.0    # below → NVMe/SSD-class: mmap random access is fine


# ── config ──────────────────────────────────────────────────────────────────
@dataclasses.dataclass(frozen=True)
class DataRuntimeConfig:
    placement: str = "auto"            # auto | ram | mmap | stream
    ram_budget_frac: float = 0.5
    ram_reserve_gb: float = 16.0
    probe: bool = True
    prefetch_batches: int = 4          # 0 = synchronous (bit-exact repro escape hatch)
    shuffle_window_docs: int = 100_000  # tier C (phase 2, unused in phase 1)

    @staticmethod
    def resolve(section=None) -> "DataRuntimeConfig":
        """Build from an optional Hydra/dict `data_runtime` section; env vars win."""
        def _get(key, default):
            if section is None:
                return default
            if isinstance(section, dict):
                v = section.get(key, default)
            else:
                v = getattr(section, key, default)
            return default if v is None else v

        env = os.environ.get
        placement = str(env("MORPH_DATA_PLACEMENT") or _get("placement", "auto")).lower()
        if placement not in ("auto", "ram", "mmap", "stream"):
            raise ValueError(f"data_runtime.placement must be auto|ram|mmap|stream, got {placement!r}")
        probe_env = env("MORPH_DATA_PROBE")
        return DataRuntimeConfig(
            placement=placement,
            ram_budget_frac=float(env("MORPH_DATA_RAM_FRAC") or _get("ram_budget_frac", 0.5)),
            ram_reserve_gb=float(env("MORPH_DATA_RAM_RESERVE_GB") or _get("ram_reserve_gb", 16.0)),
            probe=(probe_env not in ("0", "false", "False")) if probe_env is not None
                  else bool(_get("probe", True)),
            prefetch_batches=int(env("MORPH_DATA_PREFETCH") or _get("prefetch_batches", 4)),
            shuffle_window_docs=int(_get("shuffle_window_docs", 100_000)),
        )


# ── detection ───────────────────────────────────────────────────────────────
def detect_available_bytes(meminfo: str = "/proc/meminfo",
                           proc_cgroup: str = "/proc/self/cgroup",
                           cgroup_root: str = "/sys/fs/cgroup") -> int:
    """MemAvailable clamped by cgroup limits (v2 memory.max / v1 limit_in_bytes)."""
    avail = None
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) * 1024
                    break
    except OSError:
        pass
    if avail is None:                      # non-Linux fallback: sysconf
        try:
            avail = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            avail = 8 << 30                # last resort: assume a small machine
    cg = _cgroup_available_bytes(proc_cgroup, cgroup_root)
    return min(avail, cg) if cg is not None else avail


def _cgroup_available_bytes(proc_cgroup: str, cgroup_root: str) -> Optional[int]:
    """cgroup memory headroom (limit − current), or None when unlimited/undetectable."""
    def _read_int(path):
        try:
            with open(path) as f:
                s = f.read().strip()
            return None if s == "max" else int(s)
        except (OSError, ValueError):
            return None

    # v2: /proc/self/cgroup has a "0::/path" line; limit at <root>/<path>/memory.max
    try:
        with open(proc_cgroup) as f:
            for line in f:
                parts = line.strip().split(":", 2)
                if len(parts) == 3 and parts[0] == "0" and parts[1] == "":
                    d = os.path.join(cgroup_root, parts[2].lstrip("/"))
                    lim = _read_int(os.path.join(d, "memory.max"))
                    if lim is not None:
                        cur = _read_int(os.path.join(d, "memory.current")) or 0
                        return max(0, lim - cur)
    except OSError:
        pass
    # v1
    lim = _read_int(os.path.join(cgroup_root, "memory", "memory.limit_in_bytes"))
    if lim is not None and lim < (1 << 50):          # v1 "unlimited" is ~2^63
        cur = _read_int(os.path.join(cgroup_root, "memory", "memory.usage_in_bytes")) or 0
        return max(0, lim - cur)
    return None


def compute_budget_bytes(available_bytes: int, frac: float, reserve_gb: float) -> int:
    """budget = min(frac × available, available − reserve). Never negative."""
    return max(0, int(min(frac * available_bytes, available_bytes - reserve_gb * GB)))


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    rand_p50_us: float          # median random-4K read latency over the whole file
    seq_mbps: float             # one 64 MB sequential read (prices the preload)
    rotational: Optional[bool]  # /sys corroboration only — NOT decision-grade (None on ZFS/NFS)
    device: str


def probe_storage(path: str, n_reads: int = 200, read_bytes: int = 4096,
                  seq_bytes: int = 64 << 20, seed: int = 0) -> ProbeResult:
    """Microbench the ACTUAL file — don't trust filesystem flags.

    Random offsets are spread across the WHOLE file so a partially-warm page cache
    can't fake a fast device (the warm region is a small fraction; p50 still exposes
    cold latency). Costs < 3 s on HDD, < 50 ms on NVMe.
    """
    size = os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        rng = np.random.default_rng(seed)
        n = int(min(n_reads, max(1, size // read_bytes)))
        offs = (rng.integers(0, max(1, size - read_bytes + 1), size=n)
                // read_bytes) * read_bytes
        lat_ns = np.empty(n, dtype=np.float64)
        for i in range(n):
            t0 = time.perf_counter_ns()
            os.pread(fd, read_bytes, int(offs[i]))
            lat_ns[i] = time.perf_counter_ns() - t0
        p50_us = float(np.median(lat_ns)) / 1e3
        seq = int(min(seq_bytes, size))
        chunk = 4 << 20
        t0 = time.perf_counter()
        got = 0
        while got < seq:
            b = os.pread(fd, min(chunk, seq - got), got)
            if not b:
                break
            got += len(b)
        seq_mbps = (got / (1 << 20)) / max(time.perf_counter() - t0, 1e-9)
    finally:
        os.close(fd)
    rotational, device = _blockdev_info(path)
    return ProbeResult(rand_p50_us=p50_us, seq_mbps=seq_mbps,
                       rotational=rotational, device=device)


def _blockdev_info(path: str):
    """(rotational, device_name) from /sys via st_dev. Best-effort: ZFS/NFS/containers
    expose synthetic device numbers with no /sys/dev/block entry → (None, 'maj:min')."""
    try:
        st = os.stat(path)
        maj, minr = os.major(st.st_dev), os.minor(st.st_dev)
    except OSError:
        return None, "unknown"
    base = f"/sys/dev/block/{maj}:{minr}"
    device = f"{maj}:{minr}"
    try:
        with open(os.path.join(base, "uevent")) as f:
            for line in f:
                if line.startswith("DEVNAME="):
                    device = "/dev/" + line.strip().split("=", 1)[1]
                    break
    except OSError:
        pass
    for cand in (os.path.join(base, "queue", "rotational"),          # whole disk
                 os.path.join(base, "..", "queue", "rotational")):   # partition → parent
        try:
            with open(cand) as f:
                return f.read().strip() == "1", device
        except OSError:
            continue
    return None, device


# ── policy (pure — unit-testable with faked probes) ─────────────────────────
def decide_tier(size_bytes: int, budget_bytes: int, p50_us: Optional[float],
                placement: str = "auto"):
    """→ (tier, reason); tier ∈ {'ram','mmap'}. p50_us=None means the probe was skipped."""
    if placement == "stream":
        raise NotImplementedError(
            "tier C (shuffle-window streaming) is phase 2 of docs/data-placement-design.md "
            "— use placement: auto|ram|mmap")
    fits = size_bytes <= budget_bytes
    if placement == "ram":
        if fits:
            return "ram", "forced by config"
        return "mmap", (f"placement=ram REFUSED: shard {size_bytes / GB:.1f}GB > RAM budget "
                        f"{budget_bytes / GB:.1f}GB — falling back to mmap (never-OOM)")
    if placement == "mmap":
        return "mmap", "forced by config"
    if placement != "auto":
        raise ValueError(f"unknown placement {placement!r}")
    if p50_us is None:                       # probe disabled → the never-starve default
        return ("ram", "probe disabled → preload since it fits") if fits \
            else ("mmap", "probe disabled, exceeds RAM budget → mmap")
    if p50_us <= FAST_P50_US:
        return "mmap", f"fast storage (rand-4K p50 {p50_us:.0f}µs ≤ {FAST_P50_US:.0f}µs)"
    if fits:                                 # slow or gray-zone storage, fits → preload
        return "ram", (f"slow storage (rand-4K p50 {p50_us / 1e3:.2f}ms) and shard fits "
                       f"RAM budget → sequential preload beats demand paging")
    return "mmap", (f"WARNING: slow storage (rand-4K p50 {p50_us / 1e3:.2f}ms) and shard "
                    f"{size_bytes / GB:.1f}GB exceeds RAM budget {budget_bytes / GB:.1f}GB — "
                    f"mmap WILL be seek-bound; tier C streaming lands in phase 2")


# ── TokenStore (tiers A/B) ──────────────────────────────────────────────────
class TokenStore:
    """Placement-aware read-only token blob. `.array` is a flat np.ndarray view backed by
    a RAM preload (tier A) or mmap+MADV_RANDOM (tier B); slicing works identically.

    One report line is printed per store at init — no silent behavior (design §5).
    `budget_bytes` is injectable for tests; by default it is measured at call time, so a
    second store's preload accounts for RAM the first one already consumed.
    """

    def __init__(self, path: str, dtype=np.uint16, name: Optional[str] = None,
                 runtime: Optional[DataRuntimeConfig] = None,
                 budget_bytes: Optional[int] = None):
        self.path = path
        self.dtype = np.dtype(dtype)
        self.name = name or os.path.basename(os.path.dirname(os.path.abspath(path)))
        rt = runtime or DataRuntimeConfig.resolve()
        size = os.path.getsize(path)

        probe = None
        if rt.probe and rt.placement == "auto":
            probe = probe_storage(path)
        if budget_bytes is None:
            budget_bytes = compute_budget_bytes(
                detect_available_bytes(), rt.ram_budget_frac, rt.ram_reserve_gb)
        self.probe = probe
        self.budget_bytes = budget_bytes
        p50 = probe.rand_p50_us if probe is not None else None
        self.tier, reason = decide_tier(size, budget_bytes, p50, rt.placement)

        if probe is not None:
            klass = ("HDD-class" if p50 > SLOW_P50_US
                     else "NVMe/SSD-class" if p50 <= FAST_P50_US else "SATA/gray-zone")
            rot = {True: ", rotational", False: "", None: ""}[probe.rotational]
            eta = size / (1 << 20) / max(probe.seq_mbps, 1e-9)
            probe_txt = (f" on {probe.device}{rot}, rand-4K p50={_fmt_us(p50)} ({klass}), "
                         f"seq={probe.seq_mbps:.0f}MB/s")
            eta_txt = f" (~{_fmt_s(eta)} preload)" if self.tier == "ram" else ""
        else:
            probe_txt, eta_txt = " (probe skipped)", ""
        print(f"[data] {self.name}: {size / GB:.1f}GB{probe_txt}, "
              f"RAM budget {budget_bytes / GB:.0f}GB → tier "
              f"{'A PRELOAD' if self.tier == 'ram' else 'B MMAP+MADV_RANDOM'}{eta_txt} "
              f"[{reason}]", flush=True)

        self._f = self._mm = None
        if self.tier == "ram":
            t0 = time.perf_counter()
            self.array = np.fromfile(path, dtype=self.dtype)
            print(f"[data] {self.name}: preloaded {size / GB:.1f}GB in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        else:
            self._f = open(path, "rb")
            self._mm = mmap.mmap(self._f.fileno(), 0, prot=mmap.PROT_READ)
            if hasattr(self._mm, "madvise"):           # Linux, py3.8+
                self._mm.madvise(mmap.MADV_RANDOM)
            self.array = np.frombuffer(self._mm, dtype=self.dtype)

    def close(self):
        self.array = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._f is not None:
            self._f.close()
            self._f = None


def _fmt_us(us: float) -> str:
    return f"{us / 1e3:.1f}ms" if us >= 1e3 else f"{us:.0f}µs"


def _fmt_s(s: float) -> str:
    return f"{s / 60:.1f}min" if s >= 60 else f"{s:.1f}s"


# ── Prefetcher (tier D) ─────────────────────────────────────────────────────
class Prefetcher:
    """ONE producer thread running the wrapped generator into a bounded queue.

    Determinism: a single producer runs the SAME RNG sequence the consumer would, and the
    queue is FIFO — the batch stream is bit-identical to synchronous iteration for the
    lifetime of the generator. `close()` stops the producer between batches (never
    mid-draw, so the underlying loader state stays consistent) and reports how many
    in-flight batches were discarded. NOTE: at a loader rebuild (TST phase switch,
    curriculum stage step-up) those discarded batches mean the post-switch stream sits at
    a slightly different RNG offset than a prefetch=0 run — same distribution, different
    draw. Bit-exact repro across switches ⇒ `prefetch_batches: 0`.

    Producer exceptions are re-raised in the consumer on the next `next()`.
    """
    _SENTINEL = object()

    def __init__(self, gen: Iterator, depth: int, name: str = ""):
        if depth < 1:
            raise ValueError(f"prefetch depth must be >= 1, got {depth}")
        self._q: queue.Queue = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._exc: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._produce, args=(gen,), daemon=True,
            name=f"data-prefetch{('-' + name) if name else ''}")
        self._thread.start()

    def _produce(self, gen: Iterator):
        try:
            for item in gen:
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if self._stop.is_set():
                    return
        except BaseException as e:                    # surfaced to the consumer
            self._exc = e
        while not self._stop.is_set():                # signal exhaustion/failure
            try:
                self._q.put(self._SENTINEL, timeout=0.1)
                return
            except queue.Full:
                continue

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise RuntimeError("next() on a closed Prefetcher")
        item = self._q.get()
        if item is self._SENTINEL:
            self._closed = True
            if self._exc is not None:
                raise self._exc
            raise StopIteration
        return item

    def close(self) -> int:
        """Stop the producer, drain the queue; returns the number of discarded batches."""
        if self._closed:
            return 0
        self._closed = True
        self._stop.set()
        discarded = 0
        while True:
            try:
                if self._q.get_nowait() is not self._SENTINEL:
                    discarded += 1
            except queue.Empty:
                if not self._thread.is_alive():
                    break
                time.sleep(0.005)
        self._thread.join()
        return discarded
