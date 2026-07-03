"""Env-gated CUDA-event region timing for the training loop (perf pass Phase 1).

MORPH_PERF_REGIONS=1 → RegionTimer records paired CUDA events + wall clocks around
the train-step regions (data/fwd/aux/bwd/prune/clip/opt) and prints per-region means
every `report_every` steps. Default OFF → every call is a no-op returning a shared
nullcontext (zero allocation, zero kernels, bit-identical training).

Attribution model: CUDA events are stream-timestamped, so per-region GPU time is the
stream time of work ENQUEUED inside the region. The wall column is the CPU-side time
the loop spent inside the region (shows CPU-blocking costs like synchronous data
loading that never touch the stream). The per-step `found_inf` sync in
GradScaler.step drains the queue each step, so step boundaries are clean.

Aggregation is deferred: events accumulate per step and are drained with ONE
torch.cuda.synchronize at each report boundary — the timer itself adds no per-step
syncs beyond what the loop already has.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext

import torch

_NULL = nullcontext()


class RegionTimer:
    def __init__(self, enabled: bool, report_every: int = 20):
        self.enabled = bool(enabled) and torch.cuda.is_available()
        self.report_every = int(report_every)
        self._pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self._wall = defaultdict(float)
        self._n = 0

    def region(self, name: str):
        if not self.enabled:
            return _NULL
        return self._region(name)

    @contextmanager
    def _region(self, name: str):
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record()
        w0 = time.perf_counter()
        try:
            yield
        finally:
            self._wall[name] += time.perf_counter() - w0
            ev1.record()
            self._pending.append((name, ev0, ev1))

    def step_end(self, step: int, step_wall_s: float | None = None) -> None:
        if not self.enabled:
            return
        self._n += 1
        self._wall["__step__"] += float(step_wall_s or 0.0)
        if self._n % self.report_every:
            return
        torch.cuda.synchronize()
        gpu = defaultdict(float)
        for name, ev0, ev1 in self._pending:
            gpu[name] += ev0.elapsed_time(ev1)  # ms
        self._pending.clear()
        n = float(self.report_every)
        names = sorted(set(gpu) | (set(self._wall) - {"__step__"}))
        parts = [
            f"{nm} {self._wall[nm] * 1e3 / n:.1f}/{gpu[nm] / n:.1f}"
            for nm in names
        ]
        tracked_wall = sum(v for k, v in self._wall.items() if k != "__step__")
        step_ms = self._wall["__step__"] * 1e3 / n
        other = step_ms - tracked_wall * 1e3 / n
        print(
            f"[perf] step {step} mean over {self.report_every} (wall/gpu ms): "
            + "  ".join(parts)
            + f"  | untracked {other:.1f}  | step {step_ms:.1f}ms",
            flush=True,
        )
        self._wall.clear()
        self._n = 0
