"""Bounded retention for periodic training checkpoints.

Why this module exists: `checkpoints/morph/` reached **292 GB** on 2026-08-25. The
rolling pre-onset ring in `train.py` had always rotated its files; the ordinary
`ckpt_every` path never did, so it grew without bound. A four-seed 3500-step sweep at
`ckpt_every=500` wrote 63 GB, and a 100k-step run at `ckpt_every=2500` writes 90 GB.

Both ring buffers now share this one class, so their behaviour cannot drift apart.
"""
from __future__ import annotations

import os
import re
from typing import Callable, Iterable


class RetentionRing:
    """Keep the newest `keep` paths; delete the rest as new ones are added.

    `keep <= 0` disables retention entirely: `add()` still records the path but never
    deletes. That is the explicit "keep everything" setting, not an accident.

    The ring only ever removes paths that were handed to it. Files written by the abort
    guards (`DIVERGED_*.pt`, `TAKEOVER_*.pt`) are never added and so are never rotated
    away — losing the state that captured a failure is the one unacceptable outcome here.
    """

    def __init__(self, keep: int, *, tag: str = "ckpt",
                 log: Callable[[str], None] = print) -> None:
        self.keep = max(0, int(keep))
        self.paths: list[str] = []
        self.tag = tag          # log prefix, so the two rings stay distinguishable
        self._log = log

    @property
    def enabled(self) -> bool:
        return self.keep > 0

    def seed(self, paths: Iterable[str]) -> None:
        """Adopt checkpoints already on disk, oldest first, WITHOUT deleting any.

        Called once at startup so a RESUMED run enforces the bound over the whole run
        rather than only over the files this process happens to write. Seeding does not
        delete: a resume that adopts more than `keep` files trims them on the next
        `add()`, which keeps startup free of surprise deletions.
        """
        self.paths.extend(paths)

    def add(self, path: str) -> list[str]:
        """Record `path` as the newest checkpoint and delete any beyond the bound.

        Returns the paths actually removed from disk. A path that cannot be removed is
        dropped from the ring and reported, never silently swallowed: leaving it in the
        ring would make the ring retry the same failing delete forever.
        """
        self.paths.append(path)
        removed: list[str] = []
        if not self.enabled:
            return removed
        while len(self.paths) > self.keep:
            old = self.paths.pop(0)
            try:
                os.remove(old)
                removed.append(old)
                self._log(f"  [{self.tag}] rotated out {os.path.basename(old)} "
                          f"(keep_last={self.keep})")
            except OSError as e:
                self._log(f"  [{self.tag}] could not remove {old}: {e}")
        return removed


_STEP_RE = re.compile(r"step_(\d+)\.pt")


def existing_step_checkpoints(ckpt_dir: str) -> list[str]:
    """Every `step_<N>.pt` in `ckpt_dir`, oldest first.

    Sorted by the integer N, not lexically: `step_900.pt` precedes `step_1000.pt`, which
    a string sort gets backwards and would rotate away the NEWEST checkpoint first.
    `ROLL_step_*.pt` is excluded by the full match — it belongs to the other ring.
    """
    if not os.path.isdir(ckpt_dir):
        return []
    found = []
    for fn in os.listdir(ckpt_dir):
        m = _STEP_RE.fullmatch(fn)
        if m:
            found.append((int(m.group(1)), os.path.join(ckpt_dir, fn)))
    return [p for _, p in sorted(found)]
