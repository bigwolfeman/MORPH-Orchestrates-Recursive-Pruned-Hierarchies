"""Prune training checkpoints down to the ones any analysis actually reads.

A 3500-step screening arm at `ckpt_every=500` writes 7 checkpoints of ~2.2 GB —
16 GB per seed — and every analysis in this repo reads exactly ONE of them, the
pre-registered measurement step. On 2026-08-27 that arithmetic had grown to
473 GB of checkpoints, of which 364 GB was never read by anything.

DRY RUN BY DEFAULT. Nothing is deleted without `--yes`.

Kept by default:
  * `step_<measure>.pt`      the measurement step (default 3000)
  * `DIVERGED_*.pt` `TAKEOVER_*.pt`   forensic artifacts of a failed run; these
                             are the whole record of a divergence and are small
                             in number
  * anything a SYMLINK points at, anywhere under the root. `onset-sub/` is a
    curated set of symlinks into `onset-capture/`, and it is the only surviving
    statement of which onset rungs the probes use. Deleting a symlink target
    silently breaks a probe, and `du` reports the symlink as 12K, so a naive
    size-based sweep sees no cost to removing it.

REFUSES TO RUN while a trainer is alive. A live run writes checkpoints into the
tree being pruned, and this repo has already lost a run at step 800 to a cleanup
that judged directories by their contents.

    python scripts/prune_checkpoints.py                      # dry run
    python scripts/prune_checkpoints.py --keep-also step_3500 --yes
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

GB = 1073741824


def trainer_running() -> str | None:
    """The live-run guard. Bracketed pattern so the check cannot match itself."""
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except OSError:
        return "could not run ps — refusing to guess"
    for line in out.splitlines():
        if "morph.training.train" in line and "prune_checkpoints" not in line:
            return line.strip()[:120]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="checkpoints/morph")
    ap.add_argument("--measure-step", type=int, default=3000,
                    help="the step every analysis reads; kept everywhere it exists")
    ap.add_argument("--keep-also", action="append", default=[],
                    help="extra basenames to keep, e.g. step_3500. Repeatable.")
    ap.add_argument("--keep-dir", action="append", default=[],
                    help="directory whose checkpoints are kept whole. Repeatable.")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    a = ap.parse_args()

    live = trainer_running()
    if live:
        print(f"REFUSING: a trainer is running — {live}")
        print("A live run writes into this tree. Wait for it to finish.")
        return 2

    if not os.path.isdir(a.root):
        print(f"no such directory: {a.root}")
        return 2

    # Symlink targets are load-bearing and invisible to a size-based sweep.
    pinned = {os.path.realpath(p) for p in glob.glob(f"{a.root}/**/*", recursive=True)
              if os.path.islink(p)}
    keep_names = {f"step_{a.measure_step}.pt", *(f"{k}.pt" if not k.endswith(".pt") else k
                                                 for k in a.keep_also)}

    doomed = []
    for f in sorted(glob.glob(f"{a.root}/**/*.pt", recursive=True)):
        if os.path.islink(f) or not os.path.isfile(f):
            continue
        base, d = os.path.basename(f), os.path.basename(os.path.dirname(f))
        if d in a.keep_dir:
            continue
        if base.startswith(("DIVERGED_", "TAKEOVER_")):
            continue
        if os.path.realpath(f) in pinned:
            continue
        if base in keep_names:
            continue
        doomed.append(f)

    total = sum(os.path.getsize(f) for f in doomed)
    kept = [f for f in glob.glob(f"{a.root}/**/*.pt", recursive=True)
            if os.path.isfile(f) and not os.path.islink(f) and f not in set(doomed)]
    print(f"root={a.root}  measure step={a.measure_step}  also keeping={sorted(keep_names)}")
    print(f"pinned by symlink: {len(pinned)}")
    print(f"DELETE {len(doomed)} files, {total / GB:.1f} GB")
    print(f"KEEP   {len(kept)} files, "
          f"{sum(os.path.getsize(f) for f in kept) / GB:.1f} GB")
    for f in doomed[:8]:
        print(f"   - {f}")
    if len(doomed) > 8:
        print(f"   ... and {len(doomed) - 8} more")

    if not a.yes:
        print("\nDRY RUN. Re-run with --yes to delete.")
        return 0
    for f in doomed:
        os.remove(f)
    print(f"\ndeleted {len(doomed)} files, {total / GB:.1f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
