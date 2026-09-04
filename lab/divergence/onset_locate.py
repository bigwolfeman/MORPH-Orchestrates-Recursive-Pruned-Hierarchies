"""Locate the first gradient spike in a probe file and the checkpoint to replay it from.

    python lab/divergence/onset_locate.py --probe probe.jsonl --ckpt-dir checkpoints/morph/run \
        [--threshold 1e3] [--min-step 200] [--lead 10] [--key preclip/total]

Prints ONE JSON object on stdout:

    {"onset": <first step >= min_step with key > threshold>,
     "peak": <max of key over the file>, "peak_step": <its step>,
     "ckpt": <path of the newest ROLL_step_N.pt / step_N.pt with N <= onset - lead>,
     "ckpt_step": N}

Exit codes: 0 found; 3 no spike in the file; 4 spike but no usable checkpoint before it.
The lead keeps the replay's first probed steps in the calm regime, so the capture holds a
within-run control on both sides of the onset.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


def locate(probe: str, ckpt_dir: str, key: str, threshold: float, min_step: int,
           lead: int) -> tuple[dict, int]:
    onset = None
    peak, peak_step = float("-inf"), None
    with open(probe) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            v = r.get(key)
            if v is None:
                continue
            s = int(r["step"])
            if v > peak:
                peak, peak_step = float(v), s
            if onset is None and s >= min_step and v > threshold:
                onset = s
    out = {"onset": onset, "peak": peak, "peak_step": peak_step, "ckpt": None, "ckpt_step": None}
    if onset is None:
        return out, 3
    cands: list[tuple[int, str]] = []
    for p in glob.glob(os.path.join(ckpt_dir, "*.pt")):
        m = re.search(r"(?:ROLL_)?step_(\d+)\.pt$", os.path.basename(p))
        if m and int(m.group(1)) <= onset - lead:
            cands.append((int(m.group(1)), p))
    if not cands:
        return out, 4
    n, p = max(cands)
    out["ckpt"], out["ckpt_step"] = p, n
    return out, 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--probe", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--key", default="preclip/total")
    ap.add_argument("--threshold", type=float, default=1e3)
    ap.add_argument("--min-step", type=int, default=200)
    ap.add_argument("--lead", type=int, default=10)
    a = ap.parse_args()
    out, rc = locate(a.probe, a.ckpt_dir, a.key, a.threshold, a.min_step, a.lead)
    print(json.dumps(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
