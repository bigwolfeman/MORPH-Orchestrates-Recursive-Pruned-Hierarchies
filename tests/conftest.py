from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Several tests pin bit-exact float64 sums (loss, logit sum, gradient sum) measured on a
# one-thread CPU run. A multi-thread reduction changes the summation order and moves those
# sums in the 10th digit (6 failures at OMP 2, 8 at OMP 4, 0 at 1 — measured 2026-09-13).
# One thread here makes the pins a property of the code, not of the launcher's env, and
# keeps a background pytest from loading the desktop.
import torch  # noqa: E402

torch.set_num_threads(1)
