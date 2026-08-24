"""The core-takeover abort criterion.

The shipped ppl guard fires late: on the 2026-08-23 control it struck at step 2620, which
is 587 steps after the takeover had begun. The pre-clip core SHARE moves far earlier, and
being a share it needs no per-arm scale calibration.

**The rule is a fraction over a window, not a consecutive run.** The first version of this
file used "N consecutive probed steps above the threshold" and was validated on one
trajectory. It then MISSED `repl-det-b`, a run that really did take over (final core share
0.8131): b's takeover is intermittent — 0.967, 0.768, 0.264, 0.624, 0.989 — and its
longest consecutive run above 0.5 was 21 steps against a patience of 25. A consecutive rule
is brittle against exactly the ragged onset this failure mode produces.

Validated on three labelled trajectories rather than one:

=====================  =========  ==================  ====================
trajectory             outcome    final core share    rule fires at
=====================  =========  ==================  ====================
``phase1-onset-s0``    took over  ~1.0                2038
``repl-det-b``         took over  0.8131              3369
``repl-det-a``         recovered  0.0152 (peak 0.94)  never
=====================  =========  ==================  ====================

with the shipped defaults, threshold 0.25 over a 50-step window at fraction 0.3. 20 of the
27 (threshold, window, fraction) combinations swept separate the three, so this is a broad
plateau and not a knife edge (``ignore/perf/phase1/tune_guard.py``).

Note what run A proves: a run can touch a core share of **0.9369** and recover completely.
Any rule that fires on a peak, rather than on a sustained majority, aborts healthy runs.

This class is the ONE implementation. ``train.py`` drives it live and
``tests/test_divergence_guard.py`` replays the stored trajectories through the same object,
so the test cannot pass against a re-implementation that has drifted.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class CoreShareGuard:
    """Fires when the looped core has held a majority of the pre-clip gradient recently.

    Args:
        threshold: core share above which a probed step counts as an excursion.
                   0 disables the guard entirely.
        window:    how many recent probed steps to judge on.
        fraction:  the share of that window which must be excursions before firing.
                   Strictly greater than, so fraction=1.0 can never fire.
        warmup:    steps to ignore. A measured startup transient reached 0.1105 at step 122.
    """

    threshold: float = 0.0
    window: int = 50
    fraction: float = 0.3
    warmup: int = 200

    _buf: deque = field(default_factory=deque, init=False, repr=False)
    fired_at: int | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    def update(self, step: int, share: float) -> bool:
        """Feed one probed step. Returns True on the step the guard fires, once."""
        if not self.enabled or step < self.warmup or self.fired_at is not None:
            return False
        self._buf.append(share > self.threshold)
        if len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) == self.window and sum(self._buf) / self.window > self.fraction:
            self.fired_at = step
            return True
        return False

    def reason(self) -> str:
        return (f"core share > {self.threshold} on more than {self.fraction:.0%} of the "
                f"last {self.window} probed steps, at step {self.fired_at}")
