"""The core-takeover abort criteria.

TWO signals, both windowed. Prefer the block gain: it is the mechanism, the other is a
symptom.

**Block gain (preferred).** The per-block BACKWARD gain of the core. The backward runs
core.N-1 -> core.0, so a uniform per-block amplification g leaves block 0 a factor g^(N-1)
above the last block; fitting the per-block gradient profile recovers g. g <= 1 is a
non-amplifying backward. Measured on three labelled trajectories, with a 50-step window:

=====================  =========  ==================  ==================  ==============
trajectory             outcome    block gain guard    core share guard    forward gain>2
=====================  =========  ==================  ==================  ==============
``phase1-onset-s0``    took over  step 1434           step 2033           step 2063
``repl-det-b``         took over  step 3368           step 3874           step 3328
``repl-det-a``         recovered  NEVER               NEVER               step 3809
=====================  =========  ==================  ==================  ==============

with the shipped defaults of both guards. The block gain **leads the share by 599 steps on
the control and 506 on repl-det-b**, and neither fires on the run that recovered. Prefer it
because it leads and because it is the mechanism; keep the share criterion as the cheap,
independent second opinion.

(An earlier draft of this table claimed the share rule misses ``repl-det-b`` entirely. That
was true only of a stricter consecutive-run variant used during analysis, not of the
windowed rule actually shipped. Corrected.) 16 of the 36 (min_r2, window,
fraction) settings swept separate all three, so this is a plateau and not a knife edge.

The r2 condition is load-bearing and was not a guess: with ``min_r2=0.0`` the guard fires
on ``repl-det-a``, the run that recovered, at around step 250. A flat healthy profile
produces a large but meaningless gain estimate, and without the r2 floor the guard reads
that noise. The FORWARD per-iteration gain fires on all three, including the survivor, so it
is a correlate and not a criterion — do not use it.

Always read the gain WITH its r2. A healthy profile is flat and noisy, so the fit explains
nothing (r2 ~ 0.1) and the gain estimate is meaningless there; a sick one is cleanly
geometric (median r2 0.971 while the core holds >50% of the gradient). The guard therefore
requires BOTH gain > threshold and r2 >= min_r2 before a step counts.

## The core-share criterion

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
class BlockGainGuard:
    """Fires when the core's per-block BACKWARD gain has been above 1 recently.

    This is the mechanism rather than a symptom, and it is both earlier and more sensitive
    than the share criterion. Same windowed shape, for the same reason: single-step
    excursions happen in healthy runs.

    Args:
        threshold: per-block backward gain above which a step counts. 0 disables.
                   1.0 is the meaningful value — it is where the backward stops contracting.
        min_r2:    ignore steps whose geometric fit is poor. A flat healthy profile gives a
                   near-zero r2 and a meaningless gain, so without this the guard would be
                   reading noise.
    """

    threshold: float = 0.0
    min_r2: float = 0.5
    window: int = 200
    fraction: float = 0.3
    warmup: int = 200

    _buf: deque = field(default_factory=deque, init=False, repr=False)
    fired_at: int | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    def update(self, step: int, gain: float, r2: float) -> bool:
        if not self.enabled or step < self.warmup or self.fired_at is not None:
            return False
        hit = gain == gain and r2 == r2 and gain > self.threshold and r2 >= self.min_r2
        self._buf.append(hit)
        if len(self._buf) > self.window:
            self._buf.popleft()
        if len(self._buf) == self.window and sum(self._buf) / self.window > self.fraction:
            self.fired_at = step
            return True
        return False

    def reason(self) -> str:
        return (f"core per-block backward gain > {self.threshold} (r2 >= {self.min_r2}) on "
                f"more than {self.fraction:.0%} of the last {self.window} probed steps, "
                f"at step {self.fired_at}")


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
