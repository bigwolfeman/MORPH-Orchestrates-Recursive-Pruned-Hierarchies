"""The core-takeover abort criterion.

The existing ppl guard in ``train.py`` fires late. On the measured control it struck at
step 2620, which is 587 steps after the takeover had already begun and long after the run
was worth continuing. The pre-clip core SHARE moves far earlier, and it is a share, so it
needs no per-arm scale calibration.

Measured on the 2026-08-23 control (docs/experiments/results/2026-08-23-tul-onset-ordering.md):

===================================  ========  =======================
rule (sustained 25 probed steps)     fires at  warning before ppl guard
===================================  ========  =======================
pre-clip core share > 0.25              2031            589 steps
pre-clip core share > 0.50              2033            587 steps
``preclip/core`` > 1.0                  2032            588 steps
pre-clip core share > 0.90              2192            428 steps
===================================  ========  =======================

against a healthy baseline of 0.0145 and a highest healthy value of 0.031 anywhere before
step 1900.

**It must be a RATCHET, not a level.** Every pre-takeover excursion above 0.5 in that run
lasted exactly ONE probed step, and the gate arm once touched 0.3462 at step 700 and fell
back to 0.0783 without dying. A bare threshold would have false-fired at step ~1450, 570
steps early. Hence ``patience``.

This class is the ONE implementation of the rule. ``train.py`` drives it live and
``tests/test_divergence_guard.py`` replays stored control trajectories through the same
object, so the test cannot pass against a re-implementation that has drifted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CoreShareGuard:
    """Fires when the looped core's share of the pre-clip gradient stays high.

    Args:
        threshold: share above which a step counts as an excursion. 0 disables the guard.
        patience:  consecutive probed steps required before firing. Single-step excursions
                   are normal well before the takeover, so this is what separates the
                   ratchet from the noise.
        warmup:    steps to ignore. The first ~120 steps carry a startup transient that
                   reached 0.1105 in one measured run.
    """

    threshold: float = 0.0
    patience: int = 25
    warmup: int = 200

    _run: int = field(default=0, init=False)
    _run_start: int | None = field(default=None, init=False)
    fired_at: int | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        return self.threshold > 0.0

    def update(self, step: int, share: float) -> bool:
        """Feed one probed step. Returns True on the step the guard fires, once.

        ``fired_at`` records the START of the qualifying run, not its confirmation point —
        that is the step an operator would want to roll back to.
        """
        if not self.enabled or step < self.warmup or self.fired_at is not None:
            return False
        if share > self.threshold:
            if self._run == 0:
                self._run_start = step
            self._run += 1
            if self._run >= self.patience:
                self.fired_at = self._run_start
                return True
        else:
            self._run = 0
            self._run_start = None
        return False

    def reason(self) -> str:
        return (f"core share > {self.threshold} for {self.patience} consecutive probed "
                f"steps, beginning at step {self.fired_at}")
