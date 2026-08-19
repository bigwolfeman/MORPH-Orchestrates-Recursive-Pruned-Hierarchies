"""Single source of truth for the mid-run training-phase schedule.

Three schedules change what a batch looks like part-way through a run:

  TST         bag_size s -> 0           at ``tst_phase1_steps``
  TUL         slot layout off -> on     at ``tul_step``
  curriculum  stage k -> k+1            at the ``CurriculumScheduler`` boundaries

Before this module the first two were separate mutable locals in ``train.py`` and every
loader-rebuild site had to carry the others by hand. Four such sites existed. Two of them
dropped the TUL layout, and three defects followed:

1. The val loader asked ``tul_rt.activate_at == 0.0`` while the live flag asked
   ``start_step >= tul_step``. Two predicates, one question: a resume past a mid-run
   activation built a plain val loader, left the flag True, and skipped the rebuild —
   ``evaluate(tul=True)`` then unpacked a 2-tuple and raised.
2. The curriculum stage transition rebuilt the loader with no ``tul=``. Because
   ``cur_stage`` starts at -1 that site also CREATES the loader, so on a curriculum run
   TUL never started: training ran dense while the flag stayed True and validation kept
   reporting ``val/plan_nats``. Silent, and reproduced.
3. ``tst_phase1_steps`` and ``tul_step`` were derived from ``training.steps``, but the
   curriculum scheduler overrides ``total_steps`` afterwards. Both boundaries landed at
   the wrong step whenever the curriculum changed the run length.

The cure is not a fourth place to remember: it is to stop passing the phase as an
argument a call site can omit. ``PhaseSchedule`` is constructed once, AFTER
``total_steps`` is final, and ``at(step)`` is the only thing that answers "what phase is
this". Curriculum stage stays with ``CurriculumScheduler`` — it owns seq_len, micro-batch
and the RoPE re-anchor as well — but its loader rebuild reads the phase from here.

Cost per step: one frozen-dataclass construction and a two-field ``!=``. Against a step
of order 100 ms that is free, and it replaces three separate ``if <phase change>`` blocks
with one equality test. Nothing here runs inside the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["TrainPhase", "PhaseSchedule"]


@dataclass(frozen=True)
class TrainPhase:
    """What a given step implies for batch construction.

    Frozen and comparable so the training loop detects a phase change with ``!=``
    instead of one bespoke predicate per schedule.
    """

    bag_size: int      # TST superposition width; 0 = standard next-token prediction
    tul_on: bool       # does the loader build the TUL slot layout?

    def __post_init__(self) -> None:
        # Invariant 6 (docs/runtime-invariants.md §6b): TUL activates AT the TST switch,
        # never during it. The forward raises on the same combination; raising here means
        # a bad schedule is caught when the schedule is built, not at step N.
        if self.bag_size > 0 and self.tul_on:
            raise ValueError(
                f"bag_size={self.bag_size} with tul_on=True: TST superposition and the "
                f"TUL layout are mutually exclusive (runtime-invariants §6b)")


class PhaseSchedule:
    """Maps a step to its ``TrainPhase``. Build ONCE, after ``total_steps`` is final."""

    def __init__(self, *, total_steps: int, tst_bag_size: int = 0,
                 tst_ratio: float = 0.0, tul_activate_at: float | None = None) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        self.total_steps = int(total_steps)
        self.tst_bag_size = int(tst_bag_size)
        # tst_bag_size == 0 -> TST off -> boundary 0 -> every step reports bag_size 0.
        self.tst_phase1_steps = (int(tst_ratio * total_steps)
                                 if self.tst_bag_size > 0 else 0)
        # tul_activate_at is None -> TUL off. -1 sorts below every real step, so
        # `step >= tul_step` is False for all steps without a second predicate.
        self.tul_step = (int(tul_activate_at * self.total_steps)
                         if tul_activate_at is not None else -1)

        if self.tul_step >= 0 and self.tul_step < self.tst_phase1_steps:
            raise ValueError(
                f"tul activates at step {self.tul_step} but TST superposition runs to "
                f"step {self.tst_phase1_steps}; they are mutually exclusive. Raise "
                f"tul.activate_at to >= training.tst_ratio, or set tst_bag_size: 0.")

    def at(self, step: int) -> TrainPhase:
        """The phase in force AT ``step``.

        At ``step == tst_phase1_steps == tul_step`` this reports bag 0 and TUL on — the
        TST switch and the TUL activation land on the same step, which is the intended
        hand-off and the only point where both schedules move at once.
        """
        return TrainPhase(
            bag_size=self.tst_bag_size if step < self.tst_phase1_steps else 0,
            tul_on=self.tul_step >= 0 and step >= self.tul_step,
        )

    def __repr__(self) -> str:  # shown in the startup banner
        tul = "off" if self.tul_step < 0 else f"step {self.tul_step}"
        tst = ("off" if self.tst_bag_size <= 0
               else f"bag {self.tst_bag_size} to step {self.tst_phase1_steps}")
        return (f"PhaseSchedule(total_steps={self.total_steps}, tst={tst}, tul={tul})")
