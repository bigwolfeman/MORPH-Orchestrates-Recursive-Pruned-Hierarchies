"""The core-takeover abort criterion, replayed against a REAL diverging trajectory.

`tests/data_core_share_control.json` is the per-step pre-clip core share from the
2026-08-23 control run `phase1-onset-s0` (3522 steps, wandb morph-tul), the run whose
onset is written up in docs/experiments/results/2026-08-23-tul-onset-ordering.md. The
guard object under test is the same one `train.py` drives live, so these tests cannot pass
against a re-implementation that has drifted from the shipped rule.

This is acceptance criterion 4 of
.agents/notes/proposed/process/2026-08-23-divergence-root-cause-plan.md: the criterion
must fire on the stored control trajectory before step 2100.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from morph.training.divergence_guard import CoreShareGuard

CONTROL = json.loads((Path(__file__).parent / "data_core_share_control.json").read_text())
PPL_GUARD_STRUCK_AT = 2620  # the existing guard's first strike on this same run


def _replay(guard: CoreShareGuard, rows=CONTROL) -> int | None:
    for r in rows:
        if guard.update(r["step"], r["share"]):
            return guard.fired_at
    return None


def test_it_fires_on_the_real_control_before_step_2100():
    fired = _replay(CoreShareGuard(threshold=0.5))
    assert fired is not None, "the guard never fired on a trajectory that really diverged"
    assert fired < 2100, f"fired at {fired}, too late to be worth having"


def test_it_beats_the_existing_ppl_guard_by_hundreds_of_steps():
    """The whole point. The shipped ppl guard struck at 2620 on this run."""
    fired = _replay(CoreShareGuard(threshold=0.5))
    warning = PPL_GUARD_STRUCK_AT - fired
    assert warning > 400, f"only {warning} steps of warning; not worth the code"


def test_it_does_not_fire_on_the_healthy_prefix():
    """Steps before 1400 are healthy in this run — the highest share there is 0.144.
    A guard that fires on them would abort good runs."""
    healthy = [r for r in CONTROL if r["step"] < 1400]
    assert max(r["share"] for r in healthy) < 0.25  # fixture sanity
    assert _replay(CoreShareGuard(threshold=0.5), healthy) is None


def test_patience_is_load_bearing_against_single_step_excursions():
    """Every pre-takeover excursion above 0.5 in this run lasted ONE probed step. With
    patience=1 the guard false-fires long before the real takeover; with the shipped
    patience it does not."""
    impatient = _replay(CoreShareGuard(threshold=0.5, patience=1))
    patient = _replay(CoreShareGuard(threshold=0.5, patience=25))
    assert impatient is not None and patient is not None
    assert impatient < patient - 400, (
        f"patience bought only {patient - impatient} steps; the ratchet is not doing work")


def test_warmup_ignores_the_startup_transient():
    """A measured startup transient reached 0.1105 at step 122. A low threshold plus no
    warmup must be catchable, and the shipped warmup must suppress it."""
    rows = [{"step": s, "share": 0.9} for s in range(0, 60)]
    assert _replay(CoreShareGuard(threshold=0.5, patience=25, warmup=0), rows) == 0
    assert _replay(CoreShareGuard(threshold=0.5, patience=25, warmup=200), rows) is None


def test_a_fallback_resets_the_run():
    """The ratchet must count CONSECUTIVE steps. One dip below the threshold restarts it."""
    g = CoreShareGuard(threshold=0.5, patience=5, warmup=0)
    rows = [{"step": s, "share": 0.9} for s in range(4)]
    rows += [{"step": 4, "share": 0.1}]                       # the dip
    rows += [{"step": s, "share": 0.9} for s in range(5, 10)]
    assert _replay(g, rows) == 5, "the counter did not reset on the dip"


def test_it_reports_the_start_of_the_run_not_the_confirmation_step():
    g = CoreShareGuard(threshold=0.5, patience=10, warmup=0)
    rows = [{"step": s, "share": 0.9} for s in range(100, 120)]
    assert _replay(g, rows) == 100


def test_threshold_zero_disables_it_completely():
    g = CoreShareGuard(threshold=0.0)
    assert not g.enabled
    assert _replay(g) is None


def test_it_fires_only_once():
    g = CoreShareGuard(threshold=0.5)
    fired = [r["step"] for r in CONTROL if g.update(r["step"], r["share"])]
    assert len(fired) == 1, f"fired {len(fired)} times; the abort would re-trigger"


@pytest.mark.parametrize("thr", [0.25, 0.5, 0.9])
def test_every_documented_threshold_fires_where_the_results_doc_says(thr):
    """The results table names 2031 / 2033 / 2192 for thresholds 0.25 / 0.5 / 0.9.
    If the shipped rule drifts from the published numbers, this fails."""
    expected = {0.25: 2031, 0.5: 2033, 0.9: 2192}[thr]
    assert _replay(CoreShareGuard(threshold=thr)) == expected
