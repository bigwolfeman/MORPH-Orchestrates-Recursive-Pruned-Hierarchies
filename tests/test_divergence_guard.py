"""The core-takeover abort criterion, replayed against THREE real labelled trajectories.

Fixtures are the per-step pre-clip core share from three 2026-08-23 runs (wandb morph-tul):

  data_core_share_control.json  phase1-onset-s0  TOOK OVER
  data_core_share_repl_b.json   repl-det-b       TOOK OVER (final share 0.8131)
  data_core_share_repl_a.json   repl-det-a       RECOVERED (peak 0.9369, final 0.0152)

repl_a and repl_b are byte-identical runs at the same seed. One diverged and one did not,
which is both the reason this guard is needed and the reason it must be validated on more
than the trajectory it was tuned on. The first version of the rule was
"N consecutive steps above the threshold"; it passed 12 tests against the control alone
and MISSED repl_b, whose takeover is intermittent.

The object under test is the one train.py drives, so these cannot pass against a drifted
re-implementation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from morph.training.divergence_guard import CoreShareGuard

_D = Path(__file__).parent
CONTROL = json.loads((_D / "data_core_share_control.json").read_text())
REPL_A = json.loads((_D / "data_core_share_repl_a.json").read_text())
REPL_B = json.loads((_D / "data_core_share_repl_b.json").read_text())

DIVERGED = {"control": CONTROL, "repl_b": REPL_B}
PPL_GUARD_STRUCK_AT = 2620  # the existing ppl guard's first strike on the control run


def _replay(guard: CoreShareGuard, rows) -> int | None:
    for r in rows:
        if guard.update(r["step"], r["share"]):
            return guard.fired_at
    return None


# ── the contract: fire on every run that died, on none that lived ────────────

@pytest.mark.parametrize("name", list(DIVERGED))
def test_it_fires_on_every_run_that_actually_took_over(name):
    fired = _replay(CoreShareGuard(threshold=0.25), DIVERGED[name])
    assert fired is not None, f"missed {name}, which really did take over"


def test_it_does_not_fire_on_the_run_that_recovered():
    """repl_a peaked at a core share of 0.9369 and finished healthy at 0.0152. A guard
    that aborts it is worse than no guard."""
    assert max(r["share"] for r in REPL_A) > 0.9          # fixture sanity
    assert _replay(CoreShareGuard(threshold=0.25), REPL_A) is None


def test_a_consecutive_run_rule_would_miss_repl_b():
    """Pins WHY the rule is a fraction over a window. window=fraction=1.0 with a long
    window is the closest this class comes to 'N consecutive', and it must fail here —
    repl_b's longest consecutive stretch above 0.25 is shorter than its window."""
    windowed = _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.3), REPL_B)
    strict = _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.98), REPL_B)
    assert windowed is not None
    assert strict is None, "the strict rule caught it; the regression this guards is gone"


def test_it_beats_the_existing_ppl_guard_on_the_control():
    fired = _replay(CoreShareGuard(threshold=0.25), CONTROL)
    assert PPL_GUARD_STRUCK_AT - fired > 400, "not enough warning to be worth the code"


def test_it_does_not_fire_on_the_healthy_prefix_of_the_control():
    healthy = [r for r in CONTROL if r["step"] < 1400]
    assert max(r["share"] for r in healthy) < 0.25        # fixture sanity
    assert _replay(CoreShareGuard(threshold=0.25), healthy) is None


def test_the_defaults_are_the_validated_rule():
    g = CoreShareGuard(threshold=0.25)
    assert (g.window, g.fraction, g.warmup) == (50, 0.3, 200)


@pytest.mark.parametrize("rows,expected", [(CONTROL, 2038), (REPL_B, 3369)])
def test_the_shipped_defaults_fire_where_the_docs_say(rows, expected):
    """Pins code to the published numbers so they cannot drift apart silently."""
    assert _replay(CoreShareGuard(threshold=0.25), rows) == expected


# ── mechanics ────────────────────────────────────────────────────────────────

def test_a_single_spike_never_fires():
    rows = [{"step": s, "share": 0.9 if s == 500 else 0.01} for s in range(300, 900)]
    assert _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.3), rows) is None


def test_warmup_ignores_the_startup_transient():
    rows = [{"step": s, "share": 0.9} for s in range(0, 120)]
    assert _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.3, warmup=0), rows) == 49
    assert _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.3, warmup=200), rows) is None


def test_the_window_slides_so_old_excursions_expire():
    """A burst long ago must not add to a burst now."""
    rows = [{"step": s, "share": 0.9} for s in range(300, 310)]
    rows += [{"step": s, "share": 0.0} for s in range(310, 900)]
    rows += [{"step": s, "share": 0.9} for s in range(900, 910)]
    assert _replay(CoreShareGuard(threshold=0.25, window=50, fraction=0.3), rows) is None


def test_threshold_zero_disables_it_completely():
    g = CoreShareGuard(threshold=0.0)
    assert not g.enabled
    assert _replay(g, CONTROL) is None


def test_it_fires_only_once():
    g = CoreShareGuard(threshold=0.25)
    fired = [r["step"] for r in CONTROL if g.update(r["step"], r["share"])]
    assert len(fired) == 1, f"fired {len(fired)} times; the abort would re-trigger"


def test_fraction_one_can_never_fire():
    """Strictly-greater-than means a saturated window still does not trip fraction=1.0."""
    rows = [{"step": s, "share": 1.0} for s in range(300, 900)]
    assert _replay(CoreShareGuard(threshold=0.25, window=50, fraction=1.0), rows) is None


# ── the block-gain criterion: the mechanism, not the symptom ─────────────────

from morph.training.divergence_guard import BlockGainGuard  # noqa: E402

BG_CONTROL = json.loads((_D / "data_blockgain_control.json").read_text())
BG_REPL_B = json.loads((_D / "data_blockgain_repl_b.json").read_text())
BG_REPL_A = json.loads((_D / "data_blockgain_repl_a.json").read_text())
BG_DIVERGED = {"control": BG_CONTROL, "repl_b": BG_REPL_B}


def _replay_bg(guard: BlockGainGuard, rows) -> int | None:
    for r in rows:
        if guard.update(r["step"], r["gain"], r["r2"]):
            return guard.fired_at
    return None


@pytest.mark.parametrize("name", list(BG_DIVERGED))
def test_block_gain_fires_on_every_run_that_took_over(name):
    assert _replay_bg(BlockGainGuard(threshold=1.0), BG_DIVERGED[name]) is not None


def test_block_gain_stays_quiet_on_the_run_that_recovered():
    assert _replay_bg(BlockGainGuard(threshold=1.0), BG_REPL_A) is None


@pytest.mark.parametrize("bg_rows,share_rows,least_lead", [
    (BG_CONTROL, CONTROL, 500),
    (BG_REPL_B, REPL_B, 400),
])
def test_block_gain_leads_the_share_on_every_run_that_took_over(bg_rows, share_rows, least_lead):
    """The reason this criterion exists: it is the mechanism, so it moves first. On the
    control it fires at 1434 against the share's 2033, and on repl_b at 3368 against 3874."""
    gain_at = _replay_bg(BlockGainGuard(threshold=1.0), bg_rows)
    share_at = _replay(CoreShareGuard(threshold=0.5, window=50, fraction=0.5), share_rows)
    assert gain_at is not None and share_at is not None
    assert share_at - gain_at >= least_lead, (
        f"block gain fired at {gain_at}, share at {share_at}: lead only {share_at - gain_at}")


def test_the_r2_floor_is_what_keeps_it_off_the_survivor():
    """Not a design preference: with min_r2=0 the guard fires on repl_a, the run that
    recovered, early in training. A flat healthy profile yields a large but meaningless
    gain, and the r2 floor is what refuses to read it."""
    lenient = _replay_bg(BlockGainGuard(threshold=1.0, min_r2=0.0), BG_REPL_A)
    shipped = _replay_bg(BlockGainGuard(threshold=1.0), BG_REPL_A)
    assert lenient is not None, "premise stale: min_r2=0 no longer false-fires"
    assert shipped is None


def test_the_defaults_are_the_swept_rule():
    g = BlockGainGuard(threshold=1.0)
    assert (g.min_r2, g.window, g.fraction, g.warmup) == (0.5, 200, 0.3, 200)


def test_a_poor_geometric_fit_is_ignored():
    """A flat healthy profile gives a meaningless gain estimate. Without the r2 condition
    the guard would read noise, so a high gain with a low r2 must not count."""
    rows = [{"step": s, "gain": 5.0, "r2": 0.05} for s in range(300, 900)]
    assert _replay_bg(BlockGainGuard(threshold=1.0, min_r2=0.5), rows) is None
    assert _replay_bg(BlockGainGuard(threshold=1.0, min_r2=0.0), rows) is not None


def test_nan_gain_never_counts():
    rows = [{"step": s, "gain": float("nan"), "r2": float("nan")} for s in range(300, 900)]
    assert _replay_bg(BlockGainGuard(threshold=1.0), rows) is None


def test_block_gain_threshold_zero_disables_it():
    g = BlockGainGuard(threshold=0.0)
    assert not g.enabled
    assert _replay_bg(g, BG_CONTROL) is None


def test_block_gain_fires_where_the_docs_say():
    """Pins the shipped rule to the published numbers."""
    assert _replay_bg(BlockGainGuard(threshold=1.0), BG_CONTROL) == 1434
    assert _replay_bg(BlockGainGuard(threshold=1.0), BG_REPL_B) == 3368
