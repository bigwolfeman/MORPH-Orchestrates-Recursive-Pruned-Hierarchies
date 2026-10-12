"""lab/perf/bench_step.py: parsing and aggregation (CPU, no trainer)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "bench_step", Path(__file__).resolve().parents[1] / "lab" / "perf" / "bench_step.py")
bs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs)


def _lines(stamps, tokens=24576):
    out = ["noise line", "[    20/300] loss=1.0  sps=3.0  tok/s=99"]
    out += [f"[steptime] step={s} t={t:.6f} tokens={tokens}\n" for s, t in stamps]
    return out


def _clock(step_ms, n=161, t0=1000.0, jitter=0.0):
    """Stamps for steps 0..n-1: async, so the per-step stamps wobble but every 20th is exact."""
    out = []
    for s in range(n):
        t = t0 + sum(step_ms(i) for i in range(s + 1)) / 1e3
        out.append((s, t + (0.0 if s % 20 == 0 else jitter * ((s % 7) - 3))))
    return out


def test_parse_ignores_other_lines():
    rows = bs.parse_steptime(_lines([(0, 12.5), (1, 12.75)]))
    assert rows == [(0, 12.5, 24576), (1, 12.75, 24576)]


def test_blocks_skip_warmup_and_ignore_unsynced_stamps():
    # steps 0..99 slow (compile warm-up, 900 ms), steps 100..140 at 200 ms, then 100 ms
    rows = bs.parse_steptime(_lines(_clock(lambda i: 900.0 if i < 100 else (200.0 if i <= 140 else 100.0),
                                           jitter=0.05)))
    assert bs.block_ms(rows, skip=100) == pytest.approx([200.0, 200.0, 100.0])
    ss = bs.steady_state(rows, skip=100)
    assert ss["n_blocks"] == 3 and ss["median_ms"] == pytest.approx(200.0)
    assert ss["p10_ms"] == pytest.approx(120.0) and ss["p90_ms"] == pytest.approx(200.0)
    assert ss["tok_s"] == pytest.approx(24576 / 0.200)
    # the warm-up block is excluded by skip, included when skip is 0
    assert bs.block_ms(rows, skip=0)[0] == pytest.approx(900.0)


def test_steady_state_raises_without_a_block():
    with pytest.raises(ValueError):
        bs.steady_state(bs.parse_steptime(_lines(_clock(lambda i: 100.0, n=30))), skip=20)


def test_aggregate_two_arms_ratio_and_repeats():
    def run(ms_per_step):
        return bs.steady_state(bs.parse_steptime(_lines(_clock(lambda i: ms_per_step, n=61))), 20)
    res = bs.aggregate({"a": [run(100.0), run(120.0)], "b": [run(200.0), run(200.0)]})
    assert res["arms"]["a"]["repeat_median_ms"] == pytest.approx([100.0, 120.0])
    assert res["arms"]["a"]["median_ms"] == pytest.approx(110.0)
    assert res["ratio_of_medians"]["value"] == pytest.approx(0.55)
    assert (res["ratio_of_medians"]["num"], res["ratio_of_medians"]["den"]) == ("a", "b")
    assert "ratio of medians a/b = 0.5500" in bs.format_table(res)


def test_parse_arm():
    assert bs.parse_arm("x=cfg") == ("x", "cfg", [])
    assert bs.parse_arm('x=cfg:model.a=1 "model.b=[1, 2]"') == ("x", "cfg", ["model.a=1", "model.b=[1, 2]"])
    with pytest.raises(ValueError):
        bs.parse_arm("nocolon")
