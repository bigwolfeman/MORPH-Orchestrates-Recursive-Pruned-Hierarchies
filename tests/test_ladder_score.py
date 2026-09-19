"""Contract tests for `lab/divergence/ladder_score.py`, the ladder's paired reader.

Two properties decide whether a number this scorer prints means what the depth-ladder
record claims:

1. It pairs position by position and REFUSES two sweeps whose `tok_index` differs. The
   ladder's arms all cut the same validation stream, so a difference means the cut moved
   and a position-by-position subtraction would compare different text.
2. The delta is TOKEN-weighted and its interval resamples whole ROWS. A mean of per-row
   means and a token-weighted mean differ whenever rows hold different token counts, and
   a per-token bootstrap would report an interval several times too narrow.

Both are checked against hand-computed values on synthetic arrays, never against the
implementation's own output.

Record: lab/experiments/successes/2026-09-13-arc-depth-ladder-ship.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DIV = Path(__file__).resolve().parents[1] / "lab" / "divergence"
if str(_DIV) not in sys.path:  # the probes use bare imports (`from _stats import ...`)
    sys.path.insert(0, str(_DIV))

import ladder_score as ls  # noqa: E402


# ---------------------------------------------------------------- pairing refuses

def test_require_same_index_accepts_identical():
    idx = np.array([0, 1, 5, 9], dtype=np.int32)
    ls.require_same_index(idx, idx.copy(), "a", "b")   # must not raise


def test_require_same_index_refuses_different_values():
    a = np.array([0, 1, 2, 3], dtype=np.int32)
    b = np.array([0, 1, 7, 3], dtype=np.int32)
    with pytest.raises(ValueError, match="tok_index differs"):
        ls.require_same_index(a, b, "arm-a", "arm-b")


def test_require_same_index_refuses_different_length():
    a = np.arange(4, dtype=np.int32)
    b = np.arange(5, dtype=np.int32)
    with pytest.raises(ValueError, match="tok_index length differs"):
        ls.require_same_index(a, b, "arm-a", "arm-b")


def test_row_ids_refuses_counts_that_do_not_cover_the_tokens():
    with pytest.raises(ValueError, match="row counts sum to"):
        ls.row_ids(np.array([2, 2]), 5)


def test_row_ids_repeats_in_row_order():
    assert ls.row_ids(np.array([2, 3]), 5).tolist() == [0, 0, 1, 1, 1]


# ------------------------------------------------------------- the paired delta

def test_point_is_the_token_weighted_mean_not_the_row_mean():
    # row 0: 3 tokens, delta +1 each. row 1: 1 token, delta -3.
    # token-weighted: (3*1 + 1*(-3)) / 4 = 0.0   <- the contract
    # mean of per-row means: (1 + (-3)) / 2 = -1.0  <- the wrong answer
    ce_a = np.array([2.0, 2.0, 2.0, 1.0])
    ce_b = np.array([1.0, 1.0, 1.0, 4.0])
    rid = ls.row_ids(np.array([3, 1]), 4)
    res = ls.paired_delta(ce_a, ce_b, rid, n_rows=2, n_boot=16, seed=0)
    assert res["point"] == pytest.approx(0.0, abs=1e-12)
    assert res["ce_a"] == pytest.approx(7.0 / 4.0, abs=1e-12)
    assert res["ce_b"] == pytest.approx(7.0 / 4.0, abs=1e-12)
    assert res["n_tokens"] == 4


def test_constant_delta_gives_a_degenerate_interval():
    # every row carries the same per-token delta, so every resample reads exactly +0.5
    ce_b = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ce_a = ce_b + 0.5
    rid = ls.row_ids(np.array([2, 2, 2]), 6)
    res = ls.paired_delta(ce_a, ce_b, rid, n_rows=3, n_boot=200, seed=0)
    assert res["point"] == pytest.approx(0.5, abs=1e-12)
    assert res["lo"] == pytest.approx(0.5, abs=1e-12)
    assert res["hi"] == pytest.approx(0.5, abs=1e-12)


def test_interval_is_the_hand_computed_row_bootstrap_support():
    # Two rows of one token each, deltas +1 and -1. Resampling two rows with replacement
    # gives {row0,row0} -> +1, {row1,row1} -> -1, and either mixed draw -> 0. The
    # bootstrap distribution therefore has support {-1, 0, +1} with weights 1/4, 1/2,
    # 1/4, so the 2.5 % and 97.5 % percentiles are exactly -1 and +1.
    ce_a = np.array([1.0, 0.0])
    ce_b = np.array([0.0, 1.0])
    rid = ls.row_ids(np.array([1, 1]), 2)
    res = ls.paired_delta(ce_a, ce_b, rid, n_rows=2, n_boot=2000, seed=0)
    assert res["point"] == pytest.approx(0.0, abs=1e-12)
    assert res["lo"] == pytest.approx(-1.0, abs=1e-12)
    assert res["hi"] == pytest.approx(+1.0, abs=1e-12)


def test_paired_delta_refuses_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ls.paired_delta(np.zeros(3), np.zeros(4), np.zeros(3, dtype=np.int64), 1)


# ----------------------------------------------------------- end to end on files

def _write_arm(tmp: Path, label: str, tok_index, ce_by_depth, counts, step=20000):
    npz = tmp / f"sweep_{label}.{label}.tokens.npz"
    np.savez_compressed(npz, tok_index=np.asarray(tok_index, dtype=np.int32),
                        **{f"ce_{d}": np.asarray(v, dtype=np.float32)
                           for d, v in ce_by_depth.items()})
    js = tmp / f"sweep_{label}.json"
    js.write_text(json.dumps({label: {"step": step, "rows": len(counts),
                                      "row_n_tokens": list(counts),
                                      "tokens_npz": str(npz)}}))
    return str(js)


def test_score_pair_reads_the_named_depths(tmp_path):
    idx = [0, 1, 2, 3]
    a = _write_arm(tmp_path, "arm-a", idx, {3: [1.0, 1.0, 1.0, 1.0],
                                            6: [9.0, 9.0, 9.0, 9.0]}, [2, 2])
    b = _write_arm(tmp_path, "arm-b", idx, {3: [7.0, 7.0, 7.0, 7.0],
                                            6: [0.5, 0.5, 0.5, 0.5]}, [2, 2])
    res = ls.score_pair({"name": "a@3 - b@6", "a": {"json": a, "depth": 3},
                         "b": {"json": b, "depth": 6}}, n_boot=64)
    assert res["ce_a"] == pytest.approx(1.0, abs=1e-6)
    assert res["ce_b"] == pytest.approx(0.5, abs=1e-6)
    assert res["point"] == pytest.approx(0.5, abs=1e-6)
    assert res["n_rows"] == 2 and res["n_tokens"] == 4


def test_score_pair_refuses_arms_that_cut_the_stream_differently(tmp_path):
    a = _write_arm(tmp_path, "arm-a", [0, 1, 2, 3], {6: [1.0] * 4}, [2, 2])
    b = _write_arm(tmp_path, "arm-b", [0, 1, 2, 9], {6: [1.0] * 4}, [2, 2])
    with pytest.raises(ValueError, match="tok_index differs"):
        ls.score_pair({"a": {"json": a, "depth": 6}, "b": {"json": b, "depth": 6}})


def test_score_pair_refuses_a_missing_depth(tmp_path):
    a = _write_arm(tmp_path, "arm-a", [0, 1], {6: [1.0, 1.0]}, [2])
    b = _write_arm(tmp_path, "arm-b", [0, 1], {6: [1.0, 1.0]}, [2])
    with pytest.raises(ValueError, match="no depth 4"):
        ls.score_pair({"a": {"json": a, "depth": 4}, "b": {"json": b, "depth": 6}})
