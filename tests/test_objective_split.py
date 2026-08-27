"""Unit tests for the objective-split probe's arithmetic and its group masks.

The probe's whole value is that its two gates BITE. A gate that cannot fail is
theatre, so `--sabotage` exists purely so the end-to-end test can prove each
gate fails when the thing it guards is broken. The GPU test that drives it is
marked and skipped without CUDA; these run anywhere.
"""
from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest
import torch

_P = pathlib.Path(__file__).resolve().parents[1] / "lab" / "divergence" / "objective_split.py"


def _load():
    """Import the probe WITHOUT its heavy `lab._build` / morph.training imports."""
    src = _P.read_text()
    head = src.split("sys.path.insert", 1)[0]
    mod = types.ModuleType("objective_split_partial")
    mod.__dict__["torch"] = torch
    exec(compile(head, str(_P), "exec"), mod.__dict__)          # noqa: S102
    # `cos`, `region_of` and `ce_group` live below the imports; pull them by slicing.
    tail = src[src.index("CE_GROUPS ="):]
    exec(compile(tail, str(_P), "exec"), mod.__dict__)          # noqa: S102
    return mod


M = _load()


def test_cos_is_the_real_cosine():
    a = torch.tensor([1.0, 0.0, 0.0])
    assert M.cos(a, torch.tensor([1.0, 0.0, 0.0])) == pytest.approx(1.0)
    assert M.cos(a, torch.tensor([-1.0, 0.0, 0.0])) == pytest.approx(-1.0)
    assert M.cos(a, torch.tensor([0.0, 1.0, 0.0])) == pytest.approx(0.0)
    # scale invariance: the verdict must not move when one objective is upweighted
    assert M.cos(a, torch.tensor([3.0, 3.0, 0.0])) == pytest.approx(
        M.cos(a, torch.tensor([9.0, 9.0, 0.0])))


def test_cos_refuses_a_zero_vector_instead_of_returning_zero():
    """A zero gradient must read nan, NOT 0.0 — 0.0 would print 'orthogonal' and
    silently claim the objectives do not interact when one of them is absent."""
    assert torch.isnan(torch.tensor(M.cos(torch.zeros(4), torch.ones(4))))


def test_region_of_strips_the_compile_wrapper():
    assert M.region_of("_orig_mod.core.3.mlp.0.down.weight") == "core"
    assert M.region_of("core.3.mlp.0.down.weight") == "core"
    assert M.region_of("_orig_mod.embed.hybrid.euc_embed.weight") == "embed"


class _FakeRoot:
    """Minimal stand-in for the model: `ce_group` only touches `_tul_half_weights`
    and `cfg.tul.{plast,emit}_weight`."""

    def __init__(self, pw: float, ew: float, bl: int = 8):
        self.bl = bl
        self.cfg = types.SimpleNamespace(
            tul=types.SimpleNamespace(plast_weight=pw, emit_weight=ew))
        self.p_idx = torch.tensor([2, bl])       # bl = the trailing pad row
        self.z_idx = torch.tensor([4, bl])
        self._tul_half_weights = self._orig

    def _orig(self, labels, layout):
        w = torch.ones(self.bl + 1)
        w[self.p_idx] = self.cfg.tul.plast_weight
        w[self.z_idx] = self.cfg.tul.emit_weight
        return w[: self.bl], self.p_idx, self.z_idx


def _weights(root, group, unit=False):
    with M.ce_group(root, group, unit):
        w, _, _ = root._tul_half_weights(torch.zeros(1, root.bl, dtype=torch.long), None)
    return w


def test_the_three_group_masks_partition_the_full_weight_vector():
    """main + plast + emit must reconstruct the shipped weight vector EXACTLY.
    If they do not, the decomposition is not a decomposition."""
    root = _FakeRoot(pw=0.5, ew=0.5)
    full = root._orig(torch.zeros(1, root.bl, dtype=torch.long), None)[0]
    parts = sum(_weights(root, g) for g in ("main", "plast", "emit"))
    assert torch.allclose(parts, full), f"{parts} != {full}"


def test_masks_are_disjoint():
    root = _FakeRoot(pw=0.5, ew=0.5)
    ws = [_weights(root, g) for g in ("main", "plast", "emit")]
    for i in range(3):
        for j in range(i + 1, 3):
            assert float((ws[i] * ws[j]).sum()) == 0.0


def test_zero_weight_group_is_empty_at_configured_weight_and_present_at_unit():
    """emit_weight=0 (the v1a-2b arms) must give an EMPTY emit term — it really is
    absent from that objective — while the unit-weight pass still has a direction
    to measure. Getting this backwards would invent an objective the arm never had."""
    root = _FakeRoot(pw=0.5, ew=0.0)
    assert float(_weights(root, "emit").sum()) == 0.0
    unit = _weights(root, "emit", unit=True)
    assert float(unit.sum()) == 1.0                      # one real slot, one pad row
    assert float(unit[4]) == 1.0


def test_ce_group_restores_the_original_method_even_on_error():
    root = _FakeRoot(pw=0.5, ew=0.5)
    before = root._tul_half_weights
    with pytest.raises(ValueError):
        with M.ce_group(root, "not-a-group", False):
            root._tul_half_weights(torch.zeros(1, root.bl, dtype=torch.long), None)
    assert root._tul_half_weights is before


def test_full_group_is_a_no_op():
    root = _FakeRoot(pw=0.5, ew=0.5)
    before = root._tul_half_weights
    with M.ce_group(root, "full", False):
        assert root._tul_half_weights is before
