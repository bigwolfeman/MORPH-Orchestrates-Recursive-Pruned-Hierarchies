"""Measurement contract tests; the real CUDA smoke remains a separate gate."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location("huginn_sweep", Path(__file__).parents[1] / "lab/huginn/huginn_depth_sweep.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)


def test_target_offsets_include_first_token_after_boundary_and_final_target():
    from morph.model.tul_layout import BoundaryRule
    rule = BoundaryRule(np.zeros(10, dtype=bool), min_span=1, span_cap=3, eos_id=9)
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    y = torch.tensor([2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(sweep.target_offsets(x, y, rule), [1, 2, 0, 1, 2, 0])
    with pytest.raises(ValueError, match="next-token"):
        sweep.target_offsets(x, x, rule)


def test_paired_initialization_and_real_shifted_ce():
    draws = []
    class RandomModel:
        def __call__(self, input_ids, num_steps, **kwargs):
            noise = torch.randn((*input_ids.shape, 5))
            draws.append(noise.clone())
            return SimpleNamespace(logits=noise + num_steps * torch.arange(5))
    x, y = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])
    a = sweep.ce_map(RandomModel(), x, y, 1, "cpu", 9)
    b = sweep.ce_map(RandomModel(), x, y, 4, "cpu", 9)
    assert torch.equal(draws[0], draws[1])
    expected = torch.nn.functional.cross_entropy((draws[0] + torch.arange(5)).flatten(0, 1), y.flatten(), reduction="none")
    torch.testing.assert_close(a.flatten(), expected)
    assert not torch.equal(a, b)


def test_resume_preserves_counts_and_completed_profile():
    original = sweep.EarningProfile([1, 3], 2)
    for row in range(2):
        original.add(1, row, torch.tensor([2., 4.]), torch.ones(2, dtype=torch.bool), np.array([0, 1]))
    saved = original.to_json()
    restored = sweep.EarningProfile([1, 3], 2)
    sweep.restore_profile(restored, saved)
    for row in range(2):
        restored.add(3, row, torch.tensor([1., 2.]), torch.ones(2, dtype=torch.bool), np.array([0, 1]))
    np.testing.assert_array_equal(restored.row_n, [2, 2])
    np.testing.assert_array_equal(restored.row_ce[1], [6, 6])
    np.testing.assert_array_equal(restored.row_ce[3], [3, 3])
    np.testing.assert_array_equal(restored.bin_n.sum(1), [2, 2])


def test_adjacent_pairs_and_atomic_output(tmp_path):
    depths = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    rows = {str(d): [100. / d, 100. / d] for d in depths}
    result = sweep.comparisons(depths, rows, np.array([10, 10]), SimpleNamespace(n_boot=20, seed=0, level=.95))
    for a, b in zip(depths[:-1], depths[1:]):
        assert result[f"K{a}-K{b}"]["lo"] > 0
    path = tmp_path / "out.json"
    sweep.atomic_json(path, result)
    assert json.loads(path.read_text()) == result
    assert not path.with_suffix(".json.tmp").exists()
