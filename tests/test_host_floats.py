"""`train._host_floats` (2026-10-04 speed pass): the per-step probe reads its device
scalars in ONE copy. The values must be exactly `float(x)` of each, in order, for every
dtype the probe sees, with Python numbers passed through and multi-element tensors
flattened in `tolist()` order."""
from __future__ import annotations

import pytest
import torch

from morph.training.train import _host_floats


def _cases(device):
    g = torch.Generator().manual_seed(0)
    xs = [torch.randn((), generator=g).to(device, d) for d in
          (torch.float32, torch.bfloat16, torch.float16)]
    xs += [3.25, 7, torch.tensor(1e-30).to(device), torch.tensor(float("inf")).to(device)]
    xs += [torch.randn(5, generator=g).to(device), torch.randn((), generator=g).to(device)]
    return xs


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_one_copy_is_float_of_each(device):
    xs = _cases(device)
    want = []
    for x in xs:
        if torch.is_tensor(x) and x.numel() > 1:
            want.extend(x.float().tolist())
        else:
            want.append(float(x))
    got = _host_floats(xs)
    assert got == want
    assert all(type(v) is float for v in got)
    assert _host_floats([]) == [] and _host_floats([2]) == [2.0]
