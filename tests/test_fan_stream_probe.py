"""``lab/divergence/fan_stream_probe.py::stream_geometry`` — the shapes it must tell apart.

The two trainer scalars (raw mean pairwise cosine, centered participation rank) read
−1/3 and 1 for a two-against-two antipodal split and −1/3 and 3 for a regular simplex;
the probe adds the identity of the split, the norms, the shared part and whether the
one direction is a global axis. Each test builds the shape and checks the reading.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))
from fan_stream_probe import stream_geometry  # noqa: E402

N, M, C = 64, 4, 96


def _rand(n=N, c=C, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, c, generator=g)


def test_antipodal_split_per_slot_axis():
    a = _rand()                                                      # a different line per slot
    cells = torch.stack([a, a, -a, -a], dim=1)                       # [N, 4, C]
    g = stream_geometry(cells)
    assert abs(g["cos_mean"] + 1.0 / 3.0) < 1e-6
    assert abs(g["eff_rank"] - 1.0) < 1e-6
    assert max(g["sv_ratio"]) < 1e-6
    assert g["shared_frac"] < 1e-6
    assert g["top_pattern"] == "++--" and g["top_pattern_frac"] == 1.0
    assert g["axis_cos"] < 0.5                                       # random lines in 96 dims
    cm = torch.tensor(g["cos_matrix"])
    assert torch.allclose(cm[0, 1], torch.tensor(1.0, dtype=cm.dtype))
    assert torch.allclose(cm[0, 2], torch.tensor(-1.0, dtype=cm.dtype))


def test_antipodal_split_global_axis_and_unequal_norms():
    u = torch.nn.functional.normalize(_rand(1, C, seed=3), dim=-1).expand(N, C)
    scale = torch.tensor([3.0, 1.0, 2.0, 0.5]).view(1, M, 1)
    sign = torch.tensor([1.0, 1.0, -1.0, -1.0]).view(1, M, 1)
    cells = u.unsqueeze(1) * scale * sign
    g = stream_geometry(cells)
    assert abs(g["cos_mean"] + 1.0 / 3.0) < 1e-6                     # cosine is sign-only
    assert abs(g["eff_rank"] - 1.0) < 1e-6
    assert g["axis_cos"] > 1 - 1e-6                                  # one axis for every slot
    assert g["top_pattern"] == "++--" and g["top_pattern_frac"] == 1.0
    assert [round(v, 3) for v in g["norms"]] == [3.0, 1.0, 2.0, 0.5]
    # unequal magnitudes leave a shared part: mean = (3 + 1 - 2 - 0.5)/4 u = 0.375 u,
    # mean norm 1.625 -> 0.2308
    assert abs(g["shared_frac"] - 0.375 / 1.625) < 1e-6


def test_regular_simplex_is_rank_three():
    # four unit vectors with pairwise cosine -1/3: the vertices of a tetrahedron, embedded
    # in a random 3-dim subspace per slot.
    v = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float)
    v = v / v.norm(dim=-1, keepdim=True)                             # [4, 3]
    basis = torch.linalg.qr(_rand(N * C, 3, seed=5).reshape(N, C, 3))[0]  # [N, C, 3]
    cells = torch.einsum("mk,nck->nmc", v, basis)
    g = stream_geometry(cells)
    assert abs(g["cos_mean"] + 1.0 / 3.0) < 1e-5
    assert abs(g["eff_rank"] - 3.0) < 1e-5
    assert all(abs(r - 1.0) < 1e-5 for r in g["sv_ratio"][:2]) and g["sv_ratio"][2] < 1e-5
    assert g["shared_frac"] < 1e-5


def test_copies_with_noise_read_as_copies():
    a = _rand()
    noise = 0.01 * _rand(seed=9).unsqueeze(1) * torch.ones(1, M, 1)
    noise = noise + 0.01 * torch.randn(N, M, C, generator=torch.Generator().manual_seed(11))
    cells = a.unsqueeze(1).expand(N, M, C) + noise
    g = stream_geometry(cells)
    assert g["cos_mean"] > 0.99
    assert g["shared_frac"] > 0.99
    assert g["eff_rank"] > 2.0                                       # the noise is isotropic


def test_shape_checks():
    with pytest.raises(ValueError):
        stream_geometry(torch.zeros(3, 8))
    with pytest.raises(ValueError):
        stream_geometry(torch.zeros(3, 1, 8))
