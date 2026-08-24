"""The Lorentz log map is the hyperbolic channel's normalisation. It must be exact.

On the hyperboloid x0 = sqrt(1 + ||xs||^2), so acosh(x0) == asinh(||xs||) exactly and the
tangent vector is asinh(||xs||)/||xs|| * xs. The original implementation went through
acosh(x0) and sqrt(x0^2 - 1), which is catastrophic cancellation for small ||xs||, and its
guard against the resulting singularity was unreachable: it tested denom < 1e-4 while denom
was floored at sqrt(_EPS) = 1e-3. The coefficient was therefore a constant 1.3811 for
||xs|| below about 2e-3, instead of the correct limit 1.0.

These tests fail on that implementation and pass on the asinh one.
"""

from __future__ import annotations

import math

import pytest
import torch

from morph.model.embeddings import (_log_map_origin, _project_to_hyperboloid,
                                    LorentzEmbedding)


def _coeff(n: float, dtype=torch.float64) -> float:
    """Realised coefficient ||log_map(s)|| / ||s|| for a spatial vector of norm n."""
    s = torch.zeros(1, 8, dtype=dtype)
    s[0, 0] = n
    return (_log_map_origin(_project_to_hyperboloid(s)).norm() / n).item()


@pytest.mark.parametrize("n", [1e-8, 1e-6, 1e-4, 1e-3, 2e-3, 1e-2, 0.08, 1.0, 10.0])
def test_the_coefficient_is_asinh_over_norm(n):
    """The whole contract, at every scale including the one that used to be wrong."""
    assert _coeff(n) == pytest.approx(math.asinh(n) / n, rel=1e-6)


def test_the_small_norm_limit_is_one_not_1_38():
    """The specific regression: the dead guard left a 38 % error near the origin."""
    for n in (1e-8, 1e-6, 1e-4, 1e-3):
        assert _coeff(n) == pytest.approx(1.0, abs=1e-6), f"broken at ||s||={n}"


def test_there_is_no_discontinuity_across_the_old_guard_boundary():
    """The old code jumped from 1.3811 to 1.0 at ||s|| ~ 2e-3. Sweep across it and require
    the coefficient to be monotone and smooth."""
    ns = [1e-4 * (1.3 ** k) for k in range(30)]          # ~1e-4 .. 2e-2
    cs = [_coeff(n) for n in ns]
    assert all(b <= a + 1e-9 for a, b in zip(cs, cs[1:])), "coefficient is not monotone"
    steps = [abs(b - a) for a, b in zip(cs, cs[1:])]
    assert max(steps) < 0.02, f"discontinuity of {max(steps):.3f} across the sweep"


def test_it_compresses_norms_logarithmically():
    """Why this map IS the normalisation: output norm grows like ln(input norm), so a
    runaway in the raw table cannot produce a runaway embedding."""
    for n in (1e3, 1e4, 1e5):
        out = _coeff(n) * n
        assert out == pytest.approx(math.asinh(n), rel=1e-6)
        assert out < math.log(2 * n) + 1e-6


def test_zero_input_is_finite_and_zero():
    s = torch.zeros(2, 8, dtype=torch.float64)
    out = _log_map_origin(_project_to_hyperboloid(s))
    assert torch.isfinite(out).all()
    assert torch.equal(out, torch.zeros_like(out))


def test_the_gradient_is_finite_at_and_near_the_origin():
    """A kink at the origin would put a discontinuous gradient on the embedding table."""
    for n in (0.0, 1e-8, 1e-4, 1e-2):
        s = torch.zeros(1, 8, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            s[0, 0] = n
        out = _log_map_origin(_project_to_hyperboloid(s))
        g = torch.autograd.grad(out.square().sum(), s)[0]
        assert torch.isfinite(g).all(), f"non-finite gradient at ||s||={n}"


def test_direction_is_preserved():
    """The map scales, it must never rotate."""
    torch.manual_seed(0)
    s = torch.randn(4, 16, dtype=torch.float64) * 0.3
    out = _log_map_origin(_project_to_hyperboloid(s))
    cos = torch.nn.functional.cosine_similarity(s, out, dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-9)


def test_the_embedding_module_still_produces_the_documented_shape():
    m = LorentzEmbedding(32, 8)
    out = m(torch.randint(0, 32, (2, 5)))
    assert out.shape == (2, 5, 8)
    assert torch.isfinite(out).all()
