"""The diagonal carry span (model.injection_channels; arc E9).

"ctx" is the shipped model (256-wide carry on the context channel). "all" widens the
same carry to every dim with a deterministic per-dim init: the context dims keep their
legacy init, the rest start at (decay, 1 - decay) so the linear carry's fixed point is e.
Contracts: "ctx" builds the legacy parameter shapes; "all" with the widened dims set to
(decay 1, dt 0) reproduces the "ctx" forward bit for bit (the widening is a superset);
the "all" init has rho(A) < 1 on every dim and a scale-preserving fixed point.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import DiagonalInjection, MORPHConfig, MORPHTransformer

V = 64


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed=3, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg(**kw))
    m.eval()
    return m


def test_ctx_is_the_legacy_shape():
    m = _model(injection_channels="ctx")
    assert m.injection.log_A.shape == (20,) and m.injection.start == 32 and m.injection.end == 52


def test_all_widens_and_keeps_rho_below_one():
    m = _model(injection_channels="all", injection_all_decay=0.9)
    A = m.injection.log_A.exp()
    dt = m.injection.log_dt.exp()
    assert A.shape == (64,) and m.injection.start == 0 and m.injection.end == 64
    assert torch.all(A < 1.0)
    assert torch.allclose(A[32:52], torch.full((20,), 0.447)) and torch.allclose(dt[32:52], torch.ones(20))
    rest = torch.cat([A[:32], A[52:]])
    assert torch.allclose(rest, torch.full((44,), 0.9))
    # Fixed point of the linear carry h <- A h + dt e is e on the widened dims.
    e = torch.randn(64)
    h = torch.zeros(64)
    for _ in range(400):
        h = A * h + dt * e
    assert torch.allclose(h[:32], e[:32], atol=1e-4) and torch.allclose(h[52:], e[52:], atol=1e-4)


def test_all_is_a_superset_of_ctx():
    """The widened carry restricted to the context dims is the ctx-only carry bit for bit
    (same parameters there). On the widened dims A is clamped at 0.9999 by the class
    (rho < 1 by construction), so "identity on the rest" is impossible by design: with
    A = 1 -> 0.9999 and dt = 0 the whole-model loss agrees to 1e-4, not bit for bit."""
    ctx = DiagonalInjection(32, 52)
    wide = DiagonalInjection(0, 64, init_decay_vec=torch.full((64,), 0.5),
                             init_dt_vec=torch.full((64,), 0.5))
    with torch.no_grad():
        wide.log_A.fill_(0.0)
        wide.log_dt.fill_(float("-inf"))
        wide.log_A[32:52] = ctx.log_A
        wide.log_dt[32:52] = ctx.log_dt
    h = torch.randn(2, 5, 64)
    e = torch.randn(2, 5, 64)
    a, b = ctx(h, e), wide(h, e)
    assert torch.equal(a[..., 32:52], b[..., 32:52])
    assert torch.allclose(b[..., :32], 0.9999 * h[..., :32]) and torch.allclose(b[..., 52:], 0.9999 * h[..., 52:])
    m_ctx = _model(injection_channels="ctx")
    m_all = _model(injection_channels="all")
    sd = {k: v for k, v in m_ctx.state_dict().items() if not k.startswith("injection.")}
    missing, unexpected = m_all.load_state_dict(sd, strict=False)
    assert set(missing) == {"injection.log_A", "injection.log_dt"} and not unexpected
    with torch.no_grad():
        m_all.injection.log_A.fill_(0.0)
        m_all.injection.log_dt.fill_(float("-inf"))
        m_all.injection.log_A[32:52] = m_ctx.injection.log_A
        m_all.injection.log_dt[32:52] = m_ctx.injection.log_dt
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, V, (2, 32), generator=g)
    y = torch.randint(0, V, (2, 32), generator=g)
    la, lb = m_ctx(x, labels=y)["loss"], m_all(x, labels=y)["loss"]
    assert torch.allclose(la, lb, atol=1e-4), (la.item(), lb.item())


def test_bad_span_raises():
    with pytest.raises(ValueError):
        _model(injection_channels="half")


def test_diagonal_injection_vector_init_shapes():
    with pytest.raises(AssertionError):
        DiagonalInjection(0, 8, init_decay_vec=torch.ones(4), init_dt_vec=torch.ones(8))
