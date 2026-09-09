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


# ── Parcae's loop entry, faithful (2026-09-09): learned B on the injected e, uniform
# all-dim init, and the noise state init ─────────────────────────────────────────────


def test_B_identity_init_reproduces_the_forward_bit_for_bit():
    torch.manual_seed(5)
    e = torch.randn(2, 7, 64)
    h = torch.randn(2, 7, 64)
    a = DiagonalInjection(0, 64, init_decay_vec=torch.full((64,), 0.447),
                          init_dt_vec=torch.full((64,), 0.8))
    b = DiagonalInjection(0, 64, init_decay_vec=torch.full((64,), 0.447),
                          init_dt_vec=torch.full((64,), 0.8), use_B=True)
    assert b.B is not None and torch.equal(b.B, torch.eye(64)) and a.B is None
    assert torch.equal(a(h, e), b(h, e))
    with torch.no_grad():
        b.B.mul_(2.0)
    assert torch.allclose(b(h, e), 0.447 * h + 0.8 * 2.0 * e, atol=1e-5)


def test_all_with_uniform_dt_sets_every_dim_including_ctx():
    m = _model(injection_channels="all", injection_all_decay=0.447, injection_all_dt=0.8,
               injection_B=True)
    A, dt = m.injection.log_A.exp(), m.injection.log_dt.exp()
    assert torch.allclose(A, torch.full((64,), 0.447)) and torch.allclose(dt, torch.full((64,), 0.8))
    assert m.injection.B.shape == (64, 64) and torch.equal(m.injection.B, torch.eye(64))


def test_B_and_noise_init_add_no_decayed_and_no_ternary_parameters():
    from morph.training.optimizer import _split_by_decay
    m = _model(injection_channels="all", injection_all_decay=0.447, injection_all_dt=0.8,
               injection_B=True, core_state_init="noise")
    _, _, decay_n, no_decay_n = _split_by_decay(m)
    assert "injection.B" in no_decay_n and "injection.B" not in decay_n
    assert not any(n.startswith("core_init") for n, _ in m.named_parameters())


def test_noise_state_init_is_small_noise_and_the_forward_runs():
    m = _model(injection_channels="all", injection_all_decay=0.447, injection_all_dt=0.8,
               injection_B=True, core_state_init="noise", core_state_init_std=0.02)
    e = torch.randn(3, 11, 64)
    torch.manual_seed(1)
    h0 = m.core_init(e)
    assert h0.shape == e.shape and abs(float(h0.std()) - 0.02) < 0.005
    assert not torch.allclose(h0, e)
    torch.manual_seed(1)
    assert torch.equal(m.core_init(e), h0)  # seed-deterministic, e-independent
    x = torch.randint(0, V, (2, 32))
    out = m(x, labels=x)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0
    out["loss"].backward()
    assert m.injection.B.grad is not None and torch.isfinite(m.injection.B.grad).all()


def test_noise_state_init_requires_the_all_dim_carry():
    with pytest.raises(ValueError, match="requires model.injection_channels='all'"):
        _model(injection_channels="ctx", core_state_init="noise")
    with pytest.raises(ValueError, match="core_state_init must be"):
        _model(core_state_init="zeros")


def test_prelude_state_init_is_the_shipped_entry():
    m = _model()
    e = torch.randn(2, 5, 64)
    assert torch.equal(m.core_init(e), e) and m.injection.B is None


def test_train_config_mapping_carries_the_loop_entry_keys():
    """train.py maps model keys one by one; a key it does not name is dropped silently."""
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    import os
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "morph", "configs"))
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        c = compose(config_name="notul_e19_loop")
        loop = build_morph_config(c)
        c = compose(config_name="notul_e18")
        plain = build_morph_config(c)
    assert (loop.injection_channels, loop.injection_all_dt, loop.injection_B,
            loop.core_state_init, loop.core_fixed_point_lambda) == ("all", 0.8, True, "noise", 0.0)
    assert (plain.injection_channels, plain.injection_all_dt, plain.injection_B,
            plain.core_state_init) == ("ctx", None, False, "prelude")


def test_e20_configs_compose_their_one_factor_and_keep_the_e19_entry():
    """The three E20 arms (notul_e20_*.yaml) each change exactly ONE factor on top of
    notul_e19_loop; every other E19 loop-entry key must still be there. ternary_scope,
    injection_lr_mult and grad_probe_every are training.* keys `build_morph_config` never
    touches (it maps model.* one by one — see the test above), so they are read straight
    off the composed Hydra config, not through MORPHConfig.
    """
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    import os
    cfg_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "morph", "configs"))

    def _compose(name):
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            return compose(config_name=name)

    def _assert_carries_e19_entry(c, m):
        assert m.core_state_init == "noise"
        assert m.injection_B is True
        assert m.core_fixed_point_lambda == 0.0
        assert float(c.training.grad_probe_every) == 1.0

    c = _compose("notul_e20_dense_core")
    m = build_morph_config(c)
    _assert_carries_e19_entry(c, m)
    assert str(c.training.ternary_scope) == "backbone_no_core"
    # the other two arms' factors are UNCHANGED from the E19 loop entry on this arm.
    assert float(getattr(c.training, "injection_lr_mult", 1.0)) == 1.0
    assert (m.mean_depth, m.max_depth, m.bptt_depth) == (6, 8, 8)

    c = _compose("notul_e20_carry_lr20x")
    m = build_morph_config(c)
    _assert_carries_e19_entry(c, m)
    assert float(c.training.injection_lr_mult) == 20.0
    assert str(getattr(c.training, "ternary_scope", "backbone")) == "backbone"
    assert (m.mean_depth, m.max_depth, m.bptt_depth) == (6, 8, 8)

    c = _compose("notul_e20_draw8_bptt4")
    m = build_morph_config(c)
    _assert_carries_e19_entry(c, m)
    assert (m.mean_depth, m.max_depth, m.bptt_depth) == (8, 12, 4)
    assert str(getattr(c.training, "ternary_scope", "backbone")) == "backbone"
    assert float(getattr(c.training, "injection_lr_mult", 1.0)) == 1.0
