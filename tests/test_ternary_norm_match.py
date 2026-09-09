"""The norm_match ternary scale mode: symmetric codes, a scale that makes the ternary
weight's norm equal the latent weight's, pure straight-through gradients."""
import torch

from morph.model.ternary_qat import TernarySTE


def _ste(mode, shape, group=0, **kw):
    w = torch.randn(*shape) * 0.02
    m = TernarySTE(threshold=0.5, weight_shape=shape, mode=mode, group=group, weight_init=w, **kw)
    return m, w


def test_codes_match_symmetric_and_norm_matches_latent():
    torch.manual_seed(0)
    m_sym, w = _ste("symmetric", (256, 128))
    m_nm = TernarySTE(threshold=0.5, weight_shape=(256, 128), mode="norm_match", weight_init=w)
    q_sym = m_sym(w).detach()
    q_nm = m_nm(w).detach()
    assert torch.equal(torch.sign(q_sym), torch.sign(q_nm)), "codes must be the symmetric codes"
    assert abs(q_nm.norm().item() / w.norm().item() - 1.0) < 1e-3, "ternary norm must equal the latent norm (fp16-encoded scale)"
    shrink = q_sym.norm().item() / w.norm().item()
    assert 0.55 < shrink < 0.75, f"the absmean rule shrinks a Gaussian layer to ~0.67, got {shrink:.3f}"
    assert q_nm.norm() > q_sym.norm()


def test_per_group_norm_match_holds_per_group_and_on_a_ragged_tail():
    torch.manual_seed(1)
    shape, gs = (200, 64), 64  # 3 full groups + a 8-row tail
    w = torch.randn(*shape) * 0.02
    m = TernarySTE(threshold=0.5, weight_shape=shape, mode="norm_match", group=gs, weight_init=w)
    q = m(w).detach()
    for start in range(0, shape[0], gs):
        wg, qg = w[start:start + gs], q[start:start + gs]
        assert abs(qg.norm().item() / wg.norm().item() - 1.0) < 1e-3


def test_straight_through_gradient_is_identity_and_no_parameters():
    torch.manual_seed(2)
    shape = (64, 32)
    w = torch.nn.Parameter(torch.randn(*shape) * 0.02)
    m = TernarySTE(threshold=0.5, weight_shape=shape, mode="norm_match", weight_init=w.detach())
    assert sum(p.numel() for p in m.parameters()) == 0
    out = m(w)
    g = torch.randn_like(out)
    out.backward(g)
    assert torch.equal(w.grad, g), "pure STE: d(effective)/d(w) is the identity"


def test_norm_match_refuses_scale_cap_and_ema():
    import pytest
    w = torch.randn(16, 8)
    with pytest.raises(AssertionError):
        TernarySTE(threshold=0.5, weight_shape=(16, 8), mode="norm_match", weight_init=w, scale_clip_mult=2.0)
    with pytest.raises(AssertionError):
        TernarySTE(threshold=0.5, weight_shape=(16, 8), mode="norm_match", weight_init=w, scale_ema_beta=0.9)


def test_ttq_mode_still_builds_and_has_two_scales_per_group():
    torch.manual_seed(3)
    w = torch.randn(64, 32) * 0.02
    m = TernarySTE(threshold=0.5, weight_shape=(64, 32), mode="ttq", weight_init=w)
    assert sum(p.numel() for p in m.parameters()) == 2
    q = m(w)
    assert q.shape == w.shape and torch.isfinite(q).all()


def test_ttq_scales_are_in_the_no_decay_group():
    from morph.training.optimizer import _NO_DECAY_KEYWORDS
    assert any(k in "parametrizations.weight.0.gamma_pos" for k in _NO_DECAY_KEYWORDS)
    assert any(k in "parametrizations.weight.0.gamma_neg" for k in _NO_DECAY_KEYWORDS)
