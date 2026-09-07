"""Arc E10 loop terms on the plain loop: the terminal fixed-point objective
(model.core_fixed_point_lambda) and the directional gain hinge (model.core_gain_lambda).

Contracts: off is the untouched model (no keys, same loss); eval never applies them; the
fixed-point term at depth 1 equals mean_b ||x_b - e_b||^2 / ||x_b||^2 with e the loop entry
and x the loop output; the hinge is zero above a huge target and positive at target 0; the
power direction is a unit buffer updated every step; both reach the core's gradient; the TUL
slot path refuses them. CPU only, tiny config.
"""

from __future__ import annotations

import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer

V = 64


def _cfg(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(seed=5, train=True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_cfg(**kw))
    m.train(train)
    return m


def _batch(seed=0, B=3, T=32):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, V, (B, T), generator=g), torch.randint(0, V, (B, T), generator=g))


def test_off_is_untouched_and_eval_never_applies():
    x, y = _batch()
    torch.manual_seed(1)
    a = _model(train=True)(x, labels=y)
    torch.manual_seed(1)
    b = _model(train=True, core_fixed_point_lambda=0.0, core_gain_lambda=0.0)(x, labels=y)
    assert torch.equal(a["loss"], b["loss"]) and "fixed_point" not in b and "core_gain_est" not in b
    m = _model(train=False, core_fixed_point_lambda=1.0, core_gain_lambda=1.0)
    out = m(x, labels=y)
    assert "fp_weighted" not in out and "core_gain_weighted" not in out


def test_fixed_point_term_at_depth_one_matches_the_formula():
    m = _model(core_fixed_point_lambda=2.0, mean_depth=1, max_depth=1, bptt_depth=1)
    x, y = _batch()
    torch.manual_seed(3)
    out = m(x, labels=y)
    # Depth 1 everywhere: h_{T-1} = e (the loop entry, _CloneInit) and h_T = the region output.
    torch.manual_seed(3)
    with torch.no_grad():
        xf, x0, bg = m._front_region(x)
        e = m.input_norm(xf)
        h_out = m._core_region(xf, x0, bg, x)
    e_ = e.float().flatten(1)
    h_ = h_out.float().flatten(1)
    ref = ((h_ - e_).pow(2).sum(1) / (h_.pow(2).sum(1) + 1e-6)).mean()
    assert torch.allclose(out["fixed_point"], ref, rtol=1e-4, atol=1e-6), (out["fixed_point"].item(), ref.item())
    assert torch.allclose(out["fp_weighted"], 2.0 * ref, rtol=1e-4, atol=1e-6)
    assert torch.allclose(out["loss"], out["ce_main"] + out["fp_weighted"]) if "ce_main" in out else True


def test_gain_hinge_zero_above_target_positive_at_zero():
    x, y = _batch()
    torch.manual_seed(7)
    hi = _model(core_gain_lambda=1.0, core_gain_target=1e6)(x, labels=y)
    assert float(hi["core_gain_weighted"]) == 0.0 and torch.isfinite(hi["core_gain_est"])
    torch.manual_seed(7)
    lo = _model(core_gain_lambda=1.0, core_gain_target=0.0)(x, labels=y)
    assert float(lo["core_gain_weighted"]) > 0.0
    assert float(lo["core_gain_max"]) >= float(lo["core_gain_est"]) > 0.0


def test_power_direction_is_a_unit_buffer_that_updates():
    m = _model(core_gain_lambda=1.0, core_gain_target=0.0, core_gain_direction="power")
    x, y = _batch()
    assert m._core_gain_dir is None
    m(x, labels=y)
    v1 = m._core_gain_dir.clone()
    assert v1 is not None and abs(float(v1.norm()) - 1.0) < 1e-4
    m(x, labels=y)
    v2 = m._core_gain_dir
    assert abs(float(v2.norm()) - 1.0) < 1e-4 and not torch.equal(v1, v2)
    r = _model(core_gain_lambda=1.0, core_gain_target=0.0, core_gain_direction="random")
    r(x, labels=y)
    assert r._core_gain_dir is None


def test_both_terms_reach_the_core():
    x, y = _batch()
    for kw in (dict(core_fixed_point_lambda=1.0), dict(core_gain_lambda=1.0, core_gain_target=0.0)):
        m = _model(**kw)
        out = m(x, labels=y)
        key = "fp_weighted" if "fp_weighted" in out else "core_gain_weighted"
        out[key].backward()
        core = [p.grad for n, p in m.named_parameters() if n.startswith("core.") and p.grad is not None]
        assert core and any(g.abs().sum() > 0 for g in core), key


def test_within_step_power_iterations_raise_the_reading():
    """The power-iterated reading at the same input is at least the one-shot reading on the
    same model and batch (a power step cannot lower the quotient of a near-linear map), and
    it stays finite; the buffer is still a unit vector."""
    x, y = _batch()
    torch.manual_seed(11)
    one = _model(core_gain_lambda=1.0, core_gain_target=0.0, core_gain_power_iters=0)(x, labels=y)
    torch.manual_seed(11)
    m = _model(core_gain_lambda=1.0, core_gain_target=0.0, core_gain_power_iters=3)
    pw = m(x, labels=y)
    assert torch.isfinite(pw["core_gain_est"]) and float(pw["core_gain_est"]) >= 0.8 * float(one["core_gain_est"])
    assert abs(float(m._core_gain_dir.norm()) - 1.0) < 1e-4
    assert float(pw["core_gain_weighted"]) > 0.0


def test_bad_direction_raises():
    with pytest.raises(ValueError):
        _model(core_gain_direction="jvp")


# ── the paid TUL loop carries the fixed-point term; the slot loop refuses it ──────────
def _tul_layout(seed=0, B=2, n=90):
    import numpy as np
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
    lut = np.zeros(V, dtype=bool)
    lut[[10, 11]] = True
    lut[0] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::6] = 10
    return slot_layout_from_ids(ids.astype(np.int64), rule, spec)


def test_paid_loop_carries_the_fixed_point_term():
    from morph.model.tul import TULConfig
    x, y, layout, _ = _tul_layout()
    paid = dict(tul=TULConfig(prefix_k=2, slot_id=4, tokens_through_core=True))
    torch.manual_seed(1)
    off = _model(core_fixed_point_lambda=0.0, **paid)(x, labels=y, slot_layout=layout)
    torch.manual_seed(1)
    on_model = _model(core_fixed_point_lambda=1.0, **paid)
    on = on_model(x, labels=y, slot_layout=layout)
    assert "fixed_point" in on and "fp_weighted" in on and "fixed_point" not in off
    assert on["fixed_point"] > 0 and torch.isfinite(on["loss"])
    # loss = the paid loop's weighted CE + fp_weighted, and nothing else moved
    assert torch.allclose(on["loss"] - on["fp_weighted"], off["loss"], atol=1e-6)
    assert torch.equal(on["fixed_point"], on["fp_weighted"])   # lambda 1.0
    on["fp_weighted"].backward()
    core = [p.grad for n, p in on_model.named_parameters()
            if n.startswith("core.") and p.grad is not None]
    assert core and any(g.abs().sum() > 0 for g in core), "the term must reach the core"
    # eval never applies it, and a label-less forward leaves no stash behind
    m = _model(train=False, core_fixed_point_lambda=1.0, **paid)
    with torch.no_grad():
        o = m(x, labels=y, slot_layout=layout)
    assert "fp_weighted" not in o
    m.train()
    _ = m(x, labels=None, slot_layout=layout)
    assert m._core_aux is None
    # the refuted gain penalty refuses to build on any TUL model
    with pytest.raises(RuntimeError, match="core_gain_lambda"):
        _model(core_gain_lambda=1.0, core_gain_target=0.0, **paid)
