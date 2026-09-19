"""LXTUL's fan (``tul.fan_k``) — K latent streams per span through the ONE shared core.

WHAT THIS FILE HAS TO PROVE, in the order it matters:

1. OFF IS NOTHING. ``fan_k: 0`` is bit-identical to the tree before the key existed.
   Pinned against the numbers ``tests/test_tul_prefix_source.py`` already carries for the
   strict ruler at ``prefix_k 4`` — the same fixture, the same seed, measured before this
   change — and the parameter NAMES are compared set-to-set against a config built with
   no fan key at all.
2. THE REPULSION IS THE TERM THE DOCSTRING SAYS. 1.0 for identical streams, exactly 0 for
   orthogonal ones, charged over the FIRST ``fan_repel_passes`` passes ONLY (the later
   passes are still measured and reported but carry no gradient), and blind to pad slots.
3. THE MIXTURE REDUCES TO THE MEAN at equal logits, and a one-hot gate reproduces feeding
   that stream ALONE — checked end to end against the oracle's own single-stream replay,
   which is the only way to know the two paths write the same thing.
4. THE ORACLE IS A PER-SPAN MINIMUM, token-weighted. Verified by capturing the per-stream
   span tables the instrument builds and recomputing the headline from them, so a wrong
   axis, a lost mask or a plain mean instead of a token-weighted one all fail.
   NOTE, because the brief said otherwise: ``oracle_ce <= single_ce`` is true BY
   CONSTRUCTION (a minimum over K including stream 0). ``oracle_ce <= mixed_ce`` is NOT —
   a convex combination of K streams can beat every one of them — so this file asserts
   the first and only reports the second.
5. AN UNKNOWN ``tul.fan_*`` KEY STILL RAISES through ``tul_setup`` (KNOWN_TUL_KEYS).

CPU only, fp32, ``use_kernels=False``, the ``tests/test_tul_prefix_source.py`` fixtures.
fp32 and not fp64: ``_window_fallback`` hands SDPA an fp32 mask whatever the dtype of q,
which is silently WRONG at fp64 (morph/model/CLAUDE.md).

Record: lab/experiments/planned/2026-09-19-lxtul-fan4.md
Note: .agents/notes/proposed/architecture/2026-09-19-lxtul-fan-streams.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import morph.model.transformer as tf_mod
from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_fan import TULFanMix, fan_repel_term, fan_stream_cos
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.training.tul_setup import reject_unknown_tul_keys

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=2, max_depth=3, bptt_depth=3,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _spec(prefix_k: int = 4) -> TulLayoutSpec:
    return TulLayoutSpec(seq_len=64, prefix_k=prefix_k, max_slots=10, slot_id=4)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=4, slot_id=4, tg_restrict=True, tg_restrict_scope="all",
                tg_geometry="strict", emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0)
    base.update(kw)
    return TULConfig(**base)


def _model(seed: int = 1234, **tul_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.eval().float()


def _batch(prefix_k: int = 4, seed: int = 0):
    ids = _ids(seed=seed)
    inp, lab, layout, _ = slot_layout_from_ids(ids, _rule(), _spec(prefix_k))
    return ids, inp, lab, layout


def _finite_logit_sum(lg: torch.Tensor) -> float:
    return float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

# The strict ruler at prefix_k 4, pinned in tests/test_tul_prefix_source.py
# (`test_exit_is_pinned_to_the_pre_knob_values`, case "strict_k4") against a worktree of
# `d778845` on 2026-09-13 — i.e. measured on the tree BEFORE `tul.fan_k` existed.
HEAD_K4_LOSS = 5.020976543426514
HEAD_K4_LOGIT_SUM = 1136.2373420511503
HEAD_K4_GRAD_SUM = 962.8389480001819
HEAD_K4_KEYS = 211


def test_fan_k_zero_is_the_default_and_builds_nothing():
    m = _model()
    assert m.cfg.tul.fan_k == 0
    assert m.tul_fan is None
    assert m.tul_register is None
    assert not any("tul_fan" in k for k in m.state_dict())


def test_fan_k_zero_is_bit_identical_to_the_pre_key_tree():
    """Loss, logits, total gradient and the state-dict size, to the last printed digit."""
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=0).train()
    torch.manual_seed(7)
    res = m(inp, labels=lab, slot_layout=layout)
    res["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    assert res["loss"].item() == HEAD_K4_LOSS
    assert _finite_logit_sum(lg) == HEAD_K4_LOGIT_SUM
    assert g == HEAD_K4_GRAD_SUM
    assert len(m.state_dict()) == HEAD_K4_KEYS


def test_fan_k_zero_has_the_parameter_names_of_a_config_with_no_fan_key():
    a = set(_model().state_dict())
    b = set(_model(fan_k=0).state_dict())
    assert a == b


def test_fan_builds_the_register_and_the_mixture():
    m = _model(fan_k=4, slot_cells=4, fan_mix="softmax")
    assert m.tul_fan is not None and m.tul_fan.k == 4
    assert m.tul_register is not None and m.tul_register.m == 4
    assert any(k.startswith("tul_fan.gate") for k in m.state_dict())


def test_fan_mean_builds_no_gate_parameter():
    m = _model(fan_k=4, slot_cells=4, fan_mix="mean")
    assert m.tul_fan is not None and m.tul_fan.gate is None
    assert not any(k.startswith("tul_fan.") for k in m.state_dict())


def test_fan_config_refusals():
    with pytest.raises(ValueError, match="fan_k=1"):
        _tul(fan_k=1, slot_cells=1)
    with pytest.raises(ValueError, match="slot_cells"):
        _tul(fan_k=4, slot_cells=2)
    with pytest.raises(ValueError, match="fan_k=0"):
        _tul(fan_repel_lambda=0.5)
    with pytest.raises(ValueError, match="fan_k=0"):
        _tul(fan_mix="softmax")
    with pytest.raises(ValueError, match="fan_mix"):
        _tul(fan_k=4, slot_cells=4, fan_mix="argmax")


# ── 2. THE REPULSION ─────────────────────────────────────────────────────────

def _state(streams: torch.Tensor) -> torch.Tensor:
    """``[S, M, C]`` -> the compact CELL axis ``[1, S*M, C]`` ``_tul_core`` carries."""
    s, m, c = streams.shape
    return streams.reshape(1, s * m, c)


def test_repel_is_one_for_identical_streams_and_zero_for_orthogonal_ones():
    c = 8
    same = torch.ones(2, 4, c)
    valid = torch.ones(1, 2, dtype=torch.bool)
    assert fan_stream_cos(_state(same), valid, 4).item() == pytest.approx(1.0, abs=1e-6)
    orth = torch.eye(4, c).unsqueeze(0).expand(2, 4, c).contiguous()
    assert fan_stream_cos(_state(orth), valid, 4).item() == pytest.approx(0.0, abs=1e-6)
    # And it is a real gradient path, not a constant: pulling one stream toward another
    # must RAISE the term.
    near = orth.clone()
    near[:, 1] = 0.9 * orth[:, 0] + 0.1 * orth[:, 1]
    assert fan_stream_cos(_state(near), valid, 4).item() > 0.05


def test_repel_ignores_pad_slots():
    c = 8
    st = torch.eye(4, c).unsqueeze(0).expand(2, 4, c).contiguous().clone()
    st[1] = 1.0                                   # slot 1 fully collapsed
    valid = torch.tensor([[True, False]])
    assert fan_stream_cos(_state(st), valid, 4).item() == pytest.approx(0.0, abs=1e-6)
    valid_both = torch.tensor([[True, True]])
    assert fan_stream_cos(_state(st), valid_both, 4).item() == pytest.approx(0.5, abs=1e-6)


def test_repel_charges_the_first_passes_only_and_reports_every_pass():
    c = 8
    valid = torch.ones(1, 2, dtype=torch.bool)
    # Four trajectory entries: the seed and three passes, each with its own geometry so a
    # term that read the wrong pass produces a different number.
    traj = []
    for t, mix in enumerate((0.0, 0.25, 0.5, 1.0)):
        base = torch.eye(4, c).unsqueeze(0).expand(2, 4, c).contiguous().clone()
        st = (1.0 - mix) * base + mix * torch.ones_like(base)
        traj.append(_state(st).requires_grad_(True))
    stats: dict = {}
    term = fan_repel_term(traj, valid, 4, n_passes=2, stats=stats)
    # every pass reported, the seed included
    assert sorted(stats) == ["repel_terms", "stream_cos_t0", "stream_cos_t1",
                             "stream_cos_t2", "stream_cos_t3"]
    assert stats["repel_terms"] == 2.0
    assert stats["stream_cos_t0"] == pytest.approx(0.0, abs=1e-6)
    assert stats["stream_cos_t3"] == pytest.approx(1.0, abs=1e-6)
    assert term.item() == pytest.approx(
        0.5 * (stats["stream_cos_t1"] + stats["stream_cos_t2"]), abs=1e-6)
    # ONLY the penalised passes carry gradient. `allow_unused` so an unreached entry
    # comes back None rather than raising: that None IS the assertion.
    gs = torch.autograd.grad(term, traj, allow_unused=True)
    assert gs[0] is None, "the SEED state must not be repelled"
    assert gs[1] is not None and gs[2] is not None
    assert gs[3] is None, "pass 3 is past fan_repel_passes=2 and must carry no gradient"


def test_repel_term_reaches_the_loss_and_the_loop(tmp_path):
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="mean",
               fan_repel_lambda=0.5, fan_repel_passes=2).train()
    torch.manual_seed(7)
    out = m(inp, labels=lab, slot_layout=layout)
    assert "fan_repel" in out and "fan_repel_weighted" in out
    assert torch.isfinite(out["fan_repel"])
    assert float(out["fan_repel_weighted"]) == pytest.approx(
        0.5 * float(out["fan_repel"]), rel=1e-6)
    # the term must be INSIDE the reported loss and reach the register's seed projection
    out["loss"].backward()
    assert m.tul_register.W_o.weight.grad is not None
    assert float(m.tul_register.W_o.weight.grad.abs().sum()) > 0.0


def test_repel_lambda_zero_adds_no_term_but_still_reports_the_cosines():
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="mean", fan_repel_lambda=0.0).train()
    torch.manual_seed(7)
    out = m(inp, labels=lab, slot_layout=layout)
    assert "fan_repel" not in out
    assert any(str(k).startswith("fan_stream_cos_t") for k in out)


# ── 3. THE MIXTURE ───────────────────────────────────────────────────────────

def test_softmax_mixture_at_equal_logits_is_the_mean():
    torch.manual_seed(0)
    mix = TULFanMix(8, 4, "softmax")
    cells = torch.randn(2, 3, 4, 8)
    mixed, w = mix(cells)
    assert torch.allclose(w, torch.full_like(w, 0.25))
    assert torch.allclose(mixed, cells.mean(dim=2), atol=1e-6)
    # ...and it stays the mean whenever the logits are equal but NOT zero: give the gate
    # a real direction and make every stream score the same on it.
    with torch.no_grad():
        mix.gate.weight.copy_(torch.arange(8, dtype=torch.float32).view(1, 8))
    flat = torch.randn(2, 3, 1, 8).expand(2, 3, 4, 8).contiguous()
    mixed2, w2 = mix(flat)
    assert torch.allclose(w2, torch.full_like(w2, 0.25), atol=1e-6)
    assert torch.allclose(mixed2, flat.mean(dim=2), atol=1e-6)


def test_one_hot_gate_returns_that_stream_alone():
    mix = TULFanMix(8, 4, "softmax")
    with torch.no_grad():
        mix.gate.weight.copy_(torch.tensor([[50.0] + [0.0] * 7]))
    cells = torch.randn(2, 3, 4, 8)
    cells[:, :, :, 0] = -1.0
    cells[:, :, 2, 0] = 1.0                       # stream 2 wins the gate everywhere
    mixed, w = mix(cells)
    assert torch.allclose(w[..., 2], torch.ones_like(w[..., 2]), atol=1e-6)
    assert torch.allclose(mixed, cells[:, :, 2], atol=1e-5)


def test_mix_entropy_is_ln_k_at_the_mean_and_zero_at_one_hot():
    valid = torch.ones(2, 3, dtype=torch.bool)
    w = torch.full((2, 3, 4), 0.25)
    assert TULFanMix.entropy(w, valid).item() == pytest.approx(float(np.log(4)), abs=1e-6)
    hot = torch.zeros(2, 3, 4)
    hot[..., 1] = 1.0
    assert TULFanMix.entropy(hot, valid).item() == pytest.approx(0.0, abs=1e-6)


def test_model_reports_mix_entropy():
    _ids0, inp, lab, layout = _batch(4)
    for mode, want in (("mean", float(np.log(4))), ("softmax", float(np.log(4)))):
        m = _model(fan_k=4, slot_cells=4, fan_mix=mode).train()
        torch.manual_seed(7)
        out = m(inp, labels=lab, slot_layout=layout)
        assert float(out["fan_mix_entropy"]) == pytest.approx(want, abs=1e-5), mode


def test_softmax_at_init_equals_the_mean_arm_exactly():
    """The gate is zero-init, so the two arms are ONE factor after step 0 and not before."""
    _ids0, inp, lab, layout = _batch(4)
    outs = []
    for mode in ("mean", "softmax"):
        m = _model(fan_k=4, slot_cells=4, fan_mix=mode).train()
        torch.manual_seed(7)
        outs.append(float(m(inp, labels=lab, slot_layout=layout)["loss"].detach()))
    assert outs[0] == outs[1]


# ── 4. THE ORACLE ────────────────────────────────────────────────────────────

def _eval_with_oracle(m, inp, lab, layout):
    m.eval()
    with torch.no_grad():
        return m(inp, labels=lab, slot_layout=layout)


def test_oracle_is_the_token_weighted_per_span_minimum():
    """Recomputed from the per-stream span tables the instrument itself builds.

    `accumulate_span_ce` is wrapped so the test sees exactly the K tables that went into
    the minimum, then the headline is rebuilt from them with the documented formula. A
    wrong axis, a dropped validity mask or a plain mean instead of a token-weighted sum
    all fail here.
    """
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="softmax")
    # break the tie the zero-init gate and the zero-init register start at, so the four
    # streams are genuinely different objects
    with torch.no_grad():
        m.tul_register.W_o.weight.normal_(0.0, 0.3)
        m.tul_register.P_cell.normal_(0.0, 0.3)
    seen: list[torch.Tensor] = []
    real = tf_mod.accumulate_span_ce

    def spy(*a, **k):
        t = real(*a, **k)
        seen.append(t.detach().clone())
        return t

    tf_mod.accumulate_span_ce = spy
    try:
        out = _eval_with_oracle(m, inp, lab, layout)
    finally:
        tf_mod.accumulate_span_ce = real
    assert len(seen) == 5, "four single-stream replays plus the shipped write"
    gid, keep_tok, _lab, g_bins = tf_mod.span_ce_index(lab, layout)
    n_tok = tf_mod.span_token_counts(gid, keep_tok, g_bins)[:, 1:]
    ok = layout.slot_valid & (n_tok > 0)
    ce = torch.stack([t[:, 1:] for t in seen[:4]], dim=-1)
    denom = n_tok[ok].sum()
    want_oracle = float(ce.min(dim=-1).values[ok].sum() / denom)
    want_single = float(ce[..., 0][ok].sum() / denom)
    want_mixed = float(seen[4][:, 1:][ok].sum() / denom)
    assert float(out["fan_oracle_ce"]) == pytest.approx(want_oracle, rel=1e-6)
    assert float(out["fan_single_ce"]) == pytest.approx(want_single, rel=1e-6)
    assert float(out["fan_mixed_ce"]) == pytest.approx(want_mixed, rel=1e-6)
    assert float(out["fan_oracle_n_tokens"]) == pytest.approx(float(denom), rel=1e-6)
    # by construction: a minimum over K that includes stream 0
    assert float(out["fan_oracle_ce"]) <= float(out["fan_single_ce"]) + 1e-9
    for i in range(4):
        assert float(out["fan_oracle_ce"]) <= float(out[f"fan_stream_ce_k{i}"]) + 1e-9
    assert float(out["fan_stream_ce_k0"]) == pytest.approx(
        float(out["fan_single_ce"]), rel=1e-9)
    # the streams must actually differ, or the test above is vacuous
    spread = max(float(out[f"fan_stream_ce_k{i}"]) for i in range(4)) - \
        min(float(out[f"fan_stream_ce_k{i}"]) for i in range(4))
    assert spread > 1e-6, "four identical streams make the oracle test vacuous"
    assert float(out["fan_oracle_ce"]) < float(out["fan_single_ce"])


def test_one_hot_gate_makes_the_shipped_write_equal_that_stream_alone():
    """The end-to-end version of the one-hot claim: the mixture path and the oracle's
    single-stream replay must write the SAME thing, or one of the two is lying."""
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="softmax")
    with torch.no_grad():
        m.tul_register.W_o.weight.normal_(0.0, 0.3)
        m.tul_register.P_cell.normal_(0.0, 0.3)
        # P_cell[j] dominates the gate's direction, so stream j wins at every slot
        m.tul_fan.gate.weight.zero_()
        m.tul_fan.gate.weight[0, 0] = 200.0
        m.tul_register.P_cell[:, 0] = -1.0
        m.tul_register.P_cell[2, 0] = 1.0
    out = _eval_with_oracle(m, inp, lab, layout)
    assert float(out["fan_mix_w_max"]) == pytest.approx(1.0, abs=1e-5)
    assert float(out["fan_mixed_ce"]) == pytest.approx(
        float(out["fan_stream_ce_k2"]), rel=1e-5)


def test_oracle_and_the_rank_instrument_are_eval_only():
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="softmax").train()
    torch.manual_seed(7)
    out = m(inp, labels=lab, slot_layout=layout)
    assert not any(str(k).startswith("fan_oracle") for k in out)
    assert not any(str(k).startswith("fan_stream_rank") for k in out)
    ev = _eval_with_oracle(m, inp, lab, layout)
    assert "fan_oracle_ce" in ev
    assert any(str(k).startswith("fan_stream_rank_t") for k in ev)
    for k, v in ev.items():
        if str(k).startswith("fan_"):
            assert torch.isfinite(torch.as_tensor(float(v))), k


def _ranks(ev) -> list[float]:
    return [float(v) for k, v in ev.items() if str(k).startswith("fan_stream_rank_t")]


def test_stream_rank_is_bounded_by_k_and_separates_collapse_from_spread():
    """Two-sided, because a one-sided rank reading cannot be told from a broken one.

    MEASURED AND NOT ASSUMED (this test caught the wrong assumption once): at step 0 the
    register's ``W_o`` and ``P_cell`` are zero, so the four streams enter the loop as ONE
    vector — but they do NOT stay identical, because a slot's four cells sit at four
    consecutive ROW positions and CoPE, the CCA causal conv and the ``W_v_prev`` value
    shift all see that. The reading at init is ~1.15-1.19 of 4, not 1.00. That is the
    near-collapse floor on this fixture, and the register's own 1.24 of 4 at 5000 steps
    (2026-09-13) sits just above it.
    """
    _ids0, inp, lab, layout = _batch(4)
    m = _model(fan_k=4, slot_cells=4, fan_mix="mean")
    flat = _ranks(_eval_with_oracle(m, inp, lab, layout))
    assert flat and all(0.0 <= r <= 4.0 + 1e-9 for r in flat)
    assert max(flat) < 1.5, "the zero-init seed must read as a near-collapse"
    with torch.no_grad():
        m.tul_register.W_o.weight.normal_(0.0, 0.5)
        m.tul_register.P_cell.normal_(0.0, 0.5)
    spread = _ranks(_eval_with_oracle(m, inp, lab, layout))
    assert max(spread) > max(flat) + 0.2, (
        f"seeding the streams apart must MOVE the rank: {flat} -> {spread}")


# ── 5. THE KNOWN-KEY CONTRACT ────────────────────────────────────────────────

def test_unknown_fan_keys_raise_through_tul_setup():
    for k in ("fan_kk", "fan_repel", "fan_repel_weight", "fan_gate", "fan"):
        with pytest.raises(ValueError, match="unknown key"):
            reject_unknown_tul_keys({k: 1})
    for k in ("fan_k", "fan_mix", "fan_repel_lambda", "fan_repel_passes"):
        reject_unknown_tul_keys({k: 1})          # must NOT raise
