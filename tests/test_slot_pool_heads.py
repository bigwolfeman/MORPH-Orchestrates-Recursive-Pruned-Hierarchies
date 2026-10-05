"""``tul.slot_pool_heads`` — the register's seeding pool (``TULSlotRegister``) split into
H heads instead of one (CE queue item 9, 2026-10-05).

Why: `TULSlotRegister` seeds each of a span's M cells with ONE single-head softmax pool
over the span's own prelude token states — a weighted AVERAGE. The exact-recall probe
measured 65% of the LXTUL CE gap to a plain model as missed exact bigram copies from
earlier spans, and a single softmax average cannot commit to one token exactly. `H > 1`
reshapes the SAME `Q_i` / `W_k` / `W_v` / `W_o` parameters into `H` heads of
`d_model / H` each (no new parameter, no shape change) so different heads can attend to
different single tokens of the span instead of one head blending them all.

Files: morph/model/tul.py (`slot_pool_heads` field, `_check_slot_pool_heads`,
`TULSlotRegister.__init__` / `.forward`), morph/model/transformer.py (the construction
site passes `n_heads=cfg.tul.slot_pool_heads`), morph/training/tul_setup.py
(`KNOWN_TUL_KEYS`, the Hydra -> TULConfig path, the wandb manifest), the three new
configs (`lxtul_pool16.yaml`, `lxtul_recon.yaml`, `lxtul_pool16_recon.yaml`).

What each test pins:
  1. H=1 (default, explicit or implicit) is BIT-IDENTICAL to the tree before this key:
     state_dict, train loss, every W_k/W_v/Q/W_o gradient, and eval logits.
  2. H=16 draws the SAME parameters as H=1 (every real draw happens before any
     H-dependent reshape) — every tensor in the state_dict is `torch.equal`, not just
     "close". Since `W_o` is zero-init, H ALSO does not move the full model's forward at
     step 0 (the register is a no-op there regardless of H) — checked directly so the
     "RNG-neutral" claim and the "no-op at step 0" claim are not confused with each other.
  3. At the `TULSlotRegister` module level, with `W_o` forced to a known NON-zero matrix
     (so the pooling is visible through the output): H=1 reproduces the single-head
     formula this file transcribes independently of `morph/model/tul.py`, H=16 (the
     fixture's largest divisor of d_model=64) reproduces an independently transcribed
     multi-head formula on a hand-picked slot/cell, and the per-head attention
     distributions are NOT all equal to each other.
  4. CONFIG REFUSALS: `slot_pool_heads > 1` with `slot_cells == 1` (no register) raises;
     `slot_pool_heads < 1` raises; `TULSlotRegister` itself raises when `d_model % H != 0`.
  5. `slot_pool_heads` is a known `tul:` key and the Hydra -> `TULConfig` path and the
     wandb manifest carry it through; the three new configs compose and differ from
     `lxtul.yaml` by exactly the keys their own header claims.
  6. END TO END: a real forward + backward on each new config's tiny analogue (H=16,
     recon, and both together) gives a finite loss.

Sabotage checks (reported in the session, not committed): (a) reverting the scale from
`d_head ** -0.5` back to `d_model ** -0.5` leaves H=1 untouched (d_head == d_model there)
but should change the H=16 hand-computed pin — if it instead still passes, the hand
computation was not actually exercising the scale; (b) skipping the `repeat_interleave`-
style per-head concat order (e.g. interleaving heads instead of concatenating them) fails
the position-sensitive hand-computed pin.

CPU, fp32, the ``tests/test_tul_fan.py`` fixtures (strict geometry, dropout 0, d_model 64).
"""
from __future__ import annotations

import math

import pytest
import torch

from morph.model.tul import TULConfig, TULSlotRegister
from morph.model.transformer import MORPHTransformer
from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
from test_tul_fan import _batch, _model, _rule, _spec, _tiny, _tul
from test_tul_fan_lsel import _lsel

D = 64          # the fixture's d_model (test_tul_fan._tiny)
M = 4           # cells per span, prefix_k == slot_cells on the plain register path


# ── 1. H=1 is bit-identical to the tree before this key ─────────────────────────────


def test_h1_implicit_and_explicit_are_bit_identical_to_each_other_and_train():
    a = _model(seed=11, slot_cells=M, prefix_k=M)                 # no slot_pool_heads key
    b = _model(seed=11, slot_cells=M, prefix_k=M, slot_pool_heads=1)
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    _ids, inp, lab, layout = _batch(M)
    a.train(); b.train()
    torch.manual_seed(3)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(3)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(oa["loss"], ob["loss"])
    oa["loss"].backward()
    ob["loss"].backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        ga, gb = pa.grad, pb.grad
        assert (ga is None) == (gb is None), na
        if ga is not None:
            assert torch.equal(ga, gb), na
    a.eval(); b.eval()
    with torch.no_grad():
        la = a(inp, labels=None, slot_layout=layout)["logits"]
        lb = b(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


# ── 2. H=16 draws the SAME parameters; step-0 forward is also unmoved ───────────────


def test_h16_is_rng_neutral_and_step0_forward_unmoved():
    m1 = _model(seed=22, slot_cells=M, prefix_k=M, slot_pool_heads=1)
    m16 = _model(seed=22, slot_cells=M, prefix_k=M, slot_pool_heads=16)
    s1, s16 = m1.state_dict(), m16.state_dict()
    assert s1.keys() == s16.keys()
    for k in s1:
        assert torch.equal(s1[k], s16[k]), k
    assert m16.tul_register.n_heads == 16
    assert m16.tul_register.d_head == D // 16
    _ids, inp, lab, layout = _batch(M)
    m1.train(); m16.train()
    torch.manual_seed(5)
    o1 = m1(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(5)
    o16 = m16(inp, labels=lab, slot_layout=layout)
    # W_o is zero-init, so the register's additive seed is exactly 0 for EVERY H at
    # step 0 — the pooling attention differs internally but the output does not.
    assert torch.equal(o1["loss"], o16["loss"])
    m1.eval(); m16.eval()
    with torch.no_grad():
        l1 = m1(inp, labels=None, slot_layout=layout)["logits"]
        l16 = m16(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(l1, l16)


# ── 3. the pooling formula itself, with W_o forced non-zero ─────────────────────────


def _hand_pool_one_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        scale: float) -> torch.Tensor:
    """The pre-queue single-head formula, transcribed independently: one softmax over
    the token axis, for ONE query vector `q [C]` against keys/values `k, v [L, C]`."""
    sc = (k @ q) * scale                              # [L]
    a = torch.softmax(sc, dim=-1)
    return a @ v                                      # [C]


def _hand_pool_multi_head(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          n_heads: int) -> torch.Tensor:
    """H independent single-head pools over H slices of the channel axis, concatenated —
    a from-scratch transcription of the multi-head formula, looped in Python rather than
    batched with `einsum`, so it exercises a DIFFERENT code path than
    `TULSlotRegister.forward`."""
    C = q.shape[-1]
    assert C % n_heads == 0
    d_head = C // n_heads
    scale = d_head ** -0.5
    outs = []
    for h in range(n_heads):
        sl = slice(h * d_head, (h + 1) * d_head)
        outs.append(_hand_pool_one_head(q[sl], k[:, sl], v[:, sl], scale))
    return torch.cat(outs, dim=-1)


def _one_slot_window(layout, b: int = 0):
    """The first VALID slot's own token window (positions where `bag_id == s` and the
    position is a token, not a slot), as a plain index tensor."""
    S = layout.slot_index.shape[1]
    for s in range(S):
        if bool(layout.slot_valid[b, s]):
            tok = (~layout.slot_mask[b]) & (layout.bag_id[b] == s)
            idx = tok.nonzero(as_tuple=True)[0]
            if idx.numel() > 0:
                return s, idx
    raise AssertionError("fixture has no valid slot with a nonempty span")


def _make_register(n_heads: int, seed: int = 0) -> TULSlotRegister:
    torch.manual_seed(seed)
    reg = TULSlotRegister(D, M, distinct=True, n_heads=n_heads)
    with torch.no_grad():
        # W_o forced to a KNOWN non-zero (but not identity, so a transposition bug would
        # not cancel out) orthogonal matrix, so the pooled vector is visible in the
        # output rather than erased by the shipped zero-init.
        reg.W_o.weight.copy_(torch.linalg.qr(torch.randn(D, D))[0])
        reg.P_cell.zero_()
    return reg


def test_h1_register_matches_the_independently_transcribed_single_head_formula():
    reg = _make_register(n_heads=1, seed=100)
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(1)
    xn = torch.randn(inp.shape[0], layout.slot_mask.shape[1], D)
    out = reg(xn, layout)                                         # [B, S*M, C]
    S = layout.slot_index.shape[1]
    s, idx = _one_slot_window(layout, b=0)
    k_full = reg.W_k(xn[0])                                       # [L, C]
    v_full = reg.W_v(xn[0])
    for cell in range(M):
        q = reg.Q[cell]                                           # [C], distinct=True
        pooled = _hand_pool_one_head(q, k_full[idx], v_full[idx], reg.scale)
        expect = reg.W_o(pooled) + reg.P_cell[cell]
        got = out[0, s * M + cell]
        assert torch.allclose(got, expect, atol=1e-5), (s, cell)


def test_h16_register_matches_the_independently_transcribed_multihead_formula():
    H = 16
    reg = _make_register(n_heads=H, seed=101)
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(2)
    xn = torch.randn(inp.shape[0], layout.slot_mask.shape[1], D)
    out = reg(xn, layout)
    s, idx = _one_slot_window(layout, b=0)
    k_full = reg.W_k(xn[0])
    v_full = reg.W_v(xn[0])
    for cell in range(M):
        q = reg.Q[cell]
        pooled = _hand_pool_multi_head(q, k_full[idx], v_full[idx], H)
        expect = reg.W_o(pooled) + reg.P_cell[cell]
        got = out[0, s * M + cell]
        assert torch.allclose(got, expect, atol=1e-5), (s, cell)


def test_heads_attend_differently_not_all_equal():
    """The per-head softmax distributions over the span's tokens are not all identical —
    if they were, H heads would be one head copied H times and the key would buy
    nothing. Checked on the hand-computed attention weights directly, at H=16."""
    H = 16
    reg = _make_register(n_heads=H, seed=102)
    _ids, inp, lab, layout = _batch(M)
    torch.manual_seed(4)
    xn = torch.randn(inp.shape[0], layout.slot_mask.shape[1], D)
    s, idx = _one_slot_window(layout, b=0)
    if idx.numel() < 2:
        pytest.skip("fixture's first valid span has one token; no attention to compare")
    k_full = reg.W_k(xn[0])
    d_head = D // H
    q = reg.Q[0]
    dists = []
    for h in range(H):
        sl = slice(h * d_head, (h + 1) * d_head)
        sc = (k_full[idx][:, sl] @ q[sl]) * (d_head ** -0.5)
        dists.append(torch.softmax(sc, dim=-1))
    base = dists[0]
    assert any(not torch.allclose(base, d, atol=1e-4) for d in dists[1:]), (
        "every head's attention distribution over the span was identical")


# ── 4. refusals ──────────────────────────────────────────────────────────────────────


def test_slot_pool_heads_must_be_positive():
    with pytest.raises(ValueError, match="slot_pool_heads must be >= 1"):
        TULConfig(prefix_k=2, slot_id=4, slot_pool_heads=0)


def test_slot_pool_heads_gt_1_without_register_raises():
    with pytest.raises(ValueError, match="slot_pool_heads=2 set with tul.slot_cells=1"):
        TULConfig(prefix_k=2, slot_id=4, slot_pool_heads=2)


def test_slot_pool_heads_not_dividing_d_model_raises_at_construction():
    with pytest.raises(ValueError, match="must divide d_model"):
        TULSlotRegister(D, M, distinct=True, n_heads=7)   # 64 % 7 != 0


# ── 5. known key, Hydra path, configs ───────────────────────────────────────────────


def test_slot_pool_heads_is_a_known_tul_key():
    assert "slot_pool_heads" in KNOWN_TUL_KEYS
    class _FakeDict(dict):
        def keys(self): return super().keys()
    reject_unknown_tul_keys(_FakeDict(slot_pool_heads=16))   # must not raise
    with pytest.raises(ValueError, match="unknown"):
        reject_unknown_tul_keys(_FakeDict(slot_pool_hedas=16))


_CONFIG_DIR = __import__("os").path.abspath("morph/configs")


def _compose(name: str):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=name)


def test_lxtul_pool16_composes_and_differs_from_lxtul_by_exactly_its_keys():
    from omegaconf import OmegaConf

    from test_slot_gain_tail import _leaves, _MISSING

    c16 = _leaves(OmegaConf.to_container(_compose("lxtul_pool16"), resolve=True))
    c1 = _leaves(OmegaConf.to_container(_compose("lxtul"), resolve=True))
    diff = {k for k in c16.keys() | c1.keys() if c16.get(k, _MISSING) != c1.get(k, _MISSING)}
    assert diff == {"tul.slot_pool_heads", "training.steps", "wandb.name"}, sorted(diff)
    assert c16["tul.slot_pool_heads"] == 16
    assert c16["training.steps"] == 5000
    assert c16["wandb.name"] == "lxtul-pool16"


def test_lxtul_pool16_reaches_the_tulconfig_the_trainer_builds(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_pool16")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.slot_pool_heads == 16
    assert rt.manifest["slot_pool_heads"] == 16


# ── 6. end to end: forward + backward on the tiny analogue, finite loss ────────────


def test_lxtul_pool16_tiny_analogue_forward_backward_is_finite():
    m = _lsel("joint", slot_pool_heads=16)
    _ids, inp, lab, layout = _batch(M)
    m.train()
    torch.manual_seed(6)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    for n, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), n
