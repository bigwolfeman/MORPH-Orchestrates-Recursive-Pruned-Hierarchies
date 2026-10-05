"""``tul.prefix_per_cell`` — the write-all fan's WIDER WRITE (2026-10-05).

Isolates "more coda positions per slot" from "more loop streams": the loop stays at
``fan_k`` streams, but each stream now lands in ``m = prefix_per_cell`` coda positions
instead of one (``prefix_k = fan_k * m``), through ``TULSlots.prefix_project``'s cell-axis
``repeat_interleave`` expansion. Config: ``morph/configs/lxtul_fan4x2.yaml`` (one factor
off ``lxtul.yaml``), against ``lxtul_fan8.yaml`` (more STREAMS instead).

Files: morph/model/tul.py (the key, ``__post_init__`` checks, ``TULSlots.__init__``'s
W_prefix init, ``TULSlots.prefix_project``'s expansion), morph/training/tul_setup.py
(``KNOWN_TUL_KEYS``, the Hydra -> TULConfig path, the wandb manifest),
morph/inference/tul_generate_cached.py (the explicit refusal, transitively covered by the
existing ``fan_k == 0`` one).

What each test pins:
  1. m=1 is BIT-IDENTICAL to the tree before this key: loss, eval logits and every
     parameter, against a model built with no ``prefix_per_cell`` key at all.
  2. m=2 builds RNG-NEUTRALLY: every parameter but ``tul.W_prefix`` equals the m=1
     model's at the same seed; ``W_prefix``'s copy 0 of each cell is identity, copies
     ``j >= 1`` are orthogonal and not identity.
  3. POSITION ORDER: cell i's copy 0 (position ``i*m``) writes exactly what the m=1
     model's position i would write, for an arbitrary (non-identity-trivial) ``cells``
     input.
  4. A BLANK single-stream write (``_tul_fan_stream_write``'s pattern) is nonzero at
     exactly cell i's ``m`` positions.
  5. CONFIG REFUSALS: ``prefix_k != fan_k * m``, ``m > 1`` without ``fan_mix='all'``,
     ``m < 1``.
  6. END TO END: a real forward + backward of the m=2 tiny model is finite and every
     ``W_prefix`` copy (every cell, every sub-position) gets a nonzero gradient.
  7. THE LATENT-SELECTED LOOP's deployed write (``fan_loop_select='joint'``,
     ``fan_lsel_read='winner'``, the default): at eval (router-followed, no teacher) the
     routed winner's cell lands in BOTH its positions and every loser's ``m`` positions
     are exactly zero — orchestrator-specified, 2026-10-05.
  8. THE EXPLICIT GENERATION REFUSAL in ``tul_generate_cached._check_supported`` (and
     ``tul_generate_graphed``, which imports it) names ``tul.prefix_per_cell`` directly.
  9. THE CONFIG composes, differs from ``lxtul`` by exactly
     ``{prefix_per_cell, prefix_k, training.steps, wandb.name}`` and from ``lxtul_fan8``
     by exactly ``{fan_k, prefix_per_cell, model.ckpt_grad_iters, wandb.name}``.

Sabotage checks (reported in the session, not committed): (a) disabling the
``repeat_interleave`` expansion inside ``prefix_project`` makes the shape check raise,
failing every forward-based test here; (b) using identity for every copy ``j >= 1``
(instead of the orthogonal draw) fails
``test_m2_builds_rng_neutrally_and_w_prefix_is_identity_then_orthogonal`` alone (its
"not identical to copy 0" assertion), while leaving the position/gradient tests green —
confirming that test is the one actually pinning the orthogonal draw.

CPU only, fp32, the ``tests/test_tul_fan.py`` / ``tests/test_tul_lxfan.py`` tiny fixtures.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

from morph.model.tul import TULConfig
from test_tul_fan import _batch, _model, _tul
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M as FAN_K

FAN_K = int(FAN_K)


def _fan_all_kw(m: int, **kw) -> dict:
    """The write-all fan at ``fan_k=FAN_K`` with ``prefix_per_cell=m``; ``m=1`` is the
    tree before this key (``prefix_k == fan_k``, no ``prefix_per_cell`` arithmetic)."""
    base = dict(fan_k=FAN_K, slot_cells=FAN_K, prefix_k=FAN_K * m, fan_mix="all",
                fan_select_eps=0.0)
    if m != 1:
        base["prefix_per_cell"] = m
    base.update(kw)
    return base


def _fan_all_model(m: int, seed: int = 1234, **kw) -> "object":
    return _model(seed=seed, **_fan_all_kw(m, **kw))


# ── 1. m=1 is the tree before the key ───────────────────────────────────────────────


def test_m1_is_bit_identical_to_no_prefix_per_cell_key():
    a = _model(fan_k=FAN_K, slot_cells=FAN_K, prefix_k=FAN_K, fan_mix="all",
              fan_select_eps=0.0)                              # no prefix_per_cell key
    b = _model(fan_k=FAN_K, slot_cells=FAN_K, prefix_k=FAN_K, fan_mix="all",
              fan_select_eps=0.0, prefix_per_cell=1)            # explicit default
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    _ids, inp, lab, layout = _batch(FAN_K)
    a.train(); b.train()
    torch.manual_seed(7)
    oa = a(inp, labels=lab, slot_layout=layout)
    torch.manual_seed(7)
    ob = b(inp, labels=lab, slot_layout=layout)
    assert torch.equal(oa["loss"], ob["loss"])
    a.eval(); b.eval()
    with torch.no_grad():
        la = a(inp, labels=None, slot_layout=layout)["logits"]
        lb = b(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


# ── 2. m=2 is RNG-neutral; W_prefix identity-then-orthogonal ────────────────────────


def test_m2_builds_rng_neutrally_and_w_prefix_is_identity_then_orthogonal():
    m1 = _fan_all_model(1, seed=1234)
    m2 = _fan_all_model(2, seed=1234)
    s1, s2 = m1.state_dict(), m2.state_dict()
    assert set(s1.keys()) == set(s2.keys())
    for k in s1:
        if k == "tul.W_prefix":
            continue
        assert torch.equal(s1[k], s2[k]), k
    w2 = m2.tul.W_prefix.detach()
    d = w2.shape[-1]
    eye = torch.eye(d)
    assert w2.shape[0] == FAN_K * 2
    for i in range(FAN_K):
        c0 = w2[i * 2]
        assert torch.equal(c0, eye), f"cell {i} copy 0 must be identity"
        cj = w2[i * 2 + 1]
        assert torch.allclose(cj @ cj.T, eye, atol=1e-5), f"cell {i} copy 1 must be orthogonal"
        assert not torch.allclose(cj, eye, atol=1e-4), f"cell {i} copy 1 must not be identity"
        # distinct cells draw DIFFERENT orthogonal matrices (not one matrix reused)
        if i > 0:
            assert not torch.allclose(cj, w2[(i - 1) * 2 + 1], atol=1e-4)


def test_m2_build_draws_no_global_rng(monkeypatch):
    """``_model`` calls ``torch.manual_seed`` itself and (the ``test_tul_fan_lsel.py``
    precedent) construction makes exactly one ``cuda.manual_seed_all`` call regardless of
    TUL; the W_prefix orthogonal draw must add no SECOND call and move no CPU RNG state."""
    calls = []
    for fn in ("manual_seed", "manual_seed_all"):
        monkeypatch.setattr(torch.cuda, fn, lambda *a, _fn=fn, **k: calls.append(_fn))
    _fan_all_model(1, seed=55)
    after_m1 = torch.random.get_rng_state()
    calls.clear()
    _fan_all_model(2, seed=55)
    assert torch.equal(torch.random.get_rng_state(), after_m1)
    assert calls == ["manual_seed_all"], calls


# ── 3. position order: cell i -> positions i*m .. i*m+m-1 ───────────────────────────


def test_cell_i_copy0_matches_the_m1_write_exactly():
    m1 = _fan_all_model(1, seed=9)
    m2 = _fan_all_model(2, seed=9)
    _ids, inp, lab, layout = _batch(FAN_K * 2)
    B, S = layout.slot_valid.shape
    d = m1.cfg.d_model
    n = 4 if m1._is_hc else None   # HC stream count, matches _tiny's hyper-connection build
    shape = (B, S, FAN_K, n, d) if n else (B, S, FAN_K, d)
    torch.manual_seed(0)
    cells = torch.randn(*shape)
    L = int(layout.l_total)
    v1, _ = m1.tul.prefix_project(cells[:, :, 0], layout, L, cells=cells)
    v2, _ = m2.tul.prefix_project(cells[:, :, 0], layout, L, cells=cells)
    v1 = v1.view(B, S, FAN_K, *v1.shape[2:])
    v2 = v2.view(B, S, FAN_K * 2, *v2.shape[2:])
    for i in range(FAN_K):
        assert torch.equal(v1[:, :, i], v2[:, :, i * 2]), f"cell {i} copy 0"


def test_blank_single_stream_write_is_nonzero_at_exactly_m_positions():
    m2 = _fan_all_model(2, seed=3)
    _ids, inp, lab, layout = _batch(FAN_K * 2)
    B, S = layout.slot_valid.shape
    d = m2.cfg.d_model
    n = 4 if m2._is_hc else None
    shape = (B, S, FAN_K, n, d) if n else (B, S, FAN_K, d)
    torch.manual_seed(1)
    cells = torch.randn(*shape)
    L = int(layout.l_total)
    for i in range(FAN_K):
        blank = torch.zeros_like(cells)
        blank[:, :, i] = cells[:, :, i]
        values, _ = m2.tul.prefix_project(cells[:, :, i], layout, L, cells=blank)
        v = values.view(B, S, FAN_K * 2, *values.shape[2:])
        nz = v.flatten(3).abs().sum(dim=-1) > 0 if n else v.abs().sum(dim=-1) > 0  # [B,S,K]
        expect = {i * 2, i * 2 + 1}
        for b in range(B):
            for s in range(S):
                got = {j for j in range(FAN_K * 2) if bool(nz[b, s, j])}
                assert got == expect or got == set(), (b, s, got, expect)


# ── 4. config refusals ───────────────────────────────────────────────────────────────


def test_prefix_k_mismatch_raises():
    with pytest.raises(ValueError, match="prefix_k=8"):
        TULConfig(**_tul_dict(fan_k=4, slot_cells=4, fan_mix="all", prefix_k=4,
                              prefix_per_cell=2))


def test_m_gt_1_without_fan_all_raises():
    with pytest.raises(ValueError, match="tul.prefix_per_cell=2 needs"):
        TULConfig(**_tul_dict(fan_k=0, prefix_per_cell=2))
    with pytest.raises(ValueError, match="tul.prefix_per_cell=2 needs"):
        TULConfig(**_tul_dict(fan_k=4, slot_cells=4, fan_mix="mean", prefix_per_cell=2))


def test_m_lt_1_raises():
    with pytest.raises(ValueError, match="prefix_per_cell must be >= 1"):
        TULConfig(**_tul_dict(prefix_per_cell=0))


def _tul_dict(**kw) -> dict:
    base = dict(prefix_k=2, slot_id=4)
    base.update(kw)
    return base


# ── 5. end to end: forward + backward, every W_prefix copy gets gradient ───────────


def test_end_to_end_forward_backward_every_w_prefix_copy_gets_gradient():
    m = _fan_all_model(2, seed=11, fan_select_eps=0.05, fan_all_wta_lambda=1.0)
    _ids, inp, lab, layout = _batch(FAN_K * 2)
    m.train()
    torch.manual_seed(42)
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    g = m.tul.W_prefix.grad
    assert g is not None
    d = g.shape[-1]
    per_copy = g.view(FAN_K, 2, d, d).abs().sum(dim=(-1, -2))
    assert bool((per_copy > 0).all()), per_copy


# ── 6. the latent-selected loop's deployed (routed) write ──────────────────────────


def test_lsel_routed_winner_lands_in_both_positions_losers_exactly_zero():
    """Orchestrator note (2026-10-05): under ``fan_loop_select='joint'`` with the default
    ``fan_lsel_read='winner'`` the DEPLOYED write is the routed one
    (``_fan_route_cells``, every loser cell exactly zero) before it reaches
    ``prefix_project``. At ``prefix_per_cell=2`` the winner's two positions must be the
    ONLY nonzero ones and every loser's two positions must be exactly zero. The
    label-free eval forward (no teacher) always follows the router — the generation
    path's own forward."""
    m = _lsel("joint", prefix_k=FAN_K * 2, prefix_per_cell=2)
    _ids0, inp, lab, layout = _batch(FAN_K * 2)
    seen: dict = {}
    orig = m.tul.prefix_project

    def _pp(h_slots, layout_, l_total, cells=None):
        seen["cells"] = None if cells is None else cells.detach().clone()
        return orig(h_slots, layout_, l_total, cells=cells)
    m.tul.prefix_project = _pp
    m.eval()
    with torch.no_grad():
        m(inp, labels=None, slot_layout=layout)
    cells = seen["cells"]
    assert cells is not None and cells.shape[2] == FAN_K
    B, S = layout.slot_valid.shape
    valid = layout.slot_valid
    nz_cell = cells.flatten(3).abs().sum(dim=-1) > 0            # [B, S, FAN_K]
    # exactly one live fan cell per valid slot (the routed winner; losers are zeroed by
    # `_fan_route_cells` before `prefix_project` ever sees them)
    assert bool((nz_cell[valid].sum(dim=-1) == 1).all())
    winner = nz_cell.float().argmax(dim=-1)                      # [B, S]
    L = int(layout.l_total)
    values, _ = orig(cells[:, :, 0], layout, L, cells=cells)
    v = values.view(B, S, FAN_K * 2, *values.shape[2:])
    nz_pos = v.flatten(3).abs().sum(dim=-1) > 0                  # [B, S, FAN_K*2]
    for b in range(B):
        for s in range(S):
            if not bool(valid[b, s]):
                continue
            w = int(winner[b, s])
            expect = {w * 2, w * 2 + 1}
            got = {j for j in range(FAN_K * 2) if bool(nz_pos[b, s, j])}
            assert got == expect, (b, s, got, expect)


# ── 7. generation refusal names the key ─────────────────────────────────────────────


def test_cached_generation_refuses_the_fan_which_transitively_refuses_prefix_per_cell():
    """``tul.prefix_per_cell > 1`` requires ``fan_k > 0`` at CONSTRUCTION, so a model with
    both is already caught by the EXISTING ``fan_k == 0`` refusal before the explicit
    ``prefix_per_cell`` guard added beside it is ever reached — that guard is named
    explicitly so a reader does not have to trace the construction-time chain to learn
    why generation refuses. There is no legitimate TULConfig this guard's own message
    fires for; that is checked directly off the source and off TULConfig's refusal."""
    import inspect

    from morph.inference.tul_generate_cached import _check_supported
    m = _fan_all_model(2, seed=1)
    assert m.cfg.tul.prefix_per_cell == 2
    with pytest.raises(NotImplementedError, match="the fan"):
        _check_supported(m)
    src = inspect.getsource(_check_supported)
    assert 'need(int(tc.prefix_per_cell) == 1, "tul.prefix_per_cell' in src
    with pytest.raises(ValueError, match="needs tul.fan_k > 0"):
        TULConfig(prefix_k=2, slot_id=4, prefix_per_cell=2)


# ── 8. the config ────────────────────────────────────────────────────────────────────


_CONFIG_DIR = __import__("os").path.abspath("morph/configs")


def _compose(name: str):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=name)


def test_lxtul_fan4x2_composes_and_differs_from_lxtul_and_fan8():
    from omegaconf import OmegaConf

    from test_slot_gain_tail import _leaves, _MISSING

    c42 = _leaves(OmegaConf.to_container(_compose("lxtul_fan4x2"), resolve=True))
    c1 = _leaves(OmegaConf.to_container(_compose("lxtul"), resolve=True))
    c8 = _leaves(OmegaConf.to_container(_compose("lxtul_fan8"), resolve=True))

    diff1 = {k for k in c42.keys() | c1.keys() if c42.get(k, _MISSING) != c1.get(k, _MISSING)}
    assert diff1 == {"tul.prefix_per_cell", "tul.prefix_k", "training.steps", "wandb.name"}, \
        sorted(diff1)
    assert c42["tul.prefix_per_cell"] == 2 and c42["tul.prefix_k"] == 8
    assert c42["training.steps"] == 5000 and c42["wandb.name"] == "lxtul-fan4x2"

    diff8 = {k for k in c42.keys() | c8.keys() if c42.get(k, _MISSING) != c8.get(k, _MISSING)}
    assert diff8 == {"tul.fan_k", "tul.prefix_per_cell", "model.ckpt_grad_iters",
                     "wandb.name"}, sorted(diff8)
    assert c42["tul.fan_k"] == 4 and c8["tul.fan_k"] == 8
    assert c42["model.ckpt_grad_iters"] == 4 and c8["model.ckpt_grad_iters"] == -1


def test_lxtul_fan4x2_reaches_the_tulconfig_the_trainer_builds(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_fan4x2")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.prefix_per_cell == 2
    assert rt.model_cfg.prefix_k == rt.model_cfg.fan_k * 2 == 8
    assert rt.manifest["prefix_per_cell"] == 2
    assert rt.manifest["prefix_k"] == 8
