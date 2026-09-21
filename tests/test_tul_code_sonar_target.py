"""``tul.code_target_source: sonar`` — rung P3 of LXTUL-P, one test per contract.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest \
        tests/test_tul_code_sonar_target.py -q

CPU only, no tokenizer, no SONAR: the cache is written by hand in a tmp dir with KNOWN
embeddings, so every assertion below is on VALUES, not shapes. The tiny fixture runs at
``prefix_k = 16`` and ``d_model = 64`` because the frozen lift is orthogonal and needs
``prefix_k * d_model >= 1024``; that is the real constraint a small arm hits, so the test
exercises it rather than working around it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from test_tul_gl1 import DOT, SLOT_ID, V, _cfg, _rule, _tul  # noqa: E402
from test_tul_strict_geometry import _StubTok  # noqa: E402

from morph.model.sonar_cache import (SONAR_DIM, SonarSpanCache, frozen_code_expansion,
                                     next_span_hashes, span_hashes, span_token_hash,
                                     write_sonar_cache)
from morph.model.tul import TULConfig
from morph.model.tul_code import code_rmsnorm
from morph.model.tul_layout import TulLayoutSpec, slot_layout_from_ids
from morph.model.transformer import _SONAR_LIFT_SEED, MORPHTransformer

M_CELLS = 16          # prefix_k; M * d_model == 1024 == SONAR_DIM exactly
D_MODEL = 64


# ── fixtures ────────────────────────────────────────────────────────────────────

def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=64, prefix_k=M_CELLS, max_slots=6, slot_id=SLOT_ID)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(B: int = 2, n: int = 90, seed: int = 0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == SLOT_ID] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), _spec())


def _sonar_tul(**kw) -> TULConfig:
    base = dict(prefix_k=M_CELLS, slot_id=SLOT_ID, slot_seed="boundary", tg_restrict=True,
                tg_geometry="strict", tg_restrict_scope="all", sigreg_lambda=0.0,
                mux_beta=0.0, token_state_dropout=0.0, emit_weight=0.0, plast_weight=1.0,
                code=True)
    base.update(kw)
    return TULConfig(**base)


def _model(tul: TULConfig, seed: int = 3) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_cfg(tul=tul, n_core=2, retention=False, d_model=D_MODEL,
                                 mean_depth=3, max_depth=3, bptt_depth=3, dropout=0.0,
                                 core_fixed_point_lambda=0.0, ckpt_grad_iters=0))


def _known_emb(key: int) -> np.ndarray:
    """A deterministic, distinct 1024-vector per key — the tests read these back.

    Rounded through float16 because that is what the cache stores: the expected values
    below must take the SAME round trip, or the test is measuring the f16 cast.
    """
    g = np.random.default_rng(int(key % (2 ** 32)))
    return g.standard_normal(SONAR_DIM).astype(np.float16).astype(np.float32)


def _write_cache(tmp_path, layout, inp, drop_keys=()) -> str:
    """Every TARGET span of the batch, with `_known_emb` as its embedding."""
    S = int(layout.slot_index.shape[1])
    h, n_tok, _ids = span_hashes(inp.numpy(), layout.bag_id.numpy(),
                                 layout.slot_mask.numpy(), S)
    tgt, has = next_span_hashes(h, n_tok)
    keys = sorted({int(tgt[b, s]) for b in range(tgt.shape[0])
                   for s in range(S) if has[b, s]} - set(int(k) for k in drop_keys))
    emb = np.stack([_known_emb(k) for k in keys]) if keys else \
        np.zeros((0, SONAR_DIM), np.float32)
    p = str(tmp_path / "cache")
    write_sonar_cache(p, np.asarray(keys, dtype=np.uint64), emb)
    return p


def _expected_cells(layout, inp, q: torch.Tensor) -> torch.Tensor:
    """The contract, recomputed here from the definition: lift, then RMS-norm per cell."""
    S = int(layout.slot_index.shape[1])
    B = inp.shape[0]
    h, n_tok, _ = span_hashes(inp.numpy(), layout.bag_id.numpy(),
                              layout.slot_mask.numpy(), S)
    tgt, has = next_span_hashes(h, n_tok)
    out = torch.zeros(B, S, M_CELLS * D_MODEL, dtype=torch.float32)
    for b in range(B):
        for s in range(S):
            if has[b, s]:
                v = torch.from_numpy(_known_emb(int(tgt[b, s])))
                out[b, s] = q.to(torch.float32) @ v
    return out.view(B, S, M_CELLS, D_MODEL)


# ── 1. the default is the tree before the knob ──────────────────────────────────

def test_default_e_builds_nothing_and_is_bit_identical():
    a = _model(_sonar_tul())
    b = _model(_sonar_tul(code_target_source="e"))
    assert a._sonar_cache is None and b._sonar_cache is None
    assert not hasattr(a, "tul_code_sonar_q")
    for n, p in a.named_parameters():
        assert torch.equal(p, dict(b.named_parameters())[n]), f"{n} differs"

    inp, lab, layout, _ = _batch()
    torch.manual_seed(99)
    la = a(inp, labels=lab, slot_layout=layout)["loss"]
    sa = torch.random.get_rng_state()
    torch.manual_seed(99)
    lb = b(inp, labels=lab, slot_layout=layout)["loss"]
    assert torch.equal(la, lb), "the explicit 'e' default changed the loss"
    assert torch.equal(sa, torch.random.get_rng_state()), \
        "the explicit 'e' default changed the RNG consumption"


# ── 2. the target IS the expanded, rms-normed cache row of the NEXT span ────────

def test_sonar_target_is_the_cache_row_of_the_next_span(tmp_path):
    inp, lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    m = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path))

    ok = (layout.slot_valid.clone())
    S = int(layout.slot_index.shape[1])
    ok[:, S - 1] = False
    ok[:, :S - 1] &= layout.slot_valid[:, 1:S]
    got = m._sonar_target(inp, layout, ok, torch.float32)

    want_raw = _expected_cells(layout, inp, m.tul_code_sonar_q)
    want = code_rmsnorm(want_raw) * ok.view(*ok.shape, 1, 1).float()
    assert got.shape == (inp.shape[0], S, M_CELLS, D_MODEL)
    assert torch.allclose(got, want, atol=1e-5), \
        f"max |Δ| {float((got - want).abs().max())}"
    # It is the NEXT span's row, not the slot's own: swapping the shift must break it.
    h, n_tok, _ = span_hashes(inp.numpy(), layout.bag_id.numpy(),
                              layout.slot_mask.numpy(), S)
    own = torch.zeros_like(want_raw)
    for b in range(inp.shape[0]):
        for s in range(S):
            if n_tok[b, s] > 0:
                own[b, s] = (m.tul_code_sonar_q.to(torch.float32)
                             @ torch.from_numpy(_known_emb(int(h[b, s])))).view(
                                 M_CELLS, D_MODEL)
    own = code_rmsnorm(own) * ok.view(*ok.shape, 1, 1).float()
    assert not torch.allclose(got, own, atol=1e-3), \
        "slot s reads its OWN span's code — the +1 shift is missing"
    # Every valid slot is a real, unit-RMS cell, not a zero left by a silent miss.
    rms = got[ok].pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_the_code_the_forward_uses_is_the_sonar_cell(tmp_path):
    """End to end through `_tul_code_core`: the cells the coda gets in `encoder` mode
    ARE the SONAR cells, so the seam swapped every reader and not just the flow loss."""
    inp, lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    m = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path,
                          code_noise=0.0))
    m.eval()
    with torch.no_grad():
        out = m.tul_forward_ablated(inp, lab, layout, code_mode="encoder")
    S = int(layout.slot_index.shape[1])
    ok = layout.slot_valid.clone()
    ok[:, S - 1] = False
    ok[:, :S - 1] &= layout.slot_valid[:, 1:S]
    want = code_rmsnorm(_expected_cells(layout, inp, m.tul_code_sonar_q)) \
        * ok.view(*ok.shape, 1, 1).float()
    got = out["code_cells"]
    assert torch.allclose(got.float(), want, atol=1e-4), \
        f"max |Δ| {float((got.float() - want).abs().max())}"


# ── 3. the lift is orthogonal, frozen, and not a parameter ──────────────────────

def test_expansion_is_orthogonal_frozen_and_reproducible(tmp_path):
    inp, _lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    m = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path))
    q = m.tul_code_sonar_q
    assert q.shape == (M_CELLS * D_MODEL, SONAR_DIM)
    eye = q.t() @ q
    assert torch.allclose(eye, torch.eye(SONAR_DIM), atol=1e-5), \
        f"Qᵀ Q is not I: max |Δ| {float((eye - torch.eye(SONAR_DIM)).abs().max())}"
    assert not q.requires_grad
    names = {n for n, _ in m.named_parameters()}
    assert "tul_code_sonar_q" not in names, "the lift is a buffer, never a parameter"
    assert any(n == "tul_code_sonar_q" for n, _ in m.named_buffers())
    assert "tul_code_sonar_q" not in m.state_dict(), \
        "the lift must be NON-persistent so a sonar arm's checkpoint pairs with its 'e' twin"
    # Reproducible from the seed alone — an offline probe rebuilds it this way.
    assert torch.equal(q, frozen_code_expansion(SONAR_DIM, M_CELLS * D_MODEL,
                                                _SONAR_LIFT_SEED))
    # Frozen means the target cannot move: no grad ever reaches it.
    ok = layout.slot_valid.clone()
    S = int(layout.slot_index.shape[1])
    ok[:, S - 1] = False
    ok[:, :S - 1] &= layout.slot_valid[:, 1:S]
    z = m._sonar_target(inp, layout, ok, torch.float32)
    assert not z.requires_grad, "the SONAR target carries a graph — it is not frozen"


def test_expansion_refuses_a_narrower_target():
    with pytest.raises(ValueError, match="d_out >= d_in"):
        frozen_code_expansion(SONAR_DIM, 512, 0)


# ── 4. a miss raises, and says which span ───────────────────────────────────────

def test_a_miss_raises_naming_the_span_tokens(tmp_path):
    inp, _lab, layout, _ = _batch()
    S = int(layout.slot_index.shape[1])
    h, n_tok, span_ids = span_hashes(inp.numpy(), layout.bag_id.numpy(),
                                     layout.slot_mask.numpy(), S)
    tgt, has = next_span_hashes(h, n_tok)
    b0, s0 = next((b, s) for b in range(tgt.shape[0]) for s in range(S) if has[b, s])
    dropped = int(tgt[b0, s0])
    path = _write_cache(tmp_path, layout, inp, drop_keys=(dropped,))
    m = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path))
    ok = layout.slot_valid.clone()
    ok[:, S - 1] = False
    ok[:, :S - 1] &= layout.slot_valid[:, 1:S]
    with pytest.raises(KeyError) as e:
        m._sonar_target(inp, layout, ok, torch.float32)
    msg = str(e.value)
    first = span_ids[b0][s0 + 1][:16].tolist()
    assert str(first[0]) in msg and "token ids" in msg, msg
    assert str(dropped) in msg, msg


def test_a_miss_is_never_a_zero_row(tmp_path):
    """The failure this refusal exists to prevent: a zero target is learnable and silent."""
    inp, _lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp, drop_keys=())
    cache = SonarSpanCache(path)
    rows = cache.lookup(np.asarray([np.uint64(1234567)], dtype=np.uint64))
    assert int(rows[0]) == -1, "an absent key must report -1, not row 0"


# ── 5. refusals ─────────────────────────────────────────────────────────────────

def test_refusals(tmp_path):
    with pytest.raises(ValueError, match="code_target_source must be 'e' or 'sonar'"):
        _sonar_tul(code_target_source="clip")
    with pytest.raises(ValueError, match="needs tul.code_sonar_cache"):
        _sonar_tul(code_target_source="sonar")
    with pytest.raises(ValueError, match="code_discrete"):
        _sonar_tul(code_target_source="sonar", code_sonar_cache=str(tmp_path),
                   code_discrete=True)
    with pytest.raises(ValueError, match="code_fm_weight=0"):
        _sonar_tul(code_target_source="sonar", code_sonar_cache=str(tmp_path),
                   code_fm_weight=0.0)
    with pytest.raises(ValueError, match="code_target_lambda"):
        _sonar_tul(code_target_source="sonar", code_sonar_cache=str(tmp_path),
                   code_target_lambda=0.5)
    with pytest.raises(ValueError, match="nothing would read the cache"):
        _sonar_tul(code_sonar_cache=str(tmp_path))
    # No flow thinker at all: `tul.code=false` builds no encoder and no velocity head.
    with pytest.raises(ValueError, match="code_target_source"):
        _tul(code=False, code_target_source="sonar")
    with pytest.raises(ValueError, match="code_sonar_cache"):
        _tul(code=False, code_sonar_cache=str(tmp_path))


def test_a_missing_cache_directory_raises_at_build(tmp_path):
    missing = str(tmp_path / "nope")
    tul = _sonar_tul(code_target_source="sonar", code_sonar_cache=missing)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        _model(tul)
    empty = tmp_path / "empty"
    empty.mkdir()
    tul2 = _sonar_tul(code_target_source="sonar", code_sonar_cache=str(empty))
    with pytest.raises(FileNotFoundError, match="index.npy is missing"):
        _model(tul2)


# ── 6. the cache script's span cutter IS the packer's ───────────────────────────

def test_span_cutter_reproduces_the_packers_spans():
    inp, _lab, layout, _ = _batch(B=2, n=120, seed=5)
    S = int(layout.slot_index.shape[1])
    h, n_tok, span_ids = span_hashes(inp.numpy(), layout.bag_id.numpy(),
                                     layout.slot_mask.numpy(), S)
    bag = layout.bag_id.numpy()
    sm = layout.slot_mask.numpy()
    ids = inp.numpy()
    seen = 0
    for b in range(ids.shape[0]):
        for j in range(S):
            want = ids[b][(bag[b] == j) & (~sm[b])]
            assert list(span_ids[b][j]) == list(want), \
                f"row {b} span {j}: cutter gave {span_ids[b][j]} want {want}"
            assert int(n_tok[b, j]) == int(want.size)
            if want.size:
                assert int(h[b, j]) == span_token_hash(want)
                seen += 1
            else:
                assert int(h[b, j]) == 0
    assert seen >= 6, "fixture produced too few spans to be a real check"
    # The +1 shift, on the same arrays.
    tgt, has = next_span_hashes(h, n_tok)
    for b in range(ids.shape[0]):
        for s in range(S - 1):
            assert bool(has[b, s]) == (n_tok[b, s + 1] > 0)
            if has[b, s]:
                assert int(tgt[b, s]) == int(h[b, s + 1])
        assert not bool(has[b, S - 1]), "the last slot of the budget has no next span"


def test_the_key_is_the_token_ids_not_the_text():
    a = span_token_hash([5, 6, 7])
    assert a == span_token_hash(np.asarray([5, 6, 7], dtype=np.int64))
    assert a != span_token_hash([5, 7, 6]), "the key must be order sensitive"
    assert a != span_token_hash([5, 6, 7, 8])
    assert 0 <= a < 2 ** 64


def test_write_cache_sorts_dedupes_and_keeps_the_first(tmp_path):
    keys = np.asarray([9, 3, 9, 1], dtype=np.uint64)
    emb = np.stack([np.full(SONAR_DIM, float(i)) for i in range(4)]).astype(np.float32)
    p = str(tmp_path / "c")
    meta = write_sonar_cache(p, keys, emb)
    assert meta["n_rows"] == 3 and meta["n_written"] == 4
    c = SonarSpanCache(p)
    assert list(c.index) == [1, 3, 9]
    got = c.take(c.lookup(np.asarray([9, 3, 1], dtype=np.uint64)))
    assert got[0, 0] == 0.0, "the FIRST embedding of a duplicate key must win"
    assert got[1, 0] == 1.0 and got[2, 0] == 3.0


# ── 7. the arm composes through Hydra and the shipped key mapping ───────────────

def test_the_sonar_arm_composes(monkeypatch, tmp_path):
    import os

    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup

    inp, _lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_rule(), _rule().is_boundary, 0, ("\n",)))
    with initialize_config_dir(version_base=None,
                               config_dir=os.path.abspath("morph/configs")):
        cfg = compose(config_name="tul_code_cfg_tlow_sonar",
                      overrides=[f"tul.code_sonar_cache={path}"])
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt is not None
    tc = rt.model_cfg
    assert tc.code and tc.code_target_source == "sonar" and tc.code_sonar_cache == path
    assert tc.code_t_logit_mean == -1.0 and tc.code_t_logit_std == 1.0, \
        "the arm must keep tul_code_cfg_tlow's wide noise schedule"
    assert tc.code_cfg_drop == 0.1 and tc.code_cfg_scale == 2.0, \
        "the arm must keep the parent's guidance"
    assert rt.manifest["code_target_source"] == "sonar"
    assert rt.manifest["code_sonar_cache"] == path, \
        "the cache path must reach wandb, or the run is not reproducible from its config"
    assert str(cfg.wandb.name) == "tul-code-cfg-tlow-sonar"


def test_the_sonar_build_draws_no_rng(tmp_path):
    """A sonar arm's trainable weights must equal its "e" partner's, or the two cannot be
    paired: the frozen lift draws from a private generator with the global stream saved."""
    inp, _lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    e = _model(_sonar_tul())
    s = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path))
    pe = dict(e.named_parameters())
    ps = dict(s.named_parameters())
    assert set(pe) == set(ps), "the sonar arm changed the parameter set"
    for n in pe:
        assert torch.equal(pe[n], ps[n]), f"{n} differs — the sonar build consumed RNG"


def test_the_flow_loss_target_is_the_sonar_cell(tmp_path, monkeypatch):
    """The knob's headline claim, measured at the flow site: `cfm_pair` is handed the
    SONAR cells, not E's code. Captured through the SAME symbol `_tul_code_core` calls."""
    from morph.model import transformer as T
    import morph.model.tul_code as TC

    inp, lab, layout, _ = _batch()
    path = _write_cache(tmp_path, layout, inp)
    seen: list[torch.Tensor] = []

    def _spy(z, source_std, t, generator=None, z0=None):
        seen.append(z.detach().clone())
        return TC.cfm_pair(z, source_std, t, generator=generator, z0=z0)

    monkeypatch.setattr(T, "cfm_pair", _spy)
    m = _model(_sonar_tul(code_target_source="sonar", code_sonar_cache=path,
                          code_noise=0.0))
    m.train()
    m.code_phase = 2
    m(inp, labels=lab, slot_layout=layout)
    assert seen, "the flow site never ran — phase 2 did not reach `cfm_pair`"

    S = int(layout.slot_index.shape[1])
    ok = layout.slot_valid.clone()
    ok[:, S - 1] = False
    ok[:, :S - 1] &= layout.slot_valid[:, 1:S]
    want = code_rmsnorm(_expected_cells(layout, inp, m.tul_code_sonar_q)) \
        * ok.view(*ok.shape, 1, 1).float()
    assert torch.allclose(seen[0].float(), want, atol=1e-4), \
        f"the flow target is not the SONAR cell: max |Δ| {float((seen[0].float() - want).abs().max())}"

    # And it is NOT E's code: the SAME spy on an "e" model must catch a different target.
    seen.clear()
    e = _model(_sonar_tul(code_noise=0.0))
    e.train()
    e.code_phase = 2
    e(inp, labels=lab, slot_layout=layout)
    assert seen, "the 'e' arm never reached the flow site"
    assert not torch.allclose(seen[0].float()[ok], want[ok], atol=1e-2), \
        "E's code and the SONAR target are indistinguishable — the test proves nothing"
