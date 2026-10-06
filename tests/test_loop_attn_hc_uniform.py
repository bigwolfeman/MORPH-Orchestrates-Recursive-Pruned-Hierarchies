"""``tul.loop_attn_hc: uniform`` — the slot loop's core ATTENTION residual made a plain
residual (2026-10-05).

The data-flow probe found the slot loop's core attention reads a stream mix that is
99.9 % one shared vector and writes a large shared output through ONE Hyper-Connection
stream (Hpost row about [0, 0, 3.9, 0]). Under ``uniform`` each core block's
``mrr_attn`` is ``hyper_connections.UniformResidual``: ``Hpre = 1/n`` (stream mean in),
``Hpost_row = 1`` (the same write to every stream), ``Hres = I``, no parameters. The core
MLP's residual stays Cayley.

Files: morph/model/hyper_connections.py (``UniformResidual``), morph/model/transformer.py
(the build swap, the ``_core_region`` refusal), morph/model/tul.py (keys,
``_check_loop_attn``), morph/training/tul_setup.py (keys, manifest, banner),
morph/inference/tul_generate_cached.py (refusal), morph/configs/lxtul_hcuni.yaml.
Note: .agents/notes/rejected/architecture/2026-10-05-slot-loop-carrier-constant.md

What each test pins:
  1. Off (absent or explicit ``cayley``) is the tree: the master pins of
     tests/test_loop_attn_center.py (measured on ``git archive 4899fa0``).
  2. ``UniformResidual`` equals the hand-written plain residual exactly, with and
     without ``post_inject``, and equals the Cayley residual's own post-map
     (``hc_post_reference``) fed ``Hres = I`` / ``Hpost_row = 1``.
  3. In the model: every core block's attention residual is a ``UniformResidual`` with
     no parameters; the state dict has no ``core.*.mrr_attn.*`` key; every other tensor
     equals the same-seed Cayley model's (RNG-neutral build); prelude / coda attention
     residuals and every core MLP residual are untouched Cayley modules; the parameter
     count drops by exactly the removed projections (nothing dead in the optimizer).
  4. In a real forward, each core block's attention step is ``h + a`` broadcast to every
     stream, with the attention reading the stream mean (hooked, per call).
  5. A tiny training step: finite loss, gradients reach every core attention weight and
     the core MLP's Cayley projection.
  6. Config refusals, known key / Hydra / manifest path, the config's compose diff, the
     ``_core_region`` refusal.

Sabotage checks, run and reverted (results in the session report, not committed):
  (a) the build keeps the Cayley ``mrr_attn`` (Hres dynamic) -> tests 3 and 4 fail.
  (b) ``UniformResidual`` writes the output into stream 0 only -> tests 2 and 4 fail.
  (c) ``UniformResidual`` reads stream 0 instead of the mean -> tests 2 and 4 fail.

CPU, fp32, the tests/test_tul_fan_lsel.py LXTUL fixtures (M = 4 cells, strict geometry).
"""
from __future__ import annotations

import pathlib

import pytest
import torch

from morph.kernels.triton.fused_hyper_connection import hc_post_reference
from morph.model.hyper_connections import HyperConnectionResidual, UniformResidual
from morph.model.tul import TULConfig
from test_loop_attn_center import (_CONFIG_DIR, MASTER_PIN, _compose_leaves, _lx,
                                   _pin_run)
from test_tul_fan import _batch
from test_tul_lx_credit import M

M = int(M)


# ── 1. off is the tree ────────────────────────────────────────────────────────────


def test_explicit_cayley_matches_the_pins_measured_before_the_key():
    m = _lx(loop_attn_hc="cayley")
    assert all(isinstance(b.mrr_attn, HyperConnectionResidual) for b in m.core)
    assert _pin_run(m) == MASTER_PIN


# ── 2. the module ────────────────────────────────────────────────────────────────


def _sub():
    g = torch.Generator().manual_seed(3)
    w = torch.randn(16, 16, generator=g) / 4.0
    return lambda x: torch.tanh(x @ w)


@pytest.mark.parametrize("with_inject", [False, True])
def test_uniform_residual_is_the_hand_written_plain_residual(with_inject):
    g = torch.Generator().manual_seed(1)
    h = torch.randn(2, 5, 4, 16, generator=g)
    inj = torch.randn(2, 5, 16, generator=g) if with_inject else None
    f = _sub()
    out = UniformResidual(4)(h, f, post_inject=inj)
    x_bar = (h[:, :, 0] + h[:, :, 1] + h[:, :, 2] + h[:, :, 3]) / 4.0
    y = f(x_bar)
    want = torch.stack([h[:, :, j] + y for j in range(4)], dim=2)
    if with_inject:
        want = want + inj.unsqueeze(2)
    assert torch.allclose(out, want, rtol=0, atol=1e-6)
    # ... and exactly the Cayley residual's own post-map at Hres = I, Hpost_row = 1.
    x_bar_exact = h.mean(dim=-2)
    y_exact = f(x_bar_exact)
    eye = torch.eye(4).expand(2, 5, 4, 4)
    ones = torch.ones(2, 5, 4)
    assert torch.equal(out, hc_post_reference(eye, ones, h, y_exact, inj))
    assert list(UniformResidual(4).parameters()) == []


# ── 3. the build ──────────────────────────────────────────────────────────────────


def test_build_swaps_only_the_core_attention_residual_and_leaves_nothing_dead():
    cay, uni = _lx(), _lx(loop_attn_hc="uniform")
    for blk in uni.core:
        assert isinstance(blk.mrr_attn, UniformResidual)
        assert list(blk.mrr_attn.parameters()) == []
        assert isinstance(blk.mrr_mlp, HyperConnectionResidual)
    for blk in list(uni.prelude) + list(uni.coda):
        assert isinstance(blk.mrr_attn, HyperConnectionResidual)
        assert isinstance(blk.mrr_mlp, HyperConnectionResidual)
    sc, su = cay.state_dict(), uni.state_dict()
    gone = set(sc) - set(su)
    assert gone == {f"core.{i}.mrr_attn.proj.{w}" for i in range(len(uni.core))
                    for w in ("weight", "bias")}, sorted(gone)
    assert set(su) <= set(sc)
    for k in su:
        assert torch.equal(sc[k], su[k]), k
    n_proj = sum(p.numel() for b in cay.core for p in b.mrr_attn.parameters())
    assert n_proj > 0
    assert (sum(p.numel() for p in cay.parameters())
            - sum(p.numel() for p in uni.parameters())) == n_proj
    assert not any(".mrr_attn." in n and n.startswith("core.")
                   for n, _ in uni.named_parameters())


# ── 4. in the forward ─────────────────────────────────────────────────────────────


def test_core_attention_step_is_h_plus_a_on_every_stream():
    m = _lx(loop_attn_hc="uniform")
    _ids, inp, lab, layout = _batch(M)
    rec = []
    hs = []
    for i, blk in enumerate(m.core):
        cur: dict = {}

        def res_pre(mod, args, cur=cur):
            cur["h"] = args[0].detach().clone()

        def norm_pre(mod, args, cur=cur):
            cur["x_bar"] = args[0].detach().clone()

        def attn_out(mod, args, out, cur=cur):
            cur["a"] = out.detach().clone()

        def res_post(mod, args, out, cur=cur, i=i):
            rec.append((i, cur["h"], cur["x_bar"], cur["a"], out.detach().clone()))

        hs += [blk.mrr_attn.register_forward_pre_hook(res_pre),
               blk.norm_attn.register_forward_pre_hook(norm_pre),
               blk.attention.register_forward_hook(attn_out),
               blk.mrr_attn.register_forward_hook(res_post)]
    m.eval()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    for h in hs:
        h.remove()
    assert {r[0] for r in rec} == set(range(len(m.core)))
    for i, h, x_bar, a, out in rec:
        assert torch.equal(x_bar, h.mean(dim=-2)), i            # Hpre = 1/n
        assert torch.equal(out, h + a.unsqueeze(-2)), i         # Hres = I, Hpost_row = 1
        # the write is not trivially zero, and the streams differ (Hres = I keeps them)
        assert float(a.abs().max()) > 0.0
        assert not torch.equal(out[:, :, 0], out[:, :, 1])


# ── 5. a tiny training step ──────────────────────────────────────────────────────


def test_tiny_training_step_is_finite_and_reaches_the_core_attention_and_mlp_hc():
    m = _lx(loop_attn_hc="uniform")
    _ids, inp, lab, layout = _batch(M)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    for step in range(2):
        m.train()
        torch.manual_seed(30 + step)
        out = m(inp, labels=lab, slot_layout=layout)
        assert torch.isfinite(out["loss"])
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        for i, blk in enumerate(m.core):
            gs = [p.grad for p in blk.attention.parameters() if p.requires_grad]
            assert gs and all(g is not None and torch.isfinite(g).all() for g in gs), i
            assert sum(float(g.abs().sum()) for g in gs) > 0.0, i
            gp = blk.mrr_mlp.proj.weight.grad
            assert gp is not None and float(gp.abs().sum()) > 0.0, i
        opt.step()


def test_center_and_uniform_compose():
    m = _lx(loop_attn_hc="uniform", loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    m.train()
    torch.manual_seed(2)
    out = m(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    assert torch.isfinite(out["loss"])
    assert all(int(c.n_updates) == 1 for c in m._loop_attn_centers)
    assert all(isinstance(b.mrr_attn, UniformResidual) for b in m.core)


# ── 6. refusals, keys, configs ────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,msg", [
    (dict(loop_attn_hc="sinkhorn"), ValueError, "loop_attn_hc must be one of"),
    (dict(loop_attn_hc="uniform", tokens_through_core=True), NotImplementedError,
     "tokens_through_core"),
    (dict(loop_attn_hc="uniform", loop_reads_tokens=True), NotImplementedError,
     "loop_reads_tokens"),
    (dict(loop_attn_hc="uniform", code=True), NotImplementedError, "code:"),
    (dict(loop_attn_hc="uniform", xhc_streams=8), NotImplementedError, "xhc_streams"),
])
def test_config_refusals(kw, exc, msg):
    if exc is NotImplementedError:
        # this key's own refusal, not some other check that fires first
        msg = r"tul\.loop_attn_center=.* with tul\." + msg
    with pytest.raises(exc, match=msg):
        TULConfig(prefix_k=2, slot_id=4, **kw)


@pytest.mark.parametrize("key", [dict(loop_attn_hc="uniform"), dict(loop_attn_center="ema")])
@pytest.mark.parametrize("model_kw,msg", [
    (dict(core_impl="parcae"), r"with model\.core_impl='parcae'"),
    (dict(n_core=0), r"with n_core=0"),
    (dict(scse_enabled=True), r"with SCSE / core_init_scale"),
])
def test_model_level_refusals(key, model_kw, msg):
    """Built on a plain (non tg_restrict) TUL config so no OTHER refusal fires first;
    the match names this key's own message."""
    from morph.model.transformer import MORPHTransformer
    from test_tul_fan import _tiny
    with pytest.raises(NotImplementedError, match=r"tul\.loop_attn_center=.*" + msg):
        MORPHTransformer(_tiny(tul=TULConfig(prefix_k=2, slot_id=4, **key), **model_kw))


def test_core_region_refuses_a_uniform_model():
    m = _lx(loop_attn_hc="uniform")
    x = torch.zeros(1, 4, m._n_streams, m.cfg.d_model)
    with pytest.raises(RuntimeError, match="core runs inside `_tul_core` only"):
        m._core_region(x, x, None)


def test_hydra_path_and_manifest(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_hcuni")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.loop_attn_hc == "uniform"
    assert rt.model_cfg.loop_attn_center == "off"
    assert rt.manifest["loop_attn_hc"] == "uniform"


def test_lxtul_hcuni_differs_from_lxtul_by_exactly_its_keys():
    from test_slot_gain_tail import _MISSING
    a, b = _compose_leaves("lxtul_hcuni"), _compose_leaves("lxtul")
    diff = {k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING)}
    assert diff == {"tul.loop_attn_hc", "training.steps", "wandb.name"}, sorted(diff)
    assert a["tul.loop_attn_hc"] == "uniform"
    assert a["training.steps"] == 5000
    assert a["wandb.name"] == "lxtul-hcuni"


def test_cached_generator_refuses_both_keys_by_name():
    src = pathlib.Path("morph/inference/tul_generate_cached.py").read_text()
    assert '"tul.loop_attn_center (the centered slot-loop attention)"' in src
    assert '"tul.loop_attn_hc (the uniform slot-loop attention residual)"' in src
