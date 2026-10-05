"""``tul.loop_attn_center: ema`` — the slot loop's centered core attention (2026-10-05).

The data-flow probe found ~99.7 % of the slot loop's cell carrier is ONE slot-independent
vector after the first pass, built by the core attention from an input that is 99.9 %
shared. Under ``ema`` each core block's attention sublayer reads ``x_bar - mu_l``, with
``mu_l`` a per-layer fp32 BUFFER: the bias-corrected EMA of the mean of ``x_bar`` over the
valid slot cells of every pass of a training forward (``morph/model/mhc.py``,
``LoopAttnCenter``; life cycle in ``MORPHTransformer._tul_core``).

Files: morph/model/mhc.py (``LoopAttnCenter``, ``MORPHBlock.attach_attn_center`` and the
one line in ``_attn_fn``), morph/model/transformer.py (the build, the ``_tul_core``
snapshot / arm / disarm / apply, ``loop_attn_center_frozen``, the ``_core_region``
refusal), morph/model/tul.py (keys, ``_check_loop_attn``), morph/training/tul_setup.py
(``KNOWN_TUL_KEYS``, the Hydra -> TULConfig path, the manifest, the activate_at refusal,
the banner), morph/training/train.py (the compile warmup freezes the EMA),
morph/inference/tul_generate_cached.py (refusal), morph/configs/lxtul_center.yaml.
Note: .agents/notes/proposed/architecture/2026-10-05-slot-loop-carrier-constant.md

What each test pins:
  1. Off (key absent or explicit default) is bit-identical to the tree before the key:
     pins measured on ``git archive 4899fa0`` (train loss, |grad| sum, eval logit sum,
     state-dict size), plus absent == explicit on every grad.
  2. The build is RNG-neutral (every shared tensor equal) and adds exactly the two
     persistent buffers per core block; the FIRST training forward is byte-identical to
     off (mu starts at 0).
  3. The attention sublayer's input is ``x_bar - mu`` (``norm_attn``'s input, hooked,
     against the off model's at pass 0 / layer 0, where both read the same ``x_bar``).
  4. The EMA: one update per training forward, from the LOOP passes only (not the gain
     hinge's two applications, not the backward recompute), over the VALID cells only,
     with the bias-corrected rule — checked numerically over two steps against means
     the test computes itself from hooked ``x_bar``.
  5. Frozen in eval, in a train-mode no-grad forward, and inside
     ``loop_attn_center_frozen()``; train.py's warmup uses that context.
  6. The backward recompute subtracts what the forward did: checkpointed passes give the
     same loss and gradients as un-checkpointed ones with a live EMA update.
  7. Checkpoint round trip keeps ``mu`` and ``n_updates`` (``mu_used`` is not saved).
  8. A tiny training step: finite loss, gradients reach every core attention weight.
  9. Refusals, the known-key / Hydra / manifest path, the config's compose diff, and the
     ``_core_region`` runtime refusal.

Sabotage checks, run and reverted (results in the session report, not committed):
  (a) ``forward`` returns ``x`` (no subtraction) -> test 3 fails.
  (b) ``apply`` updates twice -> test 4 fails.
  (c) the weights include pad cells -> test 4 fails.
  (d) ``forward`` subtracts ``mu`` instead of the ``mu_used`` snapshot -> test 6 fails.
  (e) the arm stays on through the gain hinge -> test 4 fails.

CPU, fp32, the tests/test_tul_fan_lsel.py LXTUL fixtures (M = 4 cells, strict geometry).
"""
from __future__ import annotations

import ast
import pathlib
import traceback

import pytest
import torch

from morph.model.mhc import LoopAttnCenter
from morph.model.tul import TULConfig
from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M

M = int(M)

# The lxtul.yaml analogue on the tiny fixture: the latent-selected joint loop, router
# followed, latent weight 1, detached head input, per-pass cell RMSNorm.
LXTUL_KW = dict(slot_cell_pass_norm="rms", fan_lsel_train_follow="router",
                fan_lsel_lambda=1.0, fan_lsel_head_input="detached")

# Measured 2026-10-05 on `git archive 4899fa0` (the tree before either loop_attn key),
# one CPU thread, by the session's pin.py: the analogue below, seed 7 train forward +
# backward, then an eval forward. (loss, sum |grad|, finite logit sum, state-dict keys)
MASTER_PIN = (12.30628490447998, 2004.2731205267849, 1130.3376108596349, 244)


def _lx(**kw):
    return _lsel("joint", **{**LXTUL_KW, **kw})


def _pin_run(m):
    _ids, inp, lab, layout = _batch(M)
    m.train()
    torch.manual_seed(7)
    o = m(inp, labels=lab, slot_layout=layout)
    o["loss"].backward()
    gs = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    ls = float(torch.nan_to_num(lg.double(), nan=0.0, posinf=0.0, neginf=0.0).sum())
    return float(o["loss"].detach()), gs, ls, len(m.state_dict())


def _valid_cells(layout) -> torch.Tensor:
    return layout.slot_valid.repeat_interleave(M, dim=1)          # [B, S*M]


# ── 1. off is the tree ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kw", [{}, dict(loop_attn_center="off", loop_attn_hc="cayley",
                                         loop_attn_center_decay=0.99)])
def test_off_matches_the_pins_measured_before_the_key(kw):
    m = _lx(**kw)
    assert m._loop_attn_centers == () and not m._loop_attn_on
    assert all(b.attn_center is None for b in m.core)
    loss, gs, ls, nk = _pin_run(m)
    assert loss == MASTER_PIN[0]
    assert gs == MASTER_PIN[1]
    assert ls == MASTER_PIN[2]
    assert nk == MASTER_PIN[3]


def test_off_absent_and_explicit_have_equal_grads():
    a, b = _lx(), _lx(loop_attn_center="off")
    _ids, inp, lab, layout = _batch(M)
    for m in (a, b):
        m.train()
        torch.manual_seed(3)
        m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb
        assert (pa.grad is None) == (pb.grad is None), na
        if pa.grad is not None:
            assert torch.equal(pa.grad, pb.grad), na


# ── 2. the build ──────────────────────────────────────────────────────────────────


def test_build_is_rng_neutral_and_adds_two_persistent_buffers_per_core_block():
    off, on = _lx(), _lx(loop_attn_center="ema")
    so, sn = off.state_dict(), on.state_dict()
    extra = set(sn) - set(so)
    want = {f"core.{i}.attn_center.{b}" for i in range(len(on.core))
            for b in ("mu", "n_updates")}
    assert extra == want, sorted(extra ^ want)
    for k in so:
        assert torch.equal(so[k], sn[k]), k
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
    for i, blk in enumerate(on.core):
        c = blk.attn_center
        assert isinstance(c, LoopAttnCenter) and c is on._loop_attn_centers[i]
        assert c.mu.dtype == torch.float32 and c.mu.shape == (on.cfg.d_model,)
        assert bool((c.mu == 0).all()) and int(c.n_updates) == 0 and c.decay == 0.99
    assert all(b.attn_center is None for b in list(on.prelude) + list(on.coda))


def test_first_training_forward_is_byte_identical_to_off():
    """mu starts at 0, so the first forward subtracts exactly 0."""
    off, on = _lx(), _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    outs = []
    for m in (off, on):
        m.train()
        torch.manual_seed(5)
        outs.append(m(inp, labels=lab, slot_layout=layout)["loss"])
    assert torch.equal(outs[0], outs[1])


# ── 3. the attention reads x_bar - mu ────────────────────────────────────────────


def _capture_sublayer_io(m, which: str):
    """Per core layer, every (x_bar the residual hands its sublayer, the input the
    sublayer's norm actually receives), forward calls only. `x_bar` is captured by
    wrapping the residual's `sublayer_fn`, independently of `LoopAttnCenter`."""
    xb: dict[int, list] = {}
    nin: dict[int, list] = {}
    undo = []
    for i, blk in enumerate(m.core):
        res = getattr(blk, "mrr_attn" if which == "attn" else "mrr_mlp")
        norm = getattr(blk, "norm_attn" if which == "attn" else "norm_mlp")
        orig = res.forward

        def wrapped(h, fn, *a, _orig=orig, _i=i, **k):
            def fn2(x, *aa, **kk):
                xb.setdefault(_i, []).append(x.detach().clone())
                return fn(x, *aa, **kk)
            return _orig(h, fn2, *a, **k)
        res.forward = wrapped
        hk = norm.register_forward_pre_hook(
            lambda mod, args, _i=i: nin.setdefault(_i, []).append(args[0].detach().clone()))
        undo.append((res, hk))
    return xb, nin, undo


def _undo(undo):
    for res, hk in undo:
        del res.forward
        hk.remove()


def test_attention_sublayer_reads_x_bar_minus_mu_and_the_mlp_reads_x_bar():
    m = _lx(loop_attn_center="ema")
    g = torch.Generator().manual_seed(11)
    mus = [torch.randn(m.cfg.d_model, generator=g) for _ in m.core]
    for c, mu in zip(m._loop_attn_centers, mus):
        c.mu.copy_(mu)
        c.n_updates.fill_(5)
    _ids, inp, lab, layout = _batch(M)
    m.eval()
    for which in ("attn", "mlp"):
        xb, nin, undo = _capture_sublayer_io(m, which)
        with torch.no_grad():
            m(inp, labels=lab, slot_layout=layout)
        _undo(undo)
        for i in range(len(m.core)):
            assert len(xb[i]) == len(nin[i]) >= 1, (which, i)
            for x, n in zip(xb[i], nin[i]):
                if which == "attn":
                    assert torch.equal(n, x - mus[i]), i
                    assert not torch.equal(n, x)
                else:
                    assert torch.equal(n, x), i


def test_off_model_attention_reads_x_bar():
    """The capture itself is not vacuous: on the off model the attention norm reads x_bar."""
    m = _lx()
    _ids, inp, lab, layout = _batch(M)
    m.eval()
    xb, nin, undo = _capture_sublayer_io(m, "attn")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
    _undo(undo)
    for i in range(len(m.core)):
        assert xb[i] and all(torch.equal(n, x) for x, n in zip(xb[i], nin[i]))


# ── 4. the EMA rule ───────────────────────────────────────────────────────────────


def _loop_pass_means(m, inp, lab, layout, seed):
    """Run one train forward + backward; return, per core layer, the list of valid-cell
    means of x_bar at the LOOP passes only. Calls inside the gain hinge
    (`_slot_gain_penalty` on the stack) and the backward recompute (no `_tul_core`
    frame) are excluded by stack inspection, independently of the module's arming."""
    valid = _valid_cells(layout)
    rec: dict[int, list] = {}
    hs = []
    for i, blk in enumerate(m.core):
        def f(mod, args, i=i):
            names = [fr.name for fr in traceback.extract_stack()]
            if "_tul_core" not in names or "_slot_gain_penalty" in names:
                return
            x = args[0].detach().double()
            rec.setdefault(i, []).append(x[valid].mean(dim=0))
        hs.append(blk.attn_center.register_forward_pre_hook(f))
    m.train()
    torch.manual_seed(seed)
    m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    for h in hs:
        h.remove()
    m.zero_grad(set_to_none=True)
    return rec


def test_ema_is_the_bias_corrected_mean_over_valid_loop_passes_once_per_step():
    m = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    valid = _valid_cells(layout)
    assert bool(valid.any()) and not bool(valid.all()), "fixture must have pad cells"
    d = 0.99
    # step 1: alpha = 1, mu = m1 exactly (up to the summation order of the mean)
    r1 = _loop_pass_means(m, inp, lab, layout, seed=1)
    m1 = [torch.stack(r1[i]).mean(0) for i in range(len(m.core))]
    T1 = {len(r1[i]) for i in r1}
    assert len(T1) == 1 and T1.pop() >= 1
    for i, c in enumerate(m._loop_attn_centers):
        assert int(c.n_updates) == 1, "one update per training forward (fwd + bwd)"
        assert torch.allclose(c.mu.double(), m1[i], rtol=0, atol=1e-5), i
    mu1 = [c.mu.double().clone() for c in m._loop_attn_centers]
    # step 2: alpha = (1 - d) / (1 - d^2)
    r2 = _loop_pass_means(m, inp, lab, layout, seed=2)
    a2 = (1 - d) / (1 - d ** 2)
    for i, c in enumerate(m._loop_attn_centers):
        m2 = torch.stack(r2[i]).mean(0)
        want = mu1[i] + a2 * (m2 - mu1[i])
        assert int(c.n_updates) == 2
        assert torch.allclose(c.mu.double(), want, rtol=0, atol=1e-5), i
        # and it is not the all-cell mean: pads are excluded
        assert not torch.allclose(m1[i], torch.zeros_like(m1[i]))


def test_pad_cells_do_not_move_the_ema():
    """mu equals the valid-only mean of the loop passes and NOT the all-cell mean, which is
    what including the pad cells would give (the fixture's pads sit far enough from the
    valid cells for the two means to differ by > 1e-3)."""
    m = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    valid = _valid_cells(layout)
    allc: dict[int, list] = {}
    hs = []
    for i, blk in enumerate(m.core):
        def f(mod, args, i=i):
            names = [fr.name for fr in traceback.extract_stack()]
            if "_tul_core" in names and "_slot_gain_penalty" not in names:
                allc.setdefault(i, []).append(args[0].detach().double().mean(dim=(0, 1)))
        hs.append(blk.attn_center.register_forward_pre_hook(f))
    rec = _loop_pass_means(m, inp, lab, layout, seed=4)
    for h in hs:
        h.remove()
    for i, c in enumerate(m._loop_attn_centers):
        m_valid = torch.stack(rec[i]).mean(0)
        m_all = torch.stack(allc[i]).mean(0)
        assert (m_valid - m_all).abs().max() > 1e-3, "fixture pads indistinguishable"
        assert torch.allclose(c.mu.double(), m_valid, rtol=0, atol=1e-5)
        assert not torch.allclose(c.mu.double(), m_all, rtol=0, atol=1e-4)
    assert bool((~valid).any())


# ── 5. frozen outside a training forward ─────────────────────────────────────────


def _state(m):
    return [(c.mu.clone(), int(c.n_updates)) for c in m._loop_attn_centers]


def _same(a, b):
    return all(torch.equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


def test_eval_and_no_grad_forwards_do_not_move_the_ema():
    m = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    _loop_pass_means(m, inp, lab, layout, seed=1)        # mu != 0 now
    s0 = _state(m)
    m.eval()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)
        m(inp, labels=None, slot_layout=layout)
    m(inp, labels=lab, slot_layout=layout)               # eval mode WITH grad
    assert _same(s0, _state(m))
    m.train()
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout)           # train mode, no grad
    assert _same(s0, _state(m))
    with m.loop_attn_center_frozen():
        m(inp, labels=lab, slot_layout=layout)["loss"].backward()
    assert _same(s0, _state(m))
    assert m._loop_attn_center_frozen is False
    m(inp, labels=lab, slot_layout=layout)
    assert all(int(c.n_updates) == 2 for c in m._loop_attn_centers)


def test_train_py_warmup_runs_under_the_frozen_context():
    src = pathlib.Path("morph/training/train.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "warmup_compile_all_shapes")
    body = ast.unparse(fn)
    i_ctx = body.index("model.loop_attn_center_frozen()")
    i_fwd = body.index("model(ids, labels=labs, slot_layout=layout)")
    assert i_ctx < i_fwd


# ── 6. the backward recompute subtracts what the forward did ─────────────────────


def test_checkpointed_and_eager_passes_give_the_same_gradients_under_a_live_update():
    """`ckpt_grad_iters=-1` recomputes every grad pass in the backward, AFTER the EMA
    update has landed on `mu`; `0` keeps the activations. Equal gradients mean the
    recompute read the forward's value."""
    base = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    _loop_pass_means(base, inp, lab, layout, seed=1)      # n_updates 1, mu != 0
    sd = base.state_dict()
    res = []
    for ck in (-1, 0):
        m = _lx(loop_attn_center="ema", model_kw={"ckpt_grad_iters": ck})
        m.load_state_dict(sd)
        m.train()
        mu_before = [c.mu.clone() for c in m._loop_attn_centers]
        torch.manual_seed(9)
        out = m(inp, labels=lab, slot_layout=layout)
        out["loss"].backward()
        assert all(not torch.equal(c.mu, mb)
                   for c, mb in zip(m._loop_attn_centers, mu_before)), "update must move mu"
        res.append((out["loss"].detach(),
                    {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None},
                    [c.mu.clone() for c in m._loop_attn_centers]))
    (l0, g0, mu0), (l1, g1, mu1) = res
    assert torch.equal(l0, l1)
    assert g0.keys() == g1.keys()
    for n in g0:
        assert torch.allclose(g0[n], g1[n], rtol=1e-5, atol=1e-7), n
    for a, b in zip(mu0, mu1):
        assert torch.equal(a, b)


# ── 7. checkpoint round trip ──────────────────────────────────────────────────────


def test_checkpoint_round_trip_keeps_mu_and_count():
    m = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    _loop_pass_means(m, inp, lab, layout, seed=1)
    _loop_pass_means(m, inp, lab, layout, seed=2)
    sd = m.state_dict()
    assert not any(k.endswith("mu_used") for k in sd)
    fresh = _lx(loop_attn_center="ema")
    fresh.load_state_dict(sd)
    for a, b in zip(m._loop_attn_centers, fresh._loop_attn_centers):
        assert torch.equal(a.mu, b.mu) and int(a.n_updates) == int(b.n_updates) == 2
    m.eval()
    fresh.eval()
    with torch.no_grad():
        la = m(inp, labels=None, slot_layout=layout)["logits"]
        lb = fresh(inp, labels=None, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


# ── 8. a tiny training step ──────────────────────────────────────────────────────


def test_tiny_training_step_is_finite_and_reaches_the_core_attention():
    m = _lx(loop_attn_center="ema")
    _ids, inp, lab, layout = _batch(M)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    for step in range(2):
        m.train()
        torch.manual_seed(20 + step)
        out = m(inp, labels=lab, slot_layout=layout)
        assert torch.isfinite(out["loss"])
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        for i, blk in enumerate(m.core):
            gs = [p.grad for p in blk.attention.parameters() if p.requires_grad]
            assert gs and all(g is not None and torch.isfinite(g).all() for g in gs), i
            assert sum(float(g.abs().sum()) for g in gs) > 0.0, i
        opt.step()
    assert all(int(c.n_updates) == 2 for c in m._loop_attn_centers)
    assert all(torch.isfinite(c.mu).all() for c in m._loop_attn_centers)


# ── 9. refusals, keys, configs ────────────────────────────────────────────────────


@pytest.mark.parametrize("kw,exc,msg", [
    (dict(loop_attn_center="batch"), ValueError, "loop_attn_center must be one of"),
    (dict(loop_attn_center="ema", loop_attn_center_decay=1.0), ValueError, r"in \(0, 1\)"),
    (dict(loop_attn_center="ema", loop_attn_center_decay=0.0), ValueError, r"in \(0, 1\)"),
    (dict(loop_attn_center_decay=0.9), ValueError, "silently ignored"),
    (dict(loop_attn_center="ema", tokens_through_core=True), NotImplementedError,
     "tokens_through_core"),
    (dict(loop_attn_center="ema", loop_reads_tokens=True), NotImplementedError,
     "loop_reads_tokens"),
    (dict(loop_attn_center="ema", core_token_aux=True), NotImplementedError,
     "core_token_aux"),
    (dict(loop_attn_center="ema", core_stage_cond="sigma"), NotImplementedError,
     "core_stage_cond"),
])
def test_config_refusals(kw, exc, msg):
    if exc is NotImplementedError:
        # this key's own refusal, not some other check that fires first
        msg = r"tul\.loop_attn_center=.* with tul\." + msg
    with pytest.raises(exc, match=msg):
        TULConfig(prefix_k=2, slot_id=4, **kw)


def test_core_region_refuses_a_center_model():
    m = _lx(loop_attn_center="ema")
    x = torch.zeros(1, 4, m._n_streams, m.cfg.d_model)
    with pytest.raises(RuntimeError, match="core runs inside `_tul_core` only"):
        m._core_region(x, x, None)


def test_key_is_known_and_reaches_the_runtime(monkeypatch):
    import transformers
    from hydra import compose, initialize_config_dir

    from morph.training import tul_setup
    from test_tul_strict_geometry import _StubTok, _rule as _strict_rule

    for k in ("loop_attn_center", "loop_attn_center_decay", "loop_attn_hc"):
        assert k in KNOWN_TUL_KEYS
    reject_unknown_tul_keys({"loop_attn_center": "ema"})
    with pytest.raises(ValueError, match="unknown"):
        reject_unknown_tul_keys({"loop_attn_centre": "ema"})
    monkeypatch.setattr(transformers, "AutoTokenizer", _StubTok)
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir="": (_strict_rule(), _strict_rule().is_boundary,
                                                   0, ("\n",)))
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        cfg = compose(config_name="lxtul_center")
    rt = tul_setup.build_tul_runtime(cfg)
    assert rt.model_cfg.loop_attn_center == "ema"
    assert rt.model_cfg.loop_attn_center_decay == 0.99
    assert rt.model_cfg.loop_attn_hc == "cayley"
    assert rt.manifest["loop_attn_center"] == "ema"
    assert rt.manifest["loop_attn_center_decay"] == 0.99
    assert rt.manifest["loop_attn_hc"] == "cayley"


_CONFIG_DIR = str(pathlib.Path("morph/configs").resolve())


def _compose_leaves(name: str) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from test_slot_gain_tail import _leaves
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return _leaves(OmegaConf.to_container(compose(config_name=name), resolve=True))


def test_lxtul_center_differs_from_lxtul_by_exactly_its_keys():
    from test_slot_gain_tail import _MISSING
    a, b = _compose_leaves("lxtul_center"), _compose_leaves("lxtul")
    diff = {k for k in a.keys() | b.keys() if a.get(k, _MISSING) != b.get(k, _MISSING)}
    assert diff == {"tul.loop_attn_center", "training.steps", "wandb.name"}, sorted(diff)
    assert a["tul.loop_attn_center"] == "ema"
    assert a["training.steps"] == 5000
    assert a["wandb.name"] == "lxtul-center"
