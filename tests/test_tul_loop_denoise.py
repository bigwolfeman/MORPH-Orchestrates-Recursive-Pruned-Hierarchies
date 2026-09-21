"""``tul.loop_denoise`` — each pass of the slot loop gets a job: a noise level.

    PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 \
        python -m pytest tests/test_tul_loop_denoise.py -q

CPU only, fp32, the strict-geometry tiny fixture (``tests/test_tul_strict_geometry.py``),
the same one ``test_tul_code_target.py`` and ``test_tul_code_ref.py`` use.

WHAT THIS FILE HAS TO PROVE. Every assertion is on a VALUE, never on a shape.

1. **OFF IS THE RULER.** ``loop_denoise: false`` builds no ``TULLoopDenoiseIn`` and no
   ``TULCodeTime``, draws no RNG at build, and a whole training step — loss, logits and
   the total gradient — is bit-identical to a ``code_target`` + ``code_target_ref`` model
   built the way ``tests/test_tul_code_ref.py`` builds one. That is the regression test
   for the ``_tul_code_target_encode`` extraction as well as for the feature.
2. **THE ENTRY IS THE NOISED TARGET, AT TRAIN.** For every pass the captured state
   entering it equals ``(1 - t_i)·z0 + t_i·x0`` to the last bit, with ``t_i = i / T_s``
   the per-slot level, ``z0`` the ONE draw of that slot and ``x0`` the frozen reference
   encoder's code. ``z0`` is the SAME tensor at every pass (one line per slot), the source
   is the FROZEN code at every pass (teacher forcing, no rollout), and pass 1 enters at
   exactly ``z0`` (level 0: the target contributes nothing there).
3. **THE TERM.** One ``loop_denoise_l2_t{t}`` per REALISED pass and no more; the reported
   ``loop_denoise`` is their sum to 1e-6; ``loop_denoise_weighted`` is the weight times it
   and is inside ``loss``.
4. **THE GRADIENT REACHES THE CORE THROUGH EVERY PASS.** With the coda's CE cut off from
   the loop (``code_target_detach``, the arm's own setting) the core's gradient is nonzero
   at depth >= 2, and zeroing the term of PASS 1 ALONE changes it — so pass 1's term is
   not the only one that trains the map.
5. **THE ROLLOUT, AT EVAL.** Pass ``i+1``'s source is pass ``i``'s own prediction on the
   active slots, its entry is that prediction re-noised on the SAME line with the SAME
   ``z0``, and two eval forwards with different ``z0`` give different logits — the exit is
   a sample.
6. **THE REFUSALS.** ``loop_denoise`` without ``code_target``, without
   ``code_target_ref``, with ``code_target_weight != 0``, on a paid-loop model, with an
   unknown grid, with a truncated BPTT window, and ``_tul_core`` called without the target.
7. **THE CONFIG.** ``tul_slot_spandec_strict_denoise`` composes, resolves and builds; the
   fan variant does NOT compose and the refusal says why.
"""

from __future__ import annotations

import math

import pytest
import torch

from test_tul_strict_geometry import _pack, _tiny, _tul  # noqa: E402

import morph.model.transformer as transformer_mod
from morph.model.transformer import MORPHTransformer
from morph.model.tul_denoise import (TULLoopDenoiseIn, loop_denoise_interp,
                                     loop_denoise_levels)

DEPTH = 3


def _model(dn: bool = True, seed: int = 5, depth: int = DEPTH, **kw) -> MORPHTransformer:
    """A code-target + frozen-reference model, at a FIXED loop depth.

    The depth is fixed (``model.depth_fixed``) so a test can name pass indices; the arm
    itself runs the ordinary Poisson draw and the code is indifferent (the level grid is
    ``t / depths`` per slot, whatever ``depths`` holds).
    """
    torch.manual_seed(seed)
    tul_kw = dict(tg_geometry="strict", code_target=True, code_target_ref=True,
                  slot_depth_fixed=depth)
    if dn:
        tul_kw.update(loop_denoise=True, code_target_weight=0.0)
    tul_kw.update({k[4:]: v for k, v in kw.items() if k.startswith("tul_")})
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base = dict(n_core=2, mean_depth=depth, max_depth=depth, bptt_depth=depth,
                depth_fixed=True, retention=False, dropout=0.0,
                core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    m = MORPHTransformer(_tiny(tul=_tul(**tul_kw), **base)).float()
    m.tul_code_ref_snapshot()
    return m


def _run(m: MORPHTransformer, seed: int = 3, capture: bool = False, **kw):
    _ids, inp, lab, layout = _pack()
    m._denoise_capture = [] if capture else None
    torch.manual_seed(seed)
    out = m(inp, labels=lab, slot_layout=layout, **kw)
    cap = m._denoise_capture
    m._denoise_capture = None
    return out, cap


def _step(m: MORPHTransformer, seed: int = 3):
    """loss, a finite logit signature and the total gradient of ONE training step."""
    _ids, inp, lab, layout = _pack()
    torch.manual_seed(seed)
    out = m.train()(inp, labels=lab, slot_layout=layout)
    out["loss"].backward()
    g = sum(float(p.grad.double().abs().sum()) for p in m.parameters() if p.grad is not None)
    m.eval()
    with torch.no_grad():
        lg = m(inp, labels=None, slot_layout=layout)["logits"]
    lg = lg[torch.isfinite(lg)]
    return float(out["loss"].detach()), float(lg.double().abs().sum()), g


def _grads(m: MORPHTransformer, prefix: str) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for n, p in m.named_parameters()
                      if n.startswith(prefix) and p.grad is not None])


# ── 1. off is the ruler ──────────────────────────────────────────────────────

def test_off_builds_nothing_and_draws_no_rng():
    on, off = _model(dn=True), _model(dn=False)
    assert off.tul_loop_denoise is None and off.tul_code_time is None
    assert isinstance(on.tul_loop_denoise, TULLoopDenoiseIn)
    assert on.tul_code_time is not None, "the level has to reach the pass"
    off_p = dict(off.named_parameters())
    extra = {n for n in dict(on.named_parameters()) if n not in off_p}
    assert extra and all(n.startswith(("tul_loop_denoise.", "tul_code_time."))
                         for n in extra), sorted(extra)
    for n, p in on.named_parameters():
        if n.startswith(("tul_loop_denoise.", "tul_code_time.")):
            continue
        assert torch.equal(p, off_p[n]), f"{n}: building the denoise modules drew RNG"


def test_off_is_bit_identical_to_a_model_built_without_the_key():
    """The OFF path must equal a plain code_target + code_target_ref model, loss, logits
    and total gradient — the regression test for the hoisted target encode as well."""
    torch.manual_seed(5)
    plain = MORPHTransformer(
        _tiny(tul=_tul(tg_geometry="strict", code_target=True, code_target_ref=True,
                       slot_depth_fixed=DEPTH),
              n_core=2, mean_depth=DEPTH, max_depth=DEPTH, bptt_depth=DEPTH,
              depth_fixed=True, retention=False, dropout=0.0,
              core_fixed_point_lambda=0.0, ckpt_grad_iters=0)).float()
    plain.tul_code_ref_snapshot()
    assert _step(_model(dn=False)) == _step(plain)


def test_off_emits_no_denoise_key_and_leaves_no_stash():
    m = _model(dn=False)
    out, cap = _run(m.train(), capture=True)
    assert cap == [], "the hook fired on a model without the feature"
    assert not [k for k in out if str(k).startswith("loop_denoise")]
    assert m._loop_denoise is None


# ── 2. the entry is the noised target, at train ──────────────────────────────

def test_the_entry_of_every_pass_is_the_frozen_target_noised_to_that_level():
    m = _model().train()
    out, cap = _run(m, capture=True)
    assert len(cap) == DEPTH, f"one capture per pass, got {len(cap)}"
    z0 = cap[0]["z0"]
    _ids, _inp0, _lab0, lay0 = _pack()
    val = lay0.slot_valid
    for t, c in enumerate(cap):
        assert torch.equal(c["z0"], z0), "z0 must be ONE draw per slot, shared by its passes"
        # the level: t / T_s per slot. Valid slots run `slot_depth_fixed` passes here; a
        # PAD loops exactly once by `_pad_slot_depths`, so its own grid is t / 1.
        assert torch.allclose(c["level"][val], torch.full_like(c["level"][val], t / DEPTH))
        assert torch.allclose(c["level"][~val], torch.full_like(c["level"][~val], float(t)))
        want = loop_denoise_interp(z0, c["src"], c["level"])
        assert torch.equal(c["z_t"], want), f"pass {t}: z_t is not (1-t) z0 + t x0"
    # teacher forcing: the source is the SAME frozen code at every pass, never a rollout
    for c in cap[1:]:
        assert torch.equal(c["src"], cap[0]["src"]), "train must not roll the prediction out"
    # and it IS the frozen reference encoder's code
    _ids, inp, _lab, layout = _pack()
    z_ref, ok = m._tul_code_target_encode(None, layout, inp)
    assert torch.equal(cap[0]["src"], z_ref.float())
    assert bool(ok.any()), "the fixture must have gradeable slots"


def test_pass_one_enters_at_pure_noise_and_carries_nothing_of_the_target():
    m = _model().train()
    _out, cap = _run(m, capture=True)
    assert float(cap[0]["level"].abs().max()) == 0.0
    assert torch.equal(cap[0]["z_t"], cap[0]["z0"]), "pass 1 must be the bare source draw"


def test_pad_slots_enter_every_pass_at_exactly_zero():
    m = _model().train()
    _ids, _inp, _lab, layout = _pack()
    pad = ~layout.slot_valid
    if not bool(pad.any()):
        pytest.skip("the fixture packed no pad slots")
    _out, cap = _run(m, capture=True)
    for c in cap:
        assert float(c["z_t"][pad].abs().max()) == 0.0


def test_the_entry_is_the_denoise_map_of_z_t_plus_the_level_embedding():
    """The captured carrier equals `TULLoopDenoiseIn(z_t)` broadcast over the HC streams
    plus `TULCodeTime(level)` — so nothing else is quietly added at that seam."""
    m = _model().train()
    # `TULCodeTime.l2` is ZERO-init, so at init the level term is exactly 0 and dropping
    # it would be invisible. Move it off zero first, or this test proves nothing.
    with torch.no_grad():
        g = torch.Generator().manual_seed(77)
        m.tul_code_time.l2.weight.copy_(
            torch.empty(m.tul_code_time.l2.weight.shape).normal_(0.0, 0.05, generator=g))
        m.tul_code_time.l2.bias.copy_(
            torch.empty(m.tul_code_time.l2.bias.shape).normal_(0.0, 0.05, generator=g))
    _ids, _inp, _lab, layout = _pack()
    _out, cap = _run(m, capture=True)
    for c in cap:
        term = m.tul_code_time(c["level"])
        assert float(term.abs().max()) > 0.0, "the level term must not be zero here"
        v = m.tul_loop_denoise(c["z_t"].to(c["entry"].dtype), layout.slot_valid)
        if m._is_hc:
            v = v.unsqueeze(2).expand(-1, -1, m._n_streams, -1)
        want = m._apply_injection(v, term.to(v.dtype))
        assert torch.equal(c["entry"], want)
    # and the term DIFFERS across passes: the level is what distinguishes them
    t0 = m.tul_code_time(cap[0]["level"])
    t1 = m.tul_code_time(cap[1]["level"])
    assert not torch.equal(t0, t1), "every pass received the same level embedding"


def test_the_map_is_the_cell_mean_at_init():
    """`W_in` is `I / M`, so the entry map of M identical unit-RMS cells is that cell."""
    m = _model()
    M = int(m.cfg.tul.prefix_k)
    z = torch.randn(2, 3, M, m.cfg.d_model)
    valid = torch.ones(2, 3, dtype=torch.bool)
    assert torch.allclose(m.tul_loop_denoise(z, valid), z.mean(dim=2), atol=1e-6)


# ── 3. the term ──────────────────────────────────────────────────────────────

def test_one_term_per_realised_pass_and_the_total_is_their_sum():
    m = _model().train()
    out, _cap = _run(m)
    keys = [f"loop_denoise_l2_t{t}" for t in range(DEPTH)]
    for k in keys:
        assert k in out, k
    assert f"loop_denoise_l2_t{DEPTH}" not in out, "a term for a pass that never ran"
    total = sum(float(out[k]) for k in keys)
    assert float(out["loop_denoise"]) == pytest.approx(total, abs=1e-6)
    assert float(out["loop_denoise"]) > 0.0
    w = float(m.cfg.tul.loop_denoise_weight)
    assert float(out["loop_denoise_weighted"]) == pytest.approx(w * total, abs=1e-6)


def test_the_weighted_term_is_inside_the_loss():
    a = _model(tul_loop_denoise_weight=1.0).train()
    b = _model(tul_loop_denoise_weight=0.0).train()
    out_a, _ = _run(a)
    out_b, _ = _run(b)
    assert float(out_b["loop_denoise_weighted"]) == 0.0
    # same weights, same batch, same seed: the loss differs by exactly the weighted term
    assert float(out_a["loss"].detach()) - float(out_b["loss"].detach()) == pytest.approx(
        float(out_a["loop_denoise_weighted"]), abs=1e-5)


def test_the_per_pass_level_reading_matches_the_grid():
    m = _model().train()
    out, _cap = _run(m)
    # the reading is the mean over the slots the pass GRADED, which are valid by
    # construction, so it is exactly the valid slots' level.
    for t in range(DEPTH):
        assert float(out[f"loop_denoise_level_t{t}"]) == pytest.approx(t / DEPTH, abs=1e-6)


def test_the_last_pass_prediction_is_the_exit_cell_the_coda_reads():
    """Why `code_target_weight` must be 0: the last realised pass's prediction IS the
    exit, so its term is already the last entry of the per-pass sum."""
    m = _model().train()
    out, cap = _run(m, capture=True)
    last = cap[-1]
    # `code_target_cos` is the exit cell's cosine to the code; the last pass's own cosine
    # is read on the same mask, so the two must agree.
    assert float(out["loop_denoise_cos_t2"]) == pytest.approx(
        float(out["code_target_cos"]), abs=1e-5)
    assert float(last["mask"].sum()) > 0


# ── 4. the gradient reaches the core through every pass ──────────────────────

def test_the_core_gets_gradient_and_pass_one_is_not_the_only_source(monkeypatch):
    base = _model().train()
    out, _ = _run(base)
    out["loss"].backward()
    g_all = _grads(base, "core.")
    assert float(g_all.abs().sum()) > 0.0, "no gradient reached the core at all"

    # zero the term of PASS 1 ALONE. The per-pass calls happen inside `_tul_core`, so the
    # FIRST call of the forward is pass 1's; the write's own call comes after the loop.
    orig = transformer_mod.code_target_regression
    state = {"n": 0}

    def cut(pred, z, ok):
        loss, cos, n = orig(pred, z, ok)
        state["n"] += 1
        if state["n"] == 1:
            loss = loss * 0.0
        return loss, cos, n

    monkeypatch.setattr(transformer_mod, "code_target_regression", cut)
    cut_m = _model().train()
    out2, _ = _run(cut_m)
    out2["loss"].backward()
    g_cut = _grads(cut_m, "core.")
    assert state["n"] >= DEPTH
    assert not torch.equal(g_all, g_cut), "pass 1's term is the only one training the core"
    assert float((g_all - g_cut).abs().sum()) > 1e-9


def test_the_projection_and_the_entry_map_both_train():
    m = _model().train()
    out, _ = _run(m)
    out["loss"].backward()
    assert float(_grads(m, "tul_code_proj.").abs().sum()) > 0.0
    assert float(_grads(m, "tul_loop_denoise.").abs().sum()) > 0.0
    # the level embedding trains too: `l2` is zero-init, so `l1` reads 0 at step 0 and
    # `l2` is the tensor that must move first.
    assert float(m.tul_code_time.l2.weight.grad.abs().sum()) > 0.0


def test_the_frozen_encoder_and_the_twin_never_get_gradient():
    m = _model().train()
    out, _ = _run(m)
    out["loss"].backward()
    assert all(p.grad is None for p in m.tul_code_enc.parameters())
    assert all(p.grad is None for p in m.code_ref.parameters())


# ── 5. the rollout, at eval ──────────────────────────────────────────────────

def test_at_eval_pass_two_enters_at_the_re_noised_prediction_of_pass_one():
    m = _model().eval()
    with torch.no_grad():
        _out, cap = _run(m, capture=True)
    assert len(cap) == DEPTH
    z0 = cap[0]["z0"]
    for t in range(DEPTH - 1):
        mask = cap[t]["mask"]
        pred = cap[t]["pred"].float()
        src = cap[t + 1]["src"]
        assert torch.equal(src[mask], pred[mask]), f"pass {t + 2} did not read pass {t + 1}"
        want = loop_denoise_interp(z0, src, cap[t + 1]["level"])
        assert torch.equal(cap[t + 1]["z_t"], want)
        assert not torch.equal(src, cap[t]["src"]), "the rollout did not move"


def test_two_eval_forwards_with_different_z0_give_different_exits():
    m = _model().eval()
    _ids, inp, lab, layout = _pack()
    with torch.no_grad():
        torch.manual_seed(11)
        a = m(inp, labels=None, slot_layout=layout)
        torch.manual_seed(12)
        b = m(inp, labels=None, slot_layout=layout)
        torch.manual_seed(11)
        c = m(inp, labels=None, slot_layout=layout)
    la, lb, lc = (x["logits"] for x in (a, b, c))
    fin = torch.isfinite(la) & torch.isfinite(lb)
    assert torch.equal(la, lc), "same seed must give the same sample"
    assert not torch.equal(la[fin], lb[fin]), "the exit is not a sample"
    assert float((la[fin] - lb[fin]).abs().max()) > 1e-4


def test_no_per_pass_term_is_built_at_eval():
    m = _model().eval()
    with torch.no_grad():
        out, _ = _run(m)
    assert not [k for k in out if str(k).startswith("loop_denoise")]
    assert m._loop_denoise is None


def test_a_forced_depth_table_sets_the_grid():
    """The K-curve instrument: `slot_depths` forces T_s, and the levels follow it."""
    m = _model().eval()
    _ids, inp, lab, layout = _pack()
    table = torch.full_like(layout.slot_index, 2)
    m._denoise_capture = []
    with torch.no_grad():
        torch.manual_seed(4)
        m(inp, labels=lab, slot_layout=layout, slot_depths=table)
    cap, m._denoise_capture = m._denoise_capture, None
    val = layout.slot_valid
    assert len(cap) == 2, "the forced table decides how many passes run"
    assert float(cap[0]["level"].abs().max()) == 0.0
    assert torch.allclose(cap[1]["level"][val], torch.full_like(cap[1]["level"][val], 0.5))


# ── 6. the refusals ──────────────────────────────────────────────────────────

def test_config_refusals():
    with pytest.raises(ValueError, match="needs tul.code_target"):
        _tul(tg_geometry="strict", loop_denoise=True)
    with pytest.raises(ValueError, match="code_target_ref"):
        _tul(tg_geometry="strict", loop_denoise=True, code_target=True)
    with pytest.raises(ValueError, match="code_target_weight=0"):
        _tul(tg_geometry="strict", loop_denoise=True, code_target=True,
             code_target_ref=True)
    with pytest.raises(NotImplementedError, match="SLOT loop"):
        _tul(loop_denoise=True, code_target=True, code_target_ref=True,
             code_target_weight=0.0, tokens_through_core=True)
    with pytest.raises(ValueError, match="loop_denoise_grid"):
        _tul(tg_geometry="strict", loop_denoise=True, code_target=True,
             code_target_ref=True, code_target_weight=0.0, loop_denoise_grid="cosine")
    with pytest.raises(ValueError, match="silently ignored"):
        _tul(loop_denoise_grid="cosine")
    with pytest.raises(NotImplementedError, match="code_target_loss"):
        _tul(tg_geometry="strict", loop_denoise=True, code_target=True,
             code_target_ref=True, code_target_weight=0.0, code_target_loss="infonce")


def test_the_fan_does_not_compose_and_the_refusal_says_why():
    with pytest.raises(NotImplementedError, match="slot_cells=4"):
        _tul(tg_geometry="strict", loop_denoise=True, code_target=True,
             code_target_ref=True, code_target_weight=0.0,
             fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all")


def test_a_truncated_bptt_window_is_refused_at_build():
    with pytest.raises(ValueError, match="FULL BPTT"):
        MORPHTransformer(_tiny(
            tul=_tul(tg_geometry="strict", code_target=True, code_target_ref=True,
                     loop_denoise=True, code_target_weight=0.0),
            n_core=2, mean_depth=4, max_depth=4, bptt_depth=2, retention=False,
            dropout=0.0, core_fixed_point_lambda=0.0, ckpt_grad_iters=0))


def test_tul_core_without_the_target_raises_instead_of_running_the_feature_off():
    m = _model().train()
    _ids, inp, _lab, layout = _pack()
    x, x0, bigram = m._tul_front(inp, layout)
    with pytest.raises(RuntimeError, match="needs `code_x0`"):
        m._tul_core(x, x0, bigram, layout, input_ids=inp)


def test_halt_is_refused_because_the_grid_needs_the_depth_in_advance():
    m = _model(tul_gate=None).eval()
    _ids, inp, _lab, layout = _pack()
    x, x0, bigram = m._tul_front(inp, layout)
    z, ok = m._tul_code_target_encode(None, layout, inp)
    with pytest.raises(RuntimeError, match="halt=True"):
        m._tul_core(x, x0, bigram, layout, halt=True, input_ids=inp,
                    code_x0=z, code_ok=ok)


# ── the arithmetic, on its own ───────────────────────────────────────────────

def test_the_level_grid_is_per_slot():
    depths = torch.tensor([[2, 6, 1]])
    assert torch.equal(loop_denoise_levels(depths, 0), torch.zeros(1, 3))
    assert torch.allclose(loop_denoise_levels(depths, 1),
                          torch.tensor([[0.5, 1.0 / 6.0, 1.0]]))
    assert torch.allclose(loop_denoise_levels(depths, 5),
                          torch.tensor([[2.5, 5.0 / 6.0, 5.0]]))
    with pytest.raises(ValueError, match="unknown grid"):
        loop_denoise_levels(depths, 0, "cosine")


def test_the_last_pass_never_enters_at_the_target_itself():
    """`t_i = (i-1)/T_s` tops out at `(T_s-1)/T_s < 1`: a pass handed the target exactly
    would have nothing left to do."""
    for T in (1, 2, 6, 8):
        d = torch.tensor([[T]])
        assert float(loop_denoise_levels(d, T - 1)) == pytest.approx((T - 1) / T)
        assert float(loop_denoise_levels(d, T - 1)) < 1.0


def test_the_entry_rms_follows_the_straight_line():
    """Named in the module docstring: the path is NOT variance preserving. With unit-RMS
    x0 orthogonal to z0 the entry's RMS dips to about 0.707 at t = 0.5."""
    C = 4096
    g = torch.Generator().manual_seed(0)
    z0 = torch.randn(1, 1, 1, C, generator=g)
    x0 = torch.randn(1, 1, 1, C, generator=g)
    x0 = x0 / x0.pow(2).mean().sqrt()
    for t, want in ((0.0, 1.0), (0.5, math.sqrt(0.5)), (0.75, math.sqrt(0.625))):
        z = loop_denoise_interp(z0, x0, torch.tensor([[t]]))
        assert float(z.pow(2).mean().sqrt()) == pytest.approx(want, rel=0.05)


# ── 7. the config ────────────────────────────────────────────────────────────

def test_the_shipped_config_composes_and_builds():
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.tul_setup import build_tul_runtime
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "morph", "configs"))
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_denoise")
    assert cfg.wandb.name == "slot-spandec-strict-denoise"
    assert cfg.tul.loop_denoise is True
    assert cfg.tul.code_target is True and cfg.tul.code_target_ref is True
    assert float(cfg.tul.code_target_weight) == 0.0
    assert float(cfg.model.core_fixed_point_lambda) == 0.0
    assert int(cfg.model.bptt_depth) >= int(cfg.model.max_depth)
    rt = build_tul_runtime(cfg)
    assert rt.model_cfg.loop_denoise is True
    assert rt.model_cfg.loop_denoise_grid == "linear"
    assert rt.manifest["loop_denoise"] is True
    assert rt.manifest["loop_denoise_grid"] == "linear"
    assert rt.manifest["loop_denoise_weight"] == 1.0


def test_unknown_tul_keys_still_raise_and_the_new_ones_do_not():
    from omegaconf import OmegaConf
    from morph.training.tul_setup import reject_unknown_tul_keys
    reject_unknown_tul_keys(OmegaConf.create(
        {"loop_denoise": True, "loop_denoise_grid": "linear", "loop_denoise_weight": 1.0}))
    with pytest.raises(ValueError, match="unknown key"):
        reject_unknown_tul_keys(OmegaConf.create({"loop_denoise_schedule": "linear"}))


def test_the_shipped_config_leaves_the_new_modules_trainable():
    """`train_only` is a REPLACED list, not an extended one. `TULCodeTime`'s last layer is
    zero-init, so a frozen level embedding would add exactly 0 forever and every pass
    would be pass 1 with a different input — the one thing this arm exists to stop."""
    import os
    from hydra import compose, initialize_config_dir
    from morph.training.freeze import apply_train_only
    d = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "morph", "configs"))
    with initialize_config_dir(config_dir=d, version_base=None):
        cfg = compose(config_name="tul_slot_spandec_strict_denoise")
    prefixes = list(cfg.training.train_only)
    m = _model()
    _nt, _nf, groups = apply_train_only(m, prefixes)
    assert _nf > 0, "the arm freezes the VAE stage"
    assert groups.get("tul_loop_denoise.W_in", 0) > 0, groups
    assert any(k.startswith("tul_code_time") for k in groups), groups
    assert all(p.requires_grad for p in m.tul_loop_denoise.parameters())
    assert all(p.requires_grad for p in m.tul_code_time.parameters())
    assert not any(p.requires_grad for p in m.tul_code_enc.parameters()), "E stays frozen"
