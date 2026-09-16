"""TUL-Code (``tul.code``) — the contracts C1..C10 of docs/tul-code-spec.md §11, one test
per invariant, each failing when its mechanism is removed.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_tul_code.py -q

CPU only, the GL1 tiny fixture (tests/test_tul_gl1.py), no tokenizer.
"""

from __future__ import annotations

import pytest
import torch

from test_tul_gl1 import _batch, _cfg, _tul  # noqa: E402  (tests/ is on sys.path)
from test_tul_strict_geometry import _pack, _runtime, _tiny  # noqa: E402

from morph.model.transformer import MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_code import (code_rmsnorm, code_target_valid, code_thinker_relation,
                                  euler_sample)

CODE_CONFIGS = ["tul_code", "tul_code_seeddetach", "tul_code_nophase3", "tul_code_smoke", "tul_code_xm", "tul_code_xmn", "tul_code_lejepa", "tul_code_vae", "tul_code_ladir_tf", "tul_code_ladir_ro", "tul_code_thinker_p3",
                "tul_code_rollout1", "tul_code_renorm", "tul_code_cfg", "tul_code_jepa",
                "tul_code_thinker"]


def _model(seed: int = 3, code: bool = True, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    tul_kw = {k[4:]: v for k, v in kw.items() if k.startswith("tul_")}
    cfg_kw = {k: v for k, v in kw.items() if not k.startswith("tul_")}
    base_tul = dict(tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict",
                    sigreg_lambda=0.0, mux_beta=0.0, token_state_dropout=0.0, code=code)
    base_tul.update(tul_kw)
    base = dict(tul=_tul(**base_tul), n_core=2, mean_depth=3, max_depth=3, bptt_depth=3,
                retention=False, dropout=0.0, core_fixed_point_lambda=0.0, ckpt_grad_iters=0)
    base.update(cfg_kw)
    return MORPHTransformer(_cfg(**base))


def _arm_head(m: MORPHTransformer, seed: int = 11) -> None:
    """Give the zero-init velocity head a real weight, so the thinker MATTERS (a zero
    ``W_v`` makes every sample equal its source and blocks the flow gradient into the
    core; the tests below that need the thinker to act say so by calling this)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.tul_code_head.W_v.weight.copy_(torch.randn(m.tul_code_head.W_v.weight.shape,
                                                     generator=g) * 0.05)


def _logits(m: MORPHTransformer, x, lay, **kw):
    with torch.no_grad():
        return m.tul_forward_ablated(x, None, lay, **kw)["logits"]


def _delta(a, b):
    """Per-position max |Δ| over the vocab, ``-inf - -inf`` (the slot_id column) read as 0."""
    d = (a - b).abs().nan_to_num(0.0)
    d = torch.where(a != b, d, torch.zeros_like(d))
    return d.amax(dim=-1)


def _edit(x, lay, row: int, span: int):
    """Change ONE non-boundary token in the middle of ``span`` (its bag id) of ``row``."""
    tok = ((lay.bag_id[row] == span) & (~lay.slot_mask[row])).nonzero().flatten()
    assert tok.numel() >= 3, "fixture span too short to edit inside"
    p = int(tok[tok.numel() // 2])
    x2 = x.clone()
    x2[row, p] = 12 if int(x2[row, p]) != 12 else 13
    return x2, p


# ── C1: off is the ruler ─────────────────────────────────────────────────────

def test_c1_off_builds_nothing_and_shares_every_weight_with_the_ruler():
    on, off = _model(code=True), _model(code=False)
    assert off.tul_code_enc is None and off.tul_code_head is None and off.tul_code_time is None
    for n, p in off.named_parameters():
        q = dict(on.named_parameters())[n]
        assert torch.equal(p, q), f"{n} differs: the code modules drew from the global RNG"
    assert on.tul.W_prefix is not None and not on.tul.W_prefix.requires_grad, \
        "W_prefix is built (C1's parameter set) and inert on a code model"
    assert off.tul.W_prefix.requires_grad
    assert float(on.tul_code_enc.W_o.weight.detach().abs().sum()) > 0.0


def test_c1_off_knobs_and_wrong_geometry_raise():
    with pytest.raises(ValueError, match="code_noise"):
        _tul(code=False, code_noise=0.1)
    with pytest.raises(NotImplementedError, match="strict"):
        _tul(code=True, tg_restrict=True, sigreg_lambda=0.0)   # geometry defaults to "restrict"
    with pytest.raises(NotImplementedError, match="spandec"):
        _tul(code=True, tg_restrict=True, tg_geometry="strict", sigreg_lambda=0.0, spandec=True)
    with pytest.raises(NotImplementedError, match="slot depth"):
        _tul(code=True, tg_restrict=True, tg_geometry="strict", sigreg_lambda=0.0,
             slot_depth_fixed=3)
    with pytest.raises(NotImplementedError, match="sigreg_lambda"):
        _tul(code=True, tg_restrict=True, tg_geometry="strict")   # the fixture's 0.02


# ── C2 / C3: causality, bounded at train, plain at eval ──────────────────────

def test_c2_train_forward_is_causal_up_to_the_own_code():
    x, _y, lay, _ = _batch()
    m = _model(tul_code_noise=0.0)
    m.train()
    m.code_phase = 1
    row, span = 0, 3
    assert bool(lay.slot_valid[row, span - 1]) and bool(lay.slot_valid[row, span])
    x2, p = _edit(x, lay, row, span)
    with torch.no_grad():
        a = m(x, None, slot_layout=lay)["logits"][row]
        b = m(x2, None, slot_layout=lay)["logits"][row]
    d = _delta(a, b)
    cut = int(lay.slot_index[row, span - 1])              # slot span-1's first cell
    assert float(d[:cut].max()) == 0.0, \
        "a token of span j moved a position BEFORE slot j-1's cells: a leak past the own code"
    own_before = ((lay.bag_id[row] == span) & (~lay.slot_mask[row])).nonzero().flatten()
    own_before = own_before[own_before < p]
    assert float(d[own_before].max()) > 0.0, \
        "span j's earlier tokens did not see the edit through their own code: E is not read"
    assert float(d[cut:cut + m.cfg.tul.prefix_k].max()) > 0.0, "the cells did not change"


def test_c3_sampled_forward_is_plain_causal():
    x, _y, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    m.eval()
    row, span = 0, 3
    x2, p = _edit(x, lay, row, span)
    a = _logits(m, x, lay, code_mode="sampled", code_steps=2)[row]
    b = _logits(m, x2, lay, code_mode="sampled", code_steps=2)[row]
    d = _delta(a, b)
    assert float(d[:p].max()) == 0.0, "a sampled forward moved a position BEFORE the edit"
    assert float(d[p:].max()) > 0.0


# ── C4 / C5: the two losses share almost nothing ─────────────────────────────

def _grads(m: MORPHTransformer, x, y, lay, phase: int, seed: int = 5, **kw) -> dict:
    m.train()
    m.code_phase = phase
    m.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    m(x, y, slot_layout=lay, **kw)["loss"].backward()
    return {n: (None if p.grad is None else p.grad.clone()) for n, p in m.named_parameters()}


def test_c4_the_flow_loss_never_reaches_the_encoder_and_the_seed_ablation_holds():
    x, y, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    g1, g2 = _grads(m, x, y, lay, 1), _grads(m, x, y, lay, 2)
    for n in g1:
        if n.startswith("tul_code_enc."):
            assert torch.equal(g1[n], g2[n]), f"the flow loss reached the encoder through {n}"
    # the flow loss DOES reach the core and the head (else the term is dead) ...
    assert any(g2[n] is not None and (g1[n] is None or not torch.equal(g1[n], g2[n]))
               for n in g1 if n.startswith("core."))
    assert g1["tul_code_head.W_v.weight"] is None and g2["tul_code_head.W_v.weight"] is not None
    # ... and the seed path, unless the ablation knob detaches it.
    assert not torch.equal(g1["tul.W_sent.weight"], g2["tul.W_sent.weight"])
    md = _model(tul_code_seed_detach=True)
    _arm_head(md)
    h1, h2 = _grads(md, x, y, lay, 1), _grads(md, x, y, lay, 2)
    assert torch.equal(h1["tul.W_sent.weight"], h2["tul.W_sent.weight"]), \
        "code_seed_detach=true still let the flow loss into W_sent"


def test_c5_the_token_loss_never_reaches_the_thinker_in_phase_3():
    x, y, lay, _ = _batch()
    m = _model(tul_code_fm_weight=0.0, tul_code_rollout_p=0.5)
    _arm_head(m)
    g = _grads(m, x, y, lay, 3)
    for n, v in g.items():
        if n.startswith(("core.", "tul_code_head.", "tul_code_time.")) or n in (
                "tul_code_cell", "tul_code_clean"):
            assert v is None or float(v.abs().sum()) == 0.0, \
                f"the coda's CE reached the thinker through {n} (sampled codes not detached)"
    assert float(g["tul_code_enc.W_o.weight"].abs().sum()) > 0.0, "E must still train from CE"


# ── C6: the sampler is the Euler integrator ──────────────────────────────────

def _seed_parts(m: MORPHTransformer, x, lay):
    _fkw, _freset, _ckw, _creset = m._tul_tg_kwargs(lay)
    xf, x0, bg = m._tul_front(x, lay, attn_kwargs=_fkw, ret_reset_mask=_freset)
    xn, e, inj = m._tul_code_seed(xf, x0, bg, lay)
    xs = xn.mean(dim=2) if m._is_hc else xn
    z, ok = m.tul_code_enc(xs, lay)
    return z.float(), ok, e, inj


def test_c6_zero_head_returns_the_source_for_every_k_and_one_step_matches_by_hand():
    x, _y, lay, _ = _batch()
    m = _model().eval()
    with torch.no_grad():
        z, ok, e, inj = _seed_parts(m, x, lay)
        outs = []
        for k in (1, 3, 7):
            g = torch.Generator().manual_seed(0)
            outs.append(m._tul_code_sample(z, ok, e, inj, lay, k, generator=g))
        assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
        g = torch.Generator().manual_seed(0)
        z0 = torch.randn(z.shape, generator=g) * m.cfg.tul.code_source_std
        assert torch.equal(outs[0], z0 * ok.view(*ok.shape, 1, 1).float())
        _arm_head(m)
        g = torch.Generator().manual_seed(0)
        one = m._tul_code_sample(z, ok, e, inj, lay, 1, generator=g)
        t0 = torch.zeros(ok.shape, dtype=torch.float32)
        by_hand = (z0 + m._tul_code_thinker(z0, z, t0, e, inj, lay, False)) * ok.view(
            *ok.shape, 1, 1).float()
        assert torch.allclose(one, by_hand, atol=1e-5, rtol=1e-5)
        two = m._tul_code_sample(z, ok, e, inj, lay, 2, generator=torch.Generator().manual_seed(0))
        assert not torch.allclose(one, two), "k=2 must differ from k=1 once the head is armed"
    with pytest.raises(ValueError):
        euler_sample(lambda zz, tt: zz, torch.zeros(1, 1, 1, 4), 0)


# ── C7: the doubled sequence's relation, two-sided ───────────────────────────

def test_c7_relation_mask_is_the_documented_one():
    S, M = 3, 2
    r = code_thinker_relation(S, M, "cpu")[0, 0]
    def pos(s, copy, i): return s * 2 * M + copy * M + i
    # noisy(1,0) reads clean(0,*) and noisy(1,*); nothing else
    q = pos(1, 0, 0)
    allowed = {pos(0, 1, 0), pos(0, 1, 1), pos(1, 0, 0), pos(1, 0, 1)}
    assert set(r[q].nonzero().flatten().tolist()) == allowed
    # clean(1,1) reads clean(0,*) and clean(1,*); never a noisy cell, never a later slot
    q = pos(1, 1, 1)
    allowed = {pos(0, 1, 0), pos(0, 1, 1), pos(1, 1, 0), pos(1, 1, 1)}
    assert set(r[q].nonzero().flatten().tolist()) == allowed
    assert not bool(r[pos(0, 0, 0), pos(0, 1, 0)]), "noisy(0) must not read its own clean code"
    assert not bool(r[pos(0, 0, 0), pos(2, 1, 0)]), "noisy(0) must not read a later slot"


def test_c7_the_thinker_cannot_see_its_own_target_or_the_future():
    x, _y, lay, _ = _batch()
    m = _model().eval()
    _arm_head(m)
    with torch.no_grad():
        z, ok, e, inj = _seed_parts(m, x, lay)
        z0 = torch.randn(z.shape, generator=torch.Generator().manual_seed(1))
        t = torch.full(ok.shape, 0.3)
        v = m._tul_code_thinker(z0, z, t, e, inj, lay, False)
        s = 2
        assert bool(ok[0, s]) and bool(ok[0, s - 1]) and bool(ok[0, s + 1])
        for target, must_change in ((s, False), (s + 1, False), (s - 1, True)):
            zc = z.clone()
            zc[0, target] = zc[0, target] + 1.0
            v2 = m._tul_code_thinker(z0, zc, t, e, inj, lay, False)
            moved = float((v2[0, s] - v[0, s]).abs().max()) > 0.0
            assert moved == must_change, (
                f"perturbing the clean code of slot {target} "
                f"{'moved' if moved else 'did not move'} slot {s}'s velocity")


# ── C8: pads and the row's last slot ─────────────────────────────────────────

def test_c8_pad_and_last_slots_hold_a_zero_code_and_are_masked():
    x, _y, lay, _ = _batch()
    m = _model().eval()
    with torch.no_grad():
        z, ok, _e, _inj = _seed_parts(m, x, lay)
    assert torch.equal(ok, code_target_valid(lay) & ok)
    last = int(lay.slot_valid[0].nonzero().flatten()[-1])
    assert not bool(ok[0, last]), "the row's last valid slot precedes the tail and gets no code"
    assert not bool(ok[0, lay.slot_valid.shape[1] - 1]) or bool(lay.slot_valid[0, -1])
    assert float(z[~ok].abs().sum()) == 0.0
    good = z[ok]
    assert torch.allclose(good.pow(2).mean(-1), torch.ones(good.shape[:-1]), atol=1e-2), \
        "codes are unit-RMS per cell"
    assert torch.allclose(code_rmsnorm(good), good, atol=1e-2)


# ── C9: the config surface ───────────────────────────────────────────────────

def test_c9_known_keys_accept_the_code_block_and_reject_a_misspelling():
    from morph.training.tul_setup import KNOWN_TUL_KEYS, reject_unknown_tul_keys
    keys = ("code", "code_noise", "code_norm", "code_fm_weight", "code_source_std",
            "code_t_embed_scale", "code_phase2_at", "code_phase3_at", "code_rollout_p",
            "code_rollout_steps", "code_infer_steps", "code_seed_detach",
            "code_noise_renorm", "code_marginal_k", "code_cfg_drop", "code_cfg_scale",
            "code_target_lambda", "code_rank_abort")
    for k in keys:
        assert k in KNOWN_TUL_KEYS, k
    reject_unknown_tul_keys({k: 0 for k in keys})
    with pytest.raises(ValueError, match="code_infer_step"):
        reject_unknown_tul_keys({"code": True, "code_infer_step": 8})


# ── C10: every panel config composes, builds and runs ────────────────────────

@pytest.mark.parametrize("name", CODE_CONFIGS)
def test_c10_the_code_arms_compose_and_build_and_run(name, monkeypatch):
    cfg, rt = _runtime(name, monkeypatch)
    assert rt is not None
    tc = rt.model_cfg
    assert tc.code and tc.tg_geometry == "strict" and not tc.spandec
    assert tc.code_seed_detach == name.endswith("_seeddetach")
    _p3 = {"tul_code_nophase3": 1.0, "tul_code_thinker": 1.0, "tul_code_vae": 1.0,
           "tul_code_ladir_tf": 1.0, "tul_code_ladir_ro": 1.0, "tul_code_thinker_p3": 0.0}
    assert tc.code_phase3_at == _p3.get(name, 0.5)
    assert not bool(cfg.model.use_kernels)
    torch.manual_seed(7)
    m = MORPHTransformer(_tiny(tul=tc)).eval().float()
    _ids0, inp, lab, layout = _pack()
    with torch.no_grad():
        out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]), f"{name}: loss is not finite"
    m.train()
    m.code_phase = 2
    out = m(inp, labels=lab, slot_layout=layout)
    assert torch.isfinite(out["loss"]) and "code_fm" in out


# ── eval plumbing the spec names ─────────────────────────────────────────────

def test_eval_modes_and_refusals():
    x, y, lay, _ = _batch()
    m = _model().eval()
    _arm_head(m)
    with torch.no_grad():
        tf = m.tul_forward_ablated(x, y, lay, code_mode="encoder")
        s4 = m.tul_forward_ablated(x, y, lay, code_mode="sampled", code_steps=4)
        rl = m.tul_forward_ablated(x, y, lay, code_mode="rolled", code_steps=2)
        default = m.tul_forward_ablated(x, y, lay)
    assert float(default["ce_tokens"]) == float(
        m.tul_forward_ablated(x, y, lay, code_mode="sampled",
                              code_steps=m.cfg.tul.code_infer_steps)["ce_tokens"])
    assert float(tf["layer_passes"]) < float(s4["layer_passes"]) < float(rl["layer_passes"])
    with pytest.raises(NotImplementedError, match="wrong_seed"):
        m.tul_forward_ablated(x, y, lay, plan_mode="wrong_seed")
    with pytest.raises(NotImplementedError, match="slot_depths"):
        m.tul_forward_ablated(x, y, lay, slot_depths=torch.ones_like(lay.slot_index))
    m.train()
    with pytest.raises(ValueError, match="EVAL-ONLY"):
        m.tul_forward_ablated(x, y, lay, code_mode="encoder")
    probe = m.eval().tul_slot_state_probe(x, lay)
    assert probe["code_eff_rank"] == probe["slot_eff_rank"] > 1.0


# ── generation: one sample per span, re-encoded past ─────────────────────────

def test_generation_writes_a_span_from_one_sample_and_reencodes_the_past():
    from test_tul_gl1 import _rule, _spec
    from morph.inference.tul_generate import generate_tul, generate_tul_batch
    m = _model().eval()
    _arm_head(m)
    rule, spec = _rule(), _spec()
    prompt = [5, 6, 7, 8, 10, 5, 6, 7, 8, 9]                # a boundary (DOT) inside
    seen: list[torch.Tensor] = []
    real_forward = m.forward

    def spy(*a, **kw):
        out = real_forward(*a, **kw)
        if "code_cells" in out:
            s_open = kw["slot_layout"].slot_valid[0].sum().item() - 1
            seen.append((int(s_open), out["code_cells"][0, s_open].clone()))
        return out
    m.forward = spy
    toks, builder = generate_tul(m, prompt, rule, spec, max_new_tokens=14, temperature=1.0,
                                 seed=0, emit_source="token")
    m.forward = real_forward
    assert len(toks) == 14
    # within one open span the cells the coda read are IDENTICAL across steps (the cache),
    # and they differ across spans
    by_slot: dict[int, list[torch.Tensor]] = {}
    for s, c in seen:
        by_slot.setdefault(s, []).append(c)
    assert any(len(v) > 1 for v in by_slot.values()), "no span lasted two steps"
    for s, cs in by_slot.items():
        for c in cs[1:]:
            assert torch.equal(cs[0], c), f"slot {s}: the open span's cells changed mid-span"
    slots = sorted(by_slot)
    if len(slots) > 1:
        assert not torch.equal(by_slot[slots[0]][0], by_slot[slots[1]][0])
    with pytest.raises(NotImplementedError, match="generate_tul_batch"):
        generate_tul_batch(m, [prompt, prompt], rule, spec, max_new_tokens=2)


def test_marginal_is_a_bound_and_the_seed_acts():
    """The K-sample marginal (morph/training/code_eval.py): K = 1 equals the one-sample CE
    at seed 0 (the val/loss stream); Jensen makes it <= the K-average of one-sample CEs;
    different seeds draw different codes (else the average is a lie)."""
    from morph.training.code_eval import code_marginal_ce
    x, y, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    m.eval()
    one = code_marginal_ce(m, x, y, lay, 1, 2)
    assert abs(one["ce_marginal"] - one["ce_single_mean"]) < 1e-5
    ref = m.tul_forward_ablated(x, y, lay, code_mode="sampled", code_steps=2, code_seed=0)
    assert abs(one["ce_single_mean"] - float(ref["ce_tokens"])) < 1e-4, \
        (one, float(ref["ce_tokens"]))
    four = code_marginal_ce(m, x, y, lay, 4, 2)
    assert four["ce_marginal"] <= four["ce_single_mean"] + 1e-6
    a = m.tul_forward_ablated(x, None, lay, code_mode="sampled", code_steps=2, code_seed=0)
    b = m.tul_forward_ablated(x, None, lay, code_mode="sampled", code_steps=2, code_seed=1)
    assert not torch.equal(a["logits"], b["logits"]), "code_seed did not change the draw"
    with pytest.raises(ValueError):
        code_marginal_ce(m, x, y, lay, 0, 2)


# ── the truth cell's statistic: train == eval, in both noise modes ───────────

def _cells_into_coda(m: MORPHTransformer, run):
    """The cells `_tul_code_core` hands the coda, captured at `_tul_plan_ablate`."""
    seen = []
    orig = m._tul_plan_ablate

    def spy(h, layout, mode):
        seen.append(h.detach().float().clone())
        return orig(h, layout, mode)
    m._tul_plan_ablate = spy
    try:
        run()
    finally:
        m._tul_plan_ablate = orig
    assert len(seen) == 1
    return seen[0]


def _rms_of_valid(cells):
    """Mean per-cell RMS over the cells that carry a code (pad / last slots are zero)."""
    rms = cells.pow(2).mean(-1).sqrt()                   # [B, S, M]
    return float(rms[rms > 0].mean())


@pytest.mark.parametrize("renorm", [False, True])
def test_truth_cell_statistic_is_the_same_at_train_and_eval(renorm):
    """2026-09-15: the coda read a code's RMS as the "sample" flag because a truth cell
    reached it at RMS sqrt(1 + noise²) in training and at RMS 1 (bare z) at eval. Under
    `code_noise_renorm` both are 1; without it eval scales z to the training statistic."""
    x, y, lay, _ = _batch()
    noise = 0.5
    m = _model(tul_code_noise=noise, tul_code_noise_renorm=renorm)
    _arm_head(m)
    expected = 1.0 if renorm else (1.0 + noise * noise) ** 0.5
    m.train()
    m.code_phase = 1
    train_cells = _cells_into_coda(m, lambda: m(x, None, slot_layout=lay))
    train_rms = _rms_of_valid(train_cells)
    m.eval()
    with torch.no_grad():
        eval_cells = _cells_into_coda(
            m, lambda: m.tul_forward_ablated(x, y, lay, code_mode="encoder"))
        samp_cells = _cells_into_coda(
            m, lambda: m.tul_forward_ablated(x, y, lay, code_mode="sampled", code_steps=2))
    eval_rms, samp_rms = _rms_of_valid(eval_cells), _rms_of_valid(samp_cells)
    assert abs(eval_rms - expected) < 2e-3, (renorm, eval_rms, expected)
    assert abs(train_rms - eval_rms) < 0.03, \
        f"train truth RMS {train_rms:.4f} vs eval encoder RMS {eval_rms:.4f} (renorm={renorm})"
    assert abs(samp_rms - 1.0) < 2e-3, samp_rms
    if renorm:   # truth and sample are indistinguishable by norm
        assert abs(train_rms - samp_rms) < 0.03


# ── CFG: the null condition and the guided sampler ───────────────────────────

def test_cfg_null_condition_erases_the_past_and_guidance_acts():
    """A null-conditioned row's velocity cannot depend on the seed, the injections or the
    tape (the three conditioning paths of `_tul_code_thinker`); w = 1 is the plain sampler;
    w != 1 moves the sample; the knobs refuse an untrained null path."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_cfg_drop=0.1, tul_code_cfg_scale=2.0).eval()
    _arm_head(m)
    with torch.no_grad():
        m.tul_code_null.normal_(std=0.1)                  # a non-trivial null seed
        # the seed path's outputs, captured from a real forward (x here is token ids)
        got = {}
        orig_seed = m._tul_code_seed

        def spy(*a, **k):
            r = orig_seed(*a, **k)
            got["xn"], got["e"], got["inj"] = r
            return r
        m._tul_code_seed = spy
        try:
            m.tul_forward_ablated(x, y, lay, code_mode="encoder")
        finally:
            m._tul_code_seed = orig_seed
        xn, e, inj = got["xn"], got["e"], got["inj"]
        xs = xn.mean(dim=2) if m._is_hc else xn
        z, ok = m.tul_code_enc(xs, lay)
        B, S = ok.shape
        rows = torch.ones(B, dtype=torch.bool)
        # two different pasts: the real one and a scrambled one
        e2, inj2, tape2 = e.flip(1), inj.roll(1, dims=2), z.float().roll(1, dims=1)
        en, injn, tapen = m._tul_code_null_condition(e, inj, z.float(), rows)
        en2, injn2, tapen2 = m._tul_code_null_condition(e2, inj2, tape2, rows)
        assert torch.equal(en, en2) and torch.equal(injn, injn2) and torch.equal(tapen, tapen2)
        t = torch.full((B, S), 0.3)
        zt = torch.randn_like(z.float())
        v1 = m._tul_code_thinker(zt, tapen, t, en, injn, lay, seed_detach=False)
        v2 = m._tul_code_thinker(zt, tapen2, t, en2, injn2, lay, seed_detach=False)
        assert torch.equal(v1, v2), "a null-conditioned velocity still depends on the past"
        # a half-null batch keeps the other rows intact
        half = torch.zeros(B, dtype=torch.bool)
        half[0] = True
        eh, injh, tapeh = m._tul_code_null_condition(e, inj, z.float(), half)
        assert torch.equal(eh[1:], e[1:]) and torch.equal(injh[:, 1:], inj[:, 1:]) \
            and torch.equal(tapeh[1:], z.float()[1:])
        assert float(injh[:, 0].abs().max()) == 0.0 and float(tapeh[0].abs().max()) == 0.0
        # guidance: w = 1 (explicit) is the plain path; the config's w = 2 differs
        g = torch.Generator().manual_seed(5)
        s1 = m._tul_code_sample(z.float(), ok, e, inj, lay, 2, generator=g, guidance=1.0)
        g = torch.Generator().manual_seed(5)
        s2 = m._tul_code_sample(z.float(), ok, e, inj, lay, 2, generator=g)
        g = torch.Generator().manual_seed(5)
        s1b = m._tul_code_sample(z.float(), ok, e, inj, lay, 2, generator=g, guidance=1.0)
        assert torch.equal(s1, s1b)
        assert not torch.equal(s1, s2), "guidance w = 2 did not move the sample"
        # eval bookkeeping: a guided sample costs two thinker passes per step
        r = m.tul_forward_ablated(x, y, lay, code_mode="sampled", code_steps=3)
        assert m._code_last_passes == 6
    with pytest.raises(ValueError, match="code_cfg_scale"):
        _tul(code=True, code_cfg_scale=2.0)              # drop 0: untrained null path
    m0 = _model()                                         # no CFG: no null parameter
    assert m0.tul_code_null is None
    with pytest.raises(ValueError, match="code_target_lambda"):
        _tul(code=True, code_target_lambda=1.5)


def test_cfg_training_pass_drops_rows_and_reports_it():
    x, y, lay, _ = _batch()
    m = _model(tul_code_cfg_drop=0.999, tul_code_cfg_scale=2.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    out = m(x, y, slot_layout=lay)          # the code stats are top-level keys on the loss path
    assert float(out["code_cfg_drop_frac"]) > 0.5, sorted(out)
    m.code_phase = 1
    assert "code_cfg_drop_frac" not in m(x, y, slot_layout=lay)   # no thinker pass in phase 1


# ── code_target_lambda: C4 amended for the jepa arm ──────────────────────────

@pytest.mark.parametrize("lam", [0.0, 0.1])
def test_target_lambda_lets_the_flow_loss_reach_the_encoder_only_when_set(lam):
    x, y, lay, _ = _batch()
    m = _model(tul_code_target_lambda=lam)
    _arm_head(m)
    g1, g2 = _grads(m, x, y, lay, 1), _grads(m, x, y, lay, 2)
    reached = any(not torch.equal(g1[n], g2[n]) for n in g1 if n.startswith("tul_code_enc."))
    assert reached == (lam > 0.0), (lam, reached)


# ── training.train_only: the thinker-only regime ─────────────────────────────

def test_train_only_freezes_everything_but_the_thinker_and_the_flow_loss_still_learns():
    from morph.training.freeze import apply_train_only
    x, y, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    n_pre, n_core = m.cfg.n_prelude, m.cfg.n_core
    prefixes = ["core.", "injection.", "tul_code_head.", "tul_code_time.", "tul_code_cell",
                "tul_code_clean"] + [f"x0_injects.{i}." for i in range(n_pre, n_pre + n_core)]
    n_t, n_f, groups = apply_train_only(m, prefixes)
    assert n_t > 0 and n_f > n_t
    for name, p in m.named_parameters():
        assert p.requires_grad == name.startswith(tuple(prefixes)), name
        if name.startswith(("prelude.", "coda.", "embed", "tul_code_enc.", "tul.", "lm_mixer",
                            "final_norm", "value_embed", "input_norm")):
            assert not p.requires_grad, f"{name} must be frozen in the thinker-only regime"
    assert any(k.startswith("core.0") for k in groups) and any(k.startswith("tul_code_head") for k in groups)
    g = _grads(m, x, y, lay, 2)                                 # phase 2: flow loss on
    assert all(g[n] is not None for n, p in m.named_parameters() if p.requires_grad
               and n.startswith(("core.", "tul_code_head."))), "the thinker got no gradient"
    assert all(g[n] is None for n, p in m.named_parameters() if not p.requires_grad)
    with pytest.raises(ValueError, match="matched no parameter"):
        apply_train_only(_model(), ["nothing_here."])
    with pytest.raises(ValueError, match="EVERY"):
        apply_train_only(_model(), [""])


def test_generate_mode_open_slot_sample_follows_code_seed():
    """Generation (spec §7): the OPEN slot's code is ONE sample whose z_0 must come from the
    seeded stream when `code_seed` is given (the generator derives it from its token seed
    and the open slot), so a seeded generation is reproducible end to end; two runs of the
    same sampler differed on every span on 2026-09-15 because this draw hit the global RNG."""
    x, _, lay, _ = _batch()
    m = _model()
    _arm_head(m)
    m.eval()
    ok = code_target_valid(lay)
    open_ = lay.slot_valid & ~ok
    assert bool(open_.any()), "the test batch has no open slot; the test has no teeth"
    with torch.no_grad():
        a = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=5)["code_cells"]
        b = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=5)["code_cells"]
        c = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=6)["code_cells"]
    assert torch.equal(a[open_], b[open_]), "same code_seed, different open-slot sample"
    assert not torch.equal(a[open_], c[open_]), "code_seed did not change the open-slot sample"
    # the closed slots hold E's code of the written span and never depend on the seed
    assert torch.equal(a[ok], c[ok])


# ── Explorative Modeling (tul.code_xm_k > 1, spec §6 step 3) ────────────────────────
def _xm_capture(m, monkeypatch):
    """Record every sampler call (its z0 and endpoint), the flow pair's z0, and what the
    coda was handed (the cells entering `_tul_plan_ablate`)."""
    import morph.model.transformer as tr
    rec = {"z0": [], "hat": [], "pair_z0": None, "coda": None}
    orig_sample = m._tul_code_sample
    def spy_sample(*a, **kw):
        out = orig_sample(*a, **kw)
        rec["z0"].append(kw["z0"].clone() if kw.get("z0") is not None else None)
        rec["hat"].append(out.clone())
        return out
    monkeypatch.setattr(m, "_tul_code_sample", spy_sample)
    orig_pair = tr.cfm_pair
    def spy_pair(*a, **kw):
        rec["pair_z0"] = None if kw.get("z0") is None else kw["z0"].clone()
        return orig_pair(*a, **kw)
    monkeypatch.setattr(tr, "cfm_pair", spy_pair)
    orig_ablate = m._tul_plan_ablate
    def spy_ablate(z_coda, layout, mode):
        rec["coda"] = z_coda.detach().clone()
        return orig_ablate(z_coda, layout, mode)
    monkeypatch.setattr(m, "_tul_plan_ablate", spy_ablate)
    orig_enc = m.tul_code_enc.forward
    def spy_enc(xs, layout):
        z, ok = orig_enc(xs, layout)
        rec["z"] = z.detach().float().clone()
        return z, ok
    monkeypatch.setattr(m.tul_code_enc, "forward", spy_enc)
    return rec


def test_xm_l2_trains_the_flow_on_the_nearest_seed_and_hands_the_coda_that_sample(monkeypatch):
    """Forward XM (arXiv 2607.27372): K samples per slot, the one nearest E's code wins;
    the flow pair's z_0 is the winner's seed, and the rollout cell IS the winner."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_xm_k=3, tul_code_rollout_p=1.0, tul_code_noise=0.0)
    _arm_head(m)
    m.train()
    m.code_phase = 3
    rec = _xm_capture(m, monkeypatch)
    out = m(x, labels=y, slot_layout=lay)
    assert len(rec["hat"]) == 3, f"expected K=3 sampler calls, got {len(rec['hat'])}"
    assert all(z is not None for z in rec["z0"]), "XM must pass its own z0 into the sampler"
    ok = code_target_valid(lay)
    z = rec["z"]                                                        # E's code, [B, S, M, C]
    cands = torch.stack(rec["hat"])                                      # [K, B, S, M, C]
    score = (code_rmsnorm(cands) - z).pow(2).sum((-1, -2))               # [K, B, S]
    best = score.argmin(0)
    z0s = torch.stack(rec["z0"])
    sel_z0 = z0s.gather(0, best.view(1, *best.shape, 1, 1).expand(1, *best.shape, *z.shape[2:])).squeeze(0)
    assert rec["pair_z0"] is not None, "the flow pair did not receive XM's z0"
    assert torch.allclose(rec["pair_z0"][ok], sel_z0[ok], atol=1e-6), \
        "the flow pair's z0 is not the nearest candidate's seed"
    sel_hat = cands.gather(0, best.view(1, *best.shape, 1, 1).expand(1, *best.shape, *z.shape[2:])).squeeze(0)
    assert torch.allclose(rec["coda"][ok].float(), code_rmsnorm(sel_hat)[ok], atol=1e-4), \
        "the coda did not read the selected sample at the rollout slots"
    st = out["code_stats"] if "code_stats" in out else out
    assert st["code_xm_score_best"] <= st["code_xm_score_mean"] + 1e-6
    assert m._code_last_passes == 1 + 3 * m.cfg.tul.code_rollout_steps  # flow pass + K samplers
    out["loss"].backward()
    assert m.tul_code_head.W_v.weight.grad is not None


def test_xm_coda_selection_picks_the_lowest_span_ce_and_needs_labels(monkeypatch):
    x, y, lay, _ = _batch()
    m = _model(tul_code_xm_k=3, tul_code_rollout_p=1.0, tul_code_xm_select="coda")
    _arm_head(m)
    m.train()
    m.code_phase = 3
    scores = []
    orig = m._tul_code_span_scorer
    def spy_scorer(*a, **kw):
        f = orig(*a, **kw)
        def g(cells):
            s = f(cells); scores.append(s.clone()); return s
        g.bind = f.bind
        return g
    monkeypatch.setattr(m, "_tul_code_span_scorer", spy_scorer)
    rec = _xm_capture(m, monkeypatch)
    m(x, labels=y, slot_layout=lay)
    assert len(scores) == 3 and scores[0].shape == tuple(code_target_valid(lay).shape)
    ok = code_target_valid(lay)
    best = torch.stack(scores).argmin(0)
    cands = torch.stack(rec["hat"])
    sel = cands.gather(0, best.view(1, *best.shape, 1, 1).expand(1, *best.shape, *cands.shape[3:])).squeeze(0)
    assert torch.allclose(rec["coda"][ok].float(), code_rmsnorm(sel)[ok], atol=1e-4)
    # the spans a slot is scored on are its NEXT span: a candidate can only move its own
    # slot's score, so scores differ across candidates where ok
    assert not torch.equal(scores[0][ok], scores[1][ok])
    with pytest.raises(RuntimeError, match="coda scorer"):
        m(x, labels=None, slot_layout=lay)


def test_xm_noise_search_trains_the_lowest_flow_loss_pair_with_no_generation(monkeypatch):
    """The paper's Diffusion/Flow hybrid (App. C): K corruption noises at ONE t, each
    scored by the flow loss itself; the flow pair's z_0 is the argmin; no sampler call
    before the rollout; the rollout is a fresh draw (not an XM sample)."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_xm_k=3, tul_code_xm_mode="noise", tul_code_rollout_p=0.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    rec = _xm_capture(m, monkeypatch)
    calls = []                                       # every thinker call: (z_noisy, t, grad?)
    orig_th = m._tul_code_thinker
    def spy_th(z_noisy, z_clean, t, e, inj, layout, seed_detach):
        out = orig_th(z_noisy, z_clean, t, e, inj, layout, seed_detach)
        calls.append((z_noisy.detach().clone(), t.detach().clone(), torch.is_grad_enabled()))
        return out
    monkeypatch.setattr(m, "_tul_code_thinker", spy_th)
    out = m(x, labels=y, slot_layout=lay)
    assert rec["hat"] == [], "noise search must not generate"
    assert len(calls) == 4, f"expected K=3 no-grad passes + 1 grad pass, got {len(calls)}"
    assert [g for _, _, g in calls] == [False, False, False, True]
    assert all(torch.equal(calls[0][1], c[1]) for c in calls), "t must be shared by the candidates"
    ok = code_target_valid(lay)
    z = rec["z"]
    t = calls[0][1].view(*calls[0][1].shape, 1, 1)
    # each candidate's z_0 from its z_t: z_t = (1-t) z_0 + t z  ->  z_0 = (z_t - t z) / (1-t)
    z0s = torch.stack([(c[0] - t * z) / (1.0 - t) for c in calls[:3]])
    # the selected z_0 must be one of the candidates and be the trained pair's z_0
    sel = rec["pair_z0"]
    assert sel is not None
    dists = torch.stack([(sel - z0s[j]).pow(2).sum((-1, -2)) for j in range(3)])   # [K, B, S]
    assert bool((dists.min(0).values[ok] < 1e-6).all()), "the trained z_0 is not a candidate"
    assert torch.allclose(calls[3][0][ok], ((1 - t) * sel + t * z)[ok], atol=1e-5), \
        "the grad pass did not run on the selected pair"
    st = out
    assert st["code_xm_score_best"] <= st["code_xm_score_mean"] + 1e-6
    assert m._code_last_passes == 1 + 3
    out["loss"].backward()
    assert m.tul_code_head.W_v.weight.grad is not None


def test_xm_noise_search_selects_the_argmin_of_the_flow_loss(monkeypatch):
    """Pin the selection rule by making the thinker a known function: with v_hat = 0 the
    flow loss of candidate j is ||z - z0_j||^2, so the winner is the z_0 nearest the code."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_xm_k=4, tul_code_xm_mode="noise", tul_code_rollout_p=0.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    rec = _xm_capture(m, monkeypatch)
    seen = []
    def zero_th(z_noisy, z_clean, t, e, inj, layout, seed_detach):
        seen.append((z_noisy.detach().clone(), t.detach().clone()))
        return torch.zeros_like(z_noisy)
    monkeypatch.setattr(m, "_tul_code_thinker", zero_th)
    m(x, labels=y, slot_layout=lay)
    ok = code_target_valid(lay)
    z = rec["z"]
    t = seen[0][1].view(*seen[0][1].shape, 1, 1)
    z0s = torch.stack([(zt - t * z) / (1.0 - t) for zt, _ in seen[:4]])
    best = (z - z0s).pow(2).sum((-1, -2)).argmin(0)                    # [B, S]
    sel = z0s.gather(0, best.view(1, *best.shape, 1, 1).expand(1, *best.shape, *z.shape[2:])).squeeze(0)
    assert torch.allclose(rec["pair_z0"][ok], sel[ok], atol=1e-5), \
        "noise search did not pick the lowest-flow-loss corruption"


# ── LaDiR tape rollout (tul.code_tape_rollout_p, spec §6) ───────────────────────────
def test_tape_rollout_hands_the_thinker_its_own_sampled_tape_and_keeps_the_oracle_target(monkeypatch):
    """With p = 1 the flow pass's CONTEXT is code_rmsnorm(the sampler's output), not E's
    tape; the target pair is unchanged (v_target = z - z0 with z = E's code); the pass
    count adds the sampler's steps; the coda still reads truth (no phase-3 rollout)."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_tape_rollout_p=1.0, tul_code_rollout_p=0.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    rec = _xm_capture(m, monkeypatch)
    ctx = []
    orig_th = m._tul_code_thinker
    def spy_th(z_noisy, z_clean, t, e, inj, layout, seed_detach):
        ctx.append(z_clean.detach().clone())
        return orig_th(z_noisy, z_clean, t, e, inj, layout, seed_detach)
    monkeypatch.setattr(m, "_tul_code_thinker", spy_th)
    out = m(x, labels=y, slot_layout=lay)
    assert len(rec["hat"]) == 1, "one parallel sampler run for the tape"
    ok = code_target_valid(lay)
    z = rec["z"]
    tape_hat = code_rmsnorm(rec["hat"][0])
    flow_ctx = ctx[-1]                                   # the grad pass is the last thinker call
    assert torch.allclose(flow_ctx[ok], tape_hat[ok], atol=1e-4), \
        "the flow pass did not read the sampled tape"
    assert not torch.allclose(flow_ctx[ok], z[ok], atol=1e-3), "the context is still E's tape"
    assert out["code_tape_rollout_frac"] == 1.0
    assert m._code_last_passes == 1 + m.cfg.tul.code_rollout_steps
    # the coda read truth cells: what entered _tul_plan_ablate is E's (noised) code, not a sample
    assert rec["coda"] is not None and not torch.allclose(rec["coda"][ok].float(), tape_hat[ok], atol=1e-3)
    out["loss"].backward()
    assert m.tul_code_head.W_v.weight.grad is not None
    with pytest.raises(ValueError, match="code_tape_rollout_p"):
        _model(tul_code_tape_rollout_p=1.5)


def test_tape_rollout_off_is_the_default_path(monkeypatch):
    x, y, lay, _ = _batch()
    m = _model(tul_code_tape_rollout_p=0.0)
    _arm_head(m)
    m.train()
    m.code_phase = 2
    rec = _xm_capture(m, monkeypatch)
    out = m(x, labels=y, slot_layout=lay)
    assert rec["hat"] == [] and "code_tape_rollout_frac" not in out
    assert m._code_last_passes == 1


def test_xm_refusals_and_cfm_pair_takes_a_given_seed():
    from morph.model.tul_code import cfm_pair
    with pytest.raises(ValueError, match="code_xm_k"):
        _model(tul_code_xm_k=0)
    with pytest.raises(ValueError, match="code_xm_select"):
        _model(tul_code_xm_select="best")
    with pytest.raises(ValueError, match="code_xm_mode"):
        _model(tul_code_xm_mode="hybrid")
    with pytest.raises(ValueError, match="full generation"):
        _model(tul_code_xm_mode="noise", tul_code_xm_select="coda")
    z = torch.randn(2, 3, 2, 8)
    z0 = torch.randn(2, 3, 2, 8)
    t = torch.rand(2, 3)
    a, zt, v = cfm_pair(z, 1.0, t, z0=z0)
    assert torch.equal(a, z0) and torch.allclose(v, z - z0)
    with pytest.raises(ValueError, match="must match"):
        cfm_pair(z, 1.0, t, z0=z0[:1])


# ── LeJEPA on the code (tul.code_sigreg_lambda, spec §9) ────────────────────────────
def test_code_sigreg_reaches_the_encoder_and_is_subtracted_from_the_reported_loss():
    """SIGReg on E's code cells: a real term in the loss whose gradient reaches E (through z,
    not the detached target) and not the thinker; exposed as code_sigreg_weighted so the
    trainer keeps train/loss on the CE; off by default (bit-identical groups)."""
    x, y, lay, _ = _batch()
    m = _model(tul_code_sigreg_lambda=0.05, tul_sigreg_slices=64, tul_code_target_lambda=1.0)
    m.train(); m.code_phase = 1                      # phase 1: no flow loss, SIGReg alone
    out = m(x, labels=y, slot_layout=lay)
    assert "code_sigreg_weighted" in out and float(out["code_sigreg"]) > 0.0
    assert abs(float(out["code_sigreg_weighted"]) - 0.05 * float(out["code_sigreg"])) < 1e-6, \
        "the exposed weighted term must be lambda times the raw SIGReg value"
    # isolate the SIGReg term: its gradient reaches E and nothing else in the code path
    m2 = _model(tul_code_sigreg_lambda=0.05, tul_sigreg_slices=64, tul_code_target_lambda=1.0)
    m2.train(); m2.code_phase = 1
    o2 = m2(x, labels=y, slot_layout=lay)
    m2._code_sigreg_loss.backward()                # the live term, not the detached stat
    enc_g = sum(float(p.grad.abs().sum()) for p in m2.tul_code_enc.parameters() if p.grad is not None)
    thk_g = sum(float(p.grad.abs().sum()) for p in m2.tul_code_head.parameters() if p.grad is not None)
    assert enc_g > 0.0, "SIGReg on the code did not reach the encoder"
    assert thk_g == 0.0, "SIGReg on the code must not touch the thinker's head"
    m0 = _model(); m0.train(); m0.code_phase = 1
    o0 = m0(x, labels=y, slot_layout=lay)
    assert "code_sigreg_weighted" not in o0
    with pytest.raises(ValueError, match="code_sigreg_lambda"):
        _model(tul_code_sigreg_lambda=-1.0)
