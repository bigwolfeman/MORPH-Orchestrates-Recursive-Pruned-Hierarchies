"""LCTUL-D (``tul.code_discrete``) — the discrete code and the masked denoiser, docs/tul-code-spec.md
§16, one test per contract row D1..D12, each failing when its mechanism is removed.

    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 python -m pytest tests/test_tul_code_d.py -q

CPU only, the GL1 tiny fixture (tests/test_tul_gl1.py), no tokenizer.
"""

from __future__ import annotations

import math

import pytest
import torch

from test_tul_code import _delta, _edit, _grads, _logits, _model, _seed_parts  # noqa: E402
from test_tul_gl1 import _batch  # noqa: E402
from test_tul_strict_geometry import _runtime  # noqa: E402

from morph.model.tul_code import (code_rmsnorm, maskgit_sample, mdm_loss, mdm_mask,
                                  mdm_unmask_counts)

_D = dict(tul_code_discrete=True, tul_code_noise=0.0, tul_code_vq_codebook=32,
          tul_code_vq_groups=2, tul_code_infer_steps=4, tul_code_rollout_steps=4)


def _dmodel(seed: int = 3, **kw):
    d = dict(_D)
    d.update(kw)
    return _model(seed=seed, **d)


def _arm_sym_head(m, seed: int = 11) -> None:
    """Give the zero-init denoiser head a real weight so the thinker MATTERS."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        m.tul_code_sym_head.W_o.weight.copy_(
            torch.randn(m.tul_code_sym_head.W_o.weight.shape, generator=g) * 0.05)


# ── D1: what is built, and the base weights are the code model's ─────────────────

def test_d1_builds_the_quantiser_and_denoiser_and_no_velocity_head():
    m, c = _dmodel(), _model(seed=3, tul_code_noise=0.0)
    assert m.tul_code_vq is not None and m.tul_code_sym is not None
    assert m.tul_code_sym_head is not None and m.tul_code_head is None
    assert c.tul_code_vq is None and c.tul_code_head is not None
    assert m._code_n_sym == 2 * 2 and m._code_mask_id == 32
    assert m.tul_code_sym.weight.shape == (33, m.cfg.d_model)
    assert m.tul_code_cell.shape[0] == m._code_n_sym
    assert m._code_fm_scale == pytest.approx(4 * math.log(32))
    # RNG-neutral construction: every weight the two models share is byte-identical
    cs = dict(c.named_parameters())
    for n, p in m.named_parameters():
        if n in cs and not n.startswith("tul_code_cell"):
            assert torch.equal(p, cs[n]), f"{n} differs between the discrete and the continuous arm"
    rms = m.tul_code_sym.weight.pow(2).mean(-1).sqrt()
    assert 0.8 < float(rms.mean()) < 1.2, "symbol embeddings are not at unit per-component scale"


# ── D2: the coda reads rmsnorm(lift(symbols)), STE into E ─────────────────────────

def _spy_vq(m):
    seen = {}
    real = m.tul_code_vq.forward

    def spy(z, ok, sub_index=None):
        cells, deq, out = real(z, ok, sub_index=sub_index)
        seen["cells"], seen["index"], seen["ok"], seen["sub"] = cells, out["index"], ok, sub_index
        return cells, deq, out
    m.tul_code_vq.forward = spy
    return seen


def test_d2_truth_cells_are_the_lifted_symbols_at_unit_rms():
    x, y, lay, _ = _batch()
    m = _dmodel().eval()
    seen = _spy_vq(m)
    with torch.no_grad():
        out = m.tul_forward_ablated(x, y, lay, code_mode="encoder")
    cells = out["code_cells"]
    ok = seen["ok"]
    B, S = ok.shape
    lifted = m._tul_code_lift(seen["index"].clamp_min(0).reshape(B, S, -1), ok)
    assert torch.allclose(cells.float(), lifted, atol=1e-5), "coda cells != rmsnorm(lift(index))"
    assert torch.allclose(cells.float(), code_rmsnorm(seen["cells"].float()) * ok.view(B, S, 1, 1),
                          atol=1e-5)
    rms = cells.float().pow(2).mean(-1).sqrt()[ok]
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)
    assert float(cells[~ok].abs().max()) == 0.0
    assert seen["sub"] is None, "eval substituted symbols"


def test_d2_the_token_loss_reaches_the_encoder_through_the_straight_through_estimator():
    x, y, lay, _ = _batch()
    m = _dmodel(tul_code_sub_p=0.0, tul_code_vq_weight=0.0)
    g = _grads(m, x, y, lay, 1)
    assert g["tul_code_enc.W_o.weight"] is not None and float(g["tul_code_enc.W_o.weight"].abs().sum()) > 0
    assert g["tul_code_vq.W_vq.weight"] is not None
    assert g["tul_code_vq.W_vq_out"] is not None and float(g["tul_code_vq.W_vq_out"].abs().sum()) > 0


# ── D3: substitution touches the coda's input only ────────────────────────────────

def test_d3_substitution_changes_the_coda_cells_not_the_target_and_cuts_the_gradient():
    x, y, lay, _ = _batch()
    m = _dmodel(tul_code_sub_p=0.999, tul_code_vq_weight=0.0)
    seen = _spy_vq(m)
    calls = []
    real_th = m._tul_code_thinker_discrete

    def spy_th(idx_noisy, idx_clean, *a, **kw):
        calls.append((idx_noisy.clone(), idx_clean.clone()))
        return real_th(idx_noisy, idx_clean, *a, **kw)
    m._tul_code_thinker_discrete = spy_th
    m.train()
    m.code_phase = 2
    torch.manual_seed(0)
    out = m(x, y, slot_layout=lay)
    assert 0.9 < float(out["code_sub_frac"]) <= 1.0
    ok = seen["ok"]
    B, S = ok.shape
    truth = seen["index"].clamp_min(0).reshape(B, S, -1)
    assert torch.equal(calls[0][1], truth), "the denoiser's tape is not the encoder's own symbols"
    masked = calls[0][0] == m._code_mask_id
    assert torch.equal(calls[0][0][~masked], truth[~masked]), "unmasked noisy copies are not the truth"
    sub = seen["sub"]
    assert sub is not None and bool((sub >= 0).any())
    # the coda's cells on a substituted symbol are the substitute's lift, not the truth's
    out["loss"].backward()
    g_sub = m.tul_code_enc.W_o.weight.grad.abs().sum()
    m2 = _dmodel(tul_code_sub_p=0.0, tul_code_vq_weight=0.0)
    m2.train()
    m2.code_phase = 2
    torch.manual_seed(0)
    m2(x, y, slot_layout=lay)["loss"].backward()
    g_no = m2.tul_code_enc.W_o.weight.grad.abs().sum()
    assert float(g_sub) < 0.05 * float(g_no), \
        f"substituted symbols still carry the token gradient into E ({float(g_sub):.3g} vs {float(g_no):.3g})"


# ── D4: the masked-diffusion ELBO ─────────────────────────────────────────────────

def test_d4_mdm_loss_scores_masked_symbols_only_with_the_1_over_t_weight():
    torch.manual_seed(0)
    B, S, N, C = 2, 3, 4, 8
    t = torch.tensor([[0.2, 0.5, 0.9], [0.3, 0.7, 1.0]])
    mask = mdm_mask(t, N, generator=torch.Generator().manual_seed(1))
    ok = torch.ones(B, S, dtype=torch.bool)
    ok[1, 2] = False
    target = torch.randint(0, C, (B, S, N))
    logits = torch.zeros(B, S, N, C)                                     # uniform head
    loss, per = mdm_loss(logits, target, mask, t, ok)
    expect = mask.sum(-1).float() / t * math.log(C) * ok.float()
    assert torch.allclose(per, expect, atol=1e-5)
    assert float(loss) == pytest.approx(float(expect.sum() / ok.sum()))
    # unmasked positions do not matter
    logits2 = logits.clone()
    logits2[~mask] = torch.randn_like(logits2[~mask]) * 3
    loss2, _ = mdm_loss(logits2, target, mask, t, ok)
    assert float(loss2) == pytest.approx(float(loss))
    # a masked position does
    logits3 = logits.clone()
    logits3[mask] = torch.randn_like(logits3[mask]) * 3
    assert float(mdm_loss(logits3, target, mask, t, ok)[0]) != pytest.approx(float(loss))
    # the mask's marginal is t
    big = mdm_mask(torch.full((1, 1), 0.3), 20000, generator=torch.Generator().manual_seed(2))
    assert abs(float(big.float().mean()) - 0.3) < 0.01


def test_d4_zero_head_reads_the_uniform_floor_whatever_the_core_does():
    x, y, lay, _ = _batch()
    outs = []
    for seed in (3, 4):
        m = _dmodel(seed=3)
        if seed == 4:                              # a different core, same everything else
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n.startswith("core."):
                        p.add_(torch.randn_like(p) * 0.1)
        m.train()
        m.code_phase = 2
        torch.manual_seed(0)
        outs.append(m(x, y, slot_layout=lay))
    a, b = outs
    assert float(a["code_fm_raw"]) == pytest.approx(float(b["code_fm_raw"]), rel=1e-4), \
        "a zero-init head's ELBO depends on the core: the logits are not uniform"
    assert float(a["code_fm_rel"]) > 0 and math.isfinite(float(a["code_fm_rel"]))
    assert float(a["code_mdm_nats"]) == pytest.approx(float(a["code_fm_raw"]))
    # in expectation the uniform head reads the floor: average many draws
    m = _dmodel()
    m.train()
    m.code_phase = 2
    vals = []
    for s in range(30):
        torch.manual_seed(100 + s)
        with torch.no_grad():
            vals.append(float(m(x, y, slot_layout=lay)["code_fm_rel"]))
    mean = sum(vals) / len(vals)
    assert abs(mean - 1.0) < 0.25, f"uniform head reads {mean:.3f} of the floor over 30 draws"


# ── D5: the sampler ───────────────────────────────────────────────────────────────

def test_d5_unmask_schedule_commits_every_round_and_ends_full():
    assert mdm_unmask_counts(8, 8) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert mdm_unmask_counts(8, 1) == [8]
    assert mdm_unmask_counts(8, 3) == [3, 6, 8]
    cos = mdm_unmask_counts(8, 4, "cosine")
    assert cos[-1] == 8 and all(b > a for a, b in zip(cos, cos[1:])) and cos[0] >= 1
    with pytest.raises(ValueError):
        mdm_unmask_counts(8, 2, "bogus")


def test_d5_maskgit_sample_follows_the_schedule_and_is_seeded():
    B, S, N, C, MASK = 2, 3, 4, 8, 8
    ok = torch.ones(B, S, dtype=torch.bool)
    ok[1, 2] = False
    seen_t = []
    g = torch.Generator().manual_seed(0)
    logit_tab = torch.randn(N, C, generator=g) * 2

    def fn(idx, t):
        seen_t.append(t.clone())
        assert idx.shape == (B, S, N)
        return logit_tab.view(1, 1, N, C).expand(B, S, N, C).clone()
    for k in (1, 2, 4):
        seen_t.clear()
        out = maskgit_sample(fn, ok, N, MASK, k, generator=torch.Generator().manual_seed(1))
        assert out.shape == (B, S, N) and int(out.max()) < C and int(out.min()) >= 0
        assert int(out[1, 2].abs().sum()) == 0
        counts = mdm_unmask_counts(N, k)
        expect_t = [1.0] + [1.0 - c / N for c in counts[:-1]]
        for j, t in enumerate(seen_t):
            assert torch.allclose(t[ok], torch.full_like(t[ok], expect_t[j])), \
                f"k={k} round {j}: masked fraction {t[ok].tolist()} != {expect_t[j]}"
    a = maskgit_sample(fn, ok, N, MASK, 4, generator=torch.Generator().manual_seed(7))
    b = maskgit_sample(fn, ok, N, MASK, 4, generator=torch.Generator().manual_seed(7))
    c = maskgit_sample(fn, ok, N, MASK, 4, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and not torch.equal(a, c)
    # k = 1 is one-shot categorical sampling from the all-masked logits
    one = maskgit_sample(fn, ok, N, MASK, 1, generator=torch.Generator().manual_seed(3))
    probs = torch.softmax(logit_tab.float(), -1).view(1, 1, N, C).expand(B, S, N, C).reshape(-1, C)
    ref = torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(3)).view(B, S, N)
    assert torch.equal(one[ok], ref[ok])
    # a certain head puts its symbol everywhere, at every k
    sure = torch.full((N, C), -30.0)
    sure[:, 5] = 30.0
    for k in (1, 3, 4):
        out = maskgit_sample(lambda i, t: sure.view(1, 1, N, C).expand(B, S, N, C).clone(),
                             ok, N, MASK, k)
        assert bool((out[ok] == 5).all())


# ── D6: causality ─────────────────────────────────────────────────────────────────

def test_d6_the_denoiser_reads_earlier_slots_only_and_never_its_own_target():
    x, _y, lay, _ = _batch()
    m = _dmodel().eval()
    _arm_sym_head(m)
    B, S = lay.slot_valid.shape
    N = m._code_n_sym
    g = torch.Generator().manual_seed(0)
    clean = torch.randint(0, 32, (B, S, N), generator=g)
    noisy = torch.where(torch.rand(B, S, N, generator=g) < 0.5,
                        torch.full_like(clean, m._code_mask_id), clean)
    t = torch.full((B, S), 0.5)
    with torch.no_grad():
        _z, _ok, e, inj = _seed_parts(m, x, lay)
        base = m._tul_code_thinker_discrete(noisy, clean, t, e, inj, lay, False)
        s = 3
        c2 = clean.clone()
        c2[:, s] = (c2[:, s] + 1) % 32                                  # own target
        own = m._tul_code_thinker_discrete(noisy, c2, t, e, inj, lay, False)
        c3 = clean.clone()
        c3[:, s + 1] = (c3[:, s + 1] + 1) % 32                          # the future
        fut = m._tul_code_thinker_discrete(noisy, c3, t, e, inj, lay, False)
        c4 = clean.clone()
        c4[:, s - 1] = (c4[:, s - 1] + 1) % 32                          # the past
        past = m._tul_code_thinker_discrete(noisy, c4, t, e, inj, lay, False)
    assert torch.equal(base[:, :s + 1], own[:, :s + 1]), "slot s read its own clean code"
    assert torch.equal(base[:, :s + 1], fut[:, :s + 1]), "slot s read a future code"
    assert not torch.equal(base[:, s], past[:, s]), "slot s did not read the past code"
    assert torch.equal(base[:, :s - 1], past[:, :s - 1])


def test_d6_sampled_forward_is_plain_causal_and_train_forward_is_causal_up_to_the_own_code():
    x, _y, lay, _ = _batch()
    m = _dmodel(tul_code_sub_p=0.0).eval()      # no substitution draw: the two forwards must match
    _arm_sym_head(m)
    row, span = 0, 3
    x2, p = _edit(x, lay, row, span)
    a = _logits(m, x, lay, code_mode="sampled", code_steps=2)[row]
    b = _logits(m, x2, lay, code_mode="sampled", code_steps=2)[row]
    d = _delta(a, b)
    assert float(d[:p].max()) == 0.0, "a sampled forward moved a position BEFORE the edit"
    assert float(d[p:].max()) > 0.0
    m.train()
    m.code_phase = 1
    with torch.no_grad():
        a = m(x, None, slot_layout=lay)["logits"][row]
        b = m(x2, None, slot_layout=lay)["logits"][row]
    d = _delta(a, b)
    cut = int(lay.slot_index[row, span - 1])
    assert float(d[:cut].max()) == 0.0, "a token of span j moved a position before slot j-1's cells"


# ── D7: the two losses share almost nothing ───────────────────────────────────────

def test_d7_the_denoiser_loss_never_reaches_the_encoder_and_the_token_loss_never_reaches_the_head():
    x, y, lay, _ = _batch()
    y_none = torch.full_like(y, -100)                       # no token loss at all
    m = _dmodel(tul_code_sub_p=0.0, tul_code_vq_weight=0.0)
    _arm_sym_head(m)
    g = _grads(m, x, y_none, lay, 2)
    for n, v in g.items():
        if n.startswith("tul_code_enc.") or n.startswith("tul_code_vq."):
            assert v is None or float(v.abs().sum()) == 0.0, f"the denoiser loss reached {n}"
    assert float(g["tul_code_sym_head.W_o.weight"].abs().sum()) > 0
    assert float(g["tul_code_sym.weight"].abs().sum()) > 0
    assert any(v is not None and float(v.abs().sum()) > 0 for n, v in g.items() if n.startswith("core."))
    # phase 3 with the denoiser term off: the sampled code is no-grad, the head is untouched
    m3 = _dmodel(tul_code_fm_weight=0.0, tul_code_rollout_p=1.0, tul_code_vq_weight=0.0)
    _arm_sym_head(m3)
    g3 = _grads(m3, x, y, lay, 3)
    assert g3["tul_code_sym_head.W_o.weight"] is None or \
        float(g3["tul_code_sym_head.W_o.weight"].abs().sum()) == 0.0
    assert g3["tul_code_sym.weight"] is None or float(g3["tul_code_sym.weight"].abs().sum()) == 0.0


# ── D8: the null condition and guidance ───────────────────────────────────────────

def test_d8_null_condition_erases_the_past_on_a_discrete_tape_and_guidance_acts():
    x, _y, lay, _ = _batch()
    m = _dmodel(tul_code_cfg_drop=0.1).eval()
    _arm_sym_head(m)
    B, S = lay.slot_valid.shape
    N, MASK = m._code_n_sym, m._code_mask_id
    g = torch.Generator().manual_seed(0)
    tape_a = torch.randint(0, 32, (B, S, N), generator=g)
    tape_b = torch.randint(0, 32, (B, S, N), generator=g)
    with torch.no_grad():
        m.tul_code_null.normal_(std=0.1)
        _z, _ok, e_a, inj_a = _seed_parts(m, x, lay)
    e_b = e_a.flip(1)
    rows = torch.ones(B, dtype=torch.bool)
    ea, ia, ta = m._tul_code_null_condition(e_a, inj_a, tape_a, rows)
    eb, ib, tb = m._tul_code_null_condition(e_b, inj_a * 2, tape_b, rows)
    assert bool((ta == MASK).all()) and torch.equal(ta, tb)
    assert torch.equal(ea, eb) and float(ia.abs().sum()) == 0.0 and float(ib.abs().sum()) == 0.0
    noisy = torch.full((B, S, N), MASK)
    t = torch.ones(B, S)
    with torch.no_grad():
        la = m._tul_code_thinker_discrete(noisy, ta, t, ea, ia, lay, False)
        lb = m._tul_code_thinker_discrete(noisy, tb, t, eb, ib, lay, False)
        assert torch.equal(la, lb), "two different pasts gave different null-conditioned logits"
        lc = m._tul_code_thinker_discrete(noisy, tape_a, t, e_a, inj_a, lay, False)
        assert not torch.equal(la, lc)
        ok = lay.slot_valid.clone()
        s1 = m._tul_code_sample_idx(tape_a, ok, e_a, inj_a, lay, 2,
                                    generator=torch.Generator().manual_seed(1), guidance=1.0)
        s2 = m._tul_code_sample_idx(tape_a, ok, e_a, inj_a, lay, 2,
                                    generator=torch.Generator().manual_seed(1), guidance=3.0)
    assert not torch.equal(s1, s2), "guidance 3.0 changed nothing"
    # the training pass drops rows and reports it
    m.train()
    m.code_phase = 2
    torch.manual_seed(0)
    out = m(x, _y, slot_layout=lay)
    assert "code_cfg_drop_frac" in out


# ── D9: refusals ──────────────────────────────────────────────────────────────────

def _tc(**kw):
    from test_tul_gl1 import _tul
    base = dict(tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict",
                sigreg_lambda=0.0, mux_beta=0.0, token_state_dropout=0.0, code=True,
                code_discrete=True, code_noise=0.0)
    base.update(kw)
    return _tul(**base)


def test_d9_config_refusals():
    _tc()
    for bad, msg in ((dict(code_noise=0.5), "code_noise"), (dict(code_xm_k=2), "code_xm_k"),
                     (dict(code_infer_steps=9), "exceed"), (dict(code_rollout_steps=99), "exceed"),
                     (dict(code_mask_schedule="bogus"), "code_mask_schedule"),
                     (dict(code_sub_p=1.0), "code_sub_p"), (dict(prefix_k=1), "prefix_k"),
                     (dict(code_sigreg_lambda=0.1), "CONTINUOUS"),
                     (dict(code_target_lambda=0.5), "CONTINUOUS"),
                     (dict(code_vq_codebook=1), "code_vq_codebook"),
                     (dict(code_noise_renorm=True), "flow-thinker")):
        with pytest.raises(ValueError, match=msg):
            _tc(**bad)
    with pytest.raises(ValueError, match="code_discrete=false"):
        _tc(code_discrete=False, code_sub_p=0.1, code_noise=0.5)
    with pytest.raises(TypeError):
        m = _dmodel()
        z = torch.zeros(1, 2, 2, m.cfg.d_model)
        m._tul_code_sample(z, torch.ones(1, 2, dtype=torch.bool), None, None, None, 1)


# ── D10: the arm composes ─────────────────────────────────────────────────────────

def test_d10_tul_code_d_composes_with_the_discrete_knobs_in_the_manifest(monkeypatch):
    cfg, rt = _runtime("tul_code_d", monkeypatch)
    tc = rt.model_cfg
    assert tc.code and tc.code_discrete and tc.code_noise == 0.0 and tc.code_sub_p == 0.3
    assert tc.code_vq_codebook == 512 and tc.code_vq_groups == 4
    assert tc.code_infer_steps == tc.prefix_k * tc.code_vq_groups
    assert tc.code_cfg_drop == 0.1
    man = rt.wandb_manifest if hasattr(rt, "wandb_manifest") else None
    if man is not None:
        assert man.get("code_discrete") is True


# ── D11 / D12: eval modes, generation, the marginal ───────────────────────────────

def test_d11_eval_modes_count_their_passes_and_generation_writes_from_one_sample():
    x, y, lay, _ = _batch()
    m = _dmodel().eval()
    _arm_sym_head(m)
    S = int(lay.slot_valid.shape[1])
    with torch.no_grad():
        m.tul_forward_ablated(x, y, lay, code_mode="encoder")
        assert m._code_last_passes == 0
        m.tul_forward_ablated(x, y, lay, code_mode="sampled", code_steps=3)
        assert m._code_last_passes == 3
        m.tul_forward_ablated(x, y, lay, code_mode="rolled", code_steps=2)
        assert m._code_last_passes == 2 * S
        a = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=5)["code_cells"]
        b = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=5)["code_cells"]
        c = m(x, slot_layout=lay, code_mode="generate", code_steps=2, code_seed=6)["code_cells"]
    assert torch.equal(a, b)
    ok = lay.slot_valid
    rms = a.float().pow(2).mean(-1).sqrt()[ok]
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3), "generated cells are off the RMS shell"
    open_slot = lay.slot_valid.sum(1) - 1
    diff = any(not torch.equal(a[i, open_slot[i]], c[i, open_slot[i]]) for i in range(a.shape[0]))
    assert diff, "the open slot's sample ignored code_seed"
    from morph.training.code_eval import code_marginal_ce
    with torch.no_grad():
        r = code_marginal_ce(m, x, y, lay, 3, 2)
    assert r["ce_marginal"] <= r["ce_single_mean"] + 1e-6


def test_d12_generation_end_to_end_on_a_discrete_code():
    from test_tul_gl1 import _rule, _spec
    from morph.inference.tul_generate import generate_tul
    m = _dmodel().eval()
    _arm_sym_head(m)
    rule, spec = _rule(), _spec()
    prompt = [5, 6, 7, 8, 10, 5, 6, 7, 8, 9]
    toks, _ = generate_tul(m, prompt, rule, spec, max_new_tokens=10, temperature=1.0,
                           seed=0, emit_source="token")
    assert len(toks) == 10
