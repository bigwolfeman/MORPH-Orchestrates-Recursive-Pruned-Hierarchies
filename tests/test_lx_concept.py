"""LX-Concept groundwork (`lab/divergence/_concept.py` and its four scripts), the
amplitude-pin probe (`lab/divergence/lx_amp_pin.py`, the `_slot_pass_hook` seam in
`_tul_core`) and the K3-K6 read added to the LX scorers.

What each test pins, on tiny fp32 CPU models:
  * span keys are the next span's stream range; the key join matches spans across two
    packings whatever the row order, and a packing with a different boundary rule is caught
    by `compare_packing` (the audit's alignment proof);
  * the teacher read's codes are E's own output (unit RMS on coded slots, 0 elsewhere), its
    per-position CE reproduces the forward's `ce_tokens`, and the noised read is seeded;
  * `span_grads` is EXACT and ISOLATED: its NLL at E's code equals the teacher read's span
    NLL, its gradient matches a central finite difference of that span's NLL alone, is the
    same whether spans share a forward or not, and differs from the gradient of the summed
    NLL of every span that reads the cell (so the isolation is not vacuous);
  * CoCoMix attribution equals pre-activation times the autograd gradient w.r.t. c;
  * k-means recovers separated clusters; the TopK SAE keeps k latents, unit decoder rows,
    and learns (FVU falls);
  * the pre-check head math: set CE, the marginal prior, hits@k, the row split, a head that
    beats the prior on informative features and not on noise;
  * the student read: entry = gathered input_norm(prelude), exit = readout of the h_slots
    the parallel head reads, and the coda never runs;
  * the pin: off (and an identity hook) is the unpatched forward bit for bit; on, every
    valid slot's outer channels sit at their entry RMS after every pass while the ctx slice
    and pads are untouched; the hook is removed after, also on error; training refuses it;
  * K3-K6: the Stage 1 clauses, `paired_vs_ruler.within_arm` and `core_depth_sweep`.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, "lab/divergence")
import _concept as cc  # noqa: E402
from _rows import pack_rows  # noqa: E402
from lx_amp_pin import AmpPin, amp_pin, k_pairs, outer_mask  # noqa: E402

from morph.model.tul import gather_valid  # noqa: E402
from morph.model.tul_layout import BoundaryRule  # noqa: E402
from test_tul_code import _model as _code_model  # noqa: E402
from test_tul_gl1 import DOT, V, _batch, _spec  # noqa: E402
from test_tul_lxtul_e import _model as _lx_model  # noqa: E402
from test_tul_lxtul_e import _table  # noqa: E402
from test_tul_strict_geometry import _pack  # noqa: E402


def _idx(layout) -> torch.Tensor:
    """Stream index per position: row b's token positions numbered from b * 10_000."""
    idx = torch.full(layout.slot_mask.shape, -1, dtype=torch.long)
    tok = ~layout.slot_mask
    for b in range(idx.shape[0]):
        p = tok[b].nonzero().flatten()
        idx[b, p] = b * 10_000 + torch.arange(p.numel())
    return idx


def _rule(min_span: int = 4) -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=min_span, span_cap=8, eos_id=0)


def _rt(min_span: int = 4):
    spec = _spec()
    data = SimpleNamespace(spec_for=lambda L: spec, rule=_rule(min_span))
    return SimpleNamespace(data_cfg=data), SimpleNamespace(data=SimpleNamespace(seq_len=48))


def _stream(n: int = 1200, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=n)
    ids[ids == 4] = 5
    ids[::6] = DOT
    ids[::7] = 11
    return ids.tolist()


# ── keys, join, packing comparison ──────────────────────────────────────────────────


def test_span_keys_are_the_next_spans_stream_range_and_the_join_is_order_free():
    rt, cfg = _rt()
    batches = pack_rows(_stream(), rt, cfg, 2, False)
    inp, lab, lay, idx = batches[0]
    k = cc.span_keys(lay, idx)
    assert k["row"].size > 4
    ok = cc.coded_slots(lay)
    assert k["row"].size == int(ok.sum())
    for j in range(k["row"].size):
        b, s = int(k["row"][j]), int(k["slot"][j])
        pos = ((~lay.slot_mask[b]) & (lay.bag_id[b] == s + 1)).nonzero().flatten()
        assert int(idx[b, pos[0]]) == k["first"][j] and int(idx[b, pos[-1]]) == k["last"][j]
        assert k["n_tok"][j] == pos.numel() == k["last"][j] - k["first"][j] + 1
    perm = np.random.default_rng(1).permutation(k["row"].size)
    kp = {n: v[perm] for n, v in k.items()}
    ia, ib = cc.join_keys(k, kp)
    assert ia.size == k["row"].size
    assert np.array_equal(k["slot"][ia], kp["slot"][ib])
    assert np.array_equal(k["row"][ia], kp["row"][ib])
    dup = {n: np.concatenate([v, v[:1]]) for n, v in k.items()}
    with pytest.raises(ValueError, match="duplicate"):
        cc.join_keys(dup, k)


def test_compare_packing_is_clean_for_one_rule_and_catches_another():
    rt, cfg = _rt()
    s = _stream()
    a = pack_rows(s, rt, cfg, 2, False)
    same = cc.compare_packing(a, pack_rows(s, rt, cfg, 2, False))
    assert same["spans_joined"] == same["spans_a"] > 0
    assert same["span_token_mismatches"] == same["spans_only_a"] == 0
    assert all(v == 0 for v in same["row_field_mismatches"].values())
    rt6, _ = _rt(min_span=6)
    other = cc.compare_packing(a, pack_rows(s, rt6, cfg, 2, False))
    assert other["spans_only_a"] > 0 or other["span_token_mismatches"] > 0
    assert other["row_field_mismatches"]["bag_id"] > 0


# ── the teacher ─────────────────────────────────────────────────────────────────────


def _teacher():
    m = _code_model(tul_code_noise=3.0, tul_code_noise_renorm=True).eval()
    return m


def test_teacher_read_codes_ce_and_seeded_noise():
    x, y, lay, _ = _batch()
    m = _teacher()
    r = cc.teacher_read(m, x, y, lay, "cpu", tol=1e-5)
    assert r["dev"] < 1e-5
    with torch.no_grad():
        ref = m(x, labels=y, slot_layout=lay, code_mode="encoder")
    assert abs(float(r["ce"].sum() / r["scored"].sum()) - float(ref["ce_tokens"])) < 1e-5
    ok = r["ok"]
    assert torch.equal(ok, cc.coded_slots(lay))
    rms = r["z"].pow(2).mean(-1).sqrt()                                   # [B, S, M]
    assert torch.allclose(rms[ok], torch.ones_like(rms[ok]), atol=1e-4)
    assert float(r["z"][~ok].abs().max()) == 0.0
    n1 = cc.teacher_read(m, x, y, lay, "cpu", noise=3.0, seed=5, tol=1e-5)
    n2 = cc.teacher_read(m, x, y, lay, "cpu", noise=3.0, seed=5, tol=1e-5)
    n3 = cc.teacher_read(m, x, y, lay, "cpu", noise=3.0, seed=6, tol=1e-5)
    assert torch.equal(n1["ce"], n2["ce"]) and not torch.equal(n1["ce"], n3["ce"])
    assert not torch.equal(n1["ce"], r["ce"])
    assert torch.equal(n1["z"], r["z"])          # the clean code is what is returned
    assert "forward" not in vars(m.tul_code_enc)  # the seam is restored


def test_span_grads_are_exact_isolated_and_group_free():
    torch.manual_seed(0)
    x, y, lay, _ = _batch()
    m = _teacher()
    idx = _idx(lay)
    k = cc.span_keys(lay, idx)
    r = cc.teacher_read(m, x, y, lay, "cpu", tol=1e-5)
    nll_ref = cc.span_nll(r["ce"], r["scored"], lay, k)
    z = r["z"][torch.as_tensor(k["row"]), torch.as_tensor(k["slot"])]    # [n, M, C]
    g3, nll3 = cc.span_grads(m, x, y, lay, k["row"], k["slot"], z, "cpu", group=3)
    g1, nll1 = cc.span_grads(m, x, y, lay, k["row"], k["slot"], z, "cpu", group=1)
    np.testing.assert_allclose(nll3, nll_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(g3, g1, rtol=1e-4, atol=1e-6)
    assert np.abs(g3).max() > 0
    # a central finite difference of span j's NLL ALONE, along a random direction
    gen = torch.Generator().manual_seed(3)
    for j in (0, len(k["row"]) // 2):
        u = torch.randn(z[j].numel(), generator=gen, dtype=torch.float64).float()
        u = u / u.norm()
        h = 1e-2
        cells = torch.stack([z[j] + h * u.view_as(z[j]), z[j] - h * u.view_as(z[j])])
        _g, nl = cc.span_grads(m, x, y, lay, k["row"][[j, j]], k["slot"][[j, j]], cells, "cpu",
                               group=2)
        fd = (nl[0] - nl[1]) / (2 * h)
        an = float(torch.from_numpy(g3[j]) @ u)
        assert abs(fd - an) <= 2e-3 * max(1.0, abs(fd)), (j, fd, an)
    # isolation is not vacuous: LATER spans of the same row also read slot s's cell, so the
    # gradient of the row's summed NLL w.r.t. that cell differs from span s+1's own
    j = 0
    b, s = int(k["row"][j]), int(k["slot"][j])
    later = [i for i in range(len(k["row"])) if k["row"][i] == b and k["slot"][i] > s]
    assert later
    from morph.model.fused_ce import fused_linear_label_logprob
    from morph.model.tul_code import code_rmsnorm
    leaf = z[j].clone().requires_grad_(True)
    real = m.tul_code_enc.forward

    def enc(xs, lay_):
        zz, okk = real(xs, lay_)
        sel = torch.zeros_like(okk)
        sel[b, s] = True
        return torch.where(sel.view(*sel.shape, 1, 1), code_rmsnorm(leaf)[None, None], zz), okk

    m.tul_code_enc.forward = enc
    try:
        with torch.enable_grad(), cc.Spy(m, "_tul_group_losses") as sp:
            m(x, labels=y, slot_layout=lay, code_mode="encoder")
        xh = sp.one()[0][0]
        sc = cc.scored_mask(m, y, lay)
        top = max(int(k["slot"][i]) for i in later) + 1
        pos = (~lay.slot_mask[b]) & sc[b] & (lay.bag_id[b] >= s + 1) & (lay.bag_id[b] <= top)
        with torch.enable_grad():
            lp = fused_linear_label_logprob(xh[b][pos], m.embed.lm_weight(), y[b][pos],
                                            mask_token_id=m.cfg.tul.slot_id)
            (gtot,) = torch.autograd.grad(-lp.sum(), [leaf])
    finally:
        del m.tul_code_enc.forward
    diff = (gtot.flatten() - torch.from_numpy(g3[j])).norm() / gtot.norm()
    assert float(diff) > 1e-3


# ── attribution and dictionaries ────────────────────────────────────────────────────


def test_attribution_is_pre_times_the_autograd_gradient():
    torch.manual_seed(0)
    sae = cc.TopKSAE(12, 20, 3, seed=1)
    x = torch.randn(5, 12)
    g = torch.randn(5, 12)                     # d loss / d D(c)
    c, pre = sae.encode(x)
    cc_ = pre.clone().requires_grad_(True)     # gradient w.r.t. EVERY latent (the paper's c^pre)
    loss = (sae.decode(cc_) * g).sum()
    (gc,) = torch.autograd.grad(loss, [cc_])
    a = cc.attribution(sae, pre, g)
    torch.testing.assert_close(a, pre * gc)
    top = cc.top_attr(a, 4)
    assert torch.equal(top, torch.topk(a, 4, dim=-1).indices)
    low = cc.top_attr(a, 4, largest=False)
    assert bool((a.gather(1, low).max(1).values <= a.gather(1, top).min(1).values).all())


def test_kmeans_recovers_separated_clusters():
    g = torch.Generator().manual_seed(0)
    cent = torch.randn(5, 8, generator=g) * 10
    lab = torch.arange(500) % 5
    X = cent[lab] + 0.1 * torch.randn(500, 8, generator=g)
    c, got, inertia = cc.kmeans(X, 5, iters=30, seed=0)
    # same partition up to a relabelling
    pair = torch.stack([lab, got], 1).unique(dim=0)
    assert pair.shape[0] == 5
    assert inertia[-1] <= inertia[0]
    a2, _d = cc.assign(X, c)
    assert torch.equal(a2, got)


def test_topk_sae_is_sparse_unit_decoder_and_learns():
    g = torch.Generator().manual_seed(0)
    D, n, k = 16, 32, 4
    atoms = torch.randn(n, D, generator=g)
    idx = torch.randint(n, (2000, k), generator=g)
    X = atoms[idx].sum(1) * 0.5
    sae0 = cc.TopKSAE(D, 64, k, seed=0)
    fvu0 = cc.sae_eval(sae0, X)["fvu"]
    sae, st = cc.train_sae(X, 64, k, steps=600, batch=256, lr=3e-3, seed=0)
    ev = cc.sae_eval(sae, X)
    assert ev["fvu"] < 0.5 * fvu0, (ev, fvu0)
    c, _p = sae.encode(X[:10])
    assert bool(((c != 0).sum(1) <= k).all())
    torch.testing.assert_close(sae.W_dec.norm(dim=1), torch.ones(64), atol=1e-5, rtol=0)


# ── the pre-check head ──────────────────────────────────────────────────────────────


def test_set_ce_prior_hits_and_row_split():
    logp = torch.log_softmax(torch.tensor([[2.0, 1.0, 0.0, -1.0]]), -1)
    y = torch.tensor([[0, 2]])
    assert float(cc.set_ce(logp, y)) == pytest.approx(-(logp[0, 0] + logp[0, 2]).item() / 2)
    assert float(cc.hits(logp, y, 1)) == pytest.approx(0.5)
    assert float(cc.hits(logp, y, 3)) == pytest.approx(1.0)
    ytr = torch.tensor([[0], [0], [1], [3]])
    pr = cc.marginal_logp(ytr, 4, alpha=0.0 + 1e-9)
    torch.testing.assert_close(pr.exp(), torch.tensor([0.5, 0.25, 0.0, 0.25]), atol=1e-6, rtol=0)
    # CE of the prior on its own labels is the label entropy
    ce = float(cc.set_ce(pr.unsqueeze(0).expand(4, -1), ytr).mean())
    assert ce == pytest.approx(-(0.5 * np.log(0.5) + 2 * 0.25 * np.log(0.25)), rel=1e-5)
    from concept_precheck import row_split
    rows = np.repeat(np.arange(50), 7)
    parts = row_split(rows, seed=0)
    assert sum(int(p.sum()) for p in parts) == rows.size
    for r in range(50):
        assert sum(bool(p[rows == r].any()) for p in parts) == 1


def test_head_beats_the_prior_on_signal_and_not_on_noise():
    g = torch.Generator().manual_seed(0)
    n, d, K = 3000, 16, 8
    y = torch.randint(K, (n, 1), generator=g)
    means = torch.randn(K, d, generator=g) * 1.5
    Xs = means[y[:, 0]] + torch.randn(n, d, generator=g)
    Xn = torch.randn(n, d, generator=g)
    tr, dv, te = slice(0, 2000), slice(2000, 2500), slice(2500, n)
    prior = cc.marginal_logp(y[tr], K)
    rows = np.arange(n - 2500)
    out = {}
    for name, X in (("signal", Xs), ("noise", Xn)):
        Xf, Xd, Xt = cc.standardise(X[tr], X[dv], X[te])
        h, _info = cc.fit_head(Xf, y[tr], Xd, y[dv], K, hidden=0, max_epochs=30, prior=prior)
        with torch.no_grad():
            lp = h(Xt)
        ce_h = cc.set_ce(lp, y[te]).double().numpy()
        ce_p = cc.set_ce(prior.expand(n - 2500, -1), y[te]).double().numpy()
        out[name] = cc.row_block_ci(ce_p, ce_h, rows, n_boot=200)
        assert out[name]["point"] == pytest.approx(float(ce_p.mean() - ce_h.mean()), abs=1e-9)
    assert out["signal"]["point"] > 0.5
    assert abs(out["noise"]["point"]) < 0.05


# ── the student read ────────────────────────────────────────────────────────────────


def test_student_states_are_the_entry_and_the_heads_exit_and_skip_the_coda():
    from test_tul_lxtul_e import _Spy
    _i, inp, lab, layout = _pack()
    m = _lx_model().eval()
    with cc.Spy(m, "_back_region") as coda:
        st = cc.student_states(m, inp, lab, layout, "cpu", depth=3)
    assert len(coda.calls) == 0
    K = m._code_enum_k
    core = _Spy(m, "_tul_core")
    head = _Spy(m, "_tul_spandec_par_loss")
    with torch.no_grad():
        m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, 3))
        x = core.calls[0][0][0]
        e = gather_valid(m.input_norm(x), layout.slot_index, layout.slot_valid).mean(2)
        hs = head.calls[0][0][0]
        r = m._readout(hs)
    del m._tul_core, m._tul_spandec_par_loss
    B, S = layout.slot_index.shape
    torch.testing.assert_close(st["entry"], e.float())
    torch.testing.assert_close(st["exit"], r.float().view(K, B, S, -1).mean(0))
    torch.testing.assert_close(st["exit_raw"], hs.float().mean(2).view(K, B, S, -1).mean(0))
    assert "_tul_core" not in vars(m) and "_tul_spandec_par_loss" not in vars(m)


# ── the amplitude pin ───────────────────────────────────────────────────────────────


def _fwd(m, inp, lab, layout, d=3):
    with torch.no_grad():
        lo = m(inp, labels=lab, slot_layout=layout, slot_depths=_table(layout, d))
        lg = m(inp, slot_layout=layout, slot_depths=_table(layout, d))["logits"]
    return lo["loss"], lg


def test_pin_off_and_identity_hook_are_the_unpatched_forward():
    _i, inp, lab, layout = _pack()
    m = _lx_model().eval()
    base = _fwd(m, inp, lab, layout)
    with amp_pin(m, on=False) as pin:
        assert pin is None and m._slot_pass_hook is None
        off = _fwd(m, inp, lab, layout)
    m._slot_pass_hook = lambda t, h, hn, lay: hn
    try:
        ident = _fwd(m, inp, lab, layout)
    finally:
        m._slot_pass_hook = None
    for got in (off, ident):
        assert torch.equal(got[0], base[0]) and torch.equal(got[1], base[1])


def test_pin_holds_outer_rms_at_entry_and_leaves_the_slice_and_pads():
    _i, inp, lab, layout = _pack()
    m = _lx_model().eval()
    mask = outer_mask(m)
    assert int(mask.sum()) < m.cfg.d_model and bool((~mask).any())
    s, e = int(m.injection.start), int(m.injection.end)
    assert not bool(mask[s:e].any()) and bool(mask[:s].all()) and bool(mask[e:].all())
    base = _fwd(m, inp, lab, layout, d=3)
    rec = []
    with amp_pin(m) as pin:
        real = m._slot_pass_hook

        def spy(t, h, hn, lay):
            out = real(t, h, hn, lay)
            rec.append((t, h.detach().clone(), hn.detach().clone(), out.detach().clone(),
                        lay.slot_valid.clone(), real.entry.clone()))
            return out
        m._slot_pass_hook = spy
        pinned = _fwd(m, inp, lab, layout, d=3)
    assert m._slot_pass_hook is None
    assert pin.stats and max(pin.stats) == 2
    assert not torch.equal(pinned[1], base[1])
    ts = sorted({r[0] for r in rec})
    assert ts == [0, 1, 2]
    for t, h, hn, out, valid, entry in rec:
        if t == 0:
            ent = h.float()[..., mask].pow(2).mean(-1).sqrt()
            torch.testing.assert_close(entry, ent)
        rms = out.float()[..., mask].pow(2).mean(-1).sqrt()
        vm = valid.view(*valid.shape, 1).expand_as(rms)
        torch.testing.assert_close(rms[vm], entry[vm], rtol=1e-4, atol=1e-5)
        assert torch.equal(out[..., s:e], hn[..., s:e])                  # ctx slice untouched
        assert torch.equal(out[~valid], hn[~valid])                      # pads untouched
        before = hn.float()[..., mask].pow(2).mean(-1).sqrt()
        if t > 0:
            assert not torch.allclose(before[vm], entry[vm], rtol=1e-3)  # the pin acted
    # removed on error; refused in training
    with pytest.raises(RuntimeError, match="boom"):
        with amp_pin(m):
            raise RuntimeError("boom")
    assert m._slot_pass_hook is None
    m.train()
    with amp_pin(m), pytest.raises(RuntimeError, match="EVAL instrument"):
        m(inp, labels=lab, slot_layout=layout)
    assert m._slot_pass_hook is None
    assert isinstance(AmpPin(m.eval()), AmpPin)


# ── K3-K6 ───────────────────────────────────────────────────────────────────────────


def test_k36_in_stage1_pin_and_core_depth_sweep():
    from lxtul_e_stage1_score import _ci, k36_clauses
    rng = np.random.default_rng(0)
    n = 400
    blk = np.repeat(np.arange(20), 20)
    p = {d: {"par_mix": rng.normal(3 - 0.01 * d, 1, n), "coda_mix": rng.normal(4 - 0.02 * d, 1, n)}
         for d in (1, 3, 6)}
    q = {d: {"par_mix": rng.normal(3, 1, n), "coda_mix": rng.normal(4, 1, n)} for d in (1, 3, 6)}
    got = k36_clauses(p, q, blk, blk, [1, 3, 6], 200, 0)
    assert set(got) == {"P-3 e4 par K3-K6", "P-3 e1 par K3-K6", "P-4 e4 coda K3-K6 (mixture)",
                        "P-4 e1 coda K3-K6"}
    assert got["P-4 e4 coda K3-K6 (mixture)"] == _ci(p[3]["coda_mix"], p[6]["coda_mix"], blk,
                                                     None, 200, 0)
    assert got["P-4 e4 coda K3-K6 (mixture)"]["point"] == pytest.approx(
        float(p[3]["coda_mix"].mean() - p[6]["coda_mix"].mean()))
    assert k36_clauses(p, q, blk, blk, [1, 6], 200, 0) == {}
    kp = k_pairs({d: p[d]["coda_mix"] for d in (1, 3, 6)}, blk, 200, 0)
    assert set(kp) == {"K1-K6", "K3-K6"}
    assert kp["K3-K6"]["point"] == pytest.approx(
        float(p[3]["coda_mix"].mean() - p[6]["coda_mix"].mean()))
    from core_depth_sweep import _bootstrap_pairs
    rs = {d: rng.normal(10, 1, 30) for d in (1, 3, 6)}
    bp = _bootstrap_pairs([1, 3, 6], rs, np.full(30, 5.0))
    assert "K3-K6" in bp and bp["K3-K6"]["point"] == pytest.approx(
        float(rs[3].sum() / 150 - rs[6].sum() / 150))


def test_paired_vs_ruler_within_arm_k36(tmp_path):
    from paired_vs_ruler import within_arm
    rng = np.random.default_rng(0)
    tok = np.arange(5000)
    ce = {f"ce_{d}": rng.normal(4.0 - 0.01 * d, 1, tok.size).astype(np.float32)
          for d in (1, 2, 3, 6)}
    npz = tmp_path / "arm.tokens.npz"
    np.savez(npz, tok_index=tok, **ce)
    arm = {"row_ce_sum": {str(d): [0.0] for d in (1, 2, 3, 6)}, "row_n_tokens": [1],
           "tokens_npz": str(npz)}
    json.dump({"a": arm}, open(tmp_path / "s.json", "w"))
    w = within_arm(arm, (str(tmp_path),))
    assert set(w) == {"arm_K1-K6", "arm_K3-K6"}
    exp = float(ce["ce_3"].astype(np.float64).mean() - ce["ce_6"].astype(np.float64).mean())
    assert w["arm_K3-K6"][0] == pytest.approx(exp, abs=1e-9)
    assert w["arm_K3-K6"][1] < exp < w["arm_K3-K6"][2]
    del arm["row_ce_sum"]["3"]
    assert set(within_arm(arm, (str(tmp_path),))) == {"arm_K1-K6"}
