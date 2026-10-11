"""Contracts of the 2026-10-11 additions to `lab/divergence/cell_lens.py`.

  * COPIED vs NOVEL: the rule on a hand-built row (slot positions excluded, within-span repeats
    excluded, a token that first occurs AT the next span's start is novel), the packed-batch
    helpers against an independent python loop, and the patch split against a per-token CE
    written here (it must add back to the oracle's table).
  * PROBE split: the split sums add back to the unsplit numbers; a toy whose one cell carries
    the token only on copied rows reads copied >> novel.
  * SPECIAL: NMI is 1 for a deterministic relabelling, sits at its permutation null for random
    labels and above it for a real dependence; the contingency adds up to the span table.
  * SPANDEC: the free-running decode equals a greedy loop written here (full window, one slot at
    a time), the teacher-forced CE equals the model's own `_tul_spandec_loss`, a model without a
    decoder raises.
  * GEOM: CKA of a matrix with itself is 1 and invariant to rotation, effective rank of an
    orthonormal frame is its width, controls behave.

CPU, fp32, the tests/test_cell_lens.py fixtures.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
import cell_lens as cl  # noqa: E402
import morph.model.transformer as T  # noqa: E402
from morph.model.tul_spandec import next_span_slots  # noqa: E402
from test_cell_lens import DEV, M, _fan, _lsel  # noqa: E402
from test_tul_fan import _batch  # noqa: E402
from test_tul_lxfan import _build, _fan_kw  # noqa: E402


# ── copied vs novel ──────────────────────────────────────────────────────────────────────
def test_copied_rule_on_a_hand_built_row():
    # positions 0-3 text, 4-5 slot cells (id 99 must NOT count as text), 6-9 the next span
    x = torch.tensor([[7, 8, 9, 8, 99, 99, 3, 7, 8, 5]])
    slot_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    tok = torch.tensor([[7, 3, 8, 5, 99, 9, -100, 3]])
    start = torch.full_like(tok, 6)
    got = cl.copied_tokens(x, slot_mask, tok, start)[0].tolist()
    # 7 seen at 0; 3 FIRST occurs at 6 == start (not before it); 8 seen; 5 only inside the span;
    # 99 only at slot positions; 9 seen; no token; 3 again is still novel
    assert got == [True, False, True, False, False, True, False, False]
    # a later start admits the in-span tokens that precede it
    assert cl.copied_tokens(x, slot_mask, torch.tensor([[3, 5]]), torch.tensor([[7, 9]]))[
        0].tolist() == [True, False]


def _loop_copied(x, slot_mask, tok, start):
    seen = {int(x[p]) for p in range(int(start)) if not bool(slot_mask[p])}
    return int(tok) in seen


def test_packed_batch_helpers_match_an_independent_loop():
    _ids, inp, lab, layout = _batch(M)
    B, S = layout.slot_valid.shape
    ids, valid = next_span_slots(inp, layout, 8)
    tok = torch.where(valid, ids, torch.full_like(ids, cl.IGNORE))
    got = cl.copied_next_span(inp, layout, tok)
    n_true = 0
    for b in range(B):
        for s in range(S):
            start = int(layout.slot_index[b, s]) + int(layout.prefix_k)
            for j in range(tok.shape[2]):
                want = bool(valid[b, s, j]) and _loop_copied(inp[b], layout.slot_mask[b],
                                                             tok[b, s, j], start)
                assert bool(got[b, s, j]) == want, (b, s, j)
                n_true += want
    assert 0 < n_true < int(valid.sum())                      # both classes occur
    sc = cl.copied_scored(inp, lab, layout)
    gid, keep, _l, _g = T.span_ce_index(lab, layout)
    for b in range(B):
        for p in range(inp.shape[1]):
            if not bool(keep[b, p]):
                continue
            bag = min(int(layout.bag_id[b, p]), S)
            start = int(layout.slot_index[b, max(bag - 1, 0)]) + int(layout.prefix_k)
            assert bool(sc[b, p]) == _loop_copied(inp[b], layout.slot_mask[b], lab[b, p], start)


@pytest.mark.parametrize("make", [_fan, _lsel], ids=["ungraded_fan", "latent_selected"])
def test_patch_split_adds_back_and_matches_a_per_token_loop(make):
    m = make()
    _ids, inp, lab, layout = _batch(M)
    tb = cl.patch_batch(m, inp, lab, layout, 2, DEV)
    cl.check_split(tb, "t")
    ok = tb["ok"]
    assert tb["n_cp"][ok].sum() > 0 and tb["n_nv"][ok].sum() > 0
    # independent: per-token CE of the all-cells coda output, bins summed in python
    w = m.embed.lm_weight()
    S = layout.slot_valid.shape[1]
    want_cp = torch.zeros_like(tb["mixed"])
    want_nv = torch.zeros_like(tb["mixed"])
    with torch.no_grad():
        for b in range(inp.shape[0]):
            ce = F.cross_entropy((tb["xh_all"][b] @ w.t()).float(), lab[b].clamp_min(0),
                                 reduction="none")
            for p in range(inp.shape[1]):
                if lab[b, p] < 0 or bool(layout.slot_mask[b, p]):
                    continue
                bag = int(layout.bag_id[b, p])
                if not 1 <= bag <= S:
                    continue
                start = int(layout.slot_index[b, bag - 1]) + int(layout.prefix_k)
                tgt = want_cp if _loop_copied(inp[b], layout.slot_mask[b], lab[b, p], start) \
                    else want_nv
                tgt[b, bag - 1] += ce[p]
    assert torch.allclose(tb["mixed_cp"][ok], want_cp[ok], atol=1e-4, rtol=1e-4)
    assert torch.allclose(tb["mixed_nv"][ok], want_nv[ok], atol=1e-4, rtol=1e-4)
    res = cl.run_patch(m, [(inp, lab, layout)], [1, 3], DEV, log=lambda *_: None)
    for d, e in res.items():
        c = e["copy"]
        assert 0 < c["frac_tokens_copied"] < 1
        n_c, n_n = c["copied"]["n_tokens"], c["novel"]["n_tokens"]
        assert abs(n_c + n_n - e["n_tokens"]) < 1e-6
        for i in range(M):
            mix = (c["copied"]["cells"][i] * n_c + c["novel"]["cells"][i] * n_n) / e["n_tokens"]
            assert abs(mix - e["cells"][i]) < 1e-5
        sp = e["spans"]
        assert len(sp["win"]) == int(e["n_spans"]) == len(sp["ce"]) == len(sp["all"])
        for ce, win in zip(sp["ce"], sp["win"]):
            assert win == int(np.argmin(ce)) and len(ce) == M


# ── probe split ──────────────────────────────────────────────────────────────────────────
def test_collect_features_marks_copied_like_the_loop():
    m = _fan()
    ids, inp, lab, layout = _batch(M)
    f = cl.collect_features(m, [(inp, lab, layout)], [1, 2], 4, DEV, text_tokens=6,
                            store_dtype=torch.float32)
    assert f.copied is not None and f.copied.shape == f.text.shape
    id_, valid = next_span_slots(inp, layout, 6)
    sel = valid[:, :, 0]
    bs, ss = sel.nonzero(as_tuple=True)
    for n in range(len(bs)):
        b, s = int(bs[n]), int(ss[n])
        start = int(layout.slot_index[b, s]) + int(layout.prefix_k)
        for j in range(6):
            want = bool(valid[b, s, j]) and _loop_copied(inp[b], layout.slot_mask[b],
                                                         id_[b, s, j], start)
            assert bool(f.copied[n, j]) == want


def test_probe_split_adds_back_and_separates_copied_from_novel():
    torch.manual_seed(0)
    V, C, N, Mc = 40, 24, 1200, 4
    emb = torch.randn(V, C)
    rot, _ = torch.linalg.qr(torch.randn(C, C))
    tok = torch.randint(0, V, (N,))
    copied = (torch.arange(N) % 2 == 0)
    feats = torch.randn(N, 1, Mc, C)
    carry = emb[tok] @ rot + 0.1 * torch.randn(N, C)
    feats[:, 0, 2] = torch.where(copied.unsqueeze(1), carry, feats[:, 0, 2])   # only on copied
    row = torch.arange(N) // 6
    tr = cl.Features(feats[:800], tok[:800].view(-1, 1), tok[:800].view(-1, 1), row[:800], [3],
                     copied[:800].view(-1, 1))
    va = cl.Features(feats[800:], tok[800:].view(-1, 1), tok[800:].view(-1, 1), row[800:], [3],
                     copied[800:].view(-1, 1))
    probes = cl.train_probes(tr, emb, steps=500, batch=128, lr=3e-3, wd=0.0, device=DEV,
                             log=lambda *_: None)
    res = cl.eval_probes(probes, va, cl.unigram_stats(tr.tgt, V), DEV)
    sp = res["probe_split"]["pass3_off1"]
    n_c, n_n = res["copy"]["n_copied"][0], res["copy"]["n_novel"][0]
    assert n_c + n_n == res["n_scored"][0] == 400 and abs(res["copy"]["frac_copied"][0] - 0.5) < 1e-9
    for nm in res["names"]:
        for stat in ("ce", "top1", "top5"):
            mix = (sp[nm]["copied"][stat] * n_c + sp[nm]["novel"][stat] * n_n) / (n_c + n_n)
            assert abs(mix - res["probe"]["pass3_off1"][nm][stat]) < 1e-6, (nm, stat)
    assert sp["cell2"]["copied"]["top1"] > 0.6 and sp["cell2"]["novel"]["top1"] < 0.2
    assert sp["cell2"]["copied"]["ce"] + 1.0 < sp["cell2"]["novel"]["ce"]
    assert "novel" in cl.probe_markdown("t", res)


# ── special ──────────────────────────────────────────────────────────────────────────────
def test_first_token_classes():
    cases = {".": "punctuation", " ,": "punctuation", ".\n": "punctuation", "\n\n": "whitespace",
             " ": "whitespace", "2024": "digit", " 7": "digit", " The": "capitalised",
             "the": "lowercase", " the": "lowercase", "�": "other", "é": "lowercase",
             "É": "capitalised", "ab�": "other"}
    for text, want in cases.items():
        assert cl.first_token_class(text) == want, text


def test_quartile_bins():
    b, names = cl.quartile_bins(np.arange(100.0))
    assert sorted(set(b.tolist())) == [0, 1, 2, 3] and len(names) == 4
    assert np.bincount(b).tolist() == [25, 25, 25, 25]
    z, zn = cl.quartile_bins(np.r_[np.zeros(90), np.arange(10.0)])      # ties collapse bins
    assert len(zn) <= 3 and z.max() == len(zn) - 1


def test_nmi_is_one_for_a_deterministic_map_and_null_for_random_labels():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, 2000)
    perfect = cl.nmi_with_null(a, (a + 1) % 4, n_perm=50)
    assert abs(perfect["nmi"] - 1.0) < 1e-12 and perfect["null_p95"] < 0.05
    merged = cl.nmi_with_null(a, a // 2, n_perm=50)                       # function of a, not 1:1
    assert 0.4 < merged["nmi"] < 1.0
    rand = cl.nmi_with_null(a, rng.integers(0, 5, 2000), n_perm=200, seed=1)
    assert rand["nmi"] < 2 * rand["null_mean"] + 0.01 and rand["p_value"] > 0.01
    assert rand["null_p95"] >= rand["null_mean"] > 0
    dep = np.where(rng.random(2000) < 0.6, a, rng.integers(0, 4, 2000))
    real = cl.nmi_with_null(a, dep, n_perm=200, seed=1)
    assert real["nmi"] > 5 * real["null_p95"] and real["p_value"] < 0.01
    assert cl.nmi_from_joint(np.array([[10.0, 0], [5, 0]])) == 0.0        # constant column


def test_run_special_contingency_adds_up_to_the_span_table():
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    val = [(inp, lab, layout)]
    res = cl.run_patch(m, val, [2], DEV, log=lambda *_: None)
    sp = res[2]["spans"]
    props = cl.span_properties(val)
    # independent: the next span's length from the layout
    _i, valid = next_span_slots(inp, layout, 64)
    for b, s in zip(sp["b"], sp["slot"]):
        assert props[0]["n"][b, s] == int(valid[b, s].sum()) > 0
    out = cl.run_special(sp, props, lambda ids: "".join(chr(97 + i % 26) for i in ids), M,
                         n_perm=20, seed=0)
    n = len(sp["win"])
    assert out["n_spans"] == n
    for name, r in out["properties"].items():
        cont = np.array(r["contingency"])
        assert cont.sum() == n, name
        assert cont.sum(1).tolist() == [sp["win"].count(i) for i in range(M)], name
        assert 0.0 <= r["nmi"] <= 1.0 + 1e-9 and r["null_p95"] >= r["null_mean"], name
    assert "NMI" in cl.special_markdown("t", 2, out)


# ── spandec ──────────────────────────────────────────────────────────────────────────────
def _greedy_reference(dec, z, steps, emb, slot_id):
    """One slot at a time, the FULL window every step, argmax of the tied head."""
    out = []
    with torch.no_grad():
        for zi in z:
            ids = torch.zeros(1, 1, steps, dtype=torch.long)
            ones = torch.ones(1, 1, steps, dtype=torch.bool)
            for j in range(steps):
                st = dec.decode(zi.view(1, 1, -1), ids, ones, emb)[0, 0, j]
                lg = st @ emb.t()
                lg[slot_id] = float("-inf")
                ids[0, 0, j] = int(lg.argmax())
            out.append(ids[0, 0])
    return torch.stack(out)


def test_free_running_decode_matches_an_independent_greedy_loop():
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    si = cl.collect_spandec_inputs(m, [(inp, lab, layout)], 2, DEV)
    n = si.ids.shape[0]
    assert n >= 4 and si.z_cells.shape == (n, M, m.cfg.d_model) and si.z_trained is not None
    assert torch.allclose(si.z_trained, si.z_mean, atol=1e-4)           # spandec_read: mean
    emb = m.embed.lm_weight().detach()
    slot_id = int(m.cfg.tul.slot_id)
    steps = int(si.valid.sum(1).max())
    vars_ = cl.spandec_variants(si, seed=0)
    assert list(vars_) == [f"cell{i}" for i in range(M)] + ["mean", "trained", "noise", "shuffled"]
    rms_ratio = (vars_["noise"].pow(2).mean(-1).sqrt() / si.z_mean.pow(2).mean(-1).sqrt()).mean()
    assert 0.8 < float(rms_ratio) < 1.2
    sub = slice(0, 6)
    seen = []
    for name in ("cell0", "mean", "noise"):
        z = vars_[name][sub]
        got = cl.greedy_free_run(m.tul_spandec, z, steps, emb, slot_id, DEV, chunk=4)
        want = _greedy_reference(m.tul_spandec, z, steps, emb, slot_id)
        assert torch.equal(got, want), name
        assert not (got == slot_id).any()
        seen.append(got)
    assert not torch.equal(seen[0], seen[2]) or not torch.equal(seen[1], seen[2])   # not constant
    assert len(set(torch.cat(seen).flatten().tolist())) > 3


def test_teacher_forced_ce_equals_the_models_own_spandec_loss():
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    with cl.SpandecTap(m) as tap:
        m.tul_forward_ablated(inp, None, layout, slot_depths=cl.forced_depths(layout, 2, M))
    a = tap.args
    assert "_tul_spandec_loss" not in m.__dict__
    with torch.no_grad():
        ref = float(m._tul_spandec_loss(a["h_slots"], a["input_ids"], layout, cells=a["cells"]))
    si = cl.collect_spandec_inputs(m, [(inp, lab, layout)], 2, DEV)
    emb = m.embed.lm_weight().detach()
    ce, hit = cl.teacher_forced(m.tul_spandec, si.z_mean, si.ids, si.valid, emb,
                                int(m.cfg.tul.slot_id), DEV, chunk=3)
    mine = float(ce.sum() / si.valid.sum())
    assert abs(mine - ref) < 1e-3 * max(1.0, abs(ref)), (mine, ref)
    assert (ce[~si.valid] == 0).all() and not hit[~si.valid].any()


def test_run_spandec_scores_and_examples():
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    si = cl.collect_spandec_inputs(m, [(inp, lab, layout)], 2, DEV)
    res = cl.run_spandec(m, si, DEV, max_slots=10, log=lambda *_: None)
    assert res["n_slots"] == 10 and "noise" in res["variants"] and "shuffled" in res["variants"]
    for nm, r in res["variants"].items():
        assert 0.0 <= r["exact_match"] <= 1.0 and 0.0 <= r["free_acc_all"] <= 1.0, nm
        assert r["positions"]["1"]["n"] == 10 and r["tf_ce_all"] > 0, nm
    exs = cl.spandec_examples(res, lambda ids: " ".join(map(str, ids)), n=3)
    assert len(exs) == 3 and set(exs[0]["decodes"]) == set(res["variants"])
    assert len(exs[0]["true"].split()) == len(exs[0]["decodes"]["mean"].split()) >= 4
    assert "exact match" in cl.spandec_markdown("t", res)


def test_spandec_part_raises_without_a_decoder():
    m = _build(**_fan_kw(M, spandec=False, fan_all_wta_lambda=0.0)).eval()
    _ids, inp, lab, layout = _batch(M)
    with pytest.raises(ValueError, match="no span decoder"):
        cl.collect_spandec_inputs(m, [(inp, lab, layout)], 2, DEV)


# ── geometry ─────────────────────────────────────────────────────────────────────────────
def test_cka_of_a_matrix_with_itself_is_one_and_rotation_invariant():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(300, 16, generator=g) @ torch.randn(16, 16, generator=g)
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=g))
    assert abs(cl.linear_cka(x, x) - 1.0) < 1e-9
    assert abs(cl.linear_cka(x, x @ q) - 1.0) < 1e-9
    assert abs(cl.linear_cka(x, 3.0 * x @ q + 5.0) - 1.0) < 1e-9            # scale and shift
    ind = torch.randn(300, 16, generator=g)
    assert cl.linear_cka(x, ind) < 0.2
    assert cl.linear_cka(x, x[torch.randperm(300, generator=g)]) < 0.2     # broken pairing


def test_effective_rank():
    g = torch.Generator().manual_seed(0)
    for d in (8, 32):
        q, _ = torch.linalg.qr(torch.randn(500, d, generator=g) - 0.0)
        q = q - q.mean(0, keepdim=True)
        # a frame with equal singular values: re-orthonormalise after centring
        u, _s, vh = torch.linalg.svd(q, full_matrices=False)
        frame = u @ vh
        assert abs(cl.effective_rank(frame) - d) < 1e-6 * d
    eye = torch.eye(16)
    assert abs(cl.effective_rank(eye) - 15) < 1e-6                         # centring drops one
    r1 = torch.randn(200, 1, generator=g) @ torch.randn(1, 12, generator=g)
    assert abs(cl.effective_rank(r1) - 1.0) < 1e-6
    two = torch.cat([torch.randn(200, 1, generator=g) * 3, torch.randn(200, 1, generator=g) * 3,
                     torch.zeros(200, 4)], dim=1)
    u, _s, vh = torch.linalg.svd(two - two.mean(0), full_matrices=False)
    assert abs(cl.effective_rank(u[:, :2] @ vh[:2]) - 2.0) < 1e-6       # two equal directions


def test_within_slot_cosine_and_geom_controls():
    g = torch.Generator().manual_seed(0)
    same = torch.randn(50, 1, 8, generator=g).expand(50, 4, 8)
    assert abs(cl.within_slot_cosine(same) - 1.0) < 1e-9
    anti = torch.stack([torch.randn(50, 8, generator=g)] * 2, dim=1)
    anti[:, 1] = -anti[:, 0]
    assert abs(cl.within_slot_cosine(anti) + 1.0) < 1e-9
    # cells = shared slot signal + small private noise, over 3 passes
    n, p, mm, c = 400, 3, 3, 12
    shared = torch.randn(n, 1, 1, 4, generator=g) @ torch.randn(4, c, generator=g)   # rank 4
    feats = shared + 0.1 * torch.randn(n, p, mm, c, generator=g)
    rep = cl.geom_report(feats, seed=0)
    off = lambda mat: (np.sum(mat) - np.trace(mat)) / (mm * (mm - 1))     # noqa: E731
    real, shuf, noise = (np.array(rep[k]["cka_cells"][0]) for k in ("real", "shuffled", "noise"))
    assert off(real) > 0.9 and off(shuf) < 0.2 and off(noise) < 0.2
    assert np.allclose(np.diag(real), 1.0) and np.allclose(np.diag(shuf), 1.0)
    assert rep["real"]["cos_within"][0] > 0.9 and abs(rep["shuffled"]["cos_within"][0]) < 0.2
    # an (n x c) matrix keeps its singular values under a row shuffle; i.i.d. noise is near-full rank
    assert np.allclose(rep["real"]["erank_cells"], rep["shuffled"]["erank_cells"], rtol=1e-6)
    assert all(v > 0.8 * c for row in rep["noise"]["erank_cells"] for v in row)
    assert all(v < 0.8 * c for row in rep["real"]["erank_cells"] for v in row)
    pas = np.array(rep["real"]["cka_passes"][0])
    assert pas.shape == (p, p) and np.allclose(np.diag(pas), 1.0)
    md = cl.geom_markdown("t", [0, 1, 2], rep)
    assert "within-slot cos" in md
    between = cl.cka_between(feats, feats)
    assert len(between) == p and abs(between[0][1][1] - 1.0) < 1e-9
    with pytest.raises(ValueError):
        cl.cka_between(feats, feats[:10])
