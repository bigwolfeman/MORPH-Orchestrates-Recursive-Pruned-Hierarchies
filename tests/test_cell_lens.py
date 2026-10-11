"""Contracts of `lab/divergence/cell_lens.py`.

What each test pins:
  * PATCH, "cell i alone" is the operation the model's own val oracle runs. The patch table
    for cell i equals (a) the forward's `fan_stream_ce_k{i}` and (b) a replay written HERE
    from the model's pieces (blank the other cells under `fan_mix: all`; force the router's
    winner under the latent-selected loop), on the same rows. (b) is the independent one:
    it fails if the write is changed to "all cells".
  * PATCH, the all-cells and zero-write readings equal the label-free eval forward's own
    per-token CE summed over the same bins (`plan_mode` normal / zero).
  * PROBE features: shapes, pad masking, targets equal `next_span_slots`, the pass-t
    feature of a depth-D run equals the exit of a depth-t run, the hook is removed.
  * PROBE training: a toy where the target is a (rotated) function of ONE cell beats the
    shuffled-cell and unigram controls, and the other cells sit at the controls.

CPU, fp32, the tests/test_tul_fan.py / test_tul_lxfan.py strict fixtures.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lab", "divergence"))
import cell_lens as cl  # noqa: E402
import morph.model.transformer as T  # noqa: E402
from morph.model.tul_spandec import next_span_slots  # noqa: E402
from test_tul_fan import _batch  # noqa: E402
from test_tul_lxfan import _Spy, _build, _fan_kw  # noqa: E402

M = 4
DEV = "cpu"


def _fan():
    m = _build(**_fan_kw(M, spandec=True, fan_all_wta_lambda=0.0))
    return m.eval()


def _lsel():
    m = _build(**_fan_kw(M, spandec=True, fan_all_wta_lambda=0.0, fan_loop_select="joint"))
    assert m.tul_fan_target_build()
    return m.eval()


FAMILIES = {"ungraded_fan": _fan, "latent_selected": _lsel}


def _bin_sums(m, xh, lab, layout) -> torch.Tensor:
    """`[B, S]` summed CE of bin s+1, written here (a loop over bags, no shared helper)."""
    w = m.embed.lm_weight()
    B, S = layout.slot_valid.shape
    out = torch.zeros(B, S)
    for b in range(B):
        ce = F.cross_entropy((xh[b] @ w.t()).float(), lab[b].clamp_min(0), reduction="none")
        keep = (lab[b] >= 0) & (~layout.slot_mask[b])
        for s in range(S):
            sel = keep & (layout.bag_id[b] == s + 1)
            out[b, s] = ce[sel].sum()
    return out


def _logit_bin_sums(logits, lab, layout) -> torch.Tensor:
    B, S = layout.slot_valid.shape
    out = torch.zeros(B, S)
    for b in range(B):
        ce = F.cross_entropy(logits[b].float(), lab[b].clamp_min(0), reduction="none")
        keep = (lab[b] >= 0) & (~layout.slot_mask[b])
        for s in range(S):
            out[b, s] = ce[keep & (layout.bag_id[b] == s + 1)].sum()
    return out


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("d", [1, 3])
def test_cell_i_alone_is_the_models_own_oracle_write(family, d):
    m = FAMILIES[family]()
    _ids, inp, lab, layout = _batch(M)
    tb = cl.patch_batch(m, inp, lab, layout, d, DEV)
    sm = cl.summarise_batch(tb)
    cl.check_anchor(tb, sm, "test")            # (a) the forward's own fan_stream_ce_k{i}
    assert tb["ok"].any() and sm["cell"].shape[1] == M
    assert float(sm["cnt"].sum()) > 0
    # (b) an independent replay of "cell i alone" from the model's pieces
    spy = _Spy(m, "_tul_fan_oracle")
    with torch.no_grad():
        m.tul_forward_ablated(inp, lab, layout, slot_depths=cl.forced_depths(layout, d, M))
    a = spy.calls[0][0]
    cells, base, x0, bigram, ids, labels, lay, L, keep, ckw, reset = (
        a[0], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11])
    del m.__dict__["_tul_fan_oracle"]
    assert tuple(cells.shape[:3]) == tuple(tb["ce"].shape[:2]) + (M,)
    tables = []
    with torch.no_grad():
        for i in range(M):
            if family == "ungraded_fan":
                blank = torch.zeros_like(cells)
                blank[:, :, i] = cells[:, :, i]
                src = blank
            else:
                forced = torch.full(cells.shape[:2], i, dtype=torch.long)
                src = m._fan_route_cells(cells, {"winner": forced, "scale": None})
            values, pos = m.tul.prefix_project(cells[:, :, i], lay, L, cells=src)
            x_i = T.scatter_positions(base, pos, values)
            xh = m._back_region(x_i, x0, bigram, ids, inject_keep=keep, attn_kwargs=ckw,
                                ret_reset_mask=reset)
            tables.append(_bin_sums(m, xh, labels, lay))
    ok = tb["ok"]
    # slot s <-> bag s+1, which `_bin_sums` puts at column s (the oracle's own `[:, 1:]`)
    for i in range(M):
        assert torch.allclose(tables[i][ok], tb["ce"][..., i][ok], atol=1e-4, rtol=1e-4), (
            family, d, i)


def test_replay_differs_from_the_all_cells_write():
    """The independent replay must be able to FAIL: writing every cell is a different CE."""
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    tb = cl.patch_batch(m, inp, lab, layout, 3, DEV)
    ok = tb["ok"]
    assert not torch.allclose(tb["ce"][..., 0][ok], tb["mixed"][ok], atol=1e-5)
    # cells genuinely differ from each other, so a collapsed table could not pass either
    assert not torch.allclose(tb["ce"][..., 0][ok], tb["ce"][..., 1][ok], atol=1e-6)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_all_cells_and_zero_readings_equal_the_label_free_forward(family):
    m = FAMILIES[family]()
    _ids, inp, lab, layout = _batch(M)
    d = 2
    tb = cl.patch_batch(m, inp, lab, layout, d, DEV)
    sd = cl.forced_depths(layout, d, M)
    ok = tb["ok"]
    w = m.embed.lm_weight()
    sid = int(m.cfg.tul.slot_id)
    with torch.no_grad():
        lg = m.tul_forward_ablated(inp, None, layout, slot_depths=sd)["logits"]
        lg0 = m.tul_forward_ablated(inp, None, layout, plan_mode="zero",
                                    slot_depths=sd)["logits"]
    # The label-free head masks the slot-id logit to -inf (generation head), the oracle's
    # CE does not. So: the label-free logits must equal the tapped coda output through the
    # tied head on every OTHER column; then, with the slot-id column restored from the same
    # readout, they must reproduce the table.
    for name, logits, xh, key in (("all", lg, tb["xh_all"], "mixed"),
                                  ("zero", lg0, tb["xh_zero"], "zero")):
        ref = xh @ w.t()
        keep_cols = [c for c in range(ref.shape[-1]) if c != sid]
        assert torch.allclose(logits[..., keep_cols], ref[..., keep_cols], atol=1e-4), name
        fixed = logits.clone()
        fixed[..., sid] = ref[..., sid]
        assert torch.allclose(_logit_bin_sums(fixed, lab, layout)[ok], tb[key][ok],
                              atol=1e-3, rtol=1e-4), name
    assert not torch.allclose(tb["zero"][ok], tb["mixed"][ok], atol=1e-4)


def test_run_patch_table_is_consistent():
    m = _fan()
    _ids, inp, lab, layout = _batch(M)
    res = cl.run_patch(m, [(inp, lab, layout)], [1, 3], DEV, log=lambda *_: None)
    for d, e in res.items():
        assert len(e["cells"]) == M and abs(sum(e["win_frac"]) - 1.0) < 1e-9
        assert e["oracle_best"] <= min(e["cells"]) + 1e-9
        assert e["oracle_best"] <= e["cells"][0] + 1e-9
    assert "cell0" in res[1]["ci_vs_all"]
    assert "| 1 |" in cl.patch_markdown("t", res)


# ── probe features ───────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_feature_shapes_masks_targets_and_pass_independence(family):
    m = FAMILIES[family]()
    ids, inp, lab, layout = _batch(M)
    f = cl.collect_features(m, [(inp, lab, layout)], [1, 2, 3], 4, DEV,
                            text_tokens=6, store_dtype=torch.float32)
    assert "_slot_pass_hook" not in m.__dict__                 # the tap is removed
    n, p_n, mm, c = f.feats.shape
    assert (p_n, mm, c) == (4, M, m.cfg.d_model) and f.passes == [0, 1, 2, 3]
    ref_ids, ref_valid = next_span_slots(inp, layout, 6)
    sel = ref_valid[:, :, 0]
    assert n == int(sel.sum()) == f.tgt.shape[0] == f.text.shape[0] == f.row.shape[0]
    assert torch.equal(f.text[f.text != cl.IGNORE], ref_ids[ref_valid])      # pads masked
    assert torch.equal(f.tgt, f.text[:, :4])
    assert bool(((f.text == cl.IGNORE) == ~ref_valid[sel]).all())
    assert bool((f.text[:, 0] != cl.IGNORE).all())
    assert torch.isfinite(f.feats).all()
    # a short span is padded with IGNORE (min span 4 in the fixture, offsets beyond it)
    assert not torch.allclose(f.feats[:, 0], f.feats[:, 1])     # entry != after pass 1
    assert not torch.allclose(f.feats[:, 1], f.feats[:, 2])
    # the cell state after pass t of a depth-3 run is the exit of a depth-t run
    for t in (1, 2):
        exit_cap = {}

        def hook(tt, h, h_new, lay, exit_cap=exit_cap, t=t):
            if tt + 1 == t:
                exit_cap["h"] = h_new
                raise cl._Done()
            return h_new
        m._slot_pass_hook = hook
        try:
            with torch.no_grad():
                m.tul_forward_ablated(inp, None, layout, slot_depths=cl.forced_depths(layout, t, M))
        except cl._Done:
            pass
        finally:
            m.__dict__.pop("_slot_pass_hook")
        B, S = layout.slot_valid.shape
        cr = cl._cell_readout(exit_cap["h"].reshape(B, S, M, *exit_cap["h"].shape[2:]))
        assert torch.allclose(cr[sel], f.feats[:, t], atol=1e-5, rtol=1e-5), t
    # the patch part still runs after an early-aborted forward (no stale state)
    tb = cl.patch_batch(m, inp, lab, layout, 2, DEV)
    cl.check_anchor(tb, cl.summarise_batch(tb), "after-collect")


# ── probe training ───────────────────────────────────────────────────────────────────────
def test_probe_on_a_toy_beats_the_shuffled_and_unigram_controls():
    torch.manual_seed(0)
    V, C, N, Mc = 40, 24, 900, 4
    emb = torch.randn(V, C)
    rot, _ = torch.linalg.qr(torch.randn(C, C))                  # identity init cannot solve it
    tok = torch.randint(0, V, (N,))
    feats = torch.randn(N, 1, Mc, C)
    feats[:, 0, 2] = emb[tok] @ rot + 0.1 * torch.randn(N, C)    # ONLY cell 2 carries the token
    row = torch.arange(N) // 6
    f = cl.Features(feats, tok.view(N, 1), tok.view(N, 1), row, [3])
    tr = cl.Features(feats[:600], f.tgt[:600], f.text[:600], row[:600], [3])
    va = cl.Features(feats[600:], f.tgt[600:], f.text[600:], row[600:], [3])
    probes = cl.train_probes(tr, emb, steps=400, batch=128, lr=3e-3, wd=0.0, device=DEV,
                             log=lambda *_: None)
    res = cl.eval_probes(probes, va, cl.unigram_stats(tr.tgt, V), DEV)
    r = res["probe"]["pass3_off1"]
    assert r["cell2"]["top1"] > 0.8, r["cell2"]
    assert r["cell2"]["ce"] < 2.2
    chance = 1.0 / V
    for nm in ("cell0", "cell1", "cell3", "shuffled"):
        assert r[nm]["top1"] < 0.2, (nm, r[nm])
        assert r[nm]["ce"] > r["cell2"]["ce"] + 1.0
    assert r["cell2"]["top1"] > 10 * chance
    assert r["cell2"]["ce"] < res["unigram"]["off1"]["ce"] - 1.0
    # the mean-of-cells input dilutes the one informative cell
    assert r["mean"]["ce"] > r["cell2"]["ce"]
    assert res["same_row_frac"] < 0.1
    exs = cl.make_examples(probes, va, lambda ids: " ".join(str(i) for i in ids), DEV, n=3)
    assert len(exs) == 3 and len(exs[0]["cells"]) == Mc and len(exs[0]["cells"][2]["off1"]) == 5
