"""`lab/divergence/exploration_ledger.py`: the interventions do what their names say, the
identity patch is the shipped forward bit for bit, and the statistics match hand values.

CPU, fp32, the tests/test_tul_fan_lsel.py / tests/test_tul_fan_router.py tiny fixtures.
What each test pins:
  * identity patches (latent-selected loop, both routers, write-all fan) == the shipped
    labelled forward, and the labelled capture == the label-free forward's CE;
  * the latent-selected loop's overrides, read from `_lsel_capture`: random_search follows
    the generator's draws at every pass, fixed_lineage follows 0, loser_t{t} follows a
    non-winner at pass t ONLY, random_exit / cell_i change the LAST pass only, no_reset
    leaves every pass's carrier untouched (and the shipped forward does reset);
  * a router's exit override reaches the coda (forced-to-own-winner is the identity, a
    forced cell changes the CE); the write-all fan's write_cell blanks every other cell;
  * every patch is removed afterwards, also when the block raises;
  * the row bootstrap, the span quartiles and the per-span oracle against hand values;
  * ledger_arm end to end on the tiny latent-selected loop.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lab", "divergence"))

import exploration_ledger as XL  # noqa: E402
import morph.model.transformer as T  # noqa: E402
from test_tul_fan import _batch  # noqa: E402
from test_tul_fan_lsel import _cap, _lsel, _nowta_kw  # noqa: E402
from test_tul_lx_credit import M  # noqa: E402
from test_tul_lxfan import _build  # noqa: E402


def _router(kind: str):
    m = _build(**_nowta_kw(fan_route=kind))
    m.tul_fan_target_build()
    return m.eval()


def _nowta():
    return _build(**_nowta_kw()).eval()


def _ce(m, **patch):
    _ids, inp, lab, layout = _batch(M)
    with XL.ledger_patch(m, **patch):
        ce, scal = XL.ce_map(m, inp, lab, layout, "cpu")
    return ce, scal, (inp, lab, layout)


def _gen(seed=7):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── 1. identity == shipped ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("which", ["lsel", "reader", "latent", "nowta"])
def test_identity_patch_is_the_shipped_forward_bit_for_bit(which):
    if which == "lsel":
        m, patch = _lsel().eval(), {"select_fn": lambda t, w: w}
    elif which == "nowta":
        m, patch = _nowta(), {}
    else:
        m, patch = _router(which), {"exit_fn": lambda r, tgt: r["winner"]}
    _ids, inp, lab, layout = _batch(M)
    ce0, _s = XL.ce_map(m, inp, lab, layout, "cpu")                 # no patch at all
    ce1, scal, _b = _ce(m, **patch)
    assert torch.equal(ce0, ce1)
    tokpos = (~layout.slot_mask) & (lab >= 0)
    assert abs(float(ce1[tokpos].mean()) - scal["ce_tokens"]) < 1e-5
    with torch.no_grad():
        lg = m.tul_forward_ablated(inp, None, layout, plan_mode="normal")["logits"].float()
    lab0 = lab.clone()
    lab0[lab0 < 0] = 0
    ce_lf = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), lab0.reshape(-1),
                                              reduction="none").reshape(lab.shape)
    assert torch.allclose(ce_lf[tokpos], ce1[tokpos], atol=1e-5)


# ── 2. the latent-selected loop's overrides, read from the capture ───────────────────


def _lsel_run(m, **patch):
    cap = _cap(m)
    _ids, inp, lab, layout = _batch(M)
    with XL.ledger_patch(m, **patch):
        XL.ce_map(m, inp, lab, layout, "cpu")
    m._lsel_capture = None
    return cap, layout.slot_valid


def test_random_search_follows_the_generators_draws_every_pass():
    m = _lsel().eval()
    ship, valid = _lsel_run(m)
    g = _gen()
    cap, _v = _lsel_run(m, select_fn=lambda t, w: XL._rand_cells(w.shape, M, g, w.device))
    g2 = _gen()
    assert len(cap) == len(ship) >= 2
    moved = False
    for c, s in zip(cap, ship):
        exp = torch.randint(0, M, tuple(c["follow"].shape), generator=g2)
        assert torch.equal(c["follow"], exp)
        moved |= not torch.equal(c["follow"][valid], s["follow"][valid])
    assert moved


def test_fixed_lineage_follows_cell_zero():
    m = _lsel().eval()
    cap, _v = _lsel_run(m, select_fn=lambda t, w: torch.zeros_like(w))
    assert all(int(c["follow"].abs().sum()) == 0 for c in cap)


def test_loser_at_t_changes_pass_t_only_and_never_picks_the_winner():
    m = _lsel().eval()
    ship, valid = _lsel_run(m)
    g = _gen()
    for tt in range(len(ship) - 1):
        seen = []
        fn = XL.reading_plan("lsel", M, len(ship))
        make = dict(fn)[f"loser_t{tt}"]["make"]
        inner = make(g)["select_fn"]

        def rec(t, w, inner=inner):
            out = inner(t, w)
            seen.append((t, w.clone(), out.clone()))
            return out
        cap, _v = _lsel_run(m, select_fn=rec)
        assert [s[0] for s in seen] == list(range(len(ship)))
        for t, w_in, w_out in seen:
            if t == tt:
                assert bool((w_out != w_in).all())          # a non-winner on every slot
            else:
                assert torch.equal(w_out, w_in)             # the router's own pick
        # pass tt's input carrier is the shipped one, so its winner is the shipped pick
        if tt == 0:
            assert torch.equal(seen[tt][1], ship[tt]["follow"])
        assert torch.equal(cap[tt]["follow"], seen[tt][2])


@pytest.mark.parametrize("which", ["random_exit", "cell_2"])
def test_exit_overrides_change_the_last_pass_only(which):
    m = _lsel().eval()
    ship, valid = _lsel_run(m)
    last = len(ship) - 1
    g = _gen()
    fn = ((lambda t, w: XL._rand_cells(w.shape, M, g, w.device) if t == last else w)
          if which == "random_exit" else
          (lambda t, w: torch.full_like(w, 2) if t == last else w))
    cap, _v = _lsel_run(m, select_fn=fn)
    for c, s in zip(cap[:-1], ship[:-1]):
        assert torch.equal(c["follow"], s["follow"])
        assert torch.equal(c["post"], s["post"])
    if which == "cell_2":
        assert bool((cap[-1]["follow"] == 2).all())
    else:
        assert not torch.equal(cap[-1]["follow"][valid], ship[-1]["follow"][valid])


def test_no_reset_leaves_every_carrier_untouched_and_is_restored():
    m = _lsel().eval()
    ship, valid = _lsel_run(m)
    assert any(not torch.equal(c["pre"], c["post"]) for c in ship[:-1])   # shipped resets
    orig = T.reset_to_winner
    cap, _v = _lsel_run(m, no_reset=True)
    assert all(torch.equal(c["pre"], c["post"]) for c in cap)
    assert T.reset_to_winner is orig


def test_every_patch_is_removed_even_when_the_block_raises():
    m = _lsel().eval()
    r = _router("reader")
    w = _nowta()
    orig_reset = T.reset_to_winner
    for model, kw in ((m, {"select_fn": lambda t, x: x, "no_reset": True}),
                      (r, {"exit_fn": lambda rr, tgt: rr["winner"]}),
                      (w, {"write_cell": 1})):
        with pytest.raises(RuntimeError, match="boom"):
            with XL.ledger_patch(model, **kw):
                raise RuntimeError("boom")
        for obj in (model, model.tul, getattr(model, "tul_fan_lsel_router", None)):
            if obj is not None:
                assert not {"_tul_fan_oracle", "_tul_fan_route", "prefix_project",
                            "select"} & set(obj.__dict__)
    assert T.reset_to_winner is orig_reset


# ── 3. routers and the write-all fan ─────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["reader", "latent"])
def test_router_exit_override_reaches_the_coda(kind):
    m = _router(kind)
    ce_ship, _s, _b = _ce(m)
    forced = [_ce(m, exit_fn=(lambda i: lambda r, tgt: torch.full_like(r["winner"], i))(i))[0]
              for i in range(M)]
    assert any(not torch.equal(f, ce_ship) for f in forced)
    assert not all(torch.equal(forced[0], f) for f in forced[1:])


def test_latent_teacher_reading_follows_the_teacher_on_target_slots():
    m = _router("latent")
    seen = {}
    orig = m._tul_fan_route

    def spy(cells, xn, layout, tgt, stats):
        route = orig(cells, xn, layout, tgt, stats)
        seen["route"], seen["tgt"] = route, tgt
        return route
    m._tul_fan_route = spy                      # inner: under the ledger's own wrapper
    plan = dict(XL.reading_plan("latent", M, 0))
    kw = plan["teacher"]["make"](_gen())
    got = {}
    fn = kw["exit_fn"]
    kw["exit_fn"] = lambda r, tgt: got.setdefault("w", fn(r, tgt))
    _ce(m, **kw)
    m.__dict__.pop("_tul_fan_route")
    r, tgt = seen["route"], seen["tgt"]
    exp = torch.where(tgt["ok"], r["teacher"], r["winner"])
    assert torch.equal(got["w"], exp)


def test_write_cell_blanks_every_other_cell():
    m = _nowta()
    seen = {}
    orig = m.tul.prefix_project

    def spy(h, layout, L, cells=None):
        seen.setdefault("cells", []).append(cells.clone())
        return orig(h, layout, L, cells=cells)
    m.tul.prefix_project = spy
    _ce(m)
    full = seen["cells"][-1]
    _ce(m, write_cell=1)
    blank = seen["cells"][-1]
    del m.tul.__dict__["prefix_project"]
    assert torch.equal(blank[:, :, 1], full[:, :, 1])
    for i in (0, 2, 3):
        assert int(blank[:, :, i].abs().sum()) == 0
        assert float(full[:, :, i].abs().sum()) > 0


# ── 4. statistics against hand values ────────────────────────────────────────────────


def test_delta_ci_constant_shift_and_noisy_mean():
    rng = np.random.default_rng(0)
    n_rows, per = 40, 25
    row = np.repeat(np.arange(n_rows), per)
    base = rng.normal(3.0, 1.0, row.size)
    r = XL.delta_ci(base + 0.1, base, row, n_rows)
    assert abs(r["point"] - 0.1) < 1e-12 and abs(r["lo"] - 0.1) < 1e-9
    assert abs(r["hi"] - 0.1) < 1e-9
    noisy = base + 0.05 + rng.normal(0, 0.2, row.size)
    r2 = XL.delta_ci(noisy, base, row, n_rows)
    exp = float((noisy - base).mean())
    assert abs(r2["point"] - exp) < 1e-12
    assert r2["lo"] < 0.05 < r2["hi"] and r2["lo"] < exp < r2["hi"]
    mask = row < 10
    r3 = XL.delta_ci(noisy, base, row, n_rows, mask=mask)
    assert r3["n_units"] == 10 and r3["n_tokens"] == 10 * per
    assert abs(r3["point"] - float((noisy - base)[mask].mean())) < 1e-12


def test_span_quartiles_by_hand():
    # 8 spans of 2 tokens; span means 0..7 -> quartiles 0,0,1,1,2,2,3,3; span 8 unmatched
    key = np.repeat(np.arange(9), 2)
    plain = np.repeat(np.arange(9, dtype=float), 2)
    plain[16:] = np.nan
    q = XL.span_quartiles(plain, key)
    assert q.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, -1, -1]


def test_oracle_tokens_picks_the_lowest_span_sum_per_span():
    key = np.array([5, 5, 9, 9, 9])
    cells = np.array([[1.0, 1.0, 0.1, 0.1, 5.0],      # span 5 sum 2.0, span 9 sum 5.2
                      [0.5, 2.0, 1.0, 1.0, 1.0]])     # span 5 sum 2.5, span 9 sum 3.0
    tok, pick = XL.oracle_tokens(cells, key)
    assert pick.tolist() == [0, 0, 1, 1, 1]
    assert tok.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]


# ── 5. end to end on the tiny latent-selected loop ───────────────────────────────────


def test_ledger_arm_end_to_end_on_the_tiny_latent_selected_loop():
    m = _lsel().eval()
    _ids, inp, lab, layout = _batch(M)
    idx = torch.full(inp.shape, -1, dtype=torch.long)
    tok = ~layout.slot_mask
    idx[tok] = torch.arange(int(tok.sum()))
    block, toks = XL.ledger_arm(m, [(inp, lab, layout, idx)], "cpu", None, n_boot=50,
                                log=lambda s: None)
    passes = block["passes"]
    assert passes >= 2
    names = set(block["readings"])
    exp = {"random_exit", "random_search", "teacher", "no_reset", "fixed_lineage",
           "single_cell", "oracle", *(f"cell_{i}" for i in range(M)),
           *(f"loser_t{t}" for t in range(passes - 1))}
    assert names == exp
    cells = np.stack([toks[f"cell_{i}"] for i in range(M)])
    assert toks["oracle"].mean() <= cells.mean(axis=1).min() + 1e-12
    assert block["not_applicable"] == {}
    assert block["self_checks"]["ce_tokens_max_abs_dev"] < 1e-5
    assert block["self_checks"]["label_free_max_abs_dev_batch0"] < 1e-5
    # the shipped reading is the model's own forward
    ce0, _s = XL.ce_map(m, inp, lab, layout, "cpu")
    assert np.array_equal(toks["shipped"], ce0[(~layout.slot_mask) & (lab >= 0)].numpy()
                          .astype(np.float64))


def test_plain_join_is_shift_zero_and_splits_by_quartile():
    """The plain join pairs by the INPUT token's stream index (shift 0, `pack_rows`'
    convention for both cuts), whatever the correlation says; every reading then carries
    a Q1..Q4 split whose token counts add up to the matched tokens."""
    m = _router("reader")
    _ids, inp, lab, layout = _batch(M)
    idx = torch.full(inp.shape, -1, dtype=torch.long)
    tok = ~layout.slot_mask
    idx[tok] = torch.arange(int(tok.sum())) + 100
    scored = (tok & (lab >= 0))
    pidx = idx[scored].numpy().astype(np.int64)
    pce = np.random.default_rng(0).uniform(1, 8, pidx.size)
    block, toks = XL.ledger_arm(m, [(inp, lab, layout, idx)], "cpu", (pidx, pce),
                                n_boot=50, log=lambda s: None)
    assert block["plain"]["used"] and block["plain"]["shift"] == 0
    assert block["plain"]["matched_tokens"] == pidx.size
    q = block["readings"]["random_exit"]["by_plain_quartile"]
    assert sum(v["n_tokens"] for v in q.values()) == pidx.size
    assert set(np.unique(toks["plain_quartile"])) <= {0, 1, 2, 3}
