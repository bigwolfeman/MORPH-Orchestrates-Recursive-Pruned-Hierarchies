"""CPU tests of the lean recipe's keys: `cell_count` (+ `cell_count_eval`), `spandec_reads_cells`, and
`configs/arm/lean.yaml` (lxtul/model.py, lxtul/data.py).

What is pinned:
  1. the cell-count law (clamped Poisson, counter-based) and the loader's re-layout: tokens, labels and
     spans are the reference row's, only the cell positions change;
  2. the prefix rule: a K-role model run at M < K cells equals a model BUILT with M cells whose role
     tables are the first M rows (loss and every gradient, float64);
  3. M = K under cell_count='poisson' is the fixed-M forward, exactly;
  4. a smaller M shrinks the computed cells (the loop's and the coda's input widths), it does not mask;
  5. the keys compose with slot_compact (gather == full), ce_rows, spandec_packed, static_depth, at M = 1..K;
  6. the set reader: zero at init, invariant to permuting the cells (module and decoder level, packed and
     unpacked), and a sabotage that adds a cell-position term FAILS the invariance check;
  7. lean.yaml composes, every key is a real LXTULConfig field, and it builds and runs forward + backward;
  8. the refusals.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_lean.py
"""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import lxtul.model as lm
import parcae_lm
from lxtul.data import cell_law, relay_cells
from lxtul.model import LXTULConfig, LXTULParcae, SetCross
from lxtul.tests.test_compact import (EOS, J, K, S, SEQ, SLOT, V, _assert_same, _batch, _build,  # noqa: F401
                                      _randomize, _run, _twin, cpu_reference_kernels)

POIS2 = dict(select="none", cell_count="poisson", cell_count_mean=2.0)


def _spec():
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    return TulLayoutSpec(seq_len=SEQ, prefix_k=K, max_slots=S, slot_id=SLOT), BoundaryRule(
        lut, min_span=4, span_cap=J, eos_id=EOS)


def _at(batch, m):
    """The reference-width batch re-laid out to m cells per slot (MORPH's relayout_batch)."""
    from morph.model.tul_layout import relayout_batch
    spec, _ = _spec()
    x, y, lay = batch
    return relayout_batch(x, y, lay, m, spec, J)


@pytest.fixture(scope="module")
def data():
    x, y, lay = _batch(1)
    nv = lay.slot_valid.sum(1)
    assert nv.min() < S and nv.min() != nv.max()
    return x, y, lay


# ------------------------------------------------------------------ 1. the law and the loader
def test_law_is_the_clamped_poisson_and_counter_based():
    tul = LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, **POIS2)
    law = cell_law(tul, seed=7)
    e = math.exp(-2.0)
    want = [e * 3.0, e * 2.0, e * 4.0 / 3.0]
    want.append(1.0 - sum(want))
    assert np.allclose(law.probs, want, atol=1e-12) and abs(sum(law.probs) - 1.0) < 1e-12
    draws = [law.draw(i) for i in range(6000)]
    ms = np.array([m for m, _ in draws])
    freq = np.bincount(ms, minlength=K + 1)[1:] / len(ms)
    for p, f in zip(want, freq):                          # 4 sigma of a binomial frequency
        assert abs(f - p) < 4 * math.sqrt(p * (1 - p) / len(ms)), (want, freq.tolist())
    assert all(r.tolist() == list(range(m)) for m, r in draws), "the active cells must be the role PREFIX"
    assert [law.draw(i)[0] for i in range(50)] == [law.draw(i)[0] for i in range(50)]      # replayable
    assert [cell_law(tul, 8).draw(i)[0] for i in range(50)] != ms[:50].tolist()            # seed matters
    assert cell_law(LXTULConfig(slot_id=SLOT, max_slots=S, cells=K), 7) is None            # fixed: no law
    capped = cell_law(dataclasses.replace(tul, cell_count_max=2), 7)
    assert capped.probs[2] == capped.probs[3] == 0.0 and abs(sum(capped.probs) - 1) < 1e-12


def test_relay_cells_keeps_tokens_labels_and_spans():
    """The loader's wrapper draws M for batch i from the law and re-lays the batch to it."""
    spec, rule = _spec()
    rt = SimpleNamespace(data_cfg=SimpleNamespace(spec_for=lambda seq: spec, rule=rule))
    tul = LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, **POIS2)
    law = cell_law(tul, seed=3)
    src = [_batch(i) for i in range(10)]
    seen = set()
    for i, (x, y, lay) in enumerate(relay_cells(iter(src), rt, SEQ, law=law)):
        m = law.draw(i)[0]
        seen.add(m)
        x0, y0, l0 = src[i]
        assert lay.prefix_k == m
        for b in range(x.shape[0]):
            assert x[b][~lay.slot_mask[b]].tolist() == x0[b][~l0.slot_mask[b]].tolist(), "tokens moved"
            assert y[b][~lay.slot_mask[b]].tolist() == y0[b][~l0.slot_mask[b]].tolist(), "labels moved"
        assert (lay.slot_valid == l0.slot_valid).all()
        assert (lay.slot_mask.sum(1) >= m * lay.slot_valid.sum(1)).all()
        if m < K:
            assert x.shape[1] < x0.shape[1], "fewer cells must give a shorter row"
    assert len(seen) >= 3, f"the 10 draws covered only M in {seen}"
    # eval: a fixed M
    fixed = list(relay_cells(iter(src[:2]), rt, SEQ, eval_cells=2))
    assert all(b[2].prefix_k == 2 for b in fixed)


# ------------------------------------------------------------------ 2-4. the model at M < K
def _small_twin(base, m):
    """A model BUILT with m cells whose role tables are the first m rows of `base`'s."""
    # a one-cell model has no epivol (the model refuses repel_lambda > 0 at cells=1; M = 1 skips the term anyway)
    twin = LXTULParcae(base.config, dataclasses.replace(base.tul, cells=m, cell_count="fixed",
                                                         cell_count_mean=3.5,
                                                         repel_lambda=base.tul.repel_lambda if m > 1 else 0.0)
                       ).double().train()
    sd, have = {}, twin.state_dict()
    for k, v in base.state_dict().items():
        if k not in have and not k.startswith("W_prefix."):
            continue                                       # e.g. the epivol reservoir of a one-cell model
        if k in ("reg_q_embed", "reg_p_cell_embed"):
            v = v[:m]
        elif k.startswith("W_prefix.") and int(k.split(".")[1]) >= m:
            continue
        sd[k] = v
    twin.load_state_dict(sd)
    return twin


@pytest.fixture(scope="module", params=["plain", "reads_cells"])
def lean_base(request):
    keys = dict(POIS2)
    if request.param == "reads_cells":
        keys["spandec_reads_cells"] = True
    return _build(**keys)


@pytest.mark.parametrize("m", [1, 2, 3])
def test_prefix_rule_equals_a_model_built_with_m_cells(lean_base, data, m):
    if m == 1 and lean_base.tul.spandec_reads_cells:
        pytest.skip("a one-cell model cannot be built with spandec_reads_cells (the mean is the cell)")
    x, y, lay = _at(data, m)
    a, ga = _run(lean_base, x, y, lay)
    twin = _small_twin(lean_base, m)
    b, gb = _run(twin, x, y, lay)
    assert lay.prefix_k == m and torch.isfinite(a["loss"])
    assert abs(float(a["loss"]) - float(b["loss"])) <= 1e-9 * max(1.0, abs(float(a["loss"])))
    for n, g in gb.items():                               # every gradient of the small model
        full = ga[n]
        if n in ("reg_q_embed", "reg_p_cell_embed"):
            assert full[m:].abs().max() == 0, f"{n}: an unused role got a gradient"
            full = full[:m]
        scale = max(float(g.abs().max()), 1e-12)
        assert float((full - g).abs().max()) / scale <= 1e-8, n
    for i in range(m, K):
        assert f"W_prefix.{i}" not in ga or ga[f"W_prefix.{i}"].abs().max() == 0, "an unused role's write got a gradient"


def test_full_width_poisson_is_the_fixed_forward_exactly(data):
    base = _build(select="none")
    on = _twin(base, cell_count="poisson", cell_count_mean=2.0)
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    b, gb = _run(on, x, y, lay)
    assert float(a["loss"]) == float(b["loss"])
    assert all(torch.equal(ga[n], gb[n]) for n in ga)


@pytest.mark.parametrize("compact", ["off", "gather"])
def test_a_smaller_m_shrinks_the_computed_cells(data, compact):
    keys = dict(POIS2, **(dict(slot_depth_stratified=True, slot_compact="gather") if compact == "gather" else {}))
    m = _build(**keys)
    shapes = {}
    # norm_1 sees the layer input in the plain loop and in the compacted one (which skips block.forward)
    m.transformer.core_block[0].norm_1.register_forward_pre_hook(
        lambda mod, a: shapes.setdefault("core", []).append(a[0].shape[1]))
    m.transformer.coda[0].register_forward_pre_hook(lambda mod, a: shapes.__setitem__("coda", a[0].shape[1]))
    widths = {}
    for M in (1, 2, K):
        x, y, lay = _at(data, M)
        shapes.clear()
        _run(m, x, y, lay)
        widths[M] = (max(shapes["core"]), shapes["coda"])
        if compact == "gather":
            assert min(shapes["core"]) < max(shapes["core"]), "gather computed every cell on every pass"
        assert widths[M][1] == x.shape[1]
    assert all(widths[M][0] == S * M for M in (1, 2, K)), widths      # the loop runs on S*M cells
    assert widths[1][1] < widths[2][1] < widths[K][1], "the coda row must be shorter at fewer cells"


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_lean_keys_compose_at_every_cell_count(data, m):
    """The lean fast keys at each M: gather == full (the compaction's contract) and the packed CE /
    packed decoder / static depth are the same function as unkeyed (float64)."""
    st = dict(POIS2, spandec_reads_cells=True, slot_depth_stratified=True)
    full = _build(**st, slot_compact="full")
    x, y, lay = _at(data, m)
    a, ga = _run(full, x, y, lay)
    gat = _twin(full, slot_compact="gather")
    b, gb = _run(gat, x, y, lay)
    fast = _twin(full, slot_compact="gather", static_depth=True, cot_clip_fn=True, ce_rows=True, spandec_packed=True)
    c, gc = _run(fast, x, y, lay)
    if m == 1:
        # a softmax over ONE cell is constant: the query path's true gradient is 0 and what is left is
        # rounding noise, which a relative comparison would read as a mismatch
        for g in (ga, gb, gc):
            for n in [n for n in g if n.startswith("sd_cross.") and (".q." in n or "norm_q" in n)]:
                del g[n]
    _assert_same(a, ga, b, gb)
    _assert_same(a, ga, c, gc)


def test_epivol_is_skipped_at_one_cell_only(data):
    m = _build(**POIS2)
    for M, want in ((1, False), (2, True)):
        x, y, lay = _at(data, M)
        o, _ = _run(m, x, y, lay)
        on = LXTULParcae(m.config, dataclasses.replace(m.tul, repel_lambda=0.0)).double().train()
        on.load_state_dict(m.state_dict(), strict=False)      # same weights; no epivol reservoir
        o2, _ = _run(on, x, y, lay)
        differs = abs(float(o["loss"]) - float(o2["loss"])) > 1e-9
        assert differs == want, f"M={M}: epivol term present={differs}, expected {want}"


# ------------------------------------------------------------------ 6. the set reader
def _cross(d=64, heads=2, seed=0):
    cfg = SimpleNamespace(num_attention_heads=heads, norm_eps=1e-6, Norm=lambda dim, eps: torch.nn.RMSNorm(dim, eps=eps))
    torch.manual_seed(seed)
    c = SetCross(cfg, d).double()
    with torch.no_grad():
        c.proj.weight.normal_(0, 0.1)                     # un-zero it: the test needs a live block
    return c


def test_set_cross_is_zero_at_init():
    cfg = SimpleNamespace(num_attention_heads=2, norm_eps=1e-6, Norm=lambda dim, eps: torch.nn.RMSNorm(dim, eps=eps))
    c = SetCross(cfg, 64).double()
    out = c(torch.randn(3, 5, 64).double(), torch.randn(3, 4, 64).double())
    assert out.abs().max() == 0


def test_set_cross_is_invariant_to_the_cell_order_and_packed_matches_dense():
    c = _cross()
    g = torch.Generator().manual_seed(1)
    x, mem = torch.randn(6, 7, 64, generator=g).double(), torch.randn(6, 4, 64, generator=g).double()
    base = c(x, mem)
    assert base.abs().max() > 1e-3, "vacuous: the block outputs ~0"
    for perm in ([3, 2, 1, 0], [1, 3, 0, 2]):
        assert (c(x, mem[:, perm]) - base).abs().max() < 1e-12
    # the packed form is the same attention: B=2 rows, S=3 slots, positions pick a slot each
    Bq, Sq, P = 2, 3, 9
    xp = torch.randn(Bq, P, 64, generator=g).double()
    memp = torch.randn(Bq, Sq, 4, 64, generator=g).double()
    slot = torch.randint(0, Sq, (Bq, P), generator=g)
    got = c.forward_packed(xp, memp, slot)
    for b in range(Bq):
        for p in range(P):
            want = c(xp[b, p][None, None], memp[b, slot[b, p]][None])[0, 0]
            assert (got[b, p] - want).abs().max() < 1e-10
    assert (c.forward_packed(xp, memp[:, :, [2, 0, 3, 1]], slot) - got).abs().max() < 1e-12


def _spandec_args(model, data):
    """Run one forward and capture `_spandec`'s arguments so it can be called again on altered cells."""
    x, y, lay = data
    cap = {}
    real = model._spandec

    def spy(hv, *a, **k):
        cap["args"] = (hv.detach().clone(), *a)
        cap["kw"] = k
        return real(hv, *a, **k)
    model._spandec = spy
    try:
        _run(model, x, y, lay)
    finally:
        del model._spandec
    return cap["args"], cap["kw"]


@pytest.mark.parametrize("packed", [False, True])
def test_decoder_loss_is_invariant_to_permuting_the_cells_and_sabotage_breaks_it(data, packed, monkeypatch):
    m = _build(spandec_reads_cells=True, spandec_packed=packed, ce_rows=packed)
    with torch.no_grad():
        for blk in m.sd_cross:
            blk.proj.weight.normal_(0, 0.1)
    args, kw = _spandec_args(m, data)
    hv = args[0]
    perm = [2, 0, 3, 1]
    with torch.no_grad():
        ref = m._spandec(*args, **kw)
        again = m._spandec(hv[:, :, perm], *args[1:], **kw)
    assert torch.isfinite(ref)
    assert abs(float(ref) - float(again)) < 1e-10 * max(1.0, abs(float(ref))), "set reader depends on cell order"
    # the reader is live: without it (zero projections) the decoder loss moves
    with torch.no_grad():
        saved = [blk.proj.weight.clone() for blk in m.sd_cross]
        for blk in m.sd_cross:
            blk.proj.weight.zero_()
        off = m._spandec(*args, **kw)
        for blk, w in zip(m.sd_cross, saved):
            blk.proj.weight.copy_(w)
    assert abs(float(ref) - float(off)) > 1e-6, "vacuous: the cross block does not change the decoder"
    # sabotage: a cell-position term makes the decoder see the order. The same check must now fail.
    pos = torch.randn(K, m.config.n_embd, generator=torch.Generator().manual_seed(9)).double()
    real_f, real_p = SetCross.forward, SetCross.forward_packed
    monkeypatch.setattr(SetCross, "forward", lambda self, x, mem: real_f(self, x, mem + pos[: mem.shape[1]]))
    monkeypatch.setattr(SetCross, "forward_packed",
                        lambda self, x, mem, slot: real_p(self, x, mem + pos[: mem.shape[2]], slot))
    with torch.no_grad():
        a = m._spandec(*args, **kw)
        b = m._spandec(hv[:, :, perm], *args[1:], **kw)
    assert abs(float(a) - float(b)) > 1e-6, "the sabotage went undetected: the invariance test is vacuous"


def test_reads_cells_starts_as_the_mean_only_decoder_and_packed_equals_unpacked(data):
    x, y, lay = data
    base = _build(select="none")
    on = LXTULParcae(base.config, dataclasses.replace(base.tul, spandec_reads_cells=True)).double().train()
    missing, unexpected = on.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("sd_cross.") for k in missing), missing
    a, ga = _run(base, x, y, lay)
    b, gb = _run(on, x, y, lay)
    assert float(a["loss"]) == float(b["loss"]), "the zero-init cross block moved the loss at step 0"
    assert on.sd_cross[0].proj.weight.grad.abs().max() > 0, "the cross block gets no gradient"
    with torch.no_grad():
        for blk in on.sd_cross:
            blk.proj.weight.normal_(0, 0.1)
    c, gc = _run(on, x, y, lay)
    assert abs(float(c["loss"]) - float(b["loss"])) > 1e-6
    packed = _twin(on, spandec_packed=True, ce_rows=True)
    packed.load_state_dict(on.state_dict())
    d, gd = _run(packed, x, y, lay)
    _assert_same(c, gc, d, gd)


# ------------------------------------------------------------------ 7. lean.yaml
def _lean_cfg():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from pathlib import Path
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    with initialize_config_dir(str(Path(lm.__file__).parent / "configs"), version_base=None):
        cfg = compose("train", overrides=["arm=lean", "run.wandb_mode=disabled"])
    return OmegaConf.to_container(cfg.arm.lxtul, resolve=True), cfg


def test_lean_yaml_keys_are_real_and_the_arm_runs(data):
    keys, cfg = _lean_cfg()
    assert str(cfg.arm.kind) == "lxtul" and str(cfg.arm.name) == "lean"
    fields = {f.name for f in dataclasses.fields(LXTULConfig)}
    assert set(keys) <= fields, set(keys) - fields
    assert keys["select"] == "none" and keys["cell_count"] == "poisson" and keys["cell_count_mean"] == 2.0
    assert keys["spandec_reads_cells"] and keys["spandec_layers"] == 1
    for k in ("slot_depth_stratified", "static_depth", "cot_clip_fn", "ce_rows", "spandec_packed"):
        assert keys[k] is True, k
    assert keys["slot_compact"] == "gather" and keys["pointer_heads"] == 4
    torch.manual_seed(0)
    mc = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    m = LXTULParcae(mc, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=J, **keys))
    m.init_slot_seed()
    m = _randomize(m).double().train()
    assert len(m.sd_blocks) == 1 and len(m.sd_cross) == 1
    for M in (1, 2, 3, K):
        x, y, lay = _at(data, M)
        o, g = _run(m, x, y, lay)
        assert torch.isfinite(o["loss"]) and all(torch.isfinite(v).all() for v in g.values()), M
        assert float(o["metrics"]["cells"]) == M
        assert g["sd_cross.0.q.weight"].shape[0] == 64
    m.eval()                                              # the eval path at the reference width and at M = 2
    with torch.no_grad():
        for M in (K, 2):
            x, y, lay = _at(data, M)
            assert torch.isfinite(m(x, y, lay)["loss"])


# ------------------------------------------------------------------ 8. refusals
def _raises(match, cells=K, **keys):
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    with pytest.raises(ValueError, match=match):
        LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=cells, span_cap=J, spandec_layers=1, **keys))


def test_refusals():
    _raises("silently ignored", cell_count_mean=2.0)
    _raises("silently ignored", cell_count_eval=2)
    _raises("select='none'", cell_count="poisson")                      # select defaults to latent
    _raises("must be 'fixed' or 'poisson'", cell_count="categorical")
    _raises("min <= max <= cells", select="none", cell_count="poisson", cell_count_max=5)
    _raises("outside", select="none", cell_count="poisson", cell_count_eval=5)
    _raises("cells > 1", cells=1, spandec_reads_cells=True)
    _raises("span decoder", spandec_reads_cells=True, spandec_weight=0.0)


def test_a_layout_wider_than_the_role_table_is_refused(data):
    m = _build(**POIS2)
    x, y, lay = data
    wide = dataclasses.replace(lay, prefix_k=K + 1)
    with pytest.raises(ValueError, match="cells per slot"):
        m(x, y, wide)


# ------------------------------------------------------------------ geometry at fewer cells (CPU twin of test_geometry's GPU tests)
@pytest.mark.parametrize("m", [1, 2, 3])
def test_fewer_cells_keep_causality_and_the_loop_as_the_only_channel(m):
    model = _build(spandec_reads_cells=True, **POIS2).eval()

    def hid(x, lay):
        with torch.no_grad():
            return model(x, None, lay)["hidden"]
    x, _, lay = _at(_batch(1), m)
    base = hid(x, lay)
    for r in range(x.shape[0]):
        n = int(lay.slot_valid[r].sum())
        cut = int(lay.slot_index[r, n // 2]) + m
        x2 = x.clone()
        fut = (~lay.slot_mask[r]).nonzero().squeeze(1)
        fut = fut[fut >= cut]
        x2[r, fut] = (x2[r, fut] + 1 - 20) % (V - 20) + 20
        d = (hid(x2, lay)[r] - base[r]).abs().amax(-1)
        assert d[:cut].max() == 0, f"M={m} row {r}: a token after {cut} moved an earlier position"
        assert d[cut:].max() > 0, f"M={m} row {r}: the perturbation did nothing (test is vacuous)"
    x, _, lay = _at(_batch(2), m)
    r, s = 0, 4
    tok = (~lay.slot_mask[r]).nonzero().squeeze(1)
    prev, here = tok[lay.bag_id[r, tok] == s - 1], tok[lay.bag_id[r, tok] == s]
    x2 = x.clone()
    x2[r, prev] = (x2[r, prev] + 1 - 20) % (V - 20) + 20
    d = (hid(x2, lay)[r] - hid(x, lay)[r]).abs().amax(-1)
    assert d[here].max() > 0, f"M={m}: span s does not see span s-1 even through the loop"
    saved = [p.detach().clone() for p in model.W_prefix]
    with torch.no_grad():
        for p in model.W_prefix:
            p.zero_()
    dz = (hid(x2, lay)[r] - hid(x, lay)[r]).abs().amax(-1)
    with torch.no_grad():
        for p, v in zip(model.W_prefix, saved):
            p.copy_(v)
    assert dz[here].max() == 0, f"M={m}: span s sees span s-1 with the loop write cut: a side channel"


# ------------------------------------------------------------------ the loader with BOTH key families
def test_make_loader_composes_doc_exclusion_with_the_cell_law(monkeypatch):
    """`exclude_docs` (hygiene) and `cell_law` / `eval_cells` (lean) act on the same stream: the document
    filter wraps MORPH's create_dataloader, the re-layout wraps what comes out of it."""
    import lxtul.data as data_mod
    spec, rule = _spec()
    rt = SimpleNamespace(data_cfg=SimpleNamespace(spec_for=lambda seq: spec, rule=rule),
                         val_data_cfg=SimpleNamespace(spec_for=lambda seq: spec, rule=rule))
    src = [_batch(i) for i in range(8)]
    seen = {}

    def fake_stream(lo, hi, **kw):
        seen["range"], seen["kw"] = (lo, hi), kw
        yield from src
    monkeypatch.setattr(data_mod, "_stream_without_docs", fake_stream)
    mcfg = SimpleNamespace(data=SimpleNamespace(tokenizer="t", dataset="d", seq_len=SEQ))
    law = cell_law(LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, **POIS2), seed=3)
    it = data_mod.make_loader(mcfg, "train", 3, tul_rt=rt, prefetch=0, exclude_docs=(5, 9), cell_law=law)
    got = [next(it) for _ in range(8)]
    assert seen["range"] == (5, 9), "the document filter was bypassed"
    assert [b[2].prefix_k for b in got] == [law.draw(i)[0] for i in range(8)]
    # validation: no exclusion allowed, fixed re-layout
    with pytest.raises(ValueError, match="training stream only"):
        data_mod.make_loader(mcfg, "validation", 3, tul_rt=rt, prefetch=0, exclude_docs=(5, 9))
    monkeypatch.setattr("morph.training.data.create_dataloader", lambda **kw: iter(src))
    vi = data_mod.make_loader(mcfg, "validation", 3, tul_rt=rt, prefetch=0, eval_cells=2)
    assert all(next(vi)[2].prefix_k == 2 for _ in range(3))
