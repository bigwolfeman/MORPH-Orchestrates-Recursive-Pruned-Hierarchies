"""CPU tests of the speed keys: compact_slots, static_depth, cot_clip_fn, ce_rows, spandec_packed
(lxtul/model.py, lxtul/ce.py) and the plain loop's row_compact (parcae_lm/models/parcae/parcae.py).

Every key is compared with the key off on the same weights, the same RNG stream and the same batch:
the loss and EVERY parameter gradient, in float64 so that a rounding-free reference separates "same
function" from "close". FlexAttention has no CPU backward, so `_flex` is replaced by SDPA under the
dense mask the BlockMask encodes (the mask_mod is the object under test, not the kernel); the CE
GEMMs run in float64 (`ce._CDT`, `ce._ACC`).

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_compact.py
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, create_mask

import lxtul.data  # noqa: F401  (puts MORPH_ROOT on sys.path)
import lxtul.attention as A
import lxtul.ce as ce
import lxtul.model as lm
import parcae_lm
from lxtul.model import LXTULConfig, LXTULParcae, span_row_cap, tok_row_cap

V, SLOT, EOS, SEQ, K, S, J = 512, 4, 1, 128, 4, 16, 12
TOL = 1e-9


def _flex_ref(q, k, v, block_mask=None, enable_gqa=False):
    B, _, Tq, _ = q.shape
    m = create_mask(block_mask.mask_mod, B, 1, Tq, k.shape[2], device=q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=enable_gqa)


@pytest.fixture(autouse=True)
def cpu_reference_kernels(request, monkeypatch):
    """float64 + SDPA stand-ins on CPU. Tests marked `fp32` choose their own precision and run on
    the device in their `device` parameter; only the CPU case needs the flex/chunk stand-ins."""
    fp32 = request.node.get_closest_marker("fp32") is not None
    device = request.node.callspec.params.get("device", "cpu") if fp32 else "cpu"
    if device == "cpu":
        monkeypatch.setattr(A, "_flex", _flex_ref)
        monkeypatch.setattr(A, "_mask", create_block_mask)
        monkeypatch.setattr(ce, "_chunk", ce._chunk_impl)
    if fp32:
        monkeypatch.setattr(ce, "_CDT", torch.float32)       # same GEMM precision on both sides
        return
    monkeypatch.setattr(ce, "_CDT", torch.float64)
    monkeypatch.setattr(ce, "_ACC", torch.float64)
    # the model pins some math to fp32 (`.float()`, `dtype=torch.float32`): lift that to fp64 so the
    # comparison is against a reference with no rounding to hide a wrong term behind
    monkeypatch.setattr(torch.Tensor, "float", lambda self, *a: self.to(torch.float64))
    monkeypatch.setattr(torch, "float32", torch.float64)


def _batch(seed: int, B: int = 3, gap=(8, 14)):
    """Rows packed by MORPH's packer; boundaries every `gap` ids so that rows keep PAD slots."""
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[EOS] = True
    rule = BoundaryRule(lut, min_span=4, span_cap=J, eos_id=EOS)
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=K, max_slots=S, slot_id=SLOT)
    rng = np.random.default_rng(seed)
    buf = rng.integers(20, V, size=B * (spec.l_total + 1) * 2).tolist()
    i = 0
    while i < len(buf):
        buf[i] = int(rng.integers(10, 20))
        i += int(rng.integers(*gap))
    return pack_tul_batch(buf, rule, spec, B)


def _to(batch, device):
    x, y, lay = batch
    for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
        setattr(lay, f, getattr(lay, f).to(device))
    return x.to(device), y.to(device), lay


def _randomize(m):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        m.reg_p_cell_embed.copy_(torch.randn(m.reg_p_cell_embed.shape, generator=g) * 0.05)
    return m


def _build(**keys):
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=J,
                                     spandec_layers=1, **keys))
    m.init_slot_seed()
    return _randomize(m).double().train()


def _twin(base: LXTULParcae, **keys) -> LXTULParcae:
    """The same weights under other keys (a fresh module: spandec_packed changes the blocks' class)."""
    m = LXTULParcae(base.config, dataclasses.replace(base.tul, **keys)).double().train()
    m.load_state_dict(base.state_dict())
    return m


def _depths(B, pattern="spread", device="cpu"):
    if pattern == "spread":       # every depth 1..8 present, incl. pad slots (they are reset to 1)
        return ((torch.arange(B * S, device=device).view(B, S) * 5) % 8 + 1).double()
    return torch.full((B, S), float(pattern), device=device).double()


def _run(m, x, y, lay, pat="spread", seed=3, depth=None, monkeypatch=None):
    B = x.shape[0]
    m.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    if x.is_cuda:
        torch.cuda.manual_seed_all(seed)
    dep = _depths(B, pat, x.device)
    orig = torch.poisson
    torch.poisson = lambda t, generator=None: dep.to(t.dtype)
    try:
        o = m(x, y, lay, depth=depth)
        o["loss"].backward()
    finally:
        torch.poisson = orig
    g = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
    return o, g


def _assert_same(a, ga, b, gb, tol=TOL):
    assert torch.isfinite(a["loss"]) and torch.isfinite(b["loss"])
    assert abs(float(a["loss"]) - float(b["loss"])) <= tol * max(1.0, abs(float(a["loss"]))), \
        (float(a["loss"]), float(b["loss"]))
    assert ga.keys() == gb.keys()
    worst = 0.0
    for n in ga:
        scale = max(float(ga[n].abs().max()), 1e-12)
        err = float((ga[n] - gb[n]).abs().max()) / scale
        worst = max(worst, err)
        assert err <= tol * 10, f"{n}: relative gradient error {err:.3e}"
    return worst


@pytest.fixture(scope="module")
def data():
    x, y, lay = _batch(1)
    nv = lay.slot_valid.sum(1)
    assert nv.min() < S and nv.min() != nv.max(), f"rows must keep pad slots and differ: {nv.tolist()}"
    return x, y, lay


@pytest.fixture(scope="module", params=["default", "hot"])
def base(request):
    """`hot`: the gain hinge and the cotangent clip are both ENGAGED (the defaults leave the hinge at
    zero on this tiny random model), so the keys are compared with those two terms in the loss."""
    if request.param == "hot":
        return _build(gain_target=0.05, cot_clip=0.2)
    return _build()


def test_batch_has_pads_last_and_caps_hold():
    """The invariants the keys rest on, on many packed batches: pad slots are last; the labelled
    token rows and the span decoder's positions stay under the shape-proven caps."""
    for seed in range(12):
        for gap in ((3, 6), (8, 14), (14, 40)):
            x, y, lay = _batch(seed, B=4, gap=gap)
            v = lay.slot_valid
            assert (v[:, 1:] <= v[:, :-1]).all(), "a valid slot follows a pad slot"
            tok_lab = ((y >= 0) & ~lay.slot_mask).sum(1)
            assert tok_lab.max() <= tok_row_cap(x.shape[1], K, J, S)
            # the decoder's positions: tokens of spans 1..n-1, at most J each
            n = v.sum(1)
            assert ((~lay.slot_mask).sum(1) <= x.shape[1] - K * n).all()
    # the production shapes quote MORPH's measured bounds
    assert tok_row_cap(1280, 4, 32, 64) == 1140
    assert span_row_cap(1280, 4, 32, 64) == 1132


@pytest.mark.parametrize("pat", ["spread", "8", "1"])
def test_compact_slots_is_the_same_function(base, data, pat):
    x, y, lay = data
    a, ga = _run(base, x, y, lay, pat)
    on = _twin(base, compact_slots=True, compact_bucket_slots=4)
    b, gb = _run(on, x, y, lay, pat)
    worst = _assert_same(a, ga, b, gb)
    assert worst < TOL * 10


def test_compact_slots_packs_fewer_cells_and_keeps_every_valid_slot(base, data):
    x, y, lay = data
    on = _twin(base, compact_slots=True, compact_bucket_slots=4)
    pack = on._slot_pack(lay.slot_valid)
    B = x.shape[0]
    flat = lay.slot_valid.flatten().nonzero().squeeze(1)
    assert pack.sel.numel() < B * S, "no pad slot was dropped"
    assert pack.sel.numel() % 4 == 0
    assert set(flat.tolist()) <= set(pack.sel.tolist()), "a valid slot, finished or not, is missing"
    assert pack.sel[: flat.numel()].tolist() == flat.tolist(), "valid slots lead in row-major order"


def test_dropping_a_valid_slot_is_caught(base, data, monkeypatch):
    """Negative control: the equivalence check must FAIL when a slot that is valid is left out of
    the pack (the sabotage of the first design: a finished slot is a K/V source and must run)."""
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    on = _twin(base, compact_slots=True, compact_bucket_slots=4)
    real = on._slot_pack

    def sabotaged(valid, M=0):
        p = real(valid, M)
        keep = p.sel != p.sel[1]                  # drop one valid slot (the second one of row 0)
        sel = p.sel[keep]
        cut = sel.numel() // 4 * 4                # keep the bucket multiple
        sel = sel[:cut]
        M = on.tul.cells
        row, slot = sel // S, sel % S
        cr = row[:, None].expand(-1, M).reshape(-1)
        cs = slot[:, None].expand(-1, M).reshape(-1)
        pos = (slot[:, None] * M + torch.arange(M)).reshape(-1)
        ids = torch.full((1, sel.numel() * M), SLOT, dtype=torch.long)
        off = on.config.n_layers_in_prelude
        ves = [on._ve(off + i, ids) for i in range(len(on.transformer.core_block))]
        return lm._SlotPack(sel, A.slot_pack_mask(cr, cs), on.freqs_cis[:, pos], ves, p.bad)
    monkeypatch.setattr(on, "_slot_pack", sabotaged)
    b, gb = _run(on, x, y, lay)
    with pytest.raises(AssertionError):
        _assert_same(a, ga, b, gb)


def test_pad_slot_before_a_valid_slot_is_a_nan_loss(base, data):
    """compact_slots rests on 'pad slots are last'. A layout that breaks it must be loud."""
    x, y, lay = data
    bad = dataclasses.replace(lay, slot_valid=lay.slot_valid.clone())
    bad.slot_valid[0, 0] = False                       # slot 0 of row 0 becomes a pad before valid slots
    on = _twin(base, compact_slots=True, compact_bucket_slots=4)
    o, _ = _run(on, x, y, bad)
    assert torch.isnan(o["loss"])
    o2, _ = _run(on, x, y, lay)
    assert torch.isfinite(o2["loss"])


def test_a_finished_slot_cannot_be_skipped(base, data):
    """Why the loop cannot drop finished slots. A later slot reads the K/V of every earlier slot at
    every layer of the pass, and a finished slot's layer inputs still move with the EARLIER slots'
    states: perturb only an earlier slot and a later slot's pass output changes, cell by cell.
    Its frozen `h` and its seed `e` are untouched."""
    m = base
    d = m.config.n_embd
    g = torch.Generator().manual_seed(5)
    h = torch.randn(1, S * K, d, generator=g).double()
    e = torch.randn(1, S * K, d, generator=g).double()
    cf = m.freqs_cis[:, : S * K]
    ves = [None] * len(m.transformer.core_block)
    m._core_mask = A.core_mask(S * K, K, "cpu")
    with torch.no_grad():
        base_out = m._core_step(h, e, cf, ves)
        h2 = h.clone()
        h2[:, : 2 * K] += 1.0                                   # slots 0 and 1 only
        out2 = m._core_step(h2, e, cf, ves)
    f = 5                                                       # a "finished" later slot: h, e unchanged
    cells = slice(f * K, (f + 1) * K)
    assert (out2[:, cells] - base_out[:, cells]).abs().max() > 1e-6


@pytest.mark.parametrize("key", ["cot_clip_fn", "ce_rows", "spandec_packed", "static_depth"])
def test_single_key_is_the_same_function(base, data, key):
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    on = _twin(base, **{key: True})
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_cot_clip_fn_is_bitwise_the_hook(base, data):
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    on = _twin(base, cot_clip_fn=True)
    b, gb = _run(on, x, y, lay)
    assert float(a["loss"]) == float(b["loss"])
    assert all(torch.equal(ga[n], gb[n]) for n in ga), "the clip changed a gradient"


def test_hot_config_engages_the_hinge(data):
    x, y, lay = data
    m = _build(gain_target=0.05, cot_clip=0.2)
    o, _ = _run(m, x, y, lay)
    assert float(o["metrics"]["gain"]) > 0.05, "the hinge is idle: the hot comparison is vacuous"
    cold = _twin(m, gain_lambda=0.0)
    o2, _ = _run(cold, x, y, lay)
    assert abs(float(o["loss"]) - float(o2["loss"])) > 1e-6


def test_cot_clip_actually_clips(base, data):
    """The clip is live in this configuration: with a tiny clip factor the gradient moves."""
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    hard = _twin(base, cot_clip=1e-3)
    b, gb = _run(hard, x, y, lay)
    assert any(not torch.allclose(ga[n], gb[n]) for n in ga), "cot_clip had no effect: the test is vacuous"
    hard_fn = _twin(base, cot_clip=1e-3, cot_clip_fn=True)
    c, gc = _run(hard_fn, x, y, lay)
    assert all(torch.equal(gb[n], gc[n]) for n in gb), "the hook-free clip differs from the hook"


def test_all_keys_together(base, data):
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    on = _twin(base, compact_slots=True, compact_bucket_slots=4, static_depth=True, cot_clip_fn=True,
               ce_rows=True, spandec_packed=True)
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_eval_depth_sweep_and_forced_depth(base, data):
    x, y, lay = data
    off, on = base.eval(), _twin(base, compact_slots=True, compact_bucket_slots=4, static_depth=True).eval()
    with torch.no_grad():
        for depth in (None, 1, 3, 8):
            h0 = off(x, None, lay, depth=depth)["hidden"]
            h1 = on(x, None, lay, depth=depth)["hidden"]
            tok = ~lay.slot_mask
            assert (h0 - h1)[tok].abs().max() < 1e-9, f"depth {depth}"
    base.train()


def test_pointer_arm_and_ce_rows(data):
    x, y, lay = data
    m = _build(pointer_heads=2)
    a, ga = _run(m, x, y, lay)
    on = _twin(m, ce_rows=True, spandec_packed=True)
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_ungraded_fan_with_keys(data):
    x, y, lay = data
    m = _build(select="none")
    a, ga = _run(m, x, y, lay)
    on = _twin(m, compact_slots=True, compact_bucket_slots=4, ce_rows=True, spandec_packed=True,
               cot_clip_fn=True)
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_ckpt_passes_zero_is_the_same_function(base, data):
    """`ckpt_passes: 0` is a config change only: the recompute is the only thing it removes."""
    x, y, lay = data
    a, ga = _run(base, x, y, lay)
    on = _twin(base, ckpt_passes=0)
    b, gb = _run(on, x, y, lay)
    _assert_same(a, ga, b, gb)


def test_over_cap_batch_is_a_nan_loss(base, data, monkeypatch):
    x, y, lay = data
    real_tok, real_span = lm.tok_row_cap, lm.span_row_cap
    on = _twin(base, ce_rows=True)
    monkeypatch.setattr(lm, "tok_row_cap", lambda *a: 3)
    o, _ = _run(on, x, y, lay)
    assert torch.isnan(o["ce"])
    monkeypatch.setattr(lm, "tok_row_cap", real_tok)
    on2 = _twin(base, spandec_packed=True)
    monkeypatch.setattr(lm, "span_row_cap", lambda *a: 3)
    o2, _ = _run(on2, x, y, lay)
    assert torch.isnan(o2["spandec"])
    monkeypatch.setattr(lm, "span_row_cap", real_span)
    o3, _ = _run(on2, x, y, lay)
    assert torch.isfinite(o3["spandec"])


FP32_KEYS = {
    "compact": dict(compact_slots=True),
    "static": dict(static_depth=True),
    "cotfn": dict(cot_clip_fn=True),
    "rows": dict(ce_rows=True),
    "spandec": dict(spandec_packed=True),
    "ckpt0": dict(ckpt_passes=0),
    "all": dict(compact_slots=True, static_depth=True, cot_clip_fn=True, ce_rows=True,
                spandec_packed=True, ckpt_passes=0),
}


@pytest.mark.fp32
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
@pytest.mark.parametrize("name", list(FP32_KEYS))
def test_keys_in_fp32_on_the_real_kernels(name, device):
    """Rounding-only bound at the precision training uses (fp32 here; bf16 autocast adds its own
    noise to both sides). On CUDA this runs the real FlexAttention and the compiled CE chunk."""
    x, y, lay = _to(_batch(1), device)
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=SEQ + K * S)
    keys = dict(gain_target=0.05, cot_clip=0.2)
    m = LXTULParcae(cfg, LXTULConfig(slot_id=SLOT, max_slots=S, cells=K, span_cap=J,
                                     spandec_layers=1, **keys))
    m.init_slot_seed()
    m = _randomize(m).to(device).train()
    a, ga = _run(m, x, y, lay)
    on = LXTULParcae(cfg, dataclasses.replace(m.tul, **FP32_KEYS[name])).to(device).train()
    on.load_state_dict(m.state_dict())
    b, gb = _run(on, x, y, lay)
    assert abs(float(a["loss"]) - float(b["loss"])) <= 2e-5 * abs(float(a["loss"]))
    for n in ga:
        scale = max(float(ga[n].abs().max()), 1e-12)
        assert float((ga[n] - gb[n]).abs().max()) / scale <= 2e-3, n


# ------------------------------------------------------------------ lxtul/ce.py
def test_static_pack_roundtrip_and_grads():
    g = torch.Generator().manual_seed(2)
    keep = torch.rand(3, 40, generator=g) < 0.4
    cap = int(keep.sum(1).max()) + 2
    src, used, back, over = ce.static_pack(keep, cap)
    assert not over
    for r in range(3):
        kept = keep[r].nonzero().squeeze(1)
        assert src[r][used[r]].tolist() == kept.tolist()
        assert used[r].sum() == kept.numel()
        assert (back[r][keep[r]] == torch.arange(kept.numel())).all()
        assert (back[r][~keep[r]] == cap).all()
    x = torch.randn(3, 40, 5, dtype=torch.float64, requires_grad=True)
    y = ce.take_rows(x, src, back)
    z = ce.put_rows(y, src, used, back)
    assert torch.equal(z, x * keep[..., None])            # unkept entries read 0
    w = torch.randn_like(z)
    (z * w).sum().backward()
    assert torch.equal(x.grad, w * keep[..., None])
    _, _, _, over2 = ce.static_pack(keep, 1)
    assert over2


def test_linear_ce_rows_matches_linear_ce():
    g = torch.Generator().manual_seed(4)
    N, d, Vv = 70, 16, 50
    x = torch.randn(N, d, generator=g).double().requires_grad_()
    W = torch.randn(Vv, d, generator=g).double().requires_grad_()
    lab = torch.randint(0, Vv, (N,), generator=g)
    lab[torch.rand(N, generator=g) < 0.4] = -100
    ref = ce.linear_ce(x, W, lab, 0.7, 3)
    ref.backward()
    gx, gW = x.grad.clone(), W.grad.clone()
    x.grad = W.grad = None
    out = ce.linear_ce_rows(x, W, lab, 0.7, 3)
    out.backward()
    assert abs(float(out) - float(ref)) < 1e-12
    assert (x.grad - gx).abs().max() < 1e-12 and (W.grad - gW).abs().max() < 1e-12
    # and against autograd on the dense logits
    x.grad = W.grad = None
    lg = (x @ W.T) * 0.7
    lg = lg.clone()
    lg[:, 3] = float("-inf")
    exact = F.cross_entropy(lg, lab, ignore_index=-100)
    assert abs(float(exact) - float(out)) < 1e-12


# ------------------------------------------------------------------ plain row_compact
def _plain(row_compact: bool):
    torch.manual_seed(0)
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=64,
        mean_recurrence=6, mean_backprop_depth=3, row_compact=row_compact)
    m = cfg.construct_model(gradient_checkpointing=False)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in m.parameters():
            if p.ndim >= 2 and not p.abs().any():
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return m.double().train()


@pytest.mark.parametrize("step", [0, 1, 2, 3, 4, 5])
def test_plain_row_compact_is_the_same_function(step, monkeypatch):
    off, on = _plain(False), _plain(True)
    on.load_state_dict(off.state_dict())
    g = torch.Generator().manual_seed(9)
    x = torch.randint(0, V, (6, 64), generator=g)
    y = torch.randint(0, V, (6, 64), generator=g)
    sizes = []
    real = on.core_block_forward
    monkeypatch.setattr(on, "core_block_forward", lambda xa, *a, **k: (sizes.append(xa.shape[0]), real(xa, *a, **k))[1])
    outs = []
    for m in (off, on):
        m.zero_grad(set_to_none=True)
        m.step = step
        torch.manual_seed(7)                       # initialize_state draws from the global RNG
        o = m(x, labels=y)
        o["loss"].backward()
        outs.append((o["loss"].detach(), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}))
    (la, ga), (lb, gb) = outs
    assert abs(float(la) - float(lb)) < 1e-10 * max(1.0, abs(float(la)))
    assert ga.keys() == gb.keys()
    for n in ga:
        assert (ga[n] - gb[n]).abs().max() <= 1e-9 * max(float(ga[n].abs().max()), 1e-12), n
    n_cpu, k_cpu = on._sample_batch_depths(6, None)
    assert sum(sizes) == int(n_cpu.sum() + k_cpu.sum()), "a finished row ran a pass, or a drawn pass was skipped"
    assert max(sizes) <= 6
    if step == 0:                                  # the draw spreads: the key is not a no-op here
        assert len(set(n_cpu.tolist() + k_cpu.tolist())) > 1 and sum(sizes) < 6 * (int(n_cpu.max()) + int(k_cpu.max()))


def test_plain_row_compact_refuses_per_batch():
    cfg = parcae_lm.create_config(
        "parcae-small-140m", n_embd=64, num_attention_heads=2, num_key_value_heads=2,
        intermediate_size=128, recurrent_embedding_dimension=64,
        recurrent_intermediation_embedding_dimension=128, n_layers_in_prelude=1,
        n_layers_in_recurrent_block=2, n_layers_in_coda=1, vocab_size=V, block_size=64,
        recurrent_iteration_method="per-batch", row_compact=True)
    with pytest.raises(ValueError, match="per-sequence"):
        cfg.construct_model(gradient_checkpointing=False)
