"""`model.lmhead_row_cap` (2026-10-10, phase-2 tracker C2): the token head's per-row log-prob
on the rows a loss reads only, at a fixed row count proven from the shapes.

What each test pins:
  1. `label_logprob_static_rows` (CPU, fp32): at a kept row the log-prob is the all-rows
     call's (fp64 tolerance: the GEMMs run on other row chunks), exactly 0 at every other
     row; the gradients to x and w are the all-rows gradients of the same upstream (rows not
     kept: exactly 0 to x); deterministic run to run; a batch over the cap is NaN
     everywhere (loud); a cap at exactly the kept count is finite.
  2. `static_row_pack`: the packing's own contract (order kept, unused slots read row 0,
     `back` inverts `src` on kept rows, `over` set iff more rows than the cap).
  3. `MORPHTransformer._lmhead_row_cap` bounds the rows of REAL packer batches (every span
     at the cap, a boundary every 4 tokens, random boundaries, DITTO rows) and is tight on
     the worst case; the emit-weight term adds one row per slot.
  4. The model (tiny `lxtul_pointer` winner, CPU): the loss and every gradient with the key
     on equal the key off to fp32 summation tolerance (and the key really ran: the head saw
     the cap's row count); a cap below the batch's rows gives a NaN loss; the key reaches
     MORPHConfig from Hydra and is on in `lxtul_pointer_fast`.
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np
import pytest
import torch

from morph.model import fused_ce
from morph.model.fused_ce import (fused_linear_label_logprob, label_logprob_static_rows,
                                  static_row_pack)

MASK = 3


def _case(n=90, d=16, V=70, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g, dtype=torch.float64).float()
    w = torch.randn(V, d, generator=g, dtype=torch.float64).float() * 0.3
    lab = torch.randint(0, V, (n,), generator=g)
    lab[lab == MASK] = MASK + 1
    keep = torch.rand(n, generator=g) < 0.8
    lab[torch.rand(n, generator=g) < 0.1] = -100
    keep &= lab != -100
    up = torch.randn(n, generator=g)
    return x, w, lab, keep, up


def _run(fn, x, w, lab, up):
    xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
    out = fn(xr, wr, lab)
    (torch.nan_to_num(out) * up).sum().backward()
    return out.detach(), xr.grad, wr.grad


def test_static_rows_equal_the_all_rows_call_on_kept_rows():
    x, w, lab, keep, up = _case()
    cap = int(keep.sum()) + 7
    kw = dict(chunk_size=16, mask_token_id=MASK)
    # the reference: the all-rows call, upstream zeroed off the kept rows (what the model's
    # weighted loss does: weight 0 there)
    up_k = torch.where(keep, up, torch.zeros_like(up))
    ro, rgx, rgw = _run(lambda a, b, c: fused_linear_label_logprob(a, b, c, **kw),
                        x, w, lab, up_k)
    so, sgx, sgw = _run(lambda a, b, c: label_logprob_static_rows(a, b, c, keep, cap, **kw),
                        x, w, lab, up)
    torch.testing.assert_close(so[keep], ro[keep], rtol=0, atol=1e-5)
    assert torch.equal(so[~keep], torch.zeros_like(so[~keep]))
    torch.testing.assert_close(sgx, rgx, rtol=0, atol=1e-6)
    torch.testing.assert_close(sgw, rgw, rtol=0, atol=1e-6)
    assert torch.equal(sgx[~keep], torch.zeros_like(sgx[~keep]))
    assert float(sgx[keep].abs().max()) > 0 and float(sgw.abs().max()) > 0
    so2, sgx2, sgw2 = _run(lambda a, b, c: label_logprob_static_rows(a, b, c, keep, cap, **kw),
                           x, w, lab, up)
    assert torch.equal(so, so2) and torch.equal(sgx, sgx2) and torch.equal(sgw, sgw2)


def test_over_the_cap_is_nan_and_at_the_cap_is_finite():
    x, w, lab, keep, up = _case(seed=2)
    k = int(keep.sum())
    over = label_logprob_static_rows(x, w, lab, keep, k - 1, chunk_size=16)
    assert torch.isnan(over).all()
    at = label_logprob_static_rows(x, w, lab, keep, k, chunk_size=16)
    assert torch.isfinite(at).all()


def test_static_row_pack_contract():
    keep = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.bool)
    src, used, back, over = static_row_pack(keep, 6)
    assert src.tolist() == [1, 2, 4, 7, 0, 0] and used.tolist() == [1, 1, 1, 1, 0, 0]
    assert back.tolist() == [6, 0, 1, 6, 2, 6, 6, 3] and not bool(over)
    src, used, back, over = static_row_pack(keep, 3)
    assert src.tolist() == [1, 2, 4] and bool(over) and back.tolist()[7] == 3


# ── 3. the bound on real packer rows ─────────────────────────────────────────────────

def _pack(stream, B=6, ditto_rows=0):
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(100, dtype=bool)
    lut[1] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)
    spec = TulLayoutSpec(seq_len=1024, prefix_k=4, max_slots=64, slot_id=99)
    kw = {}
    if ditto_rows:
        kw = dict(ditto_rows=ditto_rows, ditto_rng=np.random.default_rng(0))
    return pack_tul_batch(list(stream), rule, spec, B, **kw)


@pytest.mark.parametrize("kind", ["no_boundary", "every4", "random", "ditto"])
def test_lmhead_row_cap_bounds_real_packer_rows(kind):
    from morph.model.transformer import MORPHTransformer
    rng = np.random.default_rng(3)
    n = 6 * 1300
    if kind == "no_boundary":                          # every span forced at the 32 cap
        stream = rng.integers(5, 99, n)
    elif kind == "every4":
        stream = np.where(np.arange(n) % 4 == 3, 1, rng.integers(5, 99, n))
    else:
        stream = np.where(rng.random(n) < 0.06, 1, rng.integers(5, 99, n))
    x, y, lay = _pack(stream.tolist(), ditto_rows=2 if kind == "ditto" else 0)
    B, L = x.shape
    need = (y != -100) & ~lay.slot_mask                      # emit weight 0: token rows
    need_e = need | ((y != -100) & lay.slot_mask)            # emit weight != 0: + emit rows
    cap = MORPHTransformer._lmhead_row_cap(B, L, lay, 32, False)
    cap_e = MORPHTransformer._lmhead_row_cap(B, L, lay, 32, True)
    assert cap == 6 * 1140 and cap_e >= cap
    for nd, c in ((need, cap), (need_e, cap_e)):
        per_row = nd.sum(1)
        assert int(per_row.sum()) <= c, (kind, int(per_row.sum()), c)
        assert int(per_row.max()) <= c // B
    if kind == "no_boundary":
        assert int(need.sum(1).max()) >= 1100                # within 4 % of the bound (1140)


def test_the_rows_a_loss_reads_at_the_production_shape():
    """`_lmhead_rows` on a real batch: at emit weight 0 exactly the labelled TOKEN rows, and
    under the bound per row; the emitting slot rows join only with a nonzero weight."""
    from morph.model.transformer import MORPHTransformer
    rng = np.random.default_rng(3)
    x, y, lay = _pack(rng.integers(5, 99, 6 * 1300).tolist())          # every span at the cap
    lab = y.reshape(-1)
    row_off = (torch.arange(6) * x.shape[1]).unsqueeze(1)
    z = torch.where(lay.slot_valid, lay.slot_index + row_off + lay.prefix_k - 1,
                    lab.shape[0]).reshape(-1)
    for w_emit in (0.0, 0.5):
        row_w = torch.ones(lab.shape[0] + 1).index_fill_(0, z, w_emit)[:-1]
        need = MORPHTransformer._lmhead_rows(lab, lay.slot_mask, row_w)
        tok = (lab != -100) & ~lay.slot_mask.reshape(-1)
        if w_emit == 0.0:
            assert torch.equal(need, tok)
            cap = MORPHTransformer._lmhead_row_cap(6, x.shape[1], lay, 32, False)
        else:
            assert torch.equal(need, tok | ((lab != -100) & lay.slot_mask.reshape(-1)))
            assert int(need.sum()) > int(tok.sum())
            cap = MORPHTransformer._lmhead_row_cap(6, x.shape[1], lay, 32, True)
        assert int(need.view(6, -1).sum(1).max()) <= cap // 6


# ── 4. the model ─────────────────────────────────────────────────────────────────────

def _model(monkeypatch, on: bool, seed=0):
    from test_graph_step import _TINY, _compose
    from morph.model.transformer import MORPHTransformer
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config
    cfg, rt = _compose(monkeypatch, "lxtul_pointer", f"model.lmhead_row_cap={str(on).lower()}")
    mc = dataclasses.replace(build_morph_config(cfg, tul=rt.model_cfg), **_TINY)
    assert mc.lmhead_row_cap is on
    torch.manual_seed(seed)
    m = MORPHTransformer(mc)
    apply_quantization(m, cfg)
    m.tul_fan_target_build()
    m.tul_fan_target_sync()
    return m.train(), rt


def _batch(rt):
    from morph.model.tul_layout import TulLayoutSpec, pack_tul_batch
    from test_graph_step import B, SEQ, _stream
    from test_tul_strict_geometry import _rule
    spec = TulLayoutSpec(seq_len=SEQ, prefix_k=rt.data_cfg.prefix_k, max_slots=10,
                         slot_id=rt.data_cfg.slot_id)
    return pack_tul_batch(_stream(B * 400, seed=101, p_boundary=0.12), _rule(), spec, B)


def _step(m, x, y, lay, gen):
    m.zero_grad(set_to_none=True)
    m.set_gain_pass_rng_state(gen)
    torch.manual_seed(11)
    out = m(x, labels=y, slot_layout=lay)
    out["loss"].backward()
    return out["loss"].detach(), {n: p.grad for n, p in m.named_parameters()
                                  if p.grad is not None}


def test_model_loss_and_grads_match_the_all_rows_head(monkeypatch):
    m_off, rt = _model(monkeypatch, False)
    m_on, _ = _model(monkeypatch, True)
    m_on.load_state_dict(m_off.state_dict())
    x, y, lay = _batch(rt)
    gen = m_off.gain_pass_rng_state()
    rows = []
    real = fused_ce.fused_linear_label_logprob

    def spy(xx, *a, **k):
        rows.append(int(xx.shape[0]))
        return real(xx, *a, **k)
    monkeypatch.setattr(fused_ce, "fused_linear_label_logprob", spy)
    l_off, g_off = _step(m_off, x, y, lay, gen)
    n_off = list(rows)
    rows.clear()
    l_on, g_on = _step(m_on, x, y, lay, gen)
    B, L = x.shape
    from morph.model.transformer import MORPHTransformer
    cap = MORPHTransformer._lmhead_row_cap(B, L, lay, int(m_on.cfg.tul.bound_span_cap), False)
    assert cap < B * L and rows == [cap], (rows, cap)       # the capped head really ran
    assert n_off == [], n_off                               # the transformer's own symbol
    torch.testing.assert_close(l_on, l_off, rtol=1e-6, atol=0)
    assert set(g_on) == set(g_off)
    for k in g_on:
        torch.testing.assert_close(g_on[k], g_off[k], rtol=1e-4, atol=1e-7, msg=k)


def test_model_cap_below_the_batch_is_nan(monkeypatch):
    from morph.model.transformer import MORPHTransformer
    m, rt = _model(monkeypatch, True)
    x, y, lay = _batch(rt)
    monkeypatch.setattr(MORPHTransformer, "_lmhead_row_cap",
                        staticmethod(lambda B, L, layout, J, e: 5))
    torch.manual_seed(11)
    out = m(x, labels=y, slot_layout=lay)
    assert torch.isnan(out["loss"])


def test_the_key_is_off_until_a_quality_check():
    """Not bit-exact and no 5k quality check yet: off in base AND in the fast recipe."""
    from hydra import compose, initialize_config_dir
    from morph.training.train import build_morph_config
    cdir = str(pathlib.Path(__file__).resolve().parents[1] / "morph" / "configs")
    with initialize_config_dir(config_dir=cdir, version_base=None):
        base = compose(config_name="base")
        fast = compose(config_name="lxtul_pointer_fast")
    assert build_morph_config(base).lmhead_row_cap is False
    assert fast.model.lmhead_row_cap is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels, CUDA")
def test_cuda_token_head_shape_error_against_fp64_is_the_all_rows_error():
    """The production width (3840 rows x 1024, V 49169, kernel path, bf16 autocast, kept
    logits): the static-rows head is as close to an fp64 reference as the all-rows head is
    (per-row log-prob, grad_x, grad_w), and the two differ by summation order only."""
    V, D, N, MASK = 49169, 1024, 3840, 4          # half the batch: the fp64 reference fits
    g = torch.Generator().manual_seed(5)
    x = (torch.randn(N, D, generator=g) * 0.6).to(torch.bfloat16).float().cuda()
    w = (torch.randn(V, D, generator=g) * 0.05).cuda()
    lab = torch.randint(0, V, (N,), generator=g)
    lab[lab == MASK] = MASK + 1
    keep = torch.rand(N, generator=g) < 0.87
    lab = torch.where(keep, lab, torch.full_like(lab, -100)).cuda()
    keep = keep.cuda()
    up = (torch.randn(N, generator=g) * 1e-3).cuda() * keep
    kw = dict(chunk_size=1024, mask_token_id=MASK, softmax_kernel=True, keep_logits=True)

    def run(fn):
        xr, wr = x.clone().requires_grad_(True), w.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(xr, wr)
        (torch.where(keep, out, torch.zeros_like(out)) * up).sum().backward()
        return out.detach().float(), xr.grad.float(), wr.grad.float()
    full = run(lambda a, b: fused_linear_label_logprob(a, b, lab, **kw))
    stat = run(lambda a, b: label_logprob_static_rows(a, b, lab, keep, int(keep.sum()) + 64,
                                                      **kw))
    # fp64 reference on the bf16-rounded operands the GEMMs see
    xd, wd = x.double().requires_grad_(True), w.to(torch.bfloat16).double().requires_grad_(True)
    logits = xd @ wd.t()
    logits[:, MASK] = float("-inf")
    lp = torch.log_softmax(logits, -1).gather(1, lab.clamp_min(0)[:, None]).squeeze(1)
    (torch.where(keep, lp, torch.zeros_like(lp)) * up.double()).sum().backward()
    ref = (lp.detach(), xd.grad, wd.grad)
    for name, f, s, r in zip(("lp", "grad_x", "grad_w"), full, stat, ref):
        m = keep if name == "lp" else slice(None)
        e_full = (f[m].double() - r[m]).abs().max().item()
        e_stat = (s[m].double() - r[m]).abs().max().item()
        scale = r[m].abs().max().item()
        assert e_stat <= 2.0 * e_full + 1e-6 * scale, (name, e_full, e_stat, scale)
