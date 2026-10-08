"""model.hc_region_fused: the HC region kernels against an fp64 reference.

``hc_region_entry`` / ``hc_region_exit`` (``_HCRegionEntry`` / ``_HCRegionExit``) compute the
same HC sublayer as the ``hc_fused_grad`` kernel path with other summation orders (the
projection is an in-kernel bf16 tl.dot instead of an autocast cuBLAS mm; the carrier grad is
written once instead of summed from two kernels and an addmm). Contract (blockfuse,
2026-10-08): through ``HyperConnectionResidual`` with a bf16 autocast sublayer, the region
path's error against the float64 reference is no worse than the fused_grad path's, for the
output and for the grads of the carrier, the projection weight and bias, the sublayer weight
and the folded inject term, at the production shapes (7680 and 1536 positions) and at edge
shapes (token counts that are not a multiple of the tile, a tiny grid, the inject term).
Also: an entry whose outputs are partly unused (``set_materialize_grads(False)`` path), the
CPU fallback (identical to the reference path), and the constructor's refusals.
"""

import pytest
import torch

from morph.kernels.triton import fused_hyper_connection as K
from morph.model.hyper_connections import HyperConnectionResidual
from test_graph_step import deterministic  # noqa: F401  (pytest fixture)

C = 1024
REL_SLACK, ABS_FLOOR = 0.02, 1e-7
# grad_proj_b sums graw over every token, and its error against fp64 is set by which tokens'
# bf16-rounded projections flip; two paths with different summation orders flip different
# tokens. Measured over 8 seeds at 7680 positions (perf/graph/blockfuse/seeds_gb.py,
# 2026-10-08): region/fused_grad error ratio 0.987-1.028, mean 0.999. Every other quantity
# read 0.999-1.001. A defect moves the ratio by 10x or more (the sabotage records).
SLACK = {"grad_proj_b": 0.05}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _hc(device, **kw):
    torch.manual_seed(1)
    hc = HyperConnectionResidual(C, n_streams=4, use_kernel=True, **kw).to(device)
    with torch.no_grad():                       # a mapping far from the identity init
        hc.proj.weight.normal_(0, 3.0 / (4 * C) ** 0.5)
        hc.proj.bias.normal_(0, 0.5)
    return hc


def _data(B, S, with_term, device):
    g = torch.Generator(device=device).manual_seed(0)
    h = torch.randn(B, S, 4, C, device=device, generator=g) * 2
    wf = torch.randn(C, C, device=device, generator=g) / C ** 0.5
    r = torch.randn(B, S, 4, C, device=device, generator=g)
    term = torch.randn(B, S, C, device=device, generator=g) if with_term else None
    return h, wf, r, term


def _run(hc, h0, wf0, r, term0, autocast=True):
    h = h0.detach().clone().requires_grad_(True)
    wf = wf0.detach().clone().requires_grad_(True)
    term = None if term0 is None else term0.detach().clone().requires_grad_(True)
    hc.zero_grad()
    with torch.autocast(h.device.type, dtype=torch.bfloat16, enabled=autocast):
        out = hc(h, lambda x: x @ wf.to(torch.bfloat16 if autocast else x.dtype),
                 post_inject=term)
    (out.float() * r).sum().backward()
    gs = [h.grad, hc.proj.weight.grad, hc.proj.bias.grad, wf.grad]
    if term is not None:
        gs.append(term.grad)
    return [out.detach()] + gs


def _run64(hc, h0, wf0, r, term0):
    h = h0.detach().double().requires_grad_(True)
    w = hc.proj.weight.detach().double().requires_grad_(True)
    b = hc.proj.bias.detach().double().requires_grad_(True)
    wf = wf0.detach().double().requires_grad_(True)
    term = None if term0 is None else term0.detach().double().requires_grad_(True)
    xb, hres, hpr = K.hc_pre_map_reference(h, w, b, hc.tau, hc.cayley_alpha, 3, hc.eps)
    out = K.hc_post_reference(hres, hpr, h, xb @ wf, term)
    (out * r.double()).sum().backward()
    gs = [h.grad, w.grad, b.grad, wf.grad]
    if term is not None:
        gs.append(term.grad)
    return [out.detach()] + gs


def _err(x, ref):
    return ((x.double() - ref).norm() / ref.norm().clamp_min(1e-300)).item()


CASES = [
    (6, 1280, False),   # prelude / coda / twin block: 7680 positions
    (6, 256, False),    # core pass: 1536 positions
    (3, 37, True),      # 111 tokens (partial last tile) + the folded inject term
    (2, 5, False),      # a grid smaller than one tile
]


@cuda
@pytest.mark.parametrize("B,S,with_term", CASES)
def test_region_error_no_worse_than_fused_grad(B, S, with_term):
    h, wf, r, term = _data(B, S, with_term, "cuda")
    old_hc, new_hc = _hc("cuda", fused_grad=True), _hc("cuda", region_fused=True)
    ref = _run64(old_hc, h, wf, r, term)
    old = _run(old_hc, h, wf, r, term)
    new = _run(new_hc, h, wf, r, term)
    names = ["out", "grad_h", "grad_proj_w", "grad_proj_b", "grad_sublayer_w", "grad_term"]
    rows = []
    for name, a, b, rr in zip(names, old, new, ref):
        e_old, e_new = _err(a, rr), _err(b, rr)
        rows.append(f"  {name:16s} fused_grad {e_old:.3e} region {e_new:.3e}")
        assert e_new <= e_old * (1 + SLACK.get(name, REL_SLACK)) + ABS_FLOOR, (
            f"{name}: region error {e_new:.3e} > fused_grad error {e_old:.3e}")
    print("\n".join(rows))
    assert new[1].dtype == h.dtype and new[1].stride() == h.stride()


def _entry_only(fn, h0, hc, gx, region):
    h = h0.clone().requires_grad_(True)
    w = hc.proj.weight.detach().clone().requires_grad_(True)
    b = hc.proj.bias.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        xb = fn(h, w, b, hc.tau, hc.cayley_alpha, 3, hc.eps)[0]
    (xb.float() * gx).sum().backward()
    return [xb.detach(), h.grad, w.grad, b.grad]


@cuda
def test_entry_with_unused_outputs():
    """Only x_bar reaches the loss: the exit never runs, h_link / Hres / Hpost_row get no
    grad, and the entry backward must treat them as zero. Same no-worse-than contract, against
    the old entry (``hc_pre_map``) on the same inputs."""
    h0, _, _, _ = _data(2, 96, False, "cuda")
    hc = _hc("cuda")
    gx = torch.randn(2, 96, C, device="cuda")
    new = _entry_only(K.hc_region_entry, h0, hc, gx, True)
    old = _entry_only(K.hc_pre_map, h0, hc, gx, False)
    h64 = h0.double().requires_grad_(True)
    w64 = hc.proj.weight.detach().double().requires_grad_(True)
    b64 = hc.proj.bias.detach().double().requires_grad_(True)
    xr, _, _ = K.hc_pre_map_reference(h64, w64, b64, hc.tau, hc.cayley_alpha, 3, hc.eps)
    (xr * gx.double()).sum().backward()
    ref = [xr.detach(), h64.grad, w64.grad, b64.grad]
    for name, a, b, rr in zip(["x_bar", "grad_h", "grad_proj_w", "grad_proj_b"], old, new, ref):
        e_old, e_new = _err(a, rr), _err(b, rr)
        assert e_new <= e_old * (1 + SLACK.get(name, REL_SLACK)) + ABS_FLOOR, (name, e_old, e_new)


@cuda
def test_region_falls_back_without_autocast():
    """Without bf16 autocast the old path's projection is fp32; the region key then runs the
    reference branch on both sides (one shared test), bit-identical to the plain kernel path."""
    h, wf, r, term = _data(2, 33, True, "cuda")
    a = _run(_hc("cuda", region_fused=True), h, wf, r, term, autocast=False)
    b = _run(_hc("cuda"), h, wf, r, term, autocast=False)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_cpu_fallback_is_the_reference_path():
    h, wf, r, term = _data(2, 7, True, "cpu")
    a = _run(_hc("cpu", region_fused=True), h, wf, r, term, autocast=False)
    b = _run(_hc("cpu"), h, wf, r, term, autocast=False)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


@pytest.mark.parametrize("kw", [dict(fused_norm=True), dict(n_streams=2),
                                dict(use_kernel=False), dict(cayley_iters=2)])
def test_constructor_refuses_unsupported(kw):
    base = dict(n_streams=4, use_kernel=True, region_fused=True)
    base.update(kw)
    with pytest.raises(ValueError, match="hc_region_fused"):
        HyperConnectionResidual(C, **base)


@cuda
def test_inject_fold_is_the_separate_add(monkeypatch, deterministic):
    """model.inject_fold on the tiny winner (test_graph_step's build): the forward is bit
    for bit the separate-add forward (same fp32 adds, one kernel), and every gradient agrees
    to fp32 rounding (the term's grad is summed over the 4 streams in the exit kernel instead
    of by autograd's broadcast reduction). Deterministic algorithms on (test_graph_step's
    fixture): without them two identical builds can already differ in the last bit."""
    from test_graph_step import _batches, _build, _compose
    keys = ("model.graph_safe=true", "training.capturable_optimizer=true",
            "model.hc_region_fused=true", "model.cca_prologue_tiled=true")
    res = []
    for fold in (False, True):
        cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *keys,
                           f"model.inject_fold={str(fold).lower()}")
        m, _ = _build(cfg, rt)
        assert m.cfg.inject_fold is fold
        x, y, lay = _batches(rt, 3)[0]
        torch.manual_seed(5)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)["loss"]
        loss.backward()
        res.append((loss.detach(), [p.grad for p in m.parameters()]))
    (l0, g0), (l1, g1) = res
    assert torch.equal(l0.view(torch.int32), l1.view(torch.int32)), (l0.item(), l1.item())
    worst = 0.0
    for a, b in zip(g0, g1):
        assert (a is None) == (b is None)
        if a is not None and a.norm() > 0:
            worst = max(worst, ((a - b).norm() / a.norm()).item())
    assert worst < 1e-3, worst


def test_inject_fold_needs_region():
    from morph.model.transformer import MORPHConfig, MORPHTransformer
    cfg = MORPHConfig(d_model=64, n_heads=2, n_kv_heads=2, vocab_size=64, max_seq_len=64,
                      context_len=64, n_prelude=1, n_core=1, n_coda=1,
                      channel_dims=(32, 20, 12), d_ff=96, bigram_hash_vocab=64,
                      inject_fold=True)
    with pytest.raises(ValueError, match="inject_fold"):
        MORPHTransformer(cfg)
