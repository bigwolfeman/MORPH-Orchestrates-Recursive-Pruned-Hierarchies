"""``model.slot_depth_stratified`` and ``model.slot_compact`` (morph/model/slot_compact.py).

What each test pins:
  1. The stratified draw's capacity bound ``C_t`` holds on every row (pads included) and is
     attained, at the LXTUL shape and at shapes whose thresholds sit near integers; each
     slot's marginal law is the clamped Poisson law (CPU).
  2. ``compact_order`` is a permutation with the active rows first, in order, and its
     inverse; ``take_rows`` / ``put_rows`` pass a float64 gradcheck (CPU).
  3. The strict attention kernel with query rows by position (``qpos``) equals the full
     kernel's rows at those positions, forward and every gradient (CUDA).
  4. THE MACHINERY IS EXACT: with the stratified draw on in both arms, ``"gather"`` (only the
     active rows of each pass) equals ``"full"`` (every row, frozen cells served from the
     same caches) in loss (1e-6 relative; measured bit-identical) and every gradient (1e-4 of
     each tensor's norm; measured worst 1.1e-5 fp32, 1.5e-6 bf16), in fp32 and in bf16
     autocast (CUDA, the tiny winner in train mode, dropout 0, gain hinge and fixed-point
     term on). A positive control shows the test can
     see a function change: ``"off"`` (today's recompute of frozen cells) against
     ``"full"`` differs by far more than the tolerance.
  5. Eval forwards are unchanged by both keys (they act on the training forward only).
  6. Build refusals (CPU).

Sabotage checks, run and reverted (session report, not committed):
  (a) the gather arm keeps the caches of pass 0 only (a frozen cell reads stale pre-freeze
      keys / values) -> test 4 fails;
  (b) the plans hand the kernel qpos = 0..n-1 instead of the cells' positions -> tests 4
      (and 3, through its own sabotage of the call) fail.
"""
from __future__ import annotations

import math

import pytest
import torch

from morph.model.slot_compact import (clamped_poisson_cdf, compact_order, put_rows,
                                      slot_depth_capacity, stratified_slot_depths, take_rows)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernel / CUDA")


# ── 1. the draw ──────────────────────────────────────────────────────────────────────────

def test_capacity_at_the_lxtul_shape():
    assert slot_depth_capacity(6.0, 8, 64) == (64, 63, 61, 55, 46, 36, 26, 17)


@pytest.mark.parametrize("mean,max_d,n", [(6.0, 8, 64), (6.0, 8, 10), (3.0, 5, 7),
                                          (2.0, 4, 50), (6.0, 8, 37)])
def test_capacity_bound_holds_and_is_attained(mean, max_d, n):
    torch.manual_seed(0)
    rows = 20000
    d = stratified_slot_depths(rows, n, mean, max_d, "cpu")
    assert d.min() >= 1 and d.max() <= max_d
    caps = slot_depth_capacity(mean, max_d, n)
    # pads: a random valid prefix per row, pads loop once (`_pad_slot_depths`)
    n_valid = torch.randint(0, n + 1, (rows, 1))
    dp = torch.where(torch.arange(n) < n_valid, d, torch.ones_like(d))
    for t in range(max_d):
        assert int((d > t).sum(1).max()) == caps[t], (t, "the bound must be attained")
        assert int((dp > t).sum(1).max()) <= caps[t], t


def test_bound_survives_thresholds_on_integers():
    """``n * P(D <= t)`` exactly an integer is the edge of the floor/ceil argument: a draw
    whose ``k + U`` rounds up to ``k + 1`` must not exceed the bound. Thresholds forced onto
    integers by hand, through the same comparison the draw makes."""
    torch.manual_seed(1)
    n, rows = 16, 50000
    g = torch.tensor([3.0, 8.0, 12.0], dtype=torch.float64)
    u = torch.rand(rows, 1, dtype=torch.float64)
    u[:100] = 1.0 - 2.0 ** -53                      # k + U rounds to k + 1 for k >= 1
    x = torch.arange(n, dtype=torch.float64) + u
    for i, gi in enumerate(g):
        assert int((x > gi).sum(1).max()) <= n - math.floor(float(gi)), i


def test_marginal_is_the_clamped_poisson_law():
    torch.manual_seed(2)
    rows, n, mean, max_d = 40000, 64, 6.0, 8
    d = stratified_slot_depths(rows, n, mean, max_d, "cpu")
    cdf = clamped_poisson_cdf(mean, max_d)
    for s in (0, 31, 63):                           # one slot at a time: the per-slot law
        for t in range(1, max_d):
            p = 1.0 - cdf[t]
            got = float((d[:, s] > t).double().mean())
            assert abs(got - p) < 4.5 * math.sqrt(p * (1 - p) / rows), (s, t, got, p)


# ── 2. row order and row moves ───────────────────────────────────────────────────────────

def test_compact_order_puts_active_rows_first_in_order():
    torch.manual_seed(3)
    act = torch.rand(5, 40) < 0.4
    perm, inv = compact_order(act)
    for b in range(5):
        a = act[b].nonzero().flatten()
        r = (~act[b]).nonzero().flatten()
        assert torch.equal(perm[b], torch.cat([a, r]))
    assert torch.equal(perm.gather(1, inv), torch.arange(40).expand(5, 40))


def test_row_moves_gradcheck():
    torch.manual_seed(4)
    act = torch.rand(3, 12) < 0.5
    act[:, 0] = True
    perm, inv = compact_order(act)
    n = int(act.sum(1).max())
    x = torch.randn(3, 12, 2, 5, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda x: take_rows(x, perm, inv, n), (x,))
    xq = torch.randn(4, 3, 12, 5, dtype=torch.float64, requires_grad=True)   # [L, B, N, C]
    torch.autograd.gradcheck(lambda x: take_rows(x, perm, inv, n, dim=2, bdim=1), (xq,))
    xc = torch.randn(3, n, 2, 5, dtype=torch.float64, requires_grad=True)
    fill = torch.randn(3, 12, 2, 5, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda a, f: put_rows(a, perm, inv, f, act), (xc, fill))
    # the round trip writes the active rows back and keeps the rest
    y = put_rows(take_rows(x, perm, inv, n), perm, inv, torch.zeros_like(x), act)
    assert torch.equal(y, torch.where(act.view(3, 12, 1, 1), x, torch.zeros_like(x)))


# ── 3. the kernel's query rows by position ───────────────────────────────────────────────

def _kernel_case(dtype, seed, qpos_override=None):
    from morph.kernels.triton.tg_strict_attention import tg_strict_attention, tg_strict_index
    g = torch.Generator(device="cuda").manual_seed(seed)
    B, H, M, S_slots, D = 3, 4, 4, 23, 32
    S = M * S_slots
    q, k, v = (torch.randn(B, H, S, D, device="cuda", generator=g).to(dtype)
               .requires_grad_() for _ in range(3))
    sink = torch.randn(H, device="cuda", generator=g).requires_grad_()
    act = (torch.rand(B, S_slots, device="cuda", generator=g) < 0.6).repeat_interleave(M, 1)
    act[:, :M] = True
    perm, inv = compact_order(act)
    n = int(act.sum(1).max()) + M                 # room for padding rows, as the plans have
    qpos = perm[:, :n].to(torch.int32).contiguous()
    idx = tg_strict_index("cells", cells=M)
    oc, ow = tg_strict_attention(q, k, v, sink, idx, 16, D ** -0.5)
    dc = torch.randn(B, H, n, D, device="cuda", generator=g).to(dtype)
    dw = torch.randn(B, H, n, D, device="cuda", generator=g).to(dtype)
    # the full kernel's rows at the query positions, upstream grads only on those rows
    sel = perm[:, None, :n, None].expand(B, H, n, D)
    ((oc.gather(2, sel).float() * dc.float()).sum()
     + (ow.gather(2, sel).float() * dw.float()).sum()).backward()
    ref = [t.grad.clone() for t in (q, k, v, sink)]
    out_ref = (oc.gather(2, sel).detach(), ow.gather(2, sel).detach())
    for t in (q, k, v, sink):
        t.grad = None
    qq = q.detach().gather(2, sel).requires_grad_()
    qidx = {**idx, "qpos": qpos if qpos_override is None else qpos_override(qpos)}
    oc2, ow2 = tg_strict_attention(qq, k, v, sink, qidx, 16, D ** -0.5)
    ((oc2.float() * dc.float()).sum() + (ow2.float() * dw.float()).sum()).backward()
    dq_full = torch.zeros_like(q).scatter(2, sel, qq.grad)
    got = [dq_full, k.grad, v.grad, sink.grad]
    return out_ref, (oc2.detach(), ow2.detach()), ref, got


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-6), (torch.bfloat16, 1e-2)])
def test_kernel_qpos_is_the_full_kernel_at_those_rows(dtype, tol):
    out_ref, out, ref, got = _kernel_case(dtype, seed=5)
    for a, b in zip(out, out_ref):
        assert _rel(a, b) <= tol, ("out", _rel(a, b))
    for name, a, b in zip(("dq", "dk", "dv", "dsink"), got, ref):
        assert _rel(a, b) <= tol, (name, _rel(a, b))


@cuda
def test_kernel_qpos_sabotage_is_seen():
    """The positive control of test 3: positions 0..n-1 instead of the cells' own."""
    out_ref, out, _, _ = _kernel_case(torch.float32, seed=5,
                                      qpos_override=lambda p: torch.arange(
                                          p.shape[1], device=p.device, dtype=torch.int32
                                      ).expand_as(p).contiguous())
    assert _rel(out[0], out_ref[0]) > 1e-2


# ── 4. the machinery at the model level ──────────────────────────────────────────────────

_KEYS = ("model.graph_safe=true", "model.tg_fused_attention=true", "model.ckpt_grad_iters=0",
         "model.slot_gain_no_ckpt=true", "model.slot_depth_stratified=true",
         "training.dropout=0.0")


def _run(monkeypatch, mode, autocast, n_batches=3, stratified=True):
    from test_graph_step import _batches, _build, _compose
    keys = _KEYS if stratified else tuple(k for k in _KEYS if "stratified" not in k)
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *keys, f"model.slot_compact={mode}")
    m, _ = _build(cfg, rt, seed=0)
    batches = _batches(rt, n_batches)
    torch.manual_seed(1234)
    res = []
    for x, y, lay in batches:
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)
        out["loss"].backward()
        res.append((out["loss"].detach().float(),
                    [None if p.grad is None else p.grad.detach().float().clone()
                     for p in m.parameters()]))
    return res


def _worst(a, b):
    """(loss relative difference, worst per-tensor gradient difference / the tensor's norm)."""
    lw, gw = 0.0, 0.0
    for (la, ga), (lb, gb) in zip(a, b):
        lw = max(lw, abs(float(la - lb)) / abs(float(lb)))
        for u, v in zip(ga, gb):
            assert (u is None) == (v is None)
            if u is not None and float(v.norm()) > 0:
                gw = max(gw, _rel(u, v))
            elif u is not None:
                assert float(u.norm()) == 0.0
    return lw, gw


@cuda
def test_gather_is_full_fp32(monkeypatch):
    full = _run(monkeypatch, "full", autocast=False)
    gath = _run(monkeypatch, "gather", autocast=False)
    lw, gw = _worst(gath, full)
    # measured 2026-10-08: loss bit-identical, median gradient bit-identical, worst 1.1e-5
    # (a core conv weight: the kernel's dK/dV sums the query tiles in another order)
    assert lw <= 1e-6 and gw <= 1e-4, (lw, gw)
    # positive control: today's loop (frozen cells recomputed) is a different function
    # (measured: loss 3e-4 relative, median gradient 8 %)
    off = _run(monkeypatch, "off", autocast=False)
    lo, go = _worst(off, full)
    assert lo > 100 * max(lw, 1e-9) and go > 1e-3, (lo, go)


@cuda
def test_gather_is_full_bf16(monkeypatch):
    full = _run(monkeypatch, "full", autocast=True)
    gath = _run(monkeypatch, "gather", autocast=True)
    lw, gw = _worst(gath, full)
    # measured 2026-10-08: loss bit-identical, worst gradient 1.5e-6
    assert lw <= 1e-6 and gw <= 1e-4, (lw, gw)


# ── 5. eval is unchanged ─────────────────────────────────────────────────────────────────

@cuda
def test_eval_forward_is_unchanged(monkeypatch):
    import dataclasses
    from test_graph_step import _batches, _build, _compose
    losses = []
    for keys in ((), ("model.slot_depth_stratified=true", "model.slot_compact=gather")):
        cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_KEYS[:-2], *keys)
        m, _ = _build(cfg, rt, seed=0)
        m.eval()
        x, y, lay = _batches(rt, 3)[1]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(m(x, labels=y, slot_layout=dataclasses.replace(
                lay, ditto_prev=None))["loss"].float())
    assert torch.equal(losses[0], losses[1]), losses


# ── 6. refusals ──────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("drop,needle", [
    ("model.slot_depth_stratified=true", "slot_depth_stratified"),
    ("model.tg_fused_attention=true", "tg_fused_attention"),
    ("model.graph_safe=true", "graph_safe"),
    ("model.ckpt_grad_iters=0", "ckpt_grad_iters"),
    ("model.slot_gain_no_ckpt=true", "slot_gain_no_ckpt"),
])
def test_refusals(monkeypatch, drop, needle):
    from test_graph_step import _compose

    from morph.model.slot_compact import validate_slot_compact
    from morph.training.train import build_morph_config
    keys = [k for k in _KEYS if k != drop] + ["model.slot_compact=gather"]
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *keys)
    with pytest.raises((NotImplementedError, ValueError), match=needle):
        validate_slot_compact(build_morph_config(cfg, tul=rt.model_cfg))


def test_the_recipe_is_accepted_and_unknown_modes_refused(monkeypatch):
    import dataclasses
    from test_graph_step import _compose

    from morph.model.slot_compact import validate_slot_compact
    from morph.training.train import build_morph_config
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_KEYS, "model.slot_compact=gather")
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    validate_slot_compact(mc)
    with pytest.raises(ValueError, match="slot_compact"):
        validate_slot_compact(dataclasses.replace(mc, slot_compact="gathered"))
