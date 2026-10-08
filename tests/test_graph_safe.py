"""``model.graph_safe`` (2026-10-07, the graph-captured training step, tasks 1.1 and 1.3 of
.agents/notes/proposed/architecture/2026-10-07-graph-captured-training-step.md).

A TRAINING forward + backward under the key must make no host read of a tensor value and
run one op sequence at one set of shapes for every batch of a config, so a CUDA graph
captured on one batch replays on any other. On CUDA the fixed-point and fan readers are
bit-identical to the off path (tests/test_graph_safe_exact.py, the identity
decomposition of 2026-10-07); the dense slot-column attention is not. Here, on CPU, where
the exact forms emulate CUDA's reduction order and not the CPU's, every such op is bounded
against an fp64 reference, at least as tightly as the off path's own error.

What each test pins:
  * `valid_first` is the gather with the live rows first, values and gradients, and
    `exact_mean_1d` is their mean (a CPU tolerance; CUDA bits in the exact file).
  * The prelude/coda slot-column attention's dense form (`dense_slot_cols`) against fp64:
    forward and q/k/v gradients, with and without the strict narrowing mask.
  * The fan's terms (`fan_stream_cos`, `TULFanMix.entropy`, the vol pass, the epi term
    with its masked `ridge_map`) against fp64, values and gradients; fewer than two valid
    slots read exactly 0 on both paths.
  * `LoopAttnCenter.apply(skip=)` leaves `mu` and `n_updates` where the host branch does.
  * The lead arm's train step (latent-selected loop, joint, router-followed, rank-only head,
    cell norm, fixed-point term, gain hinge, epivol): loss and every gradient match the off
    path within the bounds above; the op sequence and every shape are the same on three
    batches whose slot counts differ; nothing reads a tensor value on the host except the
    gain hinge's CPU draw (which needs no device and is kept on purpose) and two reads
    that exist on CPU tensors only (`_is_cpu_only_read`).

CPU, fp32, the tests/test_tul_fan_lsel.py lead-arm fixture.
"""
from __future__ import annotations

import math
import traceback

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from morph.model import host_shadow as hs
from morph.model.attention import _tg_slot_attention
from morph.model.mhc import LoopAttnCenter
from morph.model.tul_fan import (TULFanMix, FanReservoir, _fan_vol_pass, fan_epi_term,
                                 fan_stream_cos)
from test_host_shadow import _LEAD, _bare
from test_tul_fan import _batch
from test_tul_fan_lsel import _lsel
from test_tul_lx_credit import M

EPS32 = torch.finfo(torch.float32).eps


def _mask(shape, p, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(shape, generator=g) < p


def _bound(new: torch.Tensor, old: torch.Tensor, ref: torch.Tensor, what: str,
           ulps: float = 8.0) -> None:
    """``new``'s max error against the fp64 ``ref`` is at most twice ``old``'s, or a few
    fp32 ulps of the reference's scale when ``old`` happens to be closer than that."""
    ref = ref.double()
    e_new = (new.double() - ref).abs().max().item()
    e_old = (old.double() - ref).abs().max().item()
    floor = ulps * EPS32 * max(ref.abs().max().item(), 1e-30)
    assert e_new <= max(2.0 * e_old, floor), (what, e_new, e_old, floor)


# ── 1. the two helpers ────────────────────────────────────────────────────────────────


def test_valid_first_is_the_gather_with_zeroed_pads():
    m = _mask((3, 7), 0.5, seed=1)
    x = torch.randn(3, 7, 4, 5)
    xa, xb = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    rows, n = hs.valid_first(xb, m)
    k = int(m.sum())
    assert rows.shape == (21, 4, 5) and n.dim() == 0 and int(n) == k
    assert torch.equal(rows[:k], xa[m])
    assert torch.equal(rows[k:], torch.zeros_like(rows[k:]))
    g = torch.randn(21, 4, 5)
    (rows * g).sum().backward()
    (xa[m] * g[:k]).sum().backward()
    assert torch.equal(xa.grad, xb.grad)            # pads: exactly 0 gradient
    v = torch.randn(21)
    assert torch.allclose(hs.exact_mean_1d(v, n), v[:k].mean(), rtol=1e-6, atol=0)


# ── 2. the slot-column attention ──────────────────────────────────────────────────────


def _attn_ref(q, k, v, slot_mask, sink, scale, extra):
    """The gathered function in fp64, written densely: causal AND slot column AND extra."""
    B, H, S, D = q.shape
    pos = torch.arange(S, device=q.device)
    allow = (pos[None, :] <= pos[:, None])[None] & slot_mask[:, None, :]
    if extra is not None:
        allow = allow & extra.squeeze(1)
    sc = torch.einsum("bhid,bhjd->bhij", q.double(), k.double()) * scale
    sc = sc.masked_fill(~allow[:, None], float("-inf"))
    sc = torch.cat([sc, sink.double().view(1, H, 1, 1).expand(B, H, S, 1)], dim=-1)
    return torch.einsum("bhij,bhjd->bhid", torch.softmax(sc, -1)[..., :S], v.double())


@pytest.mark.parametrize("narrow", [False, True])
def test_dense_slot_columns_against_fp64(narrow):
    torch.manual_seed(3)
    B, H, S, D = 3, 2, 40, 8
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    sink = torch.randn(H)
    sm = _mask((B, S), 0.3, seed=4)
    sm[2] = False                                       # a row with no slot column at all
    extra = _mask((B, 1, S, S), 0.6, seed=5) if narrow else None
    outs, grads = [], []
    for dense in (False, True):
        qq, kk, vv = (t.clone().requires_grad_(True) for t in (q, k, v))
        o = _tg_slot_attention(qq, kk, vv, sm, sink, D ** -0.5, extra_mask=extra,
                               dense_slot_cols=dense)
        (o * torch.cos(torch.arange(o.numel()).view_as(o).float())).sum().backward()
        outs.append(o)
        grads.append((qq.grad, kk.grad, vv.grad))
    qq, kk, vv = (t.double().requires_grad_(True) for t in (q, k, v))
    ref = _attn_ref(qq, kk, vv, sm, sink, D ** -0.5, extra)
    (ref * torch.cos(torch.arange(ref.numel()).view_as(ref).double())).sum().backward()
    _bound(outs[1], outs[0], ref, "out")
    for name, gn, go, gr in zip("qkv", grads[1], grads[0], (qq.grad, kk.grad, vv.grad)):
        _bound(gn, go, gr, f"d{name}")


# ── 3. the fan's terms ────────────────────────────────────────────────────────────────


def _cells64(state, valid, m):
    b, sm = state.shape[:2]
    z = state.double().reshape(b, sm // m, m, *state.shape[2:])
    while z.dim() > 4:
        z = z.mean(dim=-2)
    return z[valid]                                     # [N, M, C] fp64


def _dev64(sel):
    dev = sel - sel.mean(dim=1, keepdim=True)
    return dev / sel.norm(dim=-1).mean(dim=1, keepdim=True).unsqueeze(-1).clamp_min(1e-6)


def _cos64(state, valid, m):
    n = F.normalize(_cells64(state, valid, m), dim=-1)
    g = n @ n.transpose(1, 2)
    return ((g.sum((1, 2)) - g.diagonal(dim1=1, dim2=2).sum(-1)) / (m * (m - 1))).mean()


def _vol64(state, valid, m, eta):
    d = _dev64(_cells64(state, valid, m))
    eye = torch.eye(m, dtype=d.dtype, device=d.device)
    ld = torch.linalg.slogdet(eye + eta * (d @ d.transpose(1, 2)))[1]
    return (0.5 * ld / math.log(2) / (m - 1)).mean()


def _epi64(traj, valid, m, n_passes, res, lam, eta):
    seed = _cells64(traj[0], valid, m).mean(dim=1)
    h = res(seed.float()).double()                      # the reservoir is fp32 on both paths
    f = h.shape[1]
    hs_ = (h - h.mean(0)) / h.std(0, unbiased=False).clamp_min(1e-6) / math.sqrt(f)
    a = torch.linalg.solve(hs_.T @ hs_ + lam * torch.eye(f, dtype=h.dtype, device=h.device), hs_.T)
    scores = []
    for t in range(1, n_passes + 1):
        d = _dev64(_cells64(traj[t], valid, m))
        per = []
        for i in range(m):
            w = a @ (d[:, i] - d[:, i].mean(0))
            per.append(0.5 * torch.linalg.slogdet(torch.eye(f, dtype=w.dtype, device=w.device)
                                                  + eta * (w @ w.T))[1] / math.log(2) / f)
        scores.append(torch.stack(per).mean())
    return -torch.stack(scores).mean()


def _fan_case(seed=0):
    torch.manual_seed(seed)
    B, S, m, n, C = 3, 8, 4, 2, 16
    valid = _mask((B, S), 0.7, seed=seed + 10)
    valid[0, 0] = True
    traj = [torch.randn(B, S * m, n, C) for _ in range(4)]
    return B, S, m, valid, traj


def _paths_and_ref(fn, ref_fn, traj, what):
    """``fn(traj, fixed)`` on both paths with grads into every traj entry; ``ref_fn`` on
    fp64 copies. Bounds the values and the gradients."""
    vals, grads = [], []
    for fixed in (False, True):
        tr = [t.clone().requires_grad_(True) for t in traj]
        y = fn(tr, fixed)
        y.backward()
        vals.append(y.detach())
        grads.append([t.grad if t.grad is not None else torch.zeros_like(t) for t in tr])
    tr64 = [t.double().requires_grad_(True) for t in traj]
    r = ref_fn(tr64)
    r.backward()
    _bound(vals[1], vals[0], r.detach(), what)
    for i, t in enumerate(tr64):
        gr = t.grad if t.grad is not None else torch.zeros_like(t)
        _bound(grads[1][i], grads[0][i], gr, f"{what} d/dtraj[{i}]")


@pytest.mark.parametrize("seed", [0, 1])
def test_fan_terms_against_fp64(seed):
    _B, _S, m, valid, traj = _fan_case(seed)
    res = FanReservoir(traj[0].shape[-1], 8, seed=3)
    _paths_and_ref(lambda tr, fx: fan_stream_cos(tr[1], valid, m, fixed=fx),
                   lambda tr: _cos64(tr[1], valid, m), traj, "cos")
    _paths_and_ref(lambda tr, fx: _fan_vol_pass(tr[2], valid, m, 30.0, fixed=fx),
                   lambda tr: _vol64(tr[2], valid, m, 30.0), traj, "vol")
    _paths_and_ref(lambda tr, fx: fan_epi_term(tr, valid, m, 2, res, 3.0, 30.0, fixed=fx),
                   lambda tr: _epi64(tr, valid, m, 2, res, 3.0, 30.0), traj, "epi")
    w = torch.softmax(torch.randn(*valid.shape, m), -1)
    ref = (-(w.double() * w.double().log()).sum(-1))[valid].mean()
    _bound(TULFanMix.entropy(w, valid, fixed=True), TULFanMix.entropy(w, valid), ref,
           "entropy")


@pytest.mark.parametrize("n_valid", [0, 1])
def test_fan_terms_read_zero_below_two_slots(n_valid):
    _B, _S, m, valid, traj = _fan_case(0)
    valid = torch.zeros_like(valid)
    valid.view(-1)[:n_valid] = True
    res = FanReservoir(traj[0].shape[-1], 8, seed=3)
    for fixed in (False, True):
        e = fan_epi_term(traj, valid, m, 2, res, 3.0, 30.0, fixed=fixed)
        assert e.item() == 0.0, (fixed, e)
        if n_valid == 0:
            assert fan_stream_cos(traj[1], valid, m, fixed=fixed).item() == 0.0
            assert _fan_vol_pass(traj[1], valid, m, 30.0, fixed=fixed).item() == 0.0


# ── 4. the loop-attention center's device skip ────────────────────────────────────────


@pytest.mark.parametrize("pattern", [(True, False, True), (False, True, False, False)])
def test_loop_attn_center_device_skip_equals_the_host_branch(pattern):
    torch.manual_seed(5)
    a, b = LoopAttnCenter(6, 0.9), LoopAttnCenter(6, 0.9)
    for empty in pattern:
        rec = torch.randn(6) if not empty else torch.zeros(6)
        for c in (a, b):
            c.snapshot()
            c._acc = [rec.clone()]
        a.apply(frozen=empty)
        b.apply(skip=torch.tensor(empty))
        assert torch.equal(a.mu, b.mu) and torch.equal(a.n_updates, b.n_updates)


# ── 5. the lead arm's train step ──────────────────────────────────────────────────────


def _model(graph_safe: bool, dropout: float = 0.1):
    m = _lsel("joint", model_kw={"dropout": dropout, "graph_safe": graph_safe},
              **{k: v for k, v in _LEAD.items() if k != "fan_loop_select"}).train()
    m._train_instruments = False          # the trainer's non-logging step: the captured one
    return m


def _step(m, seed: int):
    _ids, inp, lab, layout = _batch(M, seed=seed)
    m.zero_grad(set_to_none=True)
    torch.manual_seed(7)
    out = m(inp, labels=lab, slot_layout=_bare(layout).to("cpu"))
    out["loss"].backward()
    return out, {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("seed", [0, 1])
def test_train_step_matches_the_off_path(seed):
    oa, ga = _step(_model(False), seed)
    ob, gb = _step(_model(True), seed)
    assert oa.keys() == ob.keys()
    assert torch.allclose(oa["loss"], ob["loss"], rtol=4 * EPS32, atol=0)
    for k in oa:
        if torch.is_tensor(oa[k]) and oa[k].is_floating_point():
            assert torch.allclose(oa[k], ob[k], rtol=1e-5, atol=1e-6), k
    assert ga.keys() == gb.keys() and len(ga) > 50
    for n in ga:
        scale = ga[n].abs().max().item()
        assert (ga[n] - gb[n]).abs().max().item() <= 1e-4 * scale + 1e-9, n


class _OpLog(TorchDispatchMode):
    """Every aten op with its tensor arguments' shapes; and every host read of a tensor
    value (`_local_scalar_dense` is `.item()` / `bool()` / `int()`, `nonzero` the
    data-dependent shape) with the Python line that asked for it."""

    def __init__(self):
        super().__init__()
        self.ops: list = []
        self.reads: list = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket)
        shapes = tuple(tuple(a.shape) for a in args if torch.is_tensor(a))
        self.ops.append((name, shapes))
        if name in ("aten._local_scalar_dense", "aten.nonzero", "aten.masked_select"):
            self.reads.append((name, traceback.format_stack(limit=12)))
        return func(*args, **(kwargs or {}))


def _logged_step(graph_safe: bool, seed: int):
    m = _model(graph_safe)
    numpy_calls = []
    _orig = torch.Tensor.numpy

    def _numpy(self, *a, **k):
        numpy_calls.append(traceback.format_stack(limit=6))
        return _orig(self, *a, **k)

    _ids, inp, lab, layout = _batch(M, seed=seed)
    layout = _bare(layout).to("cpu")
    log = _OpLog()
    torch.manual_seed(7)
    torch.Tensor.numpy = _numpy
    try:
        with log:
            out = m(inp, labels=lab, slot_layout=layout)
            out["loss"].backward()
    finally:
        torch.Tensor.numpy = _orig
    return log, numpy_calls, layout


def _is_hinge_draw(stack: list[str]) -> bool:
    return any("torch.randint(n_grad_iters" in fr for fr in stack)


def _is_cpu_only_read(stack: list[str]) -> bool:
    """Reads that exist on a CPU tensor only, so this CPU test sees them and a CUDA step
    does not: `F.one_hot(x, n)` validates the range of ``x`` with `.item()` on CPU and
    skips the check on CUDA when ``n`` is given (the latent-selected loop's share reading,
    its exit loss, the routed write); `fused_linear_cross_entropy` divides by a host scalar
    on CPU and by a device reciprocal on CUDA (`fused_ce.py`, `if x.is_cuda else`). Both
    are absent from the 2026-10-07 census of the lead arm's CUDA step, and the sync-debug
    run of this key confirms it."""
    return any("F.one_hot(" in fr or "if x.is_cuda else float(n_valid_t.item())" in fr
               for fr in stack)


def _is_allowed(stack: list[str]) -> bool:
    return _is_hinge_draw(stack) or _is_cpu_only_read(stack)


def test_fixed_op_sequence_and_shapes_across_batches():
    logs, n_slots = [], []
    for seed in (0, 1, 2):
        log, _np, layout = _logged_step(True, seed)
        logs.append(log.ops)
        n_slots.append((int(layout.slot_valid.sum()), int(layout.slot_mask.sum(-1).max())))
    # The fixture must actually vary what the off path keys its shapes on.
    assert len(set(n_slots)) > 1, n_slots
    for i in (1, 2):
        assert len(logs[i]) == len(logs[0]), (i, len(logs[i]), len(logs[0]))
        first = next((j for j, (x, y) in enumerate(zip(logs[0], logs[i])) if x != y), None)
        assert first is None, (i, first, logs[0][first], logs[i][first])
    # ...and the off path does change shape on the same batches (the check can see it).
    off = [_logged_step(False, seed)[0].ops for seed in (0, 1)]
    assert off[0] != off[1]


def test_no_host_read_of_a_tensor_value():
    log, numpy_calls, _layout = _logged_step(True, 0)
    other = [s for name, s in log.reads if not _is_allowed(s)]
    assert not other, "".join(other[0])
    assert not numpy_calls, "".join(numpy_calls[0])
    # The gain hinge's draw is the one read left: a CPU generator, no device involved.
    assert sum(_is_hinge_draw(s) for _n, s in log.reads) == 1
    # The off path's reads (the depth table, the shadow gathers) are seen by the check.
    log_off, numpy_off, _ = _logged_step(False, 0)
    assert numpy_off or [s for _n, s in log_off.reads if not _is_allowed(s)]
