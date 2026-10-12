"""Slot-loop glue (2026-10-11): the row moves, the injection select and the compiled norms.

  1. ``take_rows`` / ``put_rows`` (gather and scatter of the pass's ``n`` positions) equal the
     full-permutation form they replaced, in value and in every gradient, ``torch.equal``.
  2. ``unbind`` gives the same gradient as ``stack[i]`` per layer, ``torch.equal`` (what
     ``_apply_core_step`` does with ``inj_terms``).
  3. ``model.loop_glue_compile``: the compiled RMS norm equals the eager one inside the fp32
     rounding of one reduction (value and both gradients, against an fp64 reference too); the
     key is refused on a model with no slot loop.
  4. A model with the key trains one step within 1e-3 of the model without it.

Sections 1 and 2 run on the CPU. 3 and 4 need CUDA (``torch.compile`` of a CUDA function).
"""
import pytest
import torch

from morph.model.slot_compact import compact_order, put_rows, take_rows

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ── 1. row moves against the full-permutation form ───────────────────────────────────────

def _inverse(perm):
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(perm.shape[1]).expand_as(perm))
    return inv


def _view(idx, like, dim, bdim):
    shape = [1] * like.dim()
    shape[bdim], shape[dim] = idx.shape[0], idx.shape[1]
    out = list(like.shape)
    out[dim] = idx.shape[1]
    return idx.view(shape).expand(out)


def _ref_take(x, perm, n, dim=1, bdim=0):
    """The form before 2026-10-11: gather by the FULL permutation, keep the first n rows."""
    return x.gather(dim, _view(perm, x, dim, bdim)).narrow(dim, 0, n)


def _ref_put(x_c, perm, fill, keep, dim=1, bdim=0):
    """The form before 2026-10-11: zero-pad to N rows, gather by the inverse, ``where``."""
    n_all = perm.shape[1]
    pad = list(x_c.shape)
    pad[dim] = n_all - x_c.shape[dim]
    full = torch.cat([x_c, x_c.new_zeros(pad)], dim=dim)
    back = full.gather(dim, _view(_inverse(perm), full, dim, bdim))
    kshape = [1] * fill.dim()
    kshape[bdim], kshape[dim] = keep.shape
    return torch.where(keep.view(kshape), back.to(fill.dtype), fill)


def _case(seed):
    torch.manual_seed(seed)
    act = torch.rand(4, 20) < 0.45
    act[:, 0] = True
    perm = compact_order(act)
    n = int(act.sum(1).max()) + 2                     # room for padding rows, as the plans have
    return act, perm, n


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_take_rows_equals_the_full_permutation_form(seed):
    act, perm, n = _case(seed)
    for shape, dim, bdim in (((4, 20, 3, 5), 1, 0), ((6, 4, 20, 5), 2, 1)):
        x = torch.randn(*shape, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a = take_rows(x, perm, n, dim=dim, bdim=bdim)
        b = _ref_take(y, perm, n, dim=dim, bdim=bdim)
        assert torch.equal(a, b)
        g = torch.randn_like(a)
        (a * g).sum().backward()
        (b * g).sum().backward()
        assert torch.equal(x.grad, y.grad)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_put_rows_equals_the_full_permutation_form(seed):
    act, perm, n = _case(seed)
    xc = torch.randn(4, n, 3, 5, requires_grad=True)
    fill = torch.randn(4, 20, 3, 5, requires_grad=True)
    xc2 = xc.detach().clone().requires_grad_()
    fill2 = fill.detach().clone().requires_grad_()
    a = put_rows(xc, perm, fill, act)
    b = _ref_put(xc2, perm, fill2, act)
    assert torch.equal(a, b)
    g = torch.randn_like(a)
    (a * g).sum().backward()
    (b * g).sum().backward()
    assert torch.equal(xc.grad, xc2.grad)
    assert torch.equal(fill.grad, fill2.grad)


def test_put_rows_with_the_callers_fill_rows_is_the_same_function():
    """`fill_c` hands over ``take_rows(fill)`` the caller already holds: same value, same
    gradients as the call that gathers it itself."""
    act, perm, n = _case(3)
    xc = torch.randn(4, n, 3, 5, requires_grad=True)
    fill = torch.randn(4, 20, 3, 5, requires_grad=True)
    out_a = put_rows(xc, perm, fill, act)
    ga = torch.autograd.grad((out_a * out_a).sum(), (xc, fill))
    fc = take_rows(fill, perm, n)
    out_b = put_rows(xc, perm, fill, act, fill_c=fc)
    gb = torch.autograd.grad((out_b * out_b).sum(), (xc, fill))
    assert torch.equal(out_a, out_b)
    for u, v in zip(ga, gb):
        assert torch.equal(u, v)


def test_put_rows_sabotage_is_seen():
    """Positive control: writing every one of the first n ranks (ignoring ``keep``) differs."""
    act, perm, n = _case(4)
    assert int(act.sum(1).min()) < n
    xc = torch.randn(4, n, 3, 5)
    fill = torch.randn(4, 20, 3, 5)
    good = put_rows(xc, perm, fill, act)
    all_rows = put_rows(xc, perm, fill, torch.ones_like(act))
    assert not torch.equal(good, all_rows)


# ── 2. the injection select ──────────────────────────────────────────────────────────────

def test_unbind_gradient_equals_the_select_gradient():
    torch.manual_seed(7)
    stack = torch.randn(6, 3, 8, 5, requires_grad=True)
    stack2 = stack.detach().clone().requires_grad_()
    w = [torch.randn(3, 8, 5) for _ in range(6)]
    sel = sum((stack[i] * w[i]).sum() for i in range(6))
    sel.backward()
    parts = stack2.unbind(0)
    ub = sum((parts[i] * w[i]).sum() for i in range(6))
    ub.backward()
    assert torch.equal(stack.grad, stack2.grad)


# ── 3. the compiled norm ─────────────────────────────────────────────────────────────────

def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_rms_norm_matches_eager_and_fp64(dtype):
    from torch.func import functional_call

    from morph.model.attention import RMSNorm
    from morph.model.loop_glue import compile_rms_norm, rms_norm_fp32
    torch.manual_seed(0)
    norm = RMSNorm(256).cuda()
    with torch.no_grad():
        norm.weight.copy_(torch.rand(256) + 0.5)
    fn = compile_rms_norm()
    x0 = (torch.randn(6, 24, 4, 256) * 3).cuda().to(dtype)
    g = torch.randn(6, 24, 4, 256, device="cuda")
    forms = {
        "module": lambda x, w: functional_call(norm, {"weight": w}, (x,)),
        "same_math": lambda x, w: rms_norm_fp32(x, w, norm.eps),
        "compiled": lambda x, w: fn(x, w, norm.eps),
    }
    outs, grads = {}, {}
    for name, f in forms.items():
        x = x0.clone().requires_grad_()
        w = norm.weight.detach().clone().requires_grad_()
        y = f(x, w)
        (y * g).sum().backward()
        outs[name], grads[name] = y.detach(), (x.grad.float(), w.grad)
    assert torch.equal(outs["module"], outs["same_math"])      # the function IS the module's
    # fp64 reference of the module's formula
    x64 = x0.double().requires_grad_()
    w64 = norm.weight.detach().double().requires_grad_()
    r = (x64 * (x64.pow(2).mean(-1, keepdim=True) + norm.eps).rsqrt()) * w64
    (r * g.double()).sum().backward()
    tol = 1e-5 if dtype == torch.float32 else 5e-3
    assert _rel(outs["compiled"], r.detach()) <= tol
    assert _rel(grads["compiled"][0], x64.grad) <= tol
    assert _rel(grads["compiled"][1], w64.grad) <= 1e-4
    # the compiled form is no worse than the module against the same reference
    assert _rel(outs["compiled"], r.detach()) <= 2 * _rel(outs["module"], r.detach()) + 1e-7


@cuda
def test_compiled_rms_norm_sabotage_is_seen():
    """Positive control: a norm with the wrong eps differs by more than the tolerance."""
    from morph.model.loop_glue import compile_rms_norm
    fn = compile_rms_norm()
    x = torch.randn(6, 24, 4, 256, device="cuda") * 1e-3
    w = torch.ones(256, device="cuda")
    assert _rel(fn(x, w, 1e-6), fn(x, w, 1e-2)) > 1e-2


def test_loop_glue_compile_is_refused_without_a_slot_loop(monkeypatch):
    import dataclasses

    from test_graph_step import _compose

    from morph.model.transformer import MORPHTransformer
    from morph.training.train import build_morph_config
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto")
    mc = build_morph_config(cfg, tul=rt.model_cfg)
    with pytest.raises(ValueError, match="loop_glue_compile"):
        MORPHTransformer(dataclasses.replace(mc, tul=None, loop_glue_compile=True))


# ── 4. a model with the key ──────────────────────────────────────────────────────────────

_KEYS = ("model.graph_safe=true", "model.tg_fused_attention=true", "model.ckpt_grad_iters=0",
         "model.slot_gain_no_ckpt=true", "model.slot_depth_stratified=true",
         "model.slot_compact=gather", "training.dropout=0.0")


def _step(monkeypatch, *extra):
    from test_graph_step import _batches, _build, _compose
    cfg, rt = _compose(monkeypatch, "lxtul_pointer_ditto", *_KEYS, *extra)
    m, _ = _build(cfg, rt, seed=0)
    x, y, lay = _batches(rt, 3)[1]
    torch.manual_seed(1234)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(x, labels=y, bag_size=0, slot_layout=lay, tul_step_mode=None)
    out["loss"].backward()
    grads = [None if p.grad is None else p.grad.detach().float().clone()
             for p in m.parameters()]
    return out["loss"].detach().float(), grads, out


def _global_grad_error(g1, g0):
    """||g1 - g0|| / ||g0|| over every parameter's gradient at once."""
    num = sum(float(((a - b).double() ** 2).sum()) for a, b in zip(g1, g0) if a is not None)
    den = sum(float((b.double() ** 2).sum()) for b in g0 if b is not None)
    return (num / den) ** 0.5


# Measured 2026-10-12 (this fixture, one step): the compiled norm moves the worst gradient
# tensor by 5e-3 on the prelude's attention parameters, because Inductor fuses
# `(x * norm).to(bf16) * weight` without the eager bf16 rounding between the two products. The
# compiled form is the closer to the fp32 math.
@cuda
def test_model_with_loop_glue_compile_matches_the_eager_loop(monkeypatch):
    key = "model.loop_glue_compile=true"
    l0, g0, o0 = _step(monkeypatch)
    l0b, g0b, _ = _step(monkeypatch)
    l1, g1, o1 = _step(monkeypatch, key)
    assert abs(float(l1 - l0)) / abs(float(l0)) <= 1e-3, (float(l0), float(l1))
    # two eager runs agree to the noise floor of this fixture; the key may add 1e-2
    assert _global_grad_error(g0b, g0) <= 1e-6
    assert _global_grad_error(g1, g0) <= 1e-2, _global_grad_error(g1, g0)
    # the hinge probes the raw map: its reading does not move with the norm's placement
    assert abs(float(o1["gain_est"]) - float(o0["gain_est"])) <= 5e-3 * abs(float(o0["gain_est"]))
