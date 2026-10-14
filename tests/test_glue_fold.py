"""model.glue_fold: every fold is exact or rounding-only against the unfolded kernels.

Each contract is pinned against the code it replaces, on CPU (the folds are dtype / layout
rewrites that do not depend on the device). The sabotage check for the main test: make
`_pad_cast` leave the pad rows uninitialised (drop the `zero_`) and
`test_pad_cast_is_bit_identical_to_pad_then_cast` fails on the odd-vocab cases.
"""
from __future__ import annotations

import pytest
import torch

from morph.model.fused_ce import (
    _gemm_operands,
    _pad_cast,
    _pad_vocab,
    fused_linear_cross_entropy,
    fused_linear_label_logprob,
)
from morph.model.transformer import MORPHConfig


@pytest.mark.parametrize("V", [49169, 100, 128, 192, 7])
@pytest.mark.parametrize("src,dst", [(torch.float32, torch.bfloat16),
                                     (torch.bfloat16, torch.bfloat16),
                                     (torch.float32, torch.float32)])
def test_pad_cast_is_bit_identical_to_pad_then_cast(V, src, dst):
    g = torch.Generator().manual_seed(V)
    w = torch.randn(V, 24, generator=g).to(src)
    ref, rV, rVp = _pad_vocab(w)
    ref = ref.to(dst)
    out, oV, oVp = _pad_cast(w, dst)
    assert (oV, oVp) == (rV, rVp)
    assert out.dtype == dst and out.shape == ref.shape and out.is_contiguous()
    assert torch.equal(out.view(torch.int16 if dst == torch.bfloat16 else torch.int32),
                       ref.view(torch.int16 if dst == torch.bfloat16 else torch.int32))
    if oVp != oV:
        assert not out[oV:].any(), "the pad rows must be exactly zero"


@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_gemm_operands_fold_matches_unfolded(x_dtype):
    g = torch.Generator().manual_seed(3)
    x = torch.randn(8, 24, generator=g).to(x_dtype)
    w = torch.randn(193, 24, generator=g)
    a = _gemm_operands(x, w, False)
    b = _gemm_operands(x, w, True)
    assert a[2:] == b[2:]
    assert torch.equal(a[0], b[0])
    assert a[1].dtype == b[1].dtype and torch.equal(a[1], b[1])


def test_cross_entropy_fold_head_matches_unfolded_loss_and_grads():
    g = torch.Generator().manual_seed(5)
    N, d, V = 37, 24, 201                                   # odd V exercises the pad
    x0 = torch.randn(N, d, generator=g)
    w0 = torch.randn(V, d, generator=g) * 0.3
    lab = torch.randint(0, V, (N,), generator=g)
    lab[::5] = -100
    outs = []
    for fold in (False, True):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        loss = fused_linear_cross_entropy(x, w, lab, chunk_size=16, fold_head=fold)
        loss.backward()
        outs.append((loss.detach(), x.grad, w.grad))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_label_logprob_fold_head_matches_unfolded():
    g = torch.Generator().manual_seed(6)
    N, d, V = 21, 24, 130
    x0 = torch.randn(N, d, generator=g)
    w0 = torch.randn(V, d, generator=g) * 0.3
    lab = torch.randint(0, V, (N,), generator=g)
    lab[::4] = -100
    outs = []
    for fold in (False, True):
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        lp = fused_linear_label_logprob(x, w, lab, chunk_size=8, fold_head=fold)
        (lp * torch.linspace(0.5, 1.5, N)).sum().backward()
        outs.append((lp.detach(), x.grad, w.grad))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_glue_fold_defaults_off():
    assert MORPHConfig.__dataclass_fields__["glue_fold"].default is False


# ── the compiled folds (CUDA: `torch.compile` of a CUDA function, like test_loop_glue) ────────

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def _both(fn_a, fn_b, inputs, g):
    """Run two functions on clones of ``inputs`` (leaf tensors that require grad), return the
    outputs and the gradients of ``sum(out * g)`` for each."""
    res = []
    for fn in (fn_a, fn_b):
        xs = [t.detach().clone().requires_grad_(t.requires_grad) for t in inputs]
        out = fn(*xs)
        outs = out if isinstance(out, tuple) else (out,)
        loss = sum((o.float() * gg).sum() for o, gg in zip(outs, g))
        grads = torch.autograd.grad(loss, [t for t in xs if t.requires_grad])
        res.append(([o.detach() for o in outs], list(grads)))
    return res


@cuda
def test_pointer_mass_is_the_attend_read_and_compiles_within_rounding():
    from morph.model.glue_fold import compile_glue
    from morph.model.tul_pointer import TULPointer, pointer_mass, token_order
    torch.manual_seed(0)
    B, L, C, H, dh = 2, 48, 64, 4, 16
    ptr = TULPointer(C, H, dh=dh).cuda()
    with torch.no_grad():                           # the init is near-uniform; make it sharp
        ptr.q.weight.normal_(std=0.3)
        ptr.k.weight.normal_(std=0.3)
    h = torch.randn(B, L, C, device="cuda")
    slot_mask = torch.rand(B, L, device="cuda") < 0.25
    labels = torch.randint(0, 6, (B, L), device="cuda")        # few ids: many `same` pairs
    from morph.model.tul_pointer import candidates_from_labels
    cand = candidates_from_labels(labels, slot_mask)
    # (1) the eager function IS the old attend() + same-mask read (the trainer's autocast)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        alpha, null, _gl, _cov = ptr.attend(h, cand, slot_mask)
        same = (cand[:, :, None] == cand[:, None, :]) & (cand[:, :, None] >= 0)
        p_ref = (alpha * same[:, None].to(alpha.dtype)).sum(-1).transpose(1, 2)
        order = token_order(slot_mask)
        ht = h.gather(1, order[..., None].expand(-1, -1, C))
        q = ptr.q(ht).view(B, L, H, dh).transpose(1, 2)
        k = ptr.k(ht).view(B, L, H, dh).transpose(1, 2)
    scale = (ptr.log_scale_bias.exp() / dh ** 0.5).float()
    n_tok = (~slot_mask).sum(1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        p, nl = pointer_mass(q, k, cand, n_tok, scale, ptr.null_k_bias)
    assert torch.equal(p, p_ref) and torch.equal(nl, null)
    assert float(p.detach().sum()) > 0.0 and float(p.detach().max()) > 0.05      # the fixture reads real mass
    # (2) compiled vs eager, value and every gradient
    g = [torch.randn_like(p), torch.randn_like(nl)]
    inputs = [q.detach().float().contiguous().requires_grad_(), k.detach().float().contiguous()
              .requires_grad_(), cand, n_tok, scale.detach().clone().requires_grad_(),
              ptr.null_k_bias.detach().clone().requires_grad_()]
    (oe, ge), (oc, gc) = _both(pointer_mass, compile_glue(pointer_mass, True), inputs, g)
    for a, b in zip(oc, oe):
        assert _rel(a, b) <= 1e-4, _rel(a, b)
    for a, b in zip(gc, ge):
        assert _rel(a, b) <= 1e-3, _rel(a, b)


@cuda
def test_lm_table_compiles_within_rounding_and_matches_fp64():
    from morph.model.embeddings import lm_table
    from morph.model.glue_fold import compile_glue
    torch.manual_seed(1)
    V = 1001
    euc = (torch.randn(V, 48, device="cuda") * 0.05).requires_grad_()
    space = (torch.randn(V, 16, device="cuda") * 0.3).requires_grad_()
    g = [torch.randn(V, 64, device="cuda")]
    (oe, ge), (oc, gc) = _both(lm_table, compile_glue(lm_table, False), [euc, space], g)
    ref = lm_table(euc.double(), space.double())
    assert _rel(oc[0], ref) <= 1e-5 and _rel(oc[0], ref) <= 2 * _rel(oe[0], ref) + 1e-7
    for a, b in zip(gc, ge):
        assert _rel(a, b) <= 1e-4, _rel(a, b)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_injection_term_and_diag_inject_compile_within_rounding(dtype):
    from morph.model.glue_fold import compile_glue, diag_inject, injection_term
    torch.manual_seed(2)
    B, S, C, cs, ce = 3, 20, 64, 8, 40
    big = torch.randn(B, S, C, device="cuda", dtype=dtype)
    lam = torch.tensor(0.7, device="cuda").requires_grad_()
    ctx = torch.randn(B, S, ce - cs, device="cuda", dtype=dtype)
    tf = injection_term(cs, ce)
    (oe, ge), (oc, gc) = _both(tf, compile_glue(tf, True), [big.requires_grad_(), lam, ctx.requires_grad_()],
                               [torch.randn(B, S, C, device="cuda")])
    tol = 1e-6 if dtype == torch.float32 else 1e-2
    assert _rel(oc[0], oe[0]) <= tol
    for a, b in zip(gc, ge):
        assert _rel(a, b) <= max(tol, 1e-5)
    h = torch.randn(B, S, 4, C, device="cuda", dtype=dtype)
    e_ctx = torch.randn(B, S, 1, ce - cs, device="cuda", dtype=dtype)
    log_A = torch.randn(ce - cs, device="cuda").mul(0.1).add(-0.8).requires_grad_()
    log_dt = torch.randn(ce - cs, device="cuda").mul(0.1).requires_grad_()
    df = diag_inject(cs, ce, dtype != torch.float32)
    (oe, ge), (oc, gc) = _both(df, compile_glue(df, True),
                               [h.requires_grad_(), e_ctx.requires_grad_(), log_A, log_dt],
                               [torch.randn(B, S, 4, C, device="cuda")])
    assert _rel(oc[0], oe[0]) <= tol
    for a, b in zip(gc, ge):
        assert _rel(a, b) <= max(tol, 1e-4), _rel(a, b)


@cuda
def test_diagonal_injection_module_folds_to_the_same_map():
    from morph.model.glue_fold import compile_glue, diag_inject
    from morph.model.transformer import DiagonalInjection
    torch.manual_seed(3)
    inj = DiagonalInjection(8, 40).cuda()
    inj.keep_dtype = True
    h = torch.randn(2, 12, 4, 64, device="cuda", dtype=torch.bfloat16)
    e = torch.randn(2, 12, 1, 64, device="cuda", dtype=torch.bfloat16)
    ref = inj(h, e)
    inj.fold_glue(compile_glue(diag_inject(8, 40, True), True))
    out = inj(h, e)
    assert out.dtype == ref.dtype == torch.bfloat16
    assert _rel(out, ref) <= 1e-2
    assert torch.equal(out[..., :8], h[..., :8]) and torch.equal(out[..., 40:], h[..., 40:])


@cuda
def test_span_decoder_folds_to_the_same_states():
    from morph.model.glue_fold import compile_glue, rms_norm_cast, rms_norm_fp32, silu_mul
    from morph.model.tul_spandec import SpanDecoder, SpanFold
    torch.manual_seed(4)
    dec = SpanDecoder(64, 4, 128, n_layers=2, max_tokens=8, reads_cells=True).cuda()
    with torch.no_grad():
        for blk in dec.blocks:
            blk.cross.proj.weight.normal_(std=0.05)     # zero-init: the cross term is off
    x = torch.randn(6, 8, 64, device="cuda", dtype=torch.bfloat16)
    mem = torch.randn(6, 3, 64, device="cuda", dtype=torch.bfloat16)

    def run():
        y = x.clone().float().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = y.to(torch.bfloat16)
            for blk in dec.blocks:
                h = blk(h, mem)
            out = dec._out_norm(h)
        (out * torch.linspace(0.5, 1.5, 64, device="cuda")).sum().backward()
        return out.detach(), y.grad.float()

    ref_out, ref_g = run()
    dec.fold_glue(SpanFold(rms_cast=compile_glue(rms_norm_cast, True),
                           rms=compile_glue(rms_norm_fp32, True),
                           silu_mul=compile_glue(silu_mul, True)))
    out, g = run()
    assert out.dtype == ref_out.dtype
    assert _rel(out, ref_out) <= 2e-2 and _rel(g, ref_g) <= 5e-2


@cuda
def test_ce_head_fold_is_bit_identical_on_the_kernel_path():
    """The Triton-softmax path (the speed base's) pads and casts the head weight in one write."""
    torch.manual_seed(5)
    N, d, V = 70, 64, 1001
    x0 = torch.randn(N, d, device="cuda") * 0.5
    w0 = torch.randn(V, d, device="cuda") * 0.3
    lab = torch.randint(0, V, (N,), device="cuda")
    lab[::6] = -100
    res = {}
    for fold in (False, True):
        for name in ("label_logprob", "cross_entropy"):
            x = x0.clone().requires_grad_()
            w = w0.clone().requires_grad_()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if name == "label_logprob":
                    out = fused_linear_label_logprob(x, w, lab, chunk_size=32,
                                                     softmax_kernel=True, fold_head=fold)
                    loss = (out * torch.linspace(0.5, 1.5, N, device="cuda")).sum()
                else:
                    out = fused_linear_cross_entropy(x, w, lab, chunk_size=32,
                                                     softmax_kernel=True, fold_head=fold)
                    loss = out
            loss.backward()
            res[(fold, name)] = (out.detach(), x.grad, w.grad)
    for name in ("label_logprob", "cross_entropy"):
        for a, b in zip(res[(False, name)], res[(True, name)]):
            assert torch.equal(a, b), name


# ── refusals and the model-level step ───────────────────────────────────────────────────────

def test_pointer_fold_refuses_the_coverage_and_cell_key_reads():
    from morph.model.tul_pointer import TULPointer
    with pytest.raises(ValueError, match="coverage"):
        TULPointer(32, 2, dh=8, coverage=True).fold_glue(lambda *a: None)
    with pytest.raises(ValueError, match="cell_key"):
        TULPointer(32, 2, dh=8, cell_key=True).fold_glue(lambda *a: None)


def test_diagonal_injection_fold_refuses_the_B_map():
    from morph.model.transformer import DiagonalInjection
    with pytest.raises(ValueError, match="injection_B"):
        DiagonalInjection(4, 12, use_B=True).fold_glue(lambda *a: None)


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
    return out["loss"].detach().float(), grads, m


@cuda
def test_model_with_glue_fold_matches_the_eager_step(monkeypatch):
    l0, g0, _ = _step(monkeypatch)
    l1, g1, m1 = _step(monkeypatch, "model.glue_fold=true")
    # the fold reached the modules it names
    assert m1.tul_pointer._mass_fn is not None and m1.injection._fold_fn is not None
    assert m1.tul_spandec._fold is not None and m1._inj_term_fn is not None
    assert m1.embed.hybrid._lm_table_fn is not lm_table_eager()
    assert abs(float(l1 - l0)) / abs(float(l0)) <= 1e-3, (float(l0), float(l1))
    num = sum(float(((a - b).double() ** 2).sum()) for a, b in zip(g1, g0) if a is not None)
    den = sum(float((b.double() ** 2).sum()) for b in g0 if b is not None)
    assert (num / den) ** 0.5 <= 2e-2, (num / den) ** 0.5


def lm_table_eager():
    from morph.model.embeddings import lm_table
    return lm_table
