"""model.block_custom_ops: the HC region entry / exit, the CCA conv and the row-tiled CCA
prologue as ``torch.library`` ops (``morph/kernels/triton/block_ops.py``).

What each test pins (2026-10-12):
  1. EAGER BITS. Every op's (HC entry / exit, CCA conv, segment-reset conv, row-tiled
     prologue) forward output and every input gradient equal the old autograd
     Function's BIT FOR BIT in eager, fp32 and bf16 inputs (``torch.equal``): the op body is the
     Function's body (shared ``*_impl``), so any difference is a wiring error (a dropped grad,
     a wrong saved tensor, a lost fold). HC: through ``HyperConnectionResidual`` with and without
     the folded inject term and on an fp32 and a bf16 carrier. The HC link (the entry's
     ``link`` output and the exit's gout-to-entry edge) is covered by the carrier grad: without
     the fold the Hres grad has no gout and ``grad_h`` / ``grad_proj_w`` differ.
  2. NO GRAPH BREAK. ``torch.compile(fullgraph=True)`` of a function that runs entry, a linear,
     exit, conv and prologue compiles with the key on, and RAISES with the fenced Functions
     (``torch.compiler.disable`` is a graph break): the claim of the key, measured.
  3. COMPILED CLOSE. The compiled key-on forward and gradients agree with the eager fenced
     path to the rounding of the casts inductor fuses (bf16 eps).
  4. REFUSALS. The key without ``hc_region_fused``, or the op prologue without ``tiled``, raises.
     Default off.

Sabotage checks (run and reverted, session report): (a) ``_exit_backward`` returning ``g_out *
0`` for the link -> test 1 HC cases fail (grad_h, grad_proj_w); (b) ``_rows_backward`` swapping
the q_conv and k_conv grads -> test 1 prologue fails; (c) ``cca_conv_op`` scaling ``w_dw`` by 1.0001
-> test 1 conv fails (fp32 cases; the bf16 rounding hides it).

CUDA only (Triton kernels).
"""
import pytest
import torch

from morph.kernels.triton.block_ops import cca_conv_op, cca_prologue_rows_op, seg_conv_op
from morph.kernels.triton.fused_cca_conv import _FusedCCAConv
from morph.kernels.triton.fused_seg_causal_conv import _FusedSegCausalConv
from morph.kernels.triton.fused_cca_prologue import _FusedCCAPrologueRows, fused_cca_prologue
from morph.model.hyper_connections import HyperConnectionResidual
from morph.model.transformer import MORPHConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

C = 1024
H, HKV, D = 8, 4, 64


def _same(a, b, name):
    assert a.dtype == b.dtype and a.shape == b.shape, (name, a.dtype, b.dtype, a.shape, b.shape)
    assert torch.equal(a, b), (name, (a.double() - b.double()).abs().max().item())


# ── 1. eager bits ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S", [(3, 128), (2, 37)])
def test_cca_conv_op_equals_function(dtype, B, S):
    G, CG, Kk = 6, 64, 4
    Cc = G * CG
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(B, S, Cc, device="cuda", generator=g).to(dtype)
    w_dw0 = torch.randn(Cc, 1, Kk, device="cuda", generator=g).to(dtype)
    w_gp0 = torch.randn(Cc, CG, Kk, device="cuda", generator=g).to(dtype) / 8
    r = torch.randn(B, Cc, S, device="cuda", generator=g)

    def run(fn):
        x = x0.clone().requires_grad_(True)      # [B, S, C] -> transposed view, as the model
        wd = w_dw0.clone().requires_grad_(True)
        wg = w_gp0.clone().requires_grad_(True)
        y = fn(x.transpose(1, 2), wd, wg, CG, Kk)
        (y.float() * r).sum().backward()
        return y.detach(), [x.grad, wd.grad, wg.grad]

    y0, g0 = run(_FusedCCAConv.apply)
    y1, g1 = run(cca_conv_op)
    _same(y0, y1, "y")
    for n, a, b in zip(("dx", "dw_dw", "dw_gp"), g0, g1):
        _same(a, b, n)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("expand", [True, False])
@pytest.mark.parametrize("need_x", [True, False])
def test_seg_conv_op_equals_function(dtype, expand, need_x):
    """The segment-reset conv (strict geometry): the op equals the Function bit for bit, with a
    batch-stride-0 ``seg`` and a dense one, and with ``x`` wanting no grad (the Function then
    skips the dx kernel; so must the op)."""
    G, CG, Kk, B, S = 4, 64, 4, 3, 128
    Cc = G * CG
    g = torch.Generator(device="cuda").manual_seed(0)
    x0 = torch.randn(B, S, Cc, device="cuda", generator=g).to(dtype)
    w_dw0 = torch.randn(Cc, 1, Kk, device="cuda", generator=g).to(dtype)
    w_gp0 = torch.randn(Cc, CG, Kk, device="cuda", generator=g).to(dtype) / 8
    r = torch.randn(B, Cc, S, device="cuda", generator=g)
    if expand:
        seg = (torch.arange(S, device="cuda") // 24).int()[None].expand(B, S)
    else:
        seg = (torch.arange(S, device="cuda")[None] // torch.arange(20, 24, device="cuda")[:B, None]
               ).int().contiguous()

    def run(fn):
        x = x0.clone().requires_grad_(need_x)
        wd = w_dw0.clone().requires_grad_(True)
        wg = w_gp0.clone().requires_grad_(True)
        y = fn(x.transpose(1, 2), wd, wg, seg, CG, Kk)
        (y.float() * r).sum().backward()
        return y.detach(), [x.grad, wd.grad, wg.grad]

    y0, g0 = run(_FusedSegCausalConv.apply)
    y1, g1 = run(lambda x, wd, wg, sg, cg, k: seg_conv_op(x, wd, wg, sg))
    _same(y0, y1, "y")
    for n, a, b in zip(("dx", "dw_dw", "dw_gp"), g0, g1):
        assert (a is None) == (b is None), n
        if a is not None:
            _same(a, b, n)


@pytest.mark.parametrize("B,S,n_skip", [(6, 256, 0), (3, 37, 5)])
def test_cca_prologue_op_equals_function(B, S, n_skip):
    from test_cca_prologue_tiled import NAMES, _inputs
    ins, cos, sin, gos = _inputs(B, S, True)

    def run(fn):
        xs = [t.detach().requires_grad_(True) for t in ins]
        outs = fn(*xs[:6], *xs[6:9], cos, sin, H, HKV, D, H // HKV, n_skip, 1e-6)
        torch.autograd.backward(list(outs), [g.to(o.dtype) for g, o in zip(gos, outs)])
        return [o.detach() for o in outs], [x.grad for x in xs]

    o0, g0 = run(_FusedCCAPrologueRows.apply)
    o1, g1 = run(cca_prologue_rows_op)
    for n, a, b in zip("qkv", o0, o1):
        _same(a, b, n)
    for n, a, b in zip(NAMES, g0, g1):
        _same(a, b, "grad_" + n)
        assert a.stride() == b.stride(), ("layout", n, a.stride(), b.stride())


def _hc(ops, bf16=True):
    torch.manual_seed(1)
    hc = HyperConnectionResidual(C, n_streams=4, use_kernel=True, region_fused=True,
                                 carrier_bf16=bf16, block_custom_ops=ops).cuda()
    with torch.no_grad():                       # a mapping far from the identity init
        hc.proj.weight.normal_(0, 3.0 / (4 * C) ** 0.5)
        hc.proj.bias.normal_(0, 0.5)
    return hc


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("B,S,with_term", [(6, 256, False), (3, 37, True)])
def test_hc_region_ops_equal_functions(dtype, B, S, with_term):
    g = torch.Generator(device="cuda").manual_seed(0)
    h0 = (torch.randn(B, S, 4, C, device="cuda", generator=g) * 2).to(dtype)
    wf0 = torch.randn(C, C, device="cuda", generator=g) / C ** 0.5
    r = torch.randn(B, S, 4, C, device="cuda", generator=g)
    t0 = torch.randn(B, S, C, device="cuda", generator=g) if with_term else None

    def run(hc):
        h = h0.clone().requires_grad_(True)
        wf = wf0.clone().requires_grad_(True)
        term = None if t0 is None else t0.clone().requires_grad_(True)
        hc.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = hc(h, lambda x: x @ wf.to(torch.bfloat16), post_inject=term)
        (out.float() * r).sum().backward()
        gs = [h.grad, hc.proj.weight.grad, hc.proj.bias.grad, wf.grad]
        return out.detach(), gs + ([] if term is None else [term.grad])

    o0, g0 = run(_hc(False))
    o1, g1 = run(_hc(True))
    _same(o0, o1, "out")
    for n, a, b in zip(("grad_h", "grad_proj_w", "grad_proj_b", "grad_sublayer_w", "grad_term"),
                       g0, g1):
        assert a is not None and b is not None, n
        _same(a, b, n)


def test_hc_ops_fall_back_where_the_region_kernels_do_not_apply():
    """No bf16 autocast: the entry returns a bare ``h_link``, the exit takes the same fallback
    as ``hc_region_exit`` (the region kernels' own gate), and the result equals the Function's."""
    h = torch.randn(2, 64, 4, C, device="cuda")
    hc_off, hc_on = _hc(False, bf16=False), _hc(True, bf16=False)
    y = torch.randn(2, 64, C, device="cuda")
    a = hc_off(h, lambda x: y)
    b = hc_on(h, lambda x: y)
    _same(a, b, "fallback out")


# ── 2 + 3. no graph break, compiled close ────────────────────────────────────

def _block_fn(ops):
    """entry -> y = x_bar @ W -> exit, then conv + prologue, with a SiLU and a cast between."""
    from morph.kernels.triton.fused_cca_conv import fused_cca_conv
    hc = _hc(ops)
    Gq = H + HKV
    wd = (torch.randn(Gq * D, 1, 4, device="cuda") / 2).bfloat16()
    wg = (torch.randn(Gq * D, D, 4, device="cuda") / 16).bfloat16()
    ws = torch.randn(C, C, device="cuda") / 32                              # the sublayer
    wf = torch.randn(C, Gq * D + HKV * D // 2, device="cuda") / 32          # q|k latents, v

    def fn(h, cos, sin, wq, wk, temp):
        h2 = hc(h, lambda x: torch.nn.functional.silu(x @ ws.to(torch.bfloat16)))
        lat = (h2[:, :, 0] @ wf.to(torch.bfloat16))
        lat = lat.to(torch.bfloat16)
        q_lat, k_lat = lat[..., :H * D], lat[..., H * D:Gq * D]
        v_curr = lat[..., Gq * D:]
        qk = lat[..., :Gq * D]
        conv = fused_cca_conv(qk.transpose(1, 2), wd, wg, Gq, 4, custom_op=ops)
        q_conv = conv[:, :H * D].transpose(1, 2)
        k_conv = conv[:, H * D:].transpose(1, 2)
        v_prev = torch.nn.functional.pad(v_curr[:, :-1], (0, 0, 1, 0))
        return fused_cca_prologue(q_lat, k_lat, q_conv, k_conv, v_curr, v_prev, wq, wk, temp,
                                  cos, sin, H, HKV, D, tiled=True, custom_op=ops)
    return fn


def _args(B=2, S=128):
    from test_cca_prologue_tiled import _inputs
    _, cos, sin, _ = _inputs(B, S, True)
    g = torch.Generator(device="cuda").manual_seed(3)
    h = (torch.randn(B, S, 4, C, device="cuda", generator=g)).bfloat16().requires_grad_(True)
    return h, cos, sin, 1 + 0.1 * torch.randn(D, device="cuda"), 1 + 0.1 * torch.randn(
        D, device="cuda"), 0.3 * torch.randn(HKV, device="cuda")


def test_ops_compile_fullgraph_and_fences_do_not():
    args = _args()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        torch._dynamo.reset()
        out = torch.compile(_block_fn(True), fullgraph=True)(*args)
        torch.autograd.backward(list(out), [torch.ones_like(o) for o in out])
        torch._dynamo.reset()
        with pytest.raises(Exception, match="(?i)disable|graph break|skip"):
            torch.compile(_block_fn(False), fullgraph=True)(*args)
        torch._dynamo.reset()


def test_compiled_ops_close_to_eager_fenced_path():
    args = _args()
    # same weights: both closures drew the same seeded HC weights; the conv / prologue weights
    # come from the global RNG, so reseed and rebuild in the same order for each.
    torch.manual_seed(5)
    ref_fn = _block_fn(False)
    torch.manual_seed(5)
    on_fn = _block_fn(True)
    res = []
    for fn, comp in ((ref_fn, False), (on_fn, True)):
        h, *rest = args
        h = h.detach().clone().requires_grad_(True)
        f = torch.compile(fn) if comp else fn
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outs = f(h, *rest)
        torch.autograd.backward(list(outs), [torch.ones_like(o) for o in outs])
        res.append(([o.detach() for o in outs], h.grad))
        torch._dynamo.reset()
    for a, b in zip(res[0][0], res[1][0]):
        assert ((a.double() - b.double()).norm() / a.double().norm()).item() < 2e-2
    assert ((res[0][1].double() - res[1][1].double()).norm() / res[0][1].double().norm()).item() \
        < 5e-2


# ── 4. refusals, default ─────────────────────────────────────────────────────

def test_refusals_and_default():
    with pytest.raises(ValueError, match="block_custom_ops"):
        HyperConnectionResidual(C, use_kernel=True, region_fused=False, block_custom_ops=True)
    assert MORPHConfig.__dataclass_fields__["block_custom_ops"].default is False
    from test_cca_prologue_tiled import _inputs
    ins, cos, sin, _ = _inputs(2, 64, True)
    with pytest.raises(ValueError, match="cca_prologue_tiled"):
        fused_cca_prologue(*ins[:6], *ins[6:9], cos, sin, H, HKV, D, tiled=False, custom_op=True)
    with pytest.raises(ValueError, match="block_custom_ops"):
        from morph.model.transformer import MORPHTransformer
        MORPHTransformer(MORPHConfig(
            d_model=256, n_heads=2, n_kv_heads=2, d_ff=688, vocab_size=512, max_seq_len=512,
            n_prelude=1, n_core=0, n_coda=1, channel_dims=(128, 80, 48), compression=2,
            csa_compress_ratio=8, hca_compress_ratio=256, top_k=256, window_size=256,
            retention=False, use_kernels=True, hc_use_kernel=True, block_custom_ops=True))
