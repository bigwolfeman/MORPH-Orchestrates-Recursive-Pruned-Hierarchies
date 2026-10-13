"""model.pdl: programmatic dependent launch for the custom Triton kernels on the step path.

What each test pins (2026-10-13):
  1. THE CONTRACT, STATICALLY. Every kernel in ``PDL_KERNELS`` declares ``PDL: tl.constexpr =
     False`` as its LAST parameter, its first statement is ``if PDL: tl.extra.cuda.gdc_wait()``
     (before any load or store), and every launch of it passes ``**pdl_opts()``. A kernel
     launched with ``launch_pdl`` and no wait races with its predecessor, so a missing wait is
     a defect, not a slowdown. ``launch_pdl`` appears nowhere but ``_pdl.py``.
  2. THE WAIT REACHES THE PTX. A kernel compiled with PDL=True has ``griddepcontrol.wait`` in
     its PTX; compiled with PDL=False it has none (key off compiles the old kernel).
  3. SAME BITS. With the switch on, the conv, prologue rows, split GEMM and kv-flat kernels
     return the same bytes as with it off, in eager and replayed from a CUDA graph (PDL
     changes when kernels start, not what they compute).
  4. OFF IS EMPTY. ``pdl_opts()`` is ``{}`` by default; ``MORPHConfig.pdl`` defaults False.

Sabotage record (2026-10-13): (a) deleting the ``gdc_wait`` line of ``_kvf_sum_kernel`` ->
test_every_pdl_kernel_waits_first and test_the_wait_reaches_the_ptx fail; (b) dropping
``**pdl_opts()`` from one launch -> test_every_launch_passes_pdl_opts fails.
CUDA only.
"""
import inspect
import re
import pathlib

import pytest
import torch

from morph.kernels.triton import _pdl

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDL_KERNELS = {
    "morph/kernels/triton/fused_hyper_connection.py": [
        "_hcr_bwd_carrier_kernel", "_hcr_bwd_reduce_kernel", "_hc_post_fwd_kernel",
        "_hcr_fwd_proj_kernel", "_hcr_exit_bwd_kernel", "_hcr_fwd_map_kernel",
        "_hcr_bwd_map_kernel"],
    "morph/kernels/triton/tg_strict_attention.py": [
        "_fwd_kernel", "_bwd_pre_kernel", "_bwd_dkdv_kernel", "_bwd_dq_kernel"],
    "morph/kernels/triton/fused_cca_prologue.py": ["_cca_rows_fwd_kernel", "_cca_rows_bwd_kernel"],
    "morph/kernels/triton/fused_cca_conv.py": [
        "_gp_fwd_kernel", "_dw_fwd_kernel", "_gp_bwd_dx_kernel", "_gp_bwd_dw_kernel",
        "_dw_bwd_kernel"],
    "morph/kernels/triton/fused_seg_causal_conv.py": [
        "_seg_gp_fwd_kernel", "_seg_dw_fwd_kernel", "_seg_gp_bwd_dx_kernel",
        "_seg_gp_bwd_dw_kernel", "_seg_dw_bwd_kernel"],
    "morph/kernels/triton/kv_flat.py": ["_kvf_gp_bwd_kernel", "_kvf_dw_bwd_kernel",
                                        "_kvf_sum_kernel"],
    "morph/kernels/triton/split_gemm.py": ["_mm_kernel"],
    "morph/kernels/triton/ce_softmax_grad.py": [
        "_ce_logprob_grad_kernel", "_ce_softmax_grad_kernel", "_ce_row_lse_kernel"],
    "morph/kernels/triton/int_row_quant.py": ["_int_row_quant_kernel"],
    "morph/kernels/triton/exact_reduce.py": ["_rowsum_kernel", "_mean1d_kernel"],
    "morph/training/ademamix_fp32_kernel.py": ["_ademamix_fp32_kernel"],
    "morph/training/ademamix_b1zero_kernel.py": ["_ademamix_b1zero_fused_kernel"],
}
ALL = [(f, k) for f, ks in PDL_KERNELS.items() for k in ks]


def _def_block(src, name):
    """(signature text, body's first statement line) of ``def name(``."""
    m = re.search(r"def %s\(" % re.escape(name), src)
    assert m, name
    i, d = m.end() - 1, 0
    for j in range(i, len(src)):
        d += src[j] == "("
        d -= src[j] == ")"
        if d == 0:
            break
    sig = src[i + 1:j]
    rest = src[src.index("\n", j) + 1:]
    stripped = rest.lstrip()
    if stripped.startswith('"""'):                 # the statement list starts after the docstring
        end = stripped.index('"""', 3) + 3
        rest = stripped[end:].lstrip("\n")
    first = next(l for l in rest.splitlines() if l.strip())
    return sig, first.strip(), rest


@pytest.mark.parametrize("path,name", ALL)
def test_every_pdl_kernel_waits_first(path, name):
    src = (ROOT / path).read_text()
    sig, first, rest = _def_block(src, name)
    flat = re.sub(r"\s+", "", "".join(l.split("#")[0] for l in sig.splitlines())).rstrip(",")
    assert flat.endswith(",PDL:tl.constexpr=False"), (name, flat[-60:])
    assert first == "if PDL:", (name, first)
    body = rest.splitlines()
    k = next(i for i, l in enumerate(body) if l.strip())
    assert body[k + 1].strip() == "tl.extra.cuda.gdc_wait()", name


@pytest.mark.parametrize("path,name", ALL)
def test_every_launch_passes_pdl_opts(path, name):
    src = (ROOT / path).read_text()
    launches = [m.start() for m in re.finditer(r"\b%s\[" % re.escape(name), src)]
    assert launches, (path, name)
    for pos in launches:
        k = src.index("]", pos)
        k = src.index("(", k)
        d, j = 0, k
        while True:
            d += src[j] == "("
            d -= src[j] == ")"
            if d == 0:
                break
            j += 1
        assert "**pdl_opts()" in src[k:j], (path, name)


def test_launch_pdl_appears_only_in_the_pdl_module():
    hits = [str(p.relative_to(ROOT)) for p in (ROOT / "morph").rglob("*.py")
            if "launch_pdl" in p.read_text()]
    assert hits == ["morph/kernels/triton/_pdl.py"], hits


def test_off_is_empty_and_default_false():
    from morph.model.transformer import MORPHConfig
    assert MORPHConfig.__dataclass_fields__["pdl"].default is False
    assert _pdl.pdl_opts() == {}


pytestmark_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def pdl_on():
    _pdl.configure(True)
    yield
    _pdl.configure(False)


@pytestmark_cuda
def test_the_wait_reaches_the_ptx(pdl_on):
    from morph.kernels.triton import kv_flat
    import triton
    part = torch.randn(3, 256, device="cuda")
    out = torch.empty(256, device="cuda", dtype=torch.bfloat16)

    def ptx(on):
        k = kv_flat._kvf_sum_kernel.warmup(
            part, out, 256, 3, 1, 2, part, out, 256, 0, 1, 0, part, out, 256, 0, 1,
            BLK=128, NB=16, PDL=on, num_warps=4, grid=(1,))
        return k.asm["ptx"]
    assert "griddepcontrol.wait" in ptx(True)
    assert "griddepcontrol.wait" not in ptx(False)


def _bits(fn):
    out = fn()
    return [t.clone() for t in (out if isinstance(out, (list, tuple)) else [out])]


def _kernel_chain():
    """Conv fwd+bwd (op path), prologue rows, split GEMM: a chain of dependent custom kernels."""
    from morph.kernels.triton.fused_cca_conv import fused_cca_conv
    from morph.kernels.triton import split_gemm
    g = torch.Generator(device="cuda").manual_seed(0)
    B, C, S, CG, K = 6, 768, 256, 64, 4
    x0 = torch.randn(B, S, C, device="cuda", generator=g).bfloat16()
    wd0 = (torch.randn(C, 1, K, device="cuda", generator=g) * 0.3).bfloat16()
    wg0 = (torch.randn(C, CG, K, device="cuda", generator=g) * 0.1).bfloat16()
    r = torch.randn(B, C, S, device="cuda", generator=g)
    a = torch.randn(384, 1024, device="cuda", generator=g).bfloat16()
    b = torch.randn(1024, 512, device="cuda", generator=g).bfloat16()

    def run(flat):
        x = x0.clone().requires_grad_(True)
        wd, wg = wd0.clone().requires_grad_(True), wg0.clone().requires_grad_(True)
        y = fused_cca_conv(x.transpose(1, 2), wd, wg, C // CG, K, custom_op=True, flat=flat)
        (y.float() * r).sum().backward()
        c = torch.empty(384, 512, device="cuda", dtype=torch.bfloat16)
        split_gemm.tuned_mm(a, b, c)
        return [y.detach(), x.grad, wd.grad, wg.grad, c]
    return run


@pytestmark_cuda
@pytest.mark.parametrize("flat", [False, True])
def test_pdl_changes_timing_not_bits_eager_and_graph(flat):
    run = _kernel_chain()
    _pdl.configure(False)
    ref = _bits(lambda: run(flat))
    _pdl.configure(True)
    try:
        got = _bits(lambda: run(flat))
        for n, u, v in zip("y dx dwd dwg c".split(), ref, got):
            assert torch.equal(u, v), ("eager", n)
        run(flat)                                       # warm: compile + tune under PDL
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = run(flat)
        g.replay()
        torch.cuda.synchronize()
        for n, u, v in zip("y dx dwd dwg c".split(), ref, out):
            assert torch.equal(u, v), ("graph", n)
    finally:
        _pdl.configure(False)
