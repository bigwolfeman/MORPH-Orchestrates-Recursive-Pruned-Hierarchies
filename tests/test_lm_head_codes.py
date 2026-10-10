"""The LM head as int codes x row scale (``morph/kernels/triton/lm_head_codes.py``).

* ``head_codes_pack`` keeps the codes only when ``fp32(code) * scale`` IS the head table
  (bit for bit), and refuses a table it cannot reproduce (no parametrization, a table one
  ulp off, a bf16 copy).
* ``head_codes_logits`` against a float64 reference of the fp32 table's logits, at the
  model's widths (V 49169, C 1024 = 768 euclidean + 256 Lorentz) and at a small width with
  a vocab tail, for 1 to 16 query rows and a strided query/output row.

Error metric ``max |got - ref| / max |ref|``, TOL 2e-5: a length-1024 fp32 dot product with
random signs has relative rounding error about sqrt(1024) * 2**-24 = 2e-6.

MUTANTS: the kernel source with one edit, loaded as its own module; the check must FAIL.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Triton kernels on CUDA")

import morph.kernels.triton.lm_head_codes as LHC  # noqa: E402
from morph.model.embed_quant import IntNRowSTE  # noqa: E402

TOL = 2e-5
WIDTHS = {"prod": (49169, 768, 256), "small": (1001, 96, 32)}


def _err(got, ref):
    ref = ref.double()
    return float((got.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _head(V, CE, CL, bits=6, seed=0):
    """An IntNRowSTE-parametrized euclidean table + an fp32 Lorentz block, as the model
    builds its tied head (``lm_weight() = cat([euc.weight, lor])``)."""
    gen = torch.Generator().manual_seed(seed)
    euc = nn.Embedding(V, CE).cuda()
    with torch.no_grad():
        euc.weight.copy_(torch.randn(V, CE, generator=gen) * 0.02)
    parametrize.register_parametrization(euc, "weight", IntNRowSTE(bits=bits))
    lor = (torch.randn(V, CL, generator=gen) * 0.05).cuda()
    with torch.no_grad():
        w_head = torch.cat([euc.weight, lor], dim=-1).contiguous()
    return euc, w_head


def check_logits(mod, V, CE, CL, R, strided=False, seed=0):
    euc, w_head = _head(V, CE, CL, seed=seed)
    pk = LHC.head_codes_pack(euc, w_head)
    assert pk is not None
    gen = torch.Generator().manual_seed(seed + 1)
    C = CE + CL
    if strided:                       # rows of a wider buffer, as the decoder's views are
        xb = torch.randn(R, C + 64, generator=gen).cuda()
        x = xb[:, 32:32 + C]
        ob = torch.full((R, V + 16), 777.0, device="cuda")
        out = ob[:, 8:8 + V]
    else:
        x = torch.randn(R, C, generator=gen).cuda()
        ob = None
        out = torch.full((R, V), 777.0, device="cuda")
    mod.head_codes_logits(x, *pk, out)
    ref = x.double() @ w_head.double().T
    e = _err(out, ref)
    if ob is not None:                # the padding around the view is untouched
        assert bool((ob[:, :8] == 777.0).all()) and bool((ob[:, 8 + V:] == 777.0).all())
    return e


@pytest.mark.parametrize("width", list(WIDTHS))
@pytest.mark.parametrize("R", [1, 3, 8, 16])
def test_logits_match_fp64(width, R):
    V, CE, CL = WIDTHS[width]
    e = check_logits(LHC, V, CE, CL, R)
    print(f"{width} R={R}: {e:.2e}")
    assert e < TOL, e


def test_strided_rows():
    e = check_logits(LHC, *WIDTHS["small"], R=5, strided=True)
    assert e < TOL, e


@pytest.mark.parametrize("bits", [6, 8])
def test_pack_is_bit_exact(bits):
    euc, w_head = _head(*WIDTHS["small"], bits=bits)
    codes, s, wl = LHC.head_codes_pack(euc, w_head)
    CE = codes.shape[1]
    assert torch.equal(codes.float() * s[:, None], w_head[:, :CE])
    assert torch.equal(wl, w_head[:, CE:])
    hi = 31 if bits == 6 else 127
    assert int(codes.abs().max()) == hi           # every row's max |w| maps to +-hi


def test_pack_refuses_what_it_cannot_reproduce():
    euc, w_head = _head(*WIDTHS["small"])
    off = w_head.clone()
    off[7, 3] = torch.nextafter(off[7, 3], torch.tensor(1.0, device="cuda"))
    assert LHC.head_codes_pack(euc, off) is None, "a table one ulp off must be refused"
    assert LHC.head_codes_pack(euc, w_head.to(torch.bfloat16)) is None
    plain = nn.Embedding(*euc.parametrizations.weight.original.shape).cuda()
    assert LHC.head_codes_pack(plain, w_head) is None, "no IntNRowSTE: no codes"
    assert LHC.head_codes_pack(euc, w_head) is not None


# ── mutants ──────────────────────────────────────────────────────────────────────────

_SRC = pathlib.Path(LHC.__file__).read_text()
MUTANTS = [
    ("w = tl.where(ke[None, :], we.to(tl.float32) * s[:, None], wl)",
     "w = tl.where(ke[None, :], we.to(tl.float32), wl)", "row scale dropped"),
    ("wl = tl.load(WL + v[:, None] * CL + (k - CE)[None, :],",
     "wl = tl.load(WL + v[:, None] * CL + (k - CE + 1)[None, :],", "Lorentz column off by one"),
    ("    for r in range(R):\n", "    for r in range(R - 1):\n", "last query row skipped"),
    ("x = tl.load(X + r * sx + k, mask=k < C, other=0.0)",
     "x = tl.load(X + r * C + k, mask=k < C, other=0.0)", "row stride ignored"),
    ("    ke = k < CE\n", "    ke = k < CE - 1\n", "last euclidean column dropped"),
    ("tl.store(OUT + r * sout + v, tl.sum(w * x[None, :], 1), mask=mv)",
     "tl.store(OUT + r * sout + v, tl.sum(w * x[None, :], 1) * 1.0, mask=mv)",
     "(control: a no-op edit must NOT be caught)"),
]


def _mutant(old, new, tmp_path):
    assert _SRC.count(old) == 1, f"mutant text not unique in the kernel source: {old!r}"
    f = tmp_path / f"lhc_mut_{abs(hash((old, new)))}.py"
    f.write_text(_SRC.replace(old, new))
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("old,new,what", MUTANTS, ids=[m[2] for m in MUTANTS])
def test_mutant_is_caught(old, new, what, tmp_path):
    mod = _mutant(old, new, tmp_path)
    e = max(check_logits(mod, *WIDTHS["small"], R=3, strided=True),
            check_logits(mod, *WIDTHS["small"], R=3))
    print(f"mutant [{what}]: {e:.2e}")
    if what.startswith("(control"):
        assert e < TOL, e
    else:
        assert e > TOL * 10, (what, e)
