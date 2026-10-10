"""``decode_front(..., kv_row=)`` (K2 of the LXTUL fused decoder) and a stride-0 ``x_hist``.

``morph/kernels/triton/fused_decode_step.py::decode_front`` is the plain engine's site
front-end (tested end to end by ``tests/test_decode_engine_widths.py``). The LXTUL fused
decoder (design note ``.agents/notes/proposed/feature/2026-10-10-lxtul-fused-decode.md``)
uses it in two new ways, pinned here at both widths of the note (production C 1024, H 8,
Hkv 4, D 64, GH 256; fixture C 256, H 4, Hkv 2, D 32, GH 64), with random fp32 weights:

* ``kv_row``: query row b writes its K/V into cache row ``kv_row[b]`` at entry
  ``pos_dev[b] % N`` and touches no other cache entry; q, xres and gateh are those of the
  engine path (``kv_row=None``) bit for bit, and so is the K/V row it writes.
* a stride-0 ``x_hist`` (one flat table expanded over the query rows) with ABSOLUTE
  ``x_off`` rows gives bit for bit what a materialised per-row history gives.

Every comparison is ``torch.equal``: the paths run the same kernels with the same
reduction order, only addresses differ.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="decode_front is a Triton kernel on CUDA")

WIDTHS = {
    "prod": dict(C=1024, H=8, Hkv=4, D=64),
    "fixture": dict(C=256, H=4, Hkv=2, D=32),
}
KC = 4                       # k_conv: the 7-row history of _conv7
SENTINEL = 12345.0


class Site:
    """Random weights of one CCA site in the engine's decode_front layout."""

    def __init__(self, C, H, Hkv, D, gen, dev="cuda"):
        self.C, self.H, self.Hkv, self.D = C, H, Hkv, D
        self.LQ, self.LK, self.VH, self.GH = H * D, Hkv * D, Hkv * D // 2, C // 4
        n_out = self.LQ + self.LK + 2 * self.VH + self.GH
        r = lambda *s, sc=1.0: (torch.randn(*s, generator=gen) * sc).to(dev)  # noqa: E731
        self.wqkv = r(n_out, C, sc=C ** -0.5)
        self.wdwq, self.wgpq = r(self.LQ, KC, sc=0.5), r(self.LQ, D, KC, sc=D ** -0.5)
        self.wdwk, self.wgpk = r(self.LK, KC, sc=0.5), r(self.LK, D, KC, sc=D ** -0.5)
        self.qnw, self.knw = 1.0 + r(D, sc=0.1), 1.0 + r(D, sc=0.1)
        self.etemp = torch.exp(r(Hkv, sc=0.1))
        self.O, self.dev = n_out, dev

    def run(self, x_hist, x_off, pos, cos, sin, win_k, win_v, kv_row=None):
        from morph.kernels.triton.fused_decode_step import decode_front
        B = x_hist.shape[0]
        q = torch.full((B, self.H, self.D), SENTINEL, device=self.dev)
        xres = torch.full((B, self.LQ), SENTINEL, device=self.dev)
        gateh = torch.full((B, self.GH), SENTINEL, device=self.dev)
        part = torch.zeros(B, 4, 7, self.O, device=self.dev)
        decode_front(x_hist, x_off, pos, self.wqkv, cos, sin, self.wdwq, self.wgpq,
                     self.wdwk, self.wgpk, self.qnw, self.knw, self.etemp, win_k, win_v,
                     q, xres, gateh, None, part, self.H, self.Hkv, self.D, self.LQ,
                     self.LK, self.VH, self.GH, KC, 1e-6, kv_row=kv_row)
        return q, xres, gateh


def _rope(B, D, gen, dev="cuda"):
    ang = torch.rand(B, D // 2, generator=gen).to(dev) * 6.0
    ang = torch.cat([ang, ang], dim=1)
    return torch.cos(ang).contiguous(), torch.sin(ang).contiguous()


def _cache(R, H, N, D, dev="cuda"):
    return (torch.full((R, H, N, D), SENTINEL, device=dev),
            torch.full((R, H, N, D), SENTINEL, device=dev))


@pytest.mark.parametrize("width", list(WIDTHS))
def test_kv_row_writes_only_the_named_row_and_entry(width):
    w = WIDTHS[width]
    gen = torch.Generator().manual_seed(1)
    site = Site(**w, gen=gen)
    B, R, N = 6, 9, 40
    # row 8 = the dummy row B of a bucket; row 2 appears twice at different entries
    kv_row = torch.tensor([8, 2, 5, 0, 2, 7], dtype=torch.int32, device="cuda")
    pos = torch.tensor([3, 17, 39, 0, 21, 40 + 11], dtype=torch.long, device="cuda")
    x_hist = torch.randn(B, 9, w["C"], generator=gen).cuda()
    x_off = torch.randint(0, 9, (B, 7), generator=gen, dtype=torch.int32).cuda()
    cos, sin = _rope(B, w["D"], gen)

    # the engine path: query row b -> cache row b (a B-row cache)
    ek, ev = _cache(B, w["H"], N, w["D"])
    q0, x0, g0 = site.run(x_hist, x_off, pos, cos, sin, ek, ev)
    # the K2 path: query row b -> cache row kv_row[b] (an R-row cache)
    k, v = _cache(R, w["H"], N, w["D"])
    q1, x1, g1 = site.run(x_hist, x_off, pos, cos, sin, k, v, kv_row=kv_row)
    torch.cuda.synchronize()

    assert torch.equal(q0, q1) and torch.equal(x0, x1) and torch.equal(g0, g1)
    assert not (q1 == SENTINEL).any() and not (g1 == SENTINEL).any()
    written = torch.zeros(R, N, dtype=torch.bool)
    for b in range(B):
        r, e = int(kv_row[b]), int(pos[b]) % N
        assert torch.equal(k[r, :, e], ek[b, :, e]), (b, r, e)
        assert torch.equal(v[r, :, e], ev[b, :, e]), (b, r, e)
        assert not (k[r, :, e] == SENTINEL).any()
        written[r, e] = True
    untouched = ~written.cuda().view(R, 1, N, 1).expand_as(k)
    assert (k[untouched] == SENTINEL).all() and (v[untouched] == SENTINEL).all()


@pytest.mark.parametrize("width", list(WIDTHS))
def test_kv_row_identity_equals_none(width):
    """kv_row = arange(B) is the engine path: same cache, bit for bit."""
    w = WIDTHS[width]
    gen = torch.Generator().manual_seed(2)
    site = Site(**w, gen=gen)
    B, N = 4, 16
    pos = torch.tensor([0, 5, 15, 33], dtype=torch.long, device="cuda")
    x_hist = torch.randn(B, 9, w["C"], generator=gen).cuda()
    x_off = torch.randint(0, 9, (B, 7), generator=gen, dtype=torch.int32).cuda()
    cos, sin = _rope(B, w["D"], gen)
    ka, va = _cache(B, w["H"], N, w["D"])
    kb, vb = _cache(B, w["H"], N, w["D"])
    site.run(x_hist, x_off, pos, cos, sin, ka, va)
    site.run(x_hist, x_off, pos, cos, sin, kb, vb,
             kv_row=torch.arange(B, dtype=torch.int64, device="cuda"))
    torch.cuda.synchronize()
    assert torch.equal(ka, kb) and torch.equal(va, vb)


@pytest.mark.parametrize("width", list(WIDTHS))
def test_stride0_x_hist_with_absolute_rows(width):
    """A flat table expanded over the query rows (stride 0) with absolute x_off rows,
    including the table's zero row, equals the materialised per-row history."""
    w = WIDTHS[width]
    gen = torch.Generator().manual_seed(3)
    site = Site(**w, gen=gen)
    B, rows, N = 8, 13, 32
    table = torch.randn(rows, w["C"], generator=gen).cuda()
    table[rows - 1] = 0.0                                   # the zero row (segment start)
    x_off = torch.randint(0, rows, (B, 7), generator=gen, dtype=torch.int32).cuda()
    x_off[:, :3] = rows - 1                                 # a cell near its segment start
    x_flat = table.unsqueeze(0).expand(B, rows, w["C"])
    assert x_flat.stride(0) == 0
    x_mat = table[x_off.long()].contiguous()                # [B, 7, C]
    x_off_mat = torch.arange(7, dtype=torch.int32, device="cuda").repeat(B, 1).contiguous()
    pos = torch.arange(B, dtype=torch.long, device="cuda") * 3
    cos, sin = _rope(B, w["D"], gen)
    kv_row = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int32, device="cuda")
    ka, va = _cache(2, w["H"], N, w["D"])
    kb, vb = _cache(2, w["H"], N, w["D"])
    qa, xa, ga = site.run(x_flat, x_off, pos, cos, sin, ka, va, kv_row=kv_row)
    qb, xb, gb = site.run(x_mat, x_off_mat, pos, cos, sin, kb, vb, kv_row=kv_row)
    torch.cuda.synchronize()
    assert torch.equal(qa, qb) and torch.equal(xa, xb) and torch.equal(ga, gb)
    assert torch.equal(ka, kb) and torch.equal(va, vb)
    assert not (qa == SENTINEL).any()


def test_kv_row_shape_is_checked():
    w = WIDTHS["fixture"]
    gen = torch.Generator().manual_seed(4)
    site = Site(**w, gen=gen)
    B = 2
    x_hist = torch.randn(B, 9, w["C"], generator=gen).cuda()
    x_off = torch.zeros(B, 7, dtype=torch.int32, device="cuda")
    cos, sin = _rope(B, w["D"], gen)
    k, v = _cache(3, w["H"], 8, w["D"])
    pos = torch.zeros(B, dtype=torch.long, device="cuda")
    with pytest.raises(AssertionError, match="kv_row"):
        site.run(x_hist, x_off, pos, cos, sin, k, v,
                 kv_row=torch.zeros(B + 1, dtype=torch.int32, device="cuda"))
    with pytest.raises(AssertionError, match="kv_row"):
        site.run(x_hist, x_off, pos, cos, sin, k, v,
                 kv_row=torch.zeros(B, dtype=torch.float32, device="cuda"))
