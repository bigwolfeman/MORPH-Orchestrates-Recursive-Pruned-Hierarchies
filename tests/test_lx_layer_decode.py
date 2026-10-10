"""The row-batched LXTUL block kernels (``morph/kernels/triton/lx_layer_decode.py``).

Design note: ``.agents/notes/proposed/feature/2026-10-10-lx-decode-megakernel.md``. Each
kernel is checked at both widths the decoder runs (production C 1024, H 8, Hkv 4, D 64,
GH 256, d_ff 2816; the decoder test fixture C 256, H 4, Hkv 2, D 32, GH 64) with random
weights:

* ``hc_partials`` + ``hc_pre``, ``lat_gemm``, ``up_post``, ``gu_swiglu``, ``dn_post``
  against a float64 PyTorch reference of the same formula;
* ``lat_post`` against the plain engine's ``decode_front`` (the tested fp32 site
  front-end, ``tests/test_decode_engine_widths.py``) on a position sequence whose
  history rows come from all three sources of the index encoding (latent table, zero
  row, this event's own partials), and against the K/V cache rows it must (and must not)
  write.

Error metric: ``max |got - ref| / max |ref|``. Tolerance ``TOL = 2e-5``, argued from fp32:
a length-K dot product with a random-sign sum has relative rounding error about
``sqrt(K) * 2**-24`` (K = 4096 for the HC projection: 4e-6); the measured worst is
printed by each test and sits below 2e-6 (``-s``).

MUTANTS: the kernel source is copied with one edit (a broken index, a dropped term, a
wrong tile offset), loaded as its own module, and the same check must FAIL (error above
``TOL * 10``; the unbroken kernels sit near 1e-7). A test that passes on a broken copy
is a test of nothing.
"""
from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Triton kernels on CUDA")

import morph.kernels.triton.lx_layer_decode as LXK  # noqa: E402
from morph.kernels.triton.fused_decode_step import ternary_pack  # noqa: E402
from morph.model.hyper_connections import cayley_orthogonal  # noqa: E402
from test_decode_front_kvrow import KC, Site, _rope  # noqa: E402

TOL = 2e-5
WIDTHS = {
    "prod": dict(C=1024, H=8, Hkv=4, D=64, FF=2816),
    "fixture": dict(C=256, H=4, Hkv=2, D=32, FF=696),     # I/4 = 174: a masked chunk tail
}
N = 4
HC = types.SimpleNamespace(tau=1.0, cayley_alpha=0.1, eps=1e-6)


def _err(got: torch.Tensor, ref: torch.Tensor) -> float:
    ref = ref.double()
    return float((got.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


def _rn(*s, gen, sc=1.0):
    return (torch.randn(*s, generator=gen) * sc).cuda()


# ── float64 references ────────────────────────────────────────────────────────────────


def _hc_ref(h, W, b, nw, neps):
    """HyperConnectionResidual._mappings + x_bar + RMSNorm, float64."""
    h, W, b, nw = h.double(), W.double(), b.double(), nw.double()
    R = h.shape[0]
    xf = h.reshape(R, -1)
    rms = xf.pow(2).mean(-1, keepdim=True).add(HC.eps).sqrt()
    raw = (xf @ W.T / rms + b).reshape(R, 3, N, N)
    hpre = torch.softmax(raw[:, 0] / HC.tau, dim=-1)
    hpost = torch.softmax(raw[:, 1] / HC.tau, dim=-2)
    hres = cayley_orthogonal(raw[:, 2], 3, HC.cayley_alpha)
    xbar = torch.einsum("rj,rjc->rc", hpre.mean(dim=-2), h)
    xn = xbar * torch.rsqrt(xbar.pow(2).mean(-1, keepdim=True) + neps) * nw
    return xn, hres.reshape(R, N * N), hpost.sum(dim=-1)


def _post_ref(hres, hpost, h, y, term):
    out = torch.einsum("rij,rjc->ric", hres.double().view(-1, N, N), h.double())
    out = out + hpost.double()[:, :, None] * y.double()[:, None, :]
    if term is not None:
        out = out + term.double()[:, None, :]
    return out


def _ternary(O, I, gen):
    w = torch.randint(-1, 2, (O, I), generator=gen).float() * 0.0371
    return w.cuda()


# ── checks (each takes the kernel module, so a mutant runs the same check) ──────────────


def _unset(*shape) -> torch.Tensor:
    """An output buffer filled with 1e9: an entry the kernel does not write fails the
    check (``torch.empty`` can hand back the previous check's correct values)."""
    return torch.full(shape, 1e9, device="cuda")


def check_hc(K, w, R=5, seed=0) -> float:
    gen = torch.Generator().manual_seed(seed)
    C = w["C"]
    h = _rn(R, N, C, gen=gen)
    W = _rn(K.HC_NO, N * C, gen=gen, sc=(N * C) ** -0.5)
    b = _rn(K.HC_NO, gen=gen, sc=0.3)
    nw = 1.0 + _rn(C, gen=gen, sc=0.1)
    rawp = K.hc_partials(h, W)
    xn, hres, hpost = _unset(R, C), _unset(R, N * N), _unset(R, N)
    K.hc_pre(h, rawp, b, HC, nw, 1e-6, xn, hres, hpost)
    rx, rh, rp = _hc_ref(h, W, b, nw, 1e-6)
    return max(_err(xn, rx), _err(hres, rh), _err(hpost, rp))


def check_lat_gemm(K, w, R=11, KS=4, seed=1) -> float:
    gen = torch.Generator().manual_seed(seed)
    C, O = w["C"], 1280 if w["C"] == 1024 else 320
    x = _rn(R, C, gen=gen)
    W = _rn(O, C, gen=gen, sc=C ** -0.5)
    part = _unset(R, KS, O)
    K.lat_gemm(x, W, part)
    return _err(part.sum(1), x.double() @ W.double().T)


def check_up_post(K, w, R=6, seed=2, term=True) -> float:
    gen = torch.Generator().manual_seed(seed)
    C, I = w["C"], w["H"] * w["D"]
    x = _rn(R, I, gen=gen)
    Wup = _rn(C, I, gen=gen, sc=I ** -0.5)
    hres = _rn(R, N * N, gen=gen, sc=0.5)
    hpost = _rn(R, N, gen=gen, sc=0.5)
    h = _rn(R, N, C, gen=gen)
    t = _rn(R, C, gen=gen) if term else None
    Wn = _rn(K.HC_NO, N * C, gen=gen, sc=(N * C) ** -0.5)
    out = _unset(*h.shape)
    rawp = _unset(R, C // K.PART_TILE, K.HC_NO)
    K.up_post(x, Wup, hres, hpost, h, out, t, Wn, rawp)
    ro = _post_ref(hres, hpost, h, x.double() @ Wup.double().T, t)
    rp = ro.reshape(R, -1) @ Wn.double().T
    return max(_err(out, ro), _err(rawp.sum(1), rp))


def check_gu(K, w, R=7, seed=3) -> float:
    gen = torch.Generator().manual_seed(seed)
    C, FF = w["C"], w["FF"]
    x = _rn(R, C, gen=gen)
    Wgu = _ternary(2 * FF, C, gen)
    codes, sc = ternary_pack(Wgu)
    out = _unset(R, FF)
    K.gu_swiglu(x, codes, sc, out)
    gu = x.double() @ Wgu.double().T
    g, u = gu[:, :FF], gu[:, FF:]
    return _err(out, g * torch.sigmoid(g) * u)


def check_dn_post(K, w, R=6, seed=4, nxt=True) -> float:
    gen = torch.Generator().manual_seed(seed)
    C, FF = w["C"], w["FF"]
    x = _rn(R, FF, gen=gen)
    Wd = _ternary(C, FF, gen)
    codes, sc = ternary_pack(Wd)
    hres = _rn(R, N * N, gen=gen, sc=0.5)
    hpost = _rn(R, N, gen=gen, sc=0.5)
    h = _rn(R, N, C, gen=gen)
    t = _rn(R, C, gen=gen)
    Wn = _rn(K.HC_NO, N * C, gen=gen, sc=(N * C) ** -0.5) if nxt else None
    out = _unset(*h.shape)
    rawp = _unset(R, C // K.PART_TILE, K.HC_NO) if nxt else None
    K.dn_post(x, codes, sc, hres, hpost, h, out, t, Wn, rawp)
    ro = _post_ref(hres, hpost, h, x.double() @ Wd.double().T, t)
    e = _err(out, ro)
    if nxt:
        e = max(e, _err(rawp.sum(1), ro.reshape(R, -1) @ Wn.double().T))
    return e


def check_lat_post(K, w, seed=5) -> float:
    """A virtual sequence of 14 positions of one cache row: positions 0..T0-1 already in
    the latent table (written by earlier lat_post calls), an event of E rows at T0..T0+E-1
    whose history reaches the table, the zero row (a segment cut at position 2) and the
    event's own earlier rows. Every output against decode_front on the materialised
    x history; also the K/V entries written into cache row 3 of 5 (``kv_row``)."""
    from morph.kernels.triton.fused_decode_step import decode_front
    gen = torch.Generator().manual_seed(seed)
    site = Site(w["C"], w["H"], w["Hkv"], w["D"], gen=gen)
    H, D, C, O = site.H, site.D, site.C, site.O
    LW = site.LQ + site.LK + site.VH
    T0, E, cut, NC = 9, 5, 2, 24
    T = T0 + E
    xs = _rn(T, C, gen=gen)
    zero_row = T
    table_x = torch.cat([xs, torch.zeros(1, C, device="cuda")])
    cos_all, sin_all = _rope(T, D, gen)

    def hist(p):
        return [hp if hp >= cut else -1 for hp in range(p - 6, p)]

    lat = torch.full((T, LW), 777.0, device="cuda")
    kc = torch.full((5, H, NC, D), 555.0, device="cuda")
    vc = torch.full_like(kc, 555.0)
    rk = torch.full((T, H, NC, D), 555.0, device="cuda")
    rv = torch.full_like(rk, 555.0)
    worst = 0.0

    def run(ps, idx_rows):
        nonlocal worst
        n = len(ps)
        x = xs[ps[0]:ps[-1] + 1].contiguous()
        part = torch.empty(n, 4, O, device="cuda")
        K.lat_gemm(x, site.wqkv, part)
        idx = torch.tensor(idx_rows, dtype=torch.int32, device="cuda")
        pos = torch.tensor(ps, dtype=torch.long, device="cuda")
        q, xres, gateh = _unset(n, H, D), _unset(n, site.LQ), _unset(n, site.GH)
        kv_row = torch.full((n,), 3, dtype=torch.long, device="cuda")
        K.lat_post(part, lat, idx, pos, cos_all[ps[0]:ps[-1] + 1], sin_all[ps[0]:ps[-1] + 1],
                   site.wdwq, K.conv_taps_major(site.wgpq), site.wdwk,
                   K.conv_taps_major(site.wgpk), site.qnw, site.knw,
                   site.etemp, kc, vc, q, xres, gateh, H, site.Hkv, D, site.LQ, site.LK,
                   site.VH, site.GH, KC, 1e-6, kv_row=kv_row)
        # the engine on the materialised history of the same positions
        x_off = torch.tensor([[zero_row if hp < 0 else hp for hp in hist(p)] + [p]
                              for p in ps], dtype=torch.int32, device="cuda")
        xh = table_x.unsqueeze(0).expand(n, *table_x.shape)
        q0 = torch.empty_like(q)
        x0 = torch.empty_like(xres)
        g0 = torch.empty_like(gateh)
        part0 = torch.zeros(n, 4, 7, O, device="cuda")
        ek = torch.full((n, H, NC, D), 555.0, device="cuda")
        ev = torch.full_like(ek, 555.0)
        decode_front(xh, x_off, pos, site.wqkv, cos_all[ps[0]:ps[-1] + 1],
                     sin_all[ps[0]:ps[-1] + 1], site.wdwq, site.wgpq, site.wdwk, site.wgpk,
                     site.qnw, site.knw, site.etemp, ek, ev, q0, x0, g0, None, part0,
                     H, site.Hkv, D, site.LQ, site.LK, site.VH, site.GH, KC, 1e-6)
        worst = max(worst, _err(q, q0), _err(xres, x0), _err(gateh, g0))
        for i, p in enumerate(ps):
            worst = max(worst, _err(kc[3, :, p % NC], ek[i, :, p % NC]),
                        _err(vc[3, :, p % NC], ev[i, :, p % NC]))

    # positions 0..T0-1 one at a time (the token sites): history from the table / zero
    for p in range(T0):
        run([p], [hist(p) + [0, p]])
    # one event of E rows (the loop cells): table rows, zero rows, event rows -2 - i
    rows = []
    for i, p in enumerate(range(T0, T)):
        hh = [hp if hp < T0 else -2 - (hp - T0) for hp in hist(p)]
        rows.append(hh + [0, -1])
    run(list(range(T0, T)), rows)
    # the table rows written: a latent row is the projection of its normed input
    ref_lat = (xs[:T0].double() @ site.wqkv.double().T)[:, :LW]
    worst = max(worst, _err(lat[:T0], ref_lat))
    assert (lat[T0:] == 777.0).all(), "an event row with write index -1 wrote the table"
    untouched = torch.ones(5, NC, dtype=torch.bool)
    untouched[3, [p % NC for p in range(T)]] = False
    um = untouched.cuda()[:, None, :, None].expand_as(kc)
    assert (kc[um] == 555.0).all() and (vc[um] == 555.0).all(), "K/V written outside (row, entry)"
    return worst


CHECKS = {"hc": check_hc, "lat_gemm": check_lat_gemm, "up_post": check_up_post,
          "gu": check_gu, "dn_post": check_dn_post, "lat_post": check_lat_post,
          "up_post_rb4": lambda K, w: check_up_post(K, w, R=17)}


@pytest.mark.parametrize("width", list(WIDTHS))
@pytest.mark.parametrize("name", list(CHECKS))
def test_kernel_matches_reference(name, width):
    e = CHECKS[name](LXK, WIDTHS[width])
    print(f"{name} {width}: {e:.2e}")
    assert e < TOL, (name, width, e)


@pytest.mark.parametrize("R", [1, 2, 3, 4, 5, 8, 9, 15, 16, 17, 33])
def test_row_counts_and_no_next(R):
    """Row counts on both sides of the 4-rows-per-CTA switch (``_rb``: lat_gemm and
    gu_swiglu from 4 rows, up_post from 16), full and ragged last row blocks, a 1-row
    event; the posts without a term / without the next partials."""
    w = WIDTHS["fixture"]
    assert check_lat_gemm(LXK, w, R=R) < TOL
    assert check_up_post(LXK, w, R=R, term=False) < TOL
    assert check_dn_post(LXK, w, R=R, nxt=False) < TOL
    assert check_gu(LXK, w, R=R) < TOL
    assert check_hc(LXK, w, R=R) < TOL


# ── mutants: one edit of the kernel source each; the check must fail ─────────────────

_SRC = pathlib.Path(LXK.__file__).read_text()

MUTANTS = [
    # (check, the exact text, its replacement, what it breaks)
    # -- the 4-rows-per-CTA code (check_lat_gemm R = 11, check_gu R = 7: a ragged block)
    ("lat_gemm", "acc3 += w * tl.load(X + (r + 3) * KDIM + kk",
     "acc3 += w * tl.load(X + (r + 2) * KDIM + kk", "4th row of a block reads the 3rd"),
    ("lat_gemm", "tl.sum(acc2, 1), mask=mo & (r + 2 < R))",
     "tl.sum(acc2, 1), mask=mo & (r + 2 < R - 4))", "3rd row of a block not stored"),
    ("gu", "                ag2 += wg * x\n", "                ag2 += wu * x\n",
     "3rd row's gate through the up weights"),
    ("gu", "x = tl.load(X + (r + 1) * I + st * I4 + j, mask=mj & (r + 1 < R),",
     "x = tl.load(X + r * I + st * I4 + j, mask=mj & (r + 1 < R),", "2nd row reads the 1st"),
    ("up_post_rb4", "RAWP, r + 3, t, tl.sum(acc3, 1), c",
     "RAWP, r + 3, t, tl.sum(acc2, 1), c", "4th row posts the 3rd row's y"),
    ("hc", "inv_tau = 1.0 / TAU", "inv_tau = 1.0 / (TAU * 2.0)", "softmax temperature"),
    ("hc", "Hpost = qe / tl.sum(qe, axis=0)[None, :]", "Hpost = qe / tl.sum(qe, axis=1)[:, None]",
     "post softmax over the wrong axis"),
    ("hc", "tm = (tt < NT)[:, None]", "tm = (tt < NT - 1)[:, None]", "a partial tile dropped"),
    ("hc", "res_s = tl.reshape(raw_res, (N, N)) * inv_rms + b_res",
     "res_s = tl.reshape(raw_res, (N, N)) * inv_rms", "res bias dropped"),
    ("lat_gemm", "for k0 in range(s * KCH, (s + 1) * KCH, BK):",
     "for k0 in range(s * KCH, (s + 1) * KCH - BK, BK):", "last k chunk of a slice dropped"),
    ("lat_gemm", "acc += w * tl.load(X + r * KDIM + kk)[None, :]",
     "acc += (w * tl.load(X + r * KDIM + kk)[None, :]) * 1.0",
     "(control: a no-op edit must NOT be caught)"),
    ("up_post", "acc += tl.load(HPOST + r * N + i) * y",
     "acc += tl.load(HPOST + r * N + (N - 1 - i)) * y", "Hpost rows reversed"),
    ("up_post", "p += tl.load(wrow + 3 * C, mask=mo, other=0.0).to(tl.float32) * o3[None, :]",
     "p += tl.load(wrow + 3 * C, mask=mo, other=0.0).to(tl.float32) * o2[None, :]",
     "partials read stream 2 twice"),
    ("up_post", "tl.store(RAWP + (r * NT + t) * NO + o, tl.sum(p, 1), mask=o < NO)",
     "tl.store(RAWP + (r * NT + t) * NO + o, tl.sum(p, 1) * 1.01, mask=o < NO)",
     "partials scaled"),
    ("gu", "pu = tl.load(W2 + (FF + o)[:, None] * I4 + j[None, :], mask=mj[None, :],",
     "pu = tl.load(W2 + (FF - 8 + o)[:, None] * I4 + j[None, :], mask=mj[None, :],",
     "up rows off by one tile"),
    ("dn_post", "tl.load(X + r * FF + st * I4 + j, mask=mj, other=0.0)[None, :]",
     "tl.load(X + r * FF + st * (I4 - 8) + j, mask=mj, other=0.0)[None, :]", "strip offset"),
    ("dn_post", "y = tl.sum(acc, 1) * tl.load(SCALE)", "y = tl.sum(acc, 1)",
     "ternary scale dropped"),
    ("gu", "x = tl.load(X + r * I + st * I4 + j, mask=mj, other=0.0)[None, :]",
     "x = tl.load(X + (r % 2) * I + st * I4 + j, mask=mj, other=0.0)[None, :]",
     "rows mixed up"),
    ("dn_post", "mj = j < I4                          # a masked lane",
     "mj = j < I4 + 4                      # a masked lane", "tail mask too wide"),
    ("lat_post", "v = tl.zeros_like(offs).to(tl.float32)",
     "v = tl.zeros_like(offs).to(tl.float32) + 1.0", "zero row not zero"),
    ("lat_post", "v = _psum1(PART + (-2 - idx).to(tl.int64) * (KS * OP), col, offs, OP, KS)",
     "v = _psum1(PART + (-1 - idx).to(tl.int64) * (KS * OP), col, offs, OP, KS)",
     "event row off by one"),
    ("lat_post", "av = _lrow(LAT, PART, i5, LQ + LK + (g - HKV // 2) * D, offs, slat, KS, OP)",
     "av = _lrow(LAT, PART, i4, LQ + LK + (g - HKV // 2) * D, offs, slat, KS, OP)",
     "value shift reads position p-2"),
    ("lat_post", "            tl.store(lw + LQ + LK + g * D + offs, vp6, mask=wlm)\n", "",
     "W_v_prev x of the position not written"),
    ("lat_post", "acc += tl.load(WGPT + 2 * CL * D + tile) * y2[None, :]",
     "acc += tl.load(WGPT + 2 * CL * D + tile) * y1[None, :]", "conv tap 2 reads tap 1's input"),
    ("lat_post", "tl.store(lw + col + offs, a6, mask=wlm)\n        qc",
     "qc", "q latent of the position not written"),
]


def _mutant(old: str, new: str, tmp_path: pathlib.Path):
    assert _SRC.count(old) >= 1, f"mutant text not in the kernel source: {old!r}"
    f = tmp_path / f"lxk_mut_{abs(hash((old, new)))}.py"
    f.write_text(_SRC.replace(old, new, 1))
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("check,old,new,what", MUTANTS, ids=[m[3] for m in MUTANTS])
def test_mutant_is_caught(check, old, new, what, tmp_path):
    mod = _mutant(old, new, tmp_path)
    e = CHECKS[check](mod, WIDTHS["fixture"])
    print(f"mutant [{what}]: {e:.2e}")
    if what.startswith("(control"):
        assert e < TOL, e
    else:
        assert e > TOL * 10, (what, e)
