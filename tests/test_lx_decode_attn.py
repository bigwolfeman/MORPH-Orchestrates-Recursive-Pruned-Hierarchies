"""``lx_attn`` (morph/kernels/triton/lx_decode_attn.py) against the strict-geometry reference.

The reference is the graphed/cached LXTUL decoder's own attention: ``_attend`` of
``morph/inference/tul_generate_cached.py`` builds the masks from key POSITIONS and slot
flags (causal order, window size, XSA self-exclusion, the loop's cell relation, the coda
cell's self-only rule) and runs ``_branches``; ``attention._CCABase._gate_combine_up``
(with ``W_up`` = identity) adds the gate and the residual alpha. Each test builds, per
query row, the key set the reference reads (positions and slot flags as the reference
assigns them) AND the decoder's cache layout plus its ``meta`` row from the design note's
meta table (``.agents/notes/proposed/feature/2026-10-10-lxtul-fused-decode.md``). So a pass
says the kernel under the table computes the reference's function, not only that the
kernel follows its own index rules. Cache entries that no range covers, and cache rows no
query points at, hold large noise, so a stray read shows.

Tolerance, argued from an fp64 run. ``_branches`` computes its slot scores in fp32 even
for fp64 queries (``q.float()``), so the fp64 truth is ``_branches_fp64`` below: the same
masks (``_attend`` builds them; this function replaces only the arithmetic) in fp64 end
to end. ``test_fp64_reference_is_the_same_function`` pins that the fp64 formula and the
fp32 ``_branches`` agree to fp32 rounding. Measured on the 5090 (2026-10-10, every case
of this file): worst |fp32 reference - fp64| ``REF32_ERR`` and worst
|kernel - fp64| ``KERNEL_ERR`` below; ``TOL`` gives the kernel about 9x its measured
worst error. Sabotages of the kernel (sink dropped, self key kept in the window, kpos
ignored, cache row b read instead of meta[b, 0]) each fail tests here; the smallest error
any of them produced in a failing assertion was 9.8e-5 (the sink dropped on a row with
many keys), most were above 0.1 (kernel agent's report, 2026-10-10).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="lx_attn is a Triton kernel on CUDA")

# measured 2026-10-10 on the 5090, worst over every case in this file (see docstring)
REF32_ERR = 2.3e-6      # |_branches fp32 + gate combine  -  fp64 truth|
KERNEL_ERR = 1.1e-6     # |lx_attn  -  fp64 truth|
TOL = 1e-5

# production widths (lxtul-pointer-rerun/step_5000.pt) and the CUDA fixture widths
WIDTHS = {
    "prod": dict(H=8, D=64, GH=256, W=256, P=32, M=4, S=97),
    "fixture": dict(H=4, D=32, GH=64, W=16, P=8, M=4, S=10),
}
KINDS = ("prelude_token", "prelude_cell", "loop_cell", "coda_token", "coda_cell")
NOISE = 40.0            # magnitude of the K/V entries no range covers


def _ref_mod():
    from morph.inference import tul_generate_cached as tgc
    return tgc


def _combine():
    from morph.model.attention import _CCABase
    return _CCABase._gate_combine_up


def _branches_fp64(impl, q, k, v, win_mask, k_s, v_s, s_allow):
    """``_branches`` in fp64 end to end (same masks, same -inf / zero-row rules)."""
    scale = impl.cca.d_head ** -0.5
    B, H, n, _ = q.shape
    s = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    s = s.masked_fill(~win_mask.view(1, 1, n, -1), float("-inf"))
    has = win_mask.any(dim=-1).view(1, 1, n, 1)
    w = torch.softmax(torch.where(has, s, torch.zeros_like(s)), dim=-1)
    out_win = torch.where(has, torch.einsum("bhij,bhjd->bhid", w, v), torch.zeros_like(q))
    if k_s is None:
        return out_win, torch.zeros_like(q)
    sc = torch.einsum("bhid,bhjd->bhij", q, k_s) * scale
    sc = sc.masked_fill(~s_allow.view(1, 1, n, -1), float("-inf"))
    sink = impl.cca.sink_logits.view(1, H, 1, 1).to(sc.dtype).expand(B, H, n, 1)
    wt = torch.softmax(torch.cat([sc, sink], dim=-1), dim=-1)
    return out_win, torch.einsum("bhij,bhjd->bhid", wt[..., :-1], v_s)


def _gate(H, D, w_gate2, alpha, dtype):
    """A stand-in ``self`` for ``_CCABase._gate_combine_up``: the real method, W_up = I."""
    lin = torch.nn.Linear(w_gate2.shape[1], 2 * H, bias=False).to(w_gate2.device, dtype)
    lin.weight.data.copy_(w_gate2.to(dtype))
    return SimpleNamespace(n_heads=H, d_head=D, latent_q_dim=H * D,
                           gate=[None, torch.nn.SiLU(), lin],
                           alpha=alpha.to(dtype).view(H, 1, 1),
                           W_up=torch.nn.Identity())


class Case:
    """B' query rows of one site kind: kernel inputs + the reference's key sets."""

    def __init__(self, kind, wd, Bq, R, gen, *, rows=None, force=None, wg_dtype=torch.float32):
        H, D, GH, P, M, S = (wd[x] for x in ("H", "D", "GH", "P", "M", "S"))
        self.kind, self.wd, dev = kind, wd, "cuda"
        Cc = M * S                          # Kp = M
        N = {"prelude_token": P + M, "prelude_cell": P + M, "loop_cell": S * M,
             "coda_token": Cc + P, "coda_cell": Cc + P}[kind]
        self.N, self.R = N, R
        rnd = lambda hi: int(torch.randint(0, hi, (1,), generator=gen))  # noqa: E731
        f = dict(force or {})
        k = torch.randn(R, H, N, D, generator=gen)
        v = torch.randn(R, H, N, D, generator=gen)
        self.kpos = None
        if kind == "coda_token":
            # every row's cells at increasing absolute positions: slot s' after a span of
            # 3..8 tokens; Kp cells at consecutive positions (tul_generate_cached `pc`).
            kp = torch.full((R, N), -(10 ** 6), dtype=torch.int32)
            self._first = []
            for r in range(R):
                p, firsts = 5, []
                for s_ in range(S):
                    p += 3 + rnd(6)
                    firsts.append(p)
                    for c in range(M):
                        kp[r, s_ * M + c] = p + c
                    p += M
                self._first.append(firsts)
            self.kpos = kp.to(dev)
        self.q = (torch.randn(Bq, H, D, generator=gen) * 3.0).to(dev)
        self.gateh = torch.randn(Bq, GH, generator=gen).to(dev)
        self.wg2 = (torch.randn(2 * H, GH, generator=gen) * 0.1).to(dev, wg_dtype)
        self.xres = torch.randn(Bq, H * D, generator=gen).to(dev)
        self.sink = torch.randn(H, generator=gen).to(dev)
        self.alpha = torch.randn(H, generator=gen).to(dev)
        meta, self.spec = [], []
        for b in range(Bq):
            row = rows[b] if rows is not None else b % (R - 1)    # row R-1: the dummy
            if kind == "prelude_token":
                j = f.get("j", rnd(P))
                base = 100 + rnd(50)
                ent = list(range(j + 1))                         # incl. itself (stored first)
                pos = [base + e for e in ent]
                slot = [False] * len(ent)
                qpos, so, cells = base + j, False, 0
                meta.append([row, 0, 0, 0, j, 0, j, 0])
            elif kind == "prelude_cell":
                n = f.get("n", 1 + rnd(P))
                i = f.get("i", rnd(M))
                base = 100 + rnd(50)
                ent = list(range(n)) + [P + c for c in range(i + 1)]
                pos = [base + e for e in range(n)] + [base + n + c for c in range(i + 1)]
                slot = [False] * n + [True] * (i + 1)
                qpos, so, cells = base + n + i, False, 0
                meta.append([row, P, P + i + 1, 0, n, P + i, n + i, 0])
            elif kind == "loop_cell":
                s = f.get("s", rnd(S))
                i = f.get("i", rnd(M))
                ent = list(range((s + 1) * M))
                pos = list(ent)                                  # compact cell index
                slot = [True] * len(ent)
                qpos, so, cells = s * M + i, False, M
                meta.append([row, 0, (s + 1) * M, 0, 0, s * M + i, 0, 0])
            elif kind == "coda_token":
                ns = f.get("ns", rnd(S + 1))                     # slots written
                c = ns * M
                j = f.get("j", rnd(P))
                kp_row = self.kpos[row].tolist()
                last = kp_row[c - 1] if c else 5
                p = last + 1 + j + f.get("gap", rnd(4))           # the span after the cells
                ent = list(range(c)) + [Cc + t for t in range(j + 1)]
                pos = kp_row[:c] + [p - j + t for t in range(j + 1)]
                slot = [True] * c + [False] * (j + 1)
                qpos, so, cells = p, False, 0
                meta.append([row, 0, c, Cc, Cc + j, 0, Cc + j, p])
            else:                                                # coda_cell
                s = f.get("s", rnd(S))
                kk = f.get("k", rnd(M))
                first = 200 + rnd(50)
                ent = [s * M + c for c in range(M)]              # the slot's Kp cells
                pos = [first + c for c in range(M)]
                slot = [True] * M
                qpos, so, cells = first + kk, True, 0
                meta.append([row, s * M + kk, s * M + kk + 1, 0, 0, s * M + kk, 0, 0])
            self.spec.append((row, ent, pos, slot, qpos, so, cells))
        self.meta = torch.tensor(meta, dtype=torch.int32, device=dev)
        # entries no query reads (other rows, the dummy row, entries past a range) are
        # NOISE x larger: a stray read moves the output by far more than TOL
        covered = torch.zeros(R, N, dtype=torch.bool)
        for row, ent, *_ in self.spec:
            covered[row, ent] = True
        far = (~covered).view(R, 1, N, 1)
        self.k = torch.where(far, k * NOISE, k).to(dev)
        self.v = torch.where(far, v * NOISE, v).to(dev)

    # ── the reference ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def reference(self, dtype):
        tgc = _ref_mod()
        H, D, W = self.wd["H"], self.wd["D"], self.wd["W"]
        impl = SimpleNamespace(cca=SimpleNamespace(d_head=D, window_size=W,
                                                   sink_logits=self.sink.to(dtype)))
        g = _gate(H, D, self.wg2, self.alpha, dtype)
        real = tgc._branches
        if dtype == torch.float64:
            tgc._branches = _branches_fp64
        outs = []
        try:
            for b, (row, ent, pos, slot, qpos, so, cells) in enumerate(self.spec):
                e = torch.tensor(ent, dtype=torch.long, device="cuda")
                keys = tgc._KV(k=self.k[row][:, e][None].to(dtype),
                               v=self.v[row][:, e][None].to(dtype),
                               pos=torch.tensor(pos, dtype=torch.long, device="cuda"),
                               slot=torch.tensor(slot, dtype=torch.bool, device="cuda"))
                qq = self.q[b].to(dtype)[None, :, None, :]
                ow, oc = tgc._attend(impl, qq, keys,
                                     torch.tensor([qpos], dtype=torch.long, device="cuda"),
                                     so, cells)
                x = torch.zeros(1, 1, 1, device="cuda", dtype=dtype)
                out = _combine()(
                    g, x, oc, ow, q_lat=self.xres[b].to(dtype).view(1, 1, -1),
                    gate_pre=self.gateh[b].to(dtype).view(1, 1, -1))
                outs.append(out.view(-1))
        finally:
            tgc._branches = real
        return torch.stack(outs)

    def kernel(self, lx_attn=None):
        if lx_attn is None:
            from morph.kernels.triton.lx_decode_attn import lx_attn
        D = self.wd["D"]
        return lx_attn(self.q, self.k, self.v, self.meta, self.kpos, self.gateh, self.wg2,
                       self.xres, self.sink, self.alpha, self.wd["W"], D ** -0.5)


def _check(case, tol=TOL):
    ref64 = case.reference(torch.float64)
    ref32 = case.reference(torch.float32)
    got = case.kernel()
    torch.cuda.synchronize()
    assert torch.isfinite(got).all(), f"{case.kind}: non-finite kernel output"
    e32 = float((ref32.double() - ref64).abs().max())
    ek = float((got.double() - ref64).abs().max())
    msg = f"{case.kind} {case.wd['D']}: |ref32-ref64| {e32:.2e}, |kernel-ref64| {ek:.2e}"
    print(msg)
    assert e32 < 1e-4, "fp64 truth and fp32 reference disagree: " + msg
    assert ek < tol, msg
    return e32, ek


# ── tests ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("width", list(WIDTHS))
def test_fp64_reference_is_the_same_function(width):
    """The fp64 truth is _branches' function: agreement to fp32 rounding on every kind."""
    gen = torch.Generator().manual_seed(11)
    for kind in KINDS:
        c = Case(kind, WIDTHS[width], 6, 4, gen)
        d = float((c.reference(torch.float32).double() - c.reference(torch.float64))
                  .abs().max())
        assert d < 1e-4, (kind, d)


@pytest.mark.parametrize("width", list(WIDTHS))
@pytest.mark.parametrize("kind", KINDS)
def test_site_kinds_match_reference(kind, width):
    gen = torch.Generator().manual_seed(hash((kind, width)) % 1000)
    _check(Case(kind, WIDTHS[width], 16, 9, gen))


@pytest.mark.parametrize("width", list(WIDTHS))
def test_empty_ranges_are_exactly_the_residual(width):
    """Prelude token at offset 0 and a coda token with no cell and offset 0: both ranges
    empty -> slot branch 0 (sink only), window branch 0, out = alpha * xres bit for bit."""
    gen = torch.Generator().manual_seed(3)
    wd = WIDTHS[width]
    for kind, force in (("prelude_token", dict(j=0)), ("coda_token", dict(ns=0, j=0))):
        c = Case(kind, wd, 8, 9, gen, force=force)
        got = c.kernel()
        want = (c.alpha.view(1, -1, 1) * c.xres.view(8, wd["H"], wd["D"])).view(8, -1)
        assert torch.isfinite(got).all()
        assert torch.equal(got, want), float((got - want).abs().max())
        _check(c)


@pytest.mark.parametrize("width", list(WIDTHS))
def test_window_shorter_than_range1(width):
    """Loop cells of the last slot and coda tokens after every slot: range 1 holds S*M
    keys, more than the window (388 > 256, 40 > 16), so the window cuts range 1."""
    wd = WIDTHS[width]
    assert wd["S"] * wd["M"] > wd["W"]
    gen = torch.Generator().manual_seed(5)
    _check(Case("loop_cell", wd, 8, 3, gen, force=dict(s=wd["S"] - 1)))
    c = Case("coda_token", wd, 8, 3, gen, force=dict(ns=wd["S"]))
    span = [c.spec[b][2][c.spec[b][3].index(True)] for b in range(8)]
    assert all(c.spec[b][4] - span[b] > wd["W"] for b in range(8))  # first cell outside
    _check(c)


@pytest.mark.parametrize("width", list(WIDTHS))
def test_loop_reads_later_siblings(width):
    """Loop cell (s=0, i=0): its only window keys are its LATER siblings 1..M-1."""
    wd = WIDTHS[width]
    gen = torch.Generator().manual_seed(7)
    c = Case("loop_cell", wd, 4, 3, gen, force=dict(s=0, i=0))
    _check(c)
    # the siblings carry weight: zeroing their values moves the output
    base = c.kernel().clone()
    c.v[:, :, 1:wd["M"]] = 0.0
    assert float((c.kernel() - base).abs().max()) > 1e-3


@pytest.mark.parametrize("width", list(WIDTHS))
def test_kpos_mode_differs_from_index_distance(width):
    """Coda token: the window over the cells is by absolute position (kpos); an index
    distance would select a different set, so the test is sensitive to the mode."""
    wd = WIDTHS[width]
    gen = torch.Generator().manual_seed(9)
    c = Case("coda_token", wd, 8, 5, gen, force=dict(ns=wd["S"] // 2, gap=0))
    for b in range(8):
        row, ent, pos, slot, qpos, _, _ = c.spec[b]
        by_pos = {e for e, p, sl in zip(ent, pos, slot) if sl and 0 < qpos - p < wd["W"]}
        by_idx = {e for e, sl in zip(ent, slot) if sl and 0 < e < wd["W"]}
        assert by_pos != by_idx
    _check(c)


@pytest.mark.parametrize("width", list(WIDTHS))
@pytest.mark.parametrize("kind", KINDS)
def test_meta_rows_point_at_other_cache_rows(kind, width):
    """R > 1, query row b reads cache row meta[b, 0] (a permutation, repeats, the last
    row), never row b. Every other row is noise of magnitude NOISE."""
    wd = WIDTHS[width]
    gen = torch.Generator().manual_seed(13)
    rows = [5, 0, 5, 2, 6, 1, 3, 3]
    _check(Case(kind, wd, 8, 7, gen, rows=rows))


@pytest.mark.parametrize("width", list(WIDTHS))
def test_bf16_gate_weight(width):
    gen = torch.Generator().manual_seed(17)
    _check(Case("coda_token", WIDTHS[width], 8, 9, gen, wg_dtype=torch.bfloat16))


@pytest.mark.parametrize("width", list(WIDTHS))
def test_graph_replay_follows_meta(width):
    """One captured launch, three replays with meta (and kpos) rewritten in place."""
    from morph.kernels.triton.lx_decode_attn import lx_attn
    wd = WIDTHS[width]
    gen = torch.Generator().manual_seed(19)
    cases = [Case("coda_token", wd, 8, 5, gen) for _ in range(3)]
    c0 = cases[0]
    st = dict(q=c0.q.clone(), k=c0.k.clone(), v=c0.v.clone(), meta=c0.meta.clone(),
              kpos=c0.kpos.clone(), gateh=c0.gateh.clone(), wg2=c0.wg2.clone(),
              xres=c0.xres.clone(), sink=c0.sink.clone(), alpha=c0.alpha.clone())

    def run():
        return lx_attn(st["q"], st["k"], st["v"], st["meta"], st["kpos"], st["gateh"],
                       st["wg2"], st["xres"], st["sink"], st["alpha"], wd["W"],
                       wd["D"] ** -0.5)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()                                           # compile + warm outside capture
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    metas = []
    for c in cases:
        for name in st:
            st[name].copy_(getattr(c, name))
        graph.replay()
        torch.cuda.synchronize()
        ref64 = c.reference(torch.float64)
        ek = float((out.double() - ref64).abs().max())
        assert ek < TOL, ek
        metas.append(c.meta.clone())
    assert not torch.equal(metas[0], metas[1])          # the replays saw different ranges

