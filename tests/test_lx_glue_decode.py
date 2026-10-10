"""The decoder's glue kernels (``morph/kernels/triton/lx_glue_decode.py``).

Each piece against a float64 reference of the PyTorch code it replaces in
``morph/inference/lx_fused_decode.py`` (``_readout``, ``_logp``, ``_pointer``, the
``argmax``):

* the entry ``tok_entry`` + ``tok_x0`` (embedding row from the int-code head table or an
  fp32 table, the hashed bigram row, the span buffers, every layer's injection term with and
  without value embeddings, the first carrier, the RoPE rows) and the hand-off ``tok_mid``
  (input_norm per stream, the register's stream mean, the coda carrier), against the
  decoder's PyTorch formulas (``_token_step_engine`` / ``_terms``) in float64;
* the slot loop's glue against ``_loop_torch`` / ``_prefix_torch``'s formulas in float64:
  ``cell_norm`` (rms cell norm, the router's layer-normed stream mean), ``route``
  (``FanRouter.scores`` + ``select``'s winner, the bias choosing, the first index among
  equal scores), ``loop_entry`` (the winner reset, the injection, the first core layer's
  HC partials per channel tile) and ``prefix_write`` (cell k through ``W_prefix[k]``, only
  the winner's cell when a winner is given);
* ``readout_gemv`` (stream mean x lm_scale, ternary mixer) and ``ptr_proj`` (q | k | gate);
* the pointer chain ``ptr_attn`` -> ``row_lse`` -> ``ptr_scatter`` -> ``mix_logp`` ->
  ``argmax_rows`` for B = 3 rows at different key counts (0 keys, inside one block, across
  blocks, the cache's last entry), candidate tokens that repeat (the scatter must add
  them), two consecutive steps on the same buffers (the dense buffer must be zero again),
  and the state it writes: the candidate update, the key stored at ti.

Error metric for log-prob rows: max |got - ref| over the finite entries, the -inf pattern
equal; TOL 2e-5 (fp32 sums of length <= 1024 and exp / log within a few ulp).

MUTANTS: the kernel source with one edit, loaded as its own module; the check must FAIL.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="Triton kernels on CUDA")

import morph.kernels.triton.lx_glue_decode as G  # noqa: E402
from morph.kernels.triton.fused_decode_step import ternary_pack  # noqa: E402

TOL = 2e-5
H, DH = 4, 64


def _rn(*s, gen, sc=1.0):
    return (torch.randn(*s, generator=gen) * sc).cuda()


def _unset(*shape, dtype=torch.float32) -> torch.Tensor:
    """A checked output buffer, filled with 1e9 (-5 for an index): an entry the kernel does
    not write fails the check (``torch.empty`` can hand back an earlier correct result)."""
    return torch.full(shape, -5 if dtype == torch.long else 1e9, device="cuda", dtype=dtype)


def _rel(got, ref):
    ref = ref.double()
    return float((got.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


# ── entry + hand-off ──────────────────────────────────────────────────────────────────


def check_entry(mod, codes=True, nve=0, C=1024, B=3, seed=0):
    """tok_entry + tok_x0 against the decoder's formulas; returns the worst relative error
    over x2, bg, the span rows, every term row, the carrier, cos / sin."""
    from morph.model.embeddings import _BIGRAM_HASH_A, _BIGRAM_HASH_B
    gen = torch.Generator().manual_seed(seed)
    V, HV, NL, P, R, D, N = 300, 500, 6, 8, B + 1, 32, 4
    CE = C * 3 // 4
    cs, cw = C // 4, 96
    tok = torch.randint(0, V, (B,), generator=gen).cuda()
    prev = torch.randint(0, V, (B,), generator=gen).cuda()
    pos = torch.randint(0, 50, (B,), generator=gen).cuda()
    j = torch.randint(0, P, (B,), generator=gen).cuda()
    code = torch.randint(-31, 32, (V, CE), generator=gen, dtype=torch.int8).cuda()
    s = torch.rand(V, generator=gen).cuda() * 0.01
    wl = _rn(V, C - CE, gen=gen, sc=0.05)
    full = torch.cat([code.float() * s[:, None], wl], 1)
    table = (code, s, wl) if codes else (full.contiguous(),)
    bgt = _rn(HV, C, gen=gen, sc=0.02)
    lam = _rn(NL, gen=gen, sc=0.5)
    cosf, sinf = _rn(64, D, gen=gen), _rn(64, D, gen=gen)
    w = (torch.randint(-1, 2, (NL * cw, C), generator=gen).float() * 0.04).cuda()
    codes_x0, gamma = ternary_pack(w)
    rs = (gamma.expand(NL * cw) * (1 + 0.2 * torch.rand(NL * cw, generator=gen).cuda()))
    rs = rs.contiguous()
    x0s = _rn(NL, gen=gen, sc=0.4)
    ve = _rn(nve, B, cw, gen=gen) if nve else None
    x2, bg2 = _unset(B, C), _unset(B, C)
    spe, spb = torch.full((R, P, C), 777.0, device="cuda"), torch.full((R, P, C), 777.0,
                                                                        device="cuda")
    terms = torch.full((NL, B, C), 777.0, device="cuda")
    h0 = torch.full((B, N, C), 777.0, device="cuda")
    cos, sin = _unset(B, D), _unset(B, D)
    mod.tok_entry(tok, prev, pos, j, table, bgt, lam, cosf, sinf, cs, cs + cw, x2, bg2, spe,
                  spb, terms, h0, cos, sin)
    mod.tok_x0(x2, bg2, codes_x0, rs, x0s, lam, cs, terms, h0, ve)
    # the decoder's formulas, float64
    x_ref = full[tok].double()
    hsh = (_BIGRAM_HASH_A * tok ^ _BIGRAM_HASH_B * prev) % HV
    bg_ref = bgt[hsh].double()
    x0f = (x_ref @ (w / gamma).double().T) * rs.double()
    x0t = x0f.view(B, NL, cw).permute(1, 0, 2) * x0s.double().view(NL, 1, 1)
    if nve:
        x0t = torch.cat([x0t[:nve] + ve.double(), x0t[nve:]])
    t_ref = lam.double().view(NL, 1, 1) * bg_ref.unsqueeze(0)
    t_ref[..., cs:cs + cw] += x0t
    h_ref = (x_ref + t_ref[0]).unsqueeze(1).expand(B, N, C)
    errs = [_rel(x2, x_ref), _rel(bg2, bg_ref), _rel(terms, t_ref), _rel(h0, h_ref),
            _rel(cos, cosf[pos]), _rel(sin, sinf[pos])]
    ar = torch.arange(B)
    errs += [_rel(spe[ar, j], x_ref), _rel(spb[ar, j], bg_ref)]
    keep = torch.ones(R, P, dtype=torch.bool)
    keep[ar, j.cpu()] = False
    if not bool((spe[keep.cuda()] == 777.0).all() and (spb[keep.cuda()] == 777.0).all()):
        return float("inf")                            # a span entry it must not touch
    return max(errs)


def check_mid(mod, C=1024, B=3, seed=0, xn=True):
    gen = torch.Generator().manual_seed(seed)
    N, P, R = 4, 8, B + 1
    h = _rn(B, N, C, gen=gen)
    nw = _rn(C, gen=gen, sc=0.2) + 1.0
    tb = _rn(B, 2, C, gen=gen)
    term = tb[:, 1]                                    # a strided term row
    j = torch.randint(0, P, (B,), generator=gen).cuda()
    xc = _unset(B, N, C)
    span_xn = torch.full((R, P, C), 777.0, device="cuda") if xn else None
    mod.tok_mid(h, nw, 1e-6, term, xc, span_xn, j)
    hd = h.double()
    e = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.double()
    err = _rel(xc, e + term.double().unsqueeze(1))
    if xn:
        err = max(err, _rel(span_xn[torch.arange(B), j], e.mean(1)))
    return err


@pytest.mark.parametrize("codes", [True, False])
@pytest.mark.parametrize("nve", [0, 2])
@pytest.mark.parametrize("C", [1024, 256])
def test_entry_matches_fp64(codes, nve, C):
    e = check_entry(G, codes, nve, C)
    print(f"entry codes={codes} nve={nve} C={C}: {e:.2e}")
    assert e < TOL, e


@pytest.mark.parametrize("C", [1024, 256])
def test_mid_matches_fp64(C):
    e = max(check_mid(G, C), check_mid(G, C, xn=False))
    print(f"mid C={C}: {e:.2e}")
    assert e < TOL, e


# ── slot-loop glue ────────────────────────────────────────────────────────────────────


def check_cell_norm(mod, C=1024, Q=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    N = 4
    h = _rn(Q, N, C, gen=gen)
    nw = _rn(C, gen=gen, sc=0.2) + 1.0
    hn, ln = _unset(Q, N, C), _unset(Q, C)
    mod.cell_norm(h, nw, 1e-6, hn, ln)
    hd = h.double()
    e = hd * torch.rsqrt(hd.pow(2).mean(-1, keepdim=True) + 1e-6) * nw.double()
    lr = torch.nn.functional.layer_norm(e.mean(1), (C,), eps=1e-5)
    hn2 = _unset(Q, N, C)
    mod.cell_norm(h, nw, 1e-6, hn2, None)
    return max(_rel(hn, e), _rel(ln, lr), _rel(hn2, e))


def _route_case(gen, nb, M, RK, C, ties=False):
    ln = _rn(nb * M, C, gen=gen)
    if ties:                                   # cells 1 and 3 of every slot identical
        ln.view(nb, M, C)[:, 3] = ln.view(nb, M, C)[:, 1]
    wc = _rn(RK, C, gen=gen, sc=C ** -0.5)
    xp = _rn(nb, RK, gen=gen, sc=0.5)
    v = _rn(RK, gen=gen, sc=RK ** -0.5)
    bias = torch.zeros(M, device="cuda") if ties else _rn(M, gen=gen, sc=0.3)
    return ln, wc, xp, v, bias


def check_route(mod, C=1024, nb=64, M=4, RK=64, seed=0):
    """Returns 0.0 when every winner equals the float64 reference's (and, on the tie
    case, every winner the reference picks is the lower of the two equal cells), inf when
    one differs. 64 slots: the bias moves about a third of the winners on this draw."""
    gen = torch.Generator().manual_seed(seed)
    bad = 0
    for ties in (False, True):
        ln, wc, xp, v, bias = _route_case(gen, nb, M, RK, C, ties)
        pick = torch.full((nb,), -7, dtype=torch.int64, device="cuda")
        sp = torch.empty(nb * M, RK // 8, device="cuda")
        mod.route(ln, wc, xp, v, bias, pick, sp)
        a = ln.double().view(nb, M, C) @ wc.double().T + xp.double().view(nb, 1, RK)
        sc = torch.relu(a) @ v.double()
        ref = (sc + bias.double()).argmax(-1)           # the first index of the maximum
        if ties:                                         # the kernel's exact tie: 1 vs 3
            ref = torch.where(ref == 3, torch.ones_like(ref), ref)
        bad += int((pick != ref).sum())
    return float("inf") if bad else 0.0


def check_loop_entry(mod, C=1024, nb=3, M=4, seed=0, reset=True):
    gen = torch.Generator().manual_seed(seed)
    N, Q = 4, nb * M
    NT = C // mod.PART_TILE
    src, e = _rn(Q, N, C, gen=gen), _rn(Q, N, C, gen=gen)
    lo, hi = C // 4, C // 4 + 96
    A = torch.rand(hi - lo, generator=gen).cuda()
    dt = torch.rand(hi - lo, generator=gen).cuda()
    tb = _rn(nb, 2, C, gen=gen)
    term = tb[:, 0]                                      # a strided term row
    w = _rn(mod.HC_NO, N * C, gen=gen, sc=0.01)
    pick = torch.randint(0, M, (nb,), generator=gen).cuda()
    out = _unset(Q, N, C)
    rawp = _unset(Q, NT, mod.HC_NO)
    mod.loop_entry(src, pick if reset else None, e, A, dt, term, lo, hi, w, out, rawp, M)
    sd = src.double().view(nb, M, N, C)
    if reset:
        sd = sd[torch.arange(nb), pick].unsqueeze(1).expand(nb, M, N, C)
    sd = sd.reshape(Q, N, C)
    ref = sd.clone()
    ref[..., lo:hi] = A.double() * sd[..., lo:hi] + dt.double() * e.double()[..., lo:hi]
    ref = ref + term.double().repeat_interleave(M, 0).unsqueeze(1)
    pr = torch.einsum("oitc,qitc->qto", w.double().view(-1, N, NT, mod.PART_TILE),
                      ref.view(Q, N, NT, mod.PART_TILE))
    return max(_rel(out, ref), _rel(rawp, pr))


def check_prefix(mod, C=1024, nb=3, KP=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    N, Q = 4, nb * KP
    hn = _rn(Q, N, C, gen=gen)
    w = _rn(KP, C, C, gen=gen, sc=C ** -0.5)
    pick = torch.randint(0, KP, (nb,), generator=gen).cuda()
    err = 0.0
    for pk in (pick, None):
        out = torch.full((Q, N, C), 777.0, device="cuda")
        mod.prefix_write(hn, pk, w, out)
        x = hn.double().view(nb, KP, N, C)
        if pk is not None:
            x = x * (torch.arange(KP, device="cuda").view(1, KP) == pk.view(nb, 1)
                     ).double().view(nb, KP, 1, 1)
        ref = torch.einsum("bkic,kco->bkio", x, w.double()).reshape(Q, N, C)
        if pk is not None and not bool((out.view(nb, KP, N, C)[
                torch.arange(KP, device="cuda").view(1, KP) != pk.view(nb, 1)] == 0).all()):
            return float("inf")                         # a masked cell must read exactly 0
        err = max(err, _rel(out, ref))
    return err


@pytest.mark.parametrize("C", [1024, 256])
def test_slot_glue_matches_fp64(C):
    e = {"cell_norm": check_cell_norm(G, C), "route": check_route(G, C),
         "loop_entry": max(check_loop_entry(G, C), check_loop_entry(G, C, reset=False)),
         "prefix": check_prefix(G, C)}
    print(f"slot glue C={C}: " + ", ".join(f"{k} {v:.2e}" for k, v in e.items()))
    assert max(e.values()) < TOL, e


# ── readout + ptr_proj ──────────────────────────────────────────────────────────────


def check_readout(mod, C=1024, B=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w = (torch.randint(-1, 2, (C, C), generator=gen).float() * 0.05).cuda()
    codes, gamma = ternary_pack(w)
    rs = (gamma.expand(C) * (1 + 0.1 * torch.rand(C, generator=gen).cuda())).contiguous()
    hb = _rn(B, 2, 4, C, gen=gen)                       # rows of a wider buffer (stride)
    h = hb[:, 1]
    sc = _rn(C, gen=gen, sc=0.3).abs() + 0.5
    out = torch.full((B, C), 777.0, device="cuda")
    mod.readout_gemv(h, sc, codes, rs, out)
    xs = h.double().mean(1) * sc.double()
    ref = (xs @ (w / gamma).double().T) * rs.double()
    return _rel(out, ref)


def check_proj(mod, C=1024, B=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    nqk = 2 * H * DH
    w = _rn(nqk + 1 + H, C, gen=gen, sc=C ** -0.5)
    gb = _rn(1 + H, gen=gen)
    x = _rn(B, C, gen=gen)
    out = _unset(B, nqk + 1 + H)
    mod.ptr_proj(x, w, gb, nqk, out)
    ref = x.double() @ w.double().T
    ref[:, nqk:] += gb.double()
    return _rel(out, ref)


@pytest.mark.parametrize("C", [1024, 256])
def test_readout_and_proj(C):
    e1, e2 = check_readout(G, C), check_proj(G, C)
    print(f"C={C}: readout {e1:.2e} proj {e2:.2e}")
    assert e1 < TOL and e2 < TOL


# ── the pointer + log-softmax + argmax chain ──────────────────────────────────────────


class _State:
    """The decoder's pointer state for B rows (keys, candidates) and one step's inputs."""

    def __init__(self, B, NK, V, gen):
        self.B, self.NK, self.V = B, NK, V
        self.keys = _rn(B, H, NK, DH, gen=gen)
        self.cand = torch.randint(5, 40, (B, NK), generator=gen).cuda()   # few ids: repeats
        self.null_k = _rn(H, DH, gen=gen)
        self.null_k = self.null_k * torch.rsqrt(self.null_k.pow(2).mean(-1, keepdim=True) + 1e-6)
        self.scale = (torch.exp(_rn(H, gen=gen, sc=0.2)) / math.sqrt(DH)).contiguous()
        self.p_dense = torch.zeros(B, V, device="cuda")


def _ref_step(st, proj, logits, ti, ti_m1, tok, slot):
    """float64 ``_logp`` + ``_pointer`` + argmax, the decoder's code line for line."""
    B, NK = st.B, st.NK
    cand = st.cand.clone()
    cand[torch.arange(B), ti_m1] = tok
    p = proj.double()
    nqk = H * DH
    rms = lambda x: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)  # noqa: E731
    q = rms(p[:, :nqk].view(B, H, DH))
    k = rms(p[:, nqk:2 * nqk].view(B, H, DH))
    lg = logits.double().clone()
    lg[:, slot] = float("-inf")
    lm = torch.log_softmax(lg, -1)
    scale = st.scale.double()
    valid = torch.arange(NK, device="cuda").view(1, -1) < ti.view(B, 1)
    s = torch.einsum("bhd,bhnd->bhn", q, st.keys.double()) * scale.view(1, -1, 1)
    s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
    s_null = (q * st.null_k.double()).sum(-1) * scale.view(1, -1)
    pr = torch.softmax(torch.cat([s, s_null.unsqueeze(-1)], -1), -1)
    alpha, null = pr[..., :-1], pr[..., -1]
    a = torch.softmax(p[:, 2 * nqk:], -1)
    w0 = a[:, 0] + (a[:, 1:] * null).sum(-1)
    beta = (alpha * a[:, 1:, None]).sum(1) * valid.double()
    p_ptr = torch.zeros_like(lm).scatter_add_(1, cand.clamp_min(0), beta)
    out = torch.log(w0.view(B, 1) * lm.exp() + p_ptr)
    keys = st.keys.clone()
    keys[torch.arange(B), :, ti] = k.float()
    return out, out.argmax(-1), cand, keys


def _run_step(mod, st, proj, logits, ti, ti_m1, tok, slot):
    B, NK, V = st.B, st.NK, st.V
    hp = 1 << (H - 1).bit_length()
    s_scr = torch.empty(B, hp, NK, device="cuda")
    beta = torch.empty(B, NK, device="cuda")
    w0 = torch.empty(B, device="cuda")
    mod.ptr_attn(proj, st.keys, st.cand, ti, ti_m1, tok, st.null_k, st.scale, s_scr, beta, w0)
    mx, ls = torch.empty(B, device="cuda"), torch.empty(B, device="cuda")
    mod.row_lse(logits, slot, mx, ls)
    mod.ptr_scatter(st.cand, beta, ti, st.p_dense)
    nt = -(-V // mod.MIX_BLOCK)
    pav = torch.empty(B, nt, device="cuda")
    pai = torch.empty(B, nt, device="cuda", dtype=torch.int32)
    out = torch.full((B, V), 777.0, device="cuda")
    mod.mix_logp(logits, mx, ls, w0, st.p_dense, slot, out, pav, pai)
    nxt = _unset(B, dtype=torch.long)
    mod.argmax_rows(pav, pai, nxt)
    return out, nxt


def _diff(got, ref):
    fg, fr = torch.isfinite(got), torch.isfinite(ref)
    if not torch.equal(fg, fr):
        return float("inf")
    return float((got.double()[fg] - ref[fr]).abs().max())


def check_chain(mod, V=5000, NK=150, seed=0, steps=2):
    """Two steps on the same state; returns the worst log-prob difference, and raises on
    a wrong token, candidate table or key cache."""
    gen = torch.Generator().manual_seed(seed)
    B = 3
    st = _State(B, NK, V, gen)
    slot = 4
    worst = 0.0
    tis = [torch.tensor([0, 37, NK - 1]), torch.tensor([1, 38, 70])]
    for k in range(steps):
        ti = tis[k].cuda()
        ti_m1 = (ti - 1).clamp_min(0)
        tok = torch.randint(5, 40, (B,), generator=gen).cuda()
        proj = _rn(B, 2 * H * DH + 1 + H, gen=gen)
        logits = _rn(B, V, gen=gen, sc=3.0)
        logits[1, 17] = logits[1].max() + 5.0          # a clear winner on row 1
        ref, ref_tok, ref_cand, ref_keys = _ref_step(st, proj, logits, ti, ti_m1, tok, slot)
        out, nxt = _run_step(mod, st, proj, logits, ti, ti_m1, tok, slot)
        worst = max(worst, _diff(out, ref))
        if not torch.equal(nxt, ref_tok):
            return float("inf")
        if not torch.equal(st.cand, ref_cand):
            return float("inf")
        worst = max(worst, float((st.keys - ref_keys).abs().max()))
        if bool((st.p_dense != 0).any()):
            return float("inf")
        st.keys = ref_keys.clone()                     # the next step starts from the reference
    return worst


def test_chain_matches_fp64():
    e = check_chain(G)
    print(f"chain worst: {e:.2e}")
    assert e < TOL, e


def test_no_pointer_is_log_softmax():
    gen = torch.Generator().manual_seed(5)
    B, V, slot = 2, 3000, 7
    logits = _rn(B, V, gen=gen, sc=4.0)
    mx, ls = torch.empty(B, device="cuda"), torch.empty(B, device="cuda")
    G.row_lse(logits, slot, mx, ls)
    nt = -(-V // G.MIX_BLOCK)
    pav, pai = torch.empty(B, nt, device="cuda"), torch.empty(B, nt, device="cuda",
                                                              dtype=torch.int32)
    out = _unset(B, V)
    G.mix_logp(logits, mx, ls, None, None, slot, out, pav, pai)
    nxt = _unset(B, dtype=torch.long)
    G.argmax_rows(pav, pai, nxt)
    lg = logits.double().clone()
    lg[:, slot] = float("-inf")
    ref = torch.log_softmax(lg, -1)
    assert _diff(out, ref) < TOL and torch.equal(nxt, ref.argmax(-1))


def test_argmax_takes_the_first_of_equal_maxima():
    B, V = 2, 9000
    logits = torch.zeros(B, V, device="cuda")
    logits[0, [10, 4000, 8999]] = 3.0                  # across three mix tiles
    logits[1, [2047, 2048]] = 3.0                      # either side of a tile edge
    mx, ls = torch.empty(B, device="cuda"), torch.empty(B, device="cuda")
    G.row_lse(logits, 5000, mx, ls)
    nt = -(-V // G.MIX_BLOCK)
    pav, pai = torch.empty(B, nt, device="cuda"), torch.empty(B, nt, device="cuda",
                                                              dtype=torch.int32)
    out = _unset(B, V)
    G.mix_logp(logits, mx, ls, None, None, 5000, out, pav, pai)
    nxt = _unset(B, dtype=torch.long)
    G.argmax_rows(pav, pai, nxt)
    assert nxt.tolist() == [10, 2047] == out.argmax(-1).tolist()


def _token_argmax(mod, logits, slot, mx=None, ls=None):
    B, V = logits.shape
    if mx is None:
        mx, ls = torch.empty(B, device="cuda"), torch.empty(B, device="cuda")
        mod.row_lse(logits, slot, mx, ls)
    nt = -(-V // mod.MIX_BLOCK)
    pav, pai = torch.empty(B, nt, device="cuda"), torch.empty(B, nt, device="cuda",
                                                              dtype=torch.int32)
    out = _unset(B, V)
    mod.mix_logp(logits, mx, ls, None, None, slot, out, pav, pai)
    nxt = _unset(B, dtype=torch.long)
    mod.argmax_rows(pav, pai, nxt)
    return out, nxt


def check_nan(mod) -> float:
    """0.0 when every NaN case picks what ``torch.argmax`` picks, inf otherwise.
    ``torch.argmax`` returns the first NaN of a row that has one; before 2026-10-10 an
    all-NaN row gave next_tok = 2**31 - 1 (out of range) and all-NaN router scores gave
    PICK = M (another slot's cell, or past the buffer)."""
    V = 9000
    logits = torch.randn(3, V, generator=torch.Generator().manual_seed(0)).cuda()
    logits[0, 4321] = float("nan")                  # one NaN: log_softmax makes the row NaN
    logits[1] = float("nan")                         # all NaN
    out, nxt = _token_argmax(mod, logits, 5000)
    ok = nxt.tolist() == out.argmax(-1).tolist()     # NaN rows: torch's first NaN
    ok &= bool((nxt >= 0).all() and (nxt < V).all())
    # finite max in tile 0, one NaN lane in tile 3 (mx / ls given finite): the NaN
    logits2 = torch.zeros(1, V, device="cuda")
    logits2[0, 10], logits2[0, 7000] = 5.0, float("nan")
    out2, nxt2 = _token_argmax(mod, logits2, 5000, torch.zeros(1, device="cuda"),
                               torch.zeros(1, device="cuda"))
    ok &= nxt2.item() == out2.argmax(-1).item() == 7000
    # the router: slot 0 one NaN input in cell 2, slot 1 every cell NaN, slot 2 finite
    nb, M, RK, C = 3, 4, 64, 256
    ln, wc, xp, v, bias = _route_case(torch.Generator().manual_seed(1), nb, M, RK, C)
    ln.view(nb, M, C)[0, 2, 7] = float("nan")
    ln.view(nb, M, C)[1] = float("nan")
    pick = torch.full((nb,), -7, dtype=torch.int64, device="cuda")
    mod.route(ln, wc, xp, v, bias, pick, torch.empty(nb * M, RK // 8, device="cuda"))
    ref = (torch.relu(ln.view(nb, M, C) @ wc.T + xp.view(nb, 1, RK)) @ v + bias).argmax(-1)
    ok &= pick.tolist() == ref.tolist() and ref.tolist()[:2] == [2, 0]
    return 0.0 if ok else float("inf")


def test_nan_rows_follow_torch_argmax():
    assert check_nan(G) == 0.0


# ── mutants ──────────────────────────────────────────────────────────────────────────

_SRC = pathlib.Path(G.__file__).read_text()
MUTANTS = [
    ("chain", "    tl.store(CAND + b * NK + tim1 + nn0, tok + nn0 * 0, mask=nn0 == 0)\n", "",
     "candidate update skipped"),
    ("chain", "alpha = tl.exp(s - m[:, None]) / z[:, None]",
     "alpha = tl.exp(s - m[:, None])", "softmax not normalised"),
    ("chain", "s_null = tl.sum(q * nk, 1) * sc", "s_null = tl.sum(q * nk, 1)",
     "null score unscaled"),
    ("chain", "+ ti * DH + d[None, :], k,", "+ ti * DH + d[None, :], pk,",
     "key stored without its RMS"),
    ("chain", "        tl.store(PD + b * V + v, tl.zeros_like(p), mask=mv)\n", "",
     "dense buffer not re-zeroed"),
    ("chain", "acc += tl.where((cn[:, None] == cm[None, :]) & vm[None, :], bm[None, :], 0.0)",
     "acc += tl.where((n[:, None] == mm[None, :]) & vm[None, :], bm[None, :], 0.0)",
     "repeated candidates not summed"),
    ("chain", "x = tl.load(LOG + b * sl + v, mask=(v < V) & (v != SLOT), other=float(\"-inf\"))\n"
              "        acc += tl.exp(x - m)",
     "x = tl.load(LOG + b * sl + v, mask=(v < V), other=float(\"-inf\"))\n"
     "        acc += tl.exp(x - m)", "slot column inside the softmax sum"),
    ("chain", "    w0 = a0 + tl.sum(tl.where(mh, ah * null, 0.0), 0)\n",
     "    w0 = a0 + 0.0 * tl.sum(tl.where(mh, ah * null, 0.0), 0)\n",
     "abstain mass not returned to the model"),
    ("readout", "s4 = (tl.load(hb + i) + tl.load(hb + C + i))",
     "s4 = (tl.load(hb + i) + tl.load(hb + i))", "stream 1 read as stream 0"),
    ("readout", "xs = (s4 * 0.25) * tl.load(SC + i)", "xs = s4 * 0.25", "lm_scale dropped"),
    ("proj", "y = tl.where(o >= NQK, y + gb, y)", "y = tl.where(o >= NQK, y, y)",
     "gate bias dropped"),
    ("entry", "hsh = ((tok * HA) ^ (prev * HB)) % HV", "hsh = ((tok * HA) ^ (tok * HB)) % HV",
     "bigram hash ignores the previous token"),
    ("entry", "x = tl.where(ke, we.to(tl.float32) * tl.load(SC + tok), wl)",
     "x = tl.where(ke, we.to(tl.float32) * tl.load(SC + prev), wl)", "row scale of prev"),
    ("entry", "    out = (c < CS) | (c >= CEX)\n", "    out = (c < CS) | (c > CEX)\n",
     "first channel after ctx not written"),
    ("entry", "ctx = tl.where(l < NVE, ctx + tl.load(VE",
     "ctx = tl.where(l < NVE - 1, ctx + tl.load(VE",
     "last value-embedding layer dropped"),
    ("entry", "tv = tl.load(LAM + l) * tl.load(BG2 + b * C + c) + ctx",
     "tv = tl.load(LAM + l) * tl.load(BG2 + b * C + c) + x0f", "x0 scale dropped"),
    ("entry", "tl.store(SPE + (b * P + j) * C + c, x)",
     "tl.store(SPE + (b * P + j + 1) * C + c, x)",
     "span row j+1"),
    ("mid", "        tl.store(XN + (b * P + j) * C + c, tl.sum(e, 0) / N, mask=mc)",
     "        tl.store(XN + (b * P + j) * C + c, tl.sum(e, 0) / (N - 1), mask=mc)",
     "register input not a mean"),
    ("mid", "(b * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)\n"
     "    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps)",
     "(b * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)\n"
     "    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps * 1e6)", "norm eps wrong"),
    ("cell_norm", "(q * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)\n"
     "    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps)",
     "(q * N + i[:, None]) * C + c[None, :], mask=mc[None, :], other=0.0)\n"
     "    r = 1.0 / tl.sqrt(tl.sum(h * h, 1) / C + eps * 1e6)", "cell norm eps wrong"),
    ("loop_entry", "    tv = tl.load(TERM + b * sterm + c)\n",
     "    tv = tl.load(TERM + b * C + c)\n",
     "term row stride ignored"),
    ("mid", "tl.store(XC + (b * N + i[:, None]) * C + c[None, :], e + tv[None, :], "
     "mask=mc[None, :])",
     "tl.store(XC + (b * N + i[:, None]) * C + c[None, :], e, mask=mc[None, :])",
     "coda term not added"),
    ("cell_norm", "dv = tl.where(mc, cr - mu, 0.0)", "dv = tl.where(mc, cr, 0.0)",
     "router layer norm not centred"),
    ("cell_norm", "        cr = tl.sum(hn, 0) / N\n", "        cr = tl.sum(h, 0) / N\n",
     "router reads the un-normed cell"),
    ("route", "    sb = sc + tl.load(BIAS + mi)\n", "    sb = sc\n", "balance bias ignored"),
    ("route", "ra = tl.where(a != a, a, tl.maximum(a, 0.0))", "ra = a", "relu dropped"),
    ("route", "a = tl.sum(acc, 1) + tl.load(XP + b * RK + o)",
     "a = tl.sum(acc, 1) + tl.load(XP + o)",
     "context of slot 0 for every slot"),
    ("route", "tl.min(tl.where(sb == best, mi, M), 0)", "tl.max(tl.where(sb == best, mi, -1), 0)",
     "last index on ties"),
    ("loop_entry", "        src = b * M + tl.load(PICK + b)\n", "        src = b * M\n",
     "reset to cell 0"),
    ("loop_entry", "inctx = (c >= lo) & (c < hi)", "inctx = (c >= lo) & (c <= hi)",
     "injection one channel too wide"),
    ("loop_entry", "_hc_part_row(WN, RAWP, q, t, o0, o1, o2, o3,",
     "_hc_part_row(WN, RAWP, q, t, o0, o1, o2, o2,", "partials miss stream 3"),
    ("prefix", "tl.where(kk == k, y, 0.0))", "tl.where(kk >= 0, y, 0.0))", "losers written"),
    ("prefix", "w = tl.load(W + k * (C * C) + d[:, None] * C + o[None, :])",
     "w = tl.load(W + d[:, None] * C + o[None, :])", "every cell through W_prefix[0]"),
    ("nan", "tl.store(OUT + b, tl.where(nidx < 2147483647, nidx, midx).to(tl.int64))",
     "tl.store(OUT + b, midx.to(tl.int64))", "token argmax ignores NaN"),
    ("nan", "tl.store(PAI + b * NT + t, tl.where(hasn, nidx, idx))",
     "tl.store(PAI + b * NT + t, idx)", "tile argmax ignores NaN"),
    ("nan", "tl.store(PICK + b, tl.where(nidx < M, nidx, midx).to(tl.int64))",
     "tl.store(PICK + b, midx.to(tl.int64))", "router pick ignores NaN"),
    ("nan", "ra = tl.where(a != a, a, tl.maximum(a, 0.0))", "ra = tl.maximum(a, 0.0)",
     "router relu drops NaN"),
    ("chain", "    tl.debug_barrier()\n", "    tl.debug_barrier()\n",
     "(control: a no-op edit must NOT be caught)"),
]
CHECKS = {"chain": check_chain, "readout": check_readout, "proj": check_proj,
          "entry": lambda mod: max(check_entry(mod, True, 2), check_entry(mod, False, 0)),
          "mid": check_mid, "cell_norm": check_cell_norm, "route": check_route,
          "loop_entry": lambda mod: max(check_loop_entry(mod), check_loop_entry(mod, reset=False)),
          "prefix": check_prefix, "nan": check_nan}


def _mutant(old, new, tmp_path):
    assert _SRC.count(old) == 1, f"mutant text not unique in the kernel source: {old!r}"
    f = tmp_path / f"glue_mut_{abs(hash((old, new)))}.py"
    f.write_text(_SRC.replace(old, new))
    spec = importlib.util.spec_from_file_location(f.stem, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("check,old,new,what", MUTANTS, ids=[m[3] for m in MUTANTS])
def test_mutant_is_caught(check, old, new, what, tmp_path):
    mod = _mutant(old, new, tmp_path)
    e = CHECKS[check](mod)
    print(f"mutant [{what}]: {e:.2e}")
    if what.startswith("(control"):
        assert e < TOL, e
    else:
        assert e > TOL * 10, (what, e)
