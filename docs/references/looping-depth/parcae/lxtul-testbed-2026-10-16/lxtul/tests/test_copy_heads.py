"""PointerHead and IdentityReach (lxtul/copy_heads.py), CPU.

1. PointerHead's p at the target equals a brute-force loop (token order, slots skipped, null key).
2. PointerHead is causal: future hidden states / labels never move an earlier token's p or gate.
3. IdentityReach is zero at init and at slot positions, causal (output at token i reads raw
   embeddings of tokens <= i only, and no future query), and equals a brute-force loop.
"""
from __future__ import annotations

import math

import torch

from lxtul.copy_heads import IdentityReach, PointerHead, _rms, mix_pointer

B, L, D, H = 2, 14, 16, 2


def _layout(seed=0):
    g = torch.Generator().manual_seed(seed)
    is_slot = torch.zeros(B, L, dtype=torch.bool)
    is_slot[:, [4, 9]] = True
    is_slot[1, 12] = True
    labels = torch.randint(0, 5, (B, L), generator=g)          # small vocab -> repeats
    labels[is_slot] = -100
    labels[0, 2] = -100                                        # an unscored token
    h = torch.randn(B, L, D, generator=g)
    return g, is_slot, labels, h


def _tok_index(is_slot, b):
    return [p for p in range(L) if not is_slot[b, p]]


def test_pointer_matches_brute_force():
    torch.manual_seed(0)
    g, is_slot, labels, h = _layout()
    m = PointerHead(D, H, dh=8)
    with torch.no_grad():
        gl, p, null = m(h, labels, is_slot)
    scale = m.log_scale_bias.exp() / math.sqrt(8)
    for b in range(B):
        tk = _tok_index(is_slot, b)
        q = _rms(m.q(h[b]).view(L, H, 8))
        k = _rms(m.k(h[b]).view(L, H, 8))
        kn = _rms(m.null_k_bias)
        for ii, i in enumerate(tk):
            for hh in range(H):
                js = [j for j in tk[:ii] if labels[b, j] >= 0]
                s = [float(q[i, hh] @ k[j, hh] * scale[hh]) for j in js]
                s_null = float(q[i, hh] @ kn[hh] * scale[hh])
                w = torch.softmax(torch.tensor(s + [s_null]), 0)
                ref = sum(float(w[n]) for n, j in enumerate(js)
                          if labels[b, i] >= 0 and labels[b, j] == labels[b, i])
                assert math.isclose(float(p[b, i, hh]), ref, abs_tol=1e-5), (b, i, hh)
                assert math.isclose(float(null[b, i, hh]), float(w[-1]), abs_tol=1e-5)


def test_pointer_is_causal():
    torch.manual_seed(1)
    g, is_slot, labels, h = _layout(1)
    m = PointerHead(D, H, dh=8)
    with torch.no_grad():
        gl, p, _ = m(h, labels, is_slot)
        t = 8                                                  # packed position
        h2, l2 = h.clone(), labels.clone()
        h2[:, t:] += torch.randn(B, L - t, D, generator=g)
        l2[:, t:] = torch.where(l2[:, t:] >= 0, (l2[:, t:] + 1) % 5, -100)
        gl2, p2, _ = m(h2, l2, is_slot)
    assert torch.equal(p[:, :t], p2[:, :t]) and torch.equal(gl[:, :t], gl2[:, :t])


def test_identity_reach_zero_init_slots_causal_and_brute_force():
    torch.manual_seed(2)
    g, is_slot, _, x = _layout(2)
    emb = torch.randn(B, L, D, generator=g)
    m = IdentityReach(D, H, dh=8)
    with torch.no_grad():
        assert m(x, emb, is_slot).abs().max() == 0                 # o zero-init
        m.o.weight.normal_(0, 0.1)
        out = m(x, emb, is_slot)
        assert out[is_slot].abs().max() == 0
        # causality: change embeddings and queries of tokens from token index t on
        t_tok = 6
        x2, e2 = x.clone(), emb.clone()
        for b in range(B):
            tk = _tok_index(is_slot, b)
            for p in tk[t_tok:]:
                x2[b, p] += 1.0
                e2[b, p] += 1.0
        out2 = m(x2, e2, is_slot)
        for b in range(B):
            tk = _tok_index(is_slot, b)
            # token index i reads e(x_{j+1}) for j < i, i.e. up to e(x_i): i < t_tok is untouched
            for i in tk[:t_tok]:
                assert torch.equal(out[b, i], out2[b, i]), (b, i)
            assert not torch.equal(out[b, tk[t_tok]], out2[b, tk[t_tok]])
        # brute force at one token
        b, tk = 1, _tok_index(is_slot, 1)
        ii = 7
        i = tk[ii]
        et = [emb[b, p] for p in tk]
        z = torch.zeros(D)
        win = [torch.cat([et[j - 1] if j > 0 else z, et[j], et[j + 1] if j + 1 < len(tk) else z])
               for j in range(len(tk))]
        q = _rms(m.q(x[b, i]).view(H, 8))
        scale = m.log_scale_bias.exp() / math.sqrt(8)
        heads = []
        for hh in range(H):
            ks = [_rms(m.k(win[j]).view(H, 8))[hh] for j in range(ii)]
            vs = [m.v(win[j]).view(H, 8)[hh] for j in range(ii)]
            s = torch.stack([q[hh] @ k * scale[hh] for k in ks] + [q[hh] @ _rms(m.null_k_bias)[hh] * scale[hh]])
            w = torch.softmax(s, 0)
            heads.append(sum(w[n] * vs[n] for n in range(ii)))
        ref = m.o(torch.cat(heads))
        assert torch.allclose(out[b, i], ref, atol=1e-5)


def test_pointer_mixture_is_normalised():
    """sum over every possible target v of the mixture's p(v) is exactly 1 (null mass returns)."""
    torch.manual_seed(5)
    g, is_slot, labels, h = _layout(5)
    m = PointerHead(D, H, dh=8)
    with torch.no_grad():
        m.gate_norm_head.bias.zero_()
        lpm = torch.log_softmax(torch.randn(B, L, 5, generator=g), -1)
        for b in range(B):
            for i in [p for p in _tok_index(is_slot, b)][3:]:
                tot = 0.0
                for v in range(5):
                    lab = labels.clone()
                    lab[b, i] = v
                    gl, p, null = m(h, lab, is_slot)
                    tot += float(mix_pointer(lpm[b, i, v], gl[b, i], p[b, i], null[b, i]).exp())
                assert math.isclose(tot, 1.0, abs_tol=1e-5), (b, i, tot)


def _tul_layout(seed=1, Bt=2, seq=64, K=4, S=12, V=200):
    import numpy as np
    import lxtul.data  # noqa: F401
    from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, pack_tul_batch
    lut = np.zeros(V, dtype=bool)
    lut[10:20] = True
    lut[1] = True
    rule = BoundaryRule(lut, min_span=4, span_cap=12, eos_id=1)
    spec = TulLayoutSpec(seq_len=seq, prefix_k=K, max_slots=S, slot_id=4)
    rng = np.random.default_rng(seed)
    buf = rng.integers(20, 40, size=Bt * (spec.l_total + 1) * 2).tolist()   # repeats
    for i in range(0, len(buf), 7):
        buf[i] = int(rng.integers(10, 20))
    return pack_tul_batch(buf, rule, spec, Bt)


def test_cell_key_zero_init_causal_and_reads_only_earlier_spans():
    torch.manual_seed(6)
    x, y, lay = _tul_layout()
    Bt, Lt = x.shape
    lab = torch.where(lay.slot_mask, torch.full_like(y, -100), y)
    h = torch.randn(Bt, Lt, D)
    plain, ck = PointerHead(D, H, dh=8), PointerHead(D, H, dh=8, cell_key=True)
    ck.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        a0 = plain(h, lab, lay.slot_mask)
        a1 = ck(h, lab, lay.slot_mask, lay)
        for u, v in zip(a0, a1):
            assert torch.allclose(u, v, atol=1e-6)                     # kc zero: no change
        ck.kc.weight.normal_(0, 0.3)
        base = ck(h, lab, lay.slot_mask, lay)
        assert not torch.allclose(base[1], a0[1], atol=1e-4)            # the term is live
        # causality: change the cells of span s (and everything after them); tokens of span <= s
        # must not move (their keys never read span s's cells: span(j) < span(i) <= s)
        b, s = 0, 3
        p0 = int(lay.slot_index[b, s])
        h2 = h.clone()
        h2[b, p0:] += 1.0
        moved = ck(h2, lab, lay.slot_mask, lay)
        tok_le_s = (~lay.slot_mask[b]) & (lay.bag_id[b] <= s) & (torch.arange(Lt) < p0)
        for u, v in zip(base, moved):
            assert torch.equal(u[b][tok_le_s], v[b][tok_le_s])
        # a query in span s+1 reads span s's cells through keys in span s
        q_next = (~lay.slot_mask[b]) & (lay.bag_id[b] == s + 1) & (lab[b] >= 0)
        h3 = h.clone()
        h3[b, p0:p0 + int(lay.prefix_k)] += 1.0                         # span s's cells only
        moved3 = ck(h3, lab, lay.slot_mask, lay)
        assert not torch.equal(base[1][b][q_next], moved3[1][b][q_next])


def test_mixed_at_matches_the_target_rule_for_every_token():
    """generation's full distribution equals forward+mix_pointer evaluated at each target v"""
    torch.manual_seed(8)
    g, is_slot, labels, h = _layout(5)
    m = PointerHead(D, H, dh=8)
    with torch.no_grad():
        m.gate_norm_head.bias.zero_()
        lpm = torch.log_softmax(torch.randn(B, L, 5, generator=g), -1)
        for b in range(B):
            toks = list(_tok_index(is_slot, b))
            for k in (3, len(toks) - 1):
                i = toks[k]
                lab = labels.clone()
                lab[b, i] = -100                                    # the query's own label unknown
                q_tok = torch.tensor([k] * B)
                full = m.mixed_at(h, lab, is_slot, lpm[:, i], q_tok)
                for v in range(5):
                    lv = lab.clone()
                    lv[b, i] = v
                    gl, p, null = m(h, lv, is_slot)
                    ref = mix_pointer(lpm[b, i, v], gl[b, i], p[b, i], null[b, i])
                    assert math.isclose(float(full[b, v]), float(ref), abs_tol=1e-5), (b, k, v)
                assert math.isclose(float(full[b].exp().sum()), 1.0, abs_tol=1e-5)
