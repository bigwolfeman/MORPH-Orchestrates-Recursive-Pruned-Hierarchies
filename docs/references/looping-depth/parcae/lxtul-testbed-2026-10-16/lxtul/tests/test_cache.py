"""Copy cache (lxtul/cache.py) and per-token CE (lxtul/ce.py), CPU.

1. `copy_probs` equals a brute-force loop on random rows with repeats and slots.
2. It is causal: changing x[t:] / y[t:] leaves every p, n at i < t unchanged.
3. `mix_logp` equals the hand-written mixture; unavailable sources get zero weight.
4. `linear_ce_tokens` gradients equal autograd on the plain bf16 expression, for an arbitrary
   per-row upstream gradient.
"""
from __future__ import annotations

import math

import torch

from lxtul.cache import copy_probs, mix_logp, packed_copy_probs
from lxtul.ce import linear_ce_tokens


def _rows(seed=0, B=3, N=60, V=7):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, V, (B, N), generator=g)
    tgt = torch.roll(ids, -1, 1)
    tgt[:, -1] = -100
    is_tok = torch.ones(B, N, dtype=torch.bool)
    is_tok[:, -5:] = False                                    # padding tail
    tgt[~is_tok] = -100
    return ids, tgt, is_tok


def _brute(ids, tgt, is_tok):
    B, N = ids.shape
    out = {k: torch.zeros(B, N) for k in ("p_bi", "n_bi", "p_uni", "n_uni")}
    for b in range(B):
        for i in range(N):
            if not is_tok[b, i]:
                continue
            for name, key in (("bi", lambda j: (ids[b, j - 1].item(), ids[b, j].item())
                               if j > 0 and is_tok[b, j - 1] else None),
                              ("uni", lambda j: ids[b, j].item())):
                ki = key(i)
                if ki is None:
                    continue
                js = [j for j in range(i) if is_tok[b, j] and tgt[b, j] >= 0 and key(j) == ki]
                out[f"n_{name}"][b, i] = len(js)
                if js:
                    out[f"p_{name}"][b, i] = sum(tgt[b, j] == tgt[b, i] for j in js).item() / len(js)
    return out


def test_copy_probs_matches_brute_force():
    ids, tgt, tok = _rows()
    c, r = copy_probs(ids, tgt, tok), _brute(ids, tgt, tok)
    for k in r:
        assert torch.allclose(c[k], r[k]), k
    assert float(c["n_bi"].sum()) > 0 and float(c["p_bi"].sum()) > 0


def test_copy_probs_is_causal():
    ids, tgt, tok = _rows(1)
    c = copy_probs(ids, tgt, tok)
    t = 30
    ids2, tgt2 = ids.clone(), tgt.clone()
    ids2[:, t:] = (ids2[:, t:] + 3) % 7
    tgt2[:, t - 1:] = torch.where(tgt2[:, t - 1:] >= 0, torch.roll(ids2, -1, 1)[:, t - 1:], -100)
    c2 = copy_probs(ids2, tgt2, tok)
    # the target of i = t-1 is x_t, which changed: compare strictly before it
    for k in c:
        assert torch.equal(c[k][:, : t - 1], c2[k][:, : t - 1]), k


def test_packed_order_round_trip():
    ids, tgt, tok = _rows(2)
    is_slot = torch.zeros_like(tok)
    is_slot[:, [10, 25, 40]] = True                           # interleave slots
    packed_ids = ids.clone(); packed_lab = tgt.clone()
    c = packed_copy_probs(packed_ids, packed_lab, is_slot)
    assert torch.equal(c["n_uni"][is_slot], torch.zeros(int(is_slot.sum())))


def test_mix_logp_is_the_hand_mixture():
    c = {"p_bi": torch.tensor([0.5, 0.0, 0.3]), "n_bi": torch.tensor([2.0, 0.0, 3.0]),
         "p_uni": torch.tensor([0.25, 0.0, 0.0]), "n_uni": torch.tensor([4.0, 0.0, 5.0])}
    g = torch.tensor([[0.0, -1.0, -2.0]] * 3)
    lpm = torch.log(torch.tensor([0.1, 0.2, 0.3]))
    got = mix_logp(lpm, g, c)
    a = torch.softmax(g[0], -1)
    assert math.isclose(float(got[0]), math.log(a[0] * 0.1 + a[1] * 0.5 + a[2] * 0.25), rel_tol=1e-5)
    assert math.isclose(float(got[1]), math.log(0.2), rel_tol=1e-5)    # no source: model only
    a2 = torch.softmax(g[2], -1)                                       # both available, p_uni 0
    assert math.isclose(float(got[2]), math.log(a2[0] * 0.3 + a2[1] * 0.3), rel_tol=1e-5)


def test_linear_ce_tokens_gradients_match_autograd():
    g = torch.Generator().manual_seed(4)
    N, d, V = 37, 16, 50
    x = torch.randn(N, d, generator=g, requires_grad=True)
    W = torch.randn(V, d, generator=g, requires_grad=True)
    lab = torch.randint(0, V, (N,), generator=g)
    lab[[3, 9]] = -100
    lab[lab == 5] = 6
    up = torch.randn(N, generator=g)
    ce = linear_ce_tokens(x, W, lab, 0.7, 5)
    (ce * up).sum().backward()
    gx, gW = x.grad.clone(), W.grad.clone()
    x.grad = W.grad = None
    lg = (x.to(torch.bfloat16) @ W.to(torch.bfloat16).T).float() * 0.7
    lg = lg.masked_fill(torch.arange(V) == 5, float("-inf"))
    ref = torch.where(lab >= 0, torch.nn.functional.cross_entropy(lg, lab.clamp_min(0),
                                                                   reduction="none"), 0.0)
    assert torch.allclose(ce, ref, atol=1e-4)
    (ref * up).sum().backward()
    assert torch.allclose(gx, x.grad, atol=2e-2, rtol=2e-2)
    assert torch.allclose(gW, W.grad, atol=2e-2, rtol=2e-2)
