"""Exact-copy cache: token identity across spans, never a hidden state.

For token i (token order, slots removed) and key k_i, every EARLIER token j < i with the same
key proposes its own target y_j ("what came next last time"):

    bigram key  (x_{i-1}, x_i)      unigram key  x_i

`copy_probs` returns, per token, the cache's probability of the TRUE target
p_c(y_i) = #{j < i : k_j = k_i, y_j = y_i} / #{j < i : k_j = k_i} and the match count. y_j is
x_{j+1}, an input at a position <= i, so the cache is causal. The model mixes
p = a0 p_model + a1 p_bi + a2 p_uni with a gate a = softmax(g(h_i, counts)); the cache enters
only the output distribution, so no earlier span's content reaches any hidden state through it.
"""
from __future__ import annotations

import torch
from torch import Tensor


def token_order(is_slot: Tensor) -> Tensor:
    """[B, L] -> permutation putting token positions first, in row order (stable)."""
    return torch.argsort(is_slot.to(torch.int8), dim=1, stable=True)


@torch.no_grad()
def copy_probs(ids: Tensor, tgt: Tensor, is_tok: Tensor) -> dict[str, Tensor]:
    """ids, tgt [B, N] in token order; is_tok [B, N] (False on slots / padding).

    Returns p_bi, n_bi, p_uni, n_uni, each [B, N] float32, in the same order."""
    B, N = ids.shape
    V = int(ids.max()) + 1
    tv = is_tok & (tgt >= 0)                                   # j proposes y_j only if scored
    prev_tok = torch.zeros_like(is_tok)
    prev_tok[:, 1:] = is_tok[:, :-1]
    k_bi = torch.where(is_tok & prev_tok, torch.roll(ids, 1, 1) * V + ids, -1)
    k_uni = torch.where(is_tok, ids, -1)
    earlier = torch.ones(N, N, dtype=torch.bool, device=ids.device).tril(-1)   # j < i
    same_y = tgt[:, :, None] == tgt[:, None, :]
    out = {}
    for name, k in (("bi", k_bi), ("uni", k_uni)):
        m = (k[:, :, None] == k[:, None, :]) & earlier & (k[:, :, None] >= 0) & tv[:, None, :]
        n = m.sum(-1).float()
        hit = (m & same_y).sum(-1).float()
        out[f"p_{name}"] = torch.where(n > 0, hit / n.clamp_min(1), 0.0)
        out[f"n_{name}"] = n
    return out


def packed_copy_probs(input_ids: Tensor, labels: Tensor, is_slot: Tensor) -> dict[str, Tensor]:
    """`copy_probs` for a packed TUL row, returned in PACKED order (slots read 0)."""
    order = token_order(is_slot)
    ids_t = input_ids.gather(1, order)
    tgt_t = torch.where(is_slot, torch.full_like(labels, -100), labels).gather(1, order)
    tok_t = (~is_slot).gather(1, order)
    c = copy_probs(ids_t, tgt_t, tok_t)
    inv = torch.argsort(order, dim=1)
    return {k: torch.where(is_slot, 0.0, v.gather(1, inv)) for k, v in c.items()}


def mix_sources(logp_model: Tensor, gate_logits: Tensor, probs: Tensor, avail: Tensor) -> Tensor:
    """log(a0 p_model(y) + sum_s a_s p_s(y)); a = softmax of `gate_logits` [..., 1+S] over the
    model and the AVAILABLE sources (`avail` [..., S]); `probs` [..., S] at the true target."""
    full = torch.cat([torch.ones_like(avail[..., :1]), avail], -1)
    la = torch.log_softmax(gate_logits.float().masked_fill(~full, float("-inf")), -1)
    s = (la[..., 1:].exp() * probs.float()).sum(-1)
    lc = torch.where(s > 0, torch.log(s.clamp_min(1e-30)), float("-inf"))
    return torch.logaddexp(la[..., 0] + logp_model, lc)


def mix_logp(logp_model: Tensor, gate_logits: Tensor, c: dict[str, Tensor]) -> Tensor:
    """The cache's two sources (bigram, unigram) through `mix_sources`."""
    return mix_sources(logp_model, gate_logits, torch.stack([c["p_bi"], c["p_uni"]], -1),
                       torch.stack([c["n_bi"] > 0, c["n_uni"] > 0], -1))


def copy_features(c: dict[str, Tensor]) -> Tensor:
    return torch.stack([torch.log1p(c["n_bi"]), torch.log1p(c["n_uni"]),
                        (c["n_bi"] > 0).float(), (c["n_uni"] > 0).float()], -1)
