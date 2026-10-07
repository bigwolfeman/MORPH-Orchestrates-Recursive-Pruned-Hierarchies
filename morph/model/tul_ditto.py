"""DITTO: the sentence-level repetition penalty (Xu et al. 2022, arXiv 2206.02369 §3, eq. 1).

``tul.ditto_rows`` rebuilds some training rows as pseudo-repetition rows
(``tul_layout.pack_ditto_row``: real text up to a span, then that span repeated). For the
l-th token of copy n >= 1, with p the model's FINAL probability of that token (the pointer
mixture when the head is on):

    L = -log(1 - |p_n - lambda * sg(p_{n-1})|)

``sg`` is a stop-gradient on the previous copy's probability. lambda = 1 asks each copy to
be no more likely than the one before (no self-reinforcement); lambda < 1 asks the
probability of a repeated span to decay by lambda per copy. The paper uses lambda 0.5 on
Wikitext-103 and mixes DITTO and MLE updates equally; here the DITTO mean over its
positions is added to the CE mean over the other positions at weight 1.

Why it is here and not See et al. coverage: coverage counts attention per POSITION, and a
language model's loop copies copy n from copy n-1, whose positions no query attended yet, so
coverage cannot see it (lab/experiments/failures/2026-10-07-lxtul-pointer-coverage.md).
DITTO scores the repeated tokens themselves.
"""
from __future__ import annotations

import torch
from torch import Tensor


def ditto_loss(lp: Tensor, ditto_prev: Tensor, lam: float) -> tuple[Tensor, Tensor, Tensor]:
    """The DITTO term over the positions ``ditto_prev`` marks.

    Args:
        lp:         ``[B, L]`` log-probability of each position's label (any dtype).
        ditto_prev: ``[B, L]`` int64; for a DITTO position, the position in the SAME row
                    whose label is the same token one copy earlier; -1 elsewhere.
        lam:        the decay factor, in [0, 1].

    Returns:
        ``(term, ratio, pos)``: the mean loss over DITTO positions (0 when there are
        none), ``sum p_n / sum p_{n-1}`` over them (detached; the loss drives it toward
        ``lam``), and the ``[B, L]`` bool mask of DITTO positions.
    """
    if lp.shape != ditto_prev.shape:
        raise ValueError(f"lp {tuple(lp.shape)} != ditto_prev {tuple(ditto_prev.shape)}")
    pos = ditto_prev >= 0
    p = lp.float().exp()
    p_prev = torch.gather(p.detach(), 1, ditto_prev.clamp_min(0))
    d = -torch.log((1.0 - (p - lam * p_prev).abs()).clamp_min(1e-6))
    n = pos.sum().clamp_min(1)
    term = (d * pos).sum() / n
    ratio = ((p.detach() * pos).sum() / (p_prev * pos).sum().clamp_min(1e-9))
    return term, ratio, pos
