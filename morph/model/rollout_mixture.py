"""The exact mixture over R enumerated (or sampled) rollouts: ONE copy of the math.

Three readers use it:

* LXTUL-GK's coda bound (``tul.gram_objective="iw"``, :func:`morph.model.tul_gram.iw_span_bound`);
* LXTUL-E's coda (``tul.code_enum_k > 1``, ``MORPHTransformer._enum_mix_losses``) and its
  deploy read (``MORPHTransformer._enum_mixture_logprobs``, the label-free forward);
* LXTUL-E's parallel span head (:func:`morph.model.tul_spandec_parallel.mixture_span_nll`).

Every one of them reduces to :func:`log_mean_exp` over the rollout axis.

THE PER-SPAN READ, AS A PREDICTOR. For one span ``g`` with scored positions ``p_1 < ... <
p_n`` and per-rollout log-probs ``lp_r(p)`` of each position's label,

    log (1/R) sum_r prod_i p_r(t_{p_i})   ==   sum_i log sum_r w_r(<p_i) p_r(t_{p_i}),

``w_r(<p) = softmax_r C_r(<p)``, ``C_r(<p) = sum_{i : p_i < p} lp_r(p_i)`` (the chain rule
of a mixture). The right-hand side is a CAUSAL predictor: at position ``p`` it needs only
the labels of the span's earlier positions, which are input tokens at or before ``p``. So
the span mixture is the log loss of a real next-token distribution, ``sum_r w_r(<p)
p_r(. | p)``, and that distribution is what a label-free forward returns
(:func:`sequential_log_weights` + :func:`mixture_logprobs`). The posterior restarts at
every span: the read is per span, not per row (the GK grouping, reused).

WHAT "EVIDENCE" IS on a packed TUL row (``morph/model/tul_layout.py``). Position ``i`` is
evidence for later positions of its span iff it is a TOKEN position whose NEXT token
position is in the SAME span: its label (the next token) is then an input token at or
before every later position of the span. A span's last token (the ``plast`` position,
label = the NEXT span's first token) is never evidence, and neither is a slot position.
:func:`evidence_labels` builds this from ``input_ids`` and the layout alone, so the
labelled and the label-free forwards weight the rollouts by the same numbers.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

from .tul_layout import SlotLayout


def log_mean_exp(S: Tensor, dim: int = 0) -> Tensor:
    """``logsumexp(S, dim) - log R``, ``R = S.shape[dim]``. At ``R = 1`` it returns the
    element exactly (logsumexp of one element is the element; nothing is subtracted)."""
    R = int(S.shape[dim])
    lse = torch.logsumexp(S, dim=dim)
    return lse - math.log(R) if R > 1 else lse


def span_segment_start(bag_id: Tensor) -> Tensor:
    """``[B, L]`` int64: the index of the first position of each position's contiguous
    run of one ``bag_id`` (on a packed row a span's tokens and its slot cells are one run;
    the tail dump bin is one run)."""
    B, L = bag_id.shape
    pos = torch.arange(L, device=bag_id.device).expand(B, L)
    start = torch.ones_like(bag_id, dtype=torch.bool)
    start[:, 1:] = bag_id[:, 1:] != bag_id[:, :-1]
    return torch.where(start, pos, torch.zeros_like(pos)).cummax(dim=1).values


def evidence_labels(input_ids: Tensor, layout: SlotLayout) -> tuple[Tensor, Tensor]:
    """``(ev_label [B, L] int64, ev_mask [B, L] bool)``.

    ``ev_mask[b, i]``: position ``i`` is a token position whose next token position ``j``
    lies in the same span (same ``bag_id``); ``ev_label[b, i] = input_ids[b, j]`` there
    (0 elsewhere, never read). On a packed row that label equals ``labels[b, i]``: the
    packer writes the next TOKEN of the stream at every token position."""
    tok = ~layout.slot_mask                                           # [B, L]
    B, L = tok.shape
    pos = torch.arange(L, device=tok.device).expand(B, L)
    big = torch.full_like(pos, L)
    # next token position strictly after i: a reverse cummin over token positions
    cand = torch.where(tok, pos, big)                                 # token pos, else L
    nxt_incl = cand.flip(1).cummin(dim=1).values.flip(1)              # first token >= i
    nxt = torch.cat([nxt_incl[:, 1:], big[:, :1]], dim=1)             # first token > i
    has = nxt < L
    nxt_c = nxt.clamp(max=L - 1)
    same = layout.bag_id.gather(1, nxt_c) == layout.bag_id
    ev_mask = tok & has & same
    ev_label = torch.where(ev_mask, input_ids.gather(1, nxt_c), torch.zeros_like(input_ids))
    return ev_label, ev_mask


def sequential_log_weights(lp_ev: Tensor, ev_mask: Tensor, seg_start: Tensor) -> Tensor:
    """``[R, B, L]`` fp32: ``log w_r(<p)``, the log posterior over the ``R`` rollouts at
    every position after the evidence of the EARLIER positions of its span.

    ``lp_ev [R, B, L]`` the per-rollout log-prob of each position's evidence label (any
    value where ``ev_mask`` is False); ``seg_start [B, L]`` from :func:`span_segment_start`.
    The exclusive running sum is taken in fp64 so a span's 32 terms do not drift."""
    a = torch.where(ev_mask.unsqueeze(0), lp_ev.double(), torch.zeros_like(lp_ev, dtype=torch.float64))
    incl = a.cumsum(dim=-1)
    excl = incl - a                                                   # sum over i < p
    R = lp_ev.shape[0]
    base = excl.gather(-1, seg_start.unsqueeze(0).expand(R, -1, -1))  # sum before the run
    C = excl - base                                                   # this span only
    return torch.log_softmax(C, dim=0).float()


def mixture_label_nll(lp: Tensor, logw: Tensor) -> Tensor:
    """``[B, L]`` ``-log sum_r w_r(<p) p_r(label_p)`` from ``lp`` / ``logw`` ``[R, B, L]``."""
    return -torch.logsumexp(logw + lp.float(), dim=0)


def mixture_logprobs(logp_r: Tensor, logw_r: Tensor, acc: Tensor | None) -> Tensor:
    """One rollout's step of the label-free deploy read: ``acc`` (``None`` at the first
    rollout) ``<- logaddexp(acc, logw_r[..., None] + logp_r)``, ``logp_r [B, L, V]`` that
    rollout's log-softmax and ``logw_r [B, L]`` its log posterior weight. Accumulated one
    rollout at a time so ``[R, B, L, V]`` never exists."""
    term = logp_r.float() + logw_r.unsqueeze(-1)
    return term if acc is None else torch.logaddexp(acc, term)
