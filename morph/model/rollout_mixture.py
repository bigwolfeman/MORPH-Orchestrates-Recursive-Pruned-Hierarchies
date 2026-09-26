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

THE CARRIED READ (LX-Carry, ``/home/wolfe/morph-scratch/tulv2/opus.md`` D2; offline only,
no forward uses it). The per-span restart is the ``eps = 1`` case of a fixed-share
switching model over the rollouts. Span ``s`` starts at a prior ``pi_s`` instead of
uniform; after it,

    alpha_s(k) ∝ pi_s(k) exp S_k(s),      pi_{s+1} = (1 - eps) alpha_s + eps / R,

``S_k(s)`` the sum of rollout ``k``'s log-probs over every SCORED position of span ``s``.
The per-position read inside the span is the same Bayes update started at ``pi_s``, so a
span's positions sum to ``-log sum_k pi_s(k) exp S_k(s)`` and a row's to the switching
model's exact log-likelihood. :func:`fixed_share_log_prior` returns ``log(R pi_s)`` per
position (0 at uniform), and :func:`sequential_log_weights` adds it before the softmax. At
``eps = 1`` every prior is ``log(R * (0 * alpha + 1/R)) = log 1 = 0`` exactly, so the read
is bit-identical to the restart (tests/test_lx_probes.py).
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


def sequential_log_weights(lp_ev: Tensor, ev_mask: Tensor, seg_start: Tensor,
                           log_prior: Tensor | None = None) -> Tensor:
    """``[R, B, L]`` fp32: ``log w_r(<p)``, the log posterior over the ``R`` rollouts at
    every position after the evidence of the EARLIER positions of its span.

    ``lp_ev [R, B, L]`` the per-rollout log-prob of each position's evidence label (any
    value where ``ev_mask`` is False); ``seg_start [B, L]`` from :func:`span_segment_start`.
    The exclusive running sum is taken in fp64 so a span's 32 terms do not drift.
    ``log_prior [R, B, L]`` (fp64, optional): the span's starting prior as ``log(R pi)``
    (:func:`fixed_share_log_prior`); ``None`` is the uniform restart, the forward's read."""
    a = torch.where(ev_mask.unsqueeze(0), lp_ev.double(), torch.zeros_like(lp_ev, dtype=torch.float64))
    incl = a.cumsum(dim=-1)
    excl = incl - a                                                   # sum over i < p
    R = lp_ev.shape[0]
    base = excl.gather(-1, seg_start.unsqueeze(0).expand(R, -1, -1))  # sum before the run
    C = excl - base                                                   # this span only
    if log_prior is not None:
        C = C + log_prior.double()
    return torch.log_softmax(C, dim=0).float()


def segment_index(seg_start: Tensor) -> Tensor:
    """``[B, L]`` int64: the 0-based index of each position's run within its row (the runs
    of :func:`span_segment_start`, in row order)."""
    B, L = seg_start.shape
    pos = torch.arange(L, device=seg_start.device).expand(B, L)
    return (seg_start == pos).long().cumsum(dim=1) - 1


def fixed_share_log_prior(lp: Tensor, scored: Tensor, seg_start: Tensor, eps: float,
                          reset_after: Tensor | None = None) -> Tensor:
    """``[R, B, L]`` fp64: ``log(R pi_s(r))`` at every position of run ``s``, the carried
    prior of the fixed-share read (module doc, THE CARRIED READ).

    ``lp [R, B, L]`` each rollout's log-prob of the position's label (any value where
    ``scored`` is False); ``scored [B, L]`` the positions whose label enters the span's
    likelihood ``S_r(s)`` (the scorer's weighted positions: every evidence position AND the
    span's last token, whose label is the next span's first token). A run with no scored
    position (none on a packed row, kept for safety) passes its prior through unchanged.
    ``reset_after [B, L]`` (optional): a run holding a True position ENDS a document, so the
    next run starts at uniform again (the caller marks EOS input tokens). Each row starts at
    uniform. ``eps`` in ``[0, 1]``; ``eps = 1`` returns zeros exactly."""
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"fixed-share eps must be in [0, 1], got {eps}")
    R, B, L = lp.shape
    seg = segment_index(seg_start)                                    # [B, L]
    G = int(seg.max()) + 1
    idx = seg.unsqueeze(0).expand(R, -1, -1)
    val = torch.where(scored.unsqueeze(0), lp.double(),
                      torch.zeros_like(lp, dtype=torch.float64))
    S = torch.zeros(R, B, G, dtype=torch.float64, device=lp.device).scatter_add_(-1, idx, val)
    has = (torch.zeros(B, G, dtype=torch.float64, device=lp.device)
           .scatter_add_(-1, seg, scored.double()) > 0)                # run has a score
    if reset_after is None:
        rst = torch.zeros(B, G, dtype=torch.bool, device=lp.device)
    else:
        rst = (torch.zeros(B, G, dtype=torch.float64, device=lp.device)
               .scatter_add_(-1, seg, reset_after.double()) > 0)
    cur = torch.zeros(R, B, dtype=torch.float64, device=lp.device)   # log(R pi), uniform
    out = torch.empty(R, B, G, dtype=torch.float64, device=lp.device)
    for g in range(G):
        out[:, :, g] = cur
        log_alpha = torch.log_softmax(cur + S[:, :, g], dim=0)        # posterior after s
        pi = (1.0 - eps) * log_alpha.exp() + eps / R
        nxt = torch.log(pi * R)                                       # eps 1: log 1 = 0
        nxt = torch.where(has[:, g].unsqueeze(0), nxt, cur)           # empty run: pass
        cur = torch.where(rst[:, g].unsqueeze(0), torch.zeros_like(nxt), nxt)
    return out.gather(-1, idx)


def mixture_label_nll(lp: Tensor, logw: Tensor) -> Tensor:
    """``[B, L]`` ``-log sum_r w_r(<p) p_r(label_p)`` from ``lp`` / ``logw`` ``[R, B, L]``."""
    return -torch.logsumexp(logw + lp.float(), dim=0)


def carried_position_nll(lp: Tensor, input_ids: Tensor, layout: SlotLayout, scored: Tensor,
                         eps: float, reset_after: Tensor | None = None) -> Tensor:
    """``[B, L]`` the per-position NLL of the label under the CARRIED read (module doc):
    the per-span Bayes read of ``MORPHTransformer._enum_position_nll`` started at the
    fixed-share prior :func:`fixed_share_log_prior` instead of uniform. Same arguments as
    that method plus ``scored`` / ``eps`` / ``reset_after``; at ``eps = 1`` the two are
    bit-identical."""
    _ev_label, ev_mask = evidence_labels(input_ids, layout)
    seg = span_segment_start(layout.bag_id)
    prior = fixed_share_log_prior(lp, scored, seg, eps, reset_after)
    return mixture_label_nll(lp, sequential_log_weights(lp, ev_mask, seg, prior))


def mixture_logprobs(logp_r: Tensor, logw_r: Tensor, acc: Tensor | None) -> Tensor:
    """One rollout's step of the label-free deploy read: ``acc`` (``None`` at the first
    rollout) ``<- logaddexp(acc, logw_r[..., None] + logp_r)``, ``logp_r [B, L, V]`` that
    rollout's log-softmax and ``logw_r [B, L]`` its log posterior weight. Accumulated one
    rollout at a time so ``[R, B, L, V]`` never exists."""
    term = logp_r.float() + logw_r.unsqueeze(-1)
    return term if acc is None else torch.logaddexp(acc, term)
