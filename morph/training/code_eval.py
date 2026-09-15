"""TUL-Code eval instruments that need several sampled forwards.

`code_marginal_ce`: the K-sample marginal. A TUL-Code model is a latent-variable LM — the
coda writes span s+1 given a code the thinker SAMPLED. One-sample CE on the true span
(`val/loss`) scores whichever draw came out; the likelihood the model is owed is the
marginal over draws, and `log (1/K) Σ_k p(span | code_k)` with the thinker as the proposal
is a lower bound on it that tightens with K (importance weighting with the prior). Reported
per token, over the same positions as `ce_tokens`. Spec: docs/tul-code-spec.md §8.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


@torch.no_grad()
def code_marginal_ce(model, x: Tensor, y: Tensor, layout, k_samples: int,
                     steps: int) -> dict[str, float]:
    """Returns ``{"ce_marginal", "ce_single_mean"}`` in nats per scored token.

    ``ce_marginal`` = −Σ_spans [logsumexp_k LL_k(span) − log K] / n_tokens, where
    ``LL_k(span)`` is the span's token log-likelihood under the coda with sampled code k
    (seed k, ``steps`` Euler passes, the encoder's codes on every other slot — the
    `sampled` eval mode). ``ce_single_mean`` = the K-average of the one-sample CE. Tokens
    before the first slot have no sampled code and contribute identically to every draw.
    """
    if k_samples < 1:
        raise ValueError(f"k_samples must be >= 1, got {k_samples}")
    B, L = y.shape
    S = int(layout.slot_valid.shape[1])
    G = S + 1  # bag ids run 0..S: bag S is the row's tail, the tokens after the last slot
    keep = (y >= 0) & (~layout.slot_mask)
    lab = y.clamp_min(0)
    gid = torch.arange(B, device=y.device).view(B, 1) * G + layout.bag_id.clamp(0, S)
    gid = torch.where(keep, gid, torch.zeros_like(gid)).reshape(-1)
    lls = []
    for k in range(int(k_samples)):
        res = model.tul_forward_ablated(x, None, layout, plan_mode="normal",
                                        code_mode="sampled", code_steps=int(steps),
                                        code_seed=k)
        logits = res["logits"].float()
        ce = F.cross_entropy(logits.reshape(B * L, -1), lab.reshape(-1),
                             reduction="none").reshape(B, L) * keep
        span_ce = torch.zeros(B * G, device=y.device, dtype=torch.float32)
        span_ce.index_add_(0, gid, ce.reshape(-1))
        lls.append(-span_ce)
    ll = torch.stack(lls)  # [K, B*G] span log-likelihoods, one row per draw
    bound = torch.logsumexp(ll, dim=0) - math.log(len(lls))
    n_tok = float(keep.sum().clamp_min(1))
    return {"ce_marginal": float(-bound.sum() / n_tok),
            "ce_single_mean": float(-ll.mean(0).sum() / n_tok)}
