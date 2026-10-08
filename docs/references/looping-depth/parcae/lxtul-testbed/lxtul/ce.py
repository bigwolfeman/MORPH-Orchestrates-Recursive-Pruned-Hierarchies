"""Chunked linear cross-entropy with one masked logit, gradients computed in the forward.

`linear_ce(x, W, labels, scale, mask_id)` = mean over rows with label != -100 of
`-log softmax(scale * x @ W.T)[label]`, where the `mask_id` logit is -inf (MORPH masks the
slot id, SPEC 5.1). It never holds the full [N, V] logits: each chunk's softmax gradient is
folded into dx and dW at once, so backward is a scale. Pass `W.detach()` to skip dW
(the span decoder reads a detached table, SPEC 5.3).
"""
from __future__ import annotations

import torch
from torch import Tensor

CHUNK = 2048


@torch.compile(dynamic=False)
def _chunk(xb: Tensor, Wb: Tensor, lab: Tensor, scale: float, mask_id: int, inv_n: Tensor):
    logits = (xb @ Wb.T).float() * scale
    logits[:, mask_id] = float("-inf")
    lse = torch.logsumexp(logits, -1)
    ok = lab >= 0
    safe = lab.clamp_min(0)
    picked = logits.gather(1, safe[:, None]).squeeze(1)
    loss = torch.where(ok, lse - picked, 0.0).sum()
    g = torch.exp(logits - lse[:, None])
    g = g.scatter_add(1, safe[:, None], -torch.ones_like(picked)[:, None])
    g = (g * (ok.float() * inv_n * scale)[:, None]).to(xb.dtype)
    return loss, g @ Wb, g


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, labels, scale, mask_id, need_w):
        rows = (labels != -100).nonzero().squeeze(1)
        n = rows.numel()
        inv_n = torch.tensor(1.0 / max(n, 1), device=x.device)
        xb_all = x.to(torch.bfloat16)
        Wb = W.to(torch.bfloat16)
        pad = (-n) % CHUNK
        rows_p = torch.cat([rows, rows.new_zeros(pad)])
        lab_p = torch.cat([labels[rows], labels.new_full((pad,), -100)])
        dx = torch.zeros_like(x, dtype=torch.float32)
        dW = torch.zeros_like(W, dtype=torch.float32) if need_w else None
        loss = torch.zeros((), device=x.device, dtype=torch.float32)
        for i in range(0, rows_p.numel(), CHUNK):
            r = rows_p[i:i + CHUNK]
            xb = xb_all[r]
            l, gx, g = _chunk(xb, Wb, lab_p[i:i + CHUNK], scale, mask_id, inv_n)
            loss = loss + l
            dx.index_add_(0, r, gx.float())
            if need_w:
                dW.add_((g.T @ xb).float())
        ctx.save_for_backward(dx.to(x.dtype), dW.to(W.dtype) if need_w else None)
        return loss * inv_n

    @staticmethod
    def backward(ctx, go):
        dx, dW = ctx.saved_tensors
        return dx * go, (dW * go if dW is not None else None), None, None, None, None


def linear_ce(x: Tensor, W: Tensor, labels: Tensor, scale: float, mask_id: int) -> Tensor:
    return _LinearCE.apply(x, W, labels, float(scale), int(mask_id), W.requires_grad)


class _LinearCETok(torch.autograd.Function):
    """Per-row CE [N] (0 where label -100). Backward recomputes each chunk's logits, so the
    upstream gradient can be any per-row weight (the copy-cache mixture needs it)."""

    @staticmethod
    def forward(ctx, x, W, labels, scale, mask_id):
        xb, Wb = x.to(torch.bfloat16), W.to(torch.bfloat16)
        ce = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        for i in range(0, x.shape[0], CHUNK):
            lg = (xb[i:i + CHUNK] @ Wb.T).float() * scale
            lg[:, mask_id] = float("-inf")
            lab = labels[i:i + CHUNK]
            ok = lab >= 0
            pk = lg.gather(1, lab.clamp_min(0)[:, None]).squeeze(1)
            ce[i:i + CHUNK] = torch.where(ok, torch.logsumexp(lg, -1) - pk, 0.0)
        ctx.save_for_backward(x, W, labels)
        ctx.scale, ctx.mask_id = scale, mask_id
        return ce

    @staticmethod
    def backward(ctx, go):
        x, W, labels = ctx.saved_tensors
        xb, Wb = x.to(torch.bfloat16), W.to(torch.bfloat16)
        dx = torch.zeros_like(x, dtype=torch.float32)
        dW = torch.zeros_like(W, dtype=torch.float32) if ctx.needs_input_grad[1] else None
        for i in range(0, x.shape[0], CHUNK):
            lg = (xb[i:i + CHUNK] @ Wb.T).float() * ctx.scale
            lg[:, ctx.mask_id] = float("-inf")
            lab = labels[i:i + CHUNK]
            w = torch.where(lab >= 0, go[i:i + CHUNK].float(), 0.0) * ctx.scale
            g = torch.softmax(lg, -1)
            g = g.scatter_add(1, lab.clamp_min(0)[:, None], -torch.ones_like(g[:, :1]))
            g = (g * w[:, None]).to(torch.bfloat16)
            dx[i:i + CHUNK] = (g @ Wb).float()
            if dW is not None:
                dW.add_((g.T @ xb[i:i + CHUNK]).float())
        return dx.to(x.dtype), (dW.to(W.dtype) if dW is not None else None), None, None, None


def linear_ce_tokens(x: Tensor, W: Tensor, labels: Tensor, scale: float, mask_id: int) -> Tensor:
    return _LinearCETok.apply(x, W, labels, float(scale), int(mask_id))
