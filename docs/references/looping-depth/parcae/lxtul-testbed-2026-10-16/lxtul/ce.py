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
# Dtypes of the vocab GEMMs and of the gradient/logit accumulators. The tests pin both to
# float64 to compare two code paths against a rounding-free reference; training never changes them.
_CDT = torch.bfloat16
_ACC = torch.float32


def _chunk_impl(xb: Tensor, Wb: Tensor, lab: Tensor, scale: float, mask_id: int, inv_n: Tensor):
    logits = (xb @ Wb.T).to(_ACC) * scale
    logits[:, mask_id] = float("-inf")
    lse = torch.logsumexp(logits, -1)
    ok = lab >= 0
    safe = lab.clamp_min(0)
    picked = logits.gather(1, safe[:, None]).squeeze(1)
    loss = torch.where(ok, lse - picked, 0.0).sum()
    g = torch.exp(logits - lse[:, None])
    g = g.scatter_add(1, safe[:, None], -torch.ones_like(picked)[:, None])
    g = (g * (ok.to(_ACC) * inv_n * scale)[:, None]).to(xb.dtype)
    return loss, g @ Wb, g


_chunk = torch.compile(_chunk_impl, dynamic=False)


class _LinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, labels, scale, mask_id, need_w):
        rows = (labels != -100).nonzero().squeeze(1)
        n = rows.numel()
        inv_n = torch.tensor(1.0 / max(n, 1), device=x.device, dtype=_ACC)
        xb_all = x.to(_CDT)
        Wb = W.to(_CDT)
        pad = (-n) % CHUNK
        rows_p = torch.cat([rows, rows.new_zeros(pad)])
        lab_p = torch.cat([labels[rows], labels.new_full((pad,), -100)])
        dx = torch.zeros_like(x, dtype=_ACC)
        dW = torch.zeros_like(W, dtype=_ACC) if need_w else None
        loss = torch.zeros((), device=x.device, dtype=_ACC)
        for i in range(0, rows_p.numel(), CHUNK):
            r = rows_p[i:i + CHUNK]
            xb = xb_all[r]
            l, gx, g = _chunk(xb, Wb, lab_p[i:i + CHUNK], scale, mask_id, inv_n)
            loss = loss + l
            dx.index_add_(0, r, gx.to(_ACC))
            if need_w:
                dW.add_((g.T @ xb).to(_ACC))
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
        xb, Wb = x.to(_CDT), W.to(_CDT)
        ce = torch.zeros(x.shape[0], device=x.device, dtype=_ACC)
        for i in range(0, x.shape[0], CHUNK):
            lg = (xb[i:i + CHUNK] @ Wb.T).to(_ACC) * scale
            if mask_id >= 0:                      # mask_id < 0: no masked logit
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
        xb, Wb = x.to(_CDT), W.to(_CDT)
        dx = torch.zeros_like(x, dtype=_ACC)
        dW = torch.zeros_like(W, dtype=_ACC) if ctx.needs_input_grad[1] else None
        for i in range(0, x.shape[0], CHUNK):
            lg = (xb[i:i + CHUNK] @ Wb.T).to(_ACC) * ctx.scale
            if ctx.mask_id >= 0:
                lg[:, ctx.mask_id] = float("-inf")
            lab = labels[i:i + CHUNK]
            w = torch.where(lab >= 0, go[i:i + CHUNK].to(_ACC), 0.0) * ctx.scale
            g = torch.softmax(lg, -1)
            g = g.scatter_add(1, lab.clamp_min(0)[:, None], -torch.ones_like(g[:, :1]))
            g = (g * w[:, None]).to(_CDT)
            dx[i:i + CHUNK] = (g @ Wb).to(_ACC)
            if dW is not None:
                dW.add_((g.T @ xb[i:i + CHUNK]).to(_ACC))
        return dx.to(x.dtype), (dW.to(W.dtype) if dW is not None else None), None, None, None


def linear_ce_tokens(x: Tensor, W: Tensor, labels: Tensor, scale: float, mask_id: int) -> Tensor:
    return _LinearCETok.apply(x, W, labels, float(scale), int(mask_id))


# ---------------------------------------------------------------------------------------------
# Static row packing (keys `ce_rows`, `spandec_packed`; lxtul/model.py). `linear_ce` finds its
# labelled rows with `nonzero()` (a host sync) and pads them to a multiple of CHUNK (up to 2047
# idle rows, 30 % of the GEMMs at the lxtul shapes). Here the labelled rows are packed to the front
# of a buffer of FIXED size proven from the shapes, with no host sync, and the chunk size divides
# that buffer. A batch with more labelled rows than the bound gives a NaN loss (loud), never a
# silent drop.
# ---------------------------------------------------------------------------------------------
def static_pack(keep: Tensor, cap: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pack the `keep` entries of every row of a [R, N] bool, in order, to the front of a [R, cap]
    buffer, with no host sync. Returns `(src, used, back, over)`:
      src  [R, cap] long: source index of each buffer slot (0 for an unused slot)
      used [R, cap] bool: the slot holds a kept entry
      back [R, N] long: each entry's slot, `cap` for an entry that is not kept (or did not fit)
      over 0-dim bool: some row kept more than `cap` entries (the caller makes that a NaN loss)
    Every scatter index is unique, so the pack is deterministic."""
    R, N = keep.shape
    pos = keep.cumsum(1) - 1
    land = keep & (pos < cap)
    over = (keep.sum(1) > cap).any()
    ar = torch.arange(N, device=keep.device).expand(R, N)
    src = torch.full((R, cap + N), N, device=keep.device, dtype=torch.long)
    src.scatter_(1, torch.where(land, pos, ar + cap), ar)
    src = src[:, :cap]
    used = src < N
    return torch.where(used, src, torch.zeros_like(src)), used, torch.where(land, pos, torch.full_like(pos, cap)), over


def _expand(idx: Tensor, trailing: tuple) -> Tensor:
    """[R, K] index -> [R, K, *trailing] (a view), the form `gather(1, .)` wants."""
    return idx.reshape(*idx.shape, *([1] * len(trailing))).expand(*idx.shape, *trailing)


class _Take(torch.autograd.Function):
    """y[r, k] = x[r, src[r, k]]  ([R, N, ...] -> [R, cap, ...]). Backward is a gather (each kept
    entry has one slot), so there are no atomics; an unused slot's gradient is never read."""

    @staticmethod
    def forward(ctx, x: Tensor, src: Tensor, back: Tensor):
        ctx.save_for_backward(back)
        return x.gather(1, _expand(src, x.shape[2:]))

    @staticmethod
    def backward(ctx, g: Tensor):
        (back,) = ctx.saved_tensors
        gp = torch.cat([g, g.new_zeros(g.shape[0], 1, *g.shape[2:])], 1)
        return gp.gather(1, _expand(back, gp.shape[2:])), None, None


class _Put(torch.autograd.Function):
    """z[r, n] = y[r, back[r, n]], 0 where the entry was not kept ([R, cap, ...] -> [R, N, ...])."""

    @staticmethod
    def forward(ctx, y: Tensor, src: Tensor, used: Tensor, back: Tensor):
        ctx.save_for_backward(src, used)
        yp = torch.cat([y, y.new_zeros(y.shape[0], 1, *y.shape[2:])], 1)
        return yp.gather(1, _expand(back, yp.shape[2:]))

    @staticmethod
    def backward(ctx, g: Tensor):
        src, used = ctx.saved_tensors
        gy = g.gather(1, _expand(src, g.shape[2:]))
        return gy * _expand(used, g.shape[2:]).to(gy.dtype), None, None, None


def take_rows(x: Tensor, src: Tensor, back: Tensor) -> Tensor:
    return _Take.apply(x, src, back)


def put_rows(y: Tensor, src: Tensor, used: Tensor, back: Tensor) -> Tensor:
    return _Put.apply(y, src, used, back)


class _LinearCERows(torch.autograd.Function):
    """`_LinearCE` on rows that are already the buffer: no nonzero(), no index_add, equal chunks.
    The row count is `labels.numel()` (static); the mean divides by the labelled count, on the device."""

    @staticmethod
    def forward(ctx, x, W, labels, scale, mask_id, need_w):
        N = x.shape[0]
        nch = -(-N // CHUNK)
        per = -(-N // nch)
        ch = -(-per // 8) * 8                            # equal chunks, one compiled shape
        pad = nch * ch - N
        inv_n = 1.0 / (labels != -100).sum().clamp_min(1).to(_ACC)
        xb_all = torch.nn.functional.pad(x.to(_CDT), (0, 0, 0, pad))
        lab_all = torch.nn.functional.pad(labels, (0, pad), value=-100)
        Wb = W.to(_CDT)
        dx = torch.empty(nch * ch, x.shape[1], device=x.device, dtype=_ACC)
        dW = torch.zeros_like(W, dtype=_ACC) if need_w else None
        loss = torch.zeros((), device=x.device, dtype=_ACC)
        for i in range(0, nch * ch, ch):
            xb = xb_all[i:i + ch]
            l, gx, g = _chunk(xb, Wb, lab_all[i:i + ch], scale, mask_id, inv_n)
            loss = loss + l
            dx[i:i + ch] = gx.to(_ACC)
            if need_w:
                dW.add_((g.T @ xb).to(_ACC))
        ctx.save_for_backward(dx[:N].to(x.dtype), dW.to(W.dtype) if need_w else None)
        return loss * inv_n

    @staticmethod
    def backward(ctx, go):
        dx, dW = ctx.saved_tensors
        return dx * go, (dW * go if dW is not None else None), None, None, None, None


def linear_ce_rows(x: Tensor, W: Tensor, labels: Tensor, scale: float, mask_id: int) -> Tensor:
    """Same value as `linear_ce` on the same labelled rows; `x` [N, d] is the packed buffer (unused
    rows carry label -100). Static shapes, no host sync."""
    return _LinearCERows.apply(x, W, labels, float(scale), int(mask_id), W.requires_grad)
