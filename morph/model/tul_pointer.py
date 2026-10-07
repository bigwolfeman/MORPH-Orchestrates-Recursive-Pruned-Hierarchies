"""``tul.pointer_heads`` — a learned pointer/copy head on the strict slot loop (2026-10-06).

The strict geometry keeps every earlier span's COMPUTATION away from a token, and the Parcae
testbed measured the price: about 90 % of strict LXTUL's CE gap to plain is tokens whose bigram
occurred in an earlier span (copying). The pointer gives the OUTPUT that copy without letting
anything cross spans inside the model (pointer-generator, See et al. 2017; pointer sentinel,
Merity et al. 2016):

    query  = the token's final hidden state h_i (after the readout norm)
    keys   = earlier tokens' h_j, j < i in TOKEN order (slot positions removed); under the strict
             geometry h_j knows only its own span and the loop's cells
    value  = the token that FOLLOWED j, c_j = x_{j+1} (an input at a position <= i: causal)
    p_ptr_h(v) = sum_j alpha_hij [c_j = v]          (a learned null key lets a head abstain)
    p(v)   = (a0 + sum_h a_h null_h) p_model(v) + sum_h a_h p_ptr_h(v),  a = softmax(gate(h_i))
             A head's abstain mass null_h goes back to the model (the pointer-sentinel rule), so
             p sums to exactly 1; a head with no candidate has null_h = 1.

Cell keys (``tul.pointer_cell_key``): a key at j in an EARLIER span than the query also gets a
term from the loop's cells for j's own span (the final hidden state at that span's slot positions,
mean over the prefix_k cells). The query's span may already read those cells under the strict
geometry, so the head sees nothing new; same-span keys get no term (their cells lie in the
query's future). Zero-initialised: step 0 is the head without them. Parcae: -0.0074 paired.

Training reads p at the true target only (``target_logprob``); generation and the sweep read the
full mixed log-probs (``mixed_logprobs``). Both use the same attention, the same gate and the
same candidates, so the two agree (tests/test_tul_pointer.py). Nothing here writes a hidden
state. Every parameter is never-ternary (``_ternary_exclude``); the gate and the 1-D parameters
carry "norm" / "bias" in their names, so morph/training/optimizer.py puts them in no-decay.

Parcae source of the same head: parcae branch lxtul-testbed, lxtul/copy_heads.PointerHead
(filed docs/experiments/failures/2026-10-06-parcae-copy-heads.md there).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def token_order(slot_mask: Tensor) -> Tensor:
    """[B, L] -> a permutation putting token positions first, in row order (stable)."""
    return torch.argsort(slot_mask.to(torch.int8), dim=1, stable=True)


def candidates_from_ids(input_ids: Tensor, slot_mask: Tensor) -> Tensor:
    """c_j = the next TOKEN's id in token order, -1 where none ([B, L], token order)."""
    order = token_order(slot_mask)
    ids_t = input_ids.gather(1, order)
    n_tok = (~slot_mask).sum(1)
    ar = torch.arange(ids_t.shape[1], device=ids_t.device)
    return torch.where(ar[None] + 1 < n_tok[:, None], torch.roll(ids_t, -1, 1), -1)


def candidates_from_labels(labels: Tensor, slot_mask: Tensor) -> Tensor:
    """c_j = a token position's label (the next token), -1 off-token / unlabelled (token order)."""
    lab = torch.where(slot_mask, torch.full_like(labels, -100), labels)
    c = lab.gather(1, token_order(slot_mask))
    return torch.where(c >= 0, c, -1)


def mix_target(logp_model: Tensor, gate_logits: Tensor, p: Tensor, null: Tensor) -> Tensor:
    """log((a0 + sum_h a_h null_h) p_model(y) + sum_h a_h p_h(y)), a = softmax(gate_logits).
    p, null [..., H]; the result is the log of a NORMALISED distribution's mass at y."""
    a = torch.softmax(gate_logits.float(), -1)
    w0 = a[..., 0] + (a[..., 1:] * null.float()).sum(-1)
    s = (a[..., 1:] * p.float()).sum(-1)
    lc = torch.where(s > 0, torch.log(s.clamp_min(1e-30)), float("-inf"))
    return torch.logaddexp(torch.log(w0.clamp_min(1e-30)) + logp_model.float(), lc)


class TULPointer(nn.Module):
    def __init__(self, d: int, heads: int, dh: int = 64, cell_key: bool = False):
        super().__init__()
        self.h, self.dh, self.cell_key = int(heads), int(dh), bool(cell_key)
        self.q = nn.Linear(d, self.h * self.dh, bias=False)
        self.k = nn.Linear(d, self.h * self.dh, bias=False)
        for m in (self.q, self.k):
            nn.init.normal_(m.weight, std=0.02)
        if self.cell_key:
            self.kc = nn.Linear(d, self.h * self.dh, bias=False)
            nn.init.zeros_(self.kc.weight)
        self.null_k_bias = nn.Parameter(torch.randn(self.h, self.dh) * 0.02)
        self.log_scale_bias = nn.Parameter(torch.full((self.h,), math.log(math.sqrt(self.dh))))
        # gate over (model, head 1..H); init a ~ (0.96, 0.04 / H each)
        self.gate_norm_head = nn.Linear(d, 1 + self.h)
        with torch.no_grad():
            self.gate_norm_head.weight.zero_()
            self.gate_norm_head.bias.copy_(
                torch.tensor([0.0] + [-3.2 - math.log(self.h)] * self.h))
        self._ternary_exclude = True
        for m in self.modules():
            m._ternary_exclude = True

    def attend(self, h: Tensor, cand: Tensor, slot_mask: Tensor, layout=None):
        """h [B, L, C] packed order; cand [B, L] token order (-1 = not a candidate).
        Returns (alpha [B, H, N, N] token order over real keys, null [B, H, N] token order
        (the abstain mass; 1 where a query has no candidate), gate logits [B, L, 1+H] packed)."""
        B, L, C = h.shape
        order = token_order(slot_mask)
        ht = h.gather(1, order[..., None].expand(-1, -1, C))
        n_tok = (~slot_mask).sum(1)
        ar = torch.arange(L, device=h.device)
        tok = ar[None] < n_tok[:, None]
        allow = (ar[None, :, None] > ar[None, None, :]) & tok[:, :, None] & (cand >= 0)[:, None, :]
        q = _rms(self.q(ht).view(B, L, self.h, self.dh).transpose(1, 2)).float()
        k = _rms(self.k(ht).view(B, L, self.h, self.dh).transpose(1, 2)).float()
        scale = (self.log_scale_bias.exp() / math.sqrt(self.dh)).float()
        s = torch.einsum("bhid,bhjd->bhij", q, k) * scale[None, :, None, None]
        if self.cell_key:
            if layout is None:
                raise ValueError("TULPointer(cell_key=True) needs the slot layout")
            s = s + self._cell_scores(h, q, layout, order, scale)
        s = s.masked_fill(~allow[:, None], float("-inf"))
        s_null = torch.einsum("bhid,hd->bhi", q, _rms(self.null_k_bias).float()) * scale[None, :, None]
        pr = torch.softmax(torch.cat([s, s_null[..., None]], -1), -1)
        return pr[..., :-1], pr[..., -1], self.gate_norm_head(h)

    def _cell_scores(self, h: Tensor, q: Tensor, layout, order: Tensor, scale: Tensor) -> Tensor:
        """[B, H, N, N] token order: q_i . kc(cells of span(j)) where span(j) < span(i), else 0."""
        B, L, C = h.shape
        S, K = layout.slot_index.shape[1], int(layout.prefix_k)
        pos = (layout.slot_index[:, :, None] + torch.arange(K, device=h.device)).clamp(max=L - 1)
        cells = h.gather(1, pos.reshape(B, S * K, 1).expand(-1, -1, C)).view(B, S, K, C).mean(2)
        cells = cells * layout.slot_valid[..., None].to(cells.dtype)
        bag_t = layout.bag_id.gather(1, order)
        ck = self.kc(cells).view(B, S, self.h, self.dh)
        ckj = ck.gather(1, bag_t.clamp(0, S - 1)[:, :, None, None].expand(-1, -1, self.h, self.dh))
        e = torch.einsum("bhid,bjhd->bhij", q, ckj.float()) * scale[None, :, None, None]
        earlier = (bag_t[:, None, :] < bag_t[:, :, None]) & (bag_t[:, None, :] >= 0) \
            & (bag_t[:, None, :] < S)
        return e * earlier[:, None].to(e.dtype)

    def target_logprob(self, h: Tensor, logp_model: Tensor, labels: Tensor,
                       slot_mask: Tensor, layout=None) -> Tensor:
        """Training: log p(y) under the mixture at every packed position [B, L]; positions that
        are not labelled token positions return `logp_model` unchanged."""
        order = token_order(slot_mask)
        inv = torch.argsort(order, dim=1)
        cand = candidates_from_labels(labels, slot_mask)                       # token order
        alpha, null, gl = self.attend(h, cand, slot_mask, layout)
        same = (cand[:, :, None] == cand[:, None, :]) & (cand[:, :, None] >= 0)  # y_i = c_i
        p = (alpha * same[:, None].to(alpha.dtype)).sum(-1).transpose(1, 2)      # [B, N, H]
        g = inv[..., None].expand(-1, -1, self.h)
        lp = mix_target(logp_model, gl, p.gather(1, g), null.transpose(1, 2).gather(1, g))
        is_tok_lab = ~slot_mask & (labels >= 0)
        return torch.where(is_tok_lab, lp, logp_model.float())

    def mixed_logprobs(self, h: Tensor, logits: Tensor, input_ids: Tensor,
                       slot_mask: Tensor, layout=None) -> Tensor:
        """Generation / sweep: full mixed log-probs [B, L, V] (slot positions: the model's own
        log-softmax). Candidates come from the inputs, so no labels are needed."""
        B, L, V = logits.shape
        order = token_order(slot_mask)
        inv = torch.argsort(order, dim=1)
        cand = candidates_from_ids(input_ids, slot_mask)
        alpha, null, gl = self.attend(h, cand, slot_mask, layout)
        gl_t = gl.gather(1, order[..., None].expand(-1, -1, 1 + self.h)).float()
        a = torch.softmax(gl_t, -1)                                             # [B, N, 1+H]
        w0 = a[..., :1] + (a[..., 1:] * null.transpose(1, 2)).sum(-1, keepdim=True)
        beta = torch.einsum("bhij,bih->bij", alpha, a[..., 1:])                # [B, N, N]
        idx = cand.clamp_min(0)[:, None, :].expand(-1, L, -1)
        p_ptr = torch.zeros(B, L, V, device=logits.device, dtype=torch.float32)
        p_ptr.scatter_add_(2, idx, beta * (cand >= 0)[:, None, :].float())
        lm = torch.log_softmax(logits.float(), -1).gather(1, order[..., None].expand(-1, -1, V))
        mixed = torch.log(w0 * lm.exp() + p_ptr)
        mixed = mixed.gather(1, inv[..., None].expand(-1, -1, V))
        return torch.where(slot_mask[..., None], torch.log_softmax(logits.float(), -1), mixed)
