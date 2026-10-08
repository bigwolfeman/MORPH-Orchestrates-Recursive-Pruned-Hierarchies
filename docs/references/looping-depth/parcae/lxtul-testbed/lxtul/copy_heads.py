"""Two learned cross-span copy channels for the strict geometry (2026-10-06).

The strict mask keeps every earlier span's COMPUTATION away from a token; the gap probe showed
the price is copying (about 90 % of the gap is tokens whose bigram occurred in an earlier span).
Both modules give tokens the IDENTITY of earlier tokens and nothing computed from other spans.
Both run in token order (slots removed, `cache.token_order`), so "previous token" skips slots.

PointerHead (pointer-generator, See et al. 2017; pointer sentinel, Merity et al. 2016)
    Output-only. Query = the token's coda output h_i; keys = earlier tokens' coda outputs h_j
    (j < i), which under the strict mask know only their own span and the loop's cells. Each
    head's attention is read as a distribution over the TARGETS y_j of the attended positions:
    p_ptr(v) = sum_j alpha_ij [y_j = v]. A learned null key lets a head abstain; its mass goes
    back to the model (pointer-sentinel rule), so p = (a0 + sum_h a_h null_h) p_model +
    sum_h a_h p_ptr_h sums to exactly 1. (Until 2026-10-06 21:50 the null mass was dropped: a
    sub-normalised mixture, pessimistic CE; the copy-heads and plain-pointer filings say so.)
    `cell_key`: a key at j in an EARLIER span than the query also gets a term from the loop's
    cells for j's own span (the coda output at that span's slot positions, mean over the K
    cells), which the strict geometry already lets the query's span read. Zero-initialised.
    Nothing enters a hidden state.

IdentityReach
    In the hidden state. Before the coda, each token attends over every earlier token's RAW input
    window [e(x_{j-1}), e(x_j), e(x_{j+1})] (token embeddings, no computation), j < i so x_{j+1}
    is at most x_i. Query from the token's coda input; output projection zero-initialised, so the
    model starts as strict LXTUL. Slot cells are never written: the loop's input is unchanged.

Dense attention over the row (N = 1280 positions); a null key/value per head makes every row
well defined (the first token has no earlier token).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lxtul.cache import token_order


def _rms(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def _attn(q: Tensor, k: Tensor, v: Tensor | None, k_null: Tensor, allow: Tensor, scale: Tensor):
    """q [B,H,N,dh], k [B,H,N,dh], v [B,H,N,dv] or None, k_null [H,dh], allow [B,N,N] (query i,
    key j). Returns (alpha [B,H,N,N] over real keys, alpha_null [B,H,N], out or None)."""
    qn, kn = _rms(q), _rms(k)
    s = torch.einsum("bhid,bhjd->bhij", qn.float(), kn.float()) * scale[None, :, None, None]
    s = s.masked_fill(~allow[:, None], float("-inf"))
    s_null = torch.einsum("bhid,hd->bhi", qn.float(), _rms(k_null).float()) * scale[None, :, None]
    p = torch.softmax(torch.cat([s, s_null[..., None]], -1), -1)
    alpha, a_null = p[..., :-1], p[..., -1]
    out = None if v is None else torch.einsum("bhij,bhjd->bhid", alpha.to(v.dtype), v)
    return alpha, a_null, out


def _earlier(n_tok: Tensor, N: int, key_ok: Tensor | None = None) -> Tensor:
    """[B,N,N] allow-matrix in token order: query i and key j are tokens and j < i."""
    ar = torch.arange(N, device=n_tok.device)
    tok = ar[None] < n_tok[:, None]                                            # [B,N]
    allow = (ar[None, :, None] > ar[None, None, :]) & tok[:, :, None] & tok[:, None, :]
    if key_ok is not None:
        allow = allow & key_ok[:, None, :]
    return allow


def mix_pointer(logp_model: Tensor, gate_logits: Tensor, p: Tensor, null: Tensor) -> Tensor:
    """log((a0 + sum_h a_h null_h) p_model(y) + sum_h a_h p_h(y)), a = softmax(gate_logits);
    p, null [..., H]. The log of a NORMALISED distribution's mass at y."""
    a = torch.softmax(gate_logits.float(), -1)
    w0 = a[..., 0] + (a[..., 1:] * null.float()).sum(-1)
    s = (a[..., 1:] * p.float()).sum(-1)
    lc = torch.where(s > 0, torch.log(s.clamp_min(1e-30)), float("-inf"))
    return torch.logaddexp(torch.log(w0.clamp_min(1e-30)) + logp_model.float(), lc)


class PointerHead(nn.Module):
    def __init__(self, d: int, heads: int, dh: int = 64, cell_key: bool = False):
        super().__init__()
        self.h, self.dh, self.cell_key = heads, dh, bool(cell_key)
        self.q = nn.Linear(d, heads * dh, bias=False)
        self.k = nn.Linear(d, heads * dh, bias=False)
        for m in (self.q, self.k):
            nn.init.normal_(m.weight, std=0.02)
        if self.cell_key:
            self.kc = nn.Linear(d, heads * dh, bias=False)
            nn.init.zeros_(self.kc.weight)                 # step 0: the pointer without cell keys
        # 1-D / tiny params are named so recpre routes them to AdamW ("bias" / "norm" groups)
        self.null_k_bias = nn.Parameter(torch.randn(heads, dh) * 0.02)
        self.log_scale_bias = nn.Parameter(torch.full((heads,), math.log(math.sqrt(dh))))
        # gate over (model, head 1..H); init a ~ (0.96, 0.04 / H each)
        self.gate_norm_head = nn.Linear(d, 1 + heads)
        with torch.no_grad():
            self.gate_norm_head.weight.zero_()
            self.gate_norm_head.bias.copy_(torch.tensor([0.0] + [-3.2 - math.log(heads)] * heads))

    def forward(self, h: Tensor, labels: Tensor, is_slot: Tensor, layout=None):
        """h [B,L,d] coda output, labels [B,L] (-100 off-token), packed order; `layout` (MORPH's
        SlotLayout) only with cell_key. Returns (gate_logits [B,L,1+H], p_ptr [B,L,H] at the
        true target, null [B,L,H] the abstain mass, 1 where a query has no candidate)."""
        alpha, null, gl, yt, inv = self._attend(h, labels, is_slot, layout)
        same = (yt[:, :, None] == yt[:, None, :]) & (yt[:, :, None] >= 0)        # [B,N,N]
        p = (alpha * same[:, None].to(alpha.dtype)).sum(-1).transpose(1, 2)       # [B,N,H]
        g = inv[..., None].expand(-1, -1, self.h)
        return gl, p.gather(1, g), null.transpose(1, 2).gather(1, g)

    @torch.no_grad()
    def mixed_at(self, h: Tensor, labels: Tensor, is_slot: Tensor, logp_q: Tensor,
                 q_tok: Tensor, layout=None) -> Tensor:
        """Generation: the mixture's full log-probs [B, V] at ONE query per row. q_tok [B] is the
        query's TOKEN-order index, logp_q [B, V] the model's log-softmax there. Candidates are
        the labelled earlier tokens (label = the token that followed). Same rule as forward:
        p(v) = (a0 + sum_h a_h null_h) p_model(v) + sum_h a_h sum_j alpha_hj [y_j = v]."""
        alpha, null, gl, yt, inv = self._attend(h, labels, is_slot, layout)
        B, V = logp_q.shape
        b = torch.arange(B, device=h.device)
        order = torch.argsort(inv, dim=1)
        a = torch.softmax(gl[b, order[b, q_tok]].float(), -1)                     # [B,1+H]
        aq, nq = alpha[b, :, q_tok], null[b, :, q_tok]                            # [B,H,N] [B,H]
        beta = (a[:, 1:, None] * aq).sum(1)                                       # [B,N]
        w0 = a[:, 0] + (a[:, 1:] * nq).sum(1)
        p = torch.zeros(B, V, device=h.device, dtype=torch.float32)
        p.scatter_add_(1, yt.clamp_min(0), beta * (yt >= 0).float())
        return torch.log(w0[:, None] * logp_q.float().exp() + p)

    def _attend(self, h: Tensor, labels: Tensor, is_slot: Tensor, layout=None):
        """-> (alpha [B,H,N,N], null [B,H,N], gate logits [B,L,1+H] packed, yt [B,N], inv);
        token order (N = L), inv maps packed positions to token order."""
        B, L, d = h.shape
        order = token_order(is_slot)
        inv = torch.argsort(order, dim=1)
        ht = h.gather(1, order[..., None].expand(-1, -1, d))
        yt = torch.where(is_slot, torch.full_like(labels, -100), labels).gather(1, order)
        n_tok = (~is_slot).sum(1)
        allow = _earlier(n_tok, L, key_ok=yt >= 0)
        q = self.q(ht).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(ht).view(B, L, self.h, self.dh).transpose(1, 2)
        scale = self.log_scale_bias.exp() / math.sqrt(self.dh)
        extra = None
        if self.cell_key:
            if layout is None:
                raise ValueError("PointerHead(cell_key=True) needs the slot layout")
            extra = self._cell_scores(h, q, layout, order, scale)
        qn, kn = _rms(q), _rms(k)
        s = torch.einsum("bhid,bhjd->bhij", qn.float(), kn.float()) * scale[None, :, None, None]
        if extra is not None:
            s = s + extra
        s = s.masked_fill(~allow[:, None], float("-inf"))
        s_null = torch.einsum("bhid,hd->bhi", qn.float(), _rms(self.null_k_bias).float()) * scale[None, :, None]
        pr = torch.softmax(torch.cat([s, s_null[..., None]], -1), -1)
        return pr[..., :-1], pr[..., -1], self.gate_norm_head(h), yt, inv

    def _cell_scores(self, h, q, layout, order, scale):
        """[B,H,N,N] extra score q_i . kc(cells of span(j)) where span(j) < span(i), else 0."""
        B, L, d = h.shape
        S = layout.slot_index.shape[1]
        K = int(layout.prefix_k)
        pos = (layout.slot_index[:, :, None] + torch.arange(K, device=h.device)).clamp(max=L - 1)
        cells = h.gather(1, pos.reshape(B, S * K, 1).expand(-1, -1, d)).view(B, S, K, d).mean(2)
        cells = cells * layout.slot_valid[..., None].to(cells.dtype)              # [B,S,d]
        bag_t = layout.bag_id.gather(1, order)                                     # token order
        span = bag_t.clamp(0, S - 1)
        ck = self.kc(cells).view(B, S, self.h, self.dh)                           # [B,S,H,dh]
        ckj = ck.gather(1, span[:, :, None, None].expand(-1, -1, self.h, self.dh))  # [B,N,H,dh]
        e = torch.einsum("bhid,bjhd->bhij", _rms(q).float(), ckj.float()) * scale[None, :, None, None]
        earlier_span = (bag_t[:, None, :] < bag_t[:, :, None]) & (bag_t[:, None, :] < S)
        return e * earlier_span[:, None].to(e.dtype)


class IdentityReach(nn.Module):
    def __init__(self, d: int, heads: int, dh: int = 128):
        super().__init__()
        self.h, self.dh = heads, dh
        self.q = nn.Linear(d, heads * dh, bias=False)
        self.k = nn.Linear(3 * d, heads * dh, bias=False)
        self.v = nn.Linear(3 * d, heads * dh, bias=False)
        self.o = nn.Linear(heads * dh, d, bias=False)
        for m in (self.q, self.k, self.v):
            nn.init.normal_(m.weight, std=0.02)
        nn.init.zeros_(self.o.weight)                       # step 0: strict LXTUL exactly
        self.null_k_bias = nn.Parameter(torch.randn(heads, dh) * 0.02)
        self.log_scale_bias = nn.Parameter(torch.full((heads,), math.log(math.sqrt(dh))))

    def forward(self, x: Tensor, emb: Tensor, is_slot: Tensor) -> Tensor:
        """x [B,L,d] coda input, emb [B,L,d] raw token embeddings, packed order.
        Returns the residual add [B,L,d] (zero at slot positions)."""
        B, L, d = x.shape
        order = token_order(is_slot)
        inv = torch.argsort(order, dim=1)
        n_tok = (~is_slot).sum(1)
        tok = torch.arange(L, device=x.device)[None] < n_tok[:, None]
        et = emb.gather(1, order[..., None].expand(-1, -1, d)) * tok[..., None].to(emb.dtype)
        prev = F.pad(et, (0, 0, 1, 0))[:, :L]                # e(x_{j-1}); zero at j = 0
        nxt = F.pad(et, (0, 0, 0, 1))[:, 1:]                 # e(x_{j+1}); zero past the last token
        win = torch.cat([prev, et, nxt], -1)
        xt = x.gather(1, order[..., None].expand(-1, -1, d))
        q = self.q(xt).view(B, L, self.h, self.dh).transpose(1, 2)
        k = self.k(win).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(win).view(B, L, self.h, self.dh).transpose(1, 2)
        allow = _earlier(n_tok, L)
        _, _, out = _attn(q, k, v, self.null_k_bias, allow, self.log_scale_bias.exp() / math.sqrt(self.dh))
        out = self.o(out.transpose(1, 2).reshape(B, L, self.h * self.dh))
        out = out * tok[..., None].to(out.dtype)
        out = out.gather(1, inv[..., None].expand(-1, -1, d))
        return out.masked_fill(is_slot[..., None], 0.0)
