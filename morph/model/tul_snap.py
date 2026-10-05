"""``tul.pseudo_k`` — the trace-free pseudo-token carrier ("snap", 2026-10-05).

LXTUL's latent-selected loop (``tul.fan_loop_select``, ``tul_fan_route.py``) writes the
loop's final winner cell into its ONE prefix position and leaves the other ``fan_k - 1``
prefix positions of that slot EXACTLY zero (``_fan_route_cells`` in ``transformer.py``
zeroes every loser cell before the write). This module fills up to ``pseudo_k`` of those
zero positions with a PonderLM-style (arXiv 2602.00959) snapped mixture of the SPAN'S OWN
TOKENS — never the next span, never the full vocabulary — so the coda gets, beside the
one looped vector, up to ``pseudo_k`` positions that can read like a raw copied token.

Why this and not a wider linear write (``tul.prefix_per_cell``, ``tul_fan.py``'s
register): a plain affine map of one ``d``-wide cell into ``N`` coda positions can show at
most one EXACT arbitrary token (`` exact_tuple_write_needs_dim`` in
``lab/theory/tul_pseudotoken/Carrier.lean``) — a real tied table spans the full ``d``-dim
space, so ``r = d`` and ``N * d <= d`` forces ``N <= 1``. A snapped vocabulary mixture
escapes that bound: once the logits put the right candidate first by a margin, the
written vector sits within ``2 (n-1) exp(-margin)`` of the EXACT embedding
(``snap_close``), whatever else the cell holds — the tied table is a free nearest-token
clean-up a linear write has no access to. Full derivation and the theorem index:
``.agents/notes/proposed/architecture/2026-10-05-trace-free-pseudo-token-carrier.md``,
``lab/theory/tul_pseudotoken/README.md``.

The mechanism, per pseudo position ``n in 1..pseudo_k`` of slot ``s``::

    c    = the winner cell (``tul.pseudo_source="exit"``: after the loop's LAST pass, as
           :meth:`TULSlots.prefix_project` reads it — the Hyper-Connection stream MEAN,
           the "state of this cell" convention every other per-cell reader in the tree
           uses (``tul_fan._cell_readout``); ``"entry"``: the slot's loop INPUT, the same
           quantity :meth:`MORPHTransformer._lsel_begin` calls ``ctx`` — a PASS-INDEPENDENT
           twin, the bypass control of ``bypass_zeroes_passes``). NOT detached under
           ``"exit"``: the coda's CE reaches the loop through it, exactly the PonderLM
           mechanism (its mixture is trained end to end, not a frozen readout).
    q_n  = W_q[n] @ RMSNorm(c)
    l_ni = exp(beta_n) * <q_n, RMSNorm(E[x_i])> / sqrt(d)    for x_i the span's OWN tokens
    p_n  = softmax_i(l_n)                                     (masked to the real span length)
    pseudo_n = g_n * sum_i p_ni * E[x_i]                      E = lm_weight().detach()

``E`` is passed in already DETACHED by the caller (:meth:`MORPHTransformer._tul_pseudo_snap_write`),
so this module never sees it with grad and the tied embedding table trains from exactly
the paths it always did — the ``mux_detach_head`` rule, applied to a READ instead of a
write. ``g_n`` starts at 0 (``tul.pseudo_gate_init``), so a freshly-built head writes
EXACTLY zero at every pseudo position: ``pseudo_k > 0`` with every ``g_n`` still at its
init is bit-identical to ``pseudo_k == 0`` (the ``W_prefix`` / ``prefix_per_cell`` zero-
and-identity-init precedent, applied to a gate rather than a weight). ``W_q`` is a plain
trainable matrix, never quantised (``_ternary_exclude``, the ``W_prefix`` /
``FanRouter`` treatment) and RNG-neutral: drawn from a PRIVATE generator so building this
module never perturbs the global RNG stream a same-seed ``pseudo_k=0`` model's later
construction draws from.

Numerics: a slot can be INVALID (padding) or, in principle, have an empty own span. The
softmax mask uses the "at least one True per row" safeguard (:func:`_safe_mask`) so no
row of an invalid slot's candidate set is ALL ``-inf`` — an all-masked softmax row is NaN,
and ``NaN * 0`` is still NaN, so a dummy invalid-slot row would otherwise poison the
gradient of ``W_q`` / ``beta`` / ``g`` for every OTHER slot in the same backward. The
caller never actually reads an invalid slot's pseudo vector: every scatter position of an
invalid slot already routes to the dump row (``TULSlots.prefix_positions``), the same
mechanism every other per-slot write in the tree relies on.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["TULPseudoSnap", "rms_norm"]

# Private init stream for W_q. Never the global RNG — the `TULSlots.W_sent` rule.
_SEED_WQ = 0x5A4921

_NEG_INF = float("-inf")


def rms_norm(x: Tensor, eps: float = 1e-6) -> Tensor:
    """Plain RMSNorm, no affine weight: ``x / sqrt(mean(x^2, dim=-1) + eps)``, fp32 in,
    native dtype out. The spec's ``RMSNorm(.)`` — :class:`attention.RMSNorm` minus the
    learned ``weight``, because here both sides of every inner product (the query and
    every candidate embedding) must sit on the SAME unit-RMS sphere with no per-side
    rescale, or the margin theorems (``snap_close``, ``exists_scale_commit``) that justify
    a free logit scale ``beta`` stop applying."""
    xf = x.float()
    norm = xf.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return (xf * norm).to(x.dtype)


def _safe_mask(valid: Tensor) -> Tensor:
    """``[..., J]`` bool -> the same, with every ALL-FALSE row forced to ALL-TRUE.

    A row with no True entry only happens for an invalid slot (or, in principle, a slot
    whose own span is empty — ``tul.min_span >= 1`` already rules that out for a real
    span). The caller never reads that slot's pseudo vector (it scatters to the dump
    row), so turning its mask into "every candidate equally valid" costs nothing and
    buys a finite, differentiable softmax instead of an all-``-inf`` row whose NaN would
    otherwise reach every other slot's gradient through the shared ``W_q`` / ``beta``."""
    return valid | (~valid.any(dim=-1, keepdim=True))


class TULPseudoSnap(nn.Module):
    """The three learnable pieces of the snap: ``W_q`` ``[k, d, d]``, ``beta`` ``[k]``,
    ``g`` ``[k]``. See the module docstring for the formula."""

    def __init__(self, d_model: int, pseudo_k: int, scale_init: float, gate_init: float):
        super().__init__()
        d, k = int(d_model), int(pseudo_k)
        if k < 1:
            raise ValueError(f"TULPseudoSnap needs pseudo_k >= 1, got {k}")
        self.k = k
        g = torch.Generator(device="cpu").manual_seed(_SEED_WQ)
        with torch.no_grad():
            w_q = torch.empty(k, d, d).normal_(mean=0.0, std=0.02, generator=g)
        self.W_q = nn.Parameter(w_q)
        self.beta = nn.Parameter(torch.full((k,), float(scale_init)))
        self.g = nn.Parameter(torch.full((k,), float(gate_init)))
        self._ternary_exclude = True

    def forward(self, c: Tensor, ids: Tensor, valid: Tensor, table: Tensor
               ) -> tuple[Tensor, Tensor, Tensor]:
        """``c`` ``[B, S, C]`` the per-slot source (grad flows in under ``"exit"``);
        ``ids`` / ``valid`` ``[B, S, J]`` the span's own candidate tokens
        (:func:`morph.model.tul_spandec.own_span_slots`); ``table`` ``[V, C]`` the
        DETACHED tied embedding (the caller detaches it; this module never re-detaches,
        so a caller mistake would show up as a gradient reaching ``E`` — pinned by
        ``tests/test_pseudo_snap.py::test_no_grad_to_embedding``).

        Returns ``(pseudo [B, S, k, C], p [B, S, k, J] fp32, vertex_mass [B, S, k] fp32)``
        — ``p`` and ``vertex_mass`` are the val instruments
        (:meth:`MORPHTransformer._tul_pseudo_snap_write`); the caller detaches them
        before any ``float()`` host sync.
        """
        d = c.shape[-1]
        cn = rms_norm(c).float()                                      # [B,S,C]
        q = torch.einsum("bsc,kdc->bskd", cn, self.W_q.float())        # [B,S,k,d]
        cand = table.float()[ids]                                      # [B,S,J,C]
        cand_n = rms_norm(cand).float()                                 # [B,S,J,C]
        logits = torch.einsum("bskd,bsjd->bskj", q, cand_n) / (d ** 0.5)
        logits = logits * self.beta.float().exp().view(1, 1, -1, 1)
        safe = _safe_mask(valid)                                        # [B,S,J]
        logits = logits.masked_fill(~safe.unsqueeze(2), _NEG_INF)
        p = F.softmax(logits, dim=-1)                                   # [B,S,k,J]
        pseudo = torch.einsum("bskj,bsjc->bskc", p, cand)               # [B,S,k,C] fp32
        pseudo = pseudo * self.g.float().view(1, 1, -1, 1)
        vertex_mass = p.amax(dim=-1)                                    # [B,S,k]
        return pseudo.to(c.dtype), p, vertex_mass
