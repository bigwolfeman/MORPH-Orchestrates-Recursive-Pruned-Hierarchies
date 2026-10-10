"""``tul.disco`` — DiscoLoop re-embedding between slot-loop passes (2026-10-10).

DiscoLoop (Fu et al. 2026, arXiv 2607.00341) re-embeds the hidden state through the tied
LM head between two passes of a looped transformer::

    Phi(h)  = sum_v softmax(W h / tau)_v W[v]               (W the tied table)
    H~(k)   = H(k) + alpha(k) * RMSNorm(Phi(H(k)))           H(k+1) = f(H~(k))
    alpha   = sigmoid(<w_a, Phi_n> + b_a) per position, or a fixed constant
    the LAST pass is ungated: the prediction reads H(K) directly.

Their diagnosis: after pass 1 the state already decodes the bridge token with P ~ 1, yet
sits at cosine ~0.3 to that token's embedding, so the next pass's retrieval (which works in
embedding space) cannot use it. The term hands the next pass the token's own embedding row.

WHERE MORPH RUNS IT (``MORPHTransformer._tul_core``, the one site, and its reasons):

* The term is part of the INPUT of pass ``t`` (``t >= 1``), added to ``_h_in`` after
  the fan trigger and BEFORE the gain hinge, so the hinge measures the map at the
  operating point the pass actually got (the term is a constant there: the hinge's
  finite difference perturbs ``_h_in`` and does not recompute Phi, the trigger's and the
  grad-pass feature's "exogenous input" reading). ``_h_in`` is a positional input of the
  checkpointed step, so a recompute replays the term with the value the pass used.
* It is added ONLY to cells whose depth continues into pass ``t`` (``depths > t``) and
  that are valid. A finished cell keeps serving its keys and values from its frozen
  carrier (runtime-invariants §6b); a term there would change what later slots read
  from it. Under ``model.slot_compact`` the same mask is the pass's active set, so the
  compacted pass computes exactly the cells that got a term.
* Nothing is added after a slot's LAST pass: the exit (what ``W_prefix`` and the coda
  read) is the raw output of that pass. That is the paper's "last pass ungated".
* The fixed-point term and the slot-cell norm read the CARRIED state ``h`` / ``_h_det``,
  which never holds the term.

WHAT IS READ AND HOW THE LOGITS ARE MADE: the carried cell after the slot-cell pass norm
(``h`` at the top of the pass), through the model's own token head: ``_readout`` (the
Hyper-Connection stream mean, ``lm_mixer``, ``final_norm``) then ``embed.attend`` with the
structural slot id at ``-inf`` — the same logits the coda's tokens are scored with, minus
the pointer mixture (an output-only copy head over earlier TOKEN states, which a slot
cell does not have). This module receives the logits and the table; it never builds them.

THE TABLE GRADIENT: ``tul.disco_detach_table`` (default true). An undetached auxiliary
head through the tied table diverged arm v1a by step 2800 (memory note
``morph-lm-head-is-weight-tied``, ``TULConfig.mux_detach_head``). With the table detached
the gradient still reaches the loop through ``h`` (the logits) and ``lm_mixer`` /
``final_norm``, which the token head shares.

THE BRIDGE (``tul.disco_bridge``):

* ``soft``  — the paper: softmax over the whole vocabulary.
* ``topk``  — softmax over the top ``tul.disco_topk`` logits only (the paper's LM uses
  top-128); the other rows get weight 0 and no gradient.
* ``hard``  — forward value is the ARGMAX row ``W[argmax]`` exactly (the top-1 of the same
  top-k call), backward is the gradient of the top-k soft mixture (straight-through:
  ``hard + (soft - soft.detach())``, whose forward value is ``hard`` to the bit because
  ``x - x`` is exactly 0). Why it exists: in DiffusionBlocks the soft ``softmax @ E``
  bridge shrank the norm to 0.27 and wrecked generation, and the argmax row fixed it
  (memory ``db-bridge-is-the-bug``). The RMSNorm below fixes the norm; it does not fix
  the direction of a blurred mixture, which the hard row does.

THE NORM: ``attention.RMSNorm`` with its learned gain (init 1), the tree's class.

THE GATE (``tul.disco_gate``): ``learned`` is ``sigmoid(<w, Phi_n> + b)`` per position,
``w`` init 0 and ``b`` init ``tul.disco_gate_bias`` (0.0 -> alpha 0.5, OPEN at step 0: the
snap arm's pseudo gate started at 0 and never opened); ``fixed`` is the constant
``tul.disco_alpha``. At ``fixed`` alpha 0 the term is exactly 0 and the forward and every
gradient equal the model without the key (``tests/test_tul_disco.py``).

HOW A d-VECTOR ENTERS THE n STREAMS: broadcast, one full copy per stream
(``MORPHTransformer._apply_injection``) — the carrier's rule for every single-stream term
(the x0 / bigram injections, the register's seed term, ``DiagonalInjection``'s per-stream
elementwise source). So the read state (the stream mean) moves by exactly the term. Full
width, like the paper's residual add and the fan trigger; NOT ``DiagonalInjection``'s
context-channel slice, which would hand the next pass only 320 of the 1024 channels of
the embedding row.

Every parameter is a deterministic constant at construction (no RNG drawn: ones, zeros
and a constant bias), so a model built with the key has the same base weights as its
same-seed twin without it. Never ternarised (``_ternary_exclude``; none of them is an
``nn.Linear``), no-decay by name (``tul_disco`` in ``optimizer._NO_DECAY_KEYWORDS``).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import RMSNorm

__all__ = ["DISCO_BRIDGES", "DISCO_GATES", "disco_bridge", "TULDisco"]

DISCO_BRIDGES = ("soft", "topk", "hard")
DISCO_GATES = ("learned", "fixed")


def disco_bridge(logits: Tensor, table: Tensor, mode: str, k: int, tau: float
                 ) -> tuple[Tensor, Tensor, Tensor]:
    """``Phi`` for ``logits`` ``[N, V]`` against ``table`` ``[V, C]``.

    Returns ``(phi [N, C] fp32, p [N, V or k] fp32, top1 [N] int64)``: the bridge's own
    distribution ``p`` (the full softmax under ``soft``, the renormalised top-k softmax
    under ``topk`` / ``hard``) and the argmax token, both for the eval instruments.

    ``table`` is used as given; the caller detaches it (``tul.disco_detach_table``). The
    logits are divided by ``tau`` in fp32. The top-k rows are gathered with
    ``F.embedding`` (an index into the table, no ``[N, V]`` one-hot); the soft mixture is
    one ``[N, V] @ [V, C]`` product."""
    lf = logits.float() / float(tau)
    if mode == "soft":
        p = torch.softmax(lf, dim=-1)                                         # [N, V]
        phi = p @ table.float()                                               # [N, C]
        return phi, p, lf.argmax(dim=-1)
    val, idx = lf.topk(int(k), dim=-1)                                        # [N, k] sorted
    p = torch.softmax(val, dim=-1)                                            # [N, k]
    rows = F.embedding(idx, table).float()                                    # [N, k, C]
    soft = torch.einsum("nk,nkc->nc", p, rows)
    if mode == "topk":
        return soft, p, idx[:, 0]
    # hard: the argmax row in the forward, the top-k mixture's gradient in the backward.
    # `soft - soft.detach()` is exactly 0 in the forward, so the value is the row itself.
    return rows[:, 0] + (soft - soft.detach()), p, idx[:, 0]


class TULDisco(nn.Module):
    """The learnable pieces of the DiscoLoop term: the RMSNorm gain on Phi and, under the
    learned gate, ``gate_w`` ``[C]`` and ``gate_b`` ``[]``. :meth:`forward` maps the head's
    logits and the table to ``(term, alpha, p, top1)``; the caller masks and adds the term.
    """

    def __init__(self, d_model: int, *, bridge: str, topk: int, tau: float, gate: str,
                 alpha: float, gate_bias: float):
        super().__init__()
        if bridge not in DISCO_BRIDGES:
            raise ValueError(f"tul.disco_bridge must be one of {DISCO_BRIDGES}, got {bridge!r}")
        if gate not in DISCO_GATES:
            raise ValueError(f"tul.disco_gate must be one of {DISCO_GATES}, got {gate!r}")
        self.bridge = bridge
        self.topk = int(topk)
        self.tau = float(tau)
        self.gate = gate
        self.alpha = float(alpha)
        d = int(d_model)
        self.norm = RMSNorm(d)                              # weight ones, no RNG
        self.gate_w: nn.Parameter | None = None
        self.gate_b: nn.Parameter | None = None
        if gate == "learned":
            self.gate_w = nn.Parameter(torch.zeros(d))
            self.gate_b = nn.Parameter(torch.full((), float(gate_bias)))
        self._ternary_exclude = True

    def forward(self, logits: Tensor, table: Tensor
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``logits`` ``[N, V]``, ``table`` ``[V, C]`` -> ``(term [N, C] fp32, alpha [N, 1]
        fp32, p, top1)``; ``term = alpha * RMSNorm(Phi)``."""
        phi, p, top1 = disco_bridge(logits, table, self.bridge, self.topk, self.tau)
        phi_n = self.norm(phi)                                                # fp32
        if self.gate == "learned":
            alpha = torch.sigmoid(phi_n @ self.gate_w.float() + self.gate_b.float())
            alpha = alpha.unsqueeze(-1)
        else:
            alpha = phi_n.new_full((phi_n.shape[0], 1), self.alpha)
        return alpha * phi_n, alpha, p, top1
