"""Span-level NextLat: a transition model on the slot loop's exit states (2026-09-25).

Paper: Teoh et al., "Next-Latent Prediction Transformers Learn Compact World Models",
arXiv 2511.05963. Their model adds one auxiliary to next-token training: a small network
``p_psi(h_t, x_{t+1})`` predicts the NEXT hidden state ``h_{t+1}`` from the current one and
the next token, under SmoothL1 against ``stop_grad(h_{t+1})`` (plus a KL through the frozen
output head). If both the token prediction and the transition are exact, ``h_t`` is a
BELIEF STATE: a sufficient statistic of the past for the future (their Theorem 3.2, d = 1).
Composed, the transition drafts: ``h -> token -> h -> token ...`` (3.3x self-speculative
decoding at 1.3B).

Here the unit is the SPAN, not the token. The slot loop's exit state ``z_s`` (the readout
of slot ``s``, the state every reader grades) summarises the text up to span ``s`` and is
the state the next span is decoded from. The transition reads ``z_s`` and the tokens of
span ``s + 1`` and predicts ``z_{s+1}``:

    z_hat_{s+1} = W_out GRU(h_0 = z_s; rmsnorm(E[t_1]), ..., rmsnorm(E[t_n]))[n]
    L = SmoothL1(z_hat_{s+1}, stop_grad(z_{s+1}))        over valid (s, s+1) pairs

A GRU over the span's tokens is the span-level form of NextLat's per-token transition: the
same map applied once per token and composed over the span, with a target only where the
model has a state to predict (the next slot). The gradient reaches ``z_s`` (the input,
not stop-graded), which is how the auxiliary shapes the loop's exit into a belief state;
the target is stop-graded, the paper's only collapse guard. The token embeddings come from
the tied table DETACHED (``morph/model`` rule: an auxiliary head never trains the table
the coda speaks through).

Why here (Wolfe's TUL intent, "think once, decode cheap"): a transition that predicts the
next slot state from the current one and the span just generated lets a decoder DRAFT the
next slot without running the loop, and the loop verifies. The draft instrument
(``nextlat_draft_ce``) measures that use directly: the committed parallel head's CE of
span ``s + 2`` read from the DRAFTED ``z_hat_{s+1}`` against the same CE read from the
loop's own ``z_{s+1}``.

Not built: the paper's KL term. It needs full-vocabulary logits of both states at every
slot and every span position (~1.5 k slots x 32 x 49 k per step here), and this tree's
readers never materialise ``[.., V]``. The paper calls it complementary; the draft
instrument reads the token-space agreement it would enforce.

Never quantised, never pruned (``_ternary_exclude``); built RNG-neutral (a forked stream),
so an arm's base weights are byte-identical to its parent's.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_BUILD_SEED = 20250925


class SpanTransition(nn.Module):
    """``(z_s [N, C], emb [N, J, C], lengths [N]) -> z_hat [N, C]``.

    A single-layer GRU whose initial hidden state is ``z_s`` steps through the span's
    ``lengths[i]`` token embeddings; the hidden state after the LAST valid token goes
    through ``out`` (identity at init, no RNG). Positions past ``lengths[i]`` are padding:
    the GRU is causal, so they never reach the gathered state.
    """

    def __init__(self, d_model: int):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_BUILD_SEED)
            self.gru = nn.GRU(d_model, d_model, num_layers=1, batch_first=True)
            self.out = nn.Linear(d_model, d_model)       # default init drawn, then replaced
        with torch.no_grad():
            self.out.weight.copy_(torch.eye(d_model))
            self.out.bias.zero_()
        self._ternary_exclude = True
        self.gru._ternary_exclude = True
        self.out._ternary_exclude = True

    def forward(self, z_s: Tensor, emb: Tensor, lengths: Tensor) -> Tensor:
        if z_s.shape[0] == 0:
            return z_s.new_zeros(z_s.shape, dtype=torch.float32)
        # fp32 with autocast off: ~1 k sequences x <= 32 steps per step, small next to the
        # coda, and cuDNN's GRU needs its input and weight dtypes to match.
        with torch.autocast(device_type=z_s.device.type, enabled=False):
            seq, _ = self.gru(emb.float(), z_s.float().unsqueeze(0).contiguous())   # [N, J, C]
            last = (lengths.long() - 1).clamp_min(0)
            h = seq.gather(1, last.view(-1, 1, 1).expand(-1, 1, seq.shape[-1])).squeeze(1)
            return self.out(h)


def nextlat_pairs(valid: Tensor) -> Tensor:
    """``[B, S]`` bool: slot ``s`` has a successor state to predict. ``valid`` is
    :func:`~morph.model.tul_spandec.span_slots` at ``shift=1``, whose validity already
    requires slot ``s``, a complete span ``s + 1`` and slot ``s + 1``. The last slot index
    can never pair (no ``s + 1`` inside the row)."""
    pair = valid.any(dim=-1)
    pair[:, -1] = False
    return pair


def span_token_embeddings(ids: Tensor, valid: Tensor, table: Tensor) -> Tensor:
    """``[.., J, C]``: RMS-normalised rows of the DETACHED tied table, zero at padding."""
    emb = F.embedding(ids.clamp_min(0), table.detach())
    emb = F.rms_norm(emb.float(), (emb.shape[-1],))
    return emb * valid.unsqueeze(-1).to(emb.dtype)
