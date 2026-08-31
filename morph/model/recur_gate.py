"""GRT recurrence gate — the Eq. 4-5 elementwise convex blend for the TUL core loop.

Gated Recurrent Transformers (arXiv 2608.15062, docs/references.md row 69; verified
against the PDF's Eqs. 2-5 and Appendix A on 2026-08-30). Per recurrence step:

    g     = sigmoid( f_g([LN(h_prev), LN(h_pre)]) / tau + eps_g )      (Eq. 5)
    h_new = g * h_prev + (1 - g) * o                                   (Eq. 4)

where ``o`` is the shared core's proposal, ``f_g`` a two-layer SiLU MLP with hidden
dimension d, and the SECOND linear's bias is initialised to +4 so g ~ 0.98 at init:
the copy branch dominates and training starts near-identity (their Table 5 note; the
bias-init sweep in their B.6 spans only 0.019 nats, so the init is a convenience,
not the mechanism). ``eps_g ~ N(0, sigma_g^2)`` per scalar, training only
(Appendix A: sigma_g = 0.1, tau = 1.0).

What we deliberately do NOT import (program note
.agents/notes/proposed/architecture/2026-08-30-gate-ladder-program.md):

- **Eq. 2's W_proj prelude re-injection** — MORPH's core already adds the
  prelude-derived injection term every iteration (``inj_core_terms`` in
  ``_tul_core``/``_core_region``), which is the same "fresh context-grounded input
  at every recurrence" their Table 7 credits with -0.198 nats.
- **State noise eps_x** (their -0.019 at 2k steps) — kept out of round 1 to isolate
  the gate.
- **Uniform {1..R} depth sampling** — arm G4's variable, not the gate's.

The hard constraint this module must respect (cond-zero probe,
lab/experiments/successes/2026-08-30-tul-condzero-probe.md): the gate is keyed on
STATE + PRELUDE only. It never sees the iteration index — index-keyed signals
poison depth-earning during formation even when the trained model abandons them.

MORPH adaptation: the TUL loop carrier is ``[B, S, n_streams, d]`` (Cayley
HyperConnection, n=4). One shared MLP is applied per stream: LN over d, concat the
gate's two inputs on the last dim, so ``g`` has the carrier's full shape and the
blend is elementwise in the whole carrier space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RecurrenceGate(nn.Module):
    """Elementwise convex-blend gate over the TUL slot carrier.

    ``forward(h_prev, e) -> g`` with ``g.shape == h_prev.shape`` and every value in
    (0, 1). The caller applies Eq. 4; this module owns only Eq. 5.
    """

    def __init__(self, d_model: int, tau: float = 1.0, bias_init: float = 4.0,
                 noise: float = 0.1):
        super().__init__()
        if tau <= 0.0:
            raise ValueError(f"recurrence-gate tau must be positive, got {tau}")
        self.tau = float(tau)
        self.noise = float(noise)
        # LayerNorm without learnable affine: Eq. 5's LN is a normalisation of the
        # gate's INPUTS, and a learnable scale here would alias with fc1's rows.
        self.norm_h = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm_e = nn.LayerNorm(d_model, elementwise_affine=False)
        self.fc1 = nn.Linear(2 * d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        nn.init.constant_(self.fc2.bias, bias_init)

    def forward(self, h_prev: Tensor, e: Tensor) -> Tensor:
        z = torch.cat([self.norm_h(h_prev), self.norm_e(e)], dim=-1)
        logits = self.fc2(F.silu(self.fc1(z))) / self.tau
        if self.training and self.noise > 0.0:
            logits = logits + torch.randn_like(logits) * self.noise
        return torch.sigmoid(logits)
