"""The ONE next-token sampling step, shared by every generator in the tree.

There are two generators — `tul_generate.generate_tul` (slots) and
`plain_generate.generate_plain` (no slots) — and their whole purpose is to be
compared against each other. If each carried its own copy of the temperature /
top-k / multinomial block, any drift between the copies would land directly on the
A1-minus-A0 number the comparison exists to produce. So the block lives here once.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["sample_next"]


def sample_next(logits: Tensor, temperature: float, top_k: int,
                generator: torch.Generator | None) -> int:
    """One token from a `[vocab]` logit row.

    ``temperature <= 0`` is greedy (argmax). Greedy is a DIAGNOSTIC, not the mode to
    rank models in: it is the mode where a healthy model still loops, so it measures
    the readout's argmax basin rather than the distribution the model learned. Rank on
    sampled modes; read greedy to see whether a loop exists at all.
    """
    logits = logits.float()
    if temperature <= 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
