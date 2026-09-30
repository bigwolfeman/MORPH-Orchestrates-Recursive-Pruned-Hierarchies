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
                generator: torch.Generator | None, top_p: float = 0.0) -> int:
    """One token from a `[vocab]` logit row.

    ``temperature <= 0`` is greedy (argmax). Greedy is a DIAGNOSTIC, not the mode to
    rank models in: it is the mode where a healthy model still loops, so it measures
    the readout's argmax basin rather than the distribution the model learned. Rank on
    sampled modes; read greedy to see whether a loop exists at all.

    ``top_p`` (nucleus sampling, Holtzman et al. 2019): ``0.0`` (default) is OFF and the
    function is byte-identical to every call site that predates this parameter — it is
    appended last and keyword-safe for that reason. At ``0 < top_p < 1`` the logits are
    sorted descending, softmax'd, and every token PAST the first one whose cumulative
    probability mass exceeds ``top_p`` is masked to ``-inf`` (the smallest set whose mass
    is ``>= top_p`` is kept; the top token is never dropped, since its own cumulative-minus-
    itself is 0). Applied AFTER ``top_k`` if both are set, on the temperature-scaled
    logits, matching the standard (Hugging Face) nucleus-filtering order.
    """
    logits = logits.float()
    if temperature <= 0.0:
        return int(logits.argmax())
    logits = logits / temperature
    if top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        # Token i is dropped when the mass STRICTLY before it already reached top_p —
        # so the token that crosses the threshold is kept, and index 0 (cum-before = 0)
        # is never dropped even if top_p is tiny.
        drop = (cum - sorted_probs) > top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1, generator=generator))
