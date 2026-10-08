"""Plain Parcae + the learned pointer head (`copy_heads.PointerHead`): the like-for-like control
for strict LXTUL + pointer. Same head, same gate, same mixture; no slots, so every earlier
token is a candidate (plain attention already reads them all, so the head is the only change).

The wrapper captures the final-norm output with a forward hook on `transformer.ln_f` and scores
the inner model's logits, so Parcae's recurrence sampling, compile and checkpointing are untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lxtul.copy_heads import PointerHead, mix_pointer


class PlainPointer(nn.Module):
    def __init__(self, inner: nn.Module, heads: int):
        super().__init__()
        self.inner = inner
        self.pointer = PointerHead(inner.config.n_embd, heads)
        self.use_pointer = True                   # mixture_off_eval flips it (the leak check)
        self._h: Tensor | None = None
        inner.transformer.ln_f.register_forward_hook(self._grab)

    @property
    def config(self):
        return self.inner.config

    def _grab(self, mod, args, out):
        self._h = out

    def _nll(self, x: Tensor, labels: Tensor, **kw) -> Tensor:
        """Per-position NLL [B, L] under the mixture (0 where label is -100)."""
        if hasattr(self, "step"):
            self.inner.step = self.step
        logits = self.inner(x, return_logits=True, **kw)["logits"]
        nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten().clamp_min(0),
                              reduction="none").view(labels.shape)
        if self.use_pointer:
            no_slot = torch.zeros_like(labels, dtype=torch.bool)
            g, p, null = self.pointer(self._h, labels, no_slot)
            nll = -mix_pointer(-nll, g, p, null)
        self._h = None
        return torch.where(labels >= 0, nll, 0.0)

    def forward(self, x: Tensor, labels: Tensor | None = None, **kw) -> dict:
        if labels is None:
            return self.inner(x, **kw)
        nll = self._nll(x, labels, **kw)
        n = (labels >= 0).sum().clamp_min(1)
        return {"loss": nll.sum() / n}

    @torch.no_grad()
    def token_nll(self, x: Tensor, labels: Tensor, **kw) -> Tensor:
        return self._nll(x, labels, **kw)
