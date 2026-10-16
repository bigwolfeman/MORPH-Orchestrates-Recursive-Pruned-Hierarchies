"""Plain Parcae (or the flat GPT, `kind: gpt`) + the learned pointer head (`copy_heads.PointerHead`): the like-for-like control
for strict LXTUL + pointer. The flat GPT has no recurrence kwargs; its `ln_f` output feeds the same head. Same head, same gate, same mixture; no slots, so every earlier
token is a candidate (plain attention already reads them all, so the head is the only change).

The wrapper captures the final-norm output with a forward hook on `transformer.ln_f` and scores
the inner model's logits, so Parcae's recurrence sampling, compile and checkpointing are untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lxtul.copy_heads import PointerHead, mix_pointer
from lxtul.plain_deep import deep_forward, head_nll


class PlainPointer(nn.Module):
    def __init__(self, inner: nn.Module, heads: int):
        super().__init__()
        self.inner = inner
        self.pointer = PointerHead(inner.config.n_embd, heads)
        self.use_pointer = True                   # mixture_off_eval flips it (the leak check)
        self.plain_deep_passes = 0                # >0: HRM deep supervision on the plain loop (lxtul/plain_deep.py), cap on passes
        self._h: Tensor | None = None
        inner.transformer.ln_f.register_forward_hook(self._grab)

    @property
    def config(self):
        return self.inner.config

    def _grab(self, mod, args, out):
        self._h = out

    def _mix(self, h: Tensor, nll: Tensor, labels: Tensor) -> Tensor:
        """The pointer mixture on the per-position CE nll [B, L] of the final-norm output h (0 where label is -100)."""
        if self.use_pointer:
            no_slot = torch.zeros_like(labels, dtype=torch.bool)
            g, p, null = self.pointer(h, labels, no_slot)
            nll = -mix_pointer(-nll, g, p, null)
        return torch.where(labels >= 0, nll, 0.0)

    def _score(self, h: Tensor, logits: Tensor, labels: Tensor) -> Tensor:
        """Per-position NLL [B, L] under the mixture from the final-norm output h and its logits (0 where label is -100)."""
        return self._mix(h, F.cross_entropy(logits.flatten(0, 1), labels.flatten().clamp_min(0),
                                            reduction="none").view(labels.shape), labels)

    def _nll(self, x: Tensor, labels: Tensor, **kw) -> Tensor:
        """Per-position NLL [B, L] under the mixture (0 where label is -100)."""
        if hasattr(self, "step"):
            self.inner.step = self.step
        if self._deep_active():
            return self._deep_nll(x, labels, **kw)
        logits = self.inner(x, return_logits=True, **kw)["logits"]
        nll = self._score(self._h, logits, labels)
        self._h = None
        return nll

    def _deep_active(self) -> bool:
        return bool(getattr(self.inner, "cycle_deep_train", False) or (self.plain_deep_passes and self.inner.training))

    def _deep_out(self, x: Tensor, labels: Tensor, backward_scale: Tensor | None = None, **kw) -> dict:
        """Run the deep-supervised model: it calls this head (the chunked CE of lxtul/ce.py, then the pointer mixture)
        after EVERY spiral cycle (`spiral.cycle.credit: deep`) or plain pass (`plain.deep_supervision`,
        lxtul/plain_deep.py). With `backward_scale` the model backpropagates each cycle's weighted CE itself."""
        head = lambda h: self._mix(h, head_nll(self.inner, h, labels), labels)
        if self.plain_deep_passes:
            if kw:
                raise ValueError(f"plain deep supervision takes no model arguments, got {sorted(kw)}")
            out = deep_forward(self.inner, x, head, self.plain_deep_passes, backward_scale)
        else:
            out = self.inner(x, head_fn=head, backward_scale=backward_scale, **kw)
        self._h = None
        real = labels >= 0
        self.last_cycle_ce = [((a.detach() * on[:, None]).sum() / (real & on[:, None]).sum().clamp_min(1))
                              for a, on in zip(out["cycle_nll"], out["cycle_active"])]     # device scalars, logged as cycle/ce_c{c}
        return out

    def _deep_nll(self, x: Tensor, labels: Tensor, **kw) -> Tensor:
        """Deep supervision with ONE backward through the sum (every cycle's graph alive at once). A row's NLL is the
        mean over its own cycles, so the loss is the mean of the per-cycle CEs (all rows share a count when `depth=` is
        given). Used where no backward follows (token_nll) and as the reference of the per-cycle backward."""
        out = self._deep_out(x, labels, **kw)
        return sum(w[:, None] * a for w, a in zip(out["cycle_weight"], out["cycle_nll"]))

    def _deep_step(self, x: Tensor, labels: Tensor, **kw) -> dict:
        """Deep supervision with a backward after EVERY cycle or pass (HRM: memory of one cycle). The model has already
        accumulated this loss's gradient into the parameters when it returns; the loss is detached, so the trainer must
        not call backward (`backward_done`)."""
        n = (labels >= 0).sum().clamp_min(1)
        out = self._deep_out(x, labels, backward_scale=n, **kw)
        loss = sum((w[:, None] * a).sum() for w, a in zip(out["cycle_weight"], out["cycle_nll"])) / n
        return {"loss": loss, "backward_done": True}

    def forward(self, x: Tensor, labels: Tensor | None = None, **kw) -> dict:
        if labels is None:
            return self.inner(x, **kw)
        if self._deep_active() and torch.is_grad_enabled():
            if hasattr(self, "step"):
                self.inner.step = self.step
            return self._deep_step(x, labels, **kw)
        nll = self._nll(x, labels, **kw)
        n = (labels >= 0).sum().clamp_min(1)
        return {"loss": nll.sum() / n}

    @torch.no_grad()
    def token_nll(self, x: Tensor, labels: Tensor, **kw) -> Tensor:
        return self._nll(x, labels, **kw)
