"""Dropout whose mask is shared by the K rollouts of a rollout-major expanded batch.

LXTUL-GK (``tul.gram_objective="iw"``, K > 1) runs the slot loop and the coda on K
rollout-major copies of each row (``transformer.repeat_along_batch``), and the
multi-sample bound credits whichever rollout explains a span best. The rollouts are meant
to differ in their Gaussian steps ALONE. A plain ``nn.Dropout`` on the expanded batch
draws an independent mask per row, so the K copies of a row would also differ in their
dropout masks: the bound could then earn width from dropout, a randomness that does not
exist at eval. Found 2026-09-23 after the first lxtul-gk4 run (its training width gain
rose to 0.0065 nats while its prior sigma fell to 3e-4 of the state RMS).

:class:`RolloutSharedDropout` draws ONE mask for the base rows (the first ``B / n_rep``)
and tiles it rollout-major. It is swapped in only on a GK model with K > 1, and only in
the modules that run on the expanded batch (the core and the coda); every other model
keeps its ``nn.Dropout`` modules and their RNG stream untouched. Eval is dropout-free
either way.
"""
from __future__ import annotations

import contextlib

import torch
from torch import Tensor, nn


class RolloutSharedDropout(nn.Dropout):
    """``nn.Dropout(p)`` whose training mask is drawn for ``x.shape[0] // n_rep`` rows and
    repeated ``n_rep`` times along dim 0 (the ``repeat_along_batch`` order).

    ``n_rep`` is fixed at construction: under checkpoint recompute the backward replays the
    same ops with the preserved RNG state, so the decision must not depend on anything
    that changes between the forward and its recompute. See :func:`bypass_rollout_sharing`
    for the one legal exception (a ``torch.no_grad()`` call, which has no recompute)."""

    def __init__(self, p: float, n_rep: int):
        super().__init__(p)
        if n_rep < 2:
            raise ValueError(f"RolloutSharedDropout needs n_rep >= 2, got {n_rep}")
        self.n_rep = int(n_rep)
        self._bypass = False   # see `bypass_rollout_sharing`

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        if self._bypass:
            # `bypass_rollout_sharing` is active: this call's batch does not hold R
            # rollout-major copies of the same rows (e.g. `_tul_fan_all`'s "map" winner
            # mode picking passes, which run on ONE row per sample after each slot has
            # already picked its rollout) — ordinary independent-per-row dropout is the
            # correct behavior here, not a relaxed version of the shared-mask rule.
            return nn.functional.dropout(x, self.p, training=True)
        rows = x.shape[0]
        if rows % self.n_rep:
            raise RuntimeError(
                f"RolloutSharedDropout: batch {rows} is not a multiple of n_rep "
                f"{self.n_rep}; a GK model's core and coda only run on the expanded batch "
                f"in training.")
        keep = 1.0 - self.p
        base = (rows // self.n_rep,) + tuple(x.shape[1:])
        mask = torch.empty(base, device=x.device, dtype=x.dtype).bernoulli_(keep).div_(keep)
        # Broadcast over the rollout axis instead of materialising the K-fold repeat, so
        # autograd saves the base-row mask only.
        return (x.reshape(self.n_rep, *base) * mask.unsqueeze(0)).reshape(x.shape)

    def extra_repr(self) -> str:
        return f"p={self.p}, n_rep={self.n_rep}"


@contextlib.contextmanager
def bypass_rollout_sharing(module: nn.Module):
    """Temporarily make every :class:`RolloutSharedDropout` under ``module`` behave like
    an ordinary ``nn.Dropout`` (independent per-row mask, no ``n_rep``-multiple
    requirement), then restore it.

    Legal ONLY around a ``torch.no_grad()`` forward. ``RolloutSharedDropout.__doc__``
    ties ``n_rep`` to the checkpoint-recompute contract: the mask decision must not
    depend on anything that changes between a forward and its backward recompute. A
    no-grad call has no recompute, so toggling this flag around one cannot violate that
    contract — there is nothing to replay. Callers outside a no-grad block must not use
    this; nothing here enforces it, so get it right at the call site (2026-09-29,
    ``.agents/notes/proposed/architecture/2026-09-29-onewinner-shared-map-rollout.md``:
    ``_tul_fan_all``'s "map" winner mode runs its M no-grad picking passes on a B0-row
    batch — ONE row per sample, each already carrying its slot's own MAP-picked
    rollout's cells, not an R-fold rollout-major batch — so there is no rollout
    comparison left within a row for a shared mask to protect; independent per-row
    dropout is not a relaxation here, it is what the batch actually calls for)."""
    mods = [m for m in module.modules() if isinstance(m, RolloutSharedDropout)]
    for m in mods:
        m._bypass = True
    try:
        yield
    finally:
        for m in mods:
            m._bypass = False


def share_dropout_across_rollouts(module: nn.Module, n_rep: int) -> int:
    """Replace every ``nn.Dropout`` (p > 0) under ``module`` with a
    :class:`RolloutSharedDropout` of the same ``p``. Returns the number replaced.
    Dropout has no parameters or buffers, so the state dict is unchanged."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child) is nn.Dropout and child.p > 0.0:
            setattr(module, name, RolloutSharedDropout(child.p, n_rep))
            n += 1
        else:
            n += share_dropout_across_rollouts(child, n_rep)
    return n
