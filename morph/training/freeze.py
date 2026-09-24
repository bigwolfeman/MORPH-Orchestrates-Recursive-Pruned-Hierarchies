"""`training.train_only`: freeze every parameter whose name does not start with one of the
listed prefixes. The optimizer (`optimizer.py`) already skips parameters without a gradient
requirement, so the freeze is complete once `requires_grad` is set before the optimizer is
built. Frozen modules whose inputs carry no gradient keep no activations for backward, so
memory falls with the frozen share.

The thinker-only regime of TUL-Code (2026-09-15, Wolfe): the encoder and the coda are
frozen at a trained checkpoint and only the thinker — the core body (the velocity field),
its injection terms, the velocity head, the time embedding and the cell markers — trains,
on the flow loss alone, for far more tokens than the joint run gave it.
"""
from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn


def apply_train_only(model: nn.Module, prefixes: list[str] | tuple[str, ...]
                     ) -> tuple[int, int, "OrderedDict[str, int]"]:
    """Set `requires_grad` from the prefix list. Returns (n_trainable, n_frozen, groups):
    `groups` maps each trainable top-level parameter group (name up to the second dot) to
    its parameter count. Raises when the list matches nothing, or matches everything."""
    prefixes = tuple(str(p) for p in prefixes)
    if not prefixes:
        raise ValueError("apply_train_only: an empty prefix list would freeze the whole model")
    n_train = n_frozen = 0
    groups: "OrderedDict[str, int]" = OrderedDict()
    for name, p in model.named_parameters():
        keep = name.startswith(prefixes)
        p.requires_grad_(keep)
        if keep:
            n_train += p.numel()
            top = ".".join(name.split(".")[:2])
            groups[top] = groups.get(top, 0) + p.numel()
        else:
            n_frozen += p.numel()
    if n_train == 0:
        raise ValueError(f"training.train_only={list(prefixes)} matched no parameter")
    if n_frozen == 0:
        raise ValueError(f"training.train_only={list(prefixes)} matched EVERY parameter; "
                         f"drop the key instead")
    return n_train, n_frozen, groups


def apply_frozen_eval(model: nn.Module, prefixes: list[str] | tuple[str, ...]) -> list[str]:
    """`training.frozen_eval`: the FROZEN model runs in eval mode, the trained modules in
    train mode. Call it wherever the trainer would call `model.train()`.

    Why. `training.train_only` freezes parameters, not behaviour: a frozen model left in
    train mode still draws its dropout masks, its token-state dropout and its Poisson slot
    depths, so the state a new head is fitted on is a noisy training-mode state and not
    the deterministic one the eval pass (and every offline scorer) reads. LXTUL-E Stage 0
    fits a head on the FROZEN ruler's exit cell, so the cell must be the eval cell.

    What it does. `model.eval()` on the whole tree (the ROOT's `training` flag is what the
    forward's `self.training` branches read: the depth draw, the token-state dropout, the
    training-only loop terms), then `.train()` on every module whose qualified name + "."
    starts with one of `prefixes`. Every prefix must end in "." (it names a whole
    module) and must match at least one module; a prefix that names a single tensor of a
    module that also holds frozen tensors (`tul.E_slot`) RAISES, because that module
    cannot be in both modes. Returns the names of the modules left in train mode."""
    prefixes = tuple(str(p) for p in prefixes)
    if not prefixes:
        raise ValueError("apply_frozen_eval: an empty prefix list leaves nothing to train")
    bad = [p for p in prefixes if not p.endswith(".")]
    if bad:
        raise ValueError(
            f"training.frozen_eval needs train_only prefixes that name WHOLE modules (end "
            f"in '.'), got {bad}: a module holding both trained and frozen tensors cannot "
            f"be in train and eval mode at once.")
    root = getattr(model, "_orig_mod", model)
    model.eval()
    root.eval()
    trained: list[str] = []
    hit = {p: False for p in prefixes}
    for name, mod in root.named_modules():
        if not name:
            continue
        key = name + "."
        for p in prefixes:
            if key.startswith(p):
                mod.train()
                trained.append(name)
                hit[p] = True
                break
    missing = [p for p, h in hit.items() if not h]
    if missing:
        raise ValueError(f"training.frozen_eval: train_only prefix(es) {missing} match no "
                         f"module")
    return trained
