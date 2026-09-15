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
