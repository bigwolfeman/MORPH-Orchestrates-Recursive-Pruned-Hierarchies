"""Host shadows: the CPU copy of a small device tensor whose value the host already knows.

WHY (2026-10-04 training-speed pass). A boolean-mask index ``x[mask]``, a ``bool(m.any())``
and an ``int(m.sum().max())`` on a CUDA tensor each stop the host until the GPU has run
everything queued before them. The slot loop did about 190 of these per training step,
most of them on masks the host had in hand before the batch ever reached the GPU (the
layout's ``slot_valid`` and ``slot_mask`` come from the CPU packer) or could have in hand
after the ONE sync it cannot avoid (the per-slot depth table, drawn on the GPU, whose max
sets the Python loop count). Every such sync drains the queue, so the host cannot launch
the next region while the GPU still runs the last one.

WHAT. :func:`attach` records a numpy copy of a device tensor's value on the tensor object
itself, keyed by the tensor's ``_version`` (an in-place write invalidates it).
:func:`shadow` returns it, or ``None``. Derived tensors (``tensor[perm]``, ``.repeat``,
``torch.where``...) are NEW objects without a shadow, so a shadow can never describe a
different value than its tensor: a reader that finds none takes the old synchronising
path. That fallback is the contract, not a degradation: every reader below must give the
same result with or without a shadow, which is what ``tests/test_host_shadow.py`` pins.

The readers are exact replacements:

* :func:`true_index` — the flat indices of a mask's True entries in ascending order, the
  same tensor as ``mask.flatten().nonzero().squeeze(1)``. With a shadow the count is known
  on the host and a stable argsort finds the positions on the device without a sync.
* :func:`masked_rows` — ``x[mask]`` for a mask over ``x``'s leading dims, as
  ``x.flatten(...)[true_index(mask)]``. Forward values are identical, and the backward is
  the same ``index_put_(accumulate=True)`` with the same (unique) indices, so the gradient
  is bit-identical too.
* :func:`any_true` / :func:`count_true` — ``bool(mask.any())`` / ``int(mask.sum())`` from
  the shadow when there is one.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

_ATTR = "_morph_host_shadow"


def attach(t: Tensor, host: np.ndarray) -> Tensor:
    """Record ``host`` (the exact value of ``t``) on ``t`` and return ``t``."""
    host = np.asarray(host)
    if tuple(host.shape) != tuple(t.shape):
        raise ValueError(f"host shadow shape {host.shape} != tensor shape {tuple(t.shape)}")
    setattr(t, _ATTR, (t._version, host))
    return t


def shadow(t: Tensor | None) -> np.ndarray | None:
    """The host copy recorded by :func:`attach`, or ``None`` (none recorded, or the
    tensor was written in place since)."""
    if t is None:
        return None
    rec = getattr(t, _ATTR, None)
    if rec is None or rec[0] != t._version:
        return None
    return rec[1]


def to_device(t: Tensor, device, non_blocking: bool = True) -> Tensor:
    """``t.to(device)`` with the source's value recorded as the result's shadow when the
    source is a CPU tensor (the value the host already holds)."""
    out = t.to(device, non_blocking=non_blocking)
    if t.device.type == "cpu":
        # A CPU target returns `t` itself; recording its own value on it is harmless and
        # lets a CPU run take the same reader paths a CUDA run takes.
        attach(out, t.numpy().copy())
    elif out is not t:
        h = shadow(t)
        if h is not None:
            attach(out, h)
    return out


def derive(out: Tensor, host: np.ndarray | None) -> Tensor:
    """Attach ``host`` to ``out`` when it is known (``None`` leaves ``out`` bare)."""
    if host is not None:
        attach(out, host)
    return out


def any_true(mask: Tensor) -> bool:
    """``bool(mask.any())``, from the shadow when there is one."""
    h = shadow(mask)
    return bool(h.any()) if h is not None else bool(mask.any())


def count_true(mask: Tensor) -> int:
    """``int(mask.sum())``, from the shadow when there is one."""
    h = shadow(mask)
    return int(h.sum()) if h is not None else int(mask.sum())


def true_index(mask: Tensor) -> Tensor:
    """Flat indices of ``mask``'s True entries, ascending: ``mask.flatten().nonzero()``.

    With a shadow: the count is the host's, and a STABLE argsort of ``~mask`` puts the True
    positions first in their original order, so the first ``n`` entries are exactly the
    nonzero indices, found without a host sync."""
    h = shadow(mask)
    flat = mask.reshape(-1)
    if h is None:
        return flat.nonzero().squeeze(1)
    n = int(h.sum())
    return torch.argsort((~flat).to(torch.uint8), stable=True)[:n]


def masked_rows(x: Tensor, mask: Tensor) -> Tensor:
    """``x[mask]`` for a boolean ``mask`` over ``x``'s leading ``mask.dim()`` dims."""
    k = mask.dim()
    if tuple(x.shape[:k]) != tuple(mask.shape):
        raise ValueError(f"masked_rows: mask {tuple(mask.shape)} does not lead x "
                         f"{tuple(x.shape)}")
    if shadow(mask) is None:
        return x[mask]
    return x.reshape(-1, *x.shape[k:])[true_index(mask)]
