"""``model.pdl``: programmatic dependent launch (PDL) for the step's custom Triton kernels.

WHAT. With PDL a kernel may start launching while the kernel before it in the stream is
still finishing, so its launch latency overlaps the predecessor's tail. Measured on this
card in a graph-replayed chain (plan note ``2026-10-12-core-pass-persistent-kernel-plan.md``,
Q2): 0.19 to 0.24 us per boundary between two Triton kernels, nothing at a cuBLAS boundary.

HOW. Two halves, and a kernel needs both:

* the launch passes ``launch_pdl=True`` (``pdl_opts()`` returns it, plus ``PDL=True``);
* the kernel body starts with ``if PDL: tl.extra.cuda.gdc_wait()``, so no load or store runs
  before the predecessor has finished and flushed its writes.

A kernel launched with ``launch_pdl`` and no ``gdc_wait`` races with its predecessor. Every
step-path kernel therefore takes ``PDL: tl.constexpr = False``, waits as its FIRST statement,
and takes ``**pdl_opts()`` at every launch; ``tests/test_pdl.py`` checks all three. There is
no early trigger (``gdc_launch_dependents``): a dependent launched early occupies SM slots
while it waits, which the plan note measured as a loss at 50 MB tensors. A dependent may
launch once every block of its predecessor has exited, which is when the predecessor's
implicit trigger fires.

Inductor's own kernels take PDL from ``torch._inductor.config.triton.enable_pdl`` (it
inserts ``gdc_wait`` where a kernel reads what the previous node wrote); ``configure`` sets
that flag with the key. Off, nothing changes: ``pdl_opts()`` is empty, ``PDL`` is False and
the kernels compile as before.
"""
from __future__ import annotations

import torch

_ON = False
_PRIOR_INDUCTOR: bool | None = None       # inductor's flag before ``configure(True)`` set it
_OPTS = {"launch_pdl": True, "PDL": True}


def configure(on: bool) -> None:
    """Set the switch ``pdl_opts`` reads (``MORPHTransformer.__init__`` calls it with
    ``model.pdl``) and, when on, inductor's flag. Refuses a device without PDL (sm_90+)."""
    global _ON, _PRIOR_INDUCTOR
    import torch._inductor.config as inductor_config
    if on:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
            raise RuntimeError("model.pdl needs a CUDA device of compute capability 9.0 or "
                               "newer (griddepcontrol)")
        if _PRIOR_INDUCTOR is None:
            _PRIOR_INDUCTOR = bool(inductor_config.triton.enable_pdl)
        inductor_config.triton.enable_pdl = True
    elif _PRIOR_INDUCTOR is not None:
        inductor_config.triton.enable_pdl = _PRIOR_INDUCTOR
        _PRIOR_INDUCTOR = None
    _ON = bool(on)


def pdl_opts() -> dict:
    """Launch keywords for a custom Triton kernel: ``{}`` off; ``launch_pdl`` and ``PDL`` on."""
    return _OPTS if _ON else {}
