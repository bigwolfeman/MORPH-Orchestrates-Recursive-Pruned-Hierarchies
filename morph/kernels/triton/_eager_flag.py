"""Process-global 'force eager' switch for the fused Triton kernels.

Lets the fused CCA/HCA/CSA/conv entry points fall back to their pure-PyTorch
reference implementations on demand — without changing call sites — so we can
run a kernel-OFF vs kernel-ON A/B in the SAME architecture (same weights, same
math up to bf16 rounding) and measure the real memory / throughput delta.

Dependency-free (imported by the fused kernels → must not import them back).
Initialised from the MORPH_FORCE_EAGER env var; also settable from the model
config (MORPHConfig.use_kernels) at build time so it is captured in the wandb run.
"""
import os

_FORCE_EAGER: bool = os.environ.get("MORPH_FORCE_EAGER", "0") == "1"


def set_force_eager(value: bool) -> None:
    global _FORCE_EAGER
    _FORCE_EAGER = bool(value)


def force_eager() -> bool:
    return _FORCE_EAGER


# Dedicated kill-switch for ONLY the fused Hyper-Connection kernels (hc_pre / hc_post /
# hc_pre_map), independent of the global force_eager that flips ALL fused kernels. Lets us run
# an isolated HC-fused vs HC-eager A/B with every OTHER kernel (CCA/HCA/CSA/conv) held ON and
# bit-identical in both arms, so any loss-curve delta is attributable to the HC kernel alone.
# Init from MORPH_HC_FORCE_EAGER env; also settable at runtime for an in-process A/B.
_HC_FORCE_EAGER: bool = os.environ.get("MORPH_HC_FORCE_EAGER", "0") == "1"


def set_hc_force_eager(value: bool) -> None:
    global _HC_FORCE_EAGER
    _HC_FORCE_EAGER = bool(value)


def hc_force_eager() -> bool:
    return _HC_FORCE_EAGER


# ── Dynamo fence for the fused-kernel dispatchers ────────────────────────────
# Historically every fused dispatcher was hard-wrapped in @torch.compiler.disable
# ("kernel is opaque", "tl.dot gets fp64 if traced"). Under torch.compile those
# fences force a graph break AT EVERY KERNEL CALL SITE, so inductor can never
# fuse the eager glue between kernels — measured 2026-07-03 (Olympiad seed,
# d512 / B32 / S64): kernels+compile == kernels+eager (21 ms/step) because every
# frame fell back to eager, while fence-free reference+compile hit 14.9 ms/step.
# Modern dynamo (torch >= 2.6, verified on 2.11) traces user Triton kernels and
# autograd.Functions natively, making the fences obsolete THERE — but they stay
# the default until parity is proven per-installation. Opt out explicitly:
#   MORPH_DYNAMO_FENCE=1  (default) keep torch.compiler.disable on dispatchers
#   MORPH_DYNAMO_FENCE=0  no fence: kernels stay INSIDE the compiled graph
# Read once at import (same contract as the other flags in this module).
_DYNAMO_FENCE: bool = os.environ.get("MORPH_DYNAMO_FENCE", "1") == "1"


def kernel_fence(fn):
    """@kernel_fence replaces @torch.compiler.disable on kernel dispatchers."""
    if _DYNAMO_FENCE:
        import torch
        return torch.compiler.disable(fn)
    return fn
