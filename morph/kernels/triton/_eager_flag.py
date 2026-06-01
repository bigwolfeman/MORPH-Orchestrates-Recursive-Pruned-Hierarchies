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
