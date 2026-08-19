"""DiffusionBlocks sampler — the reverse-diffusion decode loop.

Plan: ``docs/diffusionblocks-plan-of-action.md``. Reference: the authors'
``model.py::diffusion_step`` (``docs/diffusionblocks-reference-audit.md`` §3-§4).

How this differs from ordinary MORPH decoding. There is no next-token loop over positions
here. One call denoises the WHOLE target sequence, walking σ from ``σ_max`` down to
``σ_min``. Under ``mode="b3"`` each Euler step invokes only the block that owns its σ, so
the total layer evaluations for the walk equal about one ordinary forward — that is the
paper's inference-efficiency claim (App. H).

The Euler step's sign comes from ``next_sigma - sigma < 0``, so the σ list MUST descend.
:meth:`DBSchedule.inference_sigmas` guarantees that; do not sort it.

v1 scope, stated plainly: no KV cache (each Euler step is a full forward, same choice the
TUL generator made in its v1), and no classifier-free guidance (there is no class label to
drop for a language model — ``DBConfig`` raises if ``cfg_scale`` is set).
"""

from __future__ import annotations

import torch
from torch import Tensor

from morph.model.diffusion_blocks import DBStep, euler_step, expected_embedding

__all__ = ["db_sample", "SampleTrace"]


class SampleTrace:
    """Per-Euler-step diagnostics collected during a sample.

    ``denoised_norm`` exists because of risk R9: the bridge ``softmax(logits) @ E`` is a
    convex combination of embedding rows, so its norm collapses toward the table mean when
    the model is uncertain — while training only ever saw a full-scale target. A trajectory
    whose ``denoised_norm`` decays toward 0 is the model failing to commit, and it is
    invisible in the loss. Watch it.
    """

    def __init__(self) -> None:
        self.sigma: list[float] = []
        self.block: list[int] = []
        self.denoised_norm: list[float] = []
        self.z_norm: list[float] = []

    def add(self, sigma: float, block: int, denoised: Tensor, z: Tensor) -> None:
        self.sigma.append(float(sigma))
        self.block.append(int(block))
        self.denoised_norm.append(float(denoised.float().norm(dim=-1).mean()))
        self.z_norm.append(float(z.float().norm(dim=-1).mean()))

    def as_dict(self) -> dict:
        return {
            "db_sample/sigma": self.sigma,
            "db_sample/block": self.block,
            "db_sample/denoised_norm": self.denoised_norm,
            "db_sample/z_norm": self.z_norm,
            # A healthy walk ENDS more committed than it started.
            "db_sample/norm_ratio_end_start": (
                self.denoised_norm[-1] / self.denoised_norm[0]
                if self.denoised_norm and self.denoised_norm[0] > 0 else 0.0
            ),
        }


@torch.no_grad()
def db_sample(model, input_ids: Tensor, runtime, n_steps: int | None = None,
              generator: torch.Generator | None = None,
              trace: SampleTrace | None = None) -> tuple[Tensor, Tensor]:
    """Denoise a target sequence conditioned on ``input_ids``.

    Args:
        model:     a MORPH model with ``build_db_modules`` already called.
        input_ids: ``[B, L]`` the clean conditioning context.
        runtime:   the :class:`~morph.training.db_setup.DbRuntime`.
        n_steps:   Euler steps. ``None`` → ``1 + mean_depth + 1`` for b3 (one per block
                   application) or ``mean_depth`` for b1. Arm DB-13 sweeps this: the
                   denoiser conditions on σ rather than on Δσ, so step count is a free dial
                   at test time with NO retraining. Their AR text setting used only 4.
        trace:     optional :class:`SampleTrace` to fill (R9 monitoring).

    Returns:
        ``(logits, z)`` — final logits ``[B, L, vocab]`` and the final denoised embedding
        estimate ``[B, L, d_model]``.
    """
    sch = runtime.schedule
    pre = runtime.precond
    if n_steps is None:
        n_steps = (1 + int(model.cfg.mean_depth) + 1) if sch.n_blocks == 3 \
            else int(model.cfg.mean_depth)
    n_steps = max(int(n_steps), 2)

    sigmas = sch.inference_sigmas(n_steps).to(input_ids.device)
    if not bool((sigmas[:-1] > sigmas[1:]).all()):
        raise ValueError("inference_sigmas must be strictly descending (euler_step reads "
                         "its sign from next_sigma - sigma)")

    B, L = input_ids.shape
    d = int(model.cfg.d_model)
    # Their model.py:272-273: sqrt(1 + σ_max²), not σ_max alone.
    z = torch.randn(B, L, d, device=input_ids.device, dtype=torch.float32,
                    generator=generator) * float((1.0 + sigmas[0] ** 2).sqrt())

    lm_w = model.db_lm_weight()
    logits = None
    for i in range(n_steps - 1):
        s = sigmas[i].expand(B)
        ns = sigmas[i + 1].expand(B)
        block = int(sch.block_of_sigma(sigmas[i].view(1))[0])

        step = DBStep(block_idx=block, sigma=s, z_noisy=z.to(lm_w.dtype),
                      y_clean=z.to(lm_w.dtype), labels=input_ids)
        out = model(input_ids, db_step=step, db_precond=pre)
        logits = out["logits"]
        denoised = expected_embedding(logits, lm_w)

        if trace is not None:
            trace.add(float(sigmas[i]), block, denoised, z)
        z = euler_step(z.to(denoised.dtype), denoised, s, ns).float()

    # Final read at σ_min — the authors do this extra call too (model.py:288-290): the loop
    # above stops one short, so without it the lowest-noise block never gets to speak.
    s_min = sigmas[-1].expand(B)
    step = DBStep(block_idx=int(sch.block_of_sigma(sigmas[-1].view(1))[0]),
                  sigma=s_min, z_noisy=z.to(lm_w.dtype),
                  y_clean=z.to(lm_w.dtype), labels=input_ids)
    logits = model(input_ids, db_step=step, db_precond=pre)["logits"]
    return logits, z
