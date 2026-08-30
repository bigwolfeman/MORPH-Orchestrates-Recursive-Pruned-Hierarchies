"""Core-stage conditioning for MORPH's TUL slot loop — the faithful DiffusionBlocks
recurrent-depth recipe (arXiv 2506.14202, Appendix B "recurrent-depth architectures",
§3.3 equi-probability partitioning, Appendix C EDM preconditioning).

Mission: build the machinery ``tul.db_loop`` never had — a σ/iteration conditioning
signal reaching each core-layer application, so a SINGLE conditioned core pass can
stand in for the whole T-iteration loop at training time (the paper's Huginn recipe),
while the T-iteration loop remains available (and, with ``core_stage_cond="iter"``,
gains a depth signal it never had either).

Two pieces live here:

* :class:`CoreStageConditioning` — ONE conditioning module serving BOTH stage kinds
  (``"iter"``: which loop iteration; ``"sigma"``: EDM noise level) through the SAME
  sinusoidal-embed → MLP → per-layer AdaLN-Zero path, reusing
  :class:`morph.model.diffusion_blocks.SigmaConditioning` and
  :class:`morph.model.diffusion_blocks.AdaLNGate` verbatim rather than re-deriving a
  second copy of "sinusoidal embed a scalar, zero-init an AdaLN gate" — that machinery
  already exists for MORPH's whole-model (prelude|core|coda) DiffusionBlocks arm and
  the zero-init-at-construction discipline is identical here.
* :class:`DB1Sampler` — the equal-mass log-normal σ sampler and EDM preconditioning
  coefficients, scoped to TUL's OWN configurable ``(σ_min, σ_max, P_mean, P_std,
  σ_data)`` rather than reusing ``diffusion_blocks.DBSchedule``/``DBConfig``, which
  hardcode σ_min/σ_max as MODULE-LEVEL GLOBALS shared with the unrelated whole-model
  DB arm — changing them here would silently perturb that arm's schedule too.
  :class:`DB1Sampler` reuses :class:`morph.model.diffusion_blocks.EDMPrecond` for the
  coefficients (identical formulas, no reason to re-derive) and
  :func:`morph.model.diffusion_blocks.euler_step` for the inference ladder.

Nothing here branches on a runtime flag: :class:`CoreStageConditioning` and
:class:`DB1Sampler` are built ONLY when ``TULConfig.core_stage_cond != "none"``
(construction time, ``MORPHTransformer.__init__``), and every call site in
``transformer.py`` that can receive ``stage_cond=None`` is bit-identical to the
pre-existing forward when it does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .diffusion_blocks import AdaLNGate, EDMPrecond, SigmaConditioning

__all__ = ["CoreStageConditioning", "DB1Sampler", "iter_stage_value"]


class CoreStageConditioning(nn.Module):
    """σ- or iteration-conditioned AdaLN-Zero modulation, ONE gate per core layer.

    ``forward``-time contract: :meth:`stage_embed` turns a ``[B]`` (or ``[1]``,
    broadcasting to every sample) scalar "stage" value into a ``[B, cond_dim]``
    embedding; :meth:`modulate` applies core layer ``layer_idx``'s zero-init AdaLN
    gate to a carrier tensor ``[B, S, n, C]`` (or ``[B, S, C]``) using that embedding.

    Zero-init bit-identity (proved by
    ``tests/test_tul_dbfix.py::test_conditioning_zero_init_bit_identical``):
    ``AdaLNGate.to_mod`` is zero-initialised (weight AND bias), so
    ``modulate(x, cond, i) == x`` EXACTLY at construction, for ANY ``cond`` — the
    conditioned forward only starts doing anything once training moves the gate's
    weights off zero. This is the same "starts as a no-op" discipline as the HC
    Cayley residual init and ``ChannelInject``'s ``log_scale=0``.
    """

    def __init__(self, n_layers: int, d_model: int, cond_dim: int = 256):
        super().__init__()
        if n_layers < 1:
            raise ValueError(f"CoreStageConditioning needs n_layers >= 1, got {n_layers}")
        self.n_layers = n_layers
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.embed = SigmaConditioning(cond_dim=cond_dim)
        self.gates = nn.ModuleList([AdaLNGate(cond_dim, d_model) for _ in range(n_layers)])

    def stage_embed(self, stage: Tensor) -> Tensor:
        """``[B]`` (or ``[1]``) scalar stage value → ``[B, cond_dim]`` embedding.

        ``stage`` is EITHER ``c_noise = 0.25·log σ`` (the "sigma" mode, matching the
        EDM/authors' own timestep convention audited into
        :class:`morph.model.diffusion_blocks.EDMPrecond`) OR the raw iteration index
        ``t`` (the "iter" mode — ``t`` needs no EDM rescaling; it is already a small
        bounded integer and the sinusoidal basis in ``SigmaConditioning`` covers any
        bounded scalar equally well). Reusing ONE embedding path for both means a
        later switch between the two stage kinds is a config change, not a rewrite.
        """
        return self.embed(stage)

    def modulate(self, x: Tensor, cond: Tensor, layer_idx: int) -> Tensor:
        if not 0 <= layer_idx < self.n_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range for {self.n_layers} gates")
        return self.gates[layer_idx](x, cond)


def iter_stage_value(t: int, device) -> Tensor:
    """``[1]`` tensor carrying the iteration index — "iter" mode's stage value.

    A ``[1]`` tensor (not a Python float) so it broadcasts against a ``[B, cond_dim]``
    embedding output regardless of batch size, and so it never becomes a Python-level
    compile guard the way a bare float would (``SigmaConditioning``'s docstring flags
    exactly this risk for σ; the same applies to ``t``).
    """
    return torch.tensor([float(t)], device=device)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError(f"_norm_ppf needs q in (0,1), got {q}")
    return float(
        math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0, dtype=torch.float64)).item()
    )


@dataclass
class DB1Sampler:
    """Equal-mass log-normal σ sampler + EDM preconditioning for the TUL one-pass step.

    Mirrors ``diffusion_blocks.DBSchedule``'s ``n_blocks=1`` case (App. B: recurrent-
    depth models need no block partitioning) with TUL-LOCAL bounds — see module
    docstring for why this is not a re-use of ``DBSchedule`` directly.

    ``log σ ~ N(p_mean, p_std²)`` truncated to ``[sigma_min, sigma_max]`` and sampled
    uniformly in CDF space (§3.3), which is exactly "equal probability mass" sampling
    for ``B=1`` — there is only one block, so its σ range IS the whole schedule.
    """

    sigma_min: float
    sigma_max: float
    p_mean: float
    p_std: float
    sigma_data: float

    def __post_init__(self) -> None:
        if not 0.0 < self.sigma_min < self.sigma_max:
            raise ValueError(
                f"DB1Sampler needs 0 < sigma_min < sigma_max, got "
                f"{self.sigma_min}, {self.sigma_max}"
            )
        if self.p_std <= 0.0:
            raise ValueError(f"DB1Sampler needs p_std > 0, got {self.p_std}")
        if self.sigma_data <= 0.0:
            raise ValueError(f"DB1Sampler needs sigma_data > 0, got {self.sigma_data}")
        self.precond = EDMPrecond(sigma_data=self.sigma_data)
        self._cdf_min = _norm_cdf((math.log(self.sigma_min) - self.p_mean) / self.p_std)
        self._cdf_max = _norm_cdf((math.log(self.sigma_max) - self.p_mean) / self.p_std)

    def sample(self, n: int, device, generator: torch.Generator | None = None) -> Tensor:
        """``[n]`` σ drawn from the truncated log-normal, fp32."""
        u = torch.rand(n, dtype=torch.float64, device=device, generator=generator)
        q = self._cdf_min + (self._cdf_max - self._cdf_min) * u
        z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        return (self.p_mean + self.p_std * z).exp().float()

    def ladder(self, n_steps: int) -> Tensor:
        """``[n_steps]`` DESCENDING σ, equi-probability spaced, σ_max → σ_min.

        Descending is load-bearing: :func:`morph.model.diffusion_blocks.euler_step`
        reads its sign from ``next_sigma - sigma < 0``.
        """
        if n_steps < 1:
            raise ValueError(f"DB1Sampler.ladder needs n_steps >= 1, got {n_steps}")
        if n_steps == 1:
            return torch.tensor([self.sigma_max], dtype=torch.float32)
        qs = [
            self._cdf_min + (self._cdf_max - self._cdf_min) * (i / (n_steps - 1))
            for i in range(n_steps)
        ]
        s = [
            math.exp(self.p_mean + self.p_std * _norm_ppf(min(max(q, 1e-12), 1 - 1e-12)))
            for q in qs
        ]
        return torch.tensor(list(reversed(s)), dtype=torch.float32)

    def manifest(self) -> dict:
        return {
            "tul/db1_sigma_min": self.sigma_min,
            "tul/db1_sigma_max": self.sigma_max,
            "tul/db1_p_mean": self.p_mean,
            "tul/db1_p_std": self.p_std,
            "tul/db1_sigma_data": self.sigma_data,
        }
