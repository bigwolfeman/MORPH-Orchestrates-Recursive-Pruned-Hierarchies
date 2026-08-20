"""DiffusionBlocks for MORPH — σ schedule, EDM preconditioning, slice-wise target scaling.

Paper: DiffusionBlocks (Shing, Koyama, Akiba; ICLR 2026, arXiv:2506.14202).
Design: ``docs/diffusionblocks-morph-assessment.md``.
Arms + pre-registered numbers: ``docs/diffusionblocks-experiment-sheet.md``.
Plan: ``docs/diffusionblocks-plan-of-action.md``.
Audit of the authors' released code: ``docs/diffusionblocks-reference-audit.md``.

Mental model in one paragraph. A residual update ``z ← z + f(z)`` is an Euler step of a
reverse-diffusion ODE. If you noise the TARGET (``z_σ = y + σ·ε``) then every point on
that trajectory has ``y`` locally available, so a block can be trained WITHOUT the other
blocks and without backprop through them. The noise is what buys the removal of BPTT; σ
is a progress coordinate that tells the block how far along it is; and the inter-block
seam ``z_b = α·z_{b-1} + (1−α)·D`` with ``α = σ_b/σ_{b-1} ∈ (0,1)`` is a PRESCRIBED
contraction, which is the ``ρ(J_core) ≤ 1`` handle the nested-dynamics note asks for.

Everything here is construction-time or pure tensor math. Nothing branches on a runtime
flag in a hot path: :class:`DBConfig` is resolved once, and ``db_step=None`` never reaches
this module (the baseline forward is untouched and bit-identical).

CRITICAL sign note (audit §2). The paper's rendered Eq (3)–(5) carry a sign that makes
noise INCREASE down the schedule. The authors' own code (``model.py:283-287``) computes
``dt = next_sigma - sigma`` over a DESCENDING σ buffer, so ``dt < 0`` and the step is the
EDM form. :func:`euler_step` implements the EDM form. Do not "fix" it back to the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "DBConfig",
    "clean_noisy_mask",
    "DBSchedule",
    "DBStep",
    "EDMPrecond",
    "SliceScaler",
    "SigmaConditioning",
    "euler_step",
    "expected_embedding",
]

# EDM / DiffusionBlocks constants, read off the authors' code (audit §3). These are NOT
# "defaults to taste": the equi-probability boundaries are computed FROM them, so a
# different value is a different experiment. dblock_modules.py::get_block_sigmas and
# model.py:114.
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
P_MEAN = -1.2
P_STD = 1.2
SIGMA_DATA = 0.5

# Analytic normal CDF / inverse CDF so the schedule needs no scipy at import time
# (the authors use scipy.stats.norm; torch gives us the same to fp64 roundoff and keeps
# the module importable in a bare env).


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF via torch's erfinv (fp64)."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"_norm_ppf needs q in (0,1), got {q}")
    return float(
        math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0, dtype=torch.float64)).item()
    )


@dataclass
class DBConfig:
    """Construction-time DiffusionBlocks settings. Mirrors the Hydra ``db:`` block.

    Only keys that change PARAMETERS or SHAPES live here. The schedule key
    (``activate_at``) belongs to the training loop and never reaches the model.

    ``mode``:
      * ``"b1"`` — recurrent-depth mode (paper §5.5 / App. E.5, the Huginn setting).
        The whole prelude+core+coda net is ONE denoiser, ``B = 1``. Training is a single
        forward at a sampled σ; inference keeps the ``1 + T + 1`` Euler steps with T
        Poisson. **Run this arm first** (plan D2): it is the setting the paper validated
        on a looped model and it carries none of the per-block schedule-stretch hazards.
      * ``"b3"`` — block-wise over MORPH's own seams, prelude | core | coda. The core
        stays weight-tied and no layers are added.

    ``conditioning`` (plan O2 — Wolfe: build and test BOTH, assume neither):
      * ``"concat"`` — the paper's App. E.4 causal consistency: the clean and noisy
        sequences are concatenated under a modified causal mask so noisy targets attend
        CLEAN past. Doubles positions. See :func:`clean_noisy_mask` for the exact cutoff.
      * ``"x0_inject"`` — the cheaper alternative: ONE sequence, with the clean token
        embedding injected additively through the existing ctx-slice ``ChannelInject``,
        which MORPH already applies at every layer.

    NO SHIFT IS NEEDED, and this is worth stating because getting it wrong would silently
    leak the answer and produce a beautiful fake loss curve. MORPH's loader yields
    ``labels[t] = input_ids[t+1]`` (``training/data.py``), so the denoising target at
    position ``t`` is ``embed(labels[t]) = embed(input_ids[t+1])`` — the NEXT token. The
    conditioning that MORPH already has at position ``t`` is ``x0[t] = embed(input_ids[t])``,
    the CURRENT token. Those are different tokens, so unshifted x0 is already causally
    correct. An earlier draft of the plan called for shifting x0 by one position; that was
    a fix for a leak that does not exist, and shifting would actually DESTROY one position
    of legitimate context.
    """

    mode: str = "b3"
    conditioning: str = "concat"

    # ── σ partition ──────────────────────────────────────────────────────────
    # Mass per block in LAYER order (prelude → coda). None → equal mass 1/B, which is
    # the paper's literal rule (§3.3). For MORPH b3 the default is 1/8 : 6/8 : 1/8 so the
    # `1 + T̄ + 1 = 8` inference Euler steps land evenly spaced in probability mass at
    # T̄ = 6 — the core owns T̄ of the 8 steps because it is APPLIED T̄ times.
    block_mass: tuple[float, ...] | None = None
    # Block visit distribution during training. "uniform" is the authors' own rule
    # (App. E.1: "blocks are sampled uniformly at random for each iteration");
    # "mass" visits in proportion to block_mass. Arm DB-12 sweeps this. Uniform is the
    # default because mass-proportional visits would give the prelude and coda 1/8 of
    # steps each — 2500 effective updates over a 20k run against 20k today.
    visit: str = "uniform"
    overlap_gamma: float = 0.1        # γ, the paper's text-generation setting (App. C)

    # ── target scaling (plan O1, audit §6) ───────────────────────────────────
    # Scale the euclidean and Lorentz-tangent slices INDEPENDENTLY, each to per-component
    # std σ_data. The authors' whole-vector L2 does NOT transfer: HybridEmbedding
    # CONCATENATES a 768-dim euclidean slice (init ≈ unit) with a 256-dim Lorentz-tangent
    # slice (init std 0.005), and normalising the concatenation leaves that ~200× ratio
    # alone, so isotropic σ·ε buries the Lorentz slice at every useful σ.
    slice_scale: bool = True
    sigma_data: float = SIGMA_DATA

    # ── objective (the fork, checklist top section, 2026-08-19) ───────────────
    # "l2" (default, Option A — the paper's AR objective, App. B/C): the loss is
    # w(σ)·‖D̂ − y‖² in EMBEDDING space. No readout is used in training, so the tied-head
    # free ride (c_skip·z ≈ y at low σ → logits decode the target for free) cannot occur
    # BY CONSTRUCTION. Under "l2" the forward skips the `denoised @ E.T` matmul entirely.
    # "ce" (Option B — a MORPH extension, NOT the paper's AR method): CE on the tied
    # readout, today's behavior, kept for the db_b1_concat ablation lineage.
    loss_kind: str = "l2"

    # ── loss weighting (falsifier finding, 2026-08-19) ────────────────────────
    # "unweighted" (default): plain mean CE. "edm": the authors' w(σ)=1/c_out² L2
    # weighting. EDM's w equalizes σ contributions for an L2 REGRESSION loss; applied to
    # CROSS-ENTROPY it over-weights the trivial low-σ region (c_skip≈1 → z passes through →
    # CE≈0) by up to ~62,000×, so training collapses to the low-σ copy and never learns
    # high-σ prediction (db-b1-concat run: train loss→0, val≈ln V). Unweighted CE lets the
    # low-σ samples contribute ~0 gradient (nothing to learn there) while high-σ drives
    # learning. Ablate with loss_weighting=edm to reproduce the collapse.
    loss_weighting: str = "unweighted"

    # ── σ conditioning ───────────────────────────────────────────────────────
    cond_dim: int = 256              # width of the σ embedding fed to AdaLN

    # ── arms NOT built in v1 — configuring them RAISES rather than being ignored ──
    cfg_scale: float = 0.0           # classifier-free guidance (their ViT path has it)
    self_conditioning: bool = False

    def __post_init__(self) -> None:
        if self.mode not in ("b1", "b3"):
            raise ValueError(f"db.mode must be 'b1' or 'b3', got {self.mode!r}")
        if self.conditioning not in ("concat", "x0_inject"):
            raise ValueError(
                f"db.conditioning must be 'concat' or 'x0_inject', got {self.conditioning!r}")
        if self.visit not in ("uniform", "mass"):
            raise ValueError(f"db.visit must be 'uniform' or 'mass', got {self.visit!r}")
        if self.loss_kind not in ("l2", "ce"):
            raise ValueError(f"db.loss_kind must be 'l2' or 'ce', got {self.loss_kind!r}")
        if self.loss_weighting not in ("unweighted", "edm"):
            raise ValueError(
                f"db.loss_weighting must be 'unweighted' or 'edm', got {self.loss_weighting!r}")
        if self.overlap_gamma < 0.0:
            raise ValueError(f"db.overlap_gamma must be ≥ 0, got {self.overlap_gamma}")
        if self.sigma_data <= 0.0:
            raise ValueError(f"db.sigma_data must be > 0, got {self.sigma_data}")
        if self.mode == "b1" and self.block_mass is not None:
            raise ValueError("db.mode='b1' is B=1 by definition; leave db.block_mass unset")
        if self.block_mass is not None:
            m = tuple(float(x) for x in self.block_mass)
            if any(x <= 0.0 for x in m):
                raise ValueError(f"db.block_mass entries must be > 0, got {m}")
            total = sum(m)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(f"db.block_mass must sum to 1.0, got {total} ({m})")
            object.__setattr__(self, "block_mass", m)
        # No-theater: a silently-ignored config key is worse than a missing one.
        if float(self.cfg_scale) != 0.0:
            raise NotImplementedError(
                "db.cfg_scale > 0 — classifier-free guidance exists in the authors' ViT "
                "path but is NOT implemented here (there is no class label to drop for a "
                "language model). Leave it 0.0."
            )
        if bool(self.self_conditioning):
            raise NotImplementedError(
                "db.self_conditioning=true is not implemented in v1. Leave it false."
            )

    @property
    def n_blocks(self) -> int:
        return 1 if self.mode == "b1" else 3


@dataclass
class DBStep:
    """One sampled training step. The per-forward argument, like ``SlotLayout``.

    ``db_step=None`` in the forward means plain MORPH: no noising, no σ conditioning,
    bit-identical to today. This object is built by the training loop (never inside the
    model) so the sampling is one testable unit and shows up whole in a wandb manifest.

    Attrs:
        block_idx: which block to train, in LAYER order (0 = prelude-most).
        sigma:     ``[B]`` per-sample noise level, fp32.
        z_noisy:   ``[B, L, d]`` noised target ``y + σ·ε`` (already slice-scaled).
        y_clean:   ``[B, L, d]`` the slice-scaled clean target, kept for metrics.
        labels:    ``[B, L]`` token ids the CE is taken against.
    """

    block_idx: int
    sigma: Tensor
    z_noisy: Tensor
    y_clean: Tensor
    labels: Tensor


class DBSchedule:
    """Equi-probability σ partition, block choice, and per-block σ sampling.

    Pure math, no parameters, no autograd. Mirrors ``dblock_modules.py::get_block_sigmas``
    and ``model.py::get_sigmas`` (audit §3) with one deliberate difference: the block mass
    may be UNEQUAL, because MORPH's core is weight-tied and applied T̄ times while the
    prelude and coda run once. The paper never needs this — with no weight tying its
    B blocks are B steps, so equal mass and equal step count coincide.

    Boundaries are stored ASCENDING in σ: ``sigmas[0] == σ_min`` … ``sigmas[B] == σ_max``.
    Block ``b`` in LAYER order owns ``[sigmas[B-1-b], sigmas[B-b]]`` — the first layers
    handle the HIGHEST noise (coarse) and the last layers the lowest (fine), matching
    ``model.py::estimate_target_layer``'s ``(num_blocks - 1) - bucketize(...)`` reversal.
    """

    def __init__(self, cfg: DBConfig, mean_depth: int = 6):
        self.cfg = cfg
        self.n_blocks = cfg.n_blocks
        self.gamma = float(cfg.overlap_gamma)

        if cfg.block_mass is not None:
            mass = list(cfg.block_mass)
        elif self.n_blocks == 1:
            mass = [1.0]
        elif self.n_blocks == 3:
            # 1 prelude Euler step + T̄ core steps + 1 coda step, each equal mass.
            total_steps = 1 + int(mean_depth) + 1
            mass = [1.0 / total_steps, float(mean_depth) / total_steps, 1.0 / total_steps]
        else:
            mass = [1.0 / self.n_blocks] * self.n_blocks
        self.mass_layer_order = tuple(mass)

        # CDF-space span of the truncated log-normal.
        self._cdf_min = _norm_cdf((math.log(SIGMA_MIN) - P_MEAN) / P_STD)
        self._cdf_max = _norm_cdf((math.log(SIGMA_MAX) - P_MEAN) / P_STD)

        # Ascending σ boundaries. Mass is given in layer order (prelude first = highest
        # σ), so reverse it to walk σ upward.
        mass_ascending = list(reversed(mass))
        self.sigmas: list[float] = [SIGMA_MIN]
        acc = 0.0
        for m in mass_ascending[:-1]:
            acc += m
            self.sigmas.append(self._sigma_at_mass(acc))
        self.sigmas.append(SIGMA_MAX)
        assert len(self.sigmas) == self.n_blocks + 1

    def _sigma_at_mass(self, frac: float) -> float:
        q = self._cdf_min + (self._cdf_max - self._cdf_min) * frac
        return math.exp(P_MEAN + P_STD * _norm_ppf(q))

    def block_range(self, block_idx: int) -> tuple[float, float]:
        """γ-extended ``(σ_lo, σ_hi)`` for a block in LAYER order."""
        b_asc = self.n_blocks - 1 - block_idx
        lo, hi = self.sigmas[b_asc], self.sigmas[b_asc + 1]
        if self.gamma > 0.0:
            log_lo, log_hi = math.log(lo), math.log(hi)
            span = log_hi - log_lo
            lo = max(math.exp(log_lo - self.gamma * span), self.sigmas[0])
            hi = min(math.exp(log_hi + self.gamma * span), self.sigmas[-1])
        return lo, hi

    def visit_probs(self) -> list[float]:
        if self.cfg.visit == "uniform":
            return [1.0 / self.n_blocks] * self.n_blocks
        return list(self.mass_layer_order)

    def sample_block(self, generator: torch.Generator | None = None) -> int:
        """Choose ONE block for the whole batch (the authors' ``k=1``, audit §3)."""
        p = torch.tensor(self.visit_probs(), dtype=torch.float64)
        return int(torch.multinomial(p, 1, generator=generator).item())

    def sample_sigma(self, block_idx: int, n: int, device,
                     generator: torch.Generator | None = None) -> Tensor:
        """``[n]`` σ drawn from the log-normal RESTRICTED to the block's γ-extended range.

        Drawn uniformly in CDF space then mapped back, which is exactly sampling
        ``p_noise`` conditioned on the interval (the authors' ``np.random.uniform`` between
        the interval's two CDF values).
        """
        lo, hi = self.block_range(block_idx)
        q_lo = _norm_cdf((math.log(lo) - P_MEAN) / P_STD)
        q_hi = _norm_cdf((math.log(hi) - P_MEAN) / P_STD)
        u = torch.rand(n, dtype=torch.float64, device=device, generator=generator)
        q = q_lo + (q_hi - q_lo) * u
        # erfinv-based ppf, vectorised.
        z = math.sqrt(2.0) * torch.erfinv(2.0 * q - 1.0)
        return (P_MEAN + P_STD * z).exp().float()

    def block_of_sigma(self, sigma: Tensor) -> Tensor:
        """Map σ → block index in LAYER order (inference-side, their ``bucketize``)."""
        bounds = torch.tensor(self.sigmas[1:-1], device=sigma.device, dtype=sigma.dtype)
        asc = torch.bucketize(sigma, bounds, right=True)
        return (self.n_blocks - 1 - asc).clamp_(0, self.n_blocks - 1).long()

    def inference_sigmas(self, n_steps: int) -> Tensor:
        """``[n_steps]`` DESCENDING σ, equi-probability spaced (their ``dblock=True``).

        Descending is load-bearing: :func:`euler_step` reads the sign from
        ``next_sigma - sigma < 0``.
        """
        if n_steps < 2:
            raise ValueError(f"n_steps must be ≥ 2, got {n_steps}")
        qs = [self._cdf_min + (self._cdf_max - self._cdf_min) * (i / (n_steps - 1))
              for i in range(n_steps)]
        s = [math.exp(P_MEAN + P_STD * _norm_ppf(min(max(q, 1e-12), 1 - 1e-12))) for q in qs]
        return torch.tensor(list(reversed(s)), dtype=torch.float32)

    def manifest(self) -> dict:
        """Everything wandb needs to reproduce this schedule (global CLAUDE.md rule)."""
        return {
            "db/mode": self.cfg.mode,
            "db/conditioning": self.cfg.conditioning,
            "db/loss_kind": self.cfg.loss_kind,
            "db/loss_weighting": self.cfg.loss_weighting,
            "db/n_blocks": self.n_blocks,
            "db/block_mass": list(self.mass_layer_order),
            "db/visit": self.cfg.visit,
            "db/visit_probs": self.visit_probs(),
            "db/overlap_gamma": self.gamma,
            "db/sigma_boundaries_ascending": list(self.sigmas),
            "db/block_ranges_layer_order": [list(self.block_range(b))
                                            for b in range(self.n_blocks)],
            "db/sigma_min": SIGMA_MIN,
            "db/sigma_max": SIGMA_MAX,
            "db/p_mean": P_MEAN,
            "db/p_std": P_STD,
            "db/sigma_data": float(self.cfg.sigma_data),
            "db/slice_scale": bool(self.cfg.slice_scale),
        }


class EDMPrecond:
    """EDM preconditioning and loss weighting. Verbatim from the authors (audit §3).

    ``model.py:203-206`` and ``model.py:222``. Written as a tiny class rather than four
    loose functions so ``σ_data`` cannot drift between the coefficients and the weight.
    """

    def __init__(self, sigma_data: float = SIGMA_DATA):
        self.sigma_data = float(sigma_data)

    def coeffs(self, sigma: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(c_skip, c_out, c_in, c_noise)`` for ``[B]`` σ, computed in fp32."""
        sd = self.sigma_data
        s = sigma.float()
        s2 = s * s
        denom = s2 + sd * sd
        c_skip = (sd * sd) / denom
        c_out = s * sd / denom.sqrt()
        c_in = 1.0 / denom.sqrt()
        c_noise = 0.25 * s.log()
        return c_skip, c_out, c_in, c_noise

    def weight(self, sigma: Tensor) -> Tensor:
        """EDM loss weighting ``w(σ) = (σ² + σ_d²)/(σ·σ_d)²``.

        App. C calls this *crucial* for equi-probability partitioning to work — it
        counteracts the log-normal's sampling bias. Not optional.
        """
        sd = self.sigma_data
        s = sigma.float()
        return (s * s + sd * sd) / (s * sd) ** 2


class SliceScaler(nn.Module):
    """Pin each embedding slice's scale so ``σ_data`` is a number we actually know (O1).

    The problem (audit §6). ``HybridEmbedding`` CONCATENATES a euclidean slice with a
    Lorentz-tangent slice. The euclidean slice inits near unit norm; the Lorentz tangent
    slice inits at std 0.005 (the log-map is ≈ identity near the origin). The authors'
    ``F.normalize(x, p=2, dim=-1)`` rescales the whole vector and leaves that ~200× RATIO
    untouched — and because ``σ·ε`` is isotropic it lands the same absolute magnitude on
    the tiny Lorentz components as on the large euclidean ones, so the Lorentz slice is
    pure noise at every useful σ.

    The fix. Normalise each slice independently, then scale it so its PER-COMPONENT std is
    ``σ_data``. That is ``‖slice‖ = σ_data · √(slice_dim)``.

    Note we deliberately do NOT copy the authors' ``‖y‖ = 1``: unit norm over ``d`` dims
    means per-component std ``1/√d`` (0.031 at d=1024), which is ~16× below their
    hardcoded ``σ_data = 0.5``. Those two settings are only consistent at ``d = 4``. See
    risk R8 — we make ``σ_data`` true instead of inheriting the mismatch.

    Stateless (no parameters) and applied inside the DB conversion only, so the DB-off
    path is untouched and bit-identical.
    """

    def __init__(self, slice_dims: tuple[int, ...], sigma_data: float = SIGMA_DATA,
                 eps: float = 1e-6):
        super().__init__()
        if not slice_dims or any(d <= 0 for d in slice_dims):
            raise ValueError(f"slice_dims must be positive, got {slice_dims}")
        self.slice_dims = tuple(int(d) for d in slice_dims)
        self.sigma_data = float(sigma_data)
        self.eps = eps
        bounds, acc = [], 0
        for d in self.slice_dims:
            bounds.append((acc, acc + d))
            acc += d
        self.bounds = tuple(bounds)
        self.total_dim = acc
        # Target norm per slice so per-component std == sigma_data.
        self.target_norms = tuple(self.sigma_data * math.sqrt(d) for d in self.slice_dims)

    def forward(self, x: Tensor) -> Tensor:
        """Scale each slice of ``[..., total_dim]`` to its target norm."""
        if x.shape[-1] != self.total_dim:
            raise ValueError(
                f"SliceScaler expected last dim {self.total_dim}, got {x.shape[-1]}")
        parts = []
        for (lo, hi), target in zip(self.bounds, self.target_norms):
            sl = x[..., lo:hi]
            norm = sl.float().norm(dim=-1, keepdim=True).clamp_min(self.eps)
            parts.append((sl.float() * (target / norm)).to(x.dtype))
        return torch.cat(parts, dim=-1)

    def extra_repr(self) -> str:
        return (f"slice_dims={self.slice_dims}, sigma_data={self.sigma_data}, "
                f"target_norms={tuple(round(n, 4) for n in self.target_norms)}")


class SigmaConditioning(nn.Module):
    """σ → an embedding for AdaLN modulation.

    Input is ``c_noise = 0.25·log σ`` (the authors' timestep, audit §3), embedded with the
    standard sinusoidal basis then an MLP. torch.compile note: σ must arrive as a TENSOR.
    A Python float bakes into the guard set and forces a recompile per sampled σ.
    """

    def __init__(self, cond_dim: int = 256, n_freq: int = 128):
        super().__init__()
        if n_freq * 2 > cond_dim * 4:
            raise ValueError("n_freq too large for cond_dim")
        self.n_freq = n_freq
        self.mlp = nn.Sequential(
            nn.Linear(2 * n_freq, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        # Small init: the modulation starts near identity so step 0 is close to the
        # un-modulated forward (same argument as the HC ≈ plain-residual init).
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.zeros_(self.mlp[0].bias)
        nn.init.normal_(self.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, c_noise: Tensor) -> Tensor:
        """``[B]`` → ``[B, cond_dim]``."""
        half = self.n_freq
        freqs = torch.exp(
            torch.arange(half, device=c_noise.device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        ang = c_noise.float()[:, None] * freqs[None, :]
        basis = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(basis)


class AdaLNGate(nn.Module):
    """Zero-init AdaLN shift/scale for one norm site (DiT AdaLN-Zero).

    ``x ← norm(x)·(1 + scale) + shift`` with both produced from the σ embedding by a
    ZERO-initialised linear, so at step 0 the module is exactly the identity and the
    network reduces to its un-conditioned self. That is the same "starts as a no-op"
    discipline as the HC Cayley init and the ``ChannelInject`` ``log_scale=0``.
    """

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()
        self.to_mod = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        shift, scale = self.to_mod(cond).chunk(2, dim=-1)
        while shift.dim() < x.dim():
            shift = shift.unsqueeze(1)
            scale = scale.unsqueeze(1)
        return x * (1.0 + scale) + shift


def clean_noisy_mask(seq_len: int, device, dtype=torch.bool) -> Tensor:
    """Additive-free boolean attention mask for the ``concat`` conditioning (App. E.4).

    Layout is ``[clean_0 … clean_{L-1}, noisy_0 … noisy_{L-1}]``, total ``2L`` positions,
    where ``clean_t = embed(input_ids[t])`` and ``noisy_t = y[t] + σ·ε`` with
    ``y[t] = embed(labels[t]) = embed(input_ids[t+1])``.

    Returns ``[2L, 2L]`` with True = MAY attend. The four quadrants:

    * clean → clean: ordinary causal (``j ≤ i``). The clean stream is a plain LM context.
    * clean → noisy: **never**. Keeping the clean stream free of noise is the entire point;
      it also means the clean half's activations are reusable across σ.
    * noisy → clean: ``j ≤ i``. **This cutoff is the load-bearing line in this function.**
      ``clean_{i+1}`` is ``embed(input_ids[i+1])``, which IS the target at noisy position
      ``i``. Allowing ``j ≤ i+1`` would hand the denoiser its own answer and the loss would
      collapse to ~0 while teaching the model nothing.
    * noisy → noisy: causal (``j ≤ i``). Legal — a past noisy position carries a past
      token, just corrupted — and it is what lets the noisy stream be a sequence at all.
    """
    idx = torch.arange(seq_len, device=device)
    causal = idx[:, None] >= idx[None, :]          # [L, L], True where j <= i
    zeros = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    top = torch.cat([causal, zeros], dim=1)        # clean rows: causal clean, no noisy
    bottom = torch.cat([causal, causal], dim=1)    # noisy rows: causal clean, causal noisy
    return torch.cat([top, bottom], dim=0).to(dtype)


def euler_step(z: Tensor, denoised: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor:
    """One EDM probability-flow Euler step. ``sigma > next_sigma``.

    ``d = (z − D)/σ``; ``dt = σ_next − σ < 0``; ``z ← z + dt·d``. Equivalently
    ``z ← α·z + (1−α)·D`` with ``α = σ_next/σ ∈ (0,1)`` — a CONVEX blend, i.e. a
    prescribed contraction toward the denoiser output. A perfect denoiser lands exactly
    at ``y + σ_next·ε``.

    This is the authors' ``model.py:283-287``, NOT the paper's rendered Eq (3)–(5), whose
    sign makes noise increase down the schedule (audit §2).
    """
    s = sigma.float().view(-1, *([1] * (z.dim() - 1)))
    ns = next_sigma.float().view(-1, *([1] * (z.dim() - 1)))
    d = (z.float() - denoised.float()) / s
    return (z.float() + (ns - s) * d).to(z.dtype)


def expected_embedding(logits: Tensor, embed_weight: Tensor) -> Tensor:
    """``softmax(logits) @ E`` — the sampler's bridge from CE logits back to embedding space.

    The authors' ``model.py:280-281``. This is NOT in the paper text; it is how a
    cross-entropy-trained denoiser produces the ``D`` that the next Euler step needs.

    Risk R9 lives here: the result is a convex combination of the (scaled) embedding rows,
    so its norm SHRINKS toward the table mean as the model becomes less certain, while
    training always sees a full-scale ``y``. That is a train/inference mismatch inherent to
    the method. Log ``‖denoised‖`` per Euler step when sampling.

    Args:
        logits:       ``[..., vocab]``.
        embed_weight: ``[vocab, d]`` — the SAME transform applied to ``y`` must have been
                      applied to this matrix, or the sampler and the target disagree.
    """
    probs = logits.float().softmax(dim=-1)
    return (probs @ embed_weight.float()).to(embed_weight.dtype)
