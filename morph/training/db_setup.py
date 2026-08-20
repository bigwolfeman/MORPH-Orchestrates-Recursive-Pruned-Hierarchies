"""Resolve the Hydra ``db:`` block and build one training step's noised target.

Plan: ``docs/diffusionblocks-plan-of-action.md`` (A5). Metric contract and arms:
``docs/diffusionblocks-experiment-sheet.md``.

Kept out of ``train.py`` for the same reason ``tul_setup.py`` is: the sampling is then one
testable unit, it shows up whole in the wandb config, and ``train.py``'s DB seam is a few
lines. The whole block resolves to ONE object; ``None`` means plain MORPH — no DB
parameters are constructed, the forward never sees a ``DBStep``, and the path stays
bit-identical to today.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

from morph.model.diffusion_blocks import (
    DBConfig,
    DBSchedule,
    DBStep,
    EDMPrecond,
    SliceScaler,
)

__all__ = ["DbRuntime", "build_db_runtime", "build_db_step", "scaled_lm_weight"]

NEVER = "never"


@dataclass
class DbRuntime:
    """Everything DiffusionBlocks needs at runtime, resolved once at train start."""

    model_cfg: DBConfig
    schedule: DBSchedule
    precond: EDMPrecond
    # Built here ONLY to derive the manifest (slice dims / target norms). The scaler the
    # forward and the target actually use lives on the MODEL (`model.db_scaler`) so there is
    # exactly one. Do not use this one for tensors.
    scaler: SliceScaler | None
    activate_at: float
    manifest: dict = field(default_factory=dict)

    def activation_step(self, total_steps: int) -> int:
        return int(self.activate_at * total_steps)

    def positions_per_token(self, tul_positions_per_token: float = 1.0) -> float:
        """``L_total / seq_len`` for the perf metrics.

        The ``concat`` conditioning doubles positions (App. E.4); ``x0_inject`` does not.
        TUL's slot inflation multiplies on top of whichever is chosen — this is the
        sequence-budget collision the assessment flags, and it is why the TUL-crossed arms
        carry the widest pre-registered bands.
        """
        factor = 2.0 if self.model_cfg.conditioning == "concat" else 1.0
        return tul_positions_per_token * factor


def build_db_runtime(cfg) -> DbRuntime | None:
    """Build the DB runtime from ``cfg.db``; ``None`` when DiffusionBlocks is off.

    ``db.activate_at: never`` (the base.yaml default) returns None, which is what keeps the
    default recipe bit-identical to plain MORPH.
    """
    dc = getattr(cfg, "db", None)
    if dc is None:
        return None
    raw = dc.get("activate_at", NEVER)
    if raw is None or (isinstance(raw, str) and str(raw).lower() == NEVER):
        return None
    activate_at = float(raw)
    if not 0.0 <= activate_at < 1.0:
        raise ValueError(f"db.activate_at must be in [0,1) or 'never', got {raw!r}")

    mass = dc.get("block_mass", None)
    model_cfg = DBConfig(
        mode=str(dc.get("mode", "b3")),
        conditioning=str(dc.get("conditioning", "concat")),
        block_mass=tuple(float(x) for x in mass) if mass else None,
        visit=str(dc.get("visit", "uniform")),
        overlap_gamma=float(dc.get("overlap_gamma", 0.1)),
        slice_scale=bool(dc.get("slice_scale", True)),
        sigma_data=float(dc.get("sigma_data", 0.5)),
        loss_kind=str(dc.get("loss_kind", "l2")),
        loss_weighting=str(dc.get("loss_weighting", "unweighted")),
        cond_dim=int(dc.get("cond_dim", 256)),
        cfg_scale=float(dc.get("cfg_scale", 0.0)),
        self_conditioning=bool(dc.get("self_conditioning", False)),
    )

    schedule = DBSchedule(model_cfg, mean_depth=int(cfg.model.mean_depth))
    precond = EDMPrecond(model_cfg.sigma_data)

    scaler = None
    if model_cfg.slice_scale:
        # Slice widths must match HybridEmbedding's concatenation EXACTLY, or the scaler
        # rescales the wrong channels. Derived from the same formula the module uses
        # (embeddings.py: lorentz_dim = int(d_model * lorentz_fraction)).
        d_model = int(cfg.model.d_model)
        lorentz_dim = int(d_model * float(cfg.model.lorentz_fraction))
        euclidean_dim = d_model - lorentz_dim
        if lorentz_dim <= 0 or euclidean_dim <= 0:
            raise ValueError(
                f"lorentz_fraction={cfg.model.lorentz_fraction} gives slices "
                f"({euclidean_dim}, {lorentz_dim}); both must be > 0")
        scaler = SliceScaler((euclidean_dim, lorentz_dim), model_cfg.sigma_data)

    manifest = {
        "db/activate_at": activate_at,
        "db/cond_dim": model_cfg.cond_dim,
        **schedule.manifest(),
    }
    if scaler is not None:
        manifest["db/slice_dims"] = list(scaler.slice_dims)
        manifest["db/slice_target_norms"] = [round(n, 6) for n in scaler.target_norms]

    return DbRuntime(
        model_cfg=model_cfg,
        schedule=schedule,
        precond=precond,
        scaler=scaler,
        activate_at=activate_at,
        manifest=manifest,
    )


def scaled_lm_weight(embedding, scaler: SliceScaler | None) -> Tensor:
    """The tied LM-head matrix under the SAME slice transform applied to the target.

    Audit §4: the sampler's denoised estimate is ``softmax(logits) @ E``, which puts the
    tied head INSIDE the sampling loop. If ``y`` is slice-scaled but ``E`` is not, the
    sampler's output lives in a different space from the training target and the Euler
    steps walk in the wrong units. Gradient still flows to the underlying embedding
    parameters through ``lm_weight()`` and through the scaler (both are differentiable).
    """
    w = embedding.lm_weight()          # [vocab, d_model]
    return w if scaler is None else scaler(w)


def build_db_step(rt: DbRuntime, model, labels: Tensor,
                  generator: torch.Generator | None = None) -> DBStep:
    """Sample one ``(block, σ)`` and build the noised target for this batch.

    The target at position ``t`` is ``embed(labels[t])``. MORPH's loader gives
    ``labels[t] = input_ids[t+1]``, so this is the NEXT token's embedding and the existing
    unshifted ``x0[t] = embed(input_ids[t])`` is legitimate conditioning — different
    tokens, no leak. See :class:`DBConfig` for why no shift is applied.

    ONE block is chosen for the whole batch, matching the authors' ``random.choices(..., k=1)``
    (audit §3). σ is then drawn per SAMPLE inside that block's γ-extended range, so a batch
    spans a range of noise levels within one block's remit.

    Args:
        rt:        the resolved runtime.
        model:     the MORPH model. Its ``embed`` and its ``db_scaler`` are used — NOT
                   ``rt.scaler`` — so the target ``y`` and the tied LM-head weight
                   (``model.db_lm_weight()``) are guaranteed to be in the same space. Two
                   independently-built scalers would be numerically identical today and a
                   silent drift the first time one side's config changes.
        labels:    ``[B, L]`` token ids (already next-token shifted by the loader).
        generator: optional RNG for reproducible arms.
    """
    if labels.dim() != 2:
        raise ValueError(f"labels must be [B, L], got {tuple(labels.shape)}")
    device = labels.device
    B = labels.shape[0]

    block_idx = rt.schedule.sample_block(generator)
    sigma = rt.schedule.sample_sigma(block_idx, B, device, generator)

    y = model.embed(labels)                     # [B, L, d_model]
    scaler = getattr(model, "db_scaler", None)
    if scaler is not None:
        y = scaler(y)

    # VE noising, exactly the authors' model.py:252. Noise is drawn in the target's dtype
    # but the scaling is done in fp32 to keep small-σ steps representable.
    eps = torch.randn(y.shape, device=device, dtype=torch.float32, generator=generator)
    z = (y.float() + sigma.view(B, 1, 1) * eps).to(y.dtype)

    return DBStep(
        block_idx=block_idx,
        sigma=sigma,
        z_noisy=z,
        y_clean=y,
        labels=labels,
    )


def db_loss(pred: Tensor, step: DBStep, precond: EDMPrecond,
            ignore_index: int = -100, weighting: str = "unweighted",
            loss_kind: str = "l2") -> tuple[Tensor, dict]:
    """The DB training loss, plus the per-block metric channels.

    ``loss_kind`` (DBConfig.loss_kind) selects the objective AND what ``pred`` is:
      * ``"l2"`` (default, Option A — the paper's AR objective): ``pred`` is the DENOISED
        embedding estimate ``D̂ = c_skip·z + c_out·hidden`` (``[B, L, d]``, the forward's
        ``out["denoised"]``). Loss is the per-dim mean squared error against ``y_clean``.
        No readout is used, so the tied-head free ride cannot occur by construction.
        With ``weighting="edm"`` (the correct pairing — w(σ)=1/c_out² is DERIVED for this
        L2 regression and makes the raw network output a unit-variance target at every σ)
        the loss is the paper's ``w(σ)·‖D̂−y‖²``.
      * ``"ce"`` (Option B — MORPH extension): ``pred`` is the tied-readout logits
        (``[B, L, V]``, the forward's ``out["logits"]``); CE against ``labels``.

    ``weighting`` (DBConfig.loss_weighting):
      * ``"edm"``: multiply per-sample loss by ``w(σ)=1/c_out²``. CORRECT for ``l2``
        (App. C calls it crucial). On CE it over-weights the trivial low-σ region
        (``c_skip≈1`` → z passes through the tied head → CE≈0) by up to ~62,000×, so CE
        training collapses to the low-σ copy (db-b1-concat run, 2026-08-19). Keep
        ``ce``+``edm`` only as the ablation that reproduces that collapse.
      * ``"unweighted"``: plain mean. The corrected default for ``ce``.

    The raw (unweighted) number is always logged alongside, per block index, so a single
    failing block is visible instead of being averaged away.

    NOTE neither number is comparable to A0's ``val/ppl_tokens``. Both are conditioned on
    a σ-noised target — reconstruction numbers, not likelihoods (sheet §1.3). Never table
    them next to a baseline CE.
    """
    valid = (step.labels != ignore_index)

    if loss_kind == "l2":
        if pred.shape != step.y_clean.shape:
            raise ValueError(
                f"loss_kind='l2' expects pred = denoised embeddings {tuple(step.y_clean.shape)}, "
                f"got {tuple(pred.shape)} — did the caller pass logits?")
        per_pos = ((pred.float() - step.y_clean.float()) ** 2).mean(dim=-1)   # [B, L]
        raw_name = "db/l2_raw"
    elif loss_kind == "ce":
        B, L, V = pred.shape
        per_pos = torch.nn.functional.cross_entropy(
            pred.reshape(-1, V).float(),
            step.labels.reshape(-1),
            ignore_index=ignore_index,
            reduction="none",
        ).view(B, L)
        raw_name = "db/ce_unweighted"
    else:
        raise ValueError(f"loss_kind must be 'l2' or 'ce', got {loss_kind!r}")

    per_sample = (per_pos * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    w = precond.weight(step.sigma)
    if weighting == "edm":
        loss = (per_sample * w).mean()
    else:
        loss = per_sample.mean()

    b = step.block_idx
    metrics = {
        "db/loss": float(loss.detach()),
        raw_name: float(per_sample.mean().detach()),
        f"{raw_name}_block{b}": float(per_sample.mean().detach()),
        f"db/loss_block{b}": float(loss.detach()),
        "db/block_idx": b,
        "db/sigma_mean": float(step.sigma.detach().mean()),
        "db/sigma_min_batch": float(step.sigma.detach().min()),
        "db/sigma_max_batch": float(step.sigma.detach().max()),
        "db/weight_mean": float(w.detach().mean()),
        # Kill criterion 4 (sheet §4.5): embedding collapse. A mean pairwise cosine
        # climbing toward 1.0 means the targets are degenerating to one vector.
        "db/target_norm_mean": float(step.y_clean.detach().float().norm(dim=-1).mean()),
    }
    return loss, metrics
