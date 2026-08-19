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


def db_loss(out: dict, step: DBStep, precond: EDMPrecond, model,
            ignore_index: int = -100, chunk_size: int = 1024) -> tuple[Tensor, dict]:
    """EDM-weighted cross-entropy through the CHUNKED head, plus per-block metrics.

    Matches the authors' ``model.py:255-259`` — per-sample CE, then ``× w(σ)``, then mean —
    but never materialises the logits. ``[B, L, vocab]`` in fp32 is 2.63 GiB at batch 14 /
    seq 1024 / V 49152, and that exact allocation OOM'd the first ``db_b1`` smoke on a 32 GB
    card. ``fused_linear_cross_entropy`` streams the head in ``chunk_size`` row blocks and
    already accepts per-row ``weights``, which is precisely what ``w(σ)`` is.

    Normalisation detail that matters. The fused kernel reduces as ``Σ wᵢ·CEᵢ / Σ wᵢ`` — a
    *normalised* weighted mean — while the authors use the *unnormalised* ``(CE·w).mean()``.
    Those differ: normalising per batch would partly undo the point of EDM's weighting,
    which is to equalise gradient magnitude ACROSS σ draws, not within one. Multiplying the
    fused result by ``Σw / n_valid`` recovers the authors' form exactly, and it is a scalar
    factor so the gradient stays correct.

    NOTE this CE is NOT comparable to A0's ``val/ppl_tokens``. It is conditioned on a
    σ-noised target, so it is a reconstruction number, not a likelihood (sheet §1.3). Never
    table it next to a baseline CE.
    """
    from morph.model.fused_ce import fused_linear_cross_entropy

    denoised = out["denoised"]
    B, L, _ = denoised.shape
    labels = step.labels
    x_flat = denoised.reshape(B * L, -1)
    lab_flat = labels.reshape(B * L)

    # w(σ) is per SAMPLE; the fused kernel wants one weight per ROW (= position).
    w_sample = precond.weight(step.sigma)                      # [B]
    w_row = w_sample.view(B, 1).expand(B, L).reshape(B * L)     # [B*L]

    valid = lab_flat != ignore_index
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("db_loss got a batch with no valid label positions")

    head_w = model.db_lm_weight()
    fused = fused_linear_cross_entropy(
        x_flat, head_w, lab_flat,
        ignore_index=ignore_index, chunk_size=chunk_size,
        weights=w_row.to(x_flat.dtype),
    )
    # fused = Σ w·CE / Σ w  ->  × (Σ w / n_valid) = mean(w·CE), the authors' reduction.
    w_sum = w_row[valid].sum()
    loss = fused * (w_sum / n_valid)

    b = step.block_idx
    with torch.no_grad():
        metrics = {
            "db/loss": float(loss.detach()),
            # The weighted number is not comparable across σ ranges; this one is.
            "db/ce_weighted_norm": float(fused.detach()),
            f"db/loss_block{b}": float(loss.detach()),
            "db/block_idx": b,
            "db/sigma_mean": float(step.sigma.detach().mean()),
            "db/sigma_min_batch": float(step.sigma.detach().min()),
            "db/sigma_max_batch": float(step.sigma.detach().max()),
            "db/weight_mean": float(w_sample.detach().mean()),
            # Kill criterion 4 (sheet §4.5): embedding collapse. If the targets degenerate
            # toward one vector this norm stops moving and the pairwise cosine climbs.
            "db/target_norm_mean": float(step.y_clean.detach().float().norm(dim=-1).mean()),
        }
    return loss, metrics
