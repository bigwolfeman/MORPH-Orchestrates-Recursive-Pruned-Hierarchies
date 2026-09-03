"""Resolve the Hydra ``dmorph:`` block into the :class:`DmorphConfig` the model needs.

The ``tul_setup`` / ``fm_setup`` pattern: the whole block resolves to ONE object, and
``None`` means "no noisy stream is constructed and every path is byte-identical".
Unknown keys RAISE (``KNOWN_DMORPH_KEYS``): a retired or misspelt key must never run the
control under a name that promises something else (runtime-invariants §6b).

Everything DERIVED here (the matched source std, the input gain, the null floor, the
band bounds) goes into the wandb manifest, so a run is reproducible from its config alone
and "what source_std did we try?" is greppable (global CLAUDE.md rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from morph.model.dmorph import ARMS, DmorphConfig, band_bounds

__all__ = ["DmorphRuntime", "KNOWN_DMORPH_KEYS", "build_dmorph_runtime",
           "reject_unknown_dmorph_keys"]

KNOWN_DMORPH_KEYS = frozenset({
    "enabled", "arm", "n_blocks", "gamma", "lambda_fm", "lambda_ce", "source_std",
    "detach_ctx", "t_per_position", "sigreg_lambda", "sigreg_slices", "infer_steps",
    "cond_dim", "t_embed_scale", "in_gain", "loss_scale", "block_visit",
})


def reject_unknown_dmorph_keys(dc) -> None:
    """Raise ``ValueError`` naming every ``dmorph:`` key outside :data:`KNOWN_DMORPH_KEYS`."""
    unknown = sorted(str(k) for k in dc.keys() if str(k) not in KNOWN_DMORPH_KEYS)
    if unknown:
        raise ValueError(
            f"dmorph: has unknown key(s) {unknown}. Known keys: {sorted(KNOWN_DMORPH_KEYS)}. "
            f"Arms `stp_lambda`, `xattn`, `bcast`, a soft bridge and an EDM objective are "
            f"NOT built (design note, Alternatives considered) and raise here rather than "
            f"being ignored.")


@dataclass
class DmorphRuntime:
    """Everything dmorph needs at runtime, resolved once at train start."""

    model_cfg: DmorphConfig
    manifest: dict = field(default_factory=dict)


def build_dmorph_runtime(cfg, tul_rt) -> DmorphRuntime | None:
    """Build the dmorph runtime from ``cfg.dmorph``; ``None`` when the stream is off.

    ``dmorph.enabled: false`` (the default everywhere except the ``dmorph_*`` configs)
    returns None, which is what keeps every other config byte-identical to today.
    """
    dc = getattr(cfg, "dmorph", None)
    if dc is None:
        return None
    reject_unknown_dmorph_keys(dc)
    if not bool(dc.get("enabled", False)):
        return None
    if tul_rt is None:
        raise ValueError(
            "dmorph.enabled=true needs TUL active (tul.activate_at must not be 'never'): "
            "the noisy stream runs over the packed row and the hs target lives at the "
            "slot positions.")
    n_core = int(cfg.model.n_core)
    if n_core != 0:
        raise ValueError(
            f"dmorph.enabled=true needs model.n_core=0, got {n_core}. dmorph is the no-loop "
            f"arm (design note, Backbone); use n_prelude/n_coda for depth.")
    n_layers = int(cfg.model.n_prelude) + int(cfg.model.n_coda)
    n_blocks = int(dc.get("n_blocks", 4))
    if n_blocks < 1 or n_layers % n_blocks != 0:
        raise ValueError(
            f"dmorph.n_blocks={n_blocks} must divide n_prelude + n_coda = {n_layers}: a "
            f"block is one contiguous, equal-size group of the flat stack.")
    arm = str(dc.get("arm", "tok"))
    if arm not in ARMS:
        raise ValueError(f"dmorph.arm must be one of {ARMS}, got {arm!r}")
    d = int(cfg.model.d_model)

    raw_source = dc.get("source_std", "matched")
    if isinstance(raw_source, str):
        if raw_source.lower() != "matched":
            raise ValueError(
                f"dmorph.source_std must be a float or 'matched' (= 1/sqrt(d)), got {raw_source!r}")
        source_std = 1.0 / math.sqrt(d)
    else:
        source_std = float(raw_source)
    raw_gain = dc.get("in_gain", "auto")
    if isinstance(raw_gain, str):
        if raw_gain.lower() != "auto":
            raise ValueError(
                f"dmorph.in_gain must be a float or 'auto' (= sqrt(d)), got {raw_gain!r}")
        in_gain = math.sqrt(d)
    else:
        in_gain = float(raw_gain)
    visit = dc.get("block_visit", None)
    visit_t = None if visit is None else tuple(float(x) for x in visit)

    model_cfg = DmorphConfig(
        arm=arm,
        n_blocks=n_blocks,
        gamma=float(dc.get("gamma", 0.1)),
        lambda_fm=float(dc.get("lambda_fm", 1.0)),
        lambda_ce=float(dc.get("lambda_ce", 1.0)),
        source_std=source_std,
        detach_ctx=bool(dc.get("detach_ctx", False)),
        t_per_position=bool(dc.get("t_per_position", False)),
        sigreg_lambda=float(dc.get("sigreg_lambda", 0.0)),
        sigreg_slices=int(dc.get("sigreg_slices", 1024)),
        infer_steps=int(dc.get("infer_steps", 0)),
        cond_dim=int(dc.get("cond_dim", 256)),
        t_embed_scale=float(dc.get("t_embed_scale", 16.0)),
        in_gain=in_gain,
        loss_scale=str(dc.get("loss_scale", "auto")),
        block_visit=visit_t,
    )
    matched = 1.0 / math.sqrt(d)
    note = "MATCHED" if abs(source_std - matched) < 0.25 * matched else "MISMATCHED"
    manifest = {
        "enabled": True,
        "arm": arm,
        "n_blocks": n_blocks,
        "n_layers": n_layers,
        "layers_per_block": n_layers // n_blocks,
        "gamma": model_cfg.gamma,
        "band_bounds_widened": [list(band_bounds(b, n_blocks, model_cfg.gamma))
                                for b in range(n_blocks)],
        "lambda_fm": model_cfg.lambda_fm,
        "lambda_ce": model_cfg.lambda_ce,
        "source_std_raw": str(raw_source),
        "source_std": source_std,
        "source_std_matched": matched,
        "source_std_note": note,
        "in_gain_raw": str(raw_gain),
        "in_gain": in_gain,
        "null_floor": 1.0 + d * source_std ** 2,
        "detach_ctx": model_cfg.detach_ctx,
        "t_per_position": model_cfg.t_per_position,
        "sigreg_lambda": model_cfg.sigreg_lambda,
        "sigreg_slices": model_cfg.sigreg_slices,
        "infer_steps": model_cfg.n_infer_steps,
        "cond_dim": model_cfg.cond_dim,
        "t_embed_scale": model_cfg.t_embed_scale,
        "loss_scale": model_cfg.loss_scale,
        "block_visit": (list(model_cfg.block_visit) if model_cfg.block_visit is not None
                        else [1.0 / n_blocks] * n_blocks),
        "target_detached": not (arm == "hs" and model_cfg.sigreg_lambda > 0.0),
    }
    print(f"  dmorph ON: arm={arm} n_blocks={n_blocks} ({n_layers // n_blocks} layers each) "
          f"gamma={model_cfg.gamma} lambda_fm={model_cfg.lambda_fm} "
          f"lambda_ce={model_cfg.lambda_ce} source_std={source_std:.5f} [{note}] "
          f"in_gain={in_gain:.3f} null_floor={manifest['null_floor']:.3f} "
          f"detach_ctx={model_cfg.detach_ctx} t_per_position={model_cfg.t_per_position} "
          f"sigreg_lambda={model_cfg.sigreg_lambda} infer_steps={model_cfg.n_infer_steps}",
          flush=True)
    return DmorphRuntime(model_cfg=model_cfg, manifest=manifest)
