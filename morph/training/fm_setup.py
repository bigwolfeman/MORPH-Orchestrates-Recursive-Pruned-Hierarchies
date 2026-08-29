"""Resolve the Hydra ``fm:`` block into the :class:`FMArmConfig` the model needs.

The ``tul_setup`` pattern: the whole block resolves to ONE object, and ``None`` means
"no planner is constructed and every path is byte-identical". Kept out of ``train.py`` so
the sizing arithmetic (which must agree with the loader's slot budget exactly) is one
testable unit.
"""

from __future__ import annotations

from morph.model.tul_fm import FMArmConfig

__all__ = ["build_fm_runtime"]


def build_fm_runtime(cfg, tul_rt) -> FMArmConfig | None:
    """Build the FM1 arm config from ``cfg.fm``; ``None`` when the arm is off.

    ``fm.enabled: false`` (the default everywhere except ``tul_fm1.yaml``) returns None,
    which is what keeps every other config byte-identical to today.

    The planner is sized from the LOADER's own slot budget — ``TulDataConfig.spec_for``,
    the same call the data pipeline makes — not from ``model.max_seq_len``. Deriving it
    twice from different places is exactly how a slot-index embedding ends up one row
    short of the layout that feeds it.
    """
    fc = getattr(cfg, "fm", None)
    if fc is None or not bool(fc.get("enabled", False)):
        return None
    if tul_rt is None:
        raise ValueError(
            "fm.enabled=true needs TUL active (tul.activate_at must not be 'never'): "
            "FM1 writes its plans into the slot prefix positions, which only exist on a "
            "TUL model.")
    if int(cfg.model.n_core) != 0:
        raise ValueError(
            f"fm.enabled=true needs model.n_core=0, got {int(cfg.model.n_core)}. The FM "
            f"planner REPLACES the core loop (arm FM1 = tul_a1 skeleton + A3's coreless "
            f"body); leaving a core would build two slot-state producers.")

    spec = tul_rt.data_cfg.spec_for(int(cfg.data.seq_len))
    arm = FMArmConfig(
        d_p=int(fc.get("d_p", 512)),
        n_layers=int(fc.get("n_layers", 4)),
        n_heads=int(fc.get("n_heads", 8)),
        d_ff=int(fc.get("d_ff", 1408)),
        cond_dim=int(fc.get("cond_dim", 256)),
        objective=str(fc.get("objective", "cfm")),
        infer_steps=int(fc.get("infer_steps", 6)),
        source_std=float(fc.get("source_std", 1.0)),
        t_embed_scale=float(fc.get("t_embed_scale", 1.0)),
        dropout=float(fc.get("dropout", 0.0)),
        fm_weight=float(fc.get("fm_weight", 1.0)),
        loss_scale=str(fc.get("loss_scale", "auto")),
        sigreg_lambda=float(fc.get("sigreg_lambda", 0.02)),
        sigreg_slices=int(fc.get("sigreg_slices", 1024)),
        max_slots=int(fc.get("max_slots", 0)) or spec.max_slots,
        l_total=int(fc.get("l_total", 0)) or spec.l_total,
    )
    print(f"  FM1 ON: objective={arm.objective} T={arm.infer_steps} "
          f"fm_weight={arm.fm_weight} sigreg_lambda={arm.sigreg_lambda} "
          f"(M={arm.sigreg_slices}) source_std={arm.source_std} "
          f"planner d_p={arm.d_p}x{arm.n_layers} sized for max_slots={arm.max_slots} "
          f"L_total={arm.l_total}", flush=True)
    return arm
