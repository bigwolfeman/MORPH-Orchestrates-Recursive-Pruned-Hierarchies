"""Compose-and-run gate for the 2026-09-13 rank levers (C1 center_exit, C2 row_contrast).

A green unit test does NOT prove an arm can start: the config has to compose through
Hydra, `build_tul_runtime` has to accept every `tul.*` key, the model has to build at the
REAL `d_model`, and one forward+backward has to run. That gap has bitten this tree before
(`compose-every-config-before-queueing`, 2026-09-12: a build-time refusal on all three
energy arms that every unit test missed).

CPU only, fp32, no kernels. Prints, per arm:

  * the composed `tul.*` knobs this panel depends on,
  * the parameter count and the DELTA against the arm's one-factor partner,
  * one forward+backward at the config's own `d_model`, with the loss, the auxiliary
    term and its readout.

Usage (from the repo root)::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python lab/divergence/rank_levers_compose.py

Record: lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.training.tul_setup import build_tul_runtime

CONFIG_DIR = "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies/.claude/worktrees/agent-a60c2c22d4392cd6c/morph/configs"

# (arm config, one-factor partner) — the partner is what the delta is reported against.
PAIRS = [
    ("tul_slot_spandec_strict_center", "tul_slot_spandec_strict"),
    ("tul_slot_register_m4_center", "tul_slot_register_m4"),
    ("tul_slot_spandec_strict_contrast", "tul_slot_spandec_strict"),
    ("tul_slot_register_m4_contrast", "tul_slot_register_m4"),
]

SHOW = ("center_exit", "row_contrast_lambda", "row_contrast_tau", "slot_cells",
        "prefix_k", "spandec", "tg_geometry", "tg_coda_prefix_reach", "loop_reach",
        "mux_beta", "sigreg_lambda", "slot_seed", "max_slots", "center_bag_mean")

V = 512


def _cfg(name: str):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=name)


def _build(name: str, d_model: int, n_slots: int):
    """The REAL path: Hydra -> build_tul_runtime -> MORPHTransformer, at `d_model`."""
    from morph.model.transformer import MORPHConfig, MORPHTransformer

    cfg = _cfg(name)
    OmegaConf.set_struct(cfg, False)
    cfg.model.bigram_hash_vocab = V
    cfg.tul.max_slots = n_slots
    rt = build_tul_runtime(cfg)
    m = cfg.model
    mc = MORPHConfig(
        d_model=d_model, n_heads=int(m.n_heads), n_kv_heads=int(m.n_kv_heads),
        vocab_size=V, max_seq_len=1024, context_len=1024,
        n_prelude=int(m.n_prelude), n_core=int(m.n_core), n_coda=int(m.n_coda),
        mean_depth=int(m.mean_depth), max_depth=int(m.max_depth),
        bptt_depth=int(m.bptt_depth),
        channel_dims=_scaled_channels(tuple(int(c) for c in m.channel_dims), d_model),
        compression=int(m.compression), csa_compress_ratio=int(m.csa_compress_ratio),
        hca_compress_ratio=int(m.hca_compress_ratio), top_k=int(m.top_k),
        window_size=int(m.window_size), retention=bool(m.retention),
        d_ff=int(m.d_ff), bigram_hash_vocab=V, use_kernels=False,
        hc_use_kernel=False, dropout=0.0,
        core_fixed_point_lambda=float(getattr(m, "core_fixed_point_lambda", 0.0)),
        slot_gain_lambda=float(getattr(m, "slot_gain_lambda", 0.0)),
        slot_cot_clip=float(getattr(m, "slot_cot_clip", 0.0)),
        tul=rt.model_cfg,
    )
    return MORPHTransformer(mc), rt, cfg


def _scaled_channels(ch: tuple[int, ...], d_model: int) -> tuple[int, ...]:
    """`channel_dims` must sum to `d_model` (transformer.py: `assert sum(ch) == d`).

    At the config's own `d_model` this returns `ch` unchanged, which is the case the gate
    is FOR. A smaller `d_model` is a cheap sanity pass only, and its split is the config's
    proportions rounded with the remainder dropped into the first channel.
    """
    tot = sum(ch)
    if tot == d_model:
        return ch
    out = [max(1, round(c * d_model / tot)) for c in ch]
    out[0] += d_model - sum(out)
    if min(out) < 1:
        raise ValueError(f"d_model={d_model} is too small for channel_dims={ch}")
    return tuple(out)


def _ids(B: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[:, ::9] = 46          # a boundary char id; the rule below marks it
    return ids.astype(np.int64)


def main() -> int:
    torch.manual_seed(0)
    d_model = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    B, n_tok, n_slots, k_max = 2, 72, 8, 16

    lut = np.zeros(V, dtype=bool)
    lut[46] = True
    lut[0] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=k_max, eos_id=0)

    partners: dict[str, int] = {}
    ok = True
    for arm, partner in PAIRS:
        print("=" * 78, flush=True)
        for name in (partner, arm):
            if name in partners and name == partner:
                continue
            torch.manual_seed(0)
            try:
                model, rt, cfg = _build(name, d_model, n_slots)
            except Exception as e:                      # noqa: BLE001 - report, do not hide
                import traceback
                print(f"  {name}: BUILD FAILED {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                ok = False
                continue
            n_par = sum(p.numel() for p in model.parameters())
            partners.setdefault(name, n_par)
            knobs = " ".join(f"{k}={getattr(rt.model_cfg, k)!r}" for k in SHOW
                             if hasattr(rt.model_cfg, k))
            tag = "PARTNER" if name == partner else "ARM"
            delta = ""
            if name == arm:
                delta = f"  delta_vs_{partner}={n_par - partners[partner]:+d}"
            print(f"[{tag}] {name}", flush=True)
            print(f"  knobs: {knobs}", flush=True)
            print(f"  params: {n_par:,}{delta}", flush=True)
            print(f"  modules: tul_center={model.tul_center is not None} "
                  f"tul_contrast={model.tul_contrast is not None} "
                  f"tul_register={model.tul_register is not None}", flush=True)

            spec = TulLayoutSpec(seq_len=n_tok, max_slots=n_slots,
                                 prefix_k=int(rt.model_cfg.prefix_k),
                                 slot_id=int(rt.model_cfg.slot_id))
            ids = _ids(B, n_tok * 2, seed=7)
            x, y, layout, _st = slot_layout_from_ids(ids, rule, spec)
            model.train()
            out = model(x, labels=y, slot_layout=layout)
            loss = out["loss"]
            loss.backward()
            gsum = sum(float(p.grad.abs().sum()) for p in model.parameters()
                       if p.grad is not None)
            finite = all(torch.isfinite(p.grad).all() for p in model.parameters()
                         if p.grad is not None)
            print(f"  fwd+bwd: loss={float(loss):.10f} grad_abs_sum={gsum:.6f} "
                  f"all_finite={finite}", flush=True)
            for k in ("row_contrast", "row_contrast_weighted", "row_contrast_acc",
                      "row_contrast_n_rows", "row_contrast_n_anchors", "spandec"):
                if k in out and out[k] is not None:
                    print(f"    {k}={float(out[k]):.6f}", flush=True)
            if model.tul_center is not None:
                bc = model.tul_center.b_center
                print(f"    b_center shape={tuple(bc.shape)} norm={float(bc.norm()):.6e} "
                      f"grad_abs_sum={float(bc.grad.abs().sum()):.6e}", flush=True)
            if not finite:
                ok = False
            model.zero_grad(set_to_none=True)
    print("=" * 78, flush=True)
    print("COMPOSE GATE:", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
