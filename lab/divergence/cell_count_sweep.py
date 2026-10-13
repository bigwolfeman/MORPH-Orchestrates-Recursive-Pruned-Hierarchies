"""Token CE vs the fan's cell count M, on identical tokens (``tul.cell_count``).

Question (design note 2026-10-12-fan-variable-cell-count-design.md, gate for the sweep): how
does a trained write-all fan read at M cells per slot, M = 1..fan_k (cells are roles 0..M-1) and
M > fan_k under ``model.cell_extend`` (``cyclic``: cell i is role i mod fan_k; ``fresh``: cells
past the table take a new query, a zero offset and an identity write)?

Rows are packed ONCE at the reference width (the checkpoint's own ``prefix_k = fan_k``), and
every M re-lays the SAME rows out (``relayout_batch``), so the CE(M) curve is paired row by row:
the same tokens, the same labels, only the number of cell positions after every boundary differs.
Per-row sums are kept; the JSON carries a paired bootstrap interval for CE(M) - CE(fan_k)
(`_stats.paired_bootstrap_ci`). At M = fan_k the forward is the arm's own, so that point must equal
the filed `core_depth_sweep.py` CE at the same depth (``--depth``, default: the arm's eval depth).

Positions confound M > fan_k (design note, Risks): the coda trained to 1280 positions and the
core to 256 cells, so a CE that moves above M = fan_k may be the longer layout and not the new
cells. Below fan_k every distance is in range and the curve is clean.

Usage:
  python lab/divergence/cell_count_sweep.py \\
    --ckpt fan=nowta_pointer_fast_spdcells_speed=checkpoints/morph/<run>/step_5000.pt \\
    --cells 1,2,3,4,8,16,32 --cell-extend cyclic --depth 6 --rows 48 --out cells.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _build import build_cfg, parse_ckpt_spec, uses_sample_depth
from _rows import pack_rows, stream_from_loader
from _stats import paired_bootstrap_ci
from core_depth_sweep import ce_maps
from tul_samples import load_ckpt  # noqa: E402  (core_depth_sweep puts scripts/ on the path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH[=OVR,..]")
    ap.add_argument("--cells", default="1,2,3,4,8,16,32")
    ap.add_argument("--cell-extend", default="cyclic", choices=["cyclic", "fresh"],
                    help="the rule for cells past the role table (M > fan_k)")
    ap.add_argument("--depth", type=int, default=0,
                    help="force this slot-loop depth (0 = the arm's own eval depth)")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = [int(x) for x in a.cells.split(",")]

    from morph.model.tul_layout import relayout_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for spec_str in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(spec_str)
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        fan_k = int(cfg.tul.fan_k)
        if fan_k < 2:
            raise ValueError(f"{label}: tul.fan_k={fan_k}; the cell sweep needs a fan")
        if str(cfg.tul.get("cell_count", "fixed")) == "fixed":
            # the same weights, built so that the layout's own cell count drives the forward
            probs = [0.0] * (fan_k - 1) + [1.0]
            cfg = build_cfg(config, ["model.use_kernels=false", *ovr, "+tul.cell_count=categorical",
                                     f"+tul.cell_count_probs={probs}"])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None:
            raise ValueError(f"{label}: no TUL runtime (tul.activate_at: never)")
        if uses_sample_depth(tul_rt.model_cfg):
            raise NotImplementedError(f"{label}: a paid-loop arm has no fan")
        model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
        model.eval()
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        span_cap = tul_rt.data_cfg.rule.span_cap
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        stream = stream_from_loader(loader, a.rows * (spec.l_total + 1))
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, plain=False)[:n_batches]
        tc = model.cfg.tul
        orig = (int(tc.slot_mean_depth), int(tc.slot_max_depth), int(tc.slot_depth_fixed))
        if a.depth > 0:
            tc.slot_mean_depth = a.depth
            tc.slot_max_depth = max(a.depth, orig[1] or int(cfg.model.max_depth))
            if orig[2] > 0:
                tc.slot_depth_fixed = a.depth
        arm = {"step": step, "rows": sum(b[0].shape[0] for b in batches), "fan_k": fan_k,
               "depth": a.depth or "arm eval depth", "cell_extend": a.cell_extend, "cells": {}}
        row_sum: dict[int, np.ndarray] = {}
        row_cnt: np.ndarray | None = None
        try:
            for m in cells:
                model.cell_extend = a.cell_extend if m > fan_k else "refuse"
                rs: list[float] = []
                rc: list[float] = []
                for inp, labels, layout, _idx in batches:
                    x, y, lay = relayout_batch(inp, labels, layout, m, spec, span_cap)
                    ce, _ = ce_maps(model, x, lay.to(a.device), y, a.device)
                    tokpos = ((~lay.slot_mask) & (y >= 0))
                    rs.extend((ce.cpu() * tokpos).sum(dim=1).tolist())
                    rc.extend(tokpos.sum(dim=1).tolist())
                if row_cnt is None:
                    row_cnt = np.asarray(rc)
                elif not np.array_equal(row_cnt, np.asarray(rc)):
                    raise RuntimeError(f"M={m} scored a different token set than M={cells[0]}")
                row_sum[m] = np.asarray(rs)
                arm["cells"][m] = {"ce_tokens": float(row_sum[m].sum() / row_cnt.sum()),
                                   "n_tokens": int(row_cnt.sum())}
                print(f"{label:10s} M={m:<3d} ce={arm['cells'][m]['ce_tokens']:.4f}", flush=True)
        finally:
            model.cell_extend = "refuse"
            tc.slot_mean_depth, tc.slot_max_depth, tc.slot_depth_fixed = orig
        assert row_cnt is not None
        if fan_k in row_sum:
            arm["ci_vs_fan_k"] = {
                f"M{m}-M{fan_k}": paired_bootstrap_ci(row_sum[m], row_sum[fan_k], row_cnt)
                for m in cells if m != fan_k}
            for k, v in arm["ci_vs_fan_k"].items():
                print(f"{label:10s} {k}: {v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                      f"over {v['n_units']} rows", flush=True)
        arm["row_ce_sum"] = {str(m): row_sum[m].tolist() for m in cells}
        arm["row_n_tokens"] = row_cnt.tolist()
        results[label] = arm
        del model
        if a.device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
