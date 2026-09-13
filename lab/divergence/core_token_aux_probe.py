"""`tul.core_token_aux` — the aux path against the shipped path on IDENTICAL rows, at eval.

The run-level reading (`tul/core_token_aux_ce` against `train/loss`) is confounded: the
shipped coda input carries the 0.15 token-state dropout (`TULSlots.apply_token_dropout`)
and the aux coda input does not, so a same-batch gap mixes "the tokens went through the
core" with "no dropout". This probe removes the confound: `model.eval()`, no dropout on
either path, the same packed rows, the same weighted CE reduction (`_tul_group_losses`).

  shipped  = tul_forward_ablated(normal)["loss"] - spandec_weighted   (train.py's val rule)
  aux      = _tul_core_token_aux(...)  — `_core_region` at eval runs cfg.mean_depth uniform,
             so the aux core runs the SAME depth 6 the slot loop is forced to.

Run it on the arm AND on a checkpoint that never trained the aux (the strict ruler): the
ruler's number is what the untrained token-core path costs, the arm's is what the trained
one gives. The gap between them is the aux objective's own work.

``--depths`` (2026-09-13) turns the probe into a DEPTH SWEEP of the aux path. The aux core
is the span-restricted token loop `tul.loop_reads_tokens` promotes to the shipped forward,
and the coretok checkpoint has already trained it — so forcing `model.cfg.mean_depth` and
re-scoring the SAME rows answers "does a span-restricted token loop earn depth at all"
from a checkpoint that exists, with no training. K1-K6 and K3-K6 come with a paired
bootstrap over BATCHES (the aux CE is a batch mean inside `_tul_core_token_aux`, not a
per-row map, so batches are the resampling unit; run `--batch 1` when the interval is the
headline).

THE SHIPPED PATH IS PINNED while the aux moves. `_sample_slot_depths` at eval reads
`tul.slot_mean_depth or model.mean_depth`, so moving `mean_depth` alone would drag the
SLOT loop's depth with it and the two effects would be inseparable. The probe resolves
`slot_mean_depth` once and writes it back, so `shipped_ce` is the same number at every
depth — which is also the check that the pinning worked.

Usage:
  python lab/divergence/core_token_aux_probe.py \
      --ckpt coretok=tul_slot_spandec_strict_coretok=/path/step_5000.pt \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt --rows 480 --out X.json
  python lab/divergence/core_token_aux_probe.py \
      --ckpt coretok=tul_slot_spandec_strict_coretok=/path/step_5000.pt \
      --depths 1,2,3,6 --rows 480 --batch 3 --out X.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader
from _stats import paired_bootstrap_ci

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

@torch.no_grad()
def probe_batch(model, inp, labels, layout, device):
    inp, labels, layout = inp.to(device), labels.to(device), layout.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp, labels, layout, plan_mode="normal")
        shipped = float(res["loss"]) - float(res.get("spandec_weighted", 0.0))
        n_ship = float(res["n_targets"])
        fkw, freset, ckw, creset = model._tul_tg_kwargs(layout)   # the ONE home
        x, x0, bg = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        st: dict = {}
        aux = float(model._tul_core_token_aux(x, x0, bg, inp, labels, layout, ckw, creset,
                                              stats=st))
    n_aux = st["core_token_aux_n"]
    assert abs(n_ship - n_aux) < 0.5, (n_ship, n_aux)
    return shipped, aux, n_ship


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--depths", default="",
                    help="comma-separated forced CORE depths for the AUX path "
                         "(model.cfg.mean_depth). Empty = the checkpoint's own depth, "
                         "which is what the 2026-09-12 filing ran.")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    out: dict = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        cfg = build_cfg(config, ["model.use_kernels=false"])
        tul_rt = build_tul_runtime(cfg)
        assert tul_rt is not None and not tul_rt.model_cfg.tokens_through_core
        model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
        model.eval()
        assert not model.training
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:-(-a.rows // a.batch)]
        tc = model.cfg.tul
        orig_mean, orig_slot = int(model.cfg.mean_depth), int(tc.slot_mean_depth)
        # PIN the shipped path at the depth it already runs, so only the AUX core moves.
        tc.slot_mean_depth = orig_slot or orig_mean
        depth_list = ([int(x) for x in a.depths.split(",")] if a.depths
                      else [orig_mean])
        per_depth: dict[int, dict] = {}
        aux_by_depth: dict[int, np.ndarray] = {}
        cnt_arr = None
        try:
            for d in depth_list:
                model.cfg.mean_depth = d
                ship_sum, aux_sum, cnt = [], [], []
                for i, (inp, labels, layout, _idx) in enumerate(batches):
                    s_, x_, n_ = probe_batch(model, inp, labels, layout, a.device)
                    ship_sum.append(s_ * n_); aux_sum.append(x_ * n_); cnt.append(n_)
                    if i % 20 == 0:
                        print(f"{label} d={d} batch {i}/{len(batches)} shipped={s_:.4f} "
                              f"aux={x_:.4f} n={n_:.0f}", flush=True)
                ship_sum, aux_sum, cnt = map(np.asarray, (ship_sum, aux_sum, cnt))
                ci = paired_bootstrap_ci(aux_sum, ship_sum, cnt)  # aux - shipped, paired
                per_depth[d] = {"shipped_ce": float(ship_sum.sum() / cnt.sum()),
                                "aux_ce": float(aux_sum.sum() / cnt.sum()),
                                "aux_minus_shipped": ci}
                aux_by_depth[d] = aux_sum
                if cnt_arr is None:
                    cnt_arr = cnt
                print(f"{label:10s} d={d} shipped={per_depth[d]['shipped_ce']:.4f} "
                      f"aux={per_depth[d]['aux_ce']:.4f} aux-shipped={ci}", flush=True)
        finally:
            model.cfg.mean_depth = orig_mean
            tc.slot_mean_depth = orig_slot
        ships = {d: per_depth[d]["shipped_ce"] for d in depth_list}
        assert max(ships.values()) - min(ships.values()) < 1e-6, (
            f"the SHIPPED path moved with the aux depth ({ships}) — `slot_mean_depth` was "
            f"not pinned and the two effects are inseparable")
        entry = {"step": step, "rows": int(sum(inp.shape[0] for inp, *_ in batches)),
                 "batch": a.batch, "eval_depth_core": orig_mean,
                 "eval_depth_slots": int(tc.slot_mean_depth or orig_mean),
                 "depths": {str(d): per_depth[d] for d in depth_list},
                 "n_targets": float(cnt_arr.sum())}
        # the whole point of --depths: the aux path's own K-curve, paired over batches
        entry["ci_aux_ce"] = {
            f"K{x}-{'K' + str(y)}": paired_bootstrap_ci(aux_by_depth[x], aux_by_depth[y],
                                                        cnt_arr)
            for x, y in ((1, 6), (3, 6), (1, max(depth_list)))
            if x in aux_by_depth and y in aux_by_depth and x != y}
        # back-compatible top-level keys: the checkpoint's own depth, the 2026-09-12 columns
        _d0 = orig_mean if orig_mean in per_depth else depth_list[-1]
        entry.update({k: per_depth[_d0][k]
                      for k in ("shipped_ce", "aux_ce", "aux_minus_shipped")})
        out[label] = entry
        for k, v in entry["ci_aux_ce"].items():
            print(f"{label:10s} aux_ce {k}: {v['point']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} batches", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
