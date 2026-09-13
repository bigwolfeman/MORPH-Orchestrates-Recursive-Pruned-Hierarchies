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

Usage:
  python lab/divergence/core_token_aux_probe.py \
      --ckpt coretok=tul_slot_spandec_strict_coretok=/path/step_5000.pt \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt --rows 480 --out X.json
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

from morph.model import transformer as T  # noqa: E402


def _coda_kwargs(model, layout):
    """The coda relation and the retention reset, built the way `_forward_tul` builds them."""
    tc = model.cfg.tul
    if model._tg_strict:
        seg = T.tg_segment_ids(layout)
        allow = T.tg_strict_allow(layout, "coda", coda_prefix_reach=tc.tg_coda_prefix_reach)
        kw = {"tg_allow": allow, "tg_slot_mask": layout.slot_mask,
              "tg_comp_allow": allow, "tg_seg": seg}
        return kw, T.tg_reset_from_ids(seg)
    raise NotImplementedError("this probe is written for the strict geometry only; "
                              "add the tg_restrict branch of _forward_tul before using it "
                              "on another arm")


def _front_kwargs(model, layout):
    seg = T.tg_segment_ids(layout)
    pre = T.tg_strict_allow(layout, "prelude")
    return ({"tg_allow": pre, "tg_slot_mask": layout.slot_mask,
             "tg_comp_allow": pre, "tg_seg": seg}, T.tg_reset_from_ids(seg))


@torch.no_grad()
def probe_batch(model, inp, labels, layout, device):
    inp, labels, layout = inp.to(device), labels.to(device), layout.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp, labels, layout, plan_mode="normal")
        shipped = float(res["loss"]) - float(res.get("spandec_weighted", 0.0))
        n_ship = float(res["n_targets"])
        fkw, freset = _front_kwargs(model, layout)
        x, x0, bg = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        ckw, creset = _coda_kwargs(model, layout)
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
        ship_sum, aux_sum, cnt = [], [], []
        for i, (inp, labels, layout, _idx) in enumerate(batches):
            s, x, n = probe_batch(model, inp, labels, layout, a.device)
            ship_sum.append(s * n); aux_sum.append(x * n); cnt.append(n)
            if i % 20 == 0:
                print(f"{label} batch {i}/{len(batches)} shipped={s:.4f} aux={x:.4f} n={n:.0f}",
                      flush=True)
        ship_sum, aux_sum, cnt = map(np.asarray, (ship_sum, aux_sum, cnt))
        ci = paired_bootstrap_ci(aux_sum, ship_sum, cnt)   # aux - shipped, per-batch paired
        entry = {"step": step, "rows": int(sum(inp.shape[0] for inp, *_ in batches)),
                 "batch": a.batch, "eval_depth_core": int(model.cfg.mean_depth),
                 "eval_depth_slots": int(model.cfg.tul.slot_mean_depth or model.cfg.mean_depth),
                 "shipped_ce": float(ship_sum.sum() / cnt.sum()),
                 "aux_ce": float(aux_sum.sum() / cnt.sum()),
                 "aux_minus_shipped": ci, "n_targets": float(cnt.sum())}
        out[label] = entry
        print(f"{label:10s} step={step} shipped={entry['shipped_ce']:.4f} "
              f"aux={entry['aux_ce']:.4f} aux-shipped={ci}", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
