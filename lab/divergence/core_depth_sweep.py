"""Core-contribution depth sweep: token CE as a function of forced loop depth.

The per-iteration probe (probe.jsonl loop/delta_ratio_t*) shows every ladder arm's
loop keeps MOVING the slot state after t0 (~0.2-0.5 of its norm per iteration), but
norm motion is not nats. This measures what the iterations are WORTH: eval the same
packed rows at slot depth d = 1..max and read per-token CE (all token positions, and
span-first tokens separately — the positions the plan serves).

Depth forcing: in eval, `_sample_slot_depths` is the deterministic
`tul.slot_mean_depth or model.mean_depth` for every valid slot, so setting
`model.cfg.tul.slot_mean_depth = d` between evals forces depth d exactly (the
runtime-dispatch pattern tul_forward_ablated already uses for wrong_seed).

Rows are packed ONCE and reused for every depth -> the CE(d) curve is exactly paired.

Usage:
  python lab/divergence/core_depth_sweep.py \
    --ckpt l3=tul_l3=checkpoints/morph/tul-l3/step_4500.pt \
    --depths 1,2,3,4,5,6,7,8 --rows 48 --out .../depth_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def ce_maps(model, inp, layout, labels, device, step_mode=None) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        tul_step_mode=step_mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-mode", default="auto", choices=["auto", "force-loop"],
                    help="auto: sigma-conditioned models use the Euler ladder "
                         "(db1_ladder_steps=d). force-loop: run the plain _tul_core "
                         "loop at forced depth d even on a sigma model (measures the "
                         "bptt-trained loop of a step_mix arm; tul_step_mode='bptt' "
                         "at eval opts out of the auto-ladder).")
    a = ap.parse_args()
    device = a.device
    depths = [int(x) for x in a.depths.split(",")]

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        # pack ALL rows once — every depth sees identical batches (paired curve)
        buf: list[int] = []
        need = a.batch * (spec.l_total + 1)
        batches = []
        rows_done = 0
        while rows_done < a.rows:
            while len(buf) < need:
                buf.extend(next(loader)[0].reshape(-1).tolist())
            inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
            batches.append((inp, labels, layout.to(device)))
            rows_done += a.batch
        # scoreable masks (token pos, label valid) + span-first flags, once
        masks = []
        dump = spec.max_slots
        for inp, labels, layout in batches:
            tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
            first = torch.zeros_like(tokpos)
            for b in range(inp.shape[0]):
                seen: set[int] = set()
                for p in range(inp.shape[1]):
                    if not bool(tokpos[b, p]):
                        continue
                    bag = int(layout.bag_id[b, p])
                    if bag not in seen:
                        seen.add(bag)
                        if 0 < bag < dump:
                            first[b, p] = True
            masks.append((tokpos, first))
        orig_mean = int(model.cfg.tul.slot_mean_depth)
        orig_max = int(model.cfg.tul.slot_max_depth)
        arm = {"step": step, "rows": rows_done, "eval_mode": a.eval_mode,
               "train_eval_depth":
               orig_mean or int(cfg.model.mean_depth), "depths": {}}
        _sigma = (getattr(model.cfg.tul, "core_stage_cond", "none") == "sigma"
                  and a.eval_mode == "auto")
        _step_mode = "bptt" if a.eval_mode == "force-loop" else None
        orig_ladder = int(getattr(model.cfg.tul, "db1_ladder_steps", 0))
        try:
            for d in depths:
                if _sigma:
                    # sigma-conditioned (db1) models: eval depth = Euler-ladder steps K
                    # (transformer.py: K = k_steps or cfg.tul.db1_ladder_steps or mean_depth);
                    # slot_mean_depth is ignored by that path.
                    model.cfg.tul.db1_ladder_steps = d
                model.cfg.tul.slot_mean_depth = d
                model.cfg.tul.slot_max_depth = max(d, orig_max or int(cfg.model.max_depth))
                tot = tot_n = fst = fst_n = 0.0
                for (inp, labels, layout), (tokpos, first) in zip(batches, masks):
                    ce = ce_maps(model, inp, layout, labels, device,
                                 step_mode=_step_mode).cpu()
                    tot += float(ce[tokpos].sum()); tot_n += int(tokpos.sum())
                    fst += float(ce[first].sum()); fst_n += int(first.sum())
                arm["depths"][d] = {"ce_tokens": tot / tot_n, "ce_span_first": fst / fst_n,
                                    "n_tokens": tot_n, "n_first": fst_n}
                print(f"{label:10s} depth={d}  ce={tot/tot_n:.4f}  "
                      f"span_first={fst/fst_n:.4f}", flush=True)
        finally:
            model.cfg.tul.slot_mean_depth = orig_mean
            model.cfg.tul.slot_max_depth = orig_max
            if _sigma:
                model.cfg.tul.db1_ladder_steps = orig_ladder
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
