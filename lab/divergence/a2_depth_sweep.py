"""A2 (the paid loop, every TUL model since 2026-09-03) depth sweep: token CE vs forced PER-SAMPLE core depth.

A2 runs the ordinary per-sample core over the whole packed TUL sequence, so its
eval depth lever is `model.cfg.mean_depth` (the `else` of `_core_region`'s
`if self.training` branch) — token_depth_sweep's lever — but its batches are
packed TUL rows with a slot layout — core_depth_sweep's packing. This script is
that hybrid; the JSON schema matches core_depth_sweep (ce_tokens +
ce_span_first over identical paired batches) so the ladder tables compare
directly. token_depth_sweep REFUSES TUL configs by design; do not weaken that —
this script exists instead.

Usage:
  python lab/divergence/a2_depth_sweep.py \
    --ckpt a2=tul_a2=checkpoints/morph/tul-a2/step_5000.pt \
    --rows 48 --out .../a2_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from _build import ROOT, DepthLever, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def ce_maps(model, inp, layout, labels, device) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model(inp.to(device), labels=None, slot_layout=layout)
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
        if tul_rt is None:
            print(f"REFUSE {label}: {config} has tul.activate_at never — a plain model; use "
                  f"token_depth_sweep.py (notul). Every TUL model is the paid loop since "
                  f"2026-09-03, so no other check is needed here.")
            sys.exit(1)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg)
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
        lever = DepthLever(model, tul_rt, int(cfg.model.max_depth))
        assert lever.a2, "the REFUSE above guarantees an A2 config"
        arm = {"step": step, "rows": rows_done, "eval_mode": "a2_model_depth",
               "depth_lever": lever.name,
               "train_eval_depth": int(model.cfg.mean_depth), "depths": {}}
        try:
            for d in depths:
                lever.set(d)
                tot = tot_n = fir = fir_n = 0.0
                for (inp, labels, layout), (tokpos, first) in zip(batches, masks):
                    ce = ce_maps(model, inp, layout, labels, device).cpu()
                    tot += float(ce[tokpos].sum())
                    tot_n += float(tokpos.sum())
                    fir += float(ce[first].sum())
                    fir_n += float(first.sum())
                arm["depths"][d] = {
                    "ce_tokens": tot / tot_n, "ce_span_first": fir / fir_n,
                    "n_tokens": tot_n, "n_first": fir_n}
                print(f"{label:10s} depth={d}  ce_tokens={tot/tot_n:.4f} "
                      f"span_first={fir/fir_n:.4f}", flush=True)
        finally:
            lever.restore()
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
