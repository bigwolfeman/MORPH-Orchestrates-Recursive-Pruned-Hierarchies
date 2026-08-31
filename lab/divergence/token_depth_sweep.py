"""Token-path depth sweep: plain-forward CE as a function of forced loop depth.

The no-TUL twin of core_depth_sweep.py, for arms with `tul.activate_at=never`
(prereg 2026-08-30-tul-vs-notul-30k.md, arm B). No slots exist, so the TUL
sweep's slot_mean_depth lever is inert; the plain forward's eval depth is a
uniform `cfg.mean_depth` fill (transformer.py, the `else` of the
`if self.training` depth branch), so mutating `model.cfg.mean_depth` between
evals forces depth d exactly. Rows are drawn ONCE and reused for every depth,
so the CE(d) curve is exactly paired, like the TUL sweep.

Usage:
  python lab/divergence/token_depth_sweep.py \
    --ckpt notul=tul_l2=checkpoints/morph/notul-30k/step_30000.pt=tul.activate_at=never \
    --depths 1,2,3,4,5,6,7,8 --rows 48 --out .../token_depth_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def batch_ce(model, x, y, device) -> float:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(x.to(device), labels=y.to(device))
    return float(out["loss"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2] (overrides land in build_cfg; "
                         "pass tul.activate_at=never for the no-TUL arm)")
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    depths = [int(x) for x in a.depths.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is not None:
            # A TUL layout would route eval through the slot path and this sweep's
            # mean_depth lever would be the WRONG lever (slots read slot_mean_depth).
            print(f"REFUSE {label}: TUL is active in this config — use "
                  f"core_depth_sweep.py, or add tul.activate_at=never")
            sys.exit(1)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, None)
        model.eval()
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                   cfg.data.seq_len, a.batch,
                                   split="validation", skip_samples=0, bag_size=0,
                                   tul=None)
        batches = []
        while len(batches) * a.batch < a.rows:
            x, y = next(loader)[:2]
            batches.append((x, y))
        orig_mean = int(model.cfg.mean_depth)
        arm = {"step": step, "rows": len(batches) * a.batch,
               "train_eval_depth": orig_mean, "depths": {}}
        try:
            for d in depths:
                model.cfg.mean_depth = d
                ces = [batch_ce(model, x, y, device) for x, y in batches]
                ce = sum(ces) / len(ces)
                arm["depths"][d] = {"ce_tokens": ce,
                                    "n_batches": len(ces), "batch": a.batch}
                print(f"{label:10s} depth={d}  ce={ce:.4f}", flush=True)
        finally:
            model.cfg.mean_depth = orig_mean
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
