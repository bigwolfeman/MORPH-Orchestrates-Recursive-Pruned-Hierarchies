"""Per-row cotangent growth through the slot loop, on captured batches, to pick
`model.slot_cot_clip` before the prereg freezes it.

    python lab/divergence/cot_calibrate.py --ckpt checkpoints/morph/cap-c1-det/ROLL_step_1150.pt \
        --batches lab/experiments/results/2026-09-03-tul-onset-capture/batches/batch_001155.pt ...

Loads the checkpoint eager (deterministic settings as the trainer), replays each dumped
batch (token ids, labels, slot layout of one training step of C1) through one labelled
forward+backward with the clip armed at a ratio that never binds (1e9), so the hooks record
the per-row cotangent norm at every grad iteration and the per-row exit reference, and prints
rows[t] / ref per iteration: the max over rows, the median, and the global ratio. The
in-run probe only ever saw the global norm; the clip acts per row, so the cap has to be
read off the per-row numbers.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_layout import SlotLayout  # noqa: E402
from morph.training.tul_setup import build_tul_runtime  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_cap_c1")
    ap.add_argument("--batches", nargs="+", required=True)
    a = ap.parse_args()
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("export CUBLAS_WORKSPACE_CONFIG=:4096:8 first")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    cfg = build_cfg(a.config, ["model.use_kernels=false", "training.compile=false",
                               "model.slot_cot_clip=1e9"])
    tul_rt = build_tul_runtime(cfg)
    model, step = load_ckpt(cfg, a.ckpt, "cuda", tul_rt.model_cfg)
    model.train()
    root = getattr(model, "_orig_mod", model)
    root._probe_loop = True
    root._probe_cot = True
    print(f"checkpoint step {step}; clip armed at 1e9 (records, never binds)")
    for bp in a.batches:
        d = torch.load(bp, map_location="cuda", weights_only=False)
        layout = SlotLayout(**{k: (v.cuda() if torch.is_tensor(v) else v) for k, v in d["layout"].items()})
        torch.manual_seed(1234)
        model.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(d["x"].cuda(), labels=d["y"].cuda(), bag_size=0, slot_layout=layout)
        out["loss"].backward()
        ref = root._loop_cot_ref
        ts = sorted(root._loop_cot_rows)
        print(f"\nbatch step {d['step']}: loss {float(out['loss']):.4f}  exit ref per row "
              + " ".join(f"{v:.1f}" for v in ref.tolist()))
        print("  iter  max_row_ratio  med_row_ratio  global_ratio")
        for t in ts:
            rows = root._loop_cot_rows[t]
            ratio = rows / (ref + 1e-12)
            g = float(root._loop_cot[t]) / (float(ref.norm()) + 1e-12)
            print(f"  {t:4d}  {float(ratio.max()):13.2f}  {float(ratio.median()):13.2f}  {g:12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
