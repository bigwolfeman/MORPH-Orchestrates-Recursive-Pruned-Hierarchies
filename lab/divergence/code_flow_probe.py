"""The thinker's flow loss on a TUL-Code checkpoint, measured directly (no trainer log needed).

Runs the training-mode code branch (phase 2: one thinker pass on the CFM interpolant per
slot) under no_grad on validation rows, with every dropout off, and reports the flow loss
as a share of the null floor overall and per t-band — the same `code_fm_rel` /
`code_fm_band{0..3}_rel` the trainer logs. Reads: is the thinker still learning, and at
which end of the path (band 0 = the noise end, where the conditioning shows).

    python lab/divergence/code_flow_probe.py \
        --ckpt tul-code-thinker=tul_code_thinker=/path/step_20000.pt --rows 96 --out .../flow.json

Read-only. Same validation stream and packer as core_depth_sweep.py.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

KEYS = ("code_fm_rel", "code_fm_band0_rel", "code_fm_band1_rel", "code_fm_band2_rel",
        "code_fm_band3_rel", "code_fm_raw", "code_fm_null")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=k=v,k=v]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--repeats", type=int, default=2, help="passes over the rows (fresh t, z0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    # every dropout off: the probe wants the flow loss of the weights, not of a noisy pass
    cfg = build_cfg(config, ["model.use_kernels=false", "model.dropout=0.0",
                             "tul.token_state_dropout=0.0", *ovr])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False)), \
        "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.train()                       # the flow loss lives on the training branch
    model.code_phase = 2                # one thinker pass, no rollout
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    n_rows = sum(inp.shape[0] for inp, _, _, _ in batches)
    print(f"{label}: step {step}, {n_rows} rows x {a.repeats} passes, overrides={ovr}", flush=True)

    torch.manual_seed(a.seed)
    acc: dict[str, list[float]] = {k: [] for k in KEYS}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        for _ in range(a.repeats):
            for inp, labels, lay, _ in batches:
                out = model(inp.to(device), labels.to(device), slot_layout=lay.to(device))
                for k in KEYS:
                    if k in out and out[k] is not None:
                        acc[k].append(float(out[k]))
    res = {"label": label, "step": step, "rows": n_rows, "repeats": a.repeats,
           "mean": {k: float(np.mean(v)) for k, v in acc.items() if v},
           "sem": {k: float(np.std(v) / max(len(v), 1) ** 0.5) for k, v in acc.items() if v}}
    for k in ("code_fm_rel", "code_fm_band0_rel", "code_fm_band1_rel", "code_fm_band2_rel",
              "code_fm_band3_rel"):
        if k in res["mean"]:
            print(f"  {k:<20} {res['mean'][k]:.4f} ± {res['sem'][k]:.4f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
