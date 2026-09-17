"""Does the past carry ANYTHING about a span's code? A thinker-free reading.

For a code model (LCTUL or LCTUL-D) run the encoder on validation rows and compare the code
of slot s with the code of slot s-1 (the tape's newest entry) and with a random slot of the
batch: cosine per cell, and on a discrete code the per-position symbol match rate. If the
consecutive-pair statistics equal the random-pair ones, the codes have no temporal
structure and NO thinker can be conditional on the tape; the context-share reading is then
a property of the code, not of the thinker.

    python lab/divergence/code_continuity_probe.py \
        --ckpt tul-code-d=tul_code_d=/path/step_5000.pt --rows 96 --out .../continuity.json

Read-only. Same validation stream and packer as core_depth_sweep.py.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_code import code_target_valid  # noqa: E402


def _boot(vals: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    m = float(vals.mean())
    bs = [float(vals[rng.integers(0, len(vals), len(vals))].mean()) for _ in range(n)]
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=k=v,k=v]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lags", default="1,2,4")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    lags = [int(x) for x in a.lags.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(getattr(tul_rt.model_cfg, "code", False))
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    discrete = getattr(model, "tul_code_vq", None) is not None
    seen: dict = {}
    if discrete:
        real = model.tul_code_vq.forward

        def spy(z, ok, sub_index=None):
            cells, deq, out = real(z, ok, sub_index=sub_index)
            seen["index"] = out["index"].clone()
            return cells, deq, out
        model.tul_code_vq.forward = spy

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    print(f"{label}: step {step}, discrete={discrete}, lags={lags}", flush=True)

    cos_lag = {lag: [] for lag in lags}
    cos_rand: list[float] = []
    match_lag = {lag: [] for lag in lags}
    match_rand: list[float] = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for inp, _labels, lay, _ in batches:
            lay = lay.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                res = model.tul_forward_ablated(inp.to(device), None, lay, plan_mode="normal",
                                                code_mode="encoder")
            cells = res["code_cells"].float()                        # [B, S, M, C]
            ok = code_target_valid(lay)                               # [B, S]
            B, S, M, C = cells.shape
            cn = F.normalize(cells, dim=-1)
            idx = seen.get("index") if discrete else None            # [B, S, K, G]
            valid = ok.nonzero().tolist()
            for lag in lags:
                for b, s in valid:
                    if s - lag >= 0 and bool(ok[b, s - lag]):
                        cos_lag[lag].append(float((cn[b, s] * cn[b, s - lag]).sum(-1).mean()))
                        if idx is not None:
                            match_lag[lag].append(float((idx[b, s] == idx[b, s - lag]).float().mean()))
            # random pairs: a valid slot against a valid slot of ANOTHER row
            for b, s in valid:
                others = [(b2, s2) for b2, s2 in valid if b2 != b]
                if not others:
                    continue
                b2, s2 = others[rng.integers(0, len(others))]
                cos_rand.append(float((cn[b, s] * cn[b2, s2]).sum(-1).mean()))
                if idx is not None:
                    match_rand.append(float((idx[b, s] == idx[b2, s2]).float().mean()))
    out = {"label": label, "step": step, "discrete": discrete, "n_random": len(cos_rand),
           "cos_random": _boot(np.array(cos_rand)), "cos_lag": {}, "match_lag": {},
           "match_random": _boot(np.array(match_rand)) if match_rand else None}
    print(f"  cos random pair      {out['cos_random'][0]:+.4f} [{out['cos_random'][1]:+.4f}, {out['cos_random'][2]:+.4f}]  n={len(cos_rand)}")
    for lag in lags:
        out["cos_lag"][lag] = _boot(np.array(cos_lag[lag]))
        m, lo, hi = out["cos_lag"][lag]
        print(f"  cos lag {lag:<2d}           {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  n={len(cos_lag[lag])}")
    if match_rand:
        m, lo, hi = out["match_random"]
        print(f"  symbol match random  {m:.4f} [{lo:.4f}, {hi:.4f}]  (chance ~ 1/perplexity)")
        for lag in lags:
            out["match_lag"][lag] = _boot(np.array(match_lag[lag]))
            m, lo, hi = out["match_lag"][lag]
            print(f"  symbol match lag {lag:<2d}  {m:.4f} [{lo:.4f}, {hi:.4f}]")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
