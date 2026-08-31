"""Future-leak vs past-memory attribution probe.

Prereg: lab/experiments/planned/2026-08-31-future-leak-attribution.md.

Corrupt INPUT ids at token positions AFTER packed index k; score CE only at
token positions BEFORE k (labels there are clean). Every attention branch is
causal at scored positions, so the only path from the corrupted region into a
scored position is the acausal retention carry. If forced-depth earning
(CE@K1 - CE@K6 on the scored positions) survives corruption, the carry was
serving PAST-side memory (H-memory); if it collapses, it was reading the
FUTURE (H-leak).

Usage:
  python lab/divergence/future_leak_probe.py \
    --ckpt l2cap=tul_l2=checkpoints/morph/tul-l2-cap/step_4500.pt=model.retention_carry=acausal_final \
    --ckpt l2nc=tul_l2=checkpoints/morph/tul-l2nc/step_4500.pt \
    --ks 700,900 --depths 1,3,6 --rows 48 --out .../future_leak_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def ce_map(model, inp, layout, labels, device) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal")
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--ks", default="700,900")
    ap.add_argument("--depths", default="1,3,6")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    ks = [int(x) for x in a.ks.split(",")]
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
        carry_mode = model.cfg.retention_carry_mode
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0,
                                   tul=None)
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
        vocab = int(model.cfg.vocab_size)
        orig_mean = int(model.cfg.tul.slot_mean_depth)
        orig_max = int(model.cfg.tul.slot_max_depth)
        arm = {"step": step, "rows": rows_done, "carry_mode": carry_mode, "ks": {}}
        try:
            for k in ks:
                # scored mask: token positions with packed index < k and a valid label
                pos = torch.arange(spec.l_total)
                cells: dict[str, dict] = {"clean": {}, "corrupt": {}}
                for bi, (inp, labels, layout) in enumerate(batches):
                    tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
                    scored = tokpos & (pos < k).unsqueeze(0)
                    # corrupted twin: random non-special ids at token positions > k
                    g = torch.Generator().manual_seed(1000 + bi)
                    rand = torch.randint(100, vocab, inp.shape, generator=g)
                    after = (~layout.slot_mask.cpu()) & (pos > k).unsqueeze(0)
                    inp_c = torch.where(after, rand, inp)
                    for d in depths:
                        model.cfg.tul.slot_mean_depth = d
                        model.cfg.tul.slot_max_depth = max(
                            d, orig_max or int(cfg.model.max_depth))
                        for cond, x in (("clean", inp), ("corrupt", inp_c)):
                            ce = ce_map(model, x, layout, labels, device)
                            cell = cells[cond].setdefault(d, [0.0, 0])
                            cell[0] += float(ce[scored].sum())
                            cell[1] += int(scored.sum())
                out_k = {}
                for cond in ("clean", "corrupt"):
                    out_k[cond] = {d: cells[cond][d][0] / cells[cond][d][1]
                                   for d in depths}
                    e = out_k[cond][depths[0]] - out_k[cond][depths[-1]]
                    out_k[f"earning_{cond}"] = e
                out_k["n_scored"] = cells["clean"][depths[0]][1]
                arm["ks"][k] = out_k
                print(f"{label:8s} k={k}  " + "  ".join(
                    f"{c}: " + " ".join(f"K{d}={out_k[c][d]:.4f}" for d in depths)
                    for c in ("clean", "corrupt")))
                print(f"{'':8s} earning clean {out_k['earning_clean']:+.4f}  "
                      f"corrupt {out_k['earning_corrupt']:+.4f}  "
                      f"(n={out_k['n_scored']})", flush=True)
        finally:
            model.cfg.tul.slot_mean_depth = orig_mean
            model.cfg.tul.slot_max_depth = orig_max
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
