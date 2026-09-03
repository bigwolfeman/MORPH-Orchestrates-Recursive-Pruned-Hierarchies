"""The four-condition plan-worth scorer for dm-hs (design note, "Arm dm-hs"; prereg P4).

On one checkpoint and one fixed val set, the CE at the slot's EMIT position (the first
token of the NEXT span) read through the tied head from four slot states:

| condition | the state at the emit position                                         |
|-----------|------------------------------------------------------------------------|
| clean     | the clean target itself, ``normalize(h_final)``, through the same read-out |
| ladder    | the B-step Euler ladder's last ``D̂`` (what inference can actually produce) |
| zero      | nothing (the read-out of a zero state: ~ln V)                          |
| shuffle   | another row's slot ``s`` state at the same slot index                  |

Why the emit position and not "the next span's tokens": in a FLAT stack the post-stack
state at a slot is read by exactly one consumer, the head at that position — the coda
never sees it (it is the coda's OUTPUT). So the next span's tokens beyond the first do
not read this state at all, and the worth protocol of ``lab/divergence/slot_path_worth.py``
(which substituted the state the coda READS) reduces to the emit-position CE here. The
design note's Implementation record names this. Report the COST relative to ``clean``,
never a fraction (docs/tul-fm-probing.md §4 rule 1), with a bootstrap CI over batches.

The numbers come from the SHIPPED eval path — ``model(x, labels, slot_layout)`` in eval
mode emits ``dm_worth_*`` from ``morph.model.dmorph.eval_terms`` — so this script adds no
second implementation of the substitution (rule 9: a probe must run on the shipped path).

    python lab/dmorph/worth_scorer.py --ckpt checkpoints/morph/dmorph-hs/step_20000.pt \\
        --config dmorph_hs --batches 80 --out lab/experiments/results/<slug>/worth_hs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lab.divergence._build import build_cfg, build_model            # noqa: E402
from morph.training.data import create_dataloader                   # noqa: E402
from morph.training.train import load_checkpoint                    # noqa: E402

CONDITIONS = ("clean", "ladder", "zero", "shuffle")


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = np.array([rng.choice(x, size=len(x), replace=True).mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="dmorph_hs")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = build_cfg(a.config, a.override)
    if not bool(cfg.dmorph.enabled) or str(cfg.dmorph.arm) != "hs":
        raise SystemExit("worth_scorer scores the hs arm: pass a config with dmorph.arm=hs")
    dev = torch.device(a.device)
    model, tul_rt = build_model(cfg, device=a.device)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, dev)
    model.eval()

    loader = iter(create_dataloader(
        str(cfg.data.tokenizer), str(cfg.data.dataset), int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=50_000,
        tul=tul_rt.data_cfg))
    per: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    extra: dict[str, list[float]] = {"cos": [], "ladder_acc": [], "ce_emit_clean_head": []}
    for i in range(a.batches):
        x, y, layout = next(loader)
        x, y, layout = x.to(dev), y.to(dev), layout.to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                            enabled=(dev.type == "cuda")):
            out = model(x, labels=y, slot_layout=layout)
        for c in CONDITIONS:
            per[c].append(float(out[f"dm_worth_{c}"]))
        extra["cos"].append(float(out["dm_cos"]))
        extra["ladder_acc"].append(float(out["dm_ladder_acc"]))
        extra["ce_emit_clean_head"].append(float(out["ce_emit"]))
        print(f"  batch {i:3d}  " + "  ".join(f"{c}={per[c][-1]:.4f}" for c in CONDITIONS)
              + f"  cos={extra['cos'][-1]:.3f}", flush=True)

    res: dict = {"ckpt": a.ckpt, "config": a.config, "overrides": a.override,
                 "batches": a.batches}
    clean = np.array(per["clean"])
    for c in CONDITIONS:
        arr = np.array(per[c])
        res[c] = {"mean": float(arr.mean()), "ci95": bootstrap_ci(arr)}
        if c != "clean":
            cost = arr - clean
            res[f"cost_{c}"] = {"mean": float(cost.mean()), "ci95": bootstrap_ci(cost)}
    for k, v in extra.items():
        res[k] = {"mean": float(np.mean(v)), "ci95": bootstrap_ci(np.array(v))}
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
