"""Attention-vs-GLA branch contribution probe (plain no-TUL path).

Opus GLA-map F5 + the BG0 stall (notul-bg0 flatlined at unigram CE ~7.4 with
retention=false): if attention is quietly dead and the gated GLA branch carries
the model, removing GLA leaves nothing trainable — and a loop over a linear
summarizer earning no depth would explain every flat depth curve.

For every MORPHBlock, on real validation rows, record the RMS over [B,T] of:
  attn_out = attention(norm_attn(x))          (pre-gate, pre-drop)
  gla_out  = retention(norm_ret(x))[0]        (raw branch)
  gated    = sigmoid(ret_gate) * gla_out      (what actually enters the sum)
plus sigmoid(ret_gate). Hooks capture module outputs; RMS is per-element root
mean square so branches are comparable across shapes.

Usage:
  python lab/divergence/branch_norm_probe.py \
    --ckpt notul=notul_l2=checkpoints/morph/notul-l2nc/step_4500.pt \
    --rows 6 --device cpu --out /tmp/branch_norms.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def rms(t: torch.Tensor) -> float:
    return float(t.detach().float().pow(2).mean().sqrt())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        if build_tul_runtime(cfg) is not None:
            print(f"REFUSE {label}: TUL active; this is the plain-path probe")
            sys.exit(1)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, None)
        model.eval()

        stats: dict[str, dict[str, list[float]]] = {}
        hooks = []

        def add(name: str, block) -> None:
            ent = stats.setdefault(name, {"attn": [], "gla": [], "gated": [], "gate": []})

            def attn_hook(_m, _i, out, ent=ent):
                ent["attn"].append(rms(out))

            hooks.append(block.attention.register_forward_hook(attn_hook))
            if block.retention is not None:
                g = float(torch.sigmoid(block.ret_gate.detach().float()).mean())

                def ret_hook(_m, _i, out, ent=ent, g=g):
                    o = out[0] if isinstance(out, tuple) else out
                    ent["gla"].append(rms(o))
                    ent["gated"].append(g * rms(o))
                    ent["gate"].append(g)

                hooks.append(block.retention.register_forward_hook(ret_hook))

        for gname in ("prelude", "core", "coda"):
            for i, blk in enumerate(getattr(model, gname)):
                add(f"{gname}.{i}", blk)

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                   cfg.data.seq_len, a.batch,
                                   split="validation", skip_samples=0, bag_size=0,
                                   tul=None)
        done = 0
        with torch.no_grad():
            while done < a.rows:
                x, y = next(loader)[:2]
                if device == "cuda":
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        model(x.to(device), labels=y.to(device))
                else:
                    model(x.to(device), labels=y.to(device))
                done += a.batch
        for h in hooks:
            h.remove()

        arm = {"step": step, "rows": done, "blocks": {}}
        print(f"== {label} step={step} rows={done}")
        for name, ent in stats.items():
            row = {k: (sum(v) / len(v) if v else None) for k, v in ent.items()}
            arm["blocks"][name] = row
            if row["gla"] is not None:
                ratio = row["gated"] / row["attn"] if row["attn"] else float("inf")
                print(f"{name:10s} attn_rms={row['attn']:.4f}  gla_rms={row['gla']:.4f}  "
                      f"gate={row['gate']:.4f}  gated_rms={row['gated']:.4f}  "
                      f"gated/attn={ratio:.3f}  (n={len(ent['attn'])})")
            else:
                print(f"{name:10s} attn_rms={row['attn']:.4f}  (no retention)")
        results[label] = arm
        del model

    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
