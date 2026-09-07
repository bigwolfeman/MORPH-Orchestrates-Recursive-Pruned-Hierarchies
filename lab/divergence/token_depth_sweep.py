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
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _earning import EarningProfile, offsets_from_ids

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def batch_ce(model, x, y, device) -> float:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(x.to(device), labels=y.to(device))
    return float(out["loss"])


@torch.no_grad()
def ce_map(model, x, y, device) -> torch.Tensor:
    """[B, L] per-token CE from the full logits (the eager head; use_kernels=false)."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model(x.to(device), labels=None)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = y.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L)


@torch.no_grad()
def ce_map_heads(model, x, y, device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Arc E8: per-token CE of each lookahead head j = 2..k against labels shifted by
    j-1 (tail -100), as ``[(ce [B, L], valid [B, L]), ...]``. Empty on a 1-head model."""
    if getattr(model, "mtp", None) is None:
        return []
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model(x.to(device), labels=None)
    outs = []
    for j, lg in enumerate(res["mtp_logits"], start=2):
        lg = lg.float()
        B, L, V = lg.shape
        lab = F.pad(y.to(device)[:, j - 1:], (0, j - 1), value=-100)
        valid = lab >= 0
        lab = lab.clone()
        lab[~valid] = 0
        ce = F.cross_entropy(lg.reshape(B * L, V), lab.reshape(B * L),
                             reduction="none").reshape(B, L)
        outs.append((ce, valid))
    return outs


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
    ap.add_argument("--profile", action="store_true",
                    help="arc E0: also write per-row and per-offset-in-span CE sums per "
                         "depth (spans cut by the boundary rule on the input ids), so "
                         "score_arc_e0.py can read where the loop earns")
    a = ap.parse_args()
    device = a.device
    depths = [int(x) for x in a.depths.split(",")]

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_boundary_rule, build_tul_runtime

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
        n_rows = len(batches) * a.batch
        arm = {"step": step, "rows": n_rows,
               "train_eval_depth": orig_mean, "depths": {}}
        n_heads = len(model.mtp) if getattr(model, "mtp", None) is not None else 0
        if n_heads:
            # Arc E8: the lookahead heads' CE(depth), per head, with per-row sums so the
            # scorer can pair them (heads[j]["row_ce_sum"][d][row], row_n_tokens).
            arm["heads"] = {str(j): {"depths": {}, "row_ce_sum": {}, "row_n_tokens": {}}
                            for j in range(2, n_heads + 2)}
        prof = None
        if a.profile:
            rule = build_boundary_rule(cfg)[0]
            offs = [[offsets_from_ids(x[b].numpy(), rule) for b in range(x.shape[0])]
                    for x, _ in batches]
            prof = EarningProfile(depths, n_rows)
        try:
            for d in depths:
                model.cfg.mean_depth = d
                if prof is None:
                    ces = [batch_ce(model, x, y, device) for x, y in batches]
                    ce = sum(ces) / len(ces)
                    arm["depths"][d] = {"ce_tokens": ce,
                                        "n_batches": len(ces), "batch": a.batch}
                else:
                    tot = tot_n = 0.0
                    for i, (x, y) in enumerate(batches):
                        ce_b = ce_map(model, x, y, device)
                        valid = (y >= 0)
                        for b in range(x.shape[0]):
                            prof.add(d, i * a.batch + b, ce_b[b], valid[b], offs[i][b])
                        tot += float(ce_b.cpu()[valid].sum())
                        tot_n += float(valid.sum())
                    ce = tot / tot_n
                    arm["depths"][d] = {"ce_tokens": ce, "n_tokens": tot_n,
                                        "n_batches": len(batches), "batch": a.batch}
                if n_heads:
                    hsum = {j: 0.0 for j in range(2, n_heads + 2)}
                    hn = {j: 0.0 for j in range(2, n_heads + 2)}
                    for i, (x, y) in enumerate(batches):
                        for j, (ce_h, valid_h) in enumerate(ce_map_heads(model, x, y, device),
                                                            start=2):
                            hj = arm["heads"][str(j)]
                            rs = hj["row_ce_sum"].setdefault(str(d), [])
                            rn = hj["row_n_tokens"].setdefault(str(d), [])
                            for b in range(x.shape[0]):
                                rs.append(float(ce_h[b][valid_h[b]].sum()))
                                rn.append(float(valid_h[b].sum()))
                            hsum[j] += float(ce_h[valid_h].sum())
                            hn[j] += float(valid_h.sum())
                    for j in hsum:
                        arm["heads"][str(j)]["depths"][d] = {"ce_tokens": hsum[j] / hn[j],
                                                             "n_tokens": hn[j]}
                    heads_txt = "  ".join(f"h{j}={hsum[j] / hn[j]:.4f}" for j in hsum)
                    print(f"{label:10s} depth={d}  ce={ce:.4f}  {heads_txt}", flush=True)
                else:
                    print(f"{label:10s} depth={d}  ce={ce:.4f}", flush=True)
        finally:
            model.cfg.mean_depth = orig_mean
        if prof is not None:
            arm["profile"] = prof.to_json()
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
