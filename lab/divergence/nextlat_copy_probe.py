"""Are a row's slot exit states distinct, or one state repeated? (NextLat filing, 2026-09-25)

The span-level NextLat arm (`tul.nextlat_weight`) met its term with a no-change guess:
`val/nextlat_copy_l1` 0.0009 against the transition's 0.0010. This probe reads, on the SAME
held-out rows for every arm (with or without the NextLat module), the exit readout
``z = model._readout(h_slots)`` that the parallel head grades, at a forced depth, and reports:

  * ``copy_l1``: SmoothL1(z_s, z_{s+1}) (beta 1) over valid (s, s+1) pairs, the term's own
    no-change baseline, computed with the model's own pair rule (`nextlat_pairs`);
  * ``cos_next``: mean cosine of z_s and z_{s+1} over the same pairs;
  * ``cos_far``: mean cosine of z_s and z_{s+4} (same row, both valid slots);
  * ``cos_xrow``: mean cosine of z_s and the same slot index in the NEXT row of the batch;
  * ``rms``: the per-channel RMS of z; ``diff_rel``: RMS(z_{s+1} - z_s) / RMS(z);
  * ``rank_row`` / ``rank_all``: participation-ratio rank of the valid z of a row (mean over
    rows) and of all valid z pooled.

Rollout 0 only under `code_enum_k` (the K rollouts share the front). Usage:
    PYTHONPATH=. python lab/divergence/nextlat_copy_probe.py \
        --ckpt fp01=tul_slot_spandec_strict_e4probe_fp01=/path/step_5000.pt \
        --ckpt nextlat=tul_slot_spandec_strict_e4probe_fp01_nextlat=/path/step_5000.pt \
        --rows 96 --depth 6 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import parse_ckpt_spec  # noqa: E402
from lxtul_e_stage1_score import _Capture, _one, load_arm, val_batches  # noqa: E402

from morph.model.tul_nextlat import nextlat_pairs  # noqa: E402
from morph.model.tul_spandec import span_slots  # noqa: E402


def _pr_rank(x: torch.Tensor) -> float:
    """Participation-ratio rank of the rows of ``x`` [n, C] after centring."""
    if x.shape[0] < 2:
        return float("nan")
    x = x - x.mean(0, keepdim=True)
    ev = torch.linalg.svdvals(x.double()) ** 2
    return float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-30))


@torch.no_grad()
def probe(m, batches, depth: int, device: str) -> dict:
    J = int(m.cfg.tul.spandec_max_tokens or m.cfg.tul.bound_span_cap)
    acc = {k: [0.0, 0] for k in ("copy_l1", "cos_next", "cos_far", "cos_xrow", "diff_rel")}
    rms, ranks, pooled = [], [], []
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    for inp, labels, layout, _idx in batches:
        lay = layout.to(device)
        table = torch.full(lay.slot_index.shape, depth, dtype=torch.long)
        with ac, _Capture(m, "_tul_spandec_par_loss") as cap:
            m(inp.to(device), labels=labels.to(device), slot_layout=lay, slot_depths=table)
        a, k = _one(cap)
        h_slots = a[0]
        R = max(int(k.get("n_rollouts", a[4] if len(a) > 4 else 1)), 1)
        B0 = inp.shape[0]
        z = m._readout(h_slots).float().view(R, B0, *h_slots.shape[1:2], -1)[0]   # [B0, S, C]
        _ids, valid = span_slots(inp.to(device), lay, J, shift=1)
        pair = nextlat_pairs(valid)[:, :-1]                                       # [B0, S-1]
        zs, zt = z[:, :-1][pair], z[:, 1:][pair]
        acc["copy_l1"][0] += float(F.smooth_l1_loss(zs, zt, reduction="none").mean(-1).sum())
        acc["copy_l1"][1] += zs.shape[0]
        acc["cos_next"][0] += float(F.cosine_similarity(zs, zt, dim=-1).sum())
        acc["cos_next"][1] += zs.shape[0]
        r = z.pow(2).mean(-1).sqrt()                                              # [B0, S]
        acc["diff_rel"][0] += float(((zt - zs).pow(2).mean(-1).sqrt() / r[:, :-1][pair]).sum())
        acc["diff_rel"][1] += zs.shape[0]
        sv = lay.slot_valid.to(device).bool()
        far = sv[:, :-4] & sv[:, 4:]
        acc["cos_far"][0] += float(F.cosine_similarity(z[:, :-4][far], z[:, 4:][far], dim=-1).sum())
        acc["cos_far"][1] += int(far.sum())
        if B0 > 1:
            xr = sv & sv.roll(1, 0)
            acc["cos_xrow"][0] += float(F.cosine_similarity(z[xr], z.roll(1, 0)[xr], dim=-1).sum())
            acc["cos_xrow"][1] += int(xr.sum())
        for b in range(B0):
            zb = z[b][sv[b]]
            rms.append(float(zb.pow(2).mean().sqrt()))
            ranks.append(_pr_rank(zb))
            pooled.append(zb.cpu())
    out = {k: v[0] / max(v[1], 1) for k, v in acc.items()}
    out["n_pairs"] = acc["copy_l1"][1]
    out["rms"] = sum(rms) / len(rms)
    out["rank_row"] = sum(ranks) / len(ranks)
    out["rank_all"] = _pr_rank(torch.cat(pooled))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="1,6")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res, batches = {}, None
    for spec in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(spec)
        m, step, cfg, rt = load_arm(config, path, a.device, ovr)
        if batches is None:
            batches, _n = val_batches(cfg, rt, a.rows, a.batch)
        res[label] = {"step": step}
        for d in [int(x) for x in a.depths.split(",")]:
            res[label][f"d{d}"] = r = probe(m, batches, d, a.device)
            print(f"{label} (step {step}) depth {d}: " + "  ".join(
                f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()),
                flush=True)
        del m
        torch.cuda.empty_cache() if a.device == "cuda" else None
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
