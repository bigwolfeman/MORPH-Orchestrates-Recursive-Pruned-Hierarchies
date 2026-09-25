"""How much of what the READER gets from a slot is per-slot, and how much do passes add? (2026-09-25)

The NextLat filing (lab/experiments/failures/2026-09-25-lxtul-fp01-nextlat.md) found the
parallel head's view of the exit state, ``_readout(h_slots)``, is 93-99.9 % one vector shared
by every slot of every row, and that the arms which earn depth are the ones whose passes 2-6
ADD per-slot content. The coda does not read that view: it reads the prefix cells
``prefix_project(h_slots)`` (``W_prefix[k]`` per cell, HC streams kept). This probe reads BOTH
views on the same held-out rows at each forced depth, rollout 0 only:

  * ``share``: E|z - m|^2 / E|z|^2 over valid slots, ``m`` the mean over every valid slot in the
    probe (the per-slot share of the energy; 1 - share is the common mode);
  * ``share_row``: the same with ``m`` the mean of each ROW's valid slots (per-row common mode
    removed first);
  * ``share_nb``: diff_rel^2 / 2 from neighbouring slots, the estimate the NextLat filing used
    (equal to ``share`` when neighbours are independent);
  * ``cos_xrow``: cosine of a slot with the same slot index in the next row;
  * ``mu_top1`` / ``mu_top8`` / ``mu_top64`` / ``mu_pr`` / ``mu_top_idx``: how the common
    mode (the mean vector) spreads over coordinates: a few massive channels or a direction.

Views: ``head`` = ``_readout(h_slots)`` [C]; ``cell{k}`` = prefix cell k, all HC streams
flattened [n*C]. Usage:
    PYTHONPATH=.:lab/divergence python lab/divergence/slot_share_probe.py \
        --ckpt fp01=tul_slot_spandec_strict_e4probe_fp01=/path/step_5000.pt --rows 96 --out x.json
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


class _Acc:
    """Running sums for one view at one depth; the global mean needs two passes, so the
    vectors are kept (96 rows x 64 slots x n*C floats is ~0.1 GB in fp32 on CPU)."""

    def __init__(self):
        self.z, self.row, self.nb, self.xr = [], [], [0.0, 0], [0.0, 0]

    def add(self, z: torch.Tensor, sv: torch.Tensor) -> None:        # z [B, S, D], sv [B, S]
        z = z.float()
        for b in range(z.shape[0]):
            zb = z[b][sv[b]]
            self.z.append(zb.cpu())
            d = zb - zb.mean(0, keepdim=True)
            self.row.append((float(d.pow(2).sum()), float(zb.pow(2).sum())))
        nb = sv[:, :-1] & sv[:, 1:]
        diff = (z[:, 1:] - z[:, :-1])[nb].pow(2).mean(-1)
        rms2 = z[:, :-1][nb].pow(2).mean(-1)
        self.nb[0] += float((diff / rms2).sum())
        self.nb[1] += int(nb.sum())
        if z.shape[0] > 1:
            xr = sv & sv.roll(1, 0)
            self.xr[0] += float(F.cosine_similarity(z[xr], z.roll(1, 0)[xr], dim=-1).sum())
            self.xr[1] += int(xr.sum())

    def result(self) -> dict:
        allz = torch.cat(self.z)
        d = allz - allz.mean(0, keepdim=True)
        mu = allz.mean(0)
        e = mu.pow(2).sort(descending=True).values
        return {"share": float(d.pow(2).sum() / allz.pow(2).sum()),
                # WHAT the common mode is: the share of the mean vector's energy in its top
                # 1 / 8 / 64 coordinates, and its participation ratio (1 = one coordinate).
                "mu_top1": float(e[:1].sum() / e.sum()), "mu_top8": float(e[:8].sum() / e.sum()),
                "mu_top64": float(e[:64].sum() / e.sum()),
                "mu_pr": float(e.sum() ** 2 / e.pow(2).sum()),
                "mu_top_idx": [int(i) for i in mu.pow(2).argsort(descending=True)[:8]],
                "share_row": sum(a for a, _ in self.row) / sum(b for _, b in self.row),
                "share_nb": self.nb[0] / max(self.nb[1], 1) / 2,
                "cos_xrow": self.xr[0] / max(self.xr[1], 1),
                "n_slots": int(allz.shape[0])}


@torch.no_grad()
def probe(m, batches, depth: int, device: str) -> dict:
    views: dict[str, _Acc] = {}
    ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    K = int(m.tul.tul.prefix_k)
    for inp, labels, layout, _idx in batches:
        lay = layout.to(device)
        B0 = inp.shape[0]
        table = torch.full(lay.slot_index.shape, depth, dtype=torch.long)
        got = []
        real = m.tul.prefix_project

        def wrap(*a, **k):
            out = real(*a, **k)
            got.append(out[0])
            return out
        m.tul.prefix_project = wrap
        try:
            with ac, _Capture(m, "_tul_spandec_par_loss") as cap:
                m(inp.to(device), labels=labels.to(device), slot_layout=lay, slot_depths=table)
        finally:
            del m.tul.prefix_project
        if len(got) != 1:
            raise RuntimeError(f"prefix_project reached {len(got)} times, expected 1")
        a, _k = _one(cap)
        h_slots = a[0]
        S = lay.slot_index.shape[1]
        sv = lay.slot_valid.to(device).bool()
        head = m._readout(h_slots)[:B0]                                   # rollout 0
        views.setdefault("head", _Acc()).add(head.reshape(B0, S, -1), sv)
        vals = got[0][:B0].reshape(B0, S, K, -1)                          # slot-major s*K + k
        for k in range(K):
            views.setdefault(f"cell{k}", _Acc()).add(vals[:, :, k], sv)
    return {v: acc.result() for v, acc in views.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="1,2,3,4,5,6")
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
            print(f"{label} d{d}: " + "  ".join(
                f"{v} share {x['share']:.4f} row {x['share_row']:.4f} nb {x['share_nb']:.4f} "
                f"xcos {x['cos_xrow']:.4f}" for v, x in r.items()), flush=True)
        del m
        torch.cuda.empty_cache() if a.device == "cuda" else None
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
