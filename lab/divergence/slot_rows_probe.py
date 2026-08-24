"""Do the per-slot embedding rows re-converge?

`tul.per_slot_embed` is the only lever that improved validation CE on BOTH seeds, and it
still lost seed 0. The suspected mechanism: the 64 rows are learnable and every slot INDEX
plays nearly the same role, so they receive nearly the same gradient and can collapse back
toward each other, restoring the degeneracy the jitter broke.

`TULSlots._seat` gives every row the SAME mean vector plus jitter from a FIXED generator
(0x5107), so both arms start from a bit-identical `E_slot` and only the training seed
differs. That shared mean is why the raw pairwise cosine is the wrong statistic: it would
be dominated by the common component and would read near 1 whatever the rows do. Everything
here is measured on the CENTRED rows.

Usage:
    python lab/divergence/slot_rows_probe.py \
        --ckpt checkpoints/morph/s0-slotembed/step_2000.pt --label s0@2000
"""
from __future__ import annotations

import argparse
import json

import torch

__all__ = ["row_stats", "find_e_slot"]


def find_e_slot(state: dict) -> torch.Tensor:
    """The `[n_slots, d]` E_slot from a checkpoint's model state dict.

    Raises rather than returning None: a silent miss here would report "no rows moved".
    """
    hits = [k for k in state if k.replace("_orig_mod.", "").endswith("E_slot")]
    if not hits:
        raise KeyError("no E_slot in this checkpoint (per_slot_embed was off?)")
    if len(hits) > 1:
        raise KeyError(f"ambiguous E_slot keys: {hits}")
    t = state[hits[0]].float()
    if t.dim() != 2:
        raise ValueError(f"E_slot is {tuple(t.shape)}, not per-slot rows; "
                         "this arm ran the shared vector")
    return t


def row_stats(e: torch.Tensor) -> dict:
    """Diversity of the rows AFTER removing their common mean.

    `eff_rank` is the participation ratio of the centred singular values,
    `(sum s^2)^2 / sum s^4`, which equals n for n equal-energy orthogonal directions and 1
    for a rank-1 set. Isotropic Gaussian jitter starts it at essentially n.
    """
    mu = e.mean(dim=0, keepdim=True)
    c = e - mu
    s = torch.linalg.svdvals(c)
    p = s.square()
    eff = float(p.sum().square() / p.square().sum())
    cn = c / (c.norm(dim=1, keepdim=True) + 1e-30)
    g = cn @ cn.T
    n = e.shape[0]
    off = (g.sum() - g.diagonal().sum()) / (n * (n - 1))
    return {
        "n_rows": n,
        "d": e.shape[1],
        "eff_rank_centred": eff,
        "mean_pairwise_cos_centred": float(off),
        "spread": float(c.norm() / (mu.norm() * n ** 0.5)),
        "mean_norm": float(mu.norm()),
        # reported so the reader can SEE that the raw cosine is uninformative here
        "mean_pairwise_cos_raw": float(
            ((e / (e.norm(dim=1, keepdim=True) + 1e-30))
             @ (e / (e.norm(dim=1, keepdim=True) + 1e-30)).T).sum().sub(n)
            / (n * (n - 1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="path, or label=path")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rows = []
    print(f"{'arm':<22}{'rows':>5}{'effrank':>9}{'cos(c)':>9}{'spread':>9}{'cos(raw)':>10}")
    for spec in a.ckpt:
        label, path = spec.split("=", 1) if "=" in spec else (spec, spec)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        st = row_stats(find_e_slot(ck["model"]))
        st["label"], st["path"], st["step"] = label, path, int(ck.get("step", -1))
        rows.append(st)
        print(f"{label:<22}{st['n_rows']:>5}{st['eff_rank_centred']:>9.2f}"
              f"{st['mean_pairwise_cos_centred']:>9.4f}{st['spread']:>9.4f}"
              f"{st['mean_pairwise_cos_raw']:>10.4f}")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
