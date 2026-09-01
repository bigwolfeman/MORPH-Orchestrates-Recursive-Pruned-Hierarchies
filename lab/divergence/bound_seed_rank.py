"""E2 — bound-superposition seed: static rank check (no training).

Prereg: lab/experiments/planned/2026-09-01-bound-seed-rank.md. Compares the
shipped bag-mean slot seed against an HRR-bound seed built from the SAME
signal (`model.embed(input_ids)` — the tensor `_tul_front` hands to
`slot_input(add_e_slot=True)`), on real packed rows and the trained tul-20k
embedding path. Per row, over its valid slots: effective rank (participation
ratio of squared singular values, raw and unit-normalized — the jac_ladder
convention) and mean pairwise cosine. Both seed types, with and without the
shared E_slot term.

Usage:
  python lab/divergence/bound_seed_rank.py \
    --ckpt tul20k=tul_g0c0=checkpoints/morph/tul-20k/step_20000.pt \
    --rows 201 --out /home/wolfe/morph-scratch/tulfm/bound_seed_rank.json
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def eff_ranks(S: torch.Tensor) -> tuple[float, float]:
    """(raw, unit-normalized) participation-ratio effective rank of rows of S."""
    sv = torch.linalg.svdvals(S.double()) ** 2
    raw = float((sv.sum() ** 2) / (sv ** 2).sum())
    Sn = S / S.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    svu = torch.linalg.svdvals(Sn.double()) ** 2
    return raw, float((svu.sum() ** 2) / (svu ** 2).sum())


def mean_pair_cos(S: torch.Tensor) -> float:
    Sn = S / S.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    G = (Sn @ Sn.T).abs()
    n = S.shape[0]
    return float((G.sum() - n) / (n * (n - 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=201)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rot-seed", type=int, default=17)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None, "E2 needs a TUL config (the seed lives in TULSlots)"
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    rule = tul_rt.data_cfg.rule
    tc = model.cfg.tul
    assert tc.slot_seed == "bag_mean", f"expected shipped bag_mean seed, got {tc.slot_seed}"

    # Frozen per-offset orthogonal rotations: QR of a seeded Gaussian, one per
    # within-span offset up to span_cap. det sign is irrelevant (any orthogonal
    # binds); frozen buffers, zero trained parameters — exactly E3's design.
    d = model.cfg.d_model
    g = torch.Generator().manual_seed(a.rot_seed)
    R = torch.stack([torch.linalg.qr(torch.randn(d, d, generator=g))[0]
                     for _ in range(int(tc.span_cap))]).to(device)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    buf: list[int] = []
    need = a.batch * (spec.l_total + 1)

    per_row: list[dict] = []
    rows_done = 0
    with torch.no_grad():
        while rows_done < a.rows:
            while len(buf) < need:
                buf.extend(next(loader)[0].reshape(-1).tolist())
            inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
            layout = layout.to(device)
            inp = inp.to(device)
            signal = model.embed(inp).float()                        # [B, L, C]
            # Shipped seeds: the exact slot_input path, read back at slot positions.
            seeded = model.tul.slot_input(signal, layout, add_e_slot=True)
            token_sel = ~layout.slot_mask                             # [B, L] bool
            B, L, C = signal.shape
            for b in range(B):
                slots = torch.nonzero(layout.slot_mask[b] & layout_valid(layout, b),
                                      as_tuple=False).flatten()
                s_bag, s_bound, es = [], [], []
                for p in slots.tolist():
                    bag = int(layout.bag_id[b, p])
                    tok_pos = torch.nonzero((layout.bag_id[b] == bag) & token_sel[b],
                                            as_tuple=False).flatten()
                    n = tok_pos.numel()
                    if n == 0:
                        continue
                    e = signal[b, tok_pos]                            # [n, C]
                    bound = torch.einsum("kij,kj->i", R[:n].float(), e) / (n ** 0.5)
                    s_bag.append(seeded[b, p])
                    s_bound.append(model.tul._e_slot_term(
                        layout.bag_id[b, p:p + 1].unsqueeze(0), signal.dtype
                    ).reshape(-1) + bound)
                    es.append(e.mean(0))                              # bare bag term
                if len(s_bag) < 3:
                    continue
                Sb = torch.stack(s_bag).cpu()
                So = torch.stack(s_bound).cpu()
                Se = torch.stack(es).cpu()                            # no-E_slot bag
                row = {}
                for name, S in (("bag", Sb), ("bound", So),
                                ("bag_noeslot", Se),
                                ("bound_noeslot", So - (Sb - Se))):
                    r_raw, r_unit = eff_ranks(S)
                    row[name] = {"rank_raw": r_raw, "rank_unit": r_unit,
                                 "pair_cos": mean_pair_cos(S), "n_slots": S.shape[0]}
                per_row.append(row)
            rows_done += B
            if rows_done % 30 < a.batch:
                print(f"rows {rows_done}/{a.rows}", flush=True)

    summary: dict[str, dict] = {}
    for name in ("bag", "bound", "bag_noeslot", "bound_noeslot"):
        for key in ("rank_raw", "rank_unit", "pair_cos"):
            vals = [r[name][key] for r in per_row]
            summary.setdefault(name, {})[key] = sum(vals) / len(vals)
        summary[name]["n_rows"] = len(per_row)
    ratio = summary["bound"]["rank_unit"] / summary["bag"]["rank_unit"]
    out = {"step": step, "rows": len(per_row), "rot_seed": a.rot_seed,
           "summary": summary, "rank_unit_ratio_bound_over_bag": ratio}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    for name in ("bag", "bound", "bag_noeslot", "bound_noeslot"):
        s = summary[name]
        print(f"{name:14s} rank_unit={s['rank_unit']:.2f} rank_raw={s['rank_raw']:.2f} "
              f"|cos|={s['pair_cos']:.3f}", flush=True)
    print(f"P-B1 ratio (bound/bag, unit rank): {ratio:.2f}  "
          f"P-B2 bound |cos|: {summary['bound']['pair_cos']:.3f}", flush=True)
    print(f"wrote {a.out}")


def layout_valid(layout, b: int) -> torch.Tensor:
    """[L] bool: positions whose slot (if a slot) is a VALID slot of row b."""
    v = torch.zeros_like(layout.slot_mask[b])
    idx = layout.slot_index[b][layout.slot_valid[b]]
    v[idx] = True
    return v | ~layout.slot_mask[b]


if __name__ == "__main__":
    main()
