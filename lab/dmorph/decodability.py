"""The σ* / autoencoding trap in ``t`` clothing (design note, Risks; prereg P6).

For the tok arm the noisy input is ``x_t = (1 - t)·x0 + t·y`` with ``y = normalize(E[label])``.
Above some ``t*`` the label can be recovered from ``x_t`` by nearest-neighbour decoding
against the table, with no context at all — the FM and CE terms are then autoencoding,
not language modelling. The testbed's ``scripts/decodability.py`` measured the same thing
in σ (SliceScaler put 77–98 % of training below σ*, ``db-testbed-ladder.md`` B). This is
the ``t``-coordinate port: for each ``t`` on a grid, the fraction of labels whose
nearest table row to ``x_t`` is the label itself, and the fraction of the training ``t``
mass (uniform, or the config's ``block_visit``) above the first ``t`` where that
fraction exceeds ``--threshold`` (0.5).

Run it on the INITIAL table (``--ckpt`` omitted) and on a trained checkpoint: a trained
table clusters, which moves ``t*`` up. If the mass above ``t*`` exceeds ~40 %, the
prereg says the filing has to reshape ``t`` (``dmorph.block_visit``) and say so.

    python lab/dmorph/decodability.py --config dmorph_tok [--ckpt path] [--n 4096]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lab.divergence._build import build_cfg, build_model            # noqa: E402
from morph.model.dmorph import band_bounds, band_of_t                # noqa: E402
from morph.training.train import load_checkpoint                    # noqa: E402


@torch.no_grad()
def nn_decode_accuracy(table: torch.Tensor, labels: torch.Tensor, t: float,
                       source_std: float, gen: torch.Generator) -> float:
    y = F.normalize(table[labels], dim=-1)
    x0 = torch.randn(y.shape, device=y.device, generator=gen) * source_std
    x_t = (1.0 - t) * x0 + t * y
    tab = F.normalize(table, dim=-1)
    pred = torch.empty_like(labels)
    for s in range(0, x_t.shape[0], 512):
        pred[s:s + 512] = (x_t[s:s + 512] @ tab.t()).argmax(dim=-1)
    return float((pred == labels).float().mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dmorph_tok")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cfg = build_cfg(a.config, a.override)
    dev = torch.device(a.device)
    model, _tul = build_model(cfg, device=a.device)
    if a.ckpt:
        load_checkpoint(a.ckpt, model, torch.amp.GradScaler("cuda", enabled=False), dev)
    model.eval()
    table = model.embed.lm_weight().detach().float()
    V, d = table.shape
    s = float(model.dmorph.cfg.source_std)
    n_blocks = int(model.dmorph.cfg.n_blocks)
    visit = model.dmorph.cfg.block_visit or tuple([1.0 / n_blocks] * n_blocks)

    gen = torch.Generator(device=dev).manual_seed(0)
    labels = torch.randint(0, V, (a.n,), device=dev, generator=gen)
    ts = [i / (a.grid - 1) for i in range(a.grid)]
    accs = [nn_decode_accuracy(table, labels, t, s, gen) for t in ts]
    t_star = next((t for t, acc in zip(ts, accs) if acc >= a.threshold), 1.0)
    # Mass of the training t distribution above t*: block-first sampling, uniform t
    # inside each γ-widened band (morph/model/dmorph.py::sample_t_in_band).
    mass_above = 0.0
    for b in range(n_blocks):
        lo, hi = band_bounds(b, n_blocks, float(model.dmorph.cfg.gamma))
        frac = max(0.0, min(hi, 1.0) - max(lo, t_star)) / max(hi - lo, 1e-9)
        mass_above += visit[b] * frac
    res = {"config": a.config, "ckpt": a.ckpt or "(init)", "V": V, "d": d,
           "source_std": s, "threshold": a.threshold, "t_star": t_star,
           "mass_above_t_star": mass_above,
           "band_of_t_star": int(band_of_t(torch.tensor([t_star]), n_blocks)),
           "grid": [{"t": t, "nn_acc": acc} for t, acc in zip(ts, accs)]}
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=2))
    print("  ".join(f"t={t:.3f}:{acc:.2f}" for t, acc in zip(ts, accs)))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
