"""Fan stream geometry: WHAT SHAPE do a fan arm's K streams take, per pass, on a checkpoint.

The trainer reports two scalars per pass (``fan/stream_cos_t{t}``: mean pairwise cosine on
the RAW streams; ``fan/stream_rank_t{t}``: participation-ratio rank on the CENTERED
streams — ``morph/model/tul_fan.py::fan_stream_stats``). On ``slot-spandec-strict-fan4``
@ 5000 they read −0.326 and 1.05 of 4 together from step 500 on. The mean pairwise cosine
of K unit vectors is bounded below by −1/(K−1) = −1/3, and that floor is reached BOTH by a
regular simplex (centered rank 3) and by a rank-1 split of two copies against two negated
copies (centered rank 1). The two scalars say the streams took the second shape; they
cannot say whether the split has fixed stream identities, whether the four norms match,
or whether the ONE direction is the same axis for every slot. This probe dumps the
``[N, M, C]`` cells at every pass and reads those three things.

Usage (a probe host, never the trainer's GPU):

    python lab/divergence/fan_stream_probe.py \
        --ckpt slot-spandec-strict-fan4=tul_slot_spandec_strict_fan4=/path/step_5000.pt \
        --rows 48 --batch 3 --depth 6 --out fan_geom.json

``stream_geometry`` is the pure function (tested in ``tests/test_fan_stream_probe.py``);
``main`` loads the checkpoint exactly as ``core_depth_sweep.py`` does and captures the
per-pass cells by wrapping the name ``morph.model.transformer.fan_stream_rank``, which the
eval path calls once per pass on the live trajectory.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

import torch
import torch.nn.functional as F


def stream_geometry(cells: torch.Tensor) -> dict:
    """``[N, M, C]`` valid slots -> the shape of the M streams, averaged over the N slots.

    Keys:
      ``cos_mean``        mean pairwise cosine on the raw streams (the trainer's reading).
      ``cos_matrix``      the ``[M, M]`` pairwise cosine, averaged over slots: WHICH pairs
                          agree and which oppose.
      ``eff_rank``        centered participation-ratio rank (the trainer's reading).
      ``sv_ratio``        centered singular values ``s_k / s_1`` for k = 2..M, mean over
                          slots: 0 everywhere = one line.
      ``shared_frac``     ``‖mean_i s_i‖ / mean_i ‖s_i‖``: 1 = copies, 0 = no common part.
      ``norms``           mean ``‖s_i‖`` per stream, ``[M]``.
      ``top_pattern``     the most common sign pattern of the streams' coefficients on
                          the slot's top centered direction (canonical: first sign ``+``).
      ``top_pattern_frac`` its frequency over slots: 1.0 = the same streams are on the
                          same side in every slot (fixed identities).
      ``axis_cos``        mean ``|cos|`` between the top centered direction of slot n and
                          of slot n+1: 1 = one global axis, ~0 = a different line per slot.
    """
    if cells.dim() != 3:
        raise ValueError(f"stream_geometry wants [N, M, C], got {tuple(cells.shape)}")
    n, m, c = cells.shape
    if m < 2:
        raise ValueError(f"stream_geometry needs M >= 2 streams, got {m}")
    x = cells.double()
    norms = x.norm(dim=-1)                                            # [N, M]
    unit = F.normalize(x, dim=-1)
    cos = unit @ unit.transpose(1, 2)                                 # [N, M, M]
    off = (cos.sum((1, 2)) - cos.diagonal(dim1=1, dim2=2).sum(-1)) / float(m * (m - 1))
    xc = x - x.mean(dim=1, keepdim=True)
    g = xc @ xc.transpose(1, 2)                                       # [N, M, M] Gram
    evals, evecs = torch.linalg.eigh(g)                               # ascending
    evals = evals.clamp_min(0.0)
    tr = evals.sum(-1)
    fro2 = (evals * evals).sum(-1)
    er = torch.where(fro2 > 0, tr * tr / fro2.clamp_min(1e-300), torch.zeros_like(tr))
    sv = evals.flip(-1).sqrt()                                        # descending
    sv_ratio = torch.where(sv[:, :1] > 0, sv[:, 1:] / sv[:, :1].clamp_min(1e-300),
                           torch.zeros_like(sv[:, 1:]))
    shared = x.mean(dim=1).norm(dim=-1) / norms.mean(dim=1).clamp_min(1e-300)
    top = evecs[:, :, -1]                                             # [N, M] coefficients
    signs = torch.sign(top)
    first = signs[:, :1].clone()
    first[first == 0] = 1.0
    signs = signs * first                                             # canonical: s_0 = +
    patterns = ["".join("+" if v > 0 else ("-" if v < 0 else "0") for v in row)
                for row in signs.tolist()]
    cnt = Counter(patterns)
    pat, pat_n = cnt.most_common(1)[0]
    axis = F.normalize((top.unsqueeze(-1) * xc).sum(dim=1), dim=-1)   # [N, C]
    if n > 1:
        axis_cos = (axis[:-1] * axis[1:]).sum(-1).abs().mean()
    else:
        axis_cos = torch.tensor(float("nan"), dtype=torch.double)
    return {
        "n_slots": int(n), "m": int(m), "c": int(c),
        "cos_mean": float(off.mean()),
        "cos_matrix": cos.mean(0).tolist(),
        "eff_rank": float(er.mean()),
        "sv_ratio": sv_ratio.mean(0).tolist(),
        "shared_frac": float(shared.mean()),
        "norms": norms.mean(0).tolist(),
        "top_pattern": pat,
        "top_pattern_frac": float(pat_n) / float(n),
        "patterns": dict(cnt.most_common(6)),
        "axis_cos": float(axis_cos),
    }


def _print(label: str, t: int, g: dict) -> None:
    sv = " ".join(f"{v:.3f}" for v in g["sv_ratio"])
    print(f"{label} t={t} slots={g['n_slots']} cos={g['cos_mean']:+.4f} rank={g['eff_rank']:.3f} "
          f"sv2../sv1=[{sv}] shared={g['shared_frac']:.3f} pattern={g['top_pattern']} "
          f"({g['top_pattern_frac']:.2f}) axis_cos={g['axis_cos']:.3f} "
          f"norms={' '.join(f'{v:.1f}' for v in g['norms'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NAME=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6, help="forced slot depth, as the sweep")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    import morph.model.transformer as tr_mod
    from morph.model.tul_fan import _cell_readout

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or int(getattr(tul_rt.model_cfg, "fan_k", 0)) < 2:
        raise SystemExit(f"{label}: not a fan arm (tul.fan_k < 2)")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    m_cells = int(tc.fan_k)
    tc.slot_mean_depth = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth))
    if int(getattr(tc, "slot_depth_fixed", 0)):
        tc.slot_depth_fixed = a.depth

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

    captured: dict[int, list[torch.Tensor]] = {}
    real_rank = tr_mod.fan_stream_rank
    pass_idx = {"t": 0}

    def capture(state, valid, m):
        b, sm = state.shape[0], state.shape[1]
        s = valid.shape[1]
        z = _cell_readout(state.reshape(b, s, m, *state.shape[2:]))
        captured.setdefault(pass_idx["t"], []).append(z[valid].detach().float().cpu())
        pass_idx["t"] += 1
        return real_rank(state, valid, m)

    tr_mod.fan_stream_rank = capture
    try:
        with torch.no_grad():
            for inp, _labels, layout, _idx in batches:
                pass_idx["t"] = 0
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                    model.tul_forward_ablated(inp.to(a.device), None, layout.to(a.device),
                                              plan_mode="normal")
    finally:
        tr_mod.fan_stream_rank = real_rank
    if not captured:
        raise SystemExit("no fan cells captured: the eval path did not call fan_stream_rank")

    result = {"arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
              "batch": a.batch, "depth": a.depth, "fan_k": m_cells, "passes": {}}
    for t in sorted(captured):
        cells = torch.cat(captured[t], dim=0)
        g = stream_geometry(cells)
        result["passes"][str(t)] = g
        _print(label, t, g)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
