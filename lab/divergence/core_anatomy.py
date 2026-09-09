"""Core-loop anatomy of a PLAIN checkpoint: what each iteration does to the state.

Per forced iteration: the carrier norm, its participation-ratio rank, its distance from
and cosine to h_0 (the prelude output), the consecutive movement |h_t - h_{t-1}| / |h_{t-1}|,
and every core block's attention / MLP branch out-in norm ratio. Plus the learned
injection decay / dt and the core-0 residual-mixing parameters. The 2026-09-09 reading on
the E18 plain arm: movement 9.9 / 4.9 / 3.4 / 2.7 / 2.3 / 2.1 / 2.0 % per iteration,
branches 1-30 % (attention) and 4-13 % (MLP), decay 0.44 = its init (Parcae-140m: each
recurrent block moves the state ~40 % per pass at its fixed point).

Usage:
  python lab/divergence/core_anatomy.py --ckpt .../step_5000.pt --config notul_e18 \
      --rows 3 --depth 8 --out .../anatomy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def _flat(t: torch.Tensor) -> torch.Tensor:
    """[N, D] float view of a carrier (streams averaged when the carrier is 4-D)."""
    if t.dim() == 4:
        t = t.mean(dim=2)
    return t.float().reshape(-1, t.shape[-1])


def eff_rank(t: torch.Tensor, max_rows: int = 4096) -> float:
    """Participation-ratio rank of the centred [N, D] matrix."""
    m = _flat(t)
    m = m - m.mean(0, keepdim=True)
    s = torch.linalg.svdvals(m[:max_rows])
    p = s ** 2 / (s ** 2).sum()
    return float(1.0 / (p ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    cfg = build_cfg(a.config, ["model.use_kernels=false"])
    if build_tul_runtime(cfg) is not None:
        raise SystemExit("core_anatomy reads the PLAIN forward; pass a tul.activate_at=never config")
    model, step = load_ckpt(cfg, a.ckpt, "cuda", None)
    model.eval()
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    seq = int(cfg.data.seq_len)
    stream = stream_from_loader(loader, a.rows * (seq + 1))
    inp = pack_rows(stream, None, cfg, a.rows, True)[0][0].cuda()
    n_core = int(cfg.model.n_core)

    branch: dict[str, dict[int, list[float]]] = {"attn": {}, "mlp": {}}
    calls = {"attn": 0, "mlp": 0}
    states: list[torch.Tensor] = []
    captured: dict[str, torch.Tensor] = {}

    def branch_hook(kind: str):
        def f(_mod, args, out):
            t = calls[kind] // n_core
            calls[kind] += 1
            o = out[0] if isinstance(out, tuple) else out
            ratio = float(o.float().norm() / (args[0].float().norm() + 1e-6))
            branch[kind].setdefault(t, []).append(ratio)
        return f

    def block_hook(i: int):
        def f(_mod, _args, out):
            if i == n_core - 1:
                states.append((out if torch.is_tensor(out) else out[0]).detach())
        return f

    def h0_hook(_mod, _args, out):
        captured.setdefault("h0", out.detach())

    hooks = [model.input_norm.register_forward_hook(h0_hook)]
    for i, blk in enumerate(model.core):
        hooks += [blk.attention.register_forward_hook(branch_hook("attn")),
                  blk.mlp.register_forward_hook(branch_hook("mlp")),
                  blk.register_forward_hook(block_hook(i))]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.cfg.mean_depth = a.depth
        model(inp, labels=None)
    for h in hooks:
        h.remove()

    h0 = _flat(captured["h0"])
    rec = {
        "step": step, "rows": a.rows, "depth": a.depth,
        "h0": {"norm": float(h0.norm()), "rank": eff_rank(captured["h0"])},
        "carrier": [{"t": t + 1, "norm": float(_flat(s).norm()), "rank": eff_rank(s),
                     "dist_from_h0": float((_flat(s) - h0).norm() / h0.norm()),
                     "cos_to_h0": float(F.cosine_similarity(_flat(s), h0, dim=-1).mean())}
                    for t, s in enumerate(states)],
        "consecutive": [float((_flat(states[k]) - _flat(states[k - 1])).norm()
                              / _flat(states[k - 1]).norm()) for k in range(1, len(states))],
        "branch_out_in": {k: {str(t): v for t, v in d.items()} for k, d in branch.items()},
    }
    inj = model.injection
    rec["injection"] = {"A_mean": float(inj.log_A.exp().mean()), "A_min": float(inj.log_A.exp().min()),
                        "A_max": float(inj.log_A.exp().max()), "dt_mean": float(inj.log_dt.exp().mean()),
                        "span": [inj.start, inj.end],
                        "B_diag_mean": float(inj.B.diag().mean()) if inj.B is not None else None}
    rec["core0_params"] = {n: {"shape": list(p.shape), "norm": float(p.float().norm()),
                               "abs_mean": float(p.float().abs().mean())}
                           for n, p in model.core[0].named_parameters() if "mrr" in n}
    json.dump(rec, open(a.out, "w"), indent=1)
    print("h0 norm %.1f rank %.1f" % (rec["h0"]["norm"], rec["h0"]["rank"]))
    for c in rec["carrier"]:
        print("iter %d: |h| %.1f rank %.1f |h-h0|/|h0| %.3f cos %.3f"
              % (c["t"], c["norm"], c["rank"], c["dist_from_h0"], c["cos_to_h0"]))
    print("consecutive movement:", [round(x, 4) for x in rec["consecutive"]])
    for kind in ("attn", "mlp"):
        print(kind, "out/in per block, iteration 0 and last:",
              [round(x, 3) for x in branch[kind][0]], [round(x, 3) for x in branch[kind][max(branch[kind])]])
    print("injection", rec["injection"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
