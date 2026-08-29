"""Oracle prefix probe — can the coda decode ANY content planted at a slot prefix?

Pre-registration: lab/experiments/planned/2026-08-28-oracle-prefix-probe.md.
Named as the next step by lab/experiments/failures/2026-08-28-tul-fm1-cw.md.

Six paired conditions on one fixed eval set; the oracle conditions replace each
valid slot's plan with its TRUE target y_i (the pooled next-span prelude rep the
planner is trained toward) — perfect content, delivered through the exact same
W_prefix interface the run trained.

    python lab/tulfm/oracle_prefix_probe.py \
        --ckpt checkpoints/morph/fm1-cw-s1/step_4500.pt --config tul_fm1_cw
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402


@contextlib.contextmanager
def plans_replaced(root, kind: str, gen: torch.Generator):
    """Patch `_tul_fm_core` so h_slots become oracle/noise content for VALID slots.

    `kind`: 'oracle_unit'  -> z_i := y_i (unit-norm, 30x the trained plan scale)
            'oracle_scaled'-> z_i := y_i * mean||plan_i|| (content at trained scale)
            'noise_scaled' -> z_i := N(0, I) * mean||plan_i||/sqrt(d) (energy control)
    Pads keep their original (zeroed) state; the STREAM broadcast is preserved.
    """
    orig = root._tul_fm_core

    def patched(x, layout):
        xn, h_slots, y, geom, ctx = orig(x, layout)
        hc = h_slots.dim() == 4                       # [B,S,streams,d] under HC
        z = h_slots[:, :, 0, :] if hc else h_slots
        valid = geom.valid
        norms = z[valid].norm(dim=-1)
        scale = norms.mean() if norms.numel() else z.new_tensor(1.0)
        if kind == "oracle_unit":
            new = y
        elif kind == "oracle_scaled":
            new = y * scale
        elif kind == "noise_scaled":
            n = torch.randn(y.shape, generator=gen, device="cpu").to(y)
            new = n * (scale / float(np.sqrt(y.shape[-1])))
        else:
            raise ValueError(kind)
        new = torch.where(valid.unsqueeze(-1), new.to(z.dtype), z)
        h_new = new.unsqueeze(2).expand_as(h_slots) if hc else new
        return xn, h_new, y, geom, ctx

    root._tul_fm_core = patched
    try:
        yield
    finally:
        root._tul_fm_core = orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_fm1_cw")
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # fm.source_std must match the RUN's CLI override or the rebuilt planner samples
    # its ladder at the wrong scale (prereg Method; the run launched with 0.03125).
    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false",
                               "fm.source_std=0.03125"])
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))
    model.eval()

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    print(f"fixed eval set: {len(batches)} batches x {batches[0][0].shape[0]} rows")

    gen = torch.Generator().manual_seed(0)
    conds = [("normal", contextlib.nullcontext(), "normal"),
             ("zero", contextlib.nullcontext(), "zero"),
             ("shuffle", contextlib.nullcontext(), "shuffle"),
             ("oracle_y_unit", plans_replaced(root, "oracle_unit", gen), "normal"),
             ("oracle_y_scaled", plans_replaced(root, "oracle_scaled", gen), "normal"),
             ("zero_scaled_noise", plans_replaced(root, "noise_scaled", gen), "normal")]

    per = {}
    for name, ctx, mode in conds:
        # torch.manual_seed pins the shuffle permutation and any dropout so every
        # condition sees identical stochasticity — the pairing the prereg requires.
        torch.manual_seed(1234)
        ces = []
        with ctx, torch.no_grad():
            for bx, by, bl in batches:
                out = root.tul_fm_forward(bx, by, bl, plan_mode=mode)
                ces.append(float(out["ce_tokens"]))
        per[name] = ces
        print(f"  {name:<18} ce_tokens={np.mean(ces):.4f}")

    base = np.array(per["normal"])
    rng = np.random.default_rng(0)
    res = {"ckpt": a.ckpt, "batches": a.batches,
           "ce": {k: float(np.mean(v)) for k, v in per.items()}}
    print(f"\n{'condition':<18} {'delta_vs_normal':>16} {'95% CI (paired bootstrap)':>28}")
    for name, ces in per.items():
        if name == "normal":
            continue
        d = np.array(ces) - base
        boots = [float(np.mean(d[rng.integers(0, len(d), len(d))])) for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.setdefault("delta", {})[name] = {"mean": float(d.mean()),
                                             "ci95": [float(lo), float(hi)]}
        print(f"{name:<18} {d.mean():>16.4f} {f'[{lo:+.4f}, {hi:+.4f}]':>28}")

    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
