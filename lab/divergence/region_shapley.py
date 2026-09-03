"""Exact Shapley credit for MORPH's three regions: prelude, core loop, coda.

`delta_ablation.py` measures a LEAVE-ONE-OUT margin, which is the wrong tool for the
question it raised. Removing the core loop costs 0.0169 nats while removing the prelude
costs 3.22. That is consistent with two very different worlds:

  1. the core is useless, or
  2. the core is REDUNDANT — the coda covers for it, so removing either alone is cheap
     while removing both is not.

A leave-one-out margin cannot tell those apart. Shapley can, because it averages a
player's marginal contribution over every coalition of the others.

Three players means EXACT Shapley: 8 coalitions, no sampling. Ablation is by making a
region the identity, the same operation `delta_ablation.py` validated (prelude +3.2205,
coda +3.1051 at ROLL_step_1750).

Value of a coalition S is the nats SAVED against the empty coalition:
``v(S) = L(nothing active) - L(only S active)``, so ``v({}) = 0`` and larger is better.

    python lab/divergence/region_shapley.py --ckpt checkpoints/morph/onset-capture/ROLL_step_1750.pt
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402

REGIONS = ("prelude", "core", "coda")


@contextlib.contextmanager
def ablated(root, names):
    """Make each named region the identity for the duration."""
    undo = []
    for n in names:
        if n == "core":
            orig = root._apply_core_step
            root._apply_core_step = lambda h_in, *a, **kw: (h_in, None)
            undo.append(lambda o=orig: setattr(root, "_apply_core_step", o))
        else:
            blocks = getattr(root, n)
            origs = [b.forward for b in blocks]
            for b in blocks:
                b.forward = (lambda h, *a, **kw: h)
            undo.append(lambda bs=blocks, os=origs: [setattr(b, "forward", f)
                                                     for b, f in zip(bs, os)])
    try:
        yield
    finally:
        for f in undo:
            f()


def eval_groups(model, batches) -> dict:
    """Mean total loss and the TUL group CEs over a FIXED batch list.

    The group CEs are the point: `ce_emit` is the slot's own target (predicted WITH the
    plan) and `ce_plast` is the SAME target predicted from the previous token with NO plan.
    Both carry weight 0.5 in the training loss, so they compete for one target, and
    `ce_plast - ce_emit` is how much the plan is worth at its own job.
    """
    keys = ("loss", "ce_main", "ce_plast", "ce_emit", "ce_tokens")
    tot = {k: 0.0 for k in keys}
    n = 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, layout in batches:
            out = model(x, labels=y, slot_layout=layout)
            for k in keys:
                if k in out and out[k] is not None:
                    tot[k] += float(out[k])
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(a.ckpt, model, scaler, torch.device("cuda"))

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    print(f"  fixed eval set: {len(batches)} batches\n")

    # One evaluation per coalition. `active` is the set kept; everything else is identity.
    losses: dict[frozenset, dict] = {}
    print(f"{'active regions':<28} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} "
          f"{'ce_emit':>8} {'plast-emit':>10}")
    for r in range(len(REGIONS) + 1):
        for active in itertools.combinations(REGIONS, r):
            off = [n for n in REGIONS if n not in active]
            with ablated(root, off):
                g = eval_groups(model, batches)
            losses[frozenset(active)] = g
            cf = g["ce_plast"] - g["ce_emit"]
            print(f"{('+'.join(active) or '(none)'):<28} {g['loss']:>8.4f} "
                  f"{g['ce_main']:>8.4f} {g['ce_plast']:>9.4f} {g['ce_emit']:>8.4f} "
                  f"{cf:>+10.4f}")

    base = losses[frozenset()]

    def phi(metric: str) -> dict:
        """Exact Shapley over 3 players. v(S) = L(empty) - L(S): nats saved."""
        def v(s):
            return base[metric] - losses[frozenset(s)][metric]
        n = len(REGIONS)
        out = {}
        for i in REGIONS:
            rest = [x for x in REGIONS if x != i]
            tot = 0.0
            for r in range(len(rest) + 1):
                for S in itertools.combinations(rest, r):
                    w = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                    tot += w * (v(tuple(S) + (i,)) - v(S))
            out[i] = tot
        return out

    print()
    rows = {}
    for metric in ("loss", "ce_main", "ce_plast", "ce_emit"):
        p = phi(metric)
        rows[metric] = p
        total = sum(p.values())
        print(f"Shapley on {metric:<9} " + "  ".join(f"{k}={v:8.4f}" for k, v in p.items())
              + f"   (sum {total:.4f} = v(all) {base[metric] - losses[frozenset(REGIONS)][metric]:.4f})")

    print()
    print("Shapley values are nats SAVED, split fairly across coalitions. They sum exactly")
    print("to the value of the full model over the empty one — that identity is the check.")
    print("ce_emit is the slot's own target WITH the plan; ce_plast is the same target from")
    print("the previous token with NO plan. Both carry weight 0.5, so they compete.")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"coalitions": {"+".join(sorted(k)) or "(none)": v
                                      for k, v in losses.items()},
                       "shapley": rows}, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
