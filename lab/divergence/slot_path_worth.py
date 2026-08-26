"""Is it the LOOP that is worth nothing, or the whole SLOT PATH?

`region_shapley.py` ablates the core by making each loop iteration the identity. That
removes the LOOP. It does NOT remove the slot: the slot position still carries
`E_slot + mean(embed(span))` through the prelude, and the coda still reads a span summary
from it. So "core Shapley on ce_main = 0.0007" answers "is looping worth anything BEYOND
the span bag-mean", which is a narrower question than it looked.

This separates the two. Four conditions on one checkpoint, one fixed eval set:

| condition        | loop | slot values reaching the coda |
|------------------|------|-------------------------------|
| full             | on   | looped state                  |
| no-loop          | OFF  | un-looped slot input (bag-mean) |
| no-plan          | on   | ZERO                          |
| neither          | OFF  | ZERO                          |

`no-plan` is the honest test of "is the plan path used at all": it zeroes what
`prefix_project` writes into the shared sequence, so the coda's slot positions carry
nothing and every token predicting from them has only its own token path.

    python lab/divergence/slot_path_worth.py --ckpt checkpoints/morph/onset-capture/ROLL_step_1750.pt
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402


@contextlib.contextmanager
def loop_off(root):
    orig = root._apply_core_step
    root._apply_core_step = lambda h_in, *a, **kw: (h_in, None)
    try:
        yield
    finally:
        root._apply_core_step = orig


@contextlib.contextmanager
def plan_off(root):
    """Zero the values `prefix_project` writes into the shared sequence.

    Positions and validity are untouched, so the sequence LAYOUT is identical and only the
    content of the plan is removed. Anything the coda gains from those positions has to
    come from the plan, so this is the plan's whole worth.
    """
    tul = root.tul
    orig = tul.prefix_project

    def zeroed(h_slots, layout, l_total):
        values, pos = orig(h_slots, layout, l_total)
        return torch.zeros_like(values), pos

    tul.prefix_project = zeroed
    try:
        yield
    finally:
        tul.prefix_project = orig


def eval_groups(model, batches) -> dict:
    keys = ("loss", "ce_main", "ce_plast", "ce_emit")
    tot = {k: 0.0 for k in keys}
    n = 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, layout in batches:
            out = model(x, labels=y, slot_layout=layout)
            for k in keys:
                if out.get(k) is not None:
                    tot[k] += float(out[k])
            n += 1
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a1")
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

    @contextlib.contextmanager
    def both(root):
        with loop_off(root), plan_off(root):
            yield

    conds = [("full", contextlib.nullcontext()),
             ("no-loop (slot keeps bag-mean)", loop_off(root)),
             ("no-plan (slot values zeroed)", plan_off(root)),
             ("neither", both(root))]

    res = {}
    print(f"{'condition':<32} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name, ctx in conds:
        with ctx:
            g = eval_groups(model, batches)
        res[name] = g
        print(f"{name:<32} {g['loss']:>8.4f} {g['ce_main']:>8.4f} "
              f"{g['ce_plast']:>9.4f} {g['ce_emit']:>8.4f}")

    f = res["full"]
    print()
    print(f"{'what it is worth (nats)':<32} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name in ("no-loop (slot keeps bag-mean)", "no-plan (slot values zeroed)", "neither"):
        d = res[name]
        print(f"{'removing ' + name.split(' ')[0]:<32} "
              + " ".join(f"{d[k] - f[k]:>8.4f}" if k != 'ce_plast' else f"{d[k] - f[k]:>9.4f}"
                         for k in ("loss", "ce_main", "ce_plast", "ce_emit")))
    print()
    print("If 'no-plan' costs about the same as 'no-loop', the LOOP is the worthless part")
    print("and the span bag-mean carries the plan. If 'no-plan' also costs ~0, the whole")
    print("slot path is inert for token prediction and TUL is not doing what it claims.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()


# ── appended 2026-08-25: is the plan EMPTY, or CAPABLE but never asked? ────────────
#
# The plan is worth 0.0191 nats on ce_main because the coda can read span i's raw TOKENS
# and predict span i+1 from them directly. `token_state_dropout` (0.15) is the only tax
# that forces the coda to use the plan instead. This sweeps that tax at EVAL time.
#
# If the plan's worth on ce_main rises steeply as the token path is masked out, the plan
# HOLDS the information and the coda simply prefers tokens — a design leak, fixable by
# raising the tax. If it stays near zero even with every token state masked, the plan is
# empty and raising the tax fixes nothing.
