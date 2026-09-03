"""How many nats does the core loop actually contribute under SCSE?

A core that has been quietly switched off looks the same as a stabilised core on the
gradient-share probe: both read low. This tells them apart.

SCSE makes the separation exact. The loop exits at ``h = h* + Delta_T``, so forcing
``Delta_T = 0`` removes EVERYTHING the core loop did — with no architecture change, no
change to loop depth, and no retraining. The gap between the two evaluation losses is the
core's contribution in nats.

The ablation is applied by making ``_SCSE.update`` return zeros. Delta is then zero at
every iteration, the core still runs on ``h*`` and its output is discarded, and the exit is
exactly ``h*``. That is the same thing as zeroing the exit, and it is one instance-level
patch instead of a source edit.

Read the TREND across checkpoints, not one value: a core being progressively quieted shows
a gap that shrinks toward zero as training proceeds.

    python lab/divergence/delta_ablation.py --ckpt-dir checkpoints/morph/scse-C-long
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader            # noqa: E402
from morph.training.train import load_checkpoint             # noqa: E402


@contextlib.contextmanager
def delta_zeroed(scse):
    """Force `Delta` to zero for the duration, so the loop's whole effect is removed."""
    orig = scse.update
    scse.update = lambda delta, stack_out, rec_in=None: torch.zeros_like(delta)
    try:
        yield
    finally:
        scse.update = orig


@contextlib.contextmanager
def blocks_identity(blocks):
    """Make a ModuleList of MORPHBlocks the identity. The PROBE'S OWN CONTROL.

    A core-loop ablation that costs almost nothing is either a real finding or a broken
    patch, and the two look identical from one number. Running the SAME machinery on the
    prelude or the coda settles it: if those cost many nats and the core costs few, the
    machinery works and the core really is cheap.
    """
    origs = [b.forward for b in blocks]
    for b in blocks:
        b.forward = (lambda h, *a, **kw: h)
    try:
        yield
    finally:
        for b, f in zip(blocks, origs):
            b.forward = f


@contextlib.contextmanager
def core_identity(root):
    """The SAME ablation for a model with NO SCSE: make every core step the identity.

    Under SCSE, forcing `Delta = 0` leaves the loop exit at `h*`, the state that would
    exist if the loop did nothing. Without SCSE the equivalent is to make each core step
    return its input, which leaves the exit at `core_init(e)` — again the state that would
    exist if the loop did nothing. The two ablations remove the same thing, so the nats they
    cost are comparable, which is the whole point: a small gap on an SCSE arm means nothing
    until the same gap is known for a HEALTHY model.
    """
    orig = root._apply_core_step
    root._apply_core_step = lambda h_in, *a, **kw: (h_in, None)
    try:
        yield
    finally:
        root._apply_core_step = orig


def eval_loss(model, batches) -> float:
    """Mean loss over a FIXED batch list, so both arms of the ablation see identical data."""
    tot, n = 0.0, 0
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, layout in batches:
            out = model(x, labels=y, slot_layout=layout)
            tot += float(out["loss"])
            n += 1
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--glob", default=None, help="checkpoint filename pattern")
    ap.add_argument("--ablate", default="core", choices=["core", "prelude", "coda"],
                    help="what to remove. prelude/coda are the probe's own control: they "
                         "use the same machinery on a region that is known to matter.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-scse", action="store_true",
                    help="checkpoints from a plain (non-SCSE) run; ablate by making each "
                         "core step the identity instead of by zeroing Delta")
    a = ap.parse_args()

    overrides = ["training.batch_size=6", "model.use_kernels=false"]
    if not a.no_scse:
        overrides += ["model.scse_enabled=true", "model.scse_input_mode=state"]
    cfg = build_cfg(a.config, overrides)
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    if not a.no_scse and root.scse is None:
        raise SystemExit("model has no SCSE module; pass --no-scse to ablate a plain run")

    # A FIXED evaluation set, drawn once. Both arms of every ablation see the same tensors,
    # so the gap is the intervention and not a different batch. `skip_samples` differs from
    # the trainer's own eval window so this is not the split the run was selected on.
    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        b = next(loader)
        if len(b) == 3:
            bx, by, bl = b
            batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
        else:
            (bx, by) = b
            batches.append((bx.cuda(), by.cuda(), None))
    print(f"  fixed eval set: {len(batches)} batches\n")

    pat = a.glob or "step_*.pt"
    paths = sorted(glob.glob(os.path.join(a.ckpt_dir, pat)),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not paths:
        raise SystemExit(f"no {pat} in {a.ckpt_dir}")

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rows = []
    print(f"{'step':>6} {'loss':>9} {'loss no-core':>13} {'gap (nats)':>11} {'gap/first':>10}")
    first_gap = None
    for p in paths:
        step = int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        load_checkpoint(p, model, scaler, torch.device("cuda"))
        full = eval_loss(model, batches)
        if a.ablate == "prelude":
            ablate = blocks_identity(root.prelude)
        elif a.ablate == "coda":
            ablate = blocks_identity(root.coda)
        else:
            ablate = core_identity(root) if a.no_scse else delta_zeroed(root.scse)
        with ablate:
            ablated = eval_loss(model, batches)
        gap = ablated - full
        if first_gap is None:
            first_gap = gap
        rows.append({"step": step, "loss": full, "loss_ablated": ablated, "gap": gap})
        print(f"{step:>6} {full:>9.4f} {ablated:>13.4f} {gap:>11.4f} "
              f"{gap / first_gap if first_gap else float('nan'):>10.2f}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"\nwrote {a.out}")
    print("\ngap = how many nats the core loop is worth. A core being switched off shows a")
    print("gap shrinking toward zero. Read the trend, not one row.")


if __name__ == "__main__":
    main()
