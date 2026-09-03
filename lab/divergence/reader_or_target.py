"""Is the plan EMPTY, or is the coda BYPASSING a usable one?

This is the question the campaign has circled since 2026-08-25, when a note was appended to
`slot_path_worth.py` describing exactly this sweep. The code was never written.

WHY THE EXISTING PANELS CANNOT ANSWER IT. `plan_off` and `plan_shuffled` both run with the
coda's token path fully intact, because `apply_token_dropout` is a no-op at eval. A coda that
can read a span's own tokens has no reason to consult a plan, so those panels measure what
the coda BOTHERS to use. The measured answer — shuffling the plan costs 0.0001 nats — is
consistent with BOTH "z is empty" and "z is fine and the reader ignores it".

THE SWEEP. Raise the token tax at eval (`token_tax`) and re-measure at each level. At p=1.0
every token state is `E_mask` (Bowman's inputless decoder), so the coda has nothing but the
plans and the positions.

    shuffle cost stays ~0 at p=1.0   -> z holds no span-specific content. There is nothing
                                        to read, and no reader fix can help.
    shuffle cost RISES with p        -> z HOLDS content the coda ignores while tokens are
                                        available. The READER is the bottleneck, and an
                                        objective that writes more into z (flow matching)
                                        is well motivated rather than speculative.

Every condition drops the SAME token positions: the global RNG is reseeded before each eval,
because the drop is sampled inside the shipped `apply_token_dropout`. Without that, conditions
would differ by which tokens were masked as well as by the plan.

    python lab/divergence/reader_or_target.py --ckpt CKPT --config CFG [--out J]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model              # noqa: E402
from lab.divergence.slot_path_worth import (                          # noqa: E402
    eval_groups, plan_off, plan_shuffled, token_tax)
from morph.training.data import create_dataloader                     # noqa: E402
from morph.training.train import load_checkpoint                      # noqa: E402

TAXES = (0.0, 0.5, 0.9, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
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
    batches = [tuple(t.cuda() if hasattr(t, "cuda") else t.to("cuda")
                     for t in next(loader)) for _ in range(a.batches)]
    print(f"  {a.label or a.ckpt}: {len(batches)} eval batches\n")

    def run(tax: float, name: str, inner):
        # Reseed so EVERY condition at this tax drops the same token positions.
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        with token_tax(root, tax), inner:
            return eval_groups(model, batches)["ce_main"]

    rows = []
    print(f"{'tax':>5} {'full':>9} {'no-plan':>9} {'shuffled':>9} "
          f"{'zero cost':>10} {'shuf cost':>10} {'specificity':>12}")
    for tax in TAXES:
        full = run(tax, "full", contextlib.nullcontext())
        zero = run(tax, "no-plan", plan_off(root))
        shuf = run(tax, "shuffled", plan_shuffled(root))
        dz, ds = zero - full, shuf - full
        spec = ds / dz if abs(dz) > 1e-9 else float("nan")
        rows.append({"tax": tax, "full": full, "no_plan": zero, "shuffled": shuf,
                     "zero_cost": dz, "shuffle_cost": ds, "specificity": spec})
        print(f"{tax:>5.2f} {full:>9.4f} {zero:>9.4f} {shuf:>9.4f} "
              f"{dz:>10.4f} {ds:>10.4f} {100*spec:>11.1f}%")

    s0 = rows[0]["shuffle_cost"]
    s1 = rows[-1]["shuffle_cost"]
    print()
    print(f"SHUFFLE COST  p=0.0 -> {s0:+.4f}   p=1.0 -> {s1:+.4f}   ratio "
          f"{(s1 / s0 if abs(s0) > 1e-9 else float('nan')):.1f}x")
    print("Rising with the tax: z HOLDS span-specific content the coda ignores while the")
    print("  tokens are there. The READER is the bottleneck; writing more into z is")
    print("  motivated. Flat near zero: z holds nothing and no reader fix helps.")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"ckpt": a.ckpt, "config": a.config, "label": a.label,
                       "taxes": list(TAXES), "rows": rows}, fh, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
