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


@contextlib.contextmanager
def seed_bagmean(root):
    """Force ``slot_input`` back to the bag-mean seed, whatever the arm was TRAINED with.

    ADDED 2026-08-28, and the reason is a confound that made the round-2 panel
    unreadable. ``loop_off`` leaves the slot carrying its own INPUT, so what "no-loop"
    falls back to is arm-dependent:

        slot_seed=bag_mean  -> falls back to E_slot + mean(embed(span))   INFORMATIVE
        slot_seed=e_slot    -> falls back to E_slot alone                 A CONSTANT
        slot_seed=boundary  -> falls back to E_slot + W_sent.embed(t_last)

    So `full - no-loop` on a TG4a-style arm measures "loop vs NOTHING" while the same
    column on a bag_mean arm measures "loop vs a span summary". Reading the two side by
    side credits the loop for the seed's deletion. Measured on tg4a-s1/step_3000: the
    naive column says the loop is worth 0.0921 nats, ~25x the control band, entirely
    because its fallback was stripped of span content by design.

    Flipping ``slot_seed`` is enough because ``TULSlots.slot_input`` dispatches on it at
    CALL time, not at construction; only the ``add_e_slot=True`` (token-embedding)
    signal changes, and the bigram / value-embed signals were already the bag-mean in
    every mode.

    CAVEAT, and it is not removable without retraining: for an arm not TRAINED on the
    bag-mean this fallback is OUT OF DISTRIBUTION, so `full - no-loop-bagmean` is an
    UPPER bound on the loop's cross-arm worth. The truth for such an arm is bracketed
    BELOW that number and BELOW the naive `full - no-loop`. Report both; claim neither
    as the loop's worth.
    """
    tul_cfg = root.tul.tul
    orig = tul_cfg.slot_seed
    tul_cfg.slot_seed = "bag_mean"
    try:
        yield
    finally:
        tul_cfg.slot_seed = orig


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

    @contextlib.contextmanager
    def loop_off_bagmean(root):
        with loop_off(root), seed_bagmean(root):
            yield

    # From the BUILT model, not the YAML: this is the value slot_input actually reads.
    seed = str(root.tul.tul.slot_seed)
    # (row label, SHORT label for the worth table, context manager). The short label is
    # carried, not derived by splitting the long one: the first cut of this patch derived
    # it and printed "removing no-loop" for BOTH no-loop rows, which is precisely the
    # ambiguity the patch exists to remove.
    conds = [("full", "full", contextlib.nullcontext()),
             ("no-loop (slot keeps its seed)", "no-loop [own seed]", loop_off(root)),
             ("no-plan (slot values zeroed)", "no-plan", plan_off(root)),
             ("neither", "neither", both(root))]
    # On a bag_mean arm this condition IS `no-loop`, so running it would only spend eval
    # time to reprint the same row. Emit it exactly when the arm's seed differs.
    if seed != "bag_mean":
        conds.insert(2, (f"no-loop, bag-mean seed (was {seed})",
                         "no-loop [bag-mean]", loop_off_bagmean(root)))

    res = {}
    W = max(32, max(len(n) for n, _, _ in conds) + 1)
    print(f"{'condition':<{W}} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name, _short, ctx in conds:
        with ctx:
            g = eval_groups(model, batches)
        res[name] = g
        print(f"{name:<{W}} {g['loss']:>8.4f} {g['ce_main']:>8.4f} "
              f"{g['ce_plast']:>9.4f} {g['ce_emit']:>8.4f}")

    f = res["full"]
    print()
    print(f"{'what it is worth (nats)':<{W}} {'loss':>8} {'ce_main':>8} {'ce_plast':>9} {'ce_emit':>8}")
    for name, short, _ctx in conds[1:]:
        d = res[name]
        print(f"{'removing ' + short:<{W}} "
              + " ".join(f"{d[k] - f[k]:>8.4f}" if k != 'ce_plast' else f"{d[k] - f[k]:>9.4f}"
                         for k in ("loss", "ce_main", "ce_plast", "ce_emit")))
    print()
    print("If 'no-plan' costs about the same as 'no-loop', the LOOP is the worthless part")
    print("and the span bag-mean carries the plan. If 'no-plan' also costs ~0, the whole")
    print("slot path is inert for token prediction and TUL is not doing what it claims.")
    if seed != "bag_mean":
        print()
        print(f"slot_seed={seed}: read 'no-loop, bag-mean seed' for the CROSS-ARM loop worth.")
        print("The plain 'no-loop' row falls back to this arm's own seed, which carries less")
        print("span content than a bag_mean arm's does, so it credits the loop for the seed's")
        print("deletion. Both rows are UPPER bounds (see seed_bagmean's docstring).")
    print()
    print("CROSS-ARM CAVEAT on 'no-plan' (added 2026-08-28): under tg_restrict a token's ONLY")
    print("route to any earlier span is the slot path, so zeroing the plan removes ALL")
    print("cross-span information. An UNRESTRICTED arm keeps full causal attention and can")
    print("recover the same information directly, so its plan worth is low BY CONSTRUCTION.")
    print("Restricted-vs-unrestricted plan worth is therefore not evidence that the plan")
    print("CONTAINS more; only a direct read of z (lab/divergence/plan_content_probe.py) is.")

    # KEY RENAME 2026-08-28: the "no-loop" row used to be keyed
    # "no-loop (slot keeps bag-mean)". That label is FALSE on any arm whose slot_seed is
    # not bag_mean, and believing it is what nearly turned tg4a-s1's 0.0921 into a
    # reported 25x-control loop win. JSONs written before this date carry the old key.
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
