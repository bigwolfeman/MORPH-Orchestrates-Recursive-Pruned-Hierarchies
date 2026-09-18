"""Read the LIVE loop's cell with the FROZEN twin's reader — the only CE instrument that
survives a trainable front (2026-09-18).

Why this exists. On `tul.code_target_skip_coda` arms no coda runs at train, so the coda's
weights never move, but the prelude and the core train under a loss with no token term and
walk away from it. Measured on `tul-code-only` at 20k: the prelude drifted 0.400 of its own
norm and the core 0.279, `val/ce_tokens` climbed 9.36 -> 11.09 -> 12.64, and the coda's own
oracle ceiling (the TRUE code handed to it) sat at 8.3 nats against arm A's 1.41. At 12.64
nats the coda is 1.84 nats WORSE than uniform over the 49,152-token vocabulary, so it is
not a reader at all. Every CE instrument inherits that: the forced-depth sweep's K1-K6 read
+0.070, -0.001, +0.041, +0.029 at 5k/10k/15k/20k, each with a tight interval and the sign
flipping between checkpoints, and the worth profile read a shuffle cost of +0.297 against
arm A's +0.050 purely because the surface it sits on is broken.

What this does instead. The cell is the ONLY thing taken from the live model. Everything
that reads it — the prelude's token states, the coda, the head — comes from the frozen
VAE-stage twin (`tul.code_target_ref`), which cannot drift because nothing trains it. That
is a principled swap rather than a convenient one: the cell is regressed onto E's code and E
is defined on the twin, so the cell already lives in the twin's code space by construction.
It also puts a trainable-front arm and a frozen-front arm on ONE scale, which no other CE
instrument in this family can do. `--reader` picks which model reads the cell: "twin" (the
default) is the frozen VAE stage, "live" is the model itself. A frozen-front arm (arm A)
carries no twin at all, because it never needed one, so it is read with `--reader live` —
and on such an arm the live front IS the VAE stage, which is exactly what makes the two
readings comparable. On a trainable-front arm the gap between the two readers is the drift.

Conditions, all read by the twin, all on the same packed val rows:
  own    — the live loop's cells (the reference, matching worth_profile.py's "intact")
  zero   — no cell at all
  shuf   — each row's valid cells rolled by one, so a slot gets a FOREIGN cell from its own
           row: removes the slot-to-span correspondence and nothing else
  oracle — E's true code, the ceiling

Deltas are reported as `condition - own`, the same sign convention worth_profile.py uses for
`ablated - intact`, so the two tables can be read side by side. POSITIVE means own is better.
Stratified by the token's offset within its span, because a real write effect must DECAY with
offset and noise cannot fake that shape.

Usage:
  python lab/divergence/code_twin_read_probe.py \
    --ckpt co=tul_code_only=checkpoints/morph/tul-code-only/step_20000.pt \
    --rows 96 --out RESULTS/twin_read_co_20000.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402  (restores the twin, or LOAD_FAILs)

from _earning import BINS, bin_of  # noqa: E402  (ONE home for the offset bins)
from _rows import pack_rows, stream_from_loader  # noqa: E402  (the sweep's own packer)

# `own` is the reference every delta is taken against, so it is not in this tuple.
CONDS = ("zero", "shuf", "oracle")


def token_strata(layout, labels_row, b: int, spec):
    """[(position, bin)] for the scoreable token positions of row `b`.

    Same rule as worth_profile.py: slot positions are never scored, span 0 has no preceding
    slot to ablate FOR it, and the dump bin is excluded.
    """
    L = int(layout.slot_mask.shape[1])
    dump = spec.max_slots
    counts: dict[int, int] = {}
    out = []
    for p in range(L):
        if bool(layout.slot_mask[b, p]):
            continue
        bag = int(layout.bag_id[b, p])
        off = counts.get(bag, 0)
        counts[bag] = off + 1
        if bag == 0 or bag >= dump:
            continue
        if int(labels_row[p]) < 0:
            continue
        out.append((p, bin_of(off)))
    return out


def shuffled_cells(cells: torch.Tensor, layout) -> torch.Tensor:
    """Each row's VALID slot cells rolled by one, WITHIN the row.

    Within-row on purpose: it removes the correspondence between a slot and its span and
    leaves the row's cell distribution untouched, which is what makes the delta a
    span-SPECIFICITY number rather than a distribution-shift number. This is the same
    contract as worth_profile.py's "shuffle", not the roll-by-half-the-batch used for the
    training-time `code_target_cos_shuf` scalar.
    """
    out = cells.clone()
    for b in range(cells.shape[0]):
        idx = layout.slot_valid[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        out[b, idx] = cells[b, idx.roll(1)]
    return out


@torch.no_grad()
def live_cells(model, inp, layout, device) -> torch.Tensor:
    """The LIVE model's predicted cells [B, S, M, C] — the only thing taken from it."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal")
    if "code_cells" not in res:
        raise SystemExit("the eval forward exposed no code_cells: this is not a code-target "
                         "arm, so there is no cell to hand the twin.")
    return res["code_cells"].float()


@torch.no_grad()
def twin_ce(twin, inp, layout, labels, device, cells: torch.Tensor | None) -> torch.Tensor:
    """[B, L] per-token CE from the TWIN's forward. `cells` None => E's own code (oracle)."""
    kw = (dict(code_mode="encoder") if cells is None else
          dict(code_mode="generate", code_given=cells.to(device),
               code_given_mask=layout.slot_valid))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = twin(inp.to(device), labels=None, slot_layout=layout, **kw)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2] (extra Hydra overrides, comma-split)")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reader", default="twin", choices=("twin", "live"),
                    help="WHICH model reads the cell. 'twin' is the frozen VAE stage and "
                         "needs tul.code_target_ref; 'live' is the model itself, which is "
                         "the VAE stage anyway on a frozen-front arm and is the broken "
                         "reader on a trainable-front one.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        if a.reader == "twin":
            twin = model.code_ref
            if twin is None:
                # Without the twin the "fixed reader" would silently BE the live front,
                # which is the whole thing this probe exists to avoid. Refuse, and name the
                # flag that reads such an arm honestly.
                raise SystemExit(
                    f"{label}: --reader twin, but this model has no frozen twin "
                    f"(tul.code_target_ref is off, or the checkpoint carries no `code_ref` "
                    f"key). A frozen-front arm never needed one and is read with "
                    f"--reader live; on a trainable-front arm, re-run the training with "
                    f"code_target_ref so there is a fixed reader to hand the cell to.")
        else:
            twin = model
        twin.eval()
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        torch.manual_seed(a.seed)
        row_tokens = spec.l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

        row_sums = {c: [] for c in CONDS}
        row_counts = []
        abs_sums = {c: 0.0 for c in ("own", *CONDS)}
        abs_n = 0
        rows_done = 0
        for inp, labels, layout, _idx in batches:
            layout = layout.to(device)
            cells = live_cells(model, inp, layout, device)
            ce = {"own": twin_ce(twin, inp, layout, labels, device, cells),
                  "zero": twin_ce(twin, inp, layout, labels, device, torch.zeros_like(cells)),
                  "shuf": twin_ce(twin, inp, layout, labels, device,
                                  shuffled_cells(cells, layout)),
                  "oracle": twin_ce(twin, inp, layout, labels, device, None)}
            for b in range(inp.shape[0]):
                strata = token_strata(layout, labels[b], b, spec)
                cnt = np.zeros(len(BINS))
                sums = {c: np.zeros(len(BINS)) for c in CONDS}
                for p, bi in strata:
                    cnt[bi] += 1
                    abs_n += 1
                    for c in ("own", *CONDS):
                        abs_sums[c] += float(ce[c][b, p])
                    for c in CONDS:
                        sums[c][bi] += float(ce[c][b, p] - ce["own"][b, p])
                row_counts.append(cnt)
                for c in CONDS:
                    row_sums[c].append(sums[c])
            rows_done += inp.shape[0]
            print(f"  {label}: {rows_done}/{a.rows} rows", flush=True)

        cnts = np.stack(row_counts)
        rng = np.random.default_rng(a.seed)
        arm = {"step": step, "rows": rows_done, "reader": a.reader,
               "bins": [list(x) for x in BINS],
               "n_tokens_per_bin": cnts.sum(0).tolist(),
               "abs_ce": {c: abs_sums[c] / max(abs_n, 1) for c in ("own", *CONDS)},
               "conds": {}}
        # The absolute level is the sanity gate: a twin reading far above ln(vocab) would
        # mean the twin itself is not a reader and this probe is no better than the ones it
        # replaces. Printed first, on purpose.
        print(f"{label:10s} ABS CE  " + "  ".join(
            f"{c}={arm['abs_ce'][c]:.4f}" for c in ("own", *CONDS)), flush=True)
        for c in CONDS:
            sums = np.stack(row_sums[c])
            mean = sums.sum(0) / np.maximum(cnts.sum(0), 1)
            idx = rng.integers(0, len(sums), size=(a.boot, len(sums)))
            bmeans = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)
            lo, hi = np.percentile(bmeans, [2.5, 97.5], axis=0)
            tot = float(sums.sum() / max(cnts.sum(), 1))
            arm["conds"][c] = {"mean": mean.tolist(), "ci_lo": lo.tolist(),
                               "ci_hi": hi.tolist(), "total": tot}
            cells_s = "  ".join(f"{mu:+.3f}[{l:+.3f},{h:+.3f}]"
                                for mu, l, h in zip(mean, lo, hi))
            print(f"{label:10s} {c:8s} {cells_s}", flush=True)
            print(f"{label:10s} {c:8s} TOTAL {tot:+.4f}", flush=True)
        results[label] = arm
        del model, twin
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
