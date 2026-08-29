"""Stratified plan-worth profile — the decision-grade replacement for the all-token
worth_shuffle scalar (2026-08-29, "tea leaves" verdict).

Why: `val/plan_worth_shuffle` is a paired counterfactual (good) averaged over ALL
~1044 token positions per row (bad). The mechanism it hunts concentrates on the
tokens right after a slot; span-first tokens are ~5% of positions, so a real
0.4-nat effect there reads as ~0.02 in the mean — exactly the measured noise floor
(l1 vs l1-rep at step 1000: 0.012 vs 0.051; adjacent-eval bounce ~0.02).

What this does instead, offline on a saved checkpoint:
  1. per-token PAIRED CE deltas (ablated - intact) on the same packed val rows;
  2. stratified by the token's offset within its span (bag_id cumcount) — a real
     write effect must DECAY with offset (the plan is a prefix the coda refines);
     noise is flat and cannot fake the shape;
  3. bootstrap CI over rows, so an arm's profile carries its own error bars;
  4. all three ablations (zero / shuffle / wrong_seed) → dose-response ordering.

Tokens in span 0 (no preceding slot: nothing to ablate FOR them) and dump-bin
tokens are excluded. Slot positions are never scored (GL arms train them at
weight 0 — the emit_source lesson).

Usage:
  python lab/divergence/worth_profile.py \
    --ckpt l3=tul_l3=checkpoints/morph/tul-l3/step_4500.pt --rows 96 \
    --out /home/wolfe/morph-scratch/tulfm/worth_profile.json
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
from tul_samples import load_ckpt  # noqa: E402  (handles the _orig_mod. compile prefix)

BINS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 7), (8, 15), (16, 10 ** 9)]
MODES = ("zero", "shuffle", "wrong_seed")


def bin_of(off: int) -> int:
    for i, (lo, hi) in enumerate(BINS):
        if lo <= off <= hi:
            return i
    raise AssertionError(off)


def token_strata(layout, labels_row, b: int, spec) -> list[tuple[int, int]]:
    """[(position, bin)] for scoreable token positions of row b: real token, label
    present, in a span with a PRECEDING slot (bag_id in 1..n_valid-ish, not dump bin)."""
    L = layout.slot_mask.shape[1]
    out = []
    counts: dict[int, int] = {}
    dump = int(layout.slot_valid.shape[1])  # max_slots = the dump bin id
    for p in range(L):
        if bool(layout.slot_mask[b, p]):
            continue
        bag = int(layout.bag_id[b, p])
        off = counts.get(bag, 0)
        counts[bag] = off + 1
        if bag == 0 or bag >= dump:
            continue  # no preceding plan / dump bin
        if int(labels_row[p]) < 0:
            continue
        out.append((p, bin_of(off)))
    return out


@torch.no_grad()
def per_token_ce(model, inp, layout, labels, device, mode: str) -> torch.Tensor:
    """[B, L] CE at token positions (nan elsewhere), plan ablated per `mode`."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode=mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return ce


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2] (extra Hydra overrides, comma-split)")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.model.tul_layout import pack_tul_batch
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
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        # Same val stream, same seed for every arm -> profiles are comparable.
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        torch.manual_seed(a.seed)
        buf: list[int] = []
        need = a.batch * (spec.l_total + 1)
        # per ROW: sums[mode][bin], counts[bin]  (bootstrap unit = row)
        row_sums = {m: [] for m in MODES}
        row_counts = []
        rows_done = 0
        while rows_done < a.rows:
            while len(buf) < need:
                ids = next(loader)[0]
                buf.extend(ids.reshape(-1).tolist())
            inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
            layout = layout.to(device)
            ce = {"normal": per_token_ce(model, inp, layout, labels, device, "normal")}
            for m in MODES:
                ce[m] = per_token_ce(model, inp, layout, labels, device, m)
            for b in range(a.batch):
                strata = token_strata(layout, labels[b], b, spec)
                cnt = np.zeros(len(BINS))
                sums = {m: np.zeros(len(BINS)) for m in MODES}
                for p, bi in strata:
                    cnt[bi] += 1
                    for m in MODES:
                        sums[m][bi] += float(ce[m][b, p] - ce["normal"][b, p])
                row_counts.append(cnt)
                for m in MODES:
                    row_sums[m].append(sums[m])
            rows_done += a.batch
            print(f"  {label}: {rows_done}/{a.rows} rows", flush=True)
        cnts = np.stack(row_counts)                      # [R, bins]
        rng = np.random.default_rng(a.seed)
        arm = {"step": step, "rows": rows_done, "bins": [list(x) for x in BINS],
               "n_tokens_per_bin": cnts.sum(0).tolist(), "modes": {}}
        for m in MODES:
            sums = np.stack(row_sums[m])                 # [R, bins]
            mean = sums.sum(0) / np.maximum(cnts.sum(0), 1)
            idx = rng.integers(0, len(sums), size=(a.boot, len(sums)))
            bmeans = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)  # [boot, bins]
            lo, hi = np.percentile(bmeans, [2.5, 97.5], axis=0)
            arm["modes"][m] = {"mean": mean.tolist(), "ci_lo": lo.tolist(),
                               "ci_hi": hi.tolist()}
            cells = "  ".join(f"{mu:+.3f}[{l:+.3f},{h:+.3f}]"
                              for mu, l, h in zip(mean, lo, hi))
            print(f"{label:10s} {m:10s} {cells}", flush=True)
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
