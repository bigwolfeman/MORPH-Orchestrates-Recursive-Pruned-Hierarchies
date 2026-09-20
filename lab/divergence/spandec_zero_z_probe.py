"""Does the span decoder READ z?  `tul/spandec_ce` with the slot's exit state as trained,
zeroed, and shuffled within the row, on the same rows at a forced slot depth.

Why this exists (2026-09-20): the offset prereg's P-9 asks for the decoder's CE "reading a
zeroed z", and `plan_mode="zero"` does NOT give it — `_tul_plan_ablate` runs AFTER
`_tul_spandec_loss` has read `h_slots` (transformer.py: the spandec loss at the labelled
forward, the ablation just before `prefix_project`), so the worth profile's zero mode
zeroes what the CODA reads and leaves the decoder's input untouched.  This probe wraps
`_tul_spandec_loss` itself.  Eval-only, no model code is changed.

Caveat, stated once: a decoder never sees a zeroed z at training, so `zero` is an
off-distribution input and its CE is an UPPER bound on "the decoder without z"; `shuffle`
keeps the input distribution and removes only the correspondence (the worth doctrine's
number to report when zero is not comfortably positive).

Usage (Spark, from ~/morph-perf):
  PYTHONPATH=. .venv/bin/python lab/divergence/spandec_zero_z_probe.py \
      --ckpt off3=tul_slot_spandec_strict_off3=~/readouts/off/off3_step_5000.pt \
      --depth 6 --rows 48 --batch 3 --out ~/readouts/off/spandec_z_off3_5000.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch

from _build import ROOT, build_cfg, parse_ckpt_spec, uses_sample_depth
from _rows import pack_rows, stream_from_loader

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

MODES = ("normal", "zero", "shuffle")


def _shuffle_within_row(h: torch.Tensor, layout) -> torch.Tensor:
    """Permute whole slots among the VALID slots of each row (the `_tul_plan_ablate`
    shuffle, copied so the probe does not depend on that method's call order)."""
    B, S = layout.slot_valid.shape
    r = torch.rand(B, S, device=h.device)
    r = torch.where(layout.slot_valid, r, torch.full_like(r, 2.0))
    perm = r.argsort(dim=1)
    idx = perm.reshape(B, S, *([1] * (h.dim() - 2))).expand_as(h)
    return h.gather(1, idx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=override,...] (the depth sweep's spec)")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(triple)
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None or not bool(getattr(cfg.tul, "spandec", False)):
            raise SystemExit(f"{label}: not a span-decoder arm (tul.spandec is off)")
        if uses_sample_depth(tul_rt.model_cfg):
            raise SystemExit(f"{label}: a paid-loop arm has no slot depth to force")
        model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
        model.eval()
        tc = model.cfg.tul
        tc.slot_mean_depth = a.depth
        tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth) or int(cfg.model.max_depth))
        if int(getattr(tc, "slot_depth_fixed", 0)) > 0:
            tc.slot_depth_fixed = a.depth
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
        batches = [(inp, labels, lay.to(device)) for inp, labels, lay, _ in batches]

        orig = model._tul_spandec_loss
        state = {"mode": "normal"}

        def wrapped(h_slots, input_ids, layout, stats=None, cells=None, _orig=orig):
            m = state["mode"]
            if m == "zero":
                h_slots = torch.zeros_like(h_slots)
                cells = None if cells is None else torch.zeros_like(cells)
            elif m == "shuffle":
                with torch.random.fork_rng(devices=[h_slots.device] if h_slots.is_cuda else []):
                    torch.manual_seed(a.seed)
                    h_slots = _shuffle_within_row(h_slots, layout)
                    if cells is not None:
                        torch.manual_seed(a.seed)
                        cells = _shuffle_within_row(cells, layout)
            return _orig(h_slots, input_ids, layout, stats=stats, cells=cells)

        model._tul_spandec_loss = wrapped
        per_batch = {m: [] for m in MODES}
        counts = {m: [] for m in MODES}
        with torch.no_grad():
            for bi, (inp, labels, layout) in enumerate(batches):
                for m in MODES:
                    state["mode"] = m
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        res = model.tul_forward_ablated(inp.to(device), labels.to(device),
                                                        layout, plan_mode="normal")
                    ce = float(res["spandec_ce"]); n = float(res["spandec_n_tokens"])
                    per_batch[m].append(ce * n); counts[m].append(n)
                print(f"{label} batch {bi + 1}/{len(batches)} "
                      + " ".join(f"{m}={per_batch[m][-1] / max(counts[m][-1], 1):.4f}" for m in MODES),
                      flush=True)
        model._tul_spandec_loss = orig
        for m in MODES:
            assert counts[m] == counts["normal"], "supervised token counts differ across modes"
        s = {m: np.array(per_batch[m]) for m in MODES}
        n = np.array(counts["normal"])
        rng = np.random.default_rng(a.seed)
        idx = rng.integers(0, len(n), (2000, len(n)))
        out = {"step": step, "depth": a.depth, "rows": sum(b[0].shape[0] for b in batches),
               "batch": a.batch, "n_supervised": float(n.sum()),
               "spandec_target_offset": int(getattr(tc, "spandec_target_offset", 1)),
               "spandec_ce": {m: float(s[m].sum() / n.sum()) for m in MODES}}
        for m in ("zero", "shuffle"):
            d = (s[m] - s["normal"]); boots = d[idx].sum(1) / n[idx].sum(1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            out[f"{m}_minus_normal"] = {"mean": float(d.sum() / n.sum()),
                                        "ci95": [float(lo), float(hi)]}
        results[label] = out
        print(f"{label} step {step} depth {a.depth} rows {out['rows']} offset "
              f"{out['spandec_target_offset']}: spandec_ce normal {out['spandec_ce']['normal']:.4f} "
              f"zero {out['spandec_ce']['zero']:.4f} shuffle {out['spandec_ce']['shuffle']:.4f} | "
              f"zero-normal {out['zero_minus_normal']['mean']:+.4f} "
              f"[{out['zero_minus_normal']['ci95'][0]:+.4f}, {out['zero_minus_normal']['ci95'][1]:+.4f}] "
              f"shuffle-normal {out['shuffle_minus_normal']['mean']:+.4f}", flush=True)
    json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
