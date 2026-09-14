"""Per-pass attention-sink mass — SMELT (Wang et al., arXiv 2609.01343) §6.4's instrument,
borrowed directly (docs/references/looping-depth/2026-09-13-lit-mining/F_scale.md, paper
5, "Concrete arm for us"): SMELT reports that a second visit through repeated layers
reduces attention-sink mass and redirects it to content; the concrete arm asks whether the
SAME thing happens pass over pass in the MORPH loop even though the K-curve (CE vs forced
depth) reads flat.

WHICH ATTENTION IS COUNTED. Exactly the window branch MORPH's ``val/attn_slot_mass`` and
``morph/model/attn_lift.py`` already count (grepped, per the task): tokens and slots (or
token positions, on the plain core) compete inside ONE softmax there, and it is the only
branch where "mass on a position" has SMELT's meaning. The compressed branch pools blocks,
not positions, so it is not counted (attn_lift.py's own docstring gives the reason).

WHAT COUNTS AS "THE SINK". SMELT's sink is the first token of a sequence; ours is compact
index 0 for the slot loop — the core loop attends over the GATHERED compact slot
sequence, not the row's ``L_total`` positions, and the first REAL slot always sits at
compact index 0 (pads sit last, SlotLayout's own convention) — and simply position 0 for
the plain core, which attends the row directly with no gather.

PER-PASS BUCKETING. ``_window_fallback`` is a bare function with no iteration index, so
this instrument wraps it ONLY for the duration of the loop's OWN forward call
(``_tul_core`` / ``_core_region``, never the prelude or coda, which are separate block
objects) and buckets consecutive calls into groups of ``n_core`` — the loop calls every
core layer, in the same fixed order, exactly once per iteration
(``lab/experiments/results/2026-08-24.../measuring-the-core-map.md``'s own model; the
slot loop additionally never shrinks its active set mid-iteration — Invariant 2,
``morph/model/CLAUDE.md`` — so call count per iteration is exactly ``n_core`` on every row
at eval, with a uniform per-sequence depth for the plain core too). Call index // n_core
is the pass number.

Usage:
  python lab/divergence/sink_mass_per_pass.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --rows 96 --device cuda --out results/sink_strict.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diag_common import Arm  # noqa: E402

import morph.model.attention as _attn  # noqa: E402
from morph.model.attn_lift import _window_mask  # noqa: E402


class PassSinkStats:
    """Per-PASS accumulation of (sink mass, content mass) over live query rows."""

    def __init__(self, n_core: int):
        self.n_core = n_core
        self.calls = 0
        self.per_pass: dict[int, list[list[float]]] = {}   # pass -> [sink_mass...]

    def record(self, pass_idx: int, sink_mass: torch.Tensor) -> None:
        self.per_pass.setdefault(pass_idx, []).append(sink_mass.tolist())


@contextlib.contextmanager
def capture_sink_mass(sink_pos: torch.Tensor, stats: PassSinkStats):
    """Wrap ``_window_fallback`` for the duration of ONE ``_tul_core``/``_core_region``
    call. ``sink_pos`` is ``[B]`` int64, the sink COLUMN each row's sink key sits at."""
    orig = _attn._window_fallback
    counter = {"n": 0}

    def measured(q, k, v, window_size, device, scale, n_skip_rope=0, extra_mask=None,
                relation=None):
        out = orig(q, k, v, window_size, device, scale, n_skip_rope, extra_mask, relation)
        S = q.shape[2]
        pass_idx = counter["n"] // stats.n_core
        counter["n"] += 1
        with torch.no_grad():
            mask = _window_mask(S, window_size, n_skip_rope, device, extra_mask)
            if relation is not None:
                # A register/reach arm replaces the causal term; neither target checkpoint
                # uses one, so this instrument refuses rather than silently mis-reading.
                raise RuntimeError("sink_mass_per_pass does not support tul.slot_cells>1 "
                                   "or tul.loop_reach>0 (relation-based masks)")
            bias = torch.where(mask, 0.0, float("-inf"))
            scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale + bias
            w = torch.softmax(scores, dim=-1)                              # [B, H, S, S]
            allow = mask.expand(q.shape[0], 1, S, S)[:, 0]                  # [B, S, S]
            n_keys = allow.sum(-1).float()                                 # [B, S]
            live = (n_keys > 0) & torch.isfinite(w).all(-1).all(1)         # [B, S]
            wmean = w.mean(1)                                              # [B, S, S] over heads
            B = q.shape[0]
            bidx = torch.arange(B, device=device)
            sink_allowed = allow[bidx, :, sink_pos]                        # [B, S] bool: can query i see the sink key?
            # column `sink_pos[b]` of every query row, one gather per batch row (B is small)
            sm = torch.stack([wmean[b, :, int(sink_pos[b])] for b in range(B)], dim=0)  # [B, S]
            sel = live & sink_allowed
            stats.record(pass_idx, sm[sel].float().cpu())
        return out

    _attn._window_fallback = measured
    try:
        yield stats
    finally:
        _attn._window_fallback = orig


def run(label: str, config: str, path: str, device: str, rows: int) -> dict:
    t0 = time.time()
    arm = Arm(label, config, path, device, rows, row_batch=1)
    model = arm.model
    n_core = int(model.cfg.n_core)
    stats = PassSinkStats(n_core)
    n_rows = 0
    for i, (inp, labels, layout) in enumerate(arm.rows()):
        B = inp.shape[0]
        # The sink position: compact index 0 for the slot loop (the core loop attends
        # over the GATHERED compact slot sequence, not the row's L_total positions — the
        # first REAL slot always sits at compact index 0, SlotLayout's own convention),
        # position 0 for the plain core (which attends the row directly, no gather).
        sink_pos = torch.zeros(B, dtype=torch.long, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            with torch.no_grad():
                if arm.is_slot_loop:
                    fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
                    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw,
                                                     ret_reset_mask=freset)
                    # The hook wraps ONLY the loop's own forward — the prelude above ran
                    # over the row's L_total positions, a different S than the core's
                    # compact slot sequence, and is not a "pass" of the loop at all.
                    with capture_sink_mass(sink_pos, stats):
                        model._tul_core(x, x0, bigram, layout, input_ids=inp)
                else:
                    x, x0, bigram = model._front_region(inp)
                    with capture_sink_mass(sink_pos, stats):
                        model._core_region(x, x0, bigram)
        n_rows += B
        if i % 20 == 0:
            n_calls = sum(len(v) for v in stats.per_pass.values())
            print(f"{label} row {i} calls_so_far={n_calls}", flush=True)

    per_pass_summary: dict[int, dict] = {}
    for p in sorted(stats.per_pass):
        vals = [v for call in stats.per_pass[p] for v in call]
        if not vals:
            continue
        t = torch.tensor(vals, dtype=torch.float64)
        per_pass_summary[p] = {"n": int(t.numel()), "mean_sink_mass": float(t.mean()),
                               "mean_content_mass": float(1.0 - t.mean())}

    passes = sorted(per_pass_summary)
    # relative flatness of passes 2..6 (1-indexed; pass index here is 0-based)
    inner_1idx = [p for p in passes if 2 <= p + 1 <= 6]
    inner_vals = [per_pass_summary[p]["mean_sink_mass"] for p in inner_1idx]
    flat_rel = None
    if len(inner_vals) >= 2:
        m = sum(inner_vals) / len(inner_vals)
        flat_rel = max(abs(v - m) / max(m, 1e-9) for v in inner_vals)

    out = {
        "label": label, "config": config, "ckpt": path, "step": arm.step,
        "is_slot_loop": arm.is_slot_loop, "rows": n_rows, "n_core": n_core,
        "wall_s": time.time() - t0,
        "per_pass_1indexed": {str(p + 1): per_pass_summary[p] for p in passes},
        "passes_2_to_6_max_relative_deviation": flat_rel,
    }
    print(f"{label}: per-pass sink mass = " +
          ", ".join(f"p{p+1}={per_pass_summary[p]['mean_sink_mass']:.4f}" for p in passes) +
          (f" | max rel dev (passes 2-6) = {flat_rel:.4f}" if flat_rel is not None else ""),
          flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        out[label] = run(label, config, path, a.device, a.rows)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        for label, e in out.items():
            f.write(f"{label}: config={e['config']} step={e['step']} slot_loop={e['is_slot_loop']} "
                    f"rows={e['rows']}\n")
            for p, s in e["per_pass_1indexed"].items():
                f.write(f"  pass {p}: mean_sink_mass={s['mean_sink_mass']:.4f} "
                        f"mean_content_mass={s['mean_content_mass']:.4f} n={s['n']}\n")
            f.write(f"  max relative deviation, passes 2-6: {e['passes_2_to_6_max_relative_deviation']}\n")
    print("wrote", a.out, "and", txt, flush=True)


if __name__ == "__main__":
    main()
