"""TUL as MEMORY: token CE against INFERENCE cost, on one set of paired rows.

Every slot arm in the arc has been scored on loop CONTRIBUTION — the K-curve, the worth
profile, the per-pass cotangents — and the answer has been the same twelve times: the
passes are worth 0.0005 to 0.0033 nats. That is a verdict on the LOOP. It is not a verdict
on the SLOT, and the two have been read as one.

The slot is also a memory: a fixed-size cell per span that a later token may read instead
of re-reading the span. A memory is not priced in nats, it is priced in nats PER UNIT OF
INFERENCE COST, and nobody has written that table. This script writes it.

WHAT IT PUTS IN ONE ROW PER ARM
-------------------------------
1. **CE**, at the arm's trained loop depth, from the per-token sweep artifact
   `core_depth_sweep.py` already wrote (`sweep_<arm>_<step>.<arm>.tokens.npz`: `tok_index`,
   the STREAM index of every scored position, and `ce_<depth>`). Arms are paired on
   `tok_index`, so arms that cut the same stream into different rows still pair at the
   TOKEN level, and every reported CE is over the SAME intersection of stream positions —
   printed, so nobody has to trust it.
2. **Inference cost**, three numbers per GENERATED token at decode, all arithmetic from
   the arm's own composed config plus ONE measurement on real packed rows (the mean span
   length and the mean number of visible key positions). The cost model is defined in
   `docs/tul-as-memory.md` and nowhere else; this script is its only implementation.
3. **The fraction of the cross-span budget recovered**,
   `(CE_spanlocal - CE_arm) / (CE_spanlocal - CE_full)`, against two NAMED arms.

THE CAVEAT THAT DECIDES HOW ROW 3 IS READ, and it is not optional. `budget-web-full` and
`budget-web-span` (2026-09-11) run `model.core_impl: parcae` — a dense softmax core with no
ternary QAT — while every slot arm runs the MORPH ternary core. Their ABSOLUTE CE is
therefore not comparable to a slot arm's, and the denominator they define is a SCALE
measured on a different model. The script prints the config difference it found rather than
assuming one, refuses to print a recovered-fraction when the cores differ unless
`--allow-core-mismatch` is passed, and the prereg
`lab/experiments/planned/2026-09-13-arc-tul-as-memory.md` queues the matched control
(`plain-span-local`, the slot arms' own core with `model.span_mask: span`) that removes it.

WHAT IT IS NOT
--------------
Not a wall-clock measurement. Not a K-curve. Not a claim that any arm is good: it is a
table, and the ranking it produces is the thing to argue about.

Usage:
  python lab/divergence/memory_cost_table.py \\
      --arm slot-spandec-strict=tul_slot_spandec_strict=/…/sweep_slot-spandec-strict_5000.slot-spandec-strict.tokens.npz \\
      --arm plain-coda-matched=plain_coda_matched=/…/sweep_plain-coda-matched_5000.plain-coda-matched.tokens.npz \\
      --span-local budget-web-span --full budget-web-full \\
      --rows 480 --out OUT.json --md OUT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _build import build_cfg                                         # noqa: E402
from _rows import pack_rows, stream_from_loader                      # noqa: E402

BYTES_PER_KV = 2          # bf16 KV cache


# ── the cost model ───────────────────────────────────────────────────────────

def _visible_keys(cfg, tul_rt, layout, inp) -> tuple[float, float]:
    """``(mean visible key positions per TOKEN query, mean span length in tokens)``.

    MEASURED on real packed rows, through the SHIPPED relation builders
    (`tg_strict_allow`, `tg_allow_mask`, `span_allow_mask`), never a re-derivation:
    a second copy of an allow relation is how two paths drift apart, and this one would
    drift silently into a cost advantage.

    The relation used is the CODA's, because at decode a generation step recomputes the
    whole stack for the new position and the widest stage decides what that position can
    see. Under `strict` the prelude is same-span-only and the coda adds the earlier slot
    cells, so the coda relation is the union; under `tg_restrict` both stages carry the
    same relation; a plain model has one.
    """
    from morph.model.tul_layout import (span_allow_mask, span_ids_from_ids,
                                        tg_allow_mask, tg_strict_allow)

    if layout is None:
        # A plain arm: no slot cells in the row at all. `model.span_mask` decides.
        sm = str(getattr(cfg.model, "span_mask", "off"))
        B, L = inp.shape
        if sm == "span":
            from morph.training.tul_setup import build_boundary_rule
            rule = build_boundary_rule(cfg)[0]      # (rule, lut, eos_id, substrings)
            sid = torch.from_numpy(span_ids_from_ids(inp.numpy(), rule))
            allow = span_allow_mask(sid)[:, 0]                       # [B, L, L]
            span_len = float(L) / float(sid.max(dim=1).values.float().mean() + 1.0)
        else:
            row = torch.arange(L).unsqueeze(1)
            allow = (torch.arange(L).unsqueeze(0) <= row).unsqueeze(0).expand(B, L, L)
            span_len = float("nan")
        return float(allow.sum(dim=2).float().mean()), span_len

    tc = tul_rt.model_cfg
    if tc.tg_geometry == "strict":
        allow = tg_strict_allow(layout, "coda",
                                coda_prefix_reach=tc.tg_coda_prefix_reach)[:, 0]
    elif tc.tg_restrict:
        allow = tg_allow_mask(layout, soft_prev_span=tc.tg_soft_prev_span)[:, 0]
    else:
        B, L = layout.bag_id.shape
        row = torch.arange(L).unsqueeze(1)
        allow = (torch.arange(L).unsqueeze(0) <= row).unsqueeze(0).expand(B, L, L)
    tok = ~layout.slot_mask                                           # [B, L]
    n_keys = allow.sum(dim=2).float()                                 # [B, L]
    vis = float(n_keys[tok].mean())
    n_tok = float(tok.sum())
    n_slots = float(layout.slot_valid.sum())
    return vis, (n_tok / n_slots if n_slots else float("nan"))


def cost_model(cfg, tul_rt, vis_keys: float, span_len: float) -> dict:
    """Inference cost per GENERATED token at decode. ARITHMETIC, not a measurement.

    Definitions, and they live in `docs/tul-as-memory.md` too:

    * ``block_passes`` — transformer block applications charged to one generated token.
      The token itself always pays ``n_prelude + n_coda``. What it pays for the CORE and
      for the slot CELLS depends on the forward:
        plain / depth-matched     ``P + C·T + D``
        slot loop                 ``P + D + (K·(P + D) + C·T) / span_len``
                                  — the cells' prelude/coda cost and the core's T passes
                                    over ONE compact state per slot, amortised over the
                                    span the slot serves;
        paid loop / loop_reads_tokens
                                  ``(P + C·T + D)·(1 + K/span_len)``
                                  — every position, tokens and cells, goes through the
                                    core.
      The span DECODER (`tul.spandec`) is training-only and is charged 0 here, which is
      stated rather than silently assumed.
    * ``visible_keys`` — mean attention key positions one decode step may read, MEASURED
      on real packed rows through the shipped allow relation.
    * ``kv_bytes`` — bf16 KV cache the step must hold: ``2 · (P + D) · n_kv_heads ·
      d_head · 2 bytes · visible_keys``. The CORE contributes nothing persistent: MORPH
      recomputes K/V from the current carrier at every iteration (runtime-invariants §6b,
      the reason the slot loop cannot use the token path's active-set shrinking), so there
      is no core KV to cache.
    """
    m = cfg.model
    P, C, D = int(m.n_prelude), int(m.n_core), int(m.n_coda)
    n_kv = int(m.n_kv_heads or m.n_heads)
    d_head = int(m.d_model) // int(m.n_heads) // int(m.compression)
    if tul_rt is None:
        T = int(m.mean_depth)
        K = 0
        passes = P + C * T + D
        mode = "plain"
    else:
        tc = tul_rt.model_cfg
        K = int(tc.prefix_k)
        if tc.tokens_through_core or tc.loop_reads_tokens:
            T = int(m.mean_depth)
            passes = (P + C * T + D) * (1.0 + K / span_len)
            mode = "paid" if tc.tokens_through_core else "loop_reads_tokens"
        else:
            T = int(tc.slot_mean_depth or m.mean_depth)
            passes = P + D + (K * (P + D) + C * T) / span_len
            mode = "slot_loop"
    kv_per_pos = 2 * (P + D) * n_kv * d_head * BYTES_PER_KV
    return {"mode": mode, "n_prelude": P, "n_core": C, "n_coda": D, "eval_depth": T,
            "prefix_k": K, "span_len_tokens": span_len,
            "block_passes_per_token": passes,
            "visible_keys_per_token": vis_keys,
            "kv_bytes_per_position": kv_per_pos,
            "kv_bytes_per_token": kv_per_pos * vis_keys,
            "spandec_block_passes_at_decode": 0.0}


# ── the CE half ──────────────────────────────────────────────────────────────

def _load_npz(path: str, depth: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    key = f"ce_{depth}"
    if key not in z:
        raise SystemExit(f"{path} has no {key} (has {sorted(k for k in z if k != 'tok_index')})")
    return z["tok_index"].astype(np.int64), z[key].astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=CONFIG=TOKENS_NPZ[=DEPTH]; DEPTH defaults to --depth")
    ap.add_argument("--depth", type=int, default=6,
                    help="the loop depth to score at; an arm may override it in its --arm")
    ap.add_argument("--span-local", default="", help="arm name that defines CE_spanlocal")
    ap.add_argument("--full", default="", help="arm name that defines CE_full")
    ap.add_argument("--allow-core-mismatch", action="store_true",
                    help="print the recovered-fraction column even when the budget pair's "
                         "core_impl differs from an arm's (it is then a scale measured on "
                         "a different model; the prereg queues the matched control)")
    ap.add_argument("--rows", type=int, default=48,
                    help="packed rows used for the span-length and visible-key MEASUREMENT "
                         "(not for CE, which comes from the sweep npz)")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    arms: dict[str, dict] = {}
    ce_idx: dict[str, np.ndarray] = {}
    ce_val: dict[str, np.ndarray] = {}
    for spec in a.arm:
        parts = spec.split("=", 3)
        if len(parts) < 3:
            raise SystemExit(f"--arm needs NAME=CONFIG=NPZ, got {spec!r}")
        name, config, npz = parts[0], parts[1], parts[2]
        depth = int(parts[3]) if len(parts) == 4 and parts[3] else a.depth
        cfg = build_cfg(config, ["model.use_kernels=false"])
        tul_rt = build_tul_runtime(cfg)
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        plain = tul_rt is None
        row_tokens = (int(cfg.data.seq_len) + 1 if plain
                      else tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1)
        stream = stream_from_loader(loader, a.rows * row_tokens)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, plain)[:-(-a.rows // a.batch)]
        vis, span_len = [], []
        for inp, _lab, layout, _idx in batches:
            v, s = _visible_keys(cfg, tul_rt, layout, inp)
            vis.append(v)
            span_len.append(s)
        vis_m = float(np.mean(vis))
        span_m = float(np.nanmean(span_len)) if not all(np.isnan(span_len)) else float("nan")
        cost = cost_model(cfg, tul_rt, vis_m, span_m)
        idx, ce = _load_npz(npz, depth)
        ce_idx[name], ce_val[name] = idx, ce
        arms[name] = {"config": config, "tokens_npz": npz, "depth": depth,
                      "core_impl": str(getattr(cfg.model, "core_impl", "morph")),
                      "ternary_scale_mode": str(cfg.training.ternary_scale_mode),
                      "span_mask": str(getattr(cfg.model, "span_mask", "off")),
                      "seq_len": int(cfg.data.seq_len),
                      "measure_rows": int(sum(i.shape[0] for i, *_ in batches)),
                      "cost": cost}
        print(f"{name:34s} depth={depth} passes/token={cost['block_passes_per_token']:8.2f} "
              f"keys/token={vis_m:8.1f} span_len={span_m:6.2f} "
              f"kv_B/token={cost['kv_bytes_per_token']:12.0f}", flush=True)

    # ── paired CE on the INTERSECTION of every arm's scored stream positions ──
    common = None
    for n in arms:
        s = set(ce_idx[n].tolist())
        common = s if common is None else (common & s)
    common_arr = np.array(sorted(common), dtype=np.int64)
    if common_arr.size == 0:
        raise SystemExit("the arms share no scored stream position; they were swept on "
                         "different validation streams")
    for n in arms:
        order = np.argsort(ce_idx[n])
        pos = np.searchsorted(ce_idx[n][order], common_arr)
        sel = order[pos]
        assert np.array_equal(ce_idx[n][sel], common_arr)
        arms[n]["ce"] = float(ce_val[n][sel].mean())
        arms[n]["n_paired_tokens"] = int(common_arr.size)
    print(f"\npaired on {common_arr.size} stream positions shared by all "
          f"{len(arms)} arms", flush=True)

    # ── the recovered fraction ───────────────────────────────────────────────
    note = None
    if a.span_local and a.full:
        if a.span_local not in arms or a.full not in arms:
            raise SystemExit("--span-local / --full must name arms passed with --arm")
        lo, hi = arms[a.span_local]["ce"], arms[a.full]["ce"]
        denom = lo - hi
        cores = {arms[n]["core_impl"] for n in arms}
        mismatch = len(cores) > 1
        if mismatch and not a.allow_core_mismatch:
            note = (f"recovered fraction WITHHELD: the arms do not share a core "
                    f"({sorted(cores)}). The budget pair's CE is a scale measured on a "
                    f"different model. Pass --allow-core-mismatch to print it anyway, or "
                    f"run the matched `plain-span-local` control "
                    f"(lab/experiments/planned/2026-09-13-arc-tul-as-memory.md).")
        else:
            for n in arms:
                arms[n]["budget_recovered"] = (lo - arms[n]["ce"]) / denom if denom else None
            if mismatch:
                note = (f"recovered fraction printed under --allow-core-mismatch: cores "
                        f"{sorted(cores)} differ, so the denominator "
                        f"{denom:.4f} is a scale from a different model.")
        print(f"\nCE_spanlocal={lo:.4f} ({a.span_local})  CE_full={hi:.4f} ({a.full})  "
              f"budget={denom:+.4f}", flush=True)
    if note:
        print("\n" + note, flush=True)

    out = {"arms": arms, "paired_tokens": int(common_arr.size),
           "span_local_arm": a.span_local, "full_arm": a.full, "note": note}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)

    rows = sorted(arms.items(), key=lambda kv: kv[1]["ce"])
    hdr = ("| arm | CE @depth | block-passes/token | visible keys/token | KV B/token | "
           "budget recovered | source npz |")
    lines = [hdr, "|" + "---|" * 7]
    for n, e in rows:
        br = e.get("budget_recovered")
        lines.append(
            f"| `{n}` | {e['ce']:.4f} @{e['depth']} | {e['cost']['block_passes_per_token']:.2f} "
            f"| {e['cost']['visible_keys_per_token']:.1f} "
            f"| {e['cost']['kv_bytes_per_token']:,.0f} "
            f"| {'—' if br is None else f'{br:.3f}'} "
            f"| `{os.path.basename(e['tokens_npz'])}` |")
    md = "\n".join(lines)
    print("\n" + md)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md + "\n")
        print("wrote", a.md)


if __name__ == "__main__":
    main()
