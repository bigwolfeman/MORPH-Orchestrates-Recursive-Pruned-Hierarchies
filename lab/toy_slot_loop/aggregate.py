"""Aggregate the toy slot-loop grid JSONs into the markdown tables used by WRITEUP.md.

    python aggregate.py <dir-with-jsons> [more dirs...]

Prints every table to stdout. Mean and [min, max] over the seeds of a cell.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict


def load(dirs):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            with open(f) as fh:
                rows.append(json.load(fh))
    return rows


def key_of(r):
    """Cell identity = the label with the trailing seed marker removed."""
    lab = r["label"]
    return lab.rsplit("-s", 1)[0] if "-s" in lab else lab


def agg(vals):
    vals = list(vals)
    m = sum(vals) / len(vals)
    return m, min(vals), max(vals)


def fmt(vals, p=4):
    m, lo, hi = agg(vals)
    return f"{m:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]" if p else f"{m:.3f}"


def plain(vals, p=4):
    m, lo, hi = agg(vals)
    return f"{m:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[key_of(r)].append(r)
    return g


def get(r, path, default=None):
    o = r
    for k in path.split("."):
        if o is None:
            return default
        o = o.get(k) if isinstance(o, dict) else None
    return default if o is None else o


def table_main(g, keys, title, cols=None):
    print(f"\n### {title}\n")
    print(
        "| cell | n | value CE @6 | value acc @6 | K1-K6 (value CE) | K3-K6 | MUX CE @6 | "
        "K1-K6 (MUX) | token CE @6 | s/run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        print(
            f"| `{k}` | {len(rs)} "
            f"| {plain([r['k_curve']['6']['value_ce'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['value_acc'] for r in rs], 3)} "
            f"| {fmt([r['k1_k6_value'] for r in rs])} "
            f"| {fmt([r['k3_k6_value'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['mux_ce'] for r in rs])} "
            f"| {fmt([r['k1_k6_mux'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['token_ce'] for r in rs])} "
            f"| {agg([r['train_seconds'] for r in rs])[0]:.0f} |"
        )


def table_kcurve(g, keys, title):
    print(f"\n### {title}\n")
    depths = ["1", "2", "3", "6", "8", "12"]
    print("| cell | " + " | ".join(f"d={d}" for d in depths) + " |")
    print("|---" * (len(depths) + 1) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        cells = [f"{sum(r['k_curve'][d]['value_ce'] for r in rs)/len(rs):.3f}" for d in depths]
        print(f"| `{k}` | " + " | ".join(cells) + " |")


def table_write(g, keys, title):
    print(f"\n### {title}\n")
    print("| cell | value CE, z=exit | z=entry | z=0 | entry-exit | zero-exit | z rank | entry rank |")
    print("|---|---|---|---|---|---|---|---|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        e = [r["write_contribution"]["exit_value_ce"] for r in rs]
        en = [r["write_contribution"]["entry_value_ce"] for r in rs]
        z0 = [r["write_contribution"]["zero_value_ce"] for r in rs]
        print(
            f"| `{k}` | {plain(e, 3)} | {plain(en, 3)} | {plain(z0, 3)} "
            f"| {fmt([b - a for a, b in zip(e, en)], 3)} | {fmt([b - a for a, b in zip(e, z0)], 3)} "
            f"| {agg([r['participation']['z_rank'] for r in rs])[0]:.1f} "
            f"| {agg([r['participation']['entry_rank'] for r in rs])[0]:.1f} |"
        )


def table_grad(g, keys, title, source="total"):
    print(f"\n### {title} (source: {source})\n")
    print(
        "| cell | cotangent share p1..p6 | dW share p1..p6 | cos(dW_t, total) p1..p6 | "
        "mean pairwise cos | cancellation |"
    )
    print("|---|---|---|---|---|---|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        p = [r["gradient_probe"][source] for r in rs]
        n = len(p)

        def avg(field):
            return [sum(x[field][i] for x in p) / n for i in range(6)]

        cs = " ".join(f"{v:.3f}" for v in avg("cotangent_share"))
        ws = " ".join(f"{v:.3f}" for v in avg("dW_share"))
        co = " ".join(f"{v:+.2f}" for v in avg("cos_to_total"))
        pc = sum(x["pairwise_cos_mean"] for x in p) / n
        ca = [x["cancellation"] for x in p]
        print(f"| `{k}` | {cs} | {ws} | {co} | {pc:+.3f} | {plain(ca, 3)} |")


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = sum((a - mx) ** 2 for a in xs) ** 0.5
    dy = sum((b - my) ** 2 for b in ys) ** 0.5
    return num / (dx * dy + 1e-12)


def main():
    rows = load(sys.argv[1:])
    g = group(rows)
    print(f"loaded {len(rows)} runs in {len(g)} cells")
    sc = max(r["gradient_probe"]["total"]["selfcheck_max_rel_err"] for r in rows)
    print(f"\nper-pass tap self-check, worst over every run: max rel err {sc:.2e}")

    atts = ["exit", "mux_all", "mux_all_detach", "staged", "deep_coda", "progressive"]
    for task in ("compose", "summary"):
        keys = [f"A-{task}-{a}" for a in atts]
        table_main(g, keys, f"Grid A, task `{task}`")
        table_kcurve(g, keys, f"Grid A value CE against forced depth, task `{task}`")
        table_write(g, keys, f"Grid A write contribution, task `{task}`")
        for src in ("total", "token_ce", "mux"):
            table_grad(g, keys, f"Grid A per-pass gradient, task `{task}`", src)

    bkeys = [
        "A-compose-exit",
        "B-noise_entry",
        "B-decay09",
        "B-inject_off",
        "B-fixed_point",
        "B-pass_lora",
        "B-coda_blind",
        "B-bptt_last2",
    ]
    table_main(g, bkeys, "Grid B, one-factor extensions on `compose` at attachment `exit`")
    table_kcurve(g, bkeys, "Grid B value CE against forced depth")
    table_write(g, bkeys, "Grid B write contribution")
    table_grad(g, bkeys, "Grid B per-pass gradient", "total")

    # ladder
    lkeys = sorted(k for k in g if k.startswith("ladder"))
    if lkeys:
        table_main(g, lkeys, "Capacity ladder (trained at a fixed depth)")
        table_kcurve(g, lkeys, "Capacity ladder value CE against forced depth")

    # P8: cancellation against earning over grid A
    ga = [r for r in rows if r["label"].startswith("A-")]
    if ga:
        xs = [r["gradient_probe"]["total"]["cancellation"] for r in ga]
        ys = [r["k1_k6_value"] for r in ga]
        print(f"\nP8: corr(cancellation, K1-K6 value CE) over {len(ga)} grid-A runs = {corr(xs, ys):+.3f}")
        for task in ("compose", "summary"):
            sub = [r for r in ga if r["task"] == task]
            print(
                f"  {task}: cancellation mean {sum(r['gradient_probe']['total']['cancellation'] for r in sub)/len(sub):.3f}, "
                f"K1-K6 mean {sum(r['k1_k6_value'] for r in sub)/len(sub):+.4f}"
            )


if __name__ == "__main__":
    main()
