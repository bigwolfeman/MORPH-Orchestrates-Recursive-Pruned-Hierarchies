"""Aggregate the toy slot-loop grid JSONs into the markdown tables used by WRITEUP.md.

    python aggregate.py <dir-with-jsons> [more dirs...]

Prints every table to stdout. Mean and [min, max] over the seeds of a cell.
"""

from __future__ import annotations

import glob
import json
import os
import re
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
    """Cell identity = the label with a TRAILING -s<digits> removed.

    A plain rsplit on "-s" also cuts "ladder3-strict-..." in half, so this anchors.
    """
    return re.sub(r"-s\d+$", "", r["label"])


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
        "| cell | n | escaped | escape step | value CE @6 | value acc @6 | K1-K6 (value CE) "
        "| MUX CE @6 | K1-K6 (MUX) | token CE @6 | s/run |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        esc = [r for r in rs if r.get("escaped")]
        es = f"{len(esc)}/{len(rs)}"
        step = f"{sum(r['escape_step'] for r in esc)/len(esc):.0f}" if esc else "-"
        print(
            f"| `{k}` | {len(rs)} | {es} | {step} "
            f"| {plain([r['k_curve']['6']['value_ce'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['value_acc'] for r in rs], 3)} "
            f"| {fmt([r['k1_k6_value'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['mux_ce'] for r in rs])} "
            f"| {fmt([r['k1_k6_mux'] for r in rs])} "
            f"| {plain([r['k_curve']['6']['token_ce'] for r in rs])} "
            f"| {agg([r['train_seconds'] for r in rs])[0]:.0f} |"
        )


def table_kcurve(g, keys, title, depths=None):
    print(f"\n### {title}\n")
    depths = depths or ["1", "2", "3", "6", "8", "12"]
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


# --------------------------------------------------------------------------------------
# `eliminate` tables (2026-09-18)
# --------------------------------------------------------------------------------------

def _plateaus(row):
    """The derived plateaus of an elimination task, from the ceilings stored in the run."""
    c = row["eliminate"]["ceilings"]
    out = []
    for d, v in sorted(c["reachability"].items(), key=lambda kv: int(kv[0])):
        if v > 1e-9:
            out.append((f"depth {d} floor", v))
    for k, v in sorted(c["point_carry"].items(), key=lambda kv: int(kv[0])):
        if v > 1e-9:
            out.append((f"point carry at {k}", v))
    out.append(("solved", 0.0))
    return out


def _roles(row):
    """Role names, from the run's own spec. Runs from before 2026-09-19 carry no spec, so
    fall back to the role names their `candidate_mass` block actually has."""
    sp = row["eliminate"].get("spec")
    if sp:
        return [r[0] for r in sp["roles"]]
    return list(row["eliminate"]["candidate_mass"].keys())


def _passes(row):
    sp = row["eliminate"].get("spec")
    d = sp["probe_depth"] if sp else len(row["eliminate"]["candidate_mass"]["answer"]) - 1
    return tuple(sorted({0, 1, 2, d // 2, d - 2, d}))


def table_elim_plateau(g, keys):
    print("\n### where each seed landed, against the enumerated plateaus\n")
    print("| cell | seed | value CE @6 | value acc @6 | nearest plateau | distance |")
    print("|---|---|---|---|---|---|")
    for k in keys:
        for r in sorted(g[k], key=lambda r: r["seed"]):
            ce = r["k_curve"]["6"]["value_ce"]
            name, v = min(_plateaus(r), key=lambda p: abs(ce - p[1]))
            print(
                f"| `{k}` | {r['seed']} | {ce:.4f} | {r['k_curve']['6']['value_acc']:.3f} "
                f"| {name} ({v:.4f}) | {ce - v:+.4f} |"
            )


def _split(rs):
    return [r for r in rs if r.get("escaped")], [r for r in rs if not r.get("escaped")]


def table_elim_mass(g, keys, role):
    print(f"\n### candidate mass at role `{role}` (solved seeds | stuck seeds)\n")
    print("| cell | pass | n | survivor | dead (reachable) | alive non-survivor | entropy | top mass | alive set |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        if not g[k] or role not in g[k][0]["eliminate"]["candidate_mass"]:
            continue
        for gname, rs in (("solved", _split(g[k])[0]), ("stuck", _split(g[k])[1])):
            if not rs:
                continue
            for t in _passes(rs[0]):
                c = [r["eliminate"]["candidate_mass"][role][t] for r in rs]
                n = len(c)
                cells = " | ".join(
                    f"{sum(x[f] for x in c)/n:.3f}"
                    for f in ("survivor", "eliminated_reachable", "alive_non_survivor",
                              "entropy", "top_mass")
                )
                print(f"| `{k}` {gname} | {t} | {n} | {cells} | {c[0]['n_alive_reachable']} |")


def _mean_or_dash(vals, p=3):
    vals = [v for v in vals if v is not None]
    return f"{sum(vals)/len(vals):.{p}f}" if vals else "-"


def table_elim_probe(g, keys, fact="alive"):
    keys = [k for k in keys if g[k] and f"acc_{fact}" in g[k][0]["eliminate"]["membership_probe"]["answer"][0]]
    if not keys:
        return
    print(f"\n### membership probe, balanced accuracy on held-out rows, fact `{fact}`\n")
    ps = _passes(next(g[k][0] for k in keys if g[k]))
    print("| cell | role | " + " | ".join(f"p{t}" for t in ps) + " |")
    print("|---" * (2 + len(ps)) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        for role in _roles(rs[0]):
            cells = [
                _mean_or_dash([r["eliminate"]["membership_probe"][role][t][f"acc_{fact}"] for r in rs])
                for t in ps
            ]
            print(f"| `{k}` | {role} | " + " | ".join(cells) + " |")


def table_elim_hop(g, keys):
    """The answer slot's probe accuracy against hop distance: pass t reaches hop t."""
    rs0 = next(
        (g[k][0] for k in keys
         if g[k] and "acc_in_set" in g[k][0]["eliminate"]["membership_probe"]["answer"][0]),
        None,
    )
    if rs0 is None:
        return  # runs from before 2026-09-19 fitted the `alive` fact only
    keys = [k for k in keys
            if g[k] and "acc_in_set" in g[k][0]["eliminate"]["membership_probe"]["answer"][0]]
    print("\n### hop ladder: the ANSWER slot's probe accuracy at pass t (= hop distance t)\n")
    d = rs0["eliminate"]["spec"]["probe_depth"]
    ps = list(range(d + 1))
    print("| cell | fact | " + " | ".join(f"hop {t}" for t in ps) + " |")
    print("|---" * (2 + len(ps)) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        for fact in ("in_set", "dead", "alive"):
            cells = [
                _mean_or_dash(
                    [r["eliminate"]["membership_probe"]["answer"][t][f"acc_{fact}"] for r in rs]
                )
                for t in ps
            ]
            print(f"| `{k}` | {fact} | " + " | ".join(cells) + " |")


def table_elim_reader(g, keys):
    rs0 = next((g[k][0] for k in keys if g[k] and "adapted_reader" in g[k][0]["eliminate"]), None)
    if rs0 is None:
        return  # runs from before 2026-09-19 have no fitted reader
    keys = [k for k in keys if g[k] and "adapted_reader" in g[k][0]["eliminate"]]
    print("\n### adapted linear reader on the exit state against the coda, value CE (nats)\n")
    ds = [row["depth"] for row in rs0["eliminate"]["adapted_reader"]["by_depth"]]
    print("| cell | source | " + " | ".join(f"d={d}" for d in ds) + " |")
    print("|---" * (2 + len(ds)) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        rd = [
            _mean_or_dash([r["eliminate"]["adapted_reader"]["by_depth"][i]["reader_value_ce"] for r in rs], 4)
            for i in range(len(ds))
        ]
        cd = [
            _mean_or_dash([r["k_curve"][str(d)]["value_ce"] for r in rs], 4) for d in ds
        ]
        print(f"| `{k}` | fitted reader | " + " | ".join(rd) + " |")
        print(f"| `{k}` | coda | " + " | ".join(cd) + " |")
    print("\n### the fitted reader at each role slot, at the probe depth\n")
    print("| cell | " + " | ".join(f"{r} CE / acc" for r in _roles(rs0)) + " |")
    print("|---" * (1 + len(_roles(rs0))) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        cells = []
        for i, role in enumerate(_roles(rs[0])):
            ce = _mean_or_dash([r["eliminate"]["adapted_reader"]["by_slot"][i]["reader_value_ce"] for r in rs], 4)
            ac = _mean_or_dash([r["eliminate"]["adapted_reader"]["by_slot"][i]["reader_value_acc"] for r in rs], 3)
            cells.append(f"{ce} / {ac}")
        print(f"| `{k}` | " + " | ".join(cells) + " |")


def table_elim_twin(g, keys):
    print("\n### twin divergence (rows differing ONLY at the first elimination)\n")
    rs0 = next((g[k][0] for k in keys if g[k]), None)
    if rs0 is None:
        return
    ps = _passes(rs0)
    print("| cell | role | drop pass (expected) | " + " | ".join(f"cos p{t}" for t in ps) + " |")
    print("|---" * (3 + len(ps)) + "|")
    for k in keys:
        rs = g[k]
        if not rs:
            continue
        for role in _roles(rs[0]):
            c = [r["eliminate"]["twin_divergence"][role] for r in rs]
            drops = sorted({str(x["drop_pass"]) for x in c})
            exp = c[0]["expected_drop_pass"]
            cos = [f"{sum(x['cosine'][t] for x in c)/len(c):.4f}" for t in ps]
            print(f"| `{k}` | {role} | {'/'.join(drops)} ({exp}) | " + " | ".join(cos) + " |")


def main_eliminate(rows, g, task):
    keys = sorted({key_of(r) for r in rows})
    print(f"\n## `{task}`: {len(rows)} runs in {len(keys)} cells")
    cl = rows[0]["eliminate"]["ceilings"]
    def ladder(d):
        return " / ".join(f"{d[k]:.4f}" for k in sorted(d, key=int))
    print(f"\nchance {cl['chance']:.4f}")
    print(f"reachability by forced depth: {ladder(cl['reachability'])}")
    print(f"commitment by evidence seen:  {ladder(cl['commitment'])}")
    print(f"point carry:                  {ladder(cl['point_carry'])}")
    table_main(g, keys, f"`{task}` grid")
    kd = [d for d in rows[0]["k_curve"]]
    table_kcurve(g, keys, f"`{task}` value CE against forced depth", kd)
    table_elim_plateau(g, keys)
    table_elim_reader(g, keys)
    table_elim_hop(g, keys)
    for role in _roles(rows[0]):
        table_elim_mass(g, keys, role)
    for fact in ("alive", "in_set", "dead"):
        table_elim_probe(g, keys, fact)
    table_elim_twin(g, keys)
    table_write(g, keys, f"`{task}` write contribution")
    table_grad(g, keys, f"`{task}` per-pass gradient", "total")


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

    el = [r for r in rows if r["task"] in {"eliminate", "eliminate6"} and "eliminate" in r]
    for t in ("eliminate", "eliminate6"):
        sub = [r for r in el if r["task"] == t]
        if sub:
            main_eliminate(sub, group(sub), t)
    if el and len(el) == len(rows):
        return

    atts =["exit", "mux_all", "mux_all_detach", "staged", "deep_coda", "progressive"]
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
        "B-fixed_depth6",
    ]
    table_main(g, bkeys, "Grid B, one-factor extensions on `compose` at attachment `exit`")
    table_kcurve(g, bkeys, "Grid B value CE against forced depth")
    table_write(g, bkeys, "Grid B write contribution")
    table_grad(g, bkeys, "Grid B per-pass gradient", "total")

    ckeys = ["A-compose-exit", "C-permissive-exit", "C-permissive-mux_all"]
    table_main(g, ckeys, "Grid C, MORPH's own (permissive) geometry against the strict one")
    table_kcurve(g, ckeys, "Grid C value CE against forced depth")

    # per-span value CE at depth 6: the staircase
    print("\n### Value CE by span index at forced depth 6 (answer at the head of span j+1 needs j passes)\n")
    print("| cell | " + " | ".join(f"R{j}" for j in range(7)) + " |")
    print("|---" * 8 + "|")
    for k in [f"A-compose-{a}" for a in atts] + bkeys[1:] + ckeys[1:]:
        rs = g[k]
        if not rs or "value_ce_by_span" not in rs[0]["k_curve"]["6"]:
            continue
        m = [sum(r["k_curve"]["6"]["value_ce_by_span"][j] for r in rs) / len(rs) for j in range(7)]
        print(f"| `{k}` | " + " | ".join(f"{v:.2f}" for v in m) + " |")

    # ladder
    lkeys = sorted(k for k in g if k.startswith("ladder") or k.startswith("diag"))
    if lkeys:
        table_main(g, lkeys, "Capacity ladder (trained at a fixed depth)")
        table_kcurve(g, lkeys, "Capacity ladder value CE against forced depth")

    # escaped against stuck, pooled over every strict `compose` run
    pool = [r for r in rows if r["task"] == "compose" and r["config"].get("geometry") == "strict"
            and r["label"].startswith(("A-", "B-"))]
    esc = [r for r in pool if r.get("escaped")]
    stk = [r for r in pool if not r.get("escaped")]
    if esc and stk:
        print(f"\n### Escaped against stuck, pooled over {len(pool)} strict `compose` runs\n")
        print("| group | n | value CE @6 | cotangent share p1..p6 | dW share p1..p6 "
              "| mean pairwise cos | cancellation | z rank |")
        print("|---|---|---|---|---|---|---|---|")
        for name, sub in (("escaped", esc), ("stuck", stk)):
            p6 = [r["gradient_probe"]["total"] for r in sub]
            n = len(p6)
            cs = " ".join(f"{sum(x['cotangent_share'][i] for x in p6)/n:.3f}" for i in range(6))
            ws = " ".join(f"{sum(x['dW_share'][i] for x in p6)/n:.3f}" for i in range(6))
            pc = sum(x["pairwise_cos_mean"] for x in p6) / n
            print(
                f"| {name} | {n} | {plain([r['k_curve']['6']['value_ce'] for r in sub], 3)} "
                f"| {cs} | {ws} | {pc:+.3f} "
                f"| {plain([x['cancellation'] for x in p6], 3)} "
                f"| {agg([r['participation']['z_rank'] for r in sub])[0]:.1f} |"
            )

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
