"""Score the objective split against its frozen predictions O0-O5.

Pre-registration: lab/experiments/planned/2026-08-27-objective-split.md
Written before any checkpoint ran through the probe, so no threshold here is
fitted to the data.

Same refusal discipline as score_0827_arms.py: a letter whose input is missing
prints NOT MEASURED, and O0 (the validity gate) voids the whole panel rather
than annotating it. A cosine whose per-batch spread crosses zero is reported as
UNDECIDED, never as its accumulated sign — batch 6 is small for a cosine and the
pre-registration says so up front.

Usage:
    python lab/divergence/score_split.py /home/wolfe/morph-scratch/split
"""
from __future__ import annotations

import argparse
import glob
import json
import os

ADD_TOL = 2e-2          # additivity, as in the probe
DET_TOL = 0.99          # determinism, as in the probe
CONFLICT = -0.05        # O1 / O5 boundary
DOMINATE = 0.25         # O2: ||g_main|| / ||g_emit||
EMIT_SHARE = 0.5        # O3


def load(d: str) -> dict[str, dict]:
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        out[os.path.basename(f)[:-5]] = json.load(open(f))
    return out


def spread(rec: dict, key: str) -> tuple[float, float] | None:
    vals = [b[key] for b in rec.get("cos_per_batch", []) if key in b]
    return (min(vals), max(vals)) if vals else None


def cos_verdict(rec: dict, key: str) -> tuple[float | None, str]:
    """Accumulated cosine plus a label that refuses a sign the batches do not agree on."""
    c = rec.get("cos", {}).get(key)
    if c is None:
        alt = key.split("~")
        c = rec.get("cos", {}).get(f"{alt[1]}~{alt[0]}")
        key = f"{alt[1]}~{alt[0]}"
    if c is None:
        return None, "NOT MEASURED"
    sp = spread(rec, key)
    if sp and sp[0] < 0.0 < sp[1]:
        return c, f"UNDECIDED (per-batch spread {sp[0]:+.3f}..{sp[1]:+.3f} crosses 0)"
    lab = "CONFLICT" if c < CONFLICT else ("orthogonal" if abs(c) <= 0.05 else "aligned")
    return c, lab + (f" [{sp[0]:+.3f},{sp[1]:+.3f}]" if sp else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    recs = load(a.dir)
    if not recs:
        raise SystemExit(f"no *.json in {a.dir}")

    print("O0  VALIDITY GATE — additivity and determinism on every checkpoint")
    bad = []
    for nm, r in recs.items():
        wa = max(r.get("additivity_rel_err", [9e9]))
        wd = min(r.get("determinism_self_cos", [-1.0]))
        ok = wa <= ADD_TOL and wd >= DET_TOL
        print(f"    {nm:<18} add {wa:.2e}  det {wd:.6f}   {'pass' if ok else 'FAIL'}")
        if not ok:
            bad.append(nm)
    if bad:
        print(f"\n    O0 FAILED on {', '.join(bad)}. The panel is VOID — a decomposition")
        print("    that does not sum to the objective cannot separate conflict from noise.")
        raise SystemExit(1)
    print("    passed everywhere\n")

    print(f"{'checkpoint':<18} {'||g_main||':>11} {'||g_emit||':>11} {'||g_plast||':>11} "
          f"{'||g_mux||':>10}  {'main/emit':>9}")
    print("-" * 80)
    for nm, r in recs.items():
        n = r["norms"]
        g = lambda k: n.get(k, n.get(k + "*", float("nan")))    # noqa: E731
        ratio = g("main") / g("emit") if g("emit") else float("nan")
        print(f"{nm:<18} {g('main'):>11.3e} {g('emit'):>11.3e} {g('plast'):>11.3e} "
              f"{n.get('mux', float('nan')):>10.3e}  {ratio:>9.3f}")
    print("  (a norm read from a `*` key is a weight-0 direction: measured, but NOT part")
    print("   of that arm's objective — the emit column of any v1a2b/warmup arm is that.)")

    def rec_for(sub: str) -> tuple[str, dict] | tuple[None, None]:
        for nm, r in recs.items():
            if sub in nm:
                return nm, r
        return None, None

    print("\nPREDICTIONS")
    nm, r = rec_for("ctrl-s2-3000")
    if r is None:
        print("  O1  NOT MEASURED (no ctrl-s2-3000)")
        print("  O2  NOT MEASURED (no ctrl-s2-3000)")
    else:
        c, lab = cos_verdict(r, "main~emit")
        held = c is not None and c > CONFLICT and "UNDECIDED" not in lab
        print(f"  O1  cos(g_main, g_emit) > {CONFLICT} on the healthy control: "
              f"{c:+.4f} {lab} -> {'HELD' if held else 'FAILED'}")
        n = r["norms"]
        ratio = n.get("main", float('nan')) / n.get("emit", n.get("emit*", float('nan')))
        print(f"  O2  ||g_main||/||g_emit|| < {DOMINATE}: {ratio:.3f} "
              f"-> {'HELD' if ratio < DOMINATE else 'FAILED'}")

    nm, r = rec_for("ctrl-s1-3000")
    if r is None:
        print("  O3  NOT MEASURED (no taken-over control checkpoint)")
    else:
        n = r["norms"]
        tot = sum(v for k, v in n.items() if not k.endswith("*"))
        sh = n.get("emit", 0.0) / tot if tot else float("nan")
        print(f"  O3  emit share of the core gradient > {EMIT_SHARE} on the taken-over "
              f"control: {sh:.3f} -> {'HELD' if sh > EMIT_SHARE else 'FAILED'}")

    nm, r = rec_for("v1a2b")
    if r is None:
        print("  O4  NOT MEASURED (no v1a2b checkpoint)")
    else:
        c, lab = cos_verdict(r, "mux~main")
        held = c is not None and c > 0 and "UNDECIDED" not in lab
        print(f"  O4  cos(g_mux, g_main) > 0 on the MUX arm: "
              f"{c if c is not None else float('nan'):+.4f} {lab} "
              f"-> {'HELD' if held else 'FAILED' if c is not None else 'NOT MEASURED'}")

    nm, r = rec_for("ctrl-s2-3000")
    print("\n  O5  THE DECISION")
    if r is None:
        print("      NOT MEASURED")
    else:
        c, lab = cos_verdict(r, "main~emit")
        if c is None or "UNDECIDED" in lab:
            print("      UNDECIDED — the per-batch cosine spread crosses zero. More batches")
            print("      before any architecture decision; the sign is not established.")
        elif c < CONFLICT:
            print(f"      CONFLICT ({c:+.4f}). An MTP-shaped chain adds a FOURTH objective to")
            print("      a core whose objectives already fight. Contraindicated. Projection")
            print("      (PCGrad/CAGrad) or spectral decoupling comes first.")
        else:
            n = r["norms"]
            ratio = n.get("main", 0.0) / n.get("emit", n.get("emit*", 1.0))
            mode = "STARVATION" if c > 0.05 else "DOMINATION"
            print(f"      {mode} ({c:+.4f}, ||g_main||/||g_emit|| = {ratio:.3f}). The coda's")
            print("      route is not fought, it is small. Adding targets to the core is the")
            print("      indicated fix, and the MTP-shaped chain is on the table.")

    print("\nEvery verdict above reads one gradient at one point. It says which fix the")
    print("landscape admits, NOT which objective caused the state the checkpoint is in.")


if __name__ == "__main__":
    main()
