"""Score H18 against its pre-registered predictions, without judgement calls.

Pre-registration: docs/experiments/planned/2026-08-25-h18-positional-attention-sink.md

Written and COMMITTED BEFORE the ladder ran, so no threshold here can be fitted to the
data. The validity gate runs first and refuses the whole panel, in the same spirit as
`score_stage0.py` and `score_scse.py`: this campaign has twice read verdicts off a panel
that had not earned them.

Usage:
    python lab/divergence/score_h18.py --json attn_sink.json
"""
from __future__ import annotations

import argparse
import json

# ── pre-registered constants ───────────────────────────────────────────────────────
HEALTHY = (1625, 1650, 1675, 1700, 1725, 1750, 1775)
SICK = (1800, 1850, 1866)
AMBIGUOUS = (1825,)          # the README's "falls back" rung, excluded from BOTH sets

SELFTEST_MAX = 2.0e-2        # V1; the bf16 noise floor is 7.8e-3
P1_MIN_BLOCKS = 4            # sick rungs: blocks with ratio < 1
P2_MAX_BLOCKS = 3            # healthy rungs: blocks with ratio < 1
P3_MIN_GAP = 0.10            # mean(healthy ratio) - mean(sick ratio)
P4_MIN_AGREE = 0.80          # row_agree and cross-batch argmax agreement
P4_MIN_BLOCKS = 4
P5_MIN_RISE = 0.20           # top1 at 1850 vs 1625, relative
REFUTE_TOL = 0.10            # |healthy - sick| / healthy on pr, at EVERY cell


def win_rows(rung: dict, batch: int = 0) -> list[dict]:
    return [r for r in rung["batches"][batch] if r["branch"] == "win"]


def ratio_by_block(rung: dict) -> dict[int, float]:
    """`pr(last iteration) / pr(iteration 0)` per core block. Below 1 = concentrating."""
    out = {}
    rows = win_rows(rung)
    for b in sorted({r["block"] for r in rows}):
        rb = sorted([r for r in rows if r["block"] == b], key=lambda z: z["t"])
        out[b] = rb[-1]["pr"] / max(rb[0]["pr"], 1e-30)
    return out


def cross_batch_agree(rung: dict) -> float:
    if len(rung["batches"]) < 2:
        return float("nan")
    key = lambda z: (z["branch"], z["block"], z["t"])          # noqa: E731
    m1 = {key(z): z["argmax"] for z in rung["batches"][1]}
    hits = [key(z) in m1 and m1[key(z)] == z["argmax"] for z in rung["batches"][0]]
    return sum(hits) / max(1, len(hits))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    data = json.load(open(a.json))
    by_step = {int(r["step"]): r for r in data}

    # ── validity gate ──────────────────────────────────────────────────────────────
    print("VALIDITY GATE")
    fails = []
    iters = set()
    for st, r in sorted(by_step.items()):
        for bi, rows in enumerate(r["batches"]):
            worst = max(z["self_test"] for z in rows if z["branch"] == "win")
            if worst > SELFTEST_MAX:
                fails.append(f"V1 {st}/b{bi}: window self-test {worst:.2e}")
            iters.add(max(z["t"] for z in rows if z["branch"] == "win") + 1)
    if len(iters) != 1:
        fails.append(f"V3: loop-iteration count varies across rungs/batches: {sorted(iters)}")
    if fails:
        print("  FAILED: " + "; ".join(fails))
        print("  V2 raises inside the probe, so reaching this point means it held.")
        raise SystemExit(1)
    print(f"  V1 window self-test <= {SELFTEST_MAX:.0e} at every cell")
    print(f"  V2 core-block call order asserted inside the probe")
    print(f"  V3 loop-iteration count is {iters.pop()} at every rung\n")

    missing = [s for s in HEALTHY + SICK if s not in by_step]
    if missing:
        raise SystemExit(f"rungs missing from the json: {missing}")

    # ── the panel ──────────────────────────────────────────────────────────────────
    R = {s: ratio_by_block(r) for s, r in by_step.items()}
    nblk = len(next(iter(R.values())))

    print(f"{'step':>6} {'class':>9} " + " ".join(f"{'b%d' % b:>7}" for b in range(nblk))
          + f" {'mean':>7} {'<1':>3} {'agree':>6} {'xbatch':>7}")
    for st in sorted(by_step):
        cls = ("HEALTHY" if st in HEALTHY else "SICK" if st in SICK
               else "ambig" if st in AMBIGUOUS else "-")
        rr = R[st]
        mean = sum(rr.values()) / len(rr)
        n_lt = sum(1 for v in rr.values() if v < 1.0)
        last = max(z["t"] for z in win_rows(by_step[st]))
        agr = [z["row_agree"] for z in win_rows(by_step[st]) if z["t"] == last]
        n_ag = sum(1 for v in agr if v >= P4_MIN_AGREE)
        print(f"{st:>6} {cls:>9} " + " ".join(f"{rr[b]:>7.3f}" for b in range(nblk))
              + f" {mean:>7.3f} {n_lt:>3} {n_ag:>2}/{len(agr)} "
                f"{cross_batch_agree(by_step[st]):>7.3f}")

    def n_lt(st):
        return sum(1 for v in R[st].values() if v < 1.0)

    p1 = all(n_lt(s) >= P1_MIN_BLOCKS for s in SICK)
    p2 = all(n_lt(s) <= P2_MAX_BLOCKS for s in HEALTHY)
    mh = sum(sum(R[s].values()) / nblk for s in HEALTHY) / len(HEALTHY)
    ms = sum(sum(R[s].values()) / nblk for s in SICK) / len(SICK)
    p3 = (mh - ms) >= P3_MIN_GAP

    def p4_rung(st):
        last = max(z["t"] for z in win_rows(by_step[st]))
        agr = [z["row_agree"] for z in win_rows(by_step[st]) if z["t"] == last]
        return (sum(1 for v in agr if v >= P4_MIN_AGREE) >= P4_MIN_BLOCKS
                and cross_batch_agree(by_step[st]) >= P4_MIN_AGREE)

    p4 = all(p4_rung(s) for s in sorted(by_step))

    def top1_last(st):
        last = max(z["t"] for z in win_rows(by_step[st]))
        v = [z["top1"] for z in win_rows(by_step[st]) if z["t"] == last]
        return sum(v) / len(v)

    rise = top1_last(1850) / max(top1_last(1625), 1e-30) - 1.0
    p5 = rise >= P5_MIN_RISE

    print()
    print(f"P1  sick rungs concentrate (ratio<1 at >={P1_MIN_BLOCKS}/{nblk} blocks): "
          + ", ".join(f"{s}:{n_lt(s)}" for s in SICK) + f" -> {'HELD' if p1 else 'FAILED'}")
    print(f"P2  healthy rungs do not (ratio<1 at <={P2_MAX_BLOCKS}/{nblk}): "
          + ", ".join(f"{s}:{n_lt(s)}" for s in HEALTHY) + f" -> {'HELD' if p2 else 'FAILED'}")
    print(f"P3  mean(healthy) - mean(sick) >= {P3_MIN_GAP}: {mh:.3f} - {ms:.3f} = "
          f"{mh - ms:+.3f} -> {'HELD' if p3 else 'FAILED'}")
    print(f"P4  positional (row_agree >= {P4_MIN_AGREE} at >={P4_MIN_BLOCKS} blocks AND "
          f"cross-batch >= {P4_MIN_AGREE}, every rung) -> {'HELD' if p4 else 'FAILED'}")
    print(f"P5  top1 rises >= {P5_MIN_RISE:.0%} from 1625 to 1850: {rise:+.1%} "
          f"-> {'HELD' if p5 else 'FAILED'}")

    # ── refuter ────────────────────────────────────────────────────────────────────
    cells, close = 0, 0
    for b in range(nblk):
        rows_h = {}
        for s in HEALTHY:
            for z in win_rows(by_step[s]):
                if z["block"] == b:
                    rows_h.setdefault(z["t"], []).append(z["pr"])
        rows_s = {}
        for s in SICK:
            for z in win_rows(by_step[s]):
                if z["block"] == b:
                    rows_s.setdefault(z["t"], []).append(z["pr"])
        for t in sorted(set(rows_h) & set(rows_s)):
            h = sum(rows_h[t]) / len(rows_h[t])
            sk = sum(rows_s[t]) / len(rows_s[t])
            cells += 1
            if abs(h - sk) / max(abs(h), 1e-30) <= REFUTE_TOL:
                close += 1
    refuted = cells > 0 and close == cells
    print(f"\nREFUTER  healthy and sick pr within {REFUTE_TOL:.0%} at {close}/{cells} "
          f"cells -> {'H18 REFUTED' if refuted else 'not refuted'}")

    held = [n for n, v in (("P1", p1), ("P2", p2), ("P3", p3), ("P4", p4), ("P5", p5)) if v]
    print(f"\nSUMMARY  {len(held)}/5 predictions held: {', '.join(held) or 'none'}")
    if refuted:
        print("  The refuter fired. Whatever the panel says, H18 is dead.")


if __name__ == "__main__":
    main()
