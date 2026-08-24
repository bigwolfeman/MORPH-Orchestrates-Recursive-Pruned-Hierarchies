"""Figure for the TUL core-takeover cure
(docs/experiments/results/2026-08-24-tul-takeover-cure.md).

Six panels, in the order the argument is made:

  A  the map barely changes, its directions align  — operator vs gradient on the onset ladder
  B  the cotangent concentrates                    — effective positions, slots vs tokens
  C  what has to shrink, and by how much           — cap sweep, by family of core linear
  D  the cure holds the gain below 1               — deterministic microcosm, control vs cure
  E  the harm, and its absence                     — validation CE, control vs cure
  F  the level of sigma_max is not the criterion   — sigma_max history across four runs

COLOUR IS NEVER THE ONLY CHANNEL: every series carries its own line style and marker, and
the two arms of every pair are solid versus dotted.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
# lab/ is a spike tree, not a package (its directory names are not importable), so the
# shared loaders are pulled in by path rather than by making lab/ into one.
sys.path.insert(0, str(ROOT / "lab" / "divergence"))
from score_arms import load_probe, load_val                       # noqa: E402

FIGDIR = ROOT / "docs" / "experiments" / "figures"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
CTRL = dict(color=OI["vermillion"], ls=":", marker="x", lw=1.9, ms=4)
CURE = dict(color=OI["blue"], ls="-", marker="o", lw=1.9, ms=4)


def alignment(t):
    p = 1.0
    for v in t["rms_blocks"]:
        p *= v
    return t["rms"] / p


def smooth(xs, ys, k=25):
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        out.append(st.median(ys[lo:hi]))
    return xs, out


def panel_a(ax, path, ctrl_probe):
    lad = json.load(open(path))
    steps = [r["step"] for r in lad]
    iso = [r["sigma"]["t0"]["rms_block_gain"] for r in lad]
    ali = [alignment(r["sigma"]["t0"]) for r in lad]
    grad = [ctrl_probe.get(s, {}).get("preclip/core_block_gain", float("nan")) for s in steps]
    ax.plot(steps, iso, color=OI["blue"], ls="-", marker="o", ms=4,
            label="operator, isotropic per block")
    ax.plot(steps, grad, color=OI["orange"], ls="--", marker="s", ms=4,
            label="realized per block (gradient)")
    ax.plot(steps, ali, color=OI["purple"], ls="-.", marker="^", ms=5,
            label="alignment across the 6 blocks")
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("gain (dimensionless)")
    ax.set_title("A  the map barely changes;\nits directions align", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def panel_b(ax, slots_path, tokens_path):
    for path, style, lab in ((slots_path, CURE, "slot path (A1), 57 valid slots"),
                             (tokens_path, CTRL, "token path (A0 code path), 1152")):
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        xs = [r["step"] for r in rows]
        ys = [r["rank"]["eff_positions_per_block"]["0"] for r in rows]
        ax.plot(xs, ys, label=lab, **style)
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("effective positions at core block 0")
    ax.set_title("B  the cotangent concentrates\n(same weights, both paths)", fontsize=9)
    ax.legend(fontsize=7)


def panel_c(ax, gap_path):
    """The repair that also fails: no core weight's spectral GAP is opening.

    Power iteration aligns at rate (sigma_1/sigma_2)^k, so if the gap is flat, no single
    matrix's spectrum is driving the alignment — which is why five feature-space
    interventions all failed.
    """
    if not os.path.exists(gap_path):
        return
    rows = json.load(open(gap_path))
    steps = [r["step"] for r in rows]
    med = [sorted(v["gap"] for v in r["gap"].values())[len(r["gap"]) // 2] for r in rows]
    worst = [max(v["gap"] for v in r["gap"].values()) for r in rows]
    ax.plot(steps, med, color=OI["blue"], ls="-", marker="o", ms=4, label="median gap")
    ax.plot(steps, worst, color=OI["orange"], ls="--", marker="s", ms=4, label="worst gap")
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("sigma_1 / sigma_2 of a core linear")
    ax.set_title("C  the spectral GAP does not open\n(so the norm cure had no target)",
                 fontsize=9)
    ax.legend(fontsize=7)


def panel_d(ax, ctrl_probe, cure_probe, upto):
    for probe, style, lab in ((ctrl_probe, CTRL, "control"), (cure_probe, CURE, "cure")):
        ks = sorted(k for k in probe if k <= upto)
        v = [probe[k].get("preclip/core_block_gain", float("nan")) for k in ks]
        ks, v = smooth(ks, v)
        s = dict(style); s["marker"] = "None"
        ax.plot(ks, v, label=lab, **s)
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("block backward gain (median of 25)")
    ax.set_title("D  deterministic microcosm\nseed 0, batch 6", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def panel_e(ax, arms):
    for label, path, style in arms:
        pts = load_val(path)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, **style)
        lo = min(pts, key=lambda t: t[1])
        ax.plot([lo[0]], [lo[1]], marker="v", ms=8, color=style["color"], ls="none")
    ax.set_xlabel("step"); ax.set_ylabel("validation CE (nats)")
    ax.set_title("E  the harm is a turnaround\ntriangle = each arm's own minimum", fontsize=9)
    ax.legend(fontsize=7)


def panel_f(ax, hist_path, cure_probe_log):
    if not os.path.exists(hist_path):
        return
    hist = json.load(open(hist_path))
    styles = {
        "c23dwx4a": ("tul-a1 s0, healthy to 20k", dict(color=OI["green"], ls="-", marker="o", ms=3)),
        "0ujvtukf": ("tul-a1r s1, died at 4140", dict(color=OI["vermillion"], ls=":", marker="x", ms=4)),
        "capture": ("onset-capture, took over 1866", dict(color=OI["orange"], ls="--", marker="s", ms=3)),
        "spec-scratch": ("cure, cap 1.5", dict(color=OI["blue"], ls="-.", marker="^", ms=3)),
    }
    for key, (lab, st_) in styles.items():
        if key not in hist:
            continue
        xs = [p[0] for p in hist[key]]
        ys = [p[1] for p in hist[key]]
        ax.plot(xs, ys, label=lab, lw=1.7, markevery=max(1, len(xs) // 12), **st_)
    ax.set_xscale("log")
    ax.set_xlabel("step (log)"); ax.set_ylabel("sigma_max of the core MLP linears")
    ax.set_title("F  the LEVEL is not the criterion,\nthe RATE is", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/home/wolfe/morph-scratch")
    ap.add_argument("--out", default=str(FIGDIR / "tul_takeover_cure.png"))
    a = ap.parse_args()
    S, C = a.scratch, os.path.join(a.scratch, "cure")

    det_ctrl = load_probe(f"{S}/phase1/capture.jsonl")
    det_cure = load_probe(f"{S}/phase1/rca/spec_scratch.jsonl")

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.6))
    panel_a(axes[0][0], f"{C}/ladder.json", det_ctrl)
    panel_b(axes[0][1], f"{C}/rank_slots.json", f"{C}/rank_tokens.json")
    panel_c(axes[0][2], f"{C}/gap.json")
    panel_d(axes[1][0], det_ctrl, det_cure, 2100)
    panel_e(axes[1][1], [
        ("control", f"{C}/a35-ctrl.log", CTRL),
        ("soft cap 1.5", f"{C}/a35-spec.log",
         dict(color=OI["purple"], ls="--", marker="^", lw=1.5, ms=4)),
        ("hard cap 1.5 MLP", f"{C}/a35-proj15.log",
         dict(color=OI["orange"], ls="-.", marker="s", lw=1.5, ms=4)),
        ("hard cap 1.5 +attn", f"{C}/a35-proj15attn.log", CURE),
        ("control, batch 10", f"{C}/b10-ctrl.log",
         dict(color=OI["sky"], ls=":", marker="x", lw=1.5, ms=4)),
        ("slots 128, batch 10", f"{C}/b10-slots128.log",
         dict(color=OI["green"], ls="-", marker="D", lw=1.8, ms=4)),
    ])
    panel_f(axes[1][2], f"{C}/sigma_hist.json", None)
    for row in axes:
        for ax in row:
            ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
