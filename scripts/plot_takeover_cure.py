"""Figure for the TUL core-takeover cure (docs/experiments/results/2026-08-24-*-cure.md).

Four panels, left to right:

  A  Operator versus gradient. The per-block typical gain read off the core map's
     Jacobian (lab/divergence/jac_ladder.py) against the per-block backward gain fitted
     from gradient norms, on the ROLL_1625..1850 ladder. Two independent measurements of
     the same quantity; they either agree or the mechanism story is wrong.
  B  The deterministic microcosm. Block backward gain, control versus cure.
  C  The seed-1 real configuration. Train loss, control versus cure. The harm is a
     TURNAROUND, so this is the panel that says whether it was cured.
  D  The cap sweep. Typical block gain in the SICK state as a function of the spectral
     cap, by which family of core linears is capped.

COLOUR IS NEVER THE ONLY CHANNEL: every series also carries its own line style and
marker, and the two arms of every pair are solid versus dashed.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
# lab/ is a spike tree, not a package (its directory names are not importable), so the
# shared loaders are pulled in by path rather than by making lab/ into one.
sys.path.insert(0, str(ROOT / "lab" / "divergence"))
from score_arms import load_loss, load_probe                       # noqa: E402

FIGDIR = ROOT / "docs" / "experiments" / "figures"
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}
CTRL = dict(color=OI["vermillion"], ls=":", marker="x", lw=1.8)
CURE = dict(color=OI["blue"], ls="-", marker="o", lw=1.8, ms=3.5)


def smooth(xs, ys, k=9):
    """Centred median filter — the per-step probe is noisy and the trend is the claim."""
    out = []
    for i in range(len(ys)):
        lo, hi = max(0, i - k // 2), min(len(ys), i + k // 2 + 1)
        out.append(st.median(ys[lo:hi]))
    return xs, out


def series(probe, key, upto=None):
    ks = sorted(k for k in probe if upto is None or k <= upto)
    if key == "share":
        return ks, [probe[k]["preclip/core"] / probe[k]["preclip/total"] for k in ks]
    return ks, [probe[k].get(key, float("nan")) for k in ks]


def panel_a(ax, ladder_path, ctrl_probe):
    """The three numbers that separate 'the map grew' from 'the map aligned'.

    isotropic  = the gain a generic direction sees through one core block
    realized   = the gain the actual backward cotangent sees (fitted from grad norms)
    alignment  = whole-step gain / product of the per-block gains; below 1 the blocks'
                 amplifying directions disagree, above 1 they agree
    """
    lad = json.load(open(ladder_path))
    steps = [r["step"] for r in lad]
    iso = [r["sigma"]["t0"]["rms_block_gain"] for r in lad]
    align = []
    for r in lad:
        prod = 1.0
        for v in r["sigma"]["t0"]["rms_blocks"]:
            prod *= v
        align.append(r["sigma"]["t0"]["rms"] / prod)
    grad = [ctrl_probe.get(s, {}).get("preclip/core_block_gain", float("nan"))
            for s in steps]
    ax.plot(steps, iso, color=OI["blue"], ls="-", marker="o", ms=4,
            label="operator, isotropic per block")
    ax.plot(steps, grad, color=OI["orange"], ls="--", marker="s", ms=4,
            label="realized per block (gradient)")
    ax.plot(steps, align, color=OI["purple"], ls="-.", marker="^", ms=4,
            label="alignment across the 6 blocks")
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("gain (dimensionless)")
    ax.set_title("A  the map barely changes;\nits directions align", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def panel_b(ax, ctrl_probe, cure_probe, upto):
    for probe, style, lab in ((ctrl_probe, CTRL, "control"), (cure_probe, CURE, "cure")):
        ks, v = series(probe, "preclip/core_block_gain", upto)
        ks, v = smooth(ks, v, 25)
        ax.plot(ks, v, label=lab, markevery=200, **style)
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls="-.")
    ax.set_xlabel("step"); ax.set_ylabel("block backward gain (median of 25)")
    ax.set_title("B  deterministic microcosm\nseed 0, batch 6", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def panel_c(ax, pairs):
    for label, path, style in pairs:
        pts = load_loss(path)
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, label=label, markevery=max(1, len(xs) // 12), **style)
        lo = min(pts, key=lambda t: t[1])
        ax.plot([lo[0]], [lo[1]], marker="v", ms=7, color=style["color"], ls="none")
    ax.set_xlabel("step"); ax.set_ylabel("train loss (nats)")
    ax.set_title("C  seed 1, real configuration\ntriangle = each arm's own minimum",
                 fontsize=9)
    ax.legend(fontsize=7)


def panel_d(ax, sweeps):
    marks = {"mlp": ("o", "-", OI["blue"]), "attn": ("s", "--", OI["orange"]),
             "all": ("D", "-.", OI["green"])}
    for scope, path in sweeps:
        if not os.path.exists(path):
            continue
        rows = json.load(open(path))
        caps = [r["cap"] for r in rows]
        gains = [r["sigma"]["t0"]["rms_block_gain"] for r in rows]
        m, ls, c = marks[scope]
        order = sorted(range(len(caps)), key=lambda i: caps[i])
        ax.plot([caps[i] for i in order], [gains[i] for i in order],
                marker=m, ls=ls, color=c, label=f"cap {scope}", lw=1.8)
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls="-.")
    ax.set_xlabel("spectral cap on the core linears")
    ax.set_ylabel("typical block gain, sick state")
    ax.set_title("D  what has to shrink\n(ROLL_step_1850)", fontsize=9)
    ax.legend(fontsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/home/wolfe/morph-scratch")
    ap.add_argument("--out", default=str(FIGDIR / "tul_takeover_cure.png"))
    a = ap.parse_args()
    S, C = a.scratch, os.path.join(a.scratch, "cure")

    ctrl_probe = load_probe(f"{S}/phase1/capture.jsonl")
    cure_probe = load_probe(f"{S}/phase1/rca/spec_scratch.jsonl")

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.0))
    panel_a(axes[0], f"{C}/ladder.json", ctrl_probe)
    panel_b(axes[1], ctrl_probe, cure_probe, 2100)
    panel_c(axes[2], [("control (no penalty)", f"{C}/cure-a1r-ctrl.log", CTRL),
                      ("cure (spectral cap 1.5)", f"{C}/cure-a1r-spec.log", CURE),
                      ("dose control (cap 3.0)", f"{C}/cure-a1r-cap30.log",
                       dict(color=OI["purple"], ls="--", marker="^", lw=1.5, ms=3.5))])
    panel_d(axes[3], [("mlp", f"{C}/sweep_mlp.json"), ("attn", f"{C}/sweep_attn.json"),
                      ("all", f"{C}/sweep_all.json")])
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
