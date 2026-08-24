"""Figure for the TUL core-takeover cure
(docs/experiments/results/2026-08-24-tul-takeover-cure.md).

Six panels, in the order the argument is made:

  A  the loop stops separating the slot states  — effective rank per loop iteration, by rung
  B  the flip leads the symptom                 — rank ratio across the loop vs the core share
  C  the cotangent concentrates                 — effective positions, slot path vs token path
  D  the map barely changes, directions align   — operator vs gradient on the onset ladder
  E  the spectral gap does not open             — sigma_1/sigma_2, why a norm cap had no target
  F  the microcosm cure holds the gain below 1  — deterministic pair, control vs cure
  G  the harm is a turnaround                   — validation CE, every arm
  H  the level is not the criterion, the rate is — sigma_max history across four runs

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
    ax.set_title("D  the map barely changes;\nits directions align", fontsize=9)
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
    ax.set_title("C  the cotangent concentrates\n(same weights, both paths)", fontsize=9)
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
    ax.set_title("E  the spectral GAP does not open\n(so the norm cap had no target)",
                 fontsize=9)
    ax.legend(fontsize=7)


def panel_state(ax, path):
    """The headline: what the loop DOES to the slot states, per iteration.

    Effective rank of the 50 valid slot states in 1024 dimensions. Healthy rungs rise across
    the loop; the sick ones fall. The sign flip is the earliest indicator in the programme.
    """
    if not os.path.exists(path):
        return
    rows = json.load(open(path))
    styles = [("-", "o", OI["blue"]), ("-", "s", OI["sky"]), ("-", "^", OI["green"]),
              ("--", "D", OI["orange"]), (":", "x", OI["vermillion"]),
              (":", "+", OI["purple"])]
    for r, (ls, mk, c) in zip(rows, styles):
        pi = r["state"]["per_iter"]
        ax.plot([p["iter"] for p in pi], [p["eff_rank"] for p in pi],
                ls=ls, marker=mk, color=c, ms=4, lw=1.7, label=f"step {r['step']}")
    ax.set_xlabel("core loop iteration"); ax.set_ylabel("effective rank of 50 slot states")
    ax.set_title("A  the loop stops separating\nthe slot states", fontsize=9)
    ax.legend(fontsize=6.5, ncol=2)


def panel_flip(ax, state_path, ctrl_probe):
    """The sign flip against the symptom it precedes."""
    if not os.path.exists(state_path):
        return
    rows = json.load(open(state_path))
    steps = [r["step"] for r in rows]
    ratio = [r["state"]["per_iter"][-1]["eff_rank"] / r["state"]["per_iter"][0]["eff_rank"]
             for r in rows]
    share = [ctrl_probe.get(s, {}).get("preclip/core", float("nan"))
             / max(ctrl_probe.get(s, {}).get("preclip/total", 1.0), 1e-9) for s in steps]
    ax.plot(steps, ratio, color=OI["blue"], ls="-", marker="o", ms=5, lw=2,
            label="rank out / rank in")
    ax.axhline(1.0, color=OI["black"], lw=0.9, ls=":")
    ax.annotate("above 1: the loop separates\nbelow 1: the loop collapses", (steps[0], 1.0),
                fontsize=7, textcoords="offset points", xytext=(2, 4))
    ax2 = ax.twinx()
    ax2.plot(steps, share, color=OI["vermillion"], ls="--", marker="x", ms=5, lw=1.6,
             label="core share")
    ax2.set_ylabel("core share of the gradient")
    ax.set_xlabel("step"); ax.set_ylabel("slot-state rank ratio across the loop")
    ax.set_title("B  the flip leads the symptom", fontsize=9)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")


def panel_d(ax, ctrl_probe, cure_probe, upto):
    for probe, style, lab in ((ctrl_probe, CTRL, "control"), (cure_probe, CURE, "cure")):
        ks = sorted(k for k in probe if k <= upto)
        v = [probe[k].get("preclip/core_block_gain", float("nan")) for k in ks]
        ks, v = smooth(ks, v)
        s = dict(style); s["marker"] = "None"
        ax.plot(ks, v, label=lab, **s)
    ax.axhline(1.0, color=OI["black"], lw=0.8, ls=":")
    ax.set_xlabel("step"); ax.set_ylabel("block backward gain (median of 25)")
    ax.set_title("F  deterministic microcosm\nseed 0, batch 6", fontsize=9)
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
    ax.set_title("G  the harm is a turnaround\ntriangle = each arm's own minimum", fontsize=9)
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
    ax.set_title("H  the LEVEL is not the criterion,\nthe RATE is", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/home/wolfe/morph-scratch")
    ap.add_argument("--out", default=str(FIGDIR / "tul_takeover_cure.png"))
    a = ap.parse_args()
    S, C = a.scratch, os.path.join(a.scratch, "cure")

    det_ctrl = load_probe(f"{S}/phase1/capture.jsonl")
    det_cure = load_probe(f"{S}/phase1/rca/spec_scratch.jsonl")

    fig, axes = plt.subplots(2, 4, figsize=(21.5, 8.6))
    panel_state(axes[0][0], f"{C}/state.json")
    panel_flip(axes[0][1], f"{C}/state.json", det_ctrl)
    panel_b(axes[0][2], f"{C}/rank_slots.json", f"{C}/rank_tokens.json")
    panel_a(axes[0][3], f"{C}/ladder.json", det_ctrl)
    panel_c(axes[1][0], f"{C}/gap.json")
    panel_d(axes[1][1], det_ctrl, det_cure, 2100)
    panel_e(axes[1][2], [
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
    panel_f(axes[1][3], f"{C}/sigma_hist.json", None)
    for row in axes:
        for ax in row:
            ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
