#!/usr/bin/env python
"""Render the TUL arm figures from wandb history into docs/experiments/figures/.

    python scripts/plot_tul_arms.py            # all four figures
    python scripts/plot_tul_arms.py --only ce  # one of: ce, divergence, efficiency, order

Every wandb-derived figure is regenerated from the run history, never from a pasted
number, so a figure cannot drift from the run it describes. The one figure that is not
in wandb -- the offline order-parameter probe -- reads
`docs/experiments/results/tul_order_parameter.csv`, which is versioned next to it.

COLOUR: the palette is Okabe-Ito, which is safe for red-green colour blindness, and no
figure uses colour as its only channel -- every series also carries its own line style
and marker, and the diverged arms are additionally marked with an explicit end-of-run
annotation. Do not "improve" this by switching to a rainbow or a red/green pair.
"""
from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "docs" / "experiments" / "figures"
CSV_ORDER = ROOT / "docs" / "experiments" / "results" / "tul_order_parameter.csv"
PROJECT = "morph-tul"

# Okabe-Ito. Index 4 (yellow) is skipped on white backgrounds.
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7"}

# run id -> (label, colour, linestyle, marker, diverged?)
ARMS = {
    "l4apqgyo": ("A0  baseline",          OI["blue"],       "-",   "o", False),
    "4ltwcdof": ("A0c baseline + cap",    OI["sky"],        "--",  "s", False),
    "0doe9yu0": ("A1c TUL + cap",         OI["green"],      "-",   "^", False),
    "4lb85o25": ("A3  compute floor",     OI["orange"],     "-.",  "D", False),
    "82easori": ("A1  no cap (diverged)", OI["vermillion"], ":",   "x", True),
    "8e49z6u8": ("A1r no cap (diverged)", OI["purple"],     ":",   "+", True),
}
COMPLETED = ["l4apqgyo", "4ltwcdof", "0doe9yu0", "4lb85o25"]

# The 2026-08-22 gate bake-off re-ran the uncapped arms on a build that logs
# train/grad_norm. The 08-18 runs in ARMS above do NOT log it (n=0 rows), so the
# divergence figure uses THESE runs for both of its panels rather than showing a
# gradient panel that silently omits the only arms that detonate.
BAKEOFF = {
    "cyushbhr": ("A1  no cap (08-22)",  OI["vermillion"], ":", "x"),
    "8vdthy0r": ("A1r no cap (08-22)",  OI["purple"],     ":", "+"),
}
CAPPED = {
    "0doe9yu0": ("A1c TUL + cap",       OI["green"],      "-",  "^"),
    "4ltwcdof": ("A0c baseline + cap",  OI["sky"],        "--", "s"),
}


def _foot(fig, text):
    fig.text(0.01, -0.03, textwrap.fill(text, 108), fontsize=7.5, color="#444444",
             va="top")


def _api():
    import wandb
    return wandb.Api()


def _series(run, keys, max_samples=4000):
    """Return (steps, values) for the first key present, sorted by step."""
    for k in keys:
        rows = run.history(keys=[k], samples=max_samples, pandas=False)
        rows = [r for r in rows if r.get(k) is not None]
        if rows:
            rows.sort(key=lambda r: r["_step"])
            return [r["_step"] for r in rows], [r[k] for r in rows], k
    return [], [], None


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out = FIGDIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


def fig_ce(api):
    """Validation token CE against step: the whole run, then the part that decides it."""
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(11.0, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    finals = []
    for rid, (label, colour, ls, marker, diverged) in ARMS.items():
        run = api.run(f"{PROJECT}/{rid}")
        steps, vals, _key = _series(run, ["val/ce_tokens", "val/loss"])
        if not steps:
            print(f"  WARNING: {label} has no CE history; skipped", file=sys.stderr)
            continue
        ax.plot(steps, vals, color=colour, linestyle=ls, marker=marker,
                markevery=max(1, len(steps) // 12), markersize=4.5,
                linewidth=1.7, label=label, alpha=0.95)
        if diverged:
            ax.annotate("aborted", xy=(steps[-1], vals[-1]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7.5, color=colour)
            continue
        keep = [(st, v) for st, v in zip(steps, vals) if st >= 10000]
        if keep:
            axz.plot([st for st, _ in keep], [v for _, v in keep], color=colour,
                     linestyle=ls, marker=marker, markevery=2, markersize=4.5,
                     linewidth=1.7, label=label)
        fin = run.summary.get("val/ce_tokens_final") or run.summary.get("val/loss_final")
        if fin is not None:
            finals.append((fin, label, colour))

    ax.set_ylim(3.0, 8.0)
    _style(ax, "Every arm — 20k steps, batch 14, seq 1024", "step", "val CE (nats/token)")
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")

    # The survivors differ by 0.007-0.056 nats. On the left axis that is one line.
    finals.sort()
    for i, (fin, label, colour) in enumerate(finals):
        axz.axhline(fin, color=colour, linewidth=0.8, alpha=0.35, linestyle=(0, (1, 3)))
        # A0 and A0c are 0.0069 nats apart: at this axis their labels overlapped, so
        # stack them by rank instead of anchoring each to its own value.
        axz.annotate(f"{fin:.4f}", xy=(1.0, fin), xycoords=("axes fraction", "data"),
                     xytext=(5, -3 + (i - len(finals) / 2) * 1.5),
                     textcoords="offset points", fontsize=7, color=colour,
                     annotation_clip=False)
    _style(axz, "Survivors only, last 10k steps (note the axis)", "step", "")
    axz.tick_params(labelleft=True)

    _foot(fig, "Left: uncapped TUL detonates; ademamix_alpha_cap=1.0 carries both capped "
               "arms to 20k. Right: the same four survivors on a 10x tighter axis — the "
               "whole A1-vs-A0 result is 0.056 nats and is invisible at left-panel scale. "
               "Dotted horizontals are each arm's final eval, which is a separate pass over fresh validation batches — not the last point of the plotted curve. One seed per arm.")
    _save(fig, "tul_arms_val_ce.png")


def fig_divergence(api):
    """The detonation itself: CE turning upward, and the gradient norm behind it."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    series = {**BAKEOFF, **CAPPED}
    for rid, (label, colour, ls, marker) in series.items():
        run = api.run(f"{PROJECT}/{rid}")
        steps, vals, _ = _series(run, ["val/ce_tokens", "val/loss"])
        keep = [(st, v) for st, v in zip(steps, vals) if st <= 6500]
        if keep:
            ax1.plot([st for st, _ in keep], [v for _, v in keep], color=colour,
                     linestyle=ls, marker=marker, markevery=2, markersize=4.5,
                     linewidth=1.7, label=label)
        g_st, g_v, _ = _series(run, ["train/grad_norm"])
        keep = [(st, v) for st, v in zip(g_st, g_v) if st <= 6500 and v > 0]
        if keep:
            ax2.semilogy([st for st, _ in keep], [v for _, v in keep], color=colour,
                         linestyle=ls, linewidth=1.4, label=label)
        else:
            print(f"  WARNING: {label} has no grad_norm history", file=sys.stderr)
    ax1.set_ylim(3.0, 8.0)
    _style(ax1, "Detonation window — val CE", "step", "val CE (nats/token)")
    ax1.legend(fontsize=7.5, frameon=True, framealpha=0.92, edgecolor="none",
               loc="lower left")
    _style(ax2, "Gradient norm — same runs, log scale", "step", "train/grad_norm")
    ax2.axhline(1.0, color=OI["black"], linewidth=0.7, alpha=0.4)
    _foot(fig, "These are the 2026-08-22 gate bake-off runs, not the 08-18 runs plotted "
               "in the CE figure: only this build logs train/grad_norm, and a gradient "
               "panel that omitted the arms which detonate would be worse than none. "
               "Same recipe, same divergence, different episode — uncapped A1 reaches "
               "grad_norm 3.0e11 while capped A1c stays near 1.")
    _save(fig, "tul_arms_divergence.png")


LABEL_DY = {"l4apqgyo": -34, "4ltwcdof": 30, "0doe9yu0": -34, "4lb85o25": 30}


def fig_efficiency(api):
    """Final CE against throughput: the quality-per-token-per-second trade."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for rid in COMPLETED:
        label, colour, _ls, marker, _d = ARMS[rid]
        run = api.run(f"{PROJECT}/{rid}")
        s = run.summary
        ce = s.get("val/ce_tokens_final")
        if ce is None:
            ce = s.get("val/loss_final")
        tps = s.get("perf/tokens_per_sec")
        if ce is None or tps is None:
            print(f"  WARNING: {label} missing final CE or tok/s; skipped", file=sys.stderr)
            continue
        ax.scatter([tps], [ce], color=colour, marker=marker, s=110, zorder=3,
                   edgecolors="white", linewidths=1.0, label=label)
        # Alternate above/below: A0 and A0c sit 0.007 nats apart and their labels
        # overlapped when every annotation used the same offset.
        dy = LABEL_DY.get(rid, -34)
        ax.annotate(f"{label}\nCE {ce:.4f}   PPL {math.exp(ce):.2f}",
                    xy=(tps, ce), xytext=(0, dy), textcoords="offset points",
                    fontsize=7.5, ha="center", va="center", color=colour)
    _style(ax, "Quality against throughput — final validation CE (equal tokens, 287M)",
           "tokens / s", "final val CE (nats/token)")
    ax.margins(x=0.22, y=0.28)
    _foot(fig, "Down and to the right is better. PPL is exp(mean CE); the "
               "val/ppl_tokens logged before 2026-08-23 was mean(exp(CE)) and reads high. "
               "Every point is one seed and the gaps are 0.007-0.056 nats, so the A1r "
               "retrain noise floor — not yet measured — decides which of these are real.")
    _save(fig, "tul_arms_efficiency.png")


def fig_order(_api=None):
    """The offline order-parameter probe, three passes."""
    rows = []
    with CSV_ORDER.open() as fh:
        for row in csv.DictReader(r for r in fh if not r.startswith("#")):
            rows.append(row)
    labels = [r["checkpoint"] for r in rows]
    x = range(len(labels))
    passes = [("pass0", "1 start, random ids", OI["blue"], "o"),
              ("pass1", "4 restarts, random ids", OI["green"], "^"),
              ("pass2", "4 restarts, real text", OI["orange"], "s")]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.axhspan(21.0, 40.0, color=OI["vermillion"], alpha=0.10, zorder=0)
    ax.text(0.15, 30.0, "diverged band", fontsize=8, color=OI["vermillion"])
    for key, lab, colour, marker in passes:
        xs = [i for i, r in enumerate(rows) if r[key]]
        ys = [float(rows[i][key]) for i in xs]
        ax.plot(xs, ys, color=colour, marker=marker, linestyle="none",
                markersize=8, alpha=0.9, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    _style(ax, "Core-map order parameter by checkpoint (log scale)",
           "", "ORDER  =  sigma_max(composition) / max block")
    ax.legend(fontsize=8, frameon=False, loc="center left")
    _foot(fig, "Survivors sit at 1.1-4.9 in every pass; the diverged controls sit "
               "at 21-38. The 7.7x margin holds across both operating points, but the "
               "trajectory's direction does not: on real text the cured arm rises where "
               "on random ids it falls. Blank markers = not measured in that pass.")
    _save(fig, "tul_order_parameter.png")


# The gate bake-off is discovered from the checkpoint tree, not hardcoded: arms 2 and 3
# do not exist until arm 1 finishes, and a figure that silently drops a missing arm
# would read as "that arm was flat" rather than "that arm has not run".
BAKEOFF_ARMS = [("tul-gate", OI["green"], "-", "^"),
                ("tul-a1", OI["blue"], "--", "o"),
                ("tul-a1r", OI["orange"], "-.", "D")]


def fig_bakeoff(api):
    """The live gate bake-off: CE, degeneration, and the halting policy's depth."""
    # wandb_id.txt is OVERWRITTEN per run and is stale for any arm that has not started
    # this campaign -- checkpoints/morph/tul-a1/wandb_id.txt pointed at a deleted 08-22
    # run when this was written. Resolving it blindly would plot a PREVIOUS campaign's
    # curve as if it were tonight's. Arm 1 is definitely current, so its start time is
    # the cutoff: any run older than that is a leftover id and is dropped by name.
    def _resolve(name):
        f = ROOT / "checkpoints" / "morph" / name / "wandb_id.txt"
        if not f.exists():
            return None, f"{name}: never run"
        rid = f.read_text().strip()
        try:
            return api.run(f"{PROJECT}/{rid}"), None
        except Exception:
            return None, f"{name}: id {rid} not found (stale)"

    head, err = _resolve(BAKEOFF_ARMS[0][0])
    if head is None:
        print(f"  {err}; nothing to plot", file=sys.stderr)
        return
    cutoff = head.created_at

    ids, skipped = [], []
    for name, colour, ls, marker in BAKEOFF_ARMS:
        run, err = _resolve(name)
        if run is None:
            skipped.append(err)
        elif run.created_at < cutoff:
            skipped.append(f"{name}: id points at a run older than arm 1 (stale)")
        else:
            ids.append((name, run, colour, ls, marker))
    for m in skipped:
        print(f"  {m}", file=sys.stderr)

    fig, (a1, az, a2) = plt.subplots(1, 3, figsize=(13.5, 4.2))
    started, depths = [], []
    for name, run, colour, ls, marker in ids:
        started.append(f"{name} ({run.state}, step {run.summary.get('_step')})")
        st, v, _ = _series(run, ["val/ce_tokens", "val/loss"])
        if st:
            a1.plot(st, v, color=colour, linestyle=ls, marker=marker, markevery=3,
                    markersize=4, linewidth=1.7, label=name)
            # The survivors differ by ~0.1 nats; on the left axis that is one line.
            keep = [(x, y) for x, y in zip(st, v) if x >= 12000]
            if keep:
                az.plot([x for x, _ in keep], [y for _, y in keep], color=colour,
                        linestyle=ls, marker=marker, markersize=4.5, linewidth=1.7,
                        label=name)
        fin = run.summary.get("val/ce_tokens_final")
        if fin is not None:
            az.axhline(fin, color=colour, linewidth=0.9, alpha=0.4, linestyle=(0, (1, 3)))
            az.annotate(f"{fin:.4f}", xy=(1.0, fin), xycoords=("axes fraction", "data"),
                        xytext=(5, -3), textcoords="offset points", fontsize=7.5,
                        color=colour, annotation_clip=False)
        rs, rv, _ = _series(run, ["gen/rep4"])
        if rs:
            a2.plot(rs, rv, color=colour, linestyle=ls, marker=marker, markersize=5,
                    linewidth=1.5, label=f"{name} rep4")
        ds, dv, _ = _series(run, ["gen/distinct3"])
        if ds:
            a2.plot(ds, dv, color=colour, linestyle=(0, (4, 2)), marker=marker,
                    markersize=5, markerfacecolor="none", linewidth=1.5,
                    label=f"{name} distinct3")
        ps, pv, _ = _series(run, ["val/halt_depth_mean"])
        if pv:
            depths.append(f"{name} halt depth {min(pv):.2f}-{max(pv):.2f} (collapsed)")

    _style(a1, "Bake-off — val token CE", "step", "val CE (nats/token)")
    a1.legend(fontsize=7.5, frameon=False)
    _style(az, "Survivors, last 8k steps (note the axis)", "step", "")
    az.tick_params(labelleft=True)
    a2.set_ylim(-0.03, 1.03)
    _style(a2, "Degeneration watch — ONE decode mode", "step", "fraction")
    a2.legend(fontsize=7, frameon=False, loc="center right")

    note = "Arms present: " + "; ".join(started)
    if skipped:
        note += ". NOT PLOTTED: " + "; ".join(skipped)
    if depths:
        note += ". " + "; ".join(depths)
    _foot(fig, note + ". RIGHT PANEL IS ONE DECODE MODE (t=0.8 / top-k 50) -- the "
               "training loop's own generation test. It CANNOT see the greedy repetition "
               "loop: at 128 tokens greedy rep4 is 0.76 (gate) and 0.85 (a1) against this "
               "panel's ~0.01. See tul_decode_modes.png.")
    _save(fig, "tul_bakeoff.png")


SAMPLES_JSON = ROOT / "docs" / "experiments" / "results" / "tul_samples.json"


def fig_decode_modes(_api=None):
    """rep4 by decode mode. The figure that exists because ONE mode is not enough.

    The training loop only ever decodes at t=0.8 / top-k 50, and at that setting every
    arm looks fine. Greedy is a different picture, and the DIVERGED arm is a third one:
    it beats real text on both diversity metrics while generating fluent-shaped noise,
    which is why this plot carries CE next to every arm name.
    """
    import json
    if not SAMPLES_JSON.exists():
        print(f"  {SAMPLES_JSON} missing; run scripts/tul_samples.py", file=sys.stderr)
        return
    d = json.loads(SAMPLES_JSON.read_text())
    anchor = d.get("_real_text") or {}
    # (label, arm key, policy, CE, colour)
    ROWS = [("TUL-gate  CE 3.312", "gate_20k", "fixed", OI["green"]),
            ("TUL-gate halt", "gate_20k", "halt", OI["sky"]),
            ("TUL-A1  CE 3.418", "a1_20k", "fixed", OI["blue"]),
            ("a1r DIVERGED  CE 6.43", "a1r_DIVERGED_4160", "fixed", OI["vermillion"])]
    MODES = ["greedy", "topk50_t0.8", "sample_t1"]
    present = [(lab, k, pol, c) for lab, k, pol, c in ROWS
               if k in d and pol in d[k]]

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    w = 0.8 / max(len(present), 1)
    for i, (lab, k, pol, colour) in enumerate(present):
        ys = [d[k][pol][m]["metrics"]["rep4"] for m in MODES]
        xs = [j + i * w - 0.4 + w / 2 for j in range(len(MODES))]
        ax.bar(xs, ys, width=w * 0.92, color=colour, label=lab,
               edgecolor="white", linewidth=0.6)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.3f}", xy=(x, y), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=6.5,
                        color=colour)
    if anchor:
        ax.axhline(anchor["rep4"], color=OI["black"], linewidth=1.1,
                   linestyle=(0, (5, 3)))
        ax.annotate(f"real text  rep4 {anchor['rep4']:.3f}", xy=(0.005, anchor["rep4"]),
                    xycoords=("axes fraction", "data"), xytext=(0, 5),
                    textcoords="offset points", fontsize=7.5, color=OI["black"])
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(["greedy (t=0)", "top-k 50, t=0.8", "ancestral, t=1.0"],
                       fontsize=9)
    _style(ax, "Repetition by decode mode — 8 prompts x 128 new tokens",
           "", "rep4  (fraction of repeated 4-grams)")
    ax.legend(fontsize=8, frameon=False)
    _foot(fig, "Higher is worse. The training loop only decodes at the MIDDLE setting, "
               "where every arm looks acceptable; greedy is 4-7x worse and is invisible "
               "to it. Read CE beside every bar: the DIVERGED arm scores the BEST "
               "repetition numbers at every mode because incoherent text never repeats "
               "-- its top-k rep4 of 0.001 beats real text's 0.003 while its CE is 6.43. "
               "Diversity is a collapse detector, not a quality metric; compare it only "
               "within a band of comparable CE.")
    _save(fig, "tul_decode_modes.png")


# The 2026-08-23 batch-12 campaign: one arm died, two lived, and all three log the
# per-core-linear spectral norms that make the mechanism visible.
MECH = {
    "0ujvtukf": ("A1r seed 1 (diverged, step 4140)", OI["vermillion"], ":",  "x"),
    "c23dwx4a": ("A1  seed 0 (lived, 20k)",          OI["green"],      "-",  "^"),
    "2rk7mguo": ("gate seed 0 (lived, 20k)",         OI["blue"],       "--", "s"),
}
CORE_LIN = [f"spec/sigma/core.{i}.mlp.0.{w}" for i in range(6) for w in ("gate_up", "down")]


def fig_mechanism(api):
    """Why the TUL core detonates, in the four quantities that say it is not a seed.

    Panel 1 is the whole argument: through the onset window the loss sits at 4.5-4.8 in
    every arm while `train/grad_norm` crosses six orders of magnitude. The diverged arm's
    loss does drift upward, but only AFTER step ~2200, once the gradient has already
    exploded -- so the loss is a lagging consequence here, not the trigger.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6))
    (a1, a2), (a3, a4) = axes
    LO, HI = 1500, 3000

    for rid, (lab, col, ls, mk) in MECH.items():
        run = api.run(f"{PROJECT}/{rid}")
        rows = [r for r in run.history(keys=None, samples=25000, pandas=False)]
        rows.sort(key=lambda r: r["_step"])

        def col_of(key, lo=LO, hi=HI):
            xs = [(r["_step"], r[key]) for r in rows
                  if r.get(key) is not None and lo <= r["_step"] <= hi]
            return [x for x, _ in xs], [y for _, y in xs]

        st, ls_ = col_of("train/loss")
        a1.plot(st, ls_, color=col, linestyle=ls, linewidth=1.3, label=lab)
        st, gn = col_of("train/grad_norm")
        a2.plot(st, gn, color=col, linestyle=ls, linewidth=1.3, label=lab)
        st, gc = col_of("gradnorm/core")
        a3.plot(st, gc, color=col, linestyle=ls, marker=mk, markersize=4.5,
                linewidth=1.3, label=lab)
        # Per-core-block one-pass gain, sigma(gate_up) * sigma(down), block 0 vs the
        # median block. A runaway concentrated in ONE block is the shape to see.
        pts = [r for r in rows if r.get(CORE_LIN[0]) is not None and LO <= r["_step"] <= HI]
        st = [r["_step"] for r in pts]
        b0 = [r["spec/sigma/core.0.mlp.0.gate_up"] * r["spec/sigma/core.0.mlp.0.down"]
              for r in pts]
        rest = []
        for r in pts:
            g = sorted(r[f"spec/sigma/core.{i}.mlp.0.gate_up"]
                       * r[f"spec/sigma/core.{i}.mlp.0.down"] for i in range(1, 6))
            rest.append(g[len(g) // 2])
        a4.plot(st, b0, color=col, linestyle=ls, marker=mk, markersize=4.5,
                linewidth=1.4, label=f"{lab.split('(')[0].strip()} — block 0")
        a4.plot(st, rest, color=col, linestyle=ls, linewidth=0.9, alpha=0.45,
                label=f"{lab.split('(')[0].strip()} — median block")

    for ax in (a2,):
        ax.set_yscale("log")
    a4.set_yscale("log")
    for ax in (a1, a2, a3, a4):
        ax.axvspan(1900, 2040, color=OI["orange"], alpha=0.13, lw=0)
    a1.annotate("core gradient share starts\nratcheting here", xy=(1900, a1.get_ylim()[0]),
                xytext=(2120, a1.get_ylim()[0] + 0.25), fontsize=7.5, color=OI["orange"])

    _style(a1, "1. train/loss — flat through the onset, degrades only afterwards",
           "step", "training loss")
    _style(a2, "2. train/grad_norm — six orders of magnitude", "step", "grad norm (log)")
    _style(a3, "3. gradnorm/core — the looped weights' share of the norm",
           "step", "core share of total grad norm")
    _style(a4, "4. one-pass core gain, block 0 vs the median block",
           "step", "sigma(gate_up) x sigma(down)  (log)")
    a1.legend(fontsize=7.5, frameon=False)
    a4.legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout()
    _foot(fig, "The shaded band is steps 1900-2040. Panel 1 is the finding: through that "
               "window the loss sits at 4.5-4.8 in every arm, including the one whose "
               "gradient is about to cross six orders of magnitude, and it only degrades "
               "after step ~2200 when the damage is already done. So this is not the usual "
               "bad-batch loss spike -- "
               "and the two A1 arms eat the SAME batches (paired residual correlation of "
               "train/loss over steps 0-1900 is 0.938 at lag 0 and 0.233 at lag 1), so the "
               "seed changes the weight init, not the data. Panel 3 is the precursor: the "
               "core's share of the gradient norm ratchets 0.009 -> 0.043 -> 0.108 -> 0.90 "
               "and never returns, ~140 steps before the norm explodes. It is a RATCHET, "
               "not a level -- the gate arm touches 0.35 at step 700 and falls back. Panel "
               "4 shows the shape: the FIRST core block's gain runs away to 23x while the "
               "rest of the stack sits at 4.5-5.3 and declines.")
    _save(fig, "tul_divergence_mechanism.png")


REPAB = ROOT / "docs" / "experiments" / "results" / "tul_rep_ab.json"
# label -> (display, colour, hatch). The MATCHED pair is first: same batch 14, same seed 0,
# same alpha cap, same 20k steps, differing only in tul.activate_at.
REPAB_ARMS = [
    ("a0_acap1_b14", "A0  no TUL (b14)",   OI["blue"],       ""),
    ("a1_acap1_b14", "A1  TUL (b14)",      OI["green"],      "///"),
    ("a3_b14",       "A3  slots, no core", OI["orange"],     "..."),
    ("a1_b12",       "A1  TUL (b12)",      OI["sky"],        "\\\\"),
    ("gate_b12",     "gate TUL (b12)",     OI["purple"],     "xxx"),
]
REPAB_MODES = [("topk50_t0.8", "top-k 50, t=0.8"), ("sample_t1", "ancestral, t=1.0"),
               ("greedy", "greedy (diagnostic)")]


def fig_rep_ab(_api=None):
    """Does the slot loop repeat itself less than a plain model? Sampled decoding, 512 tokens.

    The predecessor of this figure had NO non-TUL arm in it and scored 128-token samples,
    where held-out text itself sits at rep4 0.015 with 54 % of rows at exactly 0.000 --
    a reference on the floor cannot rank anything. Both are fixed here: A0 is decoded by
    the matched plain loop, and everything is 512 tokens.
    """
    import json
    import numpy as np
    d = json.loads(REPAB.read_text())
    anchor = d.get("_real_text", {})
    have = [(k, lab, c, h) for k, lab, c, h in REPAB_ARMS if k in d]
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(12.4, 4.9),
                                  gridspec_kw={"width_ratios": [1.55, 1]})
    w = 0.8 / max(len(have), 1)
    for ai, (key, lab, col, hat) in enumerate(have):
        ys, es = [], []
        for mode, _ in REPAB_MODES:
            m = d[key]["fixed"].get(mode, {}).get("metrics", {})
            per = [x["rep4"] for x in d[key]["fixed"].get(mode, {}).get("per_prompt", [])]
            ys.append(m.get("rep4", float("nan")))
            # standard error over prompts, not the spread: the question is where the MEAN is
            es.append(np.std(per) / max(len(per) ** 0.5, 1) if per else 0.0)
        xs = [i + ai * w - 0.4 + w / 2 for i in range(len(REPAB_MODES))]
        ax.bar(xs, ys, width=w * 0.92, color=col, hatch=hat, edgecolor="white",
               linewidth=0.6, label=lab, yerr=es, capsize=2.5,
               error_kw={"elinewidth": 0.9, "ecolor": "#333333"})
    if anchor:
        ax.axhline(anchor["rep4"], color=OI["black"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.axhspan(max(anchor["rep4"] - anchor["rep4_std"], 0),
                   anchor["rep4"] + anchor["rep4_std"], color=OI["black"], alpha=0.09, lw=0)
        ax.annotate(f"held-out text  {anchor['rep4']:.3f} ± {anchor['rep4_std']:.3f} "
                    f"(n={anchor['n_rows']})",
                    xy=(0.004, anchor["rep4"]), xycoords=("axes fraction", "data"),
                    xytext=(0, 6), textcoords="offset points", fontsize=7.5)
    ax.set_xticks(range(len(REPAB_MODES)))
    ax.set_xticklabels([lab for _, lab in REPAB_MODES], fontsize=9)
    _style(ax, f"Repetition at 512 new tokens, {d.get('_meta', {}).get('n_prompts', '?')} prompts",
           "", "rep4  (fraction of repeated 4-grams; higher is worse)")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # Right panel: the PAIRED A1 - A0 difference, prompt by prompt. A mean gap of 0.03
    # against a between-prompt spread of 0.25 is unreadable unless it is paired.
    if "a0_acap1_b14" in d and "a1_acap1_b14" in d:
        for mi, (mode, mlab) in enumerate(REPAB_MODES):
            a0 = [x["rep4"] for x in d["a0_acap1_b14"]["fixed"][mode]["per_prompt"]]
            a1 = [x["rep4"] for x in d["a1_acap1_b14"]["fixed"][mode]["per_prompt"]]
            diff = np.array(a1) - np.array(a0)
            axd.scatter([mi + 0.06 * (i - len(diff) / 2) for i in range(len(diff))], diff,
                        s=16, color=OI["green"] if diff.mean() < 0 else OI["vermillion"],
                        marker="o", zorder=3)
            axd.plot([mi - 0.3, mi + 0.3], [diff.mean()] * 2, color=OI["black"],
                     linewidth=1.8, zorder=4)
            se = diff.std(ddof=1) / len(diff) ** 0.5
            axd.annotate(f"{diff.mean():+.3f}\n±{se:.3f}", xy=(mi, diff.mean()),
                         xytext=(0, 9), textcoords="offset points", ha="center", fontsize=7.5)
        axd.axhline(0.0, color=OI["black"], linewidth=0.9)
        axd.set_xticks(range(len(REPAB_MODES)))
        axd.set_xticklabels([lab for _, lab in REPAB_MODES], fontsize=8.5)
        _style(axd, "Paired A1 − A0, same prompt, same decode", "",
               "rep4 difference (below 0 = TUL repeats less)")
    fig.tight_layout()
    _foot(fig, "Matched pair: A0 and A1 both at batch 14, seed 0, alpha cap 1.0, step "
               "20000, differing only in tul.activate_at. Error bars are the standard "
               "error over prompts. Read the LEFT panel for the level and the RIGHT one "
               "for the TUL effect: the between-prompt spread is ~0.2-0.3, so an unpaired "
               "read of a ~0.03 gap says nothing, and only the paired difference does. "
               "Greedy is a diagnostic, not a ranking -- it says whether an argmax loop "
               "exists. A diverged model scores EXCELLENT repetition because incoherent "
               "text never repeats, so never rank arms on this axis across a CE gap.")
    _save(fig, "tul_rep_ab.png")


FIGS = {"ce": fig_ce, "repab": fig_rep_ab, "mechanism": fig_mechanism, "decode": fig_decode_modes, "divergence": fig_divergence, "efficiency": fig_efficiency,
        "order": fig_order, "bakeoff": fig_bakeoff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(FIGS), help="render one figure")
    a = ap.parse_args()
    names = [a.only] if a.only else list(FIGS)
    api = _api() if any(n != "order" for n in names) else None
    for n in names:
        FIGS[n](api)


if __name__ == "__main__":
    main()
