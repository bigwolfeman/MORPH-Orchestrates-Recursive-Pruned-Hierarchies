"""Summary figure + stage table for the snap vs LXTUL data-flow probe.

Reads dataflow_{snap,lxtul}.json (lab/divergence/dataflow_probe.py) and writes dataflow.png
and stage_table.txt next to them. Palette: Paul Tol high-contrast blue / yellow (validated
CVD dE 41 protan), every bar carries its value as text, LXTUL bars are hatched.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

D = os.path.dirname(os.path.abspath(__file__))
SNAP, LX = "#004488", "#DDAA33"
INK, MUTED = "#222222", "#666666"
arms = {k: json.load(open(os.path.join(D, f"dataflow_{k}.json"))) for k in ("snap", "lxtul")}


def g(arm, key, src="readings"):
    r = arms[arm][src].get(key)
    return (np.nan, np.nan, np.nan) if r is None else (r["mean"], r["lo"], r["hi"])


def bars(ax, labels, keys, fmt="{:.2f}", src="readings", scale=1.0, log=False, ref=None):
    x = np.arange(len(labels))
    w = 0.38
    for j, (arm, col, hatch) in enumerate((("snap", SNAP, None), ("lxtul", LX, "//"))):
        vals = []
        for k in keys:
            kk = k[arm] if isinstance(k, dict) else k
            vals.append(g(arm, kk, src) if kk else (np.nan,) * 3)
        m = np.array([v[0] for v in vals]) * scale
        lo = np.array([v[1] for v in vals]) * scale
        hi = np.array([v[2] for v in vals]) * scale
        pos = x + (j - 0.5) * w
        ok = ~np.isnan(m)
        ax.bar(pos[ok], m[ok], w * 0.92, color=col, hatch=hatch, edgecolor=INK, linewidth=0.6)
        err = np.vstack([np.clip(m - lo, 0, None), np.clip(hi - m, 0, None)])
        ax.errorbar(pos[ok], m[ok], yerr=err[:, ok], fmt="none", ecolor=INK, lw=0.8, capsize=2)
        for p, v, h in zip(pos[ok], m[ok], hi[ok]):
            ax.text(p, (h if not log else h * 1.15) if np.isfinite(h) else v,
                    fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK,
                    rotation=0)
        for p in pos[~ok]:
            ax.text(p, 0, "n/a", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    if log:
        ax.set_yscale("log")
    if ref is not None:
        ax.axhline(ref[0], color=MUTED, lw=1, ls="--")
        pass
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)


fig, axs = plt.subplots(4, 2, figsize=(9.6, 13.6))
axs = axs.ravel()
own = arms["snap"]["own_depth"]

# 1 seed pool
ax = axs[0]
bars(ax, ["top-1\nweight", "boundary\ntoken", "first\ntoken", "1 / span\n(uniform)"],
     [f"seed/d{own}/top1", f"seed/d{own}/boundary_mass", f"seed/d{own}/first_mass",
      f"seed/d{own}/uniform_mass"])
ax.set_title("1  Seed pool: each cell's attention over its span\n"
             f"norm. entropy snap {g('snap', f'seed/d{own}/entropy_norm')[0]:.2f}, "
             f"LXTUL {g('lxtul', f'seed/d{own}/entropy_norm')[0]:.2f} (1 = uniform)",
             fontsize=10, loc="left")
ax.set_ylabel("attention mass", fontsize=9)

# 2 loop per pass
ax = axs[1]
for arm, col, mk, ls in (("snap", SNAP, "o", "-"), ("lxtul", LX, "s", "--")):
    t = np.arange(own)
    m = [g(arm, f"loop/d{own}/t{i}/sm_winner_rel_change")[0] for i in t]
    sw = [g(arm, f"loop/d{own}/t{i}/switch")[0] for i in t]
    ax.plot(t + 1, m, color=col, marker=mk, ls=ls, lw=2, ms=7,
            label=f"{'snap' if arm == 'snap' else 'LXTUL'}: carried-state change",
            markeredgecolor=INK)
    ax.plot(t[1:] + 1, sw[1:], color=col, marker=mk, ls=":", lw=1.2, ms=5, alpha=0.9,
            markerfacecolor="white", markeredgecolor=col,
            label=f"{'snap' if arm == 'snap' else 'LXTUL'}: router switch rate")
ax.axhline(0.05, color=MUTED, lw=1, ls="--")
ax.text(1, 0.052, "D-3: 5 %", ha="left", va="bottom", fontsize=8, color=MUTED)
ax.set_yscale("log")
ax.set_xlabel("pass", fontsize=9)
ax.set_xticks(range(1, own + 1))
cm_s = g("snap", f"loop/d{own}/t{own - 1}/cm_winner_streammean", "common_mode")[0]
cm_l = g("lxtul", f"loop/d{own}/t{own - 1}/cm_winner_streammean", "common_mode")[0]
ax.set_title("2  Loop: change of the carried cell per pass\n"
             "(stream mean, ||h_t - h_t-1|| / ||h_t||)",
             fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# 3 core reads
ax = axs[2]
bars(ax, ["own slot's\ncells", "earlier\nslots' cells", "sink"],
     [f"core/d{own}/comb_own", f"core/d{own}/comb_earlier", f"core/d{own}/comb_sink"],
     ref=(0.2, "20 % (D-4)"))
cm_s = np.nanmean([g("snap", f"core/d{own}/l{i}/cm_earlier", "common_mode")[0]
                   for i in range(6)])
cm_l = np.nanmean([g("lxtul", f"core/d{own}/l{i}/cm_earlier", "common_mode")[0]
                   for i in range(6)])
ne_s = g("snap", f"core/d{own}/neff_earlier")[0]
ne_l = g("lxtul", f"core/d{own}/neff_earlier")[0]
na = g("snap", f"core/d{own}/n_earlier_cells")[0]
ax.set_title("3  Loop reads, 6 layers x 6 passes (dash: D-4)\n"
             f"earlier read {cm_s:.0%} / {cm_l:.0%} shared; ~{ne_s:.0f} of {na:.0f} cells",
             fontsize=10, loc="left")
ax.set_ylabel("attention mass", fontsize=9)

# 4 write
ax = axs[3]
bars(ax, ["winner", "pseudo\n(snap) / zero", "span\ntokens"],
     [f"write/d{own}/coda_in_l0/winner_rms",
      {"snap": f"write/d{own}/coda_in_l0/pseudo_rms", "lxtul": None},
      f"write/d{own}/coda_in_l0/token_rms"], fmt="{:.2g}", log=True)
gs = arms["snap"].get("snap_params") or {}
gtxt = ", ".join(f"{v:.3f}" for v in gs.get("g", []))
ax.text(1.19, 2e-4, "LXTUL:\nexactly 0", ha="center", va="bottom", fontsize=8, color=INK)
ax.set_title("4  Write: RMS at the coda input\n"
             f"snap gate g = [{gtxt}] (init 0)",
             fontsize=10, loc="left")
ax.set_ylabel("RMS (log)", fontsize=9)

# 5 coda reads
ax = axs[4]
bars(ax, ["all slot\npositions", "winner", "pseudo /\nzero cells", "own span\ntokens"],
     [f"coda/d{own}/comb_slot", f"coda/d{own}/comb_winner",
      {"snap": f"coda/d{own}/comb_pseudo", "lxtul": f"coda/d{own}/comb_zero"},
      f"coda/d{own}/comb_owntok"], ref=(0.10, "10 % (D-1)"))
ax.set_title("5  Coda reads: a token's attention mass\n(4 layers x 8 heads; dash: D-1)",
             fontsize=10, loc="left")
ax.set_ylabel("attention mass", fontsize=9)

# 6 depth
ax = axs[5]
bars(ax, ["K1 - K6 (CE, nats)"], [f"ce/K1-K{own}"], fmt="{:+.4f}")
ax.set_xlim(-0.8, 1.9)
txt = []
for arm in ("snap", "lxtul"):
    a1 = g(arm, "coda/d1/vnorm_winner")[0]
    a6 = g(arm, f"coda/d{own}/vnorm_winner")[0]
    s1 = g(arm, "coda/d1/comb_slot")[0]
    s6 = g(arm, f"coda/d{own}/comb_slot")[0]
    c1 = g(arm, "loop/d1/t0/cm_winner_streammean", "common_mode")[0]
    c6 = g(arm, f"loop/d{own}/t{own - 1}/cm_winner_streammean", "common_mode")[0]
    txt.append(f"{'snap' if arm == 'snap' else 'LXTUL'}  depth 1 -> 6\n"
               f" coda slot mass  {s1:.3f} -> {s6:.3f}\n"
               f" winner read |v| {a1:.2f} -> {a6:.2f}")
ax.text(0.99, 0.97, "\n\n".join(txt), transform=ax.transAxes, ha="right", va="top",
        fontsize=8.5, family="monospace", color=INK)
ax.set_title("6  Depth 1 vs 6 (same rows)", fontsize=10, loc="left")
ax.set_ylabel("nats", fontsize=9)

# 7 worth
ax = axs[6]
bars(ax, ["zero both", "winner only", "pseudo only"],
     ["worth/zero/TOTAL", "worth/winner_only/TOTAL",
      {"snap": "worth/pseudo_only/TOTAL", "lxtul": None}], fmt="{:+.3f}")
ax.set_title("7  Worth: CE rise when the write is removed\n(tokens after a slot, nats)",
             fontsize=10, loc="left")
ax.set_ylabel("nats", fontsize=9)

# 8 the per-slot share of the carried cell through the loop
ax = axs[7]
xs = ["entry"] + [f"p{i + 1}" for i in range(own)]
for arm, col, mk, ls in (("snap", SNAP, "o", "-"), ("lxtul", LX, "s", "--")):
    ys = [1 - g(arm, f"loop/d{own}/cm_entry", "common_mode")[0]]
    ys += [1 - g(arm, f"loop/d{own}/t{i}/cm_winner", "common_mode")[0] for i in range(own)]
    ax.plot(range(len(xs)), ys, color=col, marker=mk, ls=ls, lw=2, ms=7, markeredgecolor=INK,
            label="snap" if arm == "snap" else "LXTUL")
    ax.text(0.15, ys[0] * (0.75 if arm == "lxtul" else 1.0), f"{ys[0]:.0%}", fontsize=8.5,
            color=INK, va="center")
    ax.text(1.1, ys[1] * (0.62 if arm == "lxtul" else 1.25), f"{ys[1]:.1%}", fontsize=8.5,
            color=INK, va="center")
    ax.text(len(xs) - 1 + 0.12, ys[-1] * (0.8 if arm == "lxtul" else 1.15), f"{ys[-1]:.1%}",
            fontsize=8.5, color=INK, va="center")
ax.set_yscale("log")
ax.set_xticks(range(len(xs)))
ax.set_xticklabels(xs, fontsize=9)
ax.set_xlim(-0.4, len(xs) - 0.1)
ax.set_ylabel("per-slot share of energy (log)", fontsize=9)
ax.set_title("8  Loop carrier: the part that differs between slots\n"
             "(1 - shared share, all 4 streams; RMS stays ~1.05)", fontsize=10, loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=8, frameon=False, loc="upper right")

from matplotlib.patches import Patch  # noqa: E402
n = arms["snap"]["rows"]
fig.legend(handles=[Patch(facecolor=SNAP, edgecolor=INK, label="snap (lxtul_snap, 5k)"),
                    Patch(facecolor=LX, edgecolor=INK, hatch="//",
                          label="LXTUL (rank-cnorm, 5k seed 1)")],
           loc="upper left", bbox_to_anchor=(0.02, 0.975), ncol=2, fontsize=10, frameon=False)
fig.text(0.02, 0.006, f"{n} val rows per arm, batch 1; bars 95 % bootstrap over rows; "
         f"eval depth {own}, router-followed; mass = gate-combined share; 3070 @ aa127cb",
         fontsize=8, color=MUTED)
fig.suptitle("Where the slot information goes: snap vs LXTUL, stage by stage",
             fontsize=12.5, x=0.02, y=0.995, ha="left")
fig.tight_layout(rect=(0, 0.015, 1, 0.955))
out = os.path.join(D, "dataflow.png")
fig.savefig(out, dpi=130)
print("wrote", out)
