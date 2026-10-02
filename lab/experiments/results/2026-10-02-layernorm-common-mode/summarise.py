"""Tables from the ln_common_mode_probe JSONs (throwaway)."""

import glob
import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
R = {
    os.path.basename(p)[:-5]: json.load(open(p))
    for p in sorted(glob.glob(f"{D}/*.json"))
    if not os.path.basename(p).startswith("cpu_test")
}
f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"  # noqa: E731


def top(c, k=4):
    return " ".join(f"{d['ch']}:{d['mean']:+.1f}" for d in c["top_abs_mean"][:k])


print("== A. channels (token level) ==")
for k, r in R.items():
    for nm in ("chan_live", "chan_twin"):
        c = r.get(nm)
        if c:
            print(
                f"{k:11s} {nm:9s} max/med|mean| {c['max_abs_mean_over_median_abs_mean']:7.1f} "
                f"top4 E-share {c['energy_share_top4']:.3f} meanvec E-share "
                f"{c['mean_vec_energy_share']:.3f}  {top(c)}"
            )
print("== A. pooled LN target ==")
for k, r in R.items():
    for nm in ("target_plain_prelude_ln", "target_twin_ln", "target_online_ln"):
        t = r.get(nm)
        if t:
            print(
                f"{k:11s} {nm:24s} n {t['n']:5d} cos {t['mean_pairwise_cos']:.3f} "
                f"cos->u {t['mean_cos_to_mean_dir']:.3f} meanE {t['mean_vec_share_of_energy']:.3f} "
                f"PRc {t['pr_centred']:6.2f} PRstd {t['pr_standardised_fitstats']:6.2f} "
                f"var/coord {t['per_coord_var_mean']:.4f} top4var {t['centred_var_share_top4']:.3f} "
                f"var_u {t['centred_var_share_along_u']:.3f}  {top(t)}"
            )
    if r.get("target_twin_pre_ln"):
        t = r["target_twin_pre_ln"]
        print(
            f"{k:11s} twin pre-LN pooled cos {t['mean_pairwise_cos']:.3f} PRc {t['pr_centred']:.2f}"
            f" rms {t['rms_norm']:.1f}"
        )
print("== H3 exit cells raw / centred ==")
for k, r in R.items():
    h = r.get("h3_exit")
    if h:
        a, b = h["raw"], h["centred"]
        print(
            f"{k:11s} wcos {a['within_slot_cos']:+.3f}/{b['within_slot_cos']:+.3f} "
            f"wrank {a['within_slot_rank']:.3f}/{b['within_slot_rank']:.3f} "
            f"spread {a['cell_spread']:.3f}/{b['cell_spread']:.3f} "
            f"xcos {a['across_slot_cos']:+.3f}/{b['across_slot_cos']:+.3f} "
            f"xPR {a['across_slot_pr']:.2f}/{b['across_slot_pr']:.2f}"
        )
print("== H2 offline picks (per pass) ==")
for k, r in R.items():
    l = r.get("lsel")
    if not l:
        continue
    for t, p in sorted(l["per_pass"].items(), key=lambda kv: int(kv[0])):
        a, rv = p["agree_ship_vs"], p["agree_router_vs"]
        print(
            f"{k:11s} t{t} ship==zstd {a['zstd']:.3f} ==own {a['own']:.3f} ==perp {a['perp']:.3f} "
            f"==u {a['along_u']:.3f} | router== ship {rv['ship']:.3f} zstd {rv['zstd']:.3f} "
            f"own {rv['own']:.3f} perp {rv['perp']:.3f} u {rv['along_u']:.3f} | varshare_u "
            f"med {p['between_cell_var_share_along_u_median']:.3f} mean "
            f"{p['between_cell_var_share_along_u_mean']:.3f}"
        )
print("== H1 ==")
for k, r in R.items():
    l = r.get("lsel")
    if l:
        h = l["h1"]
        print(f"{k:11s} " + " ".join(f"{a}={b:.4g}" for a, b in h.items()))
print("== P per pass: geometry / step / probes ==")
for k, r in R.items():
    l = r.get("lsel")
    rows = []
    if l:
        for t, p in sorted(l["per_pass"].items(), key=lambda kv: int(kv[0])):
            rows.append(
                (
                    t,
                    p["geometry"],
                    p.get("step"),
                    p["ridge_std_target_from_router_winner"]["r2_eval"],
                    p["head_r2_std_router_winner"],
                )
            )
    elif r.get("geometry_by_depth"):
        for t, g in sorted(r["geometry_by_depth"].items(), key=lambda kv: int(kv[0])):
            rows.append((t, g, g.get("step"), None, None))
    for t, g, s, rr, hr in rows:
        st = (
            (
                f"step rel {s['rel_rms']:.3f} along_v {s['share_along_v']:.3f} coh "
                f"{s['coherence_across_cells']:.3f}"
            )
            if s
            else ""
        )
        print(
            f"{k:11s} t{t} |c| {g['rms_norm']:7.1f} cos_v {g['cos_to_v']:.3f} along {g['rms_along_v']:7.1f} "
            f"perp {g['rms_perp_v']:6.1f} cen {g['rms_centred']:6.1f} betw {g['rms_between_slot']:6.1f} "
            f"within {g['rms_within_slot']:6.1f} {st} ridgeR2 {f(rr)} headR2std {f(hr)}"
        )
print("== H2 follow CE ==")
for k, r in R.items():
    fo = r.get("follow")
    if fo:
        print(
            f"{k}: rows {fo['rows']} tokens {fo['n_tokens']} ce "
            + " ".join(f"{a}={b:.4f}" for a, b in fo["ce"].items())
        )
        for nm in ("delta_vs_router", "delta_vs_teacher_ship"):
            print(
                f"   {nm}: "
                + "  ".join(
                    f"{a} {d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]"
                    for a, d in fo[nm].items()
                )
            )
print("== v-ablation of the written cells (coda CE, paired, 96 rows) ==")
for k, r in R.items():
    if not k.startswith("vab_"):
        continue
    v = r["vablate"]
    g = lambda x: f"{x['point']:+.4f} [{x['lo']:+.4f},{x['hi']:+.4f}]"  # noqa: E731
    for dpt, d in sorted(v["by_depth"].items(), key=lambda kv: int(kv[0])):
        w = d["written_cells"]
        km = d.get("k_minus_first", {})
        print(
            f"{k[4:]:11s} K{dpt} ce {d['ce_shipped']:.4f} cos_v {w['cos_to_v']:.3f} a_bar "
            f"{w['a_bar']:7.1f} amp_std {w['amp_std']:6.1f} | mean {g(d['mean_ablate'])} zero "
            f"{g(d['zero_ablate'])}"
            + (
                f" | K-K1 ship {g(km['shipped'])} mean {g(km['mean'])} " f"zero {g(km['zero'])}"
                if km
                else ""
            )
        )
for p in sorted(glob.glob(f"{D}/follow/*.json")):
    fo = json.load(open(p))["follow"]
    print(
        f"{os.path.basename(p)[:-5]} (follow): ce "
        + " ".join(f"{a}={b:.4f}" for a, b in fo["ce"].items())
    )
    for nm in ("delta_vs_router", "delta_vs_teacher_ship"):
        print(
            f"   {nm}: "
            + "  ".join(
                f"{a} {d['point']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}]" for a, d in fo[nm].items()
            )
        )
