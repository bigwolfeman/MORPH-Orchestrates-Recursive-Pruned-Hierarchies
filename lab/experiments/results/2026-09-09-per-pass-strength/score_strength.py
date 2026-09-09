#!/usr/bin/env python
"""Score the per-pass-strength panel (three ternary scale-rule arms on the Parcae-entry
recipe, plus the bf16-everywhere ceiling when it has run) from the core_depth_sweep JSONs
(+ per-token files), the anatomy and init-probe JSONs, the probes, run logs and the queue
log. References: `parcae-entry` at depth 6 (the ternary base, `../2026-09-09-arc-e19/`) and
`depthcand-dense-core` at depth 6 (the bf16-core diagnostic, `../2026-09-09-arc-e20/`).
Pairing and readers: `lab/divergence/sweep_score.py`.

Loop CONTRIBUTION is the reading (the K-curve, the core MLP branch out/in per pass, the
state movement); CE at 5,000 steps is a horizon reading and ranks nothing.

Usage (repo root): PYTHONPATH=. python lab/experiments/results/2026-09-09-per-pass-strength/score_strength.py
                   [RESULTS_DIR] [QUEUE_LOG] [E19_RESULTS_DIR] [E20_RESULTS_DIR] [CHECKPOINT_ROOT]
The optional CHECKPOINT_ROOT (default /home/wolfe/morph-to/checkpoints/morph) enables the
TTQ learned-scale readout off `scale-ttq/step_5000.pt` (P-str-d); skipped when absent.
"""
from __future__ import annotations

import json
import os
import sys

from lab.divergence.sweep_score import (ci, fmt, load_sweep, paired, peak_gb, probe_stats,
                                        val_curve, verdicts, wall_h)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ARMS = ["scale-norm-match", "scale-ttq", "threshold-03", "precision-bf16-all"]
TERNARY_ARMS = ARMS[:3]
BASE, BF16CORE = "parcae-entry", "depthcand-dense-core"
T = "6"  # every arm trains at Poisson mean 6
CKS = [2500, 5000]
DEPTHS = ["0", "1", "2", "3", "6", "9", "12", "16"]
ART = tuple(os.path.join(REPO, "ignored", "experiment-artifacts", d)
            for d in ("2026-09-09-per-pass-strength", "2026-09-09-arc-e20", "2026-09-09-arc-e19"))
VAL_GAP_LIMIT = 0.05  # sweep-vs-trainer gap above this is a packing bug (E18/E19: ~0.025)
BASE_WALL_H = 1.00    # the Parcae-entry arm's wall clock
BF16CORE_END = 3.9275  # depthcand-dense-core depth-6 CE at 5000 (P-str-e reference)


def yes(b: bool) -> str:
    return "TRUE" if b else "FALSE"


def kdiff(s: dict, a: str, b: str, dirs) -> tuple:
    """CE at depth a minus depth b within one sweep (exact token pairing)."""
    return paired(s, a, s, b, dirs)


def ttq_gamma_readout(ckpt: str) -> list[str]:
    """Learned γ₊ / γ₋ of every TernarySTE against the latent's mean|W| (P-str-d)."""
    import torch
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    lines, ratios = [], []
    for k, gp in sd.items():
        if not k.endswith("parametrizations.weight.0.gamma_pos"):
            continue
        stem = k[: -len("parametrizations.weight.0.gamma_pos")]
        w = sd[stem + "parametrizations.weight.original"].float()
        gn = sd[stem + "parametrizations.weight.0.gamma_neg"].float()
        absmean = w.abs().mean().item()
        rp, rn = gp.float().mean().item() / absmean, gn.mean().item() / absmean
        if k.startswith("core.") or ".core." in k:
            ratios.append((rp + rn) / 2)
        lines.append(f"    {stem.rstrip('.'):48s} γ+/mean|W| {rp:.3f}  γ-/mean|W| {rn:.3f}")
    core = sum(ratios) / len(ratios) if ratios else float("nan")
    lines.insert(0, f"  core mean(γ)/mean|W| over {len(ratios)} tensors: {core:.3f}"
                    f" -> γ grew past 1.3x its absmean init: {yes(core > 1.3)}")
    return lines


def main() -> None:
    root = sys.argv[1] if len(sys.argv) > 1 else HERE
    queue = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "queue_strength.log")
    e19 = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "2026-09-09-arc-e19")
    e20 = sys.argv[4] if len(sys.argv) > 4 else os.path.join(HERE, "..", "2026-09-09-arc-e20")
    ck_root = sys.argv[5] if len(sys.argv) > 5 else "/home/wolfe/morph-to/checkpoints/morph"
    dirs = (root, HERE, e19, e20) + ART
    where = {a: root for a in ARMS}
    where[BASE], where[BF16CORE] = e19, e20
    S = {(a, ck): s for a in ARMS for ck in CKS
         if (s := load_sweep(os.path.join(root, f"sweep_{a}_{ck}.json"))) is not None}
    for ref in (BASE, BF16CORE):
        r = load_sweep(os.path.join(where[ref], f"sweep_{ref}_5000.json"))
        if r is not None:
            S[ref, 5000] = r
    present = [a for a in ARMS if (a, 5000) in S]
    missing = [a for a in ARMS if (a, 5000) not in S]
    if missing:
        print(f"NOT RUN YET (no sweep JSON): {missing}")
    vals = {}
    for a in ARMS + [BASE, BF16CORE]:
        log = os.path.join(where[a], f"run_{a}.log")
        if os.path.exists(log):
            vals[a] = val_curve(log, final=True)

    def pair(a, da, b, db):
        return paired(S[a], da, S[b], db, dirs) if (a in S and b in S) else None

    print("=== the E17 rule: sweep CE at the trained depth vs the trainer's held-out loss at that step")
    for (a, ck), s in S.items():
        v = vals.get(a, {})
        tv = v.get(ck) if ck in v else (v[max(v)] if v and ck >= max(v) else None)  # the final eval is keyed last+1
        gap = s["depths"][T]["ce_tokens"] - tv if tv is not None else float("nan")
        print(f"  {a:20s}@{ck} depth {T}: sweep {s['depths'][T]['ce_tokens']:.4f}  trainer {tv if tv is not None else float('nan'):.4f}"
              f"  gap {gap:+.4f} {'OK' if abs(gap) < VAL_GAP_LIMIT else 'BUG?'}")

    print("\n=== token CE by forced depth (480 rows) and the paired K-differences")
    print(f"{'arm':20s} {'ck':>5s} " + " ".join(f"{'T=' + d:>7s}" for d in DEPTHS) + " | K1-K6 | K3-K6 [CI] | K6-K12")
    for (a, ck), s in S.items():
        k1, k3, k12 = kdiff(s, "1", T, dirs), kdiff(s, "3", T, dirs), kdiff(s, T, "12", dirs)
        print(f"{a:20s} {ck:5d} " + " ".join(f"{s['depths'][d]['ce_tokens']:7.4f}" if d in s["depths"] else f"{'-':>7s}" for d in DEPTHS)
              + f" | {k1[0]:+.4f} | {k3[0]:+.4f} [{k3[1]:+.4f},{k3[2]:+.4f}] | {k12[0]:+.4f}")

    print("\n=== P-str-b (mechanism): core MLP branch out/in at iteration 6 (anatomy, 3 rows); ternary > 0.4, bf16-all > 0.7")
    branch = {}
    for a in present + [BASE, BF16CORE]:
        ap = os.path.join(where[a], f"anatomy_{a}.json")
        if not os.path.exists(ap):
            continue
        an = json.load(open(ap))
        attn6, mlp6 = an["branch_out_in"]["attn"].get("6", []), an["branch_out_in"]["mlp"].get("6", [])
        mv = an["consecutive"]
        bar = 0.7 if a == "precision-bf16-all" else 0.4
        ok = min(mlp6) > bar if mlp6 else False
        branch[a] = ok
        print(f"  {a:20s} mlp@6 {[round(x, 3) for x in mlp6]} (min > {bar}: {yes(ok)})  attn@6 {[round(x, 3) for x in attn6]}"
              f"  movement {[round(x, 3) for x in mv]}  carrier t8 norm {an['carrier'][-1]['norm']:.0f} rank {an['carrier'][-1]['rank']:.1f}")

    print("\n=== P-str-c (contribution) at 5000: K1-K6 > 0.08; K3-K6 > 0.005 with CI > 0; > 0.02 (the panel bar); K6-K12 within +-0.005")
    for a in present:
        s = S[a, 5000]
        k1, k3, k12 = kdiff(s, "1", T, dirs), kdiff(s, "3", T, dirs), kdiff(s, T, "12", dirs)
        print(f"  {a:20s} K1-K6 {k1[0]:+.4f} -> {yes(k1[0] > 0.08)};  K3-K6 {k3[0]:+.4f} lo {k3[1]:+.4f} -> >0.005: {yes(k3[0] > 0.005 and k3[1] > 0)}"
              f" >0.02: {yes(k3[0] > 0.02 and k3[1] > 0)};  K6-K12 {k12[0]:+.4f} -> {yes(abs(k12[0]) <= 0.005)};"
              f"  sweep's own K1-K6 {ci(s, 'K1-K6')} K3-K6 {ci(s, 'K3-K6')}")
    if ("precision-bf16-all", 5000) in S and (BF16CORE, 5000) in S:
        ka = kdiff(S["precision-bf16-all", 5000], "1", T, dirs)[0]
        kb = kdiff(S[BF16CORE, 5000], "1", T, dirs)[0]
        print(f"  bf16-all K1-K6 {ka:+.4f} vs bf16-core {kb:+.4f}: within 0.03: {yes(abs(ka - kb) <= 0.03)}")

    print("\n=== Binding check: a ternary arm with P-str-b TRUE and K1-K6 > 0.08")
    for a in [x for x in present if x in TERNARY_ARMS]:
        k1 = kdiff(S[a, 5000], "1", T, dirs)[0]
        print(f"  {a:20s} branch {yes(branch.get(a, False))}  K1-K6 > 0.08 {yes(k1 > 0.08)}  -> becomes the rule: {yes(branch.get(a, False) and k1 > 0.08)}")

    print("\n=== P-str-e (horizon readings, NOT rankings): arm@6 - parcae-entry@6 and - bf16-core@6, token-paired")
    for a in present:
        r = pair((a, 5000), T, (BASE, 5000), T)
        rb = pair((a, 5000), T, (BF16CORE, 5000), T)
        if r:
            print(f"  {a:20s} vs base {fmt(r)}  within +-0.03: {yes(abs(r[0]) <= 0.03)}" + (f"  vs bf16-core {fmt(rb)}" if rb else ""))
    if ("precision-bf16-all", 5000) in S:
        e = S["precision-bf16-all", 5000]["depths"][T]["ce_tokens"]
        print(f"  bf16-all depth-6 CE {e:.4f} under the bf16-core arm's {BF16CORE_END}: {yes(e < BF16CORE_END)}")

    print("\n=== P-str-d: TTQ learned scales at 5000 (γ over the latent's mean|W|)")
    ck = os.path.join(ck_root, "scale-ttq", "step_5000.pt")
    if os.path.exists(ck):
        for line in ttq_gamma_readout(ck):
            print(line)
    else:
        print(f"  checkpoint {ck} not present; readout skipped (the filed value: core mean 0.857, gate_up 0.50-0.57, down 0.81-1.39)")

    print("\n=== init probes (T8 CE by loop start; spread within 0.01 = start-independent fixed point)")
    for a in present + [BASE]:
        ip = os.path.join(where[a], f"init_probe_{a}.json")
        if not os.path.exists(ip):
            continue
        pr = json.load(open(ip))
        at8 = {m: (c.get("8") if "8" in c else c.get(8)) for m, c in pr["ce"].items()}
        spread = max(at8.values()) - min(at8.values())
        print(f"  {a:20s} " + " ".join(f"{m}={v:.4f}" for m, v in at8.items()) + f"  spread {spread:.4f} -> {yes(spread <= 0.01)}")

    print("\n=== P-str-a/f: survival, tripwire, wall clock, peak, held-out curve")
    wh = wall_h(queue, "") if os.path.exists(queue) else {}
    vd = verdicts(queue, "") if os.path.exists(queue) else {}
    for a in present + [BASE]:
        log = os.path.join(where[a], f"run_{a}.log")
        if not os.path.exists(log):
            print(f"  {a}: no run log")
            continue
        probe = next((p for p in [os.path.join(d, f"probe_{a}.jsonl") for d in dirs] if os.path.exists(p)), None)
        ps = probe_stats(probe) if probe else "no probe file"
        v = vals[a]
        w = wh.get(a, float("nan"))
        print(f"  {a:20s} verdict={vd.get(a, '?' if a in ARMS else 'E19')}  wall {w:.2f} h ({w / BASE_WALL_H:.2f}x base; within 1.1x: {yes(w <= 1.1 * BASE_WALL_H)})  "
              f"peak {peak_gb(log):.2f} GB (< 16: {yes(peak_gb(log) < 16)})  val@2500 {v.get(2500, float('nan')):.4f}  final {v[max(v)]:.4f}  {ps}")
    if BASE in vals:
        for a in present:
            if a in vals:
                steps = sorted(set(vals[a]) & set(vals[BASE]))
                print(f"  {a} - base held-out (trainer's batches): " +
                      " ".join(f"{s}:{vals[a][s] - vals[BASE][s]:+.3f}" for s in steps if s % 1000 == 0 or s == max(steps)))


if __name__ == "__main__":
    main()
