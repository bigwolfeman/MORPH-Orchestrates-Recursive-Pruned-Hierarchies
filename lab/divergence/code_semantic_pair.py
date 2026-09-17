"""Pair one semantic-probe condition against another ACROSS models, cut by cut.

`code_semantic_probe.py` scores every model on the same 120 validation cuts (same seed,
same rule, same packer) and saves the per-cut MiniLM cosine and content overlap under
`per_cut`. This script reads those arrays from several probe JSONs and reports the paired
gap of each arm condition minus the reference condition with a 2,000-draw bootstrap over
cuts. Before pairing it checks that every file scored the SAME cuts: the true spans written
in the sibling `.txt` files must match line for line.

    python lab/divergence/code_semantic_pair.py \
        --ref ruler=OWN=RESULTS/semantic_ctl_strict-ruler_20000.json \
        --arms dplan10k=OWN@4=RESULTS/semantic_tul-code-dplan_10000.json \
               plain=OWN=RESULTS/semantic_ctl_plain_20000.json \
        --out RESULTS/semantic_pairs_10000.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

METRICS = ("cos_true", "overlap_true")


def _parse(spec: str) -> tuple[str, str, str]:
    label, cond, path = spec.split("=", 2)
    return label, cond, path


def _true_lines(json_path: str) -> list[str]:
    txt = json_path.rsplit(".", 1)[0] + ".txt"
    if not os.path.exists(txt):
        raise FileNotFoundError(f"{txt}: the probe's .txt sibling is needed to check the cuts")
    with open(txt) as f:
        return [ln for ln in f if ln.startswith("TRUE    ")]


def _load(spec: str) -> tuple[str, dict[str, np.ndarray], list[str]]:
    label, cond, path = _parse(spec)
    with open(path) as f:
        d = json.load(f)
    if cond not in d["per_cut"]:
        raise KeyError(f"{path}: condition {cond!r} not in {sorted(d['per_cut'])}")
    arrs = {m: np.asarray(d["per_cut"][cond][m], dtype=np.float64) for m in METRICS}
    return f"{label}:{cond}", arrs, _true_lines(path)


def paired(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, draws: int = 2000) -> list[float]:
    d = x - y
    n = len(d)
    bs = [d[rng.integers(0, n, n)].mean() for _ in range(draws)]
    return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="LABEL=COND=probe.json (the reference)")
    ap.add_argument("--arms", required=True, nargs="+", help="LABEL=COND=probe.json ...")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref_name, ref, ref_true = _load(a.ref)
    rng = np.random.default_rng(0)
    out = {"ref": ref_name, "cuts": len(ref_true), "pairs": {}}
    print(f"reference {ref_name}: {len(ref_true)} cuts")
    for spec in a.arms:
        name, arm, arm_true = _load(spec)
        if arm_true != ref_true:
            n_bad = sum(p != q for p, q in zip(arm_true, ref_true)) + abs(len(arm_true) - len(ref_true))
            raise SystemExit(f"{name}: {n_bad} of {len(ref_true)} true spans differ from {ref_name}; "
                             "the probes did not score the same cuts")
        res = {m: paired(arm[m], ref[m], rng) for m in METRICS}
        res["mean"] = {m: float(arm[m].mean()) for m in METRICS}
        out["pairs"][f"{name}-{ref_name}"] = res
        print(f"  {name:22s} minus ref: cos(true) {res['cos_true'][0]:+.4f} "
              f"[{res['cos_true'][1]:+.4f}, {res['cos_true'][2]:+.4f}]  "
              f"overlap(true) {res['overlap_true'][0]:+.4f} "
              f"[{res['overlap_true'][1]:+.4f}, {res['overlap_true'][2]:+.4f}]  "
              f"(arm mean cos {res['mean']['cos_true']:.4f}, ref {ref['cos_true'].mean():.4f})")
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
