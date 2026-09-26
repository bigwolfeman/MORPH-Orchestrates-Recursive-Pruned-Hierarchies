"""LX-Concept step 4, THE GATE: can fp01's slot state predict the next span's concept?

Prereg: ``lab/experiments/planned/2026-09-26-lx-concept-precheck.md`` (Predictions, Reading
rule). Terms: ``_concept.py``'s module docstring (slot s, next span, entry, exit, exit_raw).

TARGETS of slot s (all describe span s+1, from ``concept_dict.py``):

  type64 / type256  the k-means span type of E's code (one class; softmax CE).
  sae_top4          the CoCoMix label: the 4 SAE latents with the highest attribution
                    through the teacher's decoder; loss = the mean over the 4 of the
                    softmax CE over 4,096 latents (CoCoMix's L_concept).
  sae_low4          the 4 lowest-attribution latents, same loss (reported, not graded).

FEATURES of slot s, per fp01 checkpoint (5k, 10k; forced depth 6): ``exit`` (the readout the
parallel head reads, rollout mean), ``exit_raw`` (the carrier before the readout), ``entry``
(the prelude state at the slot, which under the strict geometry has seen span s alone). The
NO-CONTEXT baseline is the marginal prior: additive-0.5 smoothed frequencies of the target
indices in the fit rows. A head is linear softmax or one hidden layer of 512 GELU units,
AdamW (lr 1e-3, wd 1e-2), on z-scored features, early-stopped on the dev rows, the linear
head's bias initialised at the prior.

SPLITS, by ROW (spans of a row are never split across sets): the train region's rows 80 / 10
/ 10 into fit / dev / test (seeded); the val region's 480 rows are the PRIMARY held-out set
(the rows every LX instrument reads); the train region's test rows are the secondary one.

READINGS per (checkpoint, feature, target, head), on each held-out set: CE (nats per target
index), the prior's CE on the same spans, ``gain = prior CE - head CE`` with a 95 % bootstrap
over ROWS, top-1 and top-5 hits (the fraction of a span's targets inside the top 1 / 5
predicted classes). And ``exit - entry``: the exit head's CE minus the entry head's on the
same spans, paired, the loop's own share of the predictability.

Reading rule (the prereg's): on the 10k checkpoint's EXIT, if the gain over the prior is
below 0.1 nats on type64, type256 AND sae_top4 (primary set, point estimate), LX-Concept is
dead at this scale. ``DEAD`` / ``ALIVE`` is printed per target and overall.

Usage (GPU for speed; CPU works):
  python lab/divergence/concept_precheck.py --dump /home/wolfe/morph-scratch/concept/dump \\
      --out /home/wolfe/morph-scratch/concept/precheck.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import _concept as cc  # noqa: E402

GATE_NATS = 0.1
GRADED = ("type64", "type256", "sae_top4")


def load_targets(rdir: str, n_latents: int) -> dict[str, tuple[np.ndarray, int]]:
    """target name -> (``[n, T]`` int64 labels, number of classes); missing files skipped."""
    out = {}
    for f in sorted(os.listdir(rdir)):
        if f.startswith("types_M") and f.endswith(".npy"):
            M = int(f[len("types_M"):-4])
            out[f"type{M}"] = (np.load(os.path.join(rdir, f)).reshape(-1, 1), M)
    for name, f in (("sae_top4", "attr_top4.npy"), ("sae_low4", "attr_low4.npy")):
        p = os.path.join(rdir, f)
        if os.path.exists(p):
            out[name] = (np.load(p), n_latents)
    return out


def row_split(rows: np.ndarray, fracs=(0.8, 0.1, 0.1), seed: int = 0) -> list[np.ndarray]:
    """Boolean masks over spans for a split of the distinct ROWS (never splits a row)."""
    u = np.unique(rows)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(u)
    cut = np.cumsum([int(round(f * u.size)) for f in fracs[:-1]])
    parts = np.split(perm, cut)
    return [np.isin(rows, p) for p in parts]


def evaluate(logp: torch.Tensor, prior: torch.Tensor, y: torch.Tensor, rows: np.ndarray,
             n_boot: int) -> dict:
    """CE of the head and of the prior on the same spans, the paired gain, hits@1/@5."""
    ce_h = cc.set_ce(logp, y).double().cpu().numpy()
    ce_p = cc.set_ce(prior.to(logp.device).expand_as(logp), y).double().cpu().numpy()
    gain = cc.row_block_ci(ce_p, ce_h, rows, n_boot=n_boot)
    return {"n_spans": int(y.shape[0]), "n_rows": int(np.unique(rows).size),
            "ce": float(ce_h.mean()), "prior_ce": float(ce_p.mean()),
            "gain": gain, "top1": float(cc.hits(logp, y, 1).mean()),
            "top5": float(cc.hits(logp, y, 5).mean()),
            "prior_top1": float(cc.hits(prior.to(logp.device).expand_as(logp), y, 1).mean()),
            "prior_top5": float(cc.hits(prior.to(logp.device).expand_as(logp), y, 5).mean()),
            "_ce_per_span": ce_h}


@torch.no_grad()
def _logp(h, X: torch.Tensor) -> torch.Tensor:
    return torch.cat([h(X[i:i + 8192]) for i in range(0, X.shape[0], 8192)])


def run(dump: str, students: list[str], features: list[str], heads: dict[str, int],
        device: str, n_boot: int, seed: int, max_epochs: int) -> dict:
    tr_dir, va_dir = os.path.join(dump, "train"), os.path.join(dump, "val")
    fitj = json.load(open(os.path.join(dump, "dict", "fit.json")))
    n_lat = int(fitj["sae"]["n_latents"])
    ttr, tva = load_targets(tr_dir, n_lat), load_targets(va_dir, n_lat)
    rows_tr = np.load(os.path.join(tr_dir, "keys.npz"))["row"]
    rows_va = np.load(os.path.join(va_dir, "keys.npz"))["row"]
    fit_m, dev_m, test_m = row_split(rows_tr, seed=seed)
    res: dict = {"n_train_region_spans": int(rows_tr.size), "n_val_spans": int(rows_va.size),
                 "split_spans": {"fit": int(fit_m.sum()), "dev": int(dev_m.sum()),
                                 "test": int(test_m.sum())},
                 "targets": {}, "readings": {}, "gate_nats": GATE_NATS}
    for tname, (ytr_all, ncls) in ttr.items():
        if tname not in tva:
            continue
        n_tr = min(ytr_all.shape[0], rows_tr.size)
        ytr_all = torch.from_numpy(ytr_all[:n_tr]).long().to(device)
        yva = torch.from_numpy(tva[tname][0]).long().to(device)
        prior = cc.marginal_logp(ytr_all[torch.from_numpy(fit_m[:n_tr]).to(device)], ncls)
        res["targets"][tname] = {"n_classes": ncls, "T": int(ytr_all.shape[1]),
                                 "prior_entropy_nats": cc.entropy_nats(prior),
                                 "log_n_classes": cc.log_uniform(ncls)}
        sets = {"val": (yva, rows_va[:yva.shape[0]], None),
                "train_test": (ytr_all[torch.from_numpy(test_m[:n_tr]).to(device)],
                               rows_tr[:n_tr][test_m[:n_tr]], test_m[:n_tr])}
        res["readings"].setdefault(tname, {})["prior"] = {
            s: {k: v for k, v in evaluate(prior.to(device).unsqueeze(0).expand(y.shape[0], -1),
                                          prior, y, r, n_boot).items() if k != "_ce_per_span"}
            for s, (y, r, _m) in sets.items()}
        for st in students:
            for feat in features:
                ptr = os.path.join(tr_dir, st, f"{feat}.npy")
                pva = os.path.join(va_dir, st, f"{feat}.npy")
                if not (os.path.exists(ptr) and os.path.exists(pva)):
                    continue
                Xtr_all = torch.from_numpy(np.load(ptr)[:n_tr]).float().to(device)
                Xva = torch.from_numpy(np.load(pva)[:yva.shape[0]]).float().to(device)
                fm = torch.from_numpy(fit_m[:n_tr]).to(device)
                dm = torch.from_numpy(dev_m[:n_tr]).to(device)
                tm = torch.from_numpy(test_m[:n_tr]).to(device)
                Xf, Xd, Xt, Xv = cc.standardise(Xtr_all[fm], Xtr_all[dm], Xtr_all[tm], Xva)
                for hname, hid in heads.items():
                    t0 = time.time()
                    h, info = cc.fit_head(Xf, ytr_all[fm], Xd, ytr_all[dm], ncls, hidden=hid,
                                          seed=seed, max_epochs=max_epochs, prior=prior)
                    ev = {"val": evaluate(_logp(h, Xv), prior, yva, rows_va[:yva.shape[0]], n_boot),
                          "train_test": evaluate(_logp(h, Xt), prior, ytr_all[tm],
                                                 rows_tr[:n_tr][test_m[:n_tr]], n_boot)}
                    key = f"{st}/{feat}/{hname}"
                    res["readings"][tname][key] = {"fit": info, "wall_s": round(time.time() - t0, 1),
                                                   **ev}
                    g = ev["val"]["gain"]
                    print(f"[{tname:8s}] {key:28s} val CE {ev['val']['ce']:.4f} prior "
                          f"{ev['val']['prior_ce']:.4f} gain {g['point']:+.4f} "
                          f"[{g['lo']:+.4f}, {g['hi']:+.4f}]  top1 {ev['val']['top1']:.3f} "
                          f"top5 {ev['val']['top5']:.3f} | test gain "
                          f"{ev['train_test']['gain']['point']:+.4f}", flush=True)
            # exit - entry, paired, per head and set
            for hname in heads:
                ke, kn = f"{st}/exit/{hname}", f"{st}/entry/{hname}"
                R = res["readings"][tname]
                if ke in R and kn in R:
                    for s, rr in (("val", rows_va[:yva.shape[0]]),
                                  ("train_test", rows_tr[:n_tr][test_m[:n_tr]])):
                        R[f"{st}/exit_minus_entry/{hname}"] = R.get(
                            f"{st}/exit_minus_entry/{hname}", {})
                        R[f"{st}/exit_minus_entry/{hname}"][s] = cc.row_block_ci(
                            R[ke][s]["_ce_per_span"], R[kn][s]["_ce_per_span"], rr, n_boot=n_boot)
    # strip the per-span arrays before writing
    for tname, R in res["readings"].items():
        for key, v in R.items():
            for s in ("val", "train_test"):
                if isinstance(v.get(s), dict):
                    v[s].pop("_ce_per_span", None)
    return res


def verdict(res: dict, student: str, heads: list[str]) -> dict:
    """The prereg's rule on ``student``'s exit / exit_raw heads (best of heads and exit
    variants), primary set: DEAD per target if the best gain < GATE_NATS."""
    out = {}
    for t in GRADED:
        R = res["readings"].get(t, {})
        g = [R[k]["val"]["gain"]["point"] for k in R
             if k.startswith(f"{student}/exit") and "minus" not in k
             and k.rsplit("/", 1)[1] in heads]
        if g:
            best = max(g)
            out[t] = {"best_exit_gain": best, "verdict": "ALIVE" if best >= GATE_NATS else "DEAD"}
    out["overall"] = ("ALIVE" if any(v["verdict"] == "ALIVE" for v in out.values()
                                     if isinstance(v, dict)) else "DEAD") if out else "NO DATA"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dump", required=True)
    ap.add_argument("--students", default="fp01_5k,fp01_10k")
    ap.add_argument("--features", default="exit,exit_raw,entry")
    ap.add_argument("--heads", default="linear:0,mlp:512")
    ap.add_argument("--gate-student", default="fp01_10k")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    heads = {h.split(":")[0]: int(h.split(":")[1]) for h in a.heads.split(",")}
    res = run(a.dump, a.students.split(","), a.features.split(","), heads, a.device,
              a.n_boot, a.seed, a.max_epochs)
    res["verdict"] = verdict(res, a.gate_student, list(heads))
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for t, v in res["verdict"].items():
        print(f"GATE {t}: {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
