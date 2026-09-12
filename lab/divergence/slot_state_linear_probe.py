"""Does the slot state carry SCOREABLE structure at all? (Step 0 of the latent-z arc.)

The question, and why it is the gate
------------------------------------
Every slot-loop arm on this tree has been scored on whether the CODA reads the loop's
state. None has asked the prior question: is there anything in that state a linear reader
could use to tell a good thought from a bad one? The literature disagrees about this by a
mile, and the disagreement is about HOW the latent was trained:

* on Huginn-style recurrent trajectories a correctness classifier on the latent reaches
  ROC-AUC near 1.0 (Latent Thinking Optimization, arXiv 2509.26314);
* on Coconut-style latents supervised only through downstream tokens it sits at chance
  (arXiv 2510.12167, Cohen's d 0.17).

MORPH's slot is the second kind. If this probe reads chance, then the two energy arms of
`lab/experiments/planned/2026-09-12-arc-latent-z-gradient.md` have nothing to descend and
the honest next move is the TARGET, not the conditioning. If it reads above chance and the
signal RISES with the pass index, the loop is already building something and the arms are
asking it for more of the right thing. Either reading is worth the run; that is why it is
step 0 and not a readout.

What it measures
----------------
For each slot, a per-pass state and one binary label.

* **The state.** ``z_t`` is the tensor ``TULSlots.prefix_project`` WOULD receive if the
  loop stopped at pass ``t`` — captured by forcing ``tul.slot_depth_fixed`` to ``t`` and
  running the forward, which is the same lever ``core_depth_sweep.py`` and
  ``slot_state_probe.py`` use. ``z_0`` is the loop's ENTRY, ``core_init(e)``, read off a
  forward hook — the same tensor ``slot_z_optimize.py`` calls ``h0`` and scores as
  ``ce_entry``.
* **The label.** ``y = 1`` when the coda's MEAN token CE over that slot's NEXT span is
  below the median over every scored slot of the run. That is the coda's own opinion of the
  span the slot is supposed to have planned, and it is computed by
  ``morph.model.tul_egrad.slot_outcome_labels`` — the SAME function the ``disc`` energy
  arm trains its critic against, so the probe and the arm cannot drift apart.

Reported per pass and for the concatenated trajectory ``z_{0..t}``: 5-fold
cross-validated ROC-AUC on out-of-fold scores, with a 200-draw bootstrap CI over slots,
against the majority baseline (0.5 by construction — a median split). Two feature
reductions are reported side by side, because the tree has never settled which one a
reader of this carrier should use: the FLATTENED Hyper-Connection streams (``n * C``, what
``prefix_project`` actually sees) and the stream MEAN (``C``, what ``_readout`` and the MUX
head see). Finding F2 says the two disagree by a lot about what the loop did.

Also reported, because two-scale latent dynamics (arXiv 2509.23314) predicts a specific
shape: the per-pass relative step ``||z_{t+1} - z_t|| / ||z_t||``, the cosine between
SUCCESSIVE steps, an isotropy reading (entropy effective rank over the dimension) and Hoyer
sparsity — each split by label, so "the good slots move differently" is falsifiable rather
than a vibe.

Self-checks, all three printed with the results
-----------------------------------------------
1. **Label shuffle** — the whole pipeline re-run on permuted labels must read ~0.5. A probe
   that reads high on shuffled labels is leaking (fold contamination, or a feature built
   from the label).
2. **Entry vs exit** — ``z_0`` alone against ``z_T`` alone. If they read the same, the loop
   adds no separable structure whatever the absolute level is.
3. **The split point** — the recorded ``z`` replayed through ``ZSplit`` must reproduce the
   trained forward's token CE EXACTLY, and ``prefix_project`` must receive the substituted
   object. Reused verbatim from ``slot_z_optimize.py``, which is where it was proved.

Usage
  python lab/divergence/slot_state_linear_probe.py \
      --ckpt LABEL=CONFIG=PATH --rows 96 --batch 3 --depth 6 --out .../LABEL.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from core_anatomy import eff_rank  # noqa: E402
from slot_z_optimize import ZSplit, guard_split_point  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

try:  # sklearn is in the MORPH-TUL venv and NOT in the training venv. Either is fine.
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


# ── metrics ───────────────────────────────────────────────────────────────────


def roc_auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney U on ranks. Ties get mid-ranks, so a constant score reads exactly 0.5."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    # mid-ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def boot_ci(score: np.ndarray, y: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        a = roc_auc(score[idx], y[idx])
        if not np.isnan(a):
            vals.append(a)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def mean_ci(v: np.ndarray) -> tuple[float, float, float]:
    """Mean and a normal 95 % interval over slots. Returns (mean, lo, hi)."""
    if v.size == 0:
        return (float("nan"),) * 3
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else 0.0
    return (m, m - 1.96 * se, m + 1.96 * se)


# ── the linear probe ──────────────────────────────────────────────────────────


def _fit_torch(X: np.ndarray, y: np.ndarray, l2: float, device: str) -> np.ndarray:
    """Ridge-logistic by LBFGS. The fallback when sklearn is not importable.

    Deterministic (no sampling, a fixed number of LBFGS iterations from a zero start), so
    two runs of this script on the same states give the same AUC to the last digit.
    """
    xb = torch.as_tensor(X, dtype=torch.float32, device=device)
    yb = torch.as_tensor(y, dtype=torch.float32, device=device)
    w = torch.zeros(X.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, history_size=20,
                            tolerance_grad=1e-7, tolerance_change=1e-9,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad(set_to_none=True)
        z = xb @ w + b
        # sklearn minimises `0.5 w'w + C * sum_i logloss_i`; divided by n and with
        # C = 1 / l2 that is `mean logloss + 0.5 * l2 * ||w||^2 / n`. Written out so the
        # two probe backends optimise the SAME objective and their AUCs are comparable.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yb) \
            + 0.5 * l2 * w.pow(2).sum() / X.shape[0]
        loss.backward()
        return loss

    opt.step(closure)
    return torch.cat([w.detach(), b.detach()]).cpu().numpy()


def cv_auc(X: np.ndarray, y: np.ndarray, folds: int, seed: int, l2: float,
           device: str) -> tuple[float, np.ndarray, float]:
    """``(pooled AUC, per-fold-standardised OOF scores, mean of the per-fold AUCs)``.

    Standardisation is fitted on each TRAIN fold only. Fitting it on the whole set would
    leak the held-out rows' scale into the fold that scores them, which on 4096 features
    and a few thousand slots is not a rounding error.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    oof = np.zeros(n, dtype=np.float64)
    fold_aucs: list[float] = []
    for f in range(folds):
        te = perm[f::folds]
        tr = np.setdiff1d(perm, te, assume_unique=False)
        mu = X[tr].mean(0, keepdims=True)
        sd = X[tr].std(0, keepdims=True) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        if _HAVE_SKLEARN:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1.0 / max(l2, 1e-12), max_iter=2000)
            clf.fit(Xtr, y[tr])
            sc = clf.decision_function(Xte)
        else:
            wb = _fit_torch(Xtr, y[tr], l2, device)
            sc = Xte @ wb[:-1] + wb[-1]
        fold_aucs.append(roc_auc(sc, y[te]))
        # POOLING FIX, and it is load-bearing. Each fold fits its own intercept and its own
        # scale, so raw decision scores from different folds are not on one axis; pooling
        # them makes the AUC read the cross-fold offsets as signal. Measured on the first
        # CPU smoke of this script: the SHUFFLED-label control read 0.576 without this line
        # and must read 0.5. Standardising within the fold removes exactly the affine part
        # a fold is free to choose and leaves the ranking inside the fold untouched.
        sc = (sc - sc.mean()) / (sc.std() + 1e-12)
        oof[te] = sc
    return roc_auc(oof, y), oof, float(np.nanmean(fold_aucs))


# ── geometry ──────────────────────────────────────────────────────────────────


def hoyer(z: np.ndarray) -> np.ndarray:
    """Per-row Hoyer sparsity in [0, 1]. 0 = every coordinate equal, 1 = one coordinate."""
    d = z.shape[1]
    l1 = np.abs(z).sum(1)
    l2 = np.sqrt((z ** 2).sum(1)) + 1e-12
    return (np.sqrt(d) - l1 / l2) / (np.sqrt(d) - 1.0)


def geometry(Z: list[np.ndarray], y: np.ndarray) -> dict:
    """Per-pass step size, successive-step cosine, isotropy and sparsity, split by label."""
    out: dict = {}
    for t, zt in enumerate(Z):
        row: dict = {}
        nrm = np.sqrt((zt ** 2).sum(1))
        row["norm"] = mean_ci(nrm)
        row["hoyer"] = mean_ci(hoyer(zt))
        row["hoyer_y1"] = mean_ci(hoyer(zt)[y == 1])
        row["hoyer_y0"] = mean_ci(hoyer(zt)[y == 0])
        # Isotropy: entropy effective rank of the slot cloud, over the dimension. 1.0 is a
        # perfectly isotropic cloud; the slot states of this tree have read 1.7-4.8 in 1024.
        er = eff_rank(torch.as_tensor(zt, dtype=torch.float32).unsqueeze(0))
        row["eff_rank"] = float(er)
        row["isotropy"] = float(er) / zt.shape[1]
        if t + 1 < len(Z):
            step = Z[t + 1] - zt
            rel = np.sqrt((step ** 2).sum(1)) / (nrm + 1e-9)
            row["step_rel"] = mean_ci(rel)
            row["step_rel_y1"] = mean_ci(rel[y == 1])
            row["step_rel_y0"] = mean_ci(rel[y == 0])
            if t >= 1:
                prev = zt - Z[t - 1]
                num = (step * prev).sum(1)
                den = np.sqrt((step ** 2).sum(1)) * np.sqrt((prev ** 2).sum(1)) + 1e-12
                cos = num / den
                row["step_cos"] = mean_ci(cos)
                row["step_cos_y1"] = mean_ci(cos[y == 1])
                row["step_cos_y0"] = mean_ci(cos[y == 0])
        out[f"t{t}"] = row
    return out


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--shuffles", type=int, default=10,
                    help="label-shuffle repeats. The null BAND they trace is "
                         "the bar every pass AUC has to clear.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--override", action="append", default=[],
                    help="extra Hydra override, repeatable. The GPU run needs NONE; a CPU "
                         "smoke on a mask arm needs model.tg_scoped_kernels=false, because "
                         "the TG-scoped path is Triton and Triton needs a device.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)
    dev = a.device

    from morph.model.tul_egrad import slot_outcome_labels
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    cfg = build_cfg(config, ["model.use_kernels=false"] + list(a.override))
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_state_linear_probe needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path, dev, tul_rt.model_cfg)
    guard_split_point(model)
    tc = model.cfg.tul
    notes = {
        "slot_depth_fixed": [int(tc.slot_depth_fixed), f"forced 1..{a.depth}"],
        "slot_gain_lambda": [float(model.cfg.slot_gain_lambda), 0.0],
        "mode": "eval (dropout OFF, token-state dropout OFF) — an instrument on a fixed "
                "function",
        "z_t": "the tensor prefix_project would receive if the loop stopped at pass t; "
               "z_0 is core_init(e), the loop's ENTRY, off a forward hook",
        "label": "coda mean token CE over the slot's NEXT span, below the RUN's median "
                 "=> 1 (morph.model.tul_egrad.slot_outcome_labels — the same function "
                 "the `disc` energy arm trains its critic against)",
        "probe": ("sklearn LogisticRegression" if _HAVE_SKLEARN
                  else "torch LBFGS ridge-logistic (sklearn not importable here)"),
        "standardisation": "fitted on each TRAIN fold only",
        "baseline": "0.5 by construction (a median split), and the shuffle control",
        "overrides": ["model.use_kernels=false"] + list(a.override),
    }
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.eval()
    model.requires_grad_(False)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    feats: dict[str, list[np.ndarray]] = {"flat": [], "mean": []}
    per_pass: dict[int, dict[str, list[np.ndarray]]] = {}
    ys: list[np.ndarray] = []
    ces: list[np.ndarray] = []
    repro_err = 0.0
    n_rows = 0

    with torch.no_grad(), ZSplit(model) as split:
        for bi, (inp, labels, layout, _) in enumerate(batches):
            inp, labels = inp.to(dev), labels.to(dev)
            layout = layout.to(dev)
            n_rows += int(inp.shape[0])

            # -- the label and the split-point gate, at the model's own exit depth -----
            tc.slot_depth_fixed = a.depth
            split.st["core0"] = 0
            out, z_exit, h0 = split.record(inp, labels, layout, want_groups=True)
            if split.st["core0"] != a.depth:
                raise RuntimeError(f"core[0] ran {split.st['core0']} times, "
                                   f"expected {a.depth}")
            ce_loop = float(split.token_ce)
            xh = split.xh
            split.replay(inp, labels, layout, z_exit, want_groups=False)
            repro_err = max(repro_err, abs(float(split.token_ce) - ce_loop))
            y, scored, ce = slot_outcome_labels(
                xh.float(), labels, layout, model.embed.lm_weight().float(),
                int(model.cfg.ce_chunk_size))
            sc = scored.reshape(-1).cpu().numpy()
            ces.append(ce.reshape(-1).float().cpu().numpy()[sc])
            del out, xh

            # -- z at every pass ------------------------------------------------------
            states = {0: h0}
            for d in range(1, a.depth + 1):
                tc.slot_depth_fixed = d
                split.st["mode"] = "off"
                split.st["core0"] = 0
                o2, z_d, _h = split.record(inp, labels, layout, want_groups=False)
                if split.st["core0"] != d:
                    raise RuntimeError(f"core[0] ran {split.st['core0']} at depth {d}")
                states[d] = z_d
                del o2
            for t, zt in states.items():
                zf = zt.float()
                fl = zf.flatten(2).reshape(-1, zf.shape[2] * zf.shape[3]).cpu().numpy() \
                    if zf.dim() == 4 else zf.reshape(-1, zf.shape[-1]).cpu().numpy()
                mn = (zf.mean(dim=2) if zf.dim() == 4 else zf)
                mn = mn.reshape(-1, mn.shape[-1]).cpu().numpy()
                per_pass.setdefault(t, {"flat": [], "mean": []})
                per_pass[t]["flat"].append(fl[sc])
                per_pass[t]["mean"].append(mn[sc])
            ys.append(y.reshape(-1).float().cpu().numpy()[sc])
            print(f"  batch {bi + 1}/{len(batches)} done "
                  f"({int(sc.sum())} scored slots)", flush=True)
            torch.cuda.empty_cache() if dev == "cuda" else None

    if repro_err != 0.0:
        raise SystemExit(f"{label}: the recorded z does NOT reproduce the trained "
                         f"forward's token CE (|d| {repro_err:.3e}) — the split point is "
                         "wrong; do not report these numbers")

    y = np.concatenate(ys)
    ce_all = np.concatenate(ces)
    # The label is a MEDIAN split per batch inside `slot_outcome_labels`; re-threshold on
    # the RUN's median so every fold sees one definition, and report both.
    med_run = float(np.median(ce_all))
    y_run = (ce_all < med_run).astype(np.float64)
    Z = {red: {t: np.concatenate(per_pass[t][red]) for t in sorted(per_pass)}
         for red in ("flat", "mean")}

    rng = np.random.default_rng(a.seed)
    y_shuf = rng.permutation(y_run)

    res: dict = {}
    null: dict = {}
    for red in ("flat", "mean"):
        traj = None
        for t in sorted(Z[red]):
            X = Z[red][t]
            auc, oof, fa = cv_auc(X, y_run, a.folds, a.seed, a.l2, dev)
            lo, hi = boot_ci(oof, y_run, a.boot, a.seed)
            res[f"{red}/pass_t{t}"] = {"auc": auc, "ci": [lo, hi], "fold_mean": fa,
                                       "dim": int(X.shape[1])}
            traj = X if traj is None else np.concatenate([traj, X], axis=1)
            auc_c, oof_c, fa_c = cv_auc(traj, y_run, a.folds, a.seed, a.l2, dev)
            lo_c, hi_c = boot_ci(oof_c, y_run, a.boot, a.seed)
            res[f"{red}/traj_0_{t}"] = {"auc": auc_c, "ci": [lo_c, hi_c],
                                        "fold_mean": fa_c, "dim": int(traj.shape[1])}
        # Control 1, the gate: the SAME pipeline on shuffled labels at the EXIT, repeated,
        # so the result is a null BAND and not one draw. Every pass AUC above must clear
        # this band's upper end before it is called signal.
        t_last = max(Z[red])
        draws = []
        for k in range(a.shuffles):
            ys_k = np.random.default_rng(a.seed + 977 * k).permutation(y_run)
            draws.append(cv_auc(Z[red][t_last], ys_k, a.folds, a.seed + k, a.l2, dev)[0])
        null[red] = draws
        res[f"{red}/shuffle_exit"] = {
            "auc": float(np.mean(draws)), "n": a.shuffles,
            "band": [float(np.min(draws)), float(np.max(draws))],
            "p95": float(np.percentile(draws, 95)) if a.shuffles > 1 else float(draws[0])}

    geo = geometry([Z["mean"][t] for t in sorted(Z["mean"])], y_run)

    rec = {"label": label, "config": config, "step": step, "rows": n_rows,
           "batch": a.batch, "depth": a.depth, "folds": a.folds, "l2": a.l2,
           "boot": a.boot, "seed": a.seed, "n_slots": int(len(y_run)),
           "pos_frac": float(y_run.mean()), "median_ce": med_run,
           "pos_frac_per_batch_label": float(y.mean()),
           "sklearn": _HAVE_SKLEARN, "notes": notes,
           "repro_abs_err": repro_err, "auc": res, "geometry": geo,
           "shuffle_null_draws": null,
           "self_check_shuffle_ok": all(
               abs(res[f"{k}/shuffle_exit"]["auc"] - 0.5) <= 0.03 for k in ("flat", "mean")),
           }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    # ── printed table ─────────────────────────────────────────────────────────
    print(f"\n{label} step {step} — {n_rows} rows, {len(y_run)} scored slots, "
          f"{y_run.mean():.3f} positive, depth {a.depth}")
    print(f"  probe: {notes['probe']}; split point BIT-EXACT (|d| {repro_err:.1e})")
    print(f"  {'feature':10s}{'AUC flat [95% CI]':>28s}{'AUC mean [95% CI]':>28s}")
    for t in sorted(Z["mean"]):
        for kind in (f"pass_t{t}", f"traj_0_{t}"):
            f_, m_ = res[f"flat/{kind}"], res[f"mean/{kind}"]
            print(f"  {kind:10s}"
                  f"{f_['auc']:14.4f} [{f_['ci'][0]:.3f},{f_['ci'][1]:.3f}]"
                  f"{m_['auc']:14.4f} [{m_['ci'][0]:.3f},{m_['ci'][1]:.3f}]")
    f_, m_ = res["flat/shuffle_exit"], res["mean/shuffle_exit"]
    print(f"  {'SHUFFLE':10s}"
          f"{f_['auc']:14.4f} [{f_['band'][0]:.3f},{f_['band'][1]:.3f}]"
          f"{m_['auc']:14.4f} [{m_['band'][0]:.3f},{m_['band'][1]:.3f}]"
          f"   <- null BAND over {a.shuffles} shuffles; p95 "
          f"{f_['p95']:.3f} / {m_['p95']:.3f}")
    bad = [k for k in ("flat", "mean") if abs(res[f"{k}/shuffle_exit"]["auc"] - 0.5) > 0.03]
    if bad:
        print(f"  !! SELF-CHECK FAILED: the shuffled-label control reads "
              f"{[round(res[k + '/shuffle_exit']['auc'], 3) for k in bad]} on {bad}, not "
              f"~0.50. The AUCs above are NOT interpretable — raise --rows or --l2.")
    print("  geometry (stream mean): step_rel = ||z_{t+1}-z_t||/||z_t||, "
          "step_cos = cos(step_t, step_{t-1})")
    print(f"  {'pass':6s}{'|z|':>10s}{'step_rel':>12s}{'step_cos':>12s}"
          f"{'eff_rank':>10s}{'isotropy':>10s}{'hoyer':>9s}"
          f"{'step_rel y1':>13s}{'step_rel y0':>13s}")
    for t in sorted(Z["mean"]):
        g = geo[f"t{t}"]
        sr = g.get("step_rel", (float("nan"),) * 3)
        sc_ = g.get("step_cos", (float("nan"),) * 3)
        s1 = g.get("step_rel_y1", (float("nan"),) * 3)
        s0 = g.get("step_rel_y0", (float("nan"),) * 3)
        print(f"  t{t:<5d}{g['norm'][0]:10.2f}{sr[0]:12.4f}{sc_[0]:12.4f}"
              f"{g['eff_rank']:10.2f}{g['isotropy']:10.4f}{g['hoyer'][0]:9.4f}"
              f"{s1[0]:13.4f}{s0[0]:13.4f}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
