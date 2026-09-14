"""Does a row's per-pass Jacobian's spectral GAP predict its own loop earning?

Wu, Zhang, Cao, "Looped Transformers with Layer Normalization Provably Learn the Power
Method" (arXiv 2606.00605), Thm 4.6-4.7 (docs/references/looping-depth/2026-09-13-lit-
mining/A_theory.md, paper 3): the number of loop iterations needed to reach a target
error is set by the ratio of the top-two eigenvalues of the per-pass linear(ized) map — a
wide gap (sigma1 >> sigma2) converges in one or two passes and every later pass only
ROTATES an already-tiny residual; a narrow gap needs many. Their concrete arm for us: "on
the per-pass core Jacobian already computed in prior audits, measure sigma_1/sigma_2 per
row and correlate against that row's K1-K6".

Per ROW (batch=1, so the active-set mask isolates exactly one row):

1. sigma1, sigma2 — the top two singular values of the pass map at iteration 0
   (``d(one core step)/d(h)`` at the live operating point, restricted to the row's own
   active positions: slot positions for the slot loop, every token position for the plain
   core), via ``_diag_common.jacobian_top2`` — the double-backward power-iteration trick
   of ``morph/training/core_jacobian.py``, extended from one vector to a 2-vector
   deflated block.
2. K1-K6 — CE(forced depth 1) - CE(forced depth 6) on the row's own token positions
   (``_diag_common.row_token_ce``), the SAME per-row K-curve unit ``core_depth_sweep.py``
   pools into its paired bootstrap.

Reports the Spearman correlation of sigma1 (and of sigma1/sigma2) against K1-K6 over the
scored rows, plus the sigma-ratio distribution.

Usage:
  python lab/divergence/jacobian_gap_vs_k.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --rows 96 --device cuda --out results/jac_strict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diag_common import Arm, jacobian_top2, row_token_ce, spearman_rho  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "morph", "training"))
from core_jacobian import CoreJacobianProbe  # noqa: E402


def _capture_iter0(model, forward_fn) -> dict:
    """Run ``forward_fn()`` (no_grad) and return the single ``iter_idx == 0`` capture
    point. Raises if the loop never reached iteration 0 (a pad-only row) or captured more
    than one such point (a caller bug, not this row)."""
    probe = CoreJacobianProbe(model, n_iter=1)  # n_iter unused here; only .capture() matters
    with probe.capture() as points:
        with torch.no_grad():
            forward_fn()
    zero = [p for p in points if int(p["iter_idx"]) == 0]
    if len(zero) != 1:
        raise RuntimeError(f"expected exactly one iter_idx==0 capture, got {len(zero)}")
    return zero[0], probe.root


def _sigma_at_point(root, point, n_iter: int, seed: int) -> tuple[float, float]:
    h0 = point["h"]
    e = point["e"].float()
    inj = point["inj"].float()
    ret = point["ret_state"]
    t = int(point["iter_idx"])
    mask = point["active"].view(*point["active"].shape,
                                *([1] * (h0.dim() - 2))).to(torch.float32)

    def step(h):
        out, _ = root._apply_core_step(h, e, None, None, None, ret_state=ret,
                                       iter_idx=t, inj_terms=inj)
        return out

    with torch.autocast("cuda", enabled=False):
        return jacobian_top2(step, h0, mask, n_iter=n_iter, seed=seed)


def run(label: str, config: str, path: str, device: str, rows: int, n_iter: int) -> dict:
    t0 = time.time()
    arm = Arm(label, config, path, device, rows, row_batch=1)
    model = arm.model
    orig_mean, orig_max, orig_fixed = arm.orig_depth()

    sigma1s: list[float] = []
    sigma2s: list[float] = []
    k1_minus_k6: list[float] = []
    n_scored = 0
    n_skipped = 0
    for i, (inp, labels, layout) in enumerate(arm.rows()):
        try:
            # ── the row's own K1-K6 (forced depth 1 vs 6) ────────────────────────
            arm.set_depth(1)
            s1_sum, s1_cnt = row_token_ce(model, inp, labels, layout, device)
            arm.set_depth(6)
            s6_sum, s6_cnt = row_token_ce(model, inp, labels, layout, device)
            arm.restore_depth(orig_mean, orig_max, orig_fixed)
            if s1_cnt == 0 or s6_cnt == 0:
                n_skipped += 1
                continue
            k = s1_sum / s1_cnt - s6_sum / s6_cnt

            # ── the pass Jacobian at iteration 0, the model's OWN eval depth ─────
            def fwd():
                if arm.is_slot_loop:
                    fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
                    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw,
                                                     ret_reset_mask=freset)
                    model._tul_core(x, x0, bigram, layout, input_ids=inp)
                else:
                    x, x0, bigram = model._front_region(inp)
                    model._core_region(x, x0, bigram)

            point, root = _capture_iter0(model, fwd)
            sigma1, sigma2 = _sigma_at_point(root, point, n_iter, seed=i)
        finally:
            arm.restore_depth(orig_mean, orig_max, orig_fixed)
            model._jac_capture = None

        sigma1s.append(sigma1)
        sigma2s.append(sigma2)
        k1_minus_k6.append(k)
        n_scored += 1
        if i % 10 == 0:
            print(f"{label} row {i} sigma1={sigma1:.4f} sigma2={sigma2:.4f} "
                  f"ratio={sigma1/max(sigma2,1e-12):.3f} K1-K6={k:.4f}", flush=True)

    ratios = [s1 / max(s2, 1e-12) for s1, s2 in zip(sigma1s, sigma2s)]
    rho_sigma1, n1 = spearman_rho(sigma1s, k1_minus_k6)
    rho_ratio, n2 = spearman_rho(ratios, k1_minus_k6)

    def quantiles(v):
        if not v:
            return {}
        t = torch.tensor(v, dtype=torch.float64)
        return {"median": float(t.median()), "q25": float(t.quantile(0.25)),
                "q75": float(t.quantile(0.75)), "mean": float(t.mean())}

    out = {
        "label": label, "config": config, "ckpt": path, "step": arm.step,
        "is_slot_loop": arm.is_slot_loop, "rows_scored": n_scored, "rows_skipped": n_skipped,
        "n_power_iter": n_iter, "wall_s": time.time() - t0,
        "sigma1": quantiles(sigma1s), "sigma2": quantiles(sigma2s),
        "sigma_ratio": quantiles(ratios),
        "spearman_sigma1_vs_K1minusK6": {"rho": rho_sigma1, "n": n1},
        "spearman_ratio_vs_K1minusK6": {"rho": rho_ratio, "n": n2},
        "k1_minus_k6": quantiles(k1_minus_k6),
        "raw": {"sigma1": sigma1s, "sigma2": sigma2s, "k1_minus_k6": k1_minus_k6},
    }
    print(f"{label}: sigma1/sigma2 median={out['sigma_ratio'].get('median', float('nan')):.3f} "
          f"spearman(sigma1, K1-K6)={rho_sigma1:.3f} spearman(ratio, K1-K6)={rho_ratio:.3f} "
          f"n={n_scored}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--n-power-iter", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        out[label] = run(label, config, path, a.device, a.rows, a.n_power_iter)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        for label, e in out.items():
            f.write(f"{label}: config={e['config']} step={e['step']} slot_loop={e['is_slot_loop']} "
                    f"rows_scored={e['rows_scored']} rows_skipped={e['rows_skipped']}\n")
            f.write(f"  sigma1 median={e['sigma1'].get('median', float('nan')):.4f} "
                    f"sigma2 median={e['sigma2'].get('median', float('nan')):.4f} "
                    f"ratio median={e['sigma_ratio'].get('median', float('nan')):.4f}\n")
            f.write(f"  spearman(sigma1, K1-K6) rho={e['spearman_sigma1_vs_K1minusK6']['rho']:.4f} "
                    f"n={e['spearman_sigma1_vs_K1minusK6']['n']}\n")
            f.write(f"  spearman(ratio, K1-K6)  rho={e['spearman_ratio_vs_K1minusK6']['rho']:.4f} "
                    f"n={e['spearman_ratio_vs_K1minusK6']['n']}\n")
    print("wrote", a.out, "and", txt, flush=True)


if __name__ == "__main__":
    main()
