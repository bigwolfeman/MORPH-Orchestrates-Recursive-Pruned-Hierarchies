"""Which objective writes to the core, and do they fight?

THE GAP THIS CLOSES. Every gradient number in this campaign — the 90 % core
share, "the direct route is already ~50 % of the gradient", every takeover
verdict — comes from `train.py::_preclip_probe`, which reads `p.grad` AFTER the
backward of the SUMMED objective. One backward of `L = L_main + L_plast + L_emit`
leaves `g_main + g_plast + g_emit` in one tensor, and that sum is IDENTICAL
whether the terms are aligned, orthogonal, or cancelling. The probe we have
cannot tell conflict from domination from starvation. This one can.

WHAT IT MEASURES. One forward+backward per objective on the same batch under the
same RNG, then per region: each objective's gradient norm, and the cosine
between every pair. The literature splits on exactly that cosine:

    cos < 0                     CONFLICT     -> projection (PCGrad, CAGrad)
    cos ~ 0, one norm tiny      DOMINATION   -> rescaling (GradNorm, MGDA)
    cos > 0, one norm tiny      STARVATION   -> decoupling, or delete the shortcut

TWO SELF-TESTS, both of which FAIL LOUDLY rather than warn.

1. ADDITIVITY. `sum_g (W_g/W) * g_g` must equal the gradient of the full CE to
   `--tol`. The reduction in `fused_linear_cross_entropy` is `Σ w·CE / Σ w`, so a
   group-only pass carries a DIFFERENT denominator; each group is rescaled by
   `W_g/W` before the backward, and this test is what proves the rescale right.
   If it fails, every cosine below is meaningless and the probe exits non-zero.

2. DETERMINISM. The same objective is run twice and the cosine between the two
   runs must be >= `--det-tol`. This is not paranoia: MORPH's bag-mean scatter
   uses atomics, and a measured 4 % per-step gradient error from that source is
   already on record. A cosine of 0.05 between two objectives means nothing if
   one objective against ITSELF only reaches 0.9.

Usage:
    PYTHONPATH=$PWD python lab/divergence/objective_split.py \
        --ckpt checkpoints/morph/ctrlworth-s2/step_3000.pt --config tul_a1 \
        --batches 4 --out split.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402

CE_GROUPS = ("main", "plast", "emit")
_FP32 = [False]      # set from --fp32 before any pass; see `grads_of`


def region_of(name: str) -> str:
    return name.replace("_orig_mod.", "").split(".")[0]


@contextlib.contextmanager
def ce_group(root, group: str, unit: bool):
    """Restrict the CE reduction to one label group.

    `unit` forces weight 1.0 even where the config sets 0. The cosine is
    scale-invariant so the DIRECTION is still this arm's emit direction; it just
    is not part of this arm's objective, which the report states.
    """
    if group == "full":
        yield
        return
    orig = root._tul_half_weights
    pw = float(root.cfg.tul.plast_weight)
    ew = float(root.cfg.tul.emit_weight)
    if unit:
        pw = pw or 1.0
        ew = ew or 1.0

    def patched(labels, layout):
        w, p_idx, z_idx = orig(labels, layout)
        bl = w.numel()
        full = torch.zeros(bl + 1, dtype=w.dtype, device=w.device)
        if group == "main":
            full[:bl] = 1.0
            full[p_idx] = 0.0
            full[z_idx] = 0.0
        elif group == "plast":
            full[p_idx] = pw
        elif group == "emit":
            full[z_idx] = ew
        else:
            raise ValueError(group)
        return full[:bl], p_idx, z_idx

    root._tul_half_weights = patched
    try:
        yield
    finally:
        root._tul_half_weights = orig


def grads_of(model, x, y, layout, group: str, seed: int, *, unit: bool,
             mux_on: bool, scale: float | None) -> tuple[dict[str, torch.Tensor], float, float]:
    """`({region: flat fp32 grad on CPU}, loss, n_targets)` for one objective.

    `scale` rescales the loss to its additive share of the full objective; None
    means "do not rescale", used for the direction-only unit-weight passes.
    """
    root = getattr(model, "_orig_mod", model)
    model.zero_grad(set_to_none=True)
    root.mux_gate.fill_(1.0 if mux_on else 0.0)
    root.sigreg_gate.fill_(1.0 if mux_on else 0.0)
    # Same depth draw and the same token-dropout mask as every other pass, or the
    # objectives are compared on two different forwards and the cosine is noise.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ctx = (contextlib.nullcontext() if _FP32[0]
           else torch.autocast("cuda", dtype=torch.bfloat16))
    with ce_group(root, group, unit), ctx:
        out = model(x, labels=y, slot_layout=layout)
    loss, n_t = out["loss"], float(out["n_targets"])
    (loss * (scale if scale is not None else 1.0)).backward()
    acc: dict[str, list[torch.Tensor]] = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            acc.setdefault(region_of(name), []).append(p.grad.detach().float().flatten().cpu())
    model.zero_grad(set_to_none=True)
    return {k: torch.cat(v) for k, v in acc.items()}, float(loss), n_t


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine, accumulated in float64.

    NOT cosmetic. The first live run reported self-cosines of 1.0156 — a cosine
    cannot exceed 1, so that was pure accumulation error: these vectors hold tens
    of millions of fp32 entries whose magnitudes reach 1e3, and a plain fp32 dot
    over 6e7 terms loses more than the 1e-2 differences this probe is trying to
    resolve. Every reduction here is float64 for the same reason.
    """
    a64, b64 = a.double(), b.double()
    na, nb = a64.norm(), b64.norm()
    if na == 0 or nb == 0:
        return float("nan")
    return float(torch.dot(a64, b64) / (na * nb))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=2e-2,
                    help="additivity: max relative error of the summed decomposition")
    ap.add_argument("--det-tol", type=float, default=0.99,
                    help="determinism: min cosine of one objective against itself")
    ap.add_argument("--region", default="core", help="region the verdict is read on")
    ap.add_argument("--fp32", action="store_true",
                    help="run the forward in fp32 instead of bf16 autocast. The DECOMPOSITION "
                         "is exact in real arithmetic, so any additivity error is rounding; "
                         "bf16 carries ~3 decimal digits and its rounding is measurement "
                         "noise, not the training signal this probe is after.")
    ap.add_argument("--out", default="")
    ap.add_argument("--sabotage", default="none", choices=["none", "scale", "seed"],
                    help="deliberately break one gate. Used by tests/test_objective_split.py "
                         "to prove the gates BITE: a gate that never fails is not a gate. "
                         "'scale' drops the W_g/W rescale so additivity must fail; 'seed' "
                         "reseeds between the two self-runs so determinism must fail.")
    a = ap.parse_args()

    _FP32[0] = a.fp32
    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    load_checkpoint(a.ckpt, model, torch.amp.GradScaler("cuda", enabled=False),
                    torch.device("cuda"))
    # TRAIN mode: the Poisson depth draw and token-state dropout are part of the
    # gradient we are decomposing. An eval-mode gradient is a different object.
    model.train()

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    pw, ew = float(root.cfg.tul.plast_weight), float(root.cfg.tul.emit_weight)
    mux_b = float(getattr(root.cfg.tul, "mux_beta", 0.0))
    print(f"  {a.ckpt}\n  config={a.config}  plast_weight={pw}  emit_weight={ew}  "
          f"mux_beta={mux_b}  batches={len(batches)}\n")

    tot: dict[str, dict[str, torch.Tensor]] = {}
    per_batch_cos: list[dict[str, float]] = []
    det_cos: list[float] = []
    add_err: list[float] = []

    for bi, (x, y, layout) in enumerate(batches):
        sd = a.seed + bi
        # W: the full CE's weight mass, the denominator every group is rescaled to.
        g_cefull, l_cefull, w_full = grads_of(model, x, y, layout, "full", sd,
                                              unit=False, mux_on=False, scale=1.0)
        g_full, _, _ = grads_of(model, x, y, layout, "full", sd,
                                unit=False, mux_on=True, scale=1.0)
        # Determinism: the SAME objective twice. Anything below --det-tol makes
        # every cross-objective cosine unreadable.
        g_cefull2, _, _ = grads_of(model, x, y, layout, "full",
                                   sd + 991 if a.sabotage == "seed" else sd,
                                   unit=False, mux_on=False, scale=1.0)
        det_cos.append(cos(g_cefull[a.region], g_cefull2[a.region]))

        parts: dict[str, dict[str, torch.Tensor]] = {}
        summed: dict[str, torch.Tensor] = {}
        for grp in CE_GROUPS:
            cw = 1.0 if grp == "main" else (pw if grp == "plast" else ew)
            # Configured-weight pass, for the additivity test and the true norm.
            if cw > 0.0:
                _, _, w_g = grads_of(model, x, y, layout, grp, sd,
                                     unit=False, mux_on=False, scale=None)
                gg, _, _ = grads_of(model, x, y, layout, grp, sd, unit=False,
                                    mux_on=False,
                                    scale=1.0 if a.sabotage == "scale"
                                    else w_g / max(w_full, 1e-12))
                for r, v in gg.items():
                    summed[r] = summed.get(r, torch.zeros_like(v)) + v
                parts[grp] = gg
            else:
                # Weight 0 in this arm: it contributes NOTHING to the objective, so
                # it is excluded from the additivity sum. Its DIRECTION is still
                # measured, at unit weight, and the report says so.
                gg, _, _ = grads_of(model, x, y, layout, grp, sd,
                                    unit=True, mux_on=False, scale=None)
                parts[grp + "*"] = gg
        if mux_b > 0.0:
            parts["mux"] = {r: g_full[r] - g_cefull[r] for r in g_full}

        ref = g_cefull[a.region].double()
        err = float((summed[a.region].double() - ref).norm() / ref.norm().clamp(min=1e-30))
        add_err.append(err)

        row = {}
        keys = list(parts)
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                row[f"{k1}~{k2}"] = cos(parts[k1][a.region], parts[k2][a.region])
        per_batch_cos.append(row)
        for k, gg in parts.items():
            for r, v in gg.items():
                tot.setdefault(k, {})
                tot[k][r] = tot[k].get(r, torch.zeros_like(v)) + v
        print(f"  batch {bi}: additivity rel-err {err:.2e}  determinism cos "
              f"{det_cos[-1]:.6f}")

    # ── the two gates ────────────────────────────────────────────────────────
    worst_add, worst_det = max(add_err), min(det_cos)
    print(f"\nGATE additivity   worst rel-err {worst_add:.2e} (tol {a.tol:.0e})   "
          f"{'PASS' if worst_add <= a.tol else 'FAIL'}")
    print(f"GATE determinism  worst self-cos {worst_det:.6f} (tol {a.det_tol})   "
          f"{'PASS' if worst_det >= a.det_tol else 'FAIL'}")
    if worst_add > a.tol or worst_det < a.det_tol:
        print("\nREFUSING to report cosines: a decomposition that does not sum to the "
              "objective, or an objective that does not reproduce against itself, "
              "cannot distinguish conflict from noise.")
        return 1

    print(f"\nregion = {a.region}   (* = weight 0 in this arm; direction only, "
          f"not part of its objective)")
    print(f"\n{'objective':<12} {'||g||':>12} {'share of ||g_full||':>20}")
    full_n = float(sum(tot[k][a.region] for k in tot if not k.endswith("*")).norm())
    for k in tot:
        n = float(tot[k][a.region].norm())
        print(f"{k:<12} {n:>12.4e} {(n / full_n if full_n else float('nan')):>20.4f}")

    print(f"\ncosines on {a.region} (accumulated over {len(batches)} batches; "
          f"per-batch spread in brackets)")
    keys = list(tot)
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            c = cos(tot[k1][a.region], tot[k2][a.region])
            vals = [r[f"{k1}~{k2}"] for r in per_batch_cos if f"{k1}~{k2}" in r]
            lo, hi = (min(vals), max(vals)) if vals else (float("nan"),) * 2
            verdict = ("CONFLICT" if c < -0.05 else
                       "orthogonal" if abs(c) <= 0.05 else "aligned")
            print(f"  {k1:>8} ~ {k2:<8} {c:>8.4f}  [{lo:+.3f},{hi:+.3f}]  {verdict}")

    if a.out:
        rec = {"ckpt": a.ckpt, "config": a.config, "region": a.region,
               "plast_weight": pw, "emit_weight": ew, "mux_beta": mux_b,
               "additivity_rel_err": add_err, "determinism_self_cos": det_cos,
               "norms": {k: float(tot[k][a.region].norm()) for k in tot},
               "norms_all_regions": {k: {r: float(v.norm()) for r, v in tot[k].items()}
                                     for k in tot},
               "cos": {f"{k1}~{k2}": cos(tot[k1][a.region], tot[k2][a.region])
                       for i, k1 in enumerate(keys) for k2 in keys[i + 1:]},
               "cos_per_batch": per_batch_cos}
        with open(a.out, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
