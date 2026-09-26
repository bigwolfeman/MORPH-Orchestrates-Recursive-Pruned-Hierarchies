"""The amplitude-matched K-curve of an LXTUL-E arm: is coda K1-K6 the loop computing, or
the enumerated code growing louder with depth?

Protocol: ``/home/wolfe/morph-scratch/tulv2/opus.md`` Section C "The single measurement to
run first" (Steps 0-2), with the calibration split of gpt.md Section C and the same
control fable.md Section C asks for. Prereg: ``lab/experiments/planned/*-lx-amplitude-
matched-k-curve.md``.

WHY. LXTUL-E (``morph/model/tul_code_enum.py``) adds ``r * rms(f(h)) * u_k`` at the end of
EVERY slot-loop pass. On the 704 channels outside ``DiagonalInjection``'s ctx slice the
carry is the identity, so a code re-added T times sums about T times. A depth-1 exit then
differs from a depth-6 exit by the loop's work AND by a quieter code, and the coda's
K1-K6 cannot tell the two apart. This instrument gives the depth-1 (2, 3) exit a louder
code, matched to depth 6's rollout separation, and asks how much K1-K6 is left.

TERMS (one meaning each; the Stage 1 terms carry over, ``lxtul_e_stage1_score.py``):

  depth d        forced slot-loop depth of every valid slot (``slot_depths`` table).
  a              the code multiplier: the model's ``tul_code_enum.ratio`` is set to
                 ``a * r`` (r = ``tul.code_enum_ratio``, 0.1 here) for the forward, eval
                 only, and restored after. a = 1 is the trained model, bit for bit.
  exit sep       ``sep_abs`` of the Stage 1 scorer: the mean over valid slots and rollout
                 pairs of the RMS distance between the K rollouts' exit states.
  code sep       ``sep_code``: exit sep restricted to span(u_k) (gpt.md's code-direction
                 component). Reported, never fitted.
  a_d            the multiplier at which exit sep at (d, a_d) equals exit sep at (6, 1),
                 found by bisection on log a over the CALIBRATION rows (the packed rows
                 right after the scored ones, disjoint from them), within ``--tol-rel``.
  coda CE        the per-token coda CE under the per-span Bayes read over the K rollouts
                 (``_enum_position_nll``), the scored rows (the 480 rows ``core_depth_
                 sweep.py`` and the Stage 2 scorer read, token-paired by stream index).
  K{d}-K6        CE(d, 1) - CE(6, 1): the unmatched reading.
  K{d}*-K6       CE(d, a_d) - CE(6, 1): what is left once the code is as loud.
  reverse_d      CE(6, 1/a_d) - CE(d, 1): depth 6 with its code quieted by the same
                 factor, against depth d (Step 2; covers a loud code on a d-pass state
                 being off-distribution for the coda).
  survive_d      (K{d}*-K6) / (K{d}-K6), a ratio with a joint block-bootstrap CI.
  code share     Step 0: the fraction of u_k's energy on the injected ctx slice
                 ``injection.start:end``, trained vs its random init (seed 0xE4C0DE) and
                 vs a 2,000-draw Gaussian null.

STATISTICS. Token-weighted means; paired block bootstrap over stream blocks of 1,024
(``sweep_score.BLOCK``), 2,000 resamples, seed 0 (``lxtul_e_stage1_score._ci``).

SELF-CHECKS. The Stage 1 scorer's per-batch check (offline coda CE vs the forward's own
``ce_tokens``, RAISES above ``--tol``). With ``--ref-npz`` the a = 1 points at depths 1 and
6 are compared token by token with the stored Stage 2 arrays of the same checkpoint
(``coda_idx`` must match exactly; RAISES when the mean CE differs by more than
``--ref-tol``).

Usage (GPU; Step 0 alone runs on CPU with ``--step0-only --device cpu``):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python lab/divergence/lx_amp_matched.py \\
      --arm fp01_5k=tul_slot_spandec_strict_e4probe_fp01=/home/wolfe/morph-to/checkpoints/morph/lxtul-e4probe-fp01/step_5000.pt \\
      --ref-npz /home/wolfe/morph-scratch/lxtul-10k/score.tokens.npz --ref-label fp01_5k \\
      --rows 480 --calib-rows 32 --out .../amp_fp01_5k.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import parse_ckpt_spec  # noqa: E402
from _earning import BINS, offsets_from_layout  # noqa: E402
from lxtul_e_stage1_score import (  # noqa: E402  (ONE copy of the Stage 1 instrument)
    N_BOOT, _blocks, _ci, _fmt, abs_path, load_arm, score_arm, val_batches)

from morph.model.tul_code_enum import TULCodeEnum, gram_schmidt_rows  # noqa: E402


@contextlib.contextmanager
def code_ratio_scale(m, a: float):
    """Run the block with the code term at ``a`` times the trained ratio; restore after.
    ``TULCodeEnum.term`` reads ``self.ratio`` on every call, and nothing else in the
    forward reads it."""
    enum = getattr(m, "tul_code_enum", None)
    if enum is None:
        raise RuntimeError("no tul_code_enum: not an LXTUL-E (code_enum_k > 1) model")
    if not a > 0.0:
        raise ValueError(f"code multiplier must be > 0, got {a}")
    base = enum.ratio
    enum.ratio = base * float(a)
    try:
        yield
    finally:
        enum.ratio = base


def match_multiplier(f, target: float, lo: float = 0.25, hi: float = 16.0,
                     tol_rel: float = 0.005, max_iter: int = 30, limit: float = 256.0):
    """``(a, trace)``: the ``a`` with ``f(a)`` within ``tol_rel`` of ``target`` for an
    INCREASING ``f``, by bisection on ``log a``. The bracket widens by factors of 2 up to
    ``[1/limit, limit]`` first; an unbracketable target RAISES. ``trace`` is every
    ``(a, f(a))`` evaluated, in order. Returns the best evaluated point when ``max_iter``
    runs out (its relative miss is in the trace)."""
    trace: list[tuple[float, float]] = []

    def ev(a: float) -> float:
        v = float(f(a))
        trace.append((a, v))
        return v

    flo, fhi = ev(lo), ev(hi)
    while flo > target:
        if lo <= 1.0 / limit:
            raise RuntimeError(f"f({lo}) = {flo} > target {target}: cannot bracket")
        hi, fhi = lo, flo
        lo /= 2.0
        flo = ev(lo)
    while fhi < target:
        if hi >= limit:
            raise RuntimeError(f"f({hi}) = {fhi} < target {target}: cannot bracket")
        lo, flo = hi, fhi
        hi *= 2.0
        fhi = ev(hi)
    for _ in range(max_iter):
        best = min(trace, key=lambda t: abs(t[1] - target))
        if abs(best[1] - target) <= tol_rel * abs(target):
            return best[0], trace
        mid = math.sqrt(lo * hi)
        fm = ev(mid)
        if fm < target:
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    best = min(trace, key=lambda t: abs(t[1] - target))
    return best[0], trace


def code_slice_energy(m, n_null: int = 2000, seed: int = 0) -> dict:
    """Step 0: the share of each code u_k's energy (and of the code subspace's) on the
    injected ctx slice, trained vs random init vs a Gaussian null."""
    enum, inj = m.tul_code_enum, m.injection
    if type(inj).__name__ != "DiagonalInjection":
        raise RuntimeError(f"model.injection is {type(inj).__name__}, not DiagonalInjection")
    s, e = int(inj.start), int(inj.end)
    C, K = int(enum.d_model), int(enum.k)
    if e - s >= C:
        raise RuntimeError(f"injection slice {s}:{e} covers every channel (injection_channels"
                           f"='all'?): the protocol's ctx-slice split does not exist")

    def stats(basis: torch.Tensor) -> tuple[list[float], float]:
        q = gram_schmidt_rows(basis.detach().double())
        u = (enum.simplex.detach().double().to(q.device) @ q) * math.sqrt(C)
        fk = ((u[:, s:e] ** 2).sum(1) / (u ** 2).sum(1)).tolist()
        fsub = float((q[:, s:e] ** 2).sum() / (K - 1))
        return fk, fsub

    tr_k, tr_sub = stats(enum.basis)
    init = TULCodeEnum(C, K, enum.ratio)
    in_k, in_sub = stats(init.basis)
    g = torch.Generator().manual_seed(seed)
    nk, ns = [], []
    for _ in range(n_null):
        fk, fsub = stats(torch.randn(K - 1, C, generator=g, dtype=torch.float64))
        nk.extend(fk)
        ns.append(fsub)
    A = inj.log_A.detach().double().exp().clamp(max=0.9999)
    return {"slice": [s, e], "d_model": C, "K": K, "expected_share": (e - s) / C,
            "trained_u_share": tr_k, "trained_subspace_share": tr_sub,
            "init_u_share": in_k, "init_subspace_share": in_sub,
            "null_u_share_q025_q975": [float(np.quantile(nk, 0.025)),
                                       float(np.quantile(nk, 0.975))],
            "null_subspace_share_q025_q975": [float(np.quantile(ns, 0.025)),
                                              float(np.quantile(ns, 0.975))],
            "injection_A_mean_min_max": [float(A.mean()), float(A.min()), float(A.max())],
            "code_ratio": float(enum.ratio)}


def ratio_ci(na: np.ndarray, nb_: np.ndarray, da: np.ndarray, db: np.ndarray,
             blocks: np.ndarray, n_boot: int, seed: int) -> dict:
    """``(mean(na) - mean(nb_)) / (mean(da) - mean(db))`` over the same tokens, with a
    joint block bootstrap (the resampling of ``_stats.paired_bootstrap_ci``)."""
    nbk = int(blocks.max()) + 1
    sums = [np.bincount(blocks, weights=x.astype(np.float64), minlength=nbk)
            for x in (na, nb_, da, db)]
    c = np.bincount(blocks, minlength=nbk).astype(np.float64)
    keep = c > 0
    sums = [x[keep] for x in sums]
    c = c[keep]

    def r(ix=None):
        s = [x.sum() if ix is None else x[ix].sum(1) for x in sums]
        n = c.sum() if ix is None else c[ix].sum(1)
        return (s[0] / n - s[1] / n) / (s[2] / n - s[3] / n)

    rng = np.random.default_rng(seed)
    ix = rng.integers(0, c.shape[0], size=(n_boot, c.shape[0]))
    boot = r(ix)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"point": float(r()), "lo": float(lo), "hi": float(hi), "n_units": int(c.shape[0]),
            "n_boot": int(n_boot)}


def run_point(m, batches, d: int, a: float, device: str, tol: float) -> dict:
    """One (depth, multiplier) point: per-token coda CE, exit / code separation."""
    with code_ratio_scale(m, a):
        p = score_arm(m, batches, [d], device, tol)["per"][d]
    return {"coda_idx": p["coda_idx"], "coda_mix": p["coda_mix"].astype(np.float64),
            "sep_abs": float(p["sep_abs"]), "sep_rel": float(p["sep_rel"]),
            "sep_code": float(p["sep_code"])}


def token_offsets(batches, coda_idx: np.ndarray) -> np.ndarray:
    """Offset-in-span (``_earning``) of every coda token, joined by stream index."""
    top = int(coda_idx.max()) + 1
    off = np.full(top, -2, dtype=np.int64)
    for _inp, _lab, layout, idx in batches:
        for b in range(idx.shape[0]):
            o = offsets_from_layout(layout, b)
            ix = idx[b].numpy()
            keep = (ix >= 0) & (ix < top)
            off[ix[keep]] = o[keep]
    got = off[coda_idx]
    if (got == -2).any():
        raise RuntimeError("a coda token has no offset: the stream map is broken")
    return got


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True, help="LABEL=CONFIG=PATH[=ovr,...]")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--calib-rows", type=int, default=32,
                    help="rounded up to whole batches; packed right after the scored rows")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--match-depths", default="1,2,3")
    ap.add_argument("--ref-depth", type=int, default=6)
    ap.add_argument("--tol-rel", type=float, default=0.005)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=2e-3)
    ap.add_argument("--ref-npz", default=None)
    ap.add_argument("--ref-label", default=None)
    ap.add_argument("--ref-tol", type=float, default=5e-4)
    ap.add_argument("--step0-only", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ovr", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    label, config, path, ovr = parse_ckpt_spec(a.arm)
    m, step, cfg, rt = load_arm(config, path, a.device, [*a.ovr, *ovr])
    if not getattr(m, "_code_enum_k", 0):
        raise SystemExit(f"{label}: not an LXTUL-E arm (tul.code_enum_k > 1)")
    res: dict = {"label": label, "config": config, "path": abs_path(path), "step": step,
                 "overrides": ovr, "step0": code_slice_energy(m)}
    s0 = res["step0"]
    print(f"[{label}] step {step}  Step 0: slice {s0['slice']} (expected share "
          f"{s0['expected_share']:.4f})  trained u share "
          f"{[round(x, 4) for x in s0['trained_u_share']]} subspace "
          f"{s0['trained_subspace_share']:.4f}  init "
          f"{[round(x, 4) for x in s0['init_u_share']]} subspace "
          f"{s0['init_subspace_share']:.4f}  null u 95% {s0['null_u_share_q025_q975']}  "
          f"A {s0['injection_A_mean_min_max']}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.step0_only:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"wrote {a.out}")
        return

    n_sc = -(-a.rows // a.batch)
    n_cal = -(-a.calib_rows // a.batch)
    batches, _got = val_batches(cfg, rt, (n_sc + n_cal) * a.batch, a.batch)
    scored, calib = batches[:n_sc], batches[n_sc:n_sc + n_cal]
    rows = sum(b[0].shape[0] for b in scored)
    crow = sum(b[0].shape[0] for b in calib)
    if rows != a.rows or crow < a.calib_rows:
        raise SystemExit(f"packed {rows} scored / {crow} calibration rows")
    res.update({"rows": rows, "calib_rows": crow, "batch": a.batch,
                "tol_rel": a.tol_rel, "ref_depth": a.ref_depth})
    D = a.ref_depth
    match = [int(x) for x in a.match_depths.split(",")]

    # ── Step 1a: calibrate a_d on the calibration rows ──────────────────────────────
    target = run_point(m, calib, D, 1.0, a.device, a.tol)
    res["calib_target"] = {k: target[k] for k in ("sep_abs", "sep_rel", "sep_code")}
    print(f"calibration: {crow} rows, target sep_abs(d{D}, a=1) = {target['sep_abs']:.4f}",
          flush=True)
    a_d: dict[int, float] = {}
    res["calib"] = {}
    for d in match:
        def f(x, d=d):
            return run_point(m, calib, d, x, a.device, a.tol)["sep_abs"]
        ad, trace = match_multiplier(f, target["sep_abs"], tol_rel=a.tol_rel)
        a_d[d] = ad
        res["calib"][d] = {"a": ad, "trace": trace}
        print(f"  a_{d} = {ad:.4f}  ({len(trace)} evals; sep {min(trace, key=lambda t: abs(t[1] - target['sep_abs']))[1]:.4f})",
              flush=True)

    # ── Step 1b / 2: score the rows ─────────────────────────────────────────────────
    points = [(D, 1.0)] + [(d, 1.0) for d in match]
    points += [(d, a_d[d]) for d in match] + [(D, 1.0 / a_d[d]) for d in match]
    pts: dict[tuple[int, float], dict] = {}
    for d, x in points:
        if (d, x) in pts:
            continue
        pts[(d, x)] = run_point(m, scored, d, x, a.device, a.tol)
        p = pts[(d, x)]
        print(f"  d{d} a={x:.4f}: coda {p['coda_mix'].mean():.4f}  sep_abs "
              f"{p['sep_abs']:.4f} sep_rel {p['sep_rel']:.4f} sep_code {p['sep_code']:.4f}",
              flush=True)
    cidx = pts[(D, 1.0)]["coda_idx"]
    for k, p in pts.items():
        if not np.array_equal(p["coda_idx"], cidx):
            raise RuntimeError(f"point {k}: the coda token set moved")
    blk = _blocks(cidx)
    off = token_offsets(scored, cidx)
    nb, sd = a.n_boot, a.seed
    ce = {k: p["coda_mix"] for k, p in pts.items()}
    res["points"] = {f"d{d}_a{x:.6g}": {"depth": d, "a": x, "coda_ce": float(ce[(d, x)].mean()),
                                        "sep_abs": pts[(d, x)]["sep_abs"],
                                        "sep_rel": pts[(d, x)]["sep_rel"],
                                        "sep_code": pts[(d, x)]["sep_code"]}
                     for d, x in pts}

    # the stored Stage 2 arrays of the same checkpoint: the a = 1 points must reproduce
    if a.ref_npz:
        z = np.load(a.ref_npz)
        rl = a.ref_label or label
        if not np.array_equal(z[f"{rl}_coda_idx"], cidx):
            raise RuntimeError("--ref-npz coda_idx differs from these rows")
        chk = {}
        for d in sorted({1, D} & {dd for dd, _x in pts}):
            st = z[f"{rl}_coda_mix_{d}"].astype(np.float64)
            md = float(ce[(d, 1.0)].mean() - st.mean())
            chk[d] = {"mean_diff": md, "max_abs_dev": float(np.abs(ce[(d, 1.0)] - st).max())}
            if abs(md) > a.ref_tol:
                raise RuntimeError(f"d{d} a=1 mean coda CE differs from the stored Stage 2 "
                                   f"value by {md:+.2e} (> {a.ref_tol})")
        res["ref_check"] = chk
        print(f"  ref check vs {a.ref_npz} [{rl}]: {chk}", flush=True)

    R = {}
    ref = ce[(D, 1.0)]
    for d in match:
        un, mt = ce[(d, 1.0)], ce[(d, a_d[d])]
        rv = ce[(D, 1.0 / a_d[d])]
        R[d] = {
            "a": a_d[d],
            f"K{d}-K{D}": _ci(un, ref, blk, None, nb, sd),
            f"K{d}*-K{D}": _ci(mt, ref, blk, None, nb, sd),
            f"survive_{d}": ratio_ci(mt, ref, un, ref, blk, nb, sd),
            f"reverse_{d} CE(d{D},1/a)-CE(d{d},1)": _ci(rv, un, blk, None, nb, sd),
            f"loud-code gain at d{d} CE(d,1)-CE(d,a)": _ci(un, mt, blk, None, nb, sd),
            "by_bin": {},
        }
        for lo, hi in [*BINS, (-1, -1)]:
            sel = (off >= lo) & (off <= hi)
            if not sel.any():
                continue
            name = "first_span_or_dump" if lo < 0 else f"{lo}-{hi}"
            R[d]["by_bin"][name] = {
                "n_tokens": int(sel.sum()),
                "ce_d1": float(un[sel].mean()), "ce_matched": float(mt[sel].mean()),
                "ce_ref": float(ref[sel].mean()), "ce_reverse": float(rv[sel].mean()),
                f"K{d}-K{D}": _ci(un, ref, blk, sel, nb, sd),
                f"K{d}*-K{D}": _ci(mt, ref, blk, sel, nb, sd),
                "reverse": _ci(rv, un, blk, sel, nb, sd),
            }
    res["readings"] = R
    res["wall_s"] = round(time.time() - t0, 1)
    npz = a.out.rsplit(".", 1)[0] + ".tokens.npz"
    np.savez_compressed(npz, coda_idx=cidx.astype(np.int64), off=off.astype(np.int16),
                        **{f"ce_d{d}_a{x:.6g}": v.astype(np.float32) for (d, x), v in ce.items()})
    res["tokens_npz"] = npz
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"[{label}] amplitude-matched K-curve, {rows} rows (calibration {crow} rows)")
    for d, r in R.items():
        sv = r[f"survive_{d}"]
        print(f"  d{d}: a_{d} = {r['a']:.4f}")
        for key in (f"K{d}-K{D}", f"K{d}*-K{D}", f"reverse_{d} CE(d{D},1/a)-CE(d{d},1)",
                    f"loud-code gain at d{d} CE(d,1)-CE(d,a)"):
            print(f"    {key:38s} {_fmt(r[key])}")
        print(f"    survive_{d:<31d} {sv['point']:+.3f} [{sv['lo']:+.3f}, {sv['hi']:+.3f}]")
        b0 = r["by_bin"].get("0-0")
        if b0:
            print(f"    offset 0: K-K6 {_fmt(b0[f'K{d}-K{D}'])}  matched "
                  f"{_fmt(b0[f'K{d}*-K{D}'])}")
    print(f"wrote {a.out} and {npz}")


if __name__ == "__main__":
    main()
