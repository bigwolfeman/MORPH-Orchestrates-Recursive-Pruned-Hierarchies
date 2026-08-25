"""Zero-deviation forcing bias in the looped core — the SCSE reading, measured.

Pre-registration: docs/experiments/planned/2026-08-24-tul-zero-deviation-forcing-bias.md

SCSE (arXiv:2607.27656, Fig. 1) contrasts a recurrence that re-injects the source at every
step, `h_{t+1} = G_theta(h_t + e)`, with one that uses the source ONCE to set an anchor and
then evolves a deviation through a core with `G_theta(0) = 0`. The first form leaves a
source-driven term `b_t(e)` alive at the anchor, so the state drifts every step and the
drift compounds with depth.

MORPH's core step is the first form twice over (`transformer.py:_apply_core_step`): a
DiagonalInjection `A*h_ctx + dt*e_ctx` on 320 of 1024 channels, then `n_core` additions of a
loop-INVARIANT term `inj_term_i` before each block. That is structure, not evidence. This
probe measures whether the resulting displacement is (a) persistent across iterations and
(b) SHARED across positions, and attributes the shared part to the two injections.

Two numbers per loop iteration, on the active positions only:

    rel_t = rms||d_t|| / rms||h_t||          d_t = f_theta(h_t) - h_t
    C_t   = P * ||mean_p d_p||^2 / mean_p||d_p||^2

`C` is the concentration of the displacement onto ONE shared direction. Displacements
spread isotropically over the P active positions give `C ~ 1`; every position moving the
same way gives `C = P`. The `P` factor is what makes the 57-slot path (arm A1) comparable
to the 1024-token path (arm A0) — without it the two baselines differ by 18x and the
comparison is unreadable, the same trap the effective-rank readings fell into.

The discriminator against oversmoothing is `rel_t`. A contraction toward a shared fixed
point must SHRINK the displacement; a persistent forcing bias must not.

Correctness gate. The replayed `f_theta(h_t)` must reproduce the NEXT captured state on the
active positions. A probe that cannot reproduce the trajectory it claims to measure is not
measuring the training path, so `--gate` raises instead of reporting.

Usage:
    PYTHONPATH=$PWD python lab/divergence/drift_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out drift.json
    PYTHONPATH=$PWD python lab/divergence/drift_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --token-path --out drift_a0.json
    PYTHONPATH=$PWD python lab/divergence/drift_probe.py --self-test
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import re
import sys

import torch
import torch.nn as nn

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lab.divergence.jac_ladder import build                    # noqa: E402
from morph.training.core_jacobian import CoreJacobianProbe     # noqa: E402
from morph.training.train import load_checkpoint               # noqa: E402


# ── geometry ───────────────────────────────────────────────────────────────────────
def select(t: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`[B, S, ...] -> [P, F]` on the True positions of `[B, S]` mask, in fp32.

    `flatten(2)` keeps the hyper-connection stream axis inside the feature vector rather
    than averaging it away: a displacement that is shared across positions but opposite
    across streams is not a shared drift, and a mean over streams would call it one.
    """
    return t.flatten(2).float()[mask]


def concentration(d: torch.Tensor) -> tuple[float, float]:
    """`(C, rms_norm)` for displacements `d` shaped `[P, F]`.

    `C = P * ||mean_p d_p||^2 / mean_p ||d_p||^2` — 1 for isotropic, P for identical.
    """
    p = d.shape[0]
    msq = d.pow(2).sum(-1).mean()
    dbar = d.mean(0)
    c = float(p * dbar.pow(2).sum() / msq.clamp_min(1e-30))
    return c, float(msq.sqrt())


def spread(d: torch.Tensor) -> tuple[float, float]:
    """`(eff_positions, max_share)` of the ENERGY of `d` over its `P` rows.

    `C` alone cannot tell two very different pictures apart, and both drive it to ~1:

      * the rows are uncorrelated directions of similar size — isotropic;
      * ONE row carries nearly all the energy and the rest are noise — a sink. Then
        `mean||d||^2 ~ ||d_big||^2 / P` and `||mean d||^2 ~ ||d_big||^2 / P^2`, so `C ~ 1`
        no matter how extreme the sink is.

    The participation ratio over rows separates them: `eff = (sum a)^2 / sum a^2` with
    `a_p = ||d_p||^2`, which is `P` when the energy is spread evenly and `1` when one row
    holds it all. `max_share` is that row's fraction, reported because a participation
    ratio near 1 and a max share near 1 are the same statement said twice and disagreeing
    is itself a finding.

    Added 2026-08-24 as Method Amendment 1 to the pre-registered method, after the first
    A1 pass returned `C -> 1` across the loop: the prediction is untouched, but `C` alone
    could not say WHICH way it got there. `a` is computed in float64 — a sink makes the
    ratio of a sum of squares to a square of sums, and bf16-sourced fp32 energies span
    enough orders of magnitude for that to lose the tail.
    """
    a = d.double().pow(2).sum(-1)
    s1, s2 = a.sum(), a.pow(2).sum()
    eff = float(s1 * s1 / s2.clamp_min(1e-300))
    return eff, float(a.max() / s1.clamp_min(1e-300))


@contextlib.contextmanager
def dropout_off(model):
    """Zero every `nn.Dropout` rate for the body of the block.

    The core map the probe replays must be a FUNCTION of `(h, e, inj)`. In `model.train()`
    every core block applies `nn.Dropout(0.1)` after each sublayer, so a replayed step draws
    a different mask than the captured one and the replay lands 24 % away from the captured
    next state — measured, and the trajectory gate is what caught it. Turning the model to
    `eval()` instead would ALSO switch the loop from Poisson depths to a uniform
    `mean_depth`, i.e. change the operating point, so the rates are zeroed and train mode is
    kept. Dropout is zero-mean and independent across positions, so it can only add an
    isotropic component to the displacement: removing it can raise the measured
    concentration `C`, never lower it. That is stated in the writeup and it applies equally
    to every arm and rung compared here.
    """
    mods = [m for m in model.modules() if isinstance(m, nn.Dropout)]
    old = [m.p for m in mods]
    for m in mods:
        m.p = 0.0
    try:
        yield len(mods)
    finally:
        for m, q in zip(mods, old):
            m.p = q


class _Identity(nn.Module):
    """Stand-in for DiagonalInjection that passes the carrier through untouched."""

    def forward(self, h, e):        # noqa: D102 - signature mirrors DiagonalInjection
        return h


@contextlib.contextmanager
def diag_off(root):
    """Replace the DiagonalInjection with a pass-through for the body of the block."""
    old = root.injection
    root.injection = _Identity()
    try:
        yield
    finally:
        root.injection = old


# ── the core map, replayed at a captured operating point ───────────────────────────
def step_at(root, point: dict, *, zero_inj: bool = False, no_diag: bool = False):
    """`f_theta(h_t)` at the captured point, optionally with an injection removed.

    Runs under the SAME autocast the capture ran under, so the replay reproduces the
    captured trajectory exactly rather than to fp32-vs-bf16 tolerance.
    """
    h, e = point["h"], point["e"]
    inj = torch.zeros_like(point["inj"]) if zero_inj else point["inj"]
    ctx = diag_off(root) if no_diag else contextlib.nullcontext()
    with dropout_off(root), ctx, torch.no_grad(), \
            torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = root._apply_core_step(h, e, None, None, None,
                                       ret_state=point["ret_state"],
                                       iter_idx=int(point["iter_idx"]), inj_terms=inj)
    return out


def trajectory_gate(root, points: list[dict], tol: float = 1e-2) -> float:
    """Max relative error between the replayed step and the next captured state.

    Only the positions active at `t` are compared: an inactive position is written back
    unchanged by `torch.where(active, h_new, h)`, so its captured value is `h_t`, not the
    step output, and comparing it would measure the freeze rather than the map. On the
    token path the active set SHRINKS to a prefix of the depth-sorted batch, so the
    comparison is over the rows both points hold.
    """
    worst = 0.0
    for a, b in zip(points, points[1:]):
        if int(b["iter_idx"]) != int(a["iter_idx"]) + 1:
            continue
        n = min(a["h"].shape[0], b["h"].shape[0])
        got = step_at(root, a)[:n].float()
        want = b["h"][:n].float()
        m = (a["active"][:n] & b["active"][:n])
        if not bool(m.any()):
            continue
        g, w = select(got, m), select(want, m)
        err = float((g - w).norm() / w.norm().clamp_min(1e-30))
        worst = max(worst, err)
    if worst > tol:
        raise RuntimeError(
            f"drift probe cannot reproduce the captured trajectory: relative error "
            f"{worst:.3e} > {tol:.0e}. The replayed map is not the map the run ran, so "
            f"every number below would be about a different operator.")
    return worst


def capture(model, x, y, layout, seed: int, token_path: bool) -> list[dict]:
    """One operating point per core-loop iteration at a FIXED depth draw."""
    probe = CoreJacobianProbe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with dropout_off(model), probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=None if token_path else layout)
    return [dict(p) for p in pts]


def drift(root, points: list[dict]) -> dict:
    """Per-iteration displacement geometry, plus the two injection ablations."""
    per_iter = []
    for p in points:
        m = p["active"]
        if not bool(m.any()):
            continue
        h = select(p["h"], m)
        h_rms = float(h.pow(2).sum(-1).mean().sqrt())
        row = {"iter": int(p["iter_idx"]), "n_pos": int(m.sum()), "h_rms": h_rms}
        eff_h, share_h = spread(h)
        row["eff_pos_h"] = eff_h
        row["max_share_h"] = share_h
        for tag, kw in (("full", {}), ("noinj", {"zero_inj": True}),
                        ("nodiag", {"no_diag": True})):
            d = select(step_at(root, p, **kw) - p["h"], m)
            c, rms = concentration(d)
            eff, share = spread(d)
            row[f"C_{tag}"] = c
            row[f"rel_{tag}"] = rms / max(h_rms, 1e-30)
            row[f"eff_pos_{tag}"] = eff
            row[f"max_share_{tag}"] = share
        # concentration of the loop-invariant additive term itself, per core layer
        inj = p["inj"]
        row["C_inj"] = [round(concentration(select(inj[i], m))[0], 3)
                        for i in range(inj.shape[0])]
        per_iter.append(row)
    return {"per_iter": per_iter}


# ── self-test ──────────────────────────────────────────────────────────────────────
def self_test() -> None:
    """`concentration` and `select` on inputs whose answers are known in closed form."""
    torch.manual_seed(0)
    p, f = 500, 64

    iso = torch.randn(p, f)
    c_iso, _ = concentration(iso)
    assert 0.5 < c_iso < 2.0, f"isotropic C should sit near 1, got {c_iso}"

    same = torch.randn(1, f).expand(p, f).contiguous()
    c_same, _ = concentration(same)
    assert abs(c_same - p) / p < 1e-4, f"identical rows should give C = P = {p}, got {c_same}"

    # C must not move when the number of positions changes at fixed geometry — this is the
    # whole reason for the P factor, and it is what makes A0 comparable to A1.
    c_small, _ = concentration(torch.randn(50, f))
    assert 0.3 < c_small < 3.0, f"C drifted with P on isotropic input: {c_small}"

    # a half-shared field: rows = v + noise. C must land between the two extremes and rise
    # with the shared amplitude.
    v = torch.randn(1, f)
    lo, _ = concentration(0.3 * v + torch.randn(p, f))
    hi, _ = concentration(3.0 * v + torch.randn(p, f))
    assert lo < hi, f"C did not rise with the shared component: {lo} vs {hi}"
    assert hi > 10 * lo, f"C is not sensitive enough to the shared component: {lo} -> {hi}"

    # select() must keep the stream axis inside the feature vector, not average it.
    t = torch.zeros(2, 3, 4, 8)
    t[0, 0, 0] = 1.0
    t[0, 0, 1] = -1.0
    mask = torch.zeros(2, 3, dtype=torch.bool)
    mask[0, 0] = True
    s = select(t, mask)
    assert s.shape == (1, 32), f"select must flatten streams into features, got {s.shape}"
    assert float(s.norm()) > 0.9, "select averaged the stream axis away (norm collapsed)"

    # spread(): the sink/isotropic separation C cannot make.
    eff_iso, sh_iso = spread(iso)
    assert eff_iso > 0.7 * p, f"isotropic rows must spread over ~P positions, got {eff_iso}"
    sink = torch.randn(p, f) * 1e-4
    sink[0] = torch.randn(f) * 1e3
    c_sink, _ = concentration(sink)
    eff_sink, sh_sink = spread(sink)
    assert c_sink < 3.0, f"a sink must ALSO read C ~ 1 (this is the confound): {c_sink}"
    assert eff_sink < 1.05, f"a sink must read eff_positions ~ 1, got {eff_sink}"
    assert sh_sink > 0.99, f"a sink must hold ~all the energy, got {sh_sink}"
    assert sh_iso < 0.05, f"isotropic max_share should be small, got {sh_iso}"

    # rms scale: a field of unit-norm rows must report rms_norm 1.
    unit = torch.nn.functional.normalize(torch.randn(p, f), dim=-1)
    _, rms = concentration(unit)
    assert abs(rms - 1.0) < 1e-4, f"rms_norm wrong: {rms}"
    print("DRIFT_PROBE_SELF_TEST_PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--token-path", action="store_true",
                    help="arm A0's code path (slot_layout=None) on the SAME weights")
    ap.add_argument("--gate-tol", type=float, default=1e-2)
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return
    if not (a.ckpt_dir and a.out):
        ap.error("--ckpt-dir and --out are required unless --self-test")

    ov = [o for o in a.overrides.split(",") if o.strip()]
    _cfg, model, x, y, layout = build(a.config_name, ov)
    model.train()                       # Poisson depths, as in training
    root = getattr(model, "_orig_mod", model)

    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    results = []
    for path in ckpts:
        step = int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
        load_checkpoint(path, model, scaler, torch.device("cuda"))
        pts = capture(model, x, y, layout, a.seed, a.token_path)
        gate = trajectory_gate(root, pts, a.gate_tol)
        d = drift(root, pts)
        d.update({"ckpt": os.path.basename(path), "step": step, "gate_rel_err": gate,
                  "token_path": a.token_path})
        results.append(d)
        pi = d["per_iter"]
        cs = " ".join(f"{r['C_full']:7.1f}" for r in pi)
        rs = " ".join(f"{r['rel_full']:7.3f}" for r in pi)
        print(f"{os.path.basename(path):<22} P={pi[0]['n_pos']:>5} gate={gate:.1e}",
              flush=True)
        print(f"{'':<22} C_full/iter  = {cs}", flush=True)
        print(f"{'':<22} rel_full/iter= {rs}", flush=True)
        es = " ".join(f"{r['eff_pos_full']:7.1f}" for r in pi)
        eh = " ".join(f"{r['eff_pos_h']:7.1f}" for r in pi)
        print(f"{'':<22} effpos_d/iter= {es}", flush=True)
        print(f"{'':<22} effpos_h/iter= {eh}", flush=True)
        print(f"{'':<22} C_noinj last = {pi[-1]['C_noinj']:.1f}  "
              f"C_nodiag last = {pi[-1]['C_nodiag']:.1f}  "
              f"maxshare_d last = {pi[-1]['max_share_full']:.3f}  "
              f"C_inj = {pi[-1]['C_inj']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
