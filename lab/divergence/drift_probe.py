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


def state_geom(h: torch.Tensor, cap: int = 2048, seed: int = 0,
               centred: bool = False) -> tuple[float, float]:
    """`(eff_rank, mean_pairwise_cos)` of the DIRECTIONS of the states `h` shaped `[P, F]`.

    Rows are unit-normalised first. The participation ratio of a norm-weighted spectrum
    reads "one position has a huge carrier" as low rank even when the directions are
    perfectly spread, and the question here is whether the positions have MERGED, which is
    a statement about directions. Same normalisation as `jac_ladder.state_geometry`'s
    `eff_rank_unit`, so the numbers are comparable to the campaign's earlier readings.

    Rows are subsampled to `cap` with a fixed seed when there are more, because the token
    path carries 6912 positions and a 6912x4096 double SVD costs minutes per rung for a
    number that does not move under subsampling. The cap is reported in the output so a
    reader never has to guess whether it bound.

    `centred` subtracts the mean state first, and it is not a cosmetic choice. These states
    carry a mean pairwise cosine near +0.5, so one large common component dominates the
    uncentred spectrum and pins the rank near 3 however spread the residuals are. The
    campaign's slot-INPUT diversity number (~28) was computed CENTRED and its post-loop
    number was not, so "the loop destroys 10x more diversity than pooling" compared two
    different measures. Both are reported here, on the same tensors, per iteration.
    """
    if h.shape[0] > cap:
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(h.shape[0], generator=g)[:cap].to(h.device)
        h = h[idx]
    if centred:
        h = h - h.mean(0, keepdim=True)
    hn = h / h.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sv = torch.linalg.svdvals(hn.double()).pow(2)
    eff = float(sv.sum().pow(2) / sv.pow(2).sum().clamp_min(1e-300))
    gram = hn.double() @ hn.double().t()
    n = gram.shape[0]
    cos = float((gram.sum() - n) / (n * (n - 1)))
    return eff, cos


class _DiagVariant(nn.Module):
    """`DiagonalInjection` with either half of it switched off.

    The real module is `h_ctx <- A * h_ctx + dt * e_ctx`. Replacing the WHOLE module with a
    pass-through removes both halves at once — the fresh per-slot injection `dt * e_ctx`
    AND the decay `A * h_ctx` — so an effect measured that way cannot be attributed to
    either. The first pass of this probe did exactly that and the writeup over-claimed from
    it. These flags separate them:

    * `drop_inject` sets `dt = 0`: the decay still runs, no fresh per-slot content enters.
      This is the ablation that tests whether the INJECTION carries the position-specific
      signal, i.e. the term SCSE replaces with a once-only anchor.
    * `drop_decay` sets `A = 1`: the fresh content still enters, the ctx channels no longer
      contract toward it.

    Both flags together reproduce the pass-through, which is the self-test.

    The base module is stored with `object.__setattr__` so it is NOT registered as a
    submodule: registering it would graft a second name for `log_A` / `log_dt` onto the
    model for the duration of the swap, which a later `state_dict` or parameter walk would
    see.
    """

    def __init__(self, base, drop_inject: bool = False, drop_decay: bool = False):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.drop_inject = drop_inject
        self.drop_decay = drop_decay

    def forward(self, h, e):        # signature mirrors DiagonalInjection
        b = self.base
        a = b.log_A.exp().clamp(max=0.9999)
        dt = b.log_dt.exp()
        if self.drop_decay:
            a = torch.ones_like(a)
        if self.drop_inject:
            dt = torch.zeros_like(dt)
        s, e_ = b.start, b.end
        new_ctx = a * h[..., s:e_] + dt * e[..., s:e_]
        return torch.cat([h[..., :s], new_ctx, h[..., e_:]], dim=-1)


class _OnceOnlyDiag(nn.Module):
    """`DiagonalInjection` that fires on the FIRST core iteration only.

    This is MORPH's analogue of how SCSE handles the source: use `e` ONCE to set an anchor,
    then evolve without re-injecting it. Iteration 0 runs the real
    `A * h_ctx + dt * e_ctx`; every later iteration passes the carrier through untouched,
    i.e. `A = 1, dt = 0`, so the identity written at iteration 0 PERSISTS instead of
    decaying away.

    That last part is the whole point, and it is what the `dt = 0` ablation gets wrong as an
    SCSE stand-in: with the decay still running, killing the injection lets the ctx band
    decay toward zero, so the positions lose their identity by erasure rather than by the
    substitution SCSE actually proposes. This module tests the substitution.

    The iteration index is a CALL COUNTER, not an argument, because the module's signature
    is `(h, e)`. `_apply_core_step` calls it exactly once per loop iteration on both code
    paths, so call `k` is iteration `k` within one forward. `reset()` must be called before
    every forward; `self_test` checks the counter, and the caller checks that iteration 0 of
    the patched capture is bit-identical to the baseline's.

    What this does NOT model: SCSE's bias-free core (`G_theta(0) = 0`) and its zero-deviation
    mask. It isolates the source handling, which is the only part the de-correlation argument
    turns on.
    """

    def __init__(self, base):
        super().__init__()
        object.__setattr__(self, "base", base)
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def forward(self, h, e):        # signature mirrors DiagonalInjection
        k, self.calls = self.calls, self.calls + 1
        if k > 0:
            return h
        return self.base(h, e)


@contextlib.contextmanager
def diag_once(root):
    """Swap in the once-only injection for the body of the block, counter reset."""
    old = root.injection
    mod = _OnceOnlyDiag(old)
    mod.reset()
    root.injection = mod
    try:
        yield mod
    finally:
        root.injection = old


@contextlib.contextmanager
def diag_variant(root, *, drop_inject: bool = False, drop_decay: bool = False):
    """Swap the DiagonalInjection for a variant with one or both halves off."""
    old = root.injection
    root.injection = _DiagVariant(old, drop_inject, drop_decay)
    try:
        yield
    finally:
        root.injection = old


# ── the core map, replayed at a captured operating point ───────────────────────────
def step_at(root, point: dict, *, zero_inj: bool = False, no_diag: bool = False,
            no_dt: bool = False, no_decay: bool = False):
    """`f_theta(h_t)` at the captured point, optionally with an injection removed.

    Runs under the SAME autocast the capture ran under, so the replay reproduces the
    captured trajectory exactly rather than to fp32-vs-bf16 tolerance.
    """
    h, e = point["h"], point["e"]
    inj = torch.zeros_like(point["inj"]) if zero_inj else point["inj"]
    if no_diag or no_dt or no_decay:
        ctx = diag_variant(root, drop_inject=no_diag or no_dt,
                           drop_decay=no_diag or no_decay)
    else:
        ctx = contextlib.nullcontext()
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


def capture(model, x, y, layout, seed: int, token_path: bool,
            once_only: bool = False) -> list[dict]:
    """One operating point per core-loop iteration at a FIXED depth draw.

    `once_only` runs the WHOLE forward with the source injected at iteration 0 only, so the
    captured trajectory is the counterfactual one rather than a one-step counterfactual off
    the real trajectory. The depth draw is seeded identically, so the two trajectories are
    compared at the same input and the same active sets.
    """
    root = getattr(model, "_orig_mod", model)
    probe = CoreJacobianProbe(model)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    patch = diag_once(root) if once_only else contextlib.nullcontext()
    with dropout_off(model), patch, probe.capture() as pts:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=None if token_path else layout)
    return [dict(p) for p in pts]


def drift(root, points: list[dict], plain: bool = False, once_only: bool = False) -> dict:
    """Per-iteration displacement geometry, plus the injection ablations.

    Two position sets are reported, and the difference matters. The per-iteration `active`
    set SHRINKS as the Poisson depth draw freezes positions — 342 to 96 on the slot path —
    and an effective rank read off fewer samples is biased downward, so a rank measured on
    `active` at iteration 0 and at iteration 7 is a comparison across sample sizes, which
    this campaign's own trap list forbids. `*_common` repeats the state geometry on the
    INTERSECTION of every iteration's active set, a fixed group, which is the comparison a
    trend across iterations may actually be read from.
    """
    per_iter = []
    common = None
    for p in points:
        if not bool(p["active"].any()):
            continue
        a = p["active"]
        if common is None:
            common = a.clone()
        else:
            n = min(common.shape[0], a.shape[0])
            common = common[:n] & a[:n]
    for p in points:
        m = p["active"]
        if not bool(m.any()):
            continue
        nrow = min(m.shape[0], common.shape[0])
        cm = common[:nrow]
        h = select(p["h"], m)
        h_rms = float(h.pow(2).sum(-1).mean().sqrt())
        row = {"iter": int(p["iter_idx"]), "n_pos": int(m.sum()), "h_rms": h_rms}
        eff_h, share_h = spread(h)
        row["eff_pos_h"] = eff_h
        row["max_share_h"] = share_h
        rank_h, cos_h = state_geom(h)
        rank_c, cos_c = state_geom(h, centred=True)
        row["rank_h"] = rank_h
        row["cos_h"] = cos_h
        row["rank_h_centred"] = rank_c
        row["cos_h_centred"] = cos_c
        hc = select(p["h"][:nrow], cm)
        row["n_pos_common"] = int(cm.sum())
        row["rank_h_common"] = state_geom(hc)[0]
        row["rank_h_common_centred"], row["cos_h_common"] = state_geom(hc, centred=True)[0], \
            state_geom(hc)[1]
        row["rank_capped"] = bool(h.shape[0] > 2048)
        # A once-only trajectory must be replayed with the once-only MAP. Replaying it with
        # the real injection restored measures what the real map would do at counterfactual
        # states, which is a different question and — because iteration 0 is shared — makes
        # the first two iterations identical by construction. That defect shipped once.
        base_kw = {"no_diag": True} if (once_only and int(p["iter_idx"]) > 0) else {}
        modes = (("full", base_kw),) if plain else (
            ("full", base_kw), ("noinj", {"zero_inj": True}), ("nodiag", {"no_diag": True}),
            ("nodt", {"no_dt": True}), ("nodecay", {"no_decay": True}))
        for tag, kw in modes:
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

    # _DiagVariant with both flags off must be the module it replaces, exactly; with both
    # flags on it must be the pass-through the first version of this probe used.
    from morph.model.transformer import DiagonalInjection
    torch.manual_seed(1)
    di = DiagonalInjection(4, 10)
    with torch.no_grad():
        di.log_A.normal_(-0.5, 0.3)
        di.log_dt.normal_(-0.2, 0.3)
    hh, ee = torch.randn(2, 5, 16), torch.randn(2, 5, 16)
    ref = di(hh, ee)
    assert torch.equal(_DiagVariant(di)(hh, ee), ref), "_DiagVariant is not the identity swap"
    both = _DiagVariant(di, drop_inject=True, drop_decay=True)(hh, ee)
    assert torch.equal(both, hh), "both flags on must reproduce the pass-through"
    only_dt = _DiagVariant(di, drop_inject=True)(hh, ee)
    only_decay = _DiagVariant(di, drop_decay=True)(hh, ee)
    assert not torch.equal(only_dt, ref) and not torch.equal(only_decay, ref), \
        "a single flag changed nothing"
    assert not torch.equal(only_dt, only_decay), "the two flags are not distinguishable"
    # the flags must touch ONLY the ctx band
    for v in (only_dt, only_decay, both):
        assert torch.equal(v[..., :4], hh[..., :4]) and torch.equal(v[..., 10:], hh[..., 10:]), \
            "a variant modified channels outside [start, end)"
    assert not hasattr(_DiagVariant(di), "_modules") or \
        "base" not in _DiagVariant(di)._modules, "base leaked in as a registered submodule"

    # _OnceOnlyDiag: real on call 0, pass-through after, and resettable.
    once = _OnceOnlyDiag(di)
    once.reset()
    assert torch.equal(once(hh, ee), ref), "once-only must run the real map on call 0"
    assert torch.equal(once(hh, ee), hh), "once-only must pass through on call 1"
    assert torch.equal(once(hh, ee), hh), "once-only must pass through on call 2"
    once.reset()
    assert torch.equal(once(hh, ee), ref), "reset() did not restore the first-call behaviour"

    # state_geom: orthogonal rows are full rank and mutually orthogonal; copies of ONE
    # direction are rank 1 and cos 1. A sign-flipped pair must read cos 0, not 1 — the
    # measure has to see anti-alignment, which an abs() would hide.
    q = torch.linalg.qr(torch.randn(64, 64))[0]
    r_o, c_o = state_geom(q)
    assert r_o > 60, f"orthogonal rows should be near-full rank, got {r_o}"
    assert abs(c_o) < 0.05, f"orthogonal rows should have cos ~ 0, got {c_o}"
    one = torch.randn(1, 64).expand(64, 64).contiguous()
    r_1, c_1 = state_geom(one)
    assert r_1 < 1.05, f"identical directions should be rank 1, got {r_1}"
    assert c_1 > 0.99, f"identical directions should have cos 1, got {c_1}"
    pair = torch.cat([one[:32], -one[:32]])
    r_p, c_p = state_geom(pair)
    assert r_p < 1.05 and abs(c_p) < 0.02, f"sign-flipped pair: rank {r_p}, cos {c_p}"
    # centring: a field with a big common component reads LOW uncentred and HIGH centred.
    # This is the artefact that made the campaign's pre/post loop comparison invalid, so it
    # gets an assertion rather than a comment.
    shared = 5.0 * torch.randn(1, 64) + torch.randn(200, 64)
    r_un, _ = state_geom(shared)
    r_ct, _ = state_geom(shared, centred=True)
    assert r_un < 5.0, f"uncentred rank should be dominated by the common part: {r_un}"
    assert r_ct > 8 * r_un, f"centring must expose the residual spread: {r_un} -> {r_ct}"

    # the subsample cap must not move the answer on homogeneous input
    big = torch.randn(5000, 64)
    r_b, _ = state_geom(big)
    assert 55 < r_b < 65, f"capped eff rank drifted: {r_b}"

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
    ap.add_argument("--once-only", action="store_true",
                    help="also capture the counterfactual trajectory with the source "
                         "injected at iteration 0 ONLY (SCSE's source handling)")
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
        if a.once_only:
            once = capture(model, x, y, layout, a.seed, a.token_path, once_only=True)
            # Gate: iteration 0 is BEFORE the injection has fired differently, so the two
            # trajectories must agree there exactly. If they do not, the patch changed
            # something other than the source handling and the comparison is void.
            if not torch.equal(once[0]["h"], pts[0]["h"]):
                raise RuntimeError(
                    "once-only capture differs from the baseline at iteration 0 — the patch "
                    "moved something other than the injection, so the counterfactual is not "
                    "comparable to the trajectory it is being compared against.")
            if len(once) != len(pts):
                raise RuntimeError(f"depth draw differed: {len(pts)} vs {len(once)} iters")
            d["once_only"] = drift(root, once, plain=True,
                                   once_only=True)["per_iter"]
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
        print(f"{'':<22} rank_h/iter  = " +
              " ".join(f"{r['rank_h']:7.2f}" for r in pi), flush=True)
        print(f"{'':<22} cos_h/iter   = " +
              " ".join(f"{r['cos_h']:+7.3f}" for r in pi), flush=True)
        print(f"{'':<22} rankC_h/iter = " +
              " ".join(f"{r['rank_h_centred']:7.2f}" for r in pi), flush=True)
        print(f"{'':<22} COMMON(P={pi[0]['n_pos_common']:>3}) rank = " +
              " ".join(f"{r['rank_h_common']:6.2f}" for r in pi), flush=True)
        print(f"{'':<22} COMMON centred rank = " +
              " ".join(f"{r['rank_h_common_centred']:6.2f}" for r in pi), flush=True)
        if a.once_only:
            o = d["once_only"]
            print(f"{'':<22} ONCE C/iter  = " +
                  " ".join(f"{r['C_full']:7.1f}" for r in o), flush=True)
            print(f"{'':<22} ONCE rank_h  = " +
                  " ".join(f"{r['rank_h']:7.2f}" for r in o), flush=True)
            print(f"{'':<22} ONCE cos_h   = " +
                  " ".join(f"{r['cos_h']:+7.3f}" for r in o), flush=True)
            print(f"{'':<22} ONCE rankC_h = " +
                  " ".join(f"{r['rank_h_centred']:7.2f}" for r in o), flush=True)
            print(f"{'':<22} ONCE COMMON centred = " +
                  " ".join(f"{r['rank_h_common_centred']:6.2f}" for r in o), flush=True)
        z = pi[-1]
        print(f"{'':<22} C last: full={z['C_full']:.2f} noinj={z['C_noinj']:.2f} "
              f"nodt={z['C_nodt']:.2f} nodecay={z['C_nodecay']:.2f} "
              f"nodiag={z['C_nodiag']:.2f}   maxshare_d={z['max_share_full']:.3f}",
              flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
