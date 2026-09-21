"""ANISOTROPIC GAIN: does one core pass contract a slot's stream DEVIATIONS faster than
its stream MEAN?

Probe 2 of Part 1 of
``.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md``.

THE QUESTION. On a fan arm (``tul.fan_k`` K) a span's slot state is K stream cells. The
trainer's per-pass centred rank (``fan/stream_rank_t{t}``, ``morph/model/tul_fan.py::
fan_stream_stats``) falls from 2.8 at pass 1 to 2.0 at pass 6 on ``fan4-all``. A UNIFORM
contraction of every deviation cannot change a rank: it rescales every singular value of
the K x C centred matrix by the same factor and the participation ratio is scale-free. So
the core must treat some directions differently from others. The first split worth
measuring is the one the rank statistic is built on: the K-axis splits orthogonally into
the stream MEAN (the K averaging projector P) and the DEVIATION subspace (Q = I - P), and
the rank reads Q alone. This probe measures the realised gain of ONE core pass separately
on the two.

WHAT IS DIFFERENTIATED — "one pass" is ``MORPHTransformer._apply_core_step``, the same
method ``morph/training/core_jacobian.py`` measures: the SSM diagonal injection followed
by the ``n_core`` shared blocks, with the per-layer x0/bigram injection stack, the
retention carry and the cell relation mask of this arm. On the slot loop that method is
the whole of ``_tul_core``'s ``_core_step`` whenever ``tul.reread``, ``tul.loop_carry``,
``tul.slot_chain`` and ``tul.grad_pass`` are all off (fan4-all: all off) — the probe
CHECKS that by pairing on the input tensor's storage and refuses otherwise, because a
pre-step injection would put part of the map outside what is differentiated.

HOW THE PASS-t OPERATING POINT IS OBTAINED — from the trajectory the EVAL path builds,
not from a re-run. The forward is ``tul_forward_ablated(..., plan_mode="normal")`` over
the trainer's own packed rows (the sibling probe ``fan_stream_probe.py`` captures the same
trajectory by wrapping ``morph.model.transformer.fan_stream_rank``; this one uses the
model's built-in capture hook instead). ``MORPHTransformer._jac_capture``, a list attached
for the duration, receives one dict per loop iteration holding the detached carrier ``h``
the iteration is about to consume, its seed ``e``, the injection stack, the retention
state, the iteration index and the ACTIVE set ``active & slot_valid``. Iteration index
``t`` (0-based) is reported as pass ``t + 1``, so pass 1's operating point is the SEED
state — the same indexing ``fan_repel_term`` uses for ``traj[0]``. The recorded
``attn_kw`` (the cell relation, ``slot_cell_relation``) and ``stage_cond`` come from a
recorder wrapped around ``_apply_core_step`` itself, so the replayed map carries the
arm's own attention relation and not the default.

``J v`` comes from the double-backward identity of ``core_jacobian.py`` (``g = J^T u`` is
linear in ``u``, so ``d g / d u`` contracted with ``v`` is ``J v``) — exact, no finite
differences, one forward per operating point and one backward per direction. fp32,
autocast off, restricted to the active set: a pad cell enters the loop at ``h = 0`` and an
RMSNorm at 0 has a Jacobian of order ``1/eps``, which is why the mask is not optional.

WHAT IS REPORTED, per pass.

* ``gain_mean_dir`` / ``gain_dev_dir`` — the PER-SLOT gain, averaged over slots and draws.
  A per-slot gain needs a per-slot operator: the direction is supported on ONE slot and
  the output is read at that SAME slot, which is exactly the diagonal block ``J_ss``. That
  restriction is not cosmetic here — at ``tul.loop_reach 0`` the cell relation is
  block-CAUSAL (a cell reads every cell of its own slot and every cell of earlier slots),
  so a whole-state draw would fold every earlier slot's inflow into a later slot's
  reading. ``leak_*`` reports how much of ``||J v||`` lands outside the perturbed slot, so
  the size of what the diagonal block leaves out is on the table beside the number.
* ``gain_realised_dev`` / ``gain_realised_mean`` — the gain along the state's OWN centred
  cells and along its own stream mean, normalised. The random-direction average and the
  realised direction can differ by a lot: the loop is a power iteration and the realised
  direction is the one that has already been through t passes of it
  (``morph-loop-is-a-power-iteration``, E7).
* ``gain_global_mean_dir`` / ``gain_global_dev_dir`` — every active slot perturbed at
  once, the whole active output read. This is the operator-wide counterpart, cross-slot
  inflow included, and it is the number to quote when the per-slot leak is large.
* ``cv_*`` / ``d_eff_*`` — the spread of the gain ACROSS draws inside one slot, and the
  effective number of directions it implies (:func:`_d_eff`). The mean/deviation ratio
  can only see anisotropy BETWEEN the two subspaces; a deviation subspace that is itself
  anisotropic (some of its ``(K-1) * n * C`` directions contracted harder than others) is
  also a way to move a rank, and only a within-family spread sees it.
* ``traj_rank`` / ``traj_cos`` — the trainer's own per-pass stream statistics
  (``fan_stream_stats``) at the very operating point being differentiated, over the
  ACTIVE slots, so the gain table can be read against the rank fall it is meant to
  explain without leaving the file.
* standard errors over SLOTS (per-slot means first, then the spread across slots).

KERNELS. ``main`` forces ``model.tg_scoped_kernels=false``. The fused HC-Cayley / CCA /
window path has a hand-written backward with no graph, so the second derivative does not
exist there and the JVP raises. Every number is therefore the EAGER map's. The 2026-09-08
"eager reads the gain 0.07 high" audit does not apply: that is a finite-difference noise
term and this probe takes no finite difference.

Usage (a probe host, never the trainer's GPU):

    python lab/divergence/fan_gain_probe.py \
        --ckpt fan4-all=tul_slot_spandec_strict_fan4_all=/path/step_5000.pt \
        --rows 24 --batch 3 --depth 6 --draws 16 --out fan_gain.json

``pass_gains`` is the pure function (tested against a synthetic split map in
``tests/test_fan_gain_probe.py``); ``main`` loads the checkpoint exactly as
``fan_stream_probe.py`` does, over the same loader, the same row packing and the same
forced depth, so the two probes read the same rows — the one difference is the eager
kernels above, which is why ``traj_rank`` is reported here as well: it is the stream
probe's own statistic on THIS forward's trajectory, and the two should agree.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

import torch


# ── the pure function ────────────────────────────────────────────────────────────────
def _unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.reshape(-1).double().norm().clamp_min(1e-30).to(v.dtype)


def _slot_direction(kind: str, m: int, trail: tuple[int, ...], gen: torch.Generator,
                    device, dtype, cells: torch.Tensor | None) -> torch.Tensor | None:
    """One unit direction on ONE slot, shaped ``[m, *trail]``.

    ``mean``  — the SAME Gaussian draw on all K streams: zero deviation component.
    ``dev``   — an independent Gaussian draw per stream, stream-sum removed: zero mean
                component. Isotropic in the ``(K-1) * prod(trail)`` deviation subspace.
    ``realised_mean`` / ``realised_dev`` — the two components of the slot's OWN state.
    Returns ``None`` when the direction is degenerate (a zero deviation, e.g. a slot whose
    K cells are exact copies), so the caller drops that slot instead of normalising noise.
    """
    if kind in ("mean", "dev"):
        n = 1 if kind == "mean" else m
        d = torch.randn((n, *trail), generator=gen).to(device=device, dtype=torch.float32)
        if kind == "mean":
            d = d.expand(m, *trail).contiguous()
        else:
            d = d - d.mean(dim=0, keepdim=True)
    elif kind in ("realised_mean", "realised_dev"):
        if cells is None:
            raise ValueError(f"{kind} needs the slot's live cells")
        z = cells.float()
        mu = z.mean(dim=0, keepdim=True)
        d = mu.expand(m, *trail).contiguous() if kind == "realised_mean" else z - mu
    else:
        raise ValueError(f"unknown direction family {kind!r}")
    if float(d.reshape(-1).double().norm()) < 1e-20:
        return None
    return _unit(d).to(dtype)


def _d_eff(cv: float, d: int) -> float:
    """Participation ratio of the map's SQUARED spectrum on a ``d``-dimensional subspace,
    inverted from the spread of ``||J v||`` across isotropic unit draws inside it.

    For ``v`` uniform on the unit sphere of a ``d``-dimensional subspace and
    ``X = ||J v||^2 = sum_i sigma_i^2 v_i^2``:

        E X   = m2 / d
        Var X = 2 (d m4 - m2^2) / (d^2 (d + 2))        m2 = sum sigma^2, m4 = sum sigma^4

    (exact, from ``E v_i^4 = 3 / (d(d+2))`` and ``E v_i^2 v_j^2 = 1 / (d(d+2))`` — the
    familiar ``Var X ~ 2 m4 / d^2`` drops the ``- m2^2`` term and is wrong by a factor of
    2 at ``d_eff = d/2``, which is the regime this probe is looking for). Writing
    ``d_eff = m2^2 / m4`` and ``cv`` for the coefficient of variation of the GAIN
    ``sqrt(X)`` (so ``cv(X) = 2 cv``) and solving:

        d_eff = d / (2 cv^2 (d + 2) + 1)

    A uniform scaling of the subspace gives ``cv = 0`` and ``d_eff = d`` — the map spreads
    its gain over every direction. A map that keeps half the subspace and kills the rest
    gives ``d / 2``. The result is clamped to ``[1, d]``.

    Read it as an order of magnitude, not a rank: ``cv`` is a second moment estimated
    from ``draws`` samples per slot, and it is blind to WHICH directions are large.
    """
    if d < 1:
        raise ValueError(f"_d_eff needs a subspace dimension >= 1, got {d}")
    if not (cv == cv):
        return float("nan")
    return min(float(d), max(1.0, d / (2.0 * cv * cv * (d + 2) + 1.0)))


def pass_gains(jvp, state: torch.Tensor, active: torch.Tensor, m_cells: int,
               draws: int = 16, seed: int = 0, slots_per_row: int = 0,
               global_draws: int | None = None) -> dict:
    """Realised per-pass gain of ``jvp`` on the MEAN and DEVIATION stream directions.

    ``jvp(v) -> J v``, both shaped like ``state`` ``[B, S*M, *trail]`` (``trail`` is
    ``(C,)`` or, on a Hyper-Connection carrier, ``(n_streams, C)``; the K-axis split is on
    the CELL axis only and the carrier axis is left isotropic). ``active`` is ``[B, S*M]``
    bool and must be constant within a slot — it is ``active & slot_valid`` from the loop,
    and both factors are per-slot, so a mixed slot means the layout was misread.

    ``slots_per_row`` caps the probed slots per row (0 = every valid slot); the kept ones
    are evenly spaced through the row so a row's late slots are represented. ``draws``
    random directions per slot per family. One ``jvp`` call carries one slot per ROW (rows
    do not interact), so the cost is ``ceil(slots_per_row) * draws * 2`` calls for the
    per-slot readings plus ``global_draws * 2`` for the whole-state ones.
    """
    if state.dim() < 3:
        raise ValueError(f"pass_gains wants [B, S*M, *trail], got {tuple(state.shape)}")
    if active.dim() != 2 or active.shape[:2] != state.shape[:2]:
        raise ValueError(f"active {tuple(active.shape)} does not index state "
                         f"{tuple(state.shape)}")
    if m_cells < 2:
        raise ValueError(f"pass_gains needs m_cells >= 2, got {m_cells}")
    b, sm = state.shape[:2]
    trail = tuple(state.shape[2:])
    if sm % m_cells:
        raise ValueError(f"compact axis {sm} is not a multiple of M = {m_cells}")
    s = sm // m_cells
    dev, dt = state.device, state.dtype
    if global_draws is None:
        global_draws = draws

    av = active.view(b, s, m_cells)
    full, anyc = av.all(dim=-1), av.any(dim=-1)
    if not torch.equal(full, anyc):
        raise ValueError("the active mask is not constant within a slot: "
                         "active & slot_valid should be per-slot")
    cell_mask = active.view(b, sm, *([1] * len(trail))).to(torch.float32)

    sel: list[list[int]] = []
    for r in range(b):
        idx = full[r].nonzero().flatten().tolist()
        if slots_per_row > 0 and len(idx) > slots_per_row:
            step = len(idx) / float(slots_per_row)
            idx = [idx[int(i * step)] for i in range(slots_per_row)]
        sel.append(idx)
    n_calls = max((len(i) for i in sel), default=0)
    if n_calls == 0:
        raise ValueError("no valid slot in this operating point")

    zc = state.view(b, s, m_cells, *trail)
    fams = ("mean", "dev", "realised_mean", "realised_dev")
    # per-slot accumulators, keyed by family -> list over probed slots of the mean gain
    per_slot: dict[str, list[float]] = {f: [] for f in fams}
    per_slot_cv: dict[str, list[float]] = {"mean": [], "dev": []}
    leak: dict[str, list[float]] = {"mean": [], "dev": []}
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    for call in range(n_calls):
        rows = [r for r in range(b) if len(sel[r]) > call]
        slots = {r: sel[r][call] for r in rows}
        for fam in fams:
            n_draw = draws if fam in ("mean", "dev") else 1
            acc = {r: [] for r in rows}
            lk = {r: [] for r in rows}
            for _ in range(n_draw):
                v = torch.zeros_like(state, dtype=torch.float32)
                used: list[int] = []
                for r in rows:
                    sl = slots[r]
                    d = _slot_direction(fam, m_cells, trail, gen, dev, torch.float32,
                                        zc[r, sl])
                    if d is None:
                        continue
                    v[r, sl * m_cells:(sl + 1) * m_cells] = d
                    used.append(r)
                if not used:
                    continue
                jv = jvp(v.to(dt)).float() * cell_mask
                for r in used:
                    sl = slots[r]
                    blk = jv[r, sl * m_cells:(sl + 1) * m_cells]
                    gn = float(blk.reshape(-1).double().norm())
                    acc[r].append(gn)
                    if fam in ("mean", "dev"):
                        tot = float(jv[r].reshape(-1).double().norm())
                        out = max(tot * tot - gn * gn, 0.0) ** 0.5
                        lk[r].append(out / max(tot, 1e-30))
            for r in rows:
                if acc[r]:
                    mu_r = sum(acc[r]) / len(acc[r])
                    per_slot[fam].append(mu_r)
                    # WITHIN-slot spread over draws. A family whose subspace the map
                    # scales UNIFORMLY gives every draw the same gain; a spread means
                    # the map treats directions inside that subspace differently, which
                    # is the only way a contraction can move a rank. `d_eff` below turns
                    # it into an effective direction count.
                    if fam in ("mean", "dev") and len(acc[r]) >= 2 and mu_r > 0:
                        v_r = sum((x - mu_r) ** 2 for x in acc[r]) / (len(acc[r]) - 1)
                        per_slot_cv[fam].append(math.sqrt(v_r) / mu_r)
                if fam in ("mean", "dev") and lk[r]:
                    leak[fam].append(sum(lk[r]) / len(lk[r]))

    # whole-state directions: every valid slot perturbed at once, whole active output read
    glob: dict[str, list[float]] = {"mean": [], "dev": []}
    for fam in ("mean", "dev"):
        for _ in range(max(0, int(global_draws))):
            n = 1 if fam == "mean" else m_cells
            d = torch.randn((b, s, n, *trail), generator=gen).to(device=dev,
                                                                 dtype=torch.float32)
            if fam == "mean":
                d = d.expand(b, s, m_cells, *trail).contiguous()
            else:
                d = d - d.mean(dim=2, keepdim=True)
            v = (d.reshape(b, sm, *trail) * cell_mask)
            vn = float(v.reshape(-1).double().norm())
            if vn < 1e-20:
                continue
            v = v / vn
            jv = jvp(v.to(dt)).float() * cell_mask
            glob[fam].append(float(jv.reshape(-1).double().norm()))

    def agg(xs: list[float]) -> tuple[float, float, int]:
        n = len(xs)
        if n == 0:
            return float("nan"), float("nan"), 0
        mu = sum(xs) / n
        if n < 2:
            return mu, float("nan"), n
        var = sum((x - mu) ** 2 for x in xs) / (n - 1)
        return mu, math.sqrt(var / n), n

    out: dict = {"n_cells": int(active.sum()), "m": int(m_cells), "draws": int(draws),
                 "slots_probed": len(per_slot["dev"])}
    for fam, key in (("mean", "mean_dir"), ("dev", "dev_dir"),
                     ("realised_mean", "realised_mean"),
                     ("realised_dev", "realised_dev")):
        mu, se, n = agg(per_slot[fam])
        out[f"gain_{key}"] = mu
        out[f"se_{key}"] = se
        out[f"n_{key}"] = n
    for fam in ("mean", "dev"):
        mu, se, _ = agg(leak[fam])
        out[f"leak_{fam}_dir"] = mu
        out[f"leak_se_{fam}_dir"] = se
        mu, se, _ = agg(glob[fam])
        out[f"gain_global_{fam}_dir"] = mu
        out[f"se_global_{fam}_dir"] = se
        cv, cv_se, _ = agg(per_slot_cv[fam])
        # the subspace the family draws in: prod(trail) for the stream MEAN, and
        # (K-1) * prod(trail) for the DEVIATION (the deviations sum to zero).
        dsub = int(math.prod(trail)) * (1 if fam == "mean" else (m_cells - 1))
        out[f"cv_{fam}_dir"] = cv
        out[f"cv_se_{fam}_dir"] = cv_se
        out[f"dim_{fam}_dir"] = dsub
        out[f"d_eff_{fam}_dir"] = _d_eff(cv, dsub)
    out["ratio_dev_over_mean"] = (out["gain_dev_dir"] / out["gain_mean_dir"]
                                  if out["gain_mean_dir"] else float("nan"))
    out["ratio_realised_dev_over_mean"] = (
        out["gain_realised_dev"] / out["gain_realised_mean"]
        if out["gain_realised_mean"] else float("nan"))
    out["ratio_global_dev_over_mean"] = (
        out["gain_global_dev_dir"] / out["gain_global_mean_dir"]
        if out["gain_global_mean_dir"] else float("nan"))
    return out


# ── the live-model side ──────────────────────────────────────────────────────────────
def _jvp_closure(fn, h0: torch.Tensor):
    """``v -> J v`` at ``h0`` for ``fn``, from the double-backward identity.

    One forward builds the graph; every direction then costs one backward, exactly as
    ``core_jacobian._jacobian_stats`` power-iterates. fp32 throughout.
    """
    h = h0.detach().float().requires_grad_(True)
    y = fn(h)
    if y.shape != h.shape:
        raise RuntimeError(f"core map is not an endomorphism: in {tuple(h.shape)} "
                           f"out {tuple(y.shape)}")
    y = y.float()
    u = torch.zeros_like(y, requires_grad=True)
    (g,) = torch.autograd.grad(y, h, grad_outputs=u, create_graph=True)

    def jvp(v: torch.Tensor) -> torch.Tensor:
        (jv,) = torch.autograd.grad(g, u, grad_outputs=v.float(), retain_graph=True)
        return jv

    return jvp


def fd_check(step, h0: torch.Tensor, jvp, v: torch.Tensor, eps: float = 1e-3) -> dict:
    """Central finite difference against the exact JVP, on the REAL map.

    The synthetic tests pin :func:`pass_gains`; they cannot pin ``_jvp_closure`` on a
    stack of custom attention modules, and a double backward that silently returns the
    wrong thing (a kernel whose backward is written without a graph is the live hazard
    here — see KERNELS above) would produce a perfectly plausible table. This is the
    check that the operator being reported is the derivative of the map that ran:
    ``(f(h + eps v) - f(h - eps v)) / (2 eps)`` against ``J v``, both in fp32.

    Returns the relative error and both norms. Expect ~1e-3 in fp32 at ``eps = 1e-3``
    relative to the carrier's RMS; a wrong operator reads O(1).
    """
    scale = float(h0.detach().float().abs().mean()) + 1e-12
    e = float(eps) * scale / max(float(v.abs().mean()), 1e-12)
    with torch.no_grad():
        hp = step((h0.detach().float() + e * v).float())
        hm = step((h0.detach().float() - e * v).float())
    num = ((hp - hm).float() / (2.0 * e))
    exact = jvp(v)
    dn = float((num - exact).reshape(-1).double().norm())
    en = float(exact.reshape(-1).double().norm())
    return {"eps": e, "rel_err": dn / max(en, 1e-30), "norm_exact": en,
            "norm_fd": float(num.reshape(-1).double().norm())}


def _print(label: str, t: int, g: dict) -> None:
    print(f"{label} pass={t} slots={g['slots_probed']:3d} cells={g['n_cells']:5d} "
          f"rank={g.get('traj_rank', float('nan')):.3f} "
          f"cos={g.get('traj_cos', float('nan')):+.3f} | "
          f"mean={g['gain_mean_dir']:.4f}±{g['se_mean_dir']:.4f} "
          f"dev={g['gain_dev_dir']:.4f}±{g['se_dev_dir']:.4f} "
          f"dev/mean={g['ratio_dev_over_mean']:.4f} | "
          f"real_mean={g['gain_realised_mean']:.4f}±{g['se_realised_mean']:.4f} "
          f"real_dev={g['gain_realised_dev']:.4f}±{g['se_realised_dev']:.4f} "
          f"r={g['ratio_realised_dev_over_mean']:.4f} | "
          f"glob_mean={g['gain_global_mean_dir']:.4f} "
          f"glob_dev={g['gain_global_dev_dir']:.4f} | "
          f"cv={g.get('cv_mean_dir', float('nan')):.4f}/"
          f"{g.get('cv_dev_dir', float('nan')):.4f} "
          f"d_eff={g.get('d_eff_mean_dir', float('nan')):.0f}/"
          f"{g.get('d_eff_dev_dir', float('nan')):.0f} "
          f"leak={g['leak_mean_dir']:.3f}/{g['leak_dev_dir']:.3f}")




def refuse_unsupported(tc, label: str) -> None:
    """RAISE unless every pre-step injection of ``_tul_core`` is off on this arm.

    ``tul.reread``, ``tul.loop_carry``, ``tul.slot_chain`` and ``tul.grad_pass`` each add a
    term to the carrier INSIDE ``_core_step`` and BEFORE ``_apply_core_step``, so on such
    an arm the method this probe differentiates is not the whole pass. The storage pairing
    in :func:`measure_batches` would catch it anyway (the captured ``h`` is no longer the
    tensor the step received), but a named refusal says which knob did it.
    """
    for key in ("reread", "slot_chain", "grad_pass"):
        if bool(getattr(tc, key, False)):
            raise SystemExit(f"{label}: tul.{key} adds a pre-step injection outside "
                             "_apply_core_step; the probe would differentiate the "
                             "wrong map")
    if str(getattr(tc, "loop_carry", "none")) != "none":
        raise SystemExit(f"{label}: tul.loop_carry re-injects before _apply_core_step")
    if bool(getattr(tc, "tokens_through_core", False)):
        raise SystemExit(f"{label}: the paid loop has no slot-loop core stage to probe")
    if int(getattr(tc, "fan_k", 0)) < 2:
        raise SystemExit(f"{label}: not a fan arm (tul.fan_k < 2)")


def measure_batches(model, batches, m_cells: int, device: str, draws: int,
                    slots_per_row: int, seed: int, label: str,
                    verbose: bool = True, fd_checks: int = 0,
                    fd_eps: float = 1e-3) -> dict[int, list[dict]]:
    """Run the eval forward on every batch and measure every captured operating point.

    Returns ``{iter_idx: [per-batch reading]}``. One ``_jvp_closure`` per operating point,
    so the core step's forward runs once per (batch, pass) and every direction after that
    is one backward.
    """
    root = getattr(model, "_orig_mod", model)
    # The recorder: the EXACT call `_core_step` makes, keyed by the input carrier's
    # storage, so the replayed map carries this arm's own cell relation (`attn_kw`) and
    # stage conditioning rather than the defaults. Pairing on the storage is also the
    # check that nothing was injected between the captured `h` and the step's input.
    real_step = type(root)._apply_core_step
    calls: dict[int, dict] = {}

    def recording_step(self, h_in, e_in, ids, x0_terms, bg, **kw):
        calls[h_in.data_ptr()] = {"e": e_in, "ids": ids, "x0": x0_terms, "bg": bg,
                                  "kw": dict(kw), "shape": tuple(h_in.shape)}
        return real_step(self, h_in, e_in, ids, x0_terms, bg, **kw)

    acc: dict[int, list[dict]] = {}
    t0 = time.time()
    type(root)._apply_core_step = recording_step
    try:
        for bi, (inp, _labels, layout, _idx) in enumerate(batches):
            calls.clear()
            root._jac_capture = points = []
            try:
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16,
                                        enabled=device == "cuda"):
                        model.tul_forward_ablated(inp.to(device), None,
                                                  layout.to(device), plan_mode="normal")
            finally:
                root._jac_capture = None
            if not points:
                raise SystemExit("no operating point captured: the forward did not reach "
                                 "_tul_core's loop")
            for p in points:
                if p["scse"]:
                    raise SystemExit("SCSE: 'h' is the deviation and 'e' the anchor; the "
                                     "mean/deviation split named here is another object")
                rec = calls.get(p["h"].data_ptr())
                if rec is None or rec["shape"] != tuple(p["h"].shape):
                    raise SystemExit(
                        f"pass {p['iter_idx'] + 1}: the captured carrier is not the tensor "
                        "_apply_core_step received — something injects between the capture "
                        "site and the step, and the probe would measure the wrong map")
                t = int(p["iter_idx"])
                e = rec["e"].float() if torch.is_tensor(rec["e"]) else rec["e"]
                kw = dict(rec["kw"])
                for k in ("inj_terms", "ret_state"):
                    if torch.is_tensor(kw.get(k)):
                        kw[k] = kw[k].float()

                def step(h, _e=e, _kw=kw, _rec=rec):
                    out, _ = real_step(root, h, _e, _rec["ids"], _rec["x0"], _rec["bg"],
                                       **_kw)
                    return out

                with torch.autocast("cuda", enabled=False):
                    jvp = _jvp_closure(step, p["h"])
                    if fd_checks > 0:
                        fd_checks -= 1
                        gq = torch.Generator(device="cpu").manual_seed(seed + t)
                        vv = torch.randn(p["h"].shape, generator=gq).to(
                            device=p["h"].device, dtype=torch.float32)
                        vv = vv.reshape(p["h"].shape[0], -1, m_cells,
                                        *p["h"].shape[2:])
                        vv = (vv - vv.mean(dim=2, keepdim=True)).reshape(p["h"].shape)
                        vv = vv * p["active"].view(*p["active"].shape,
                                                   *([1] * (p["h"].dim() - 2)))
                        vv = vv / vv.reshape(-1).double().norm().clamp_min(1e-30).float()
                        fd = fd_check(step, p["h"], jvp, vv, eps=fd_eps)
                        print(f"[{label}] FD-CHECK pass {t + 1}: rel_err="
                              f"{fd['rel_err']:.3e} |Jv|={fd['norm_exact']:.4f} "
                              f"|fd|={fd['norm_fd']:.4f} eps={fd['eps']:.2e}",
                              flush=True)
                    g = pass_gains(jvp, p["h"].detach().float(), p["active"], m_cells,
                                   draws=draws, seed=seed + 1009 * t,
                                   slots_per_row=slots_per_row)
                del jvp
                # The trajectory's OWN stream geometry at this operating point, from the
                # trainer's own statistic, so the gain table can be read against the
                # rank fall it is meant to explain (`fan/stream_rank_t{t}` and
                # `lab/divergence/fan_stream_probe.py`). Over the ACTIVE slots only,
                # which is the set the gain is measured on — the stream probe reads
                # every valid slot, so the two need not match to the last digit.
                from morph.model.tul_fan import _cell_readout, fan_stream_stats
                bb, smm = p["h"].shape[:2]
                ss = smm // m_cells
                z = _cell_readout(p["h"].float().reshape(bb, ss, m_cells,
                                                         *p["h"].shape[2:]))
                ok = p["active"].view(bb, ss, m_cells).all(dim=-1)
                er, cs = fan_stream_stats(z[ok])
                g["traj_rank"] = float(er.mean())
                g["traj_cos"] = float(cs.mean())
                acc.setdefault(t, []).append(g)
            points.clear()
            calls.clear()
            if device == "cuda":
                torch.cuda.empty_cache()
            if verbose:
                print(f"[{label}] batch {bi + 1}/{len(batches)} done "
                      f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        type(root)._apply_core_step = real_step
    return acc


def pool_batches(gs: list[dict], m_cells: int, draws: int) -> dict:
    """Pool one pass's per-batch readings into one row.

    A per-slot mean is already inside each batch's reading, so the pooled figure weights
    batches by their probed-slot count and the pooled SE is recombined from the per-batch
    spreads plus the between-batch spread.
    """
    pooled: dict = {"n_batches": len(gs), "m": int(m_cells), "draws": int(draws),
                    "slots_probed": sum(g["slots_probed"] for g in gs),
                    "n_cells": sum(g["n_cells"] for g in gs)}
    for key in ("mean_dir", "dev_dir", "realised_mean", "realised_dev"):
        n = sum(g[f"n_{key}"] for g in gs)
        mu = sum(g[f"gain_{key}"] * g[f"n_{key}"] for g in gs) / max(n, 1)
        ss = 0.0
        for g in gs:
            nb = g[f"n_{key}"]
            se = g[f"se_{key}"]
            if nb >= 2 and se == se:
                ss += (se ** 2) * nb * (nb - 1)
            ss += nb * (g[f"gain_{key}"] - mu) ** 2
        pooled[f"gain_{key}"] = mu
        pooled[f"se_{key}"] = math.sqrt(ss / (n * (n - 1))) if n >= 2 else float("nan")
        pooled[f"n_{key}"] = n
    for key in ("leak_mean_dir", "leak_dev_dir", "gain_global_mean_dir",
                "gain_global_dev_dir", "cv_mean_dir", "cv_dev_dir",
                "traj_rank", "traj_cos"):
        vals = [g[key] for g in gs if key in g and g[key] == g[key]]
        pooled[key] = sum(vals) / len(vals) if vals else float("nan")
    for fam in ("mean", "dev"):
        dsub = int(gs[0][f"dim_{fam}_dir"])
        pooled[f"dim_{fam}_dir"] = dsub
        pooled[f"d_eff_{fam}_dir"] = _d_eff(pooled[f"cv_{fam}_dir"], dsub)
    for name, num, den in (("ratio_dev_over_mean", "gain_dev_dir", "gain_mean_dir"),
                           ("ratio_realised_dev_over_mean", "gain_realised_dev",
                            "gain_realised_mean"),
                           ("ratio_global_dev_over_mean", "gain_global_dev_dir",
                            "gain_global_mean_dir")):
        pooled[name] = pooled[num] / pooled[den] if pooled[den] else float("nan")
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="NAME=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=24)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6, help="forced slot depth, as the sweep")
    ap.add_argument("--draws", type=int, default=16, help="random directions per slot")
    ap.add_argument("--slots-per-row", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fd-checks", type=int, default=0,
                    help="verify the JVP against a central finite difference on the "
                         "real map, on the first N operating points")
    ap.add_argument("--fd-eps", type=float, default=1e-3,
                    help="finite-difference step, as a fraction of the carrier's scale")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    # `model.tg_scoped_kernels=false` is NOT a taste choice: the fused HC-Cayley / CCA /
    # window path is a custom autograd.Function whose backward is built without a graph,
    # so `d g / d u` does not exist and the double-backward JVP raises "One of the
    # differentiated Tensors appears to not have been used in the graph" (measured on
    # fan4-all, 2026-09-21). Every reading below is therefore on the EAGER map. The
    # 2026-09-08 audit's "eager reads the gain 0.07 high" does NOT transfer: that is a
    # finite-difference noise term ((bf16 noise / eps)^2) and this probe takes no finite
    # difference — it differentiates exactly, in fp32. What DOES differ is the captured
    # trajectory, by fp32/bf16 rounding against the fused forward the stream probe ran.
    cfg = build_cfg(config, ["model.use_kernels=false",
                             "model.tg_scoped_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit(f"{label}: not a TUL arm")
    refuse_unsupported(tul_rt.model_cfg, label)
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    m_cells = int(tc.fan_k)
    # The SAME forced depth the stream-rank probe uses (`fan_stream_probe.py`), so the two
    # probes read the same trajectory on the same rows.
    tc.slot_mean_depth = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth))
    if int(getattr(tc, "slot_depth_fixed", 0)):
        tc.slot_depth_fixed = a.depth

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
    if not batches:
        raise SystemExit("no packed batch: raise --rows")

    t_start = time.time()
    acc = measure_batches(model, batches, m_cells, a.device, a.draws, a.slots_per_row,
                          a.seed, label, fd_checks=a.fd_checks, fd_eps=a.fd_eps)
    result = {"arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
              "batch": a.batch, "depth": a.depth, "fan_k": m_cells, "draws": a.draws,
              "slots_per_row": a.slots_per_row, "seed": a.seed, "passes": {}}
    for t in sorted(acc):
        pooled = pool_batches(acc[t], m_cells, a.draws)
        result["passes"][str(t + 1)] = pooled
        _print(label, t + 1, pooled)
    result["wall_s"] = time.time() - t_start
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {a.out} in {result['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
