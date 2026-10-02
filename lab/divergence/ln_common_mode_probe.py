"""The LayerNorm common-mode probe: is the latent target one shared direction, and what
does that do to the latent-selected loop's teacher, its pull and our cell instruments?

Why (Wolfe 2026-10-02, "chase down the layer norm hypothesis"): the latent target of every
latent arm since 2026-09-30 is the EMA prelude twin's output, mean-pooled over the NEXT
span's tokens and LayerNormed with no affine (``pooled_span_states``). A stage-1 build
read about 98 % of it as one shared direction. This probe measures that on the trained
checkpoints and tests four claims (note
``.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md``):

  A  channels    per-channel mean / std of the prelude output (live front, EMA twin, plain
                 model) at token positions; the pooled LN target's cosine, centred PR, the
                 share of its centred variance in the top-|mean| channels.
  H1 tax         does the exit loss pull every cell toward ONE constant vector? Reads the
                 residual e = g(c_win) - z (its shared part and its part along the target's
                 mean direction u) and the gradient of the exit loss on the cells (its shared
                 part across slots).
  H2 teacher     the teacher's argmin under the shipped distance vs a standardised and a
                 u-removed distance, offline on the same cells; with ``--follow`` the coda CE
                 when the loop follows each teacher (ledger machinery, paired bootstrap).
  H3 instruments within-slot cosine / rank / spread, raw vs globally centred.
  P  puzzle      per pass: the cells' split into the shared direction and the rest, the
                 step size between passes, and a ridge probe of the standardised target.
  V  --vablate   (a separate call) Sun et al.'s mean-vs-zero ablation of the written
                 cells' shared direction, at each forced depth: is it a bias or content?

DISTANCE ALGEBRA (why "centre z" is not enough for H2): ``argmin_i |g_i - z|^2`` is unchanged
by subtracting ONE vector from both ``g_i`` and ``z``. The common mode can steer the pick
only through the cells' own disagreement along it, or through the per-coordinate scale.
So the variants are: ``zstd`` (each coordinate over z's std), ``own`` (g and z each
standardised by their own fixed stats), ``perp`` (the component along u removed).

Rows: the first ``--rows`` packed val rows are the EVAL rows (the ledger's row source);
the next ``--fit_rows`` are the FIT rows (every fixed statistic: mu, sigma, u, the ridge
probe). All arms read the same packed rows (``--pack_config``); the plain model reads the
same token stream with the slot positions removed.

Usage (GPU, one checkpoint per call keeps each GPU hold short):
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock /home/wolfe/morph-scratch/gpu.lock \\
    python lab/divergence/ln_common_mode_probe.py --ckpt LABEL=CONFIG=PATH --out OUT.json \\
    [--follow] [--rows 96 --fit_rows 96 --batch 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import ROOT, build_cfg, parse_ckpt_spec  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402

TOPK = 10
N_MASSIVE = 4


# ── small math ───────────────────────────────────────────────────────────────────────


def _ln(x: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],))


def pool_raw(x: torch.Tensor, gid: torch.Tensor, keep_tok: torch.Tensor, g_bins: int):
    """``pooled_span_states`` WITHOUT the LayerNorm: ``[B, S, C]`` span means (bin s+1)."""
    xs = x.float()
    while xs.dim() > 3:
        xs = xs.mean(dim=-2)
    B, L, C = xs.shape
    w = keep_tok.reshape(-1, 1).to(xs.dtype)
    sums = xs.new_zeros(B * g_bins, C).index_add_(0, gid.reshape(-1), xs.reshape(-1, C) * w)
    cnt = xs.new_zeros(B * g_bins).index_add_(0, gid.reshape(-1),
                                              keep_tok.reshape(-1).to(xs.dtype))
    return (sums / cnt.clamp_min(1.0).unsqueeze(-1)).view(B, g_bins, C)[:, 1:]


class ChanAcc:
    """Per-channel running sum / sum of squares / count over token rows, fp64."""

    def __init__(self):
        self.s = self.ss = None
        self.n = 0

    def add(self, rows: torch.Tensor) -> None:
        r = rows.double()
        if self.s is None:
            self.s = torch.zeros(r.shape[-1], dtype=torch.float64, device=r.device)
            self.ss = torch.zeros_like(self.s)
        self.s += r.sum(0)
        self.ss += r.square().sum(0)
        self.n += int(r.shape[0])

    def summary(self) -> dict:
        mean = self.s / self.n
        ms = self.ss / self.n
        std = (ms - mean.square()).clamp_min(0).sqrt()
        top = mean.abs().argsort(descending=True)[:TOPK]
        e2 = ms.sum()
        return {
            "n_tokens": self.n,
            "top_abs_mean": [{"ch": int(c), "mean": float(mean[c]), "std": float(std[c])}
                             for c in top],
            "median_abs_mean": float(mean.abs().median()),
            "median_std": float(std.median()),
            "max_abs_mean_over_median_abs_mean": float(mean.abs().max()
                                                       / mean.abs().median().clamp_min(1e-12)),
            # share of E|x|^2 carried by the top-N_MASSIVE |mean| channels
            "energy_share_top4": float(ms[top[:N_MASSIVE]].sum() / e2),
            "mean_vec_energy_share": float(mean.square().sum() / e2),
        }


def mean_pairwise_cos(z: torch.Tensor) -> float:
    """Exact mean cosine over distinct pairs of the rows of ``z`` ``[N, C]``."""
    u = F.normalize(z.double(), dim=-1)
    n = u.shape[0]
    s = u.sum(0)
    return float((s.dot(s) - n) / (n * (n - 1)))


def pr_cov(z: torch.Tensor) -> float:
    """Participation ratio of the centred covariance (``(sum l)^2 / sum l^2``)."""
    zc = z.double() - z.double().mean(0, keepdim=True)
    cov = zc.t() @ zc / max(zc.shape[0] - 1, 1)
    return float(torch.trace(cov) ** 2 / cov.square().sum())


def target_stats(z: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> dict:
    """``z`` ``[N, C]`` LN targets of the ok slots; ``mu``/``sd`` fixed (fit rows)."""
    zd = z.double()
    m = zd.mean(0)
    var = zd.var(0, unbiased=False)
    top = m.abs().argsort(descending=True)[:TOPK]
    u = F.normalize(m, dim=0)
    zs = (zd - mu.double()) / sd.double()
    return {
        "n": int(z.shape[0]),
        "mean_pairwise_cos": mean_pairwise_cos(zd),
        "mean_cos_to_mean_dir": float(F.normalize(zd, dim=-1).mv(u).mean()),
        "mean_vec_share_of_energy": float(m.square().sum() / zd.square().sum(1).mean()),
        "pr_centred": pr_cov(zd),
        "pr_standardised_fitstats": pr_cov(zs),
        "per_coord_var_mean": float(var.mean()),
        "per_coord_var_median": float(var.median()),
        "top_abs_mean": [{"ch": int(c), "mean": float(m[c]), "std": float(var[c].sqrt())}
                         for c in top],
        "centred_var_share_top4": float(var[top[:N_MASSIVE]].sum() / var.sum()),
        "centred_var_share_top10": float(var[top].sum() / var.sum()),
        # the u direction carries what share of the CENTRED variance (amplitude jitter
        # of the common mode)
        "centred_var_share_along_u": float(((zd - m) @ u).var(unbiased=False) / var.sum()),
    }


def ridge_r2(xf, yf, xe, ye, lams=(1e-1, 1e0, 1e1, 1e2, 1e3)) -> dict:
    """Ridge from ``x`` to standardised ``y``: lambda picked on a split of the FIT rows,
    then refit on all fit rows and scored on the eval rows. R^2 = 1 - SSE/SST per
    coordinate, averaged (y standardised, so every coordinate weighs the same)."""
    def _fit(x, y, lam):
        xm, ym = x.mean(0, keepdim=True), y.mean(0, keepdim=True)
        xc = x - xm
        a = xc.t() @ xc + lam * x.shape[0] * torch.eye(x.shape[1], dtype=x.dtype,
                                                       device=x.device)
        w = torch.linalg.solve(a, xc.t() @ (y - ym))
        return xm, ym, w

    def _r2(x, y, p):
        xm, ym, w = p
        pred = (x - xm) @ w + ym
        sse = (y - pred).square().sum(0)
        sst = (y - y.mean(0, keepdim=True)).square().sum(0).clamp_min(1e-12)
        return float((1 - sse / sst).mean())
    xf, yf, xe, ye = (t.double() for t in (xf, yf, xe, ye))
    sx = xf.std(0, keepdim=True).clamp_min(1e-6)
    mx = xf.mean(0, keepdim=True)
    xf, xe = (xf - mx) / sx, (xe - mx) / sx
    h = xf.shape[0] // 2
    sel = {lam: _r2(xf[h:], yf[h:], _fit(xf[:h], yf[:h], lam)) for lam in lams}
    best = max(sel, key=sel.get)
    return {"r2_eval": _r2(xe, ye, _fit(xf, yf, best)), "lambda": best,
            "r2_fit_half_by_lambda": {str(k): v for k, v in sel.items()}}


# ── capture ──────────────────────────────────────────────────────────────────────────


class Capture:
    """Instance-level patches that record what one forward computed, removed after."""

    def __init__(self, model, T_mod):
        self.model, self.T = model, T_mod
        self.front_live: list = []
        self.front_twin: list = []
        self.pooled: list = []
        self.pp_cells: list = []
        self._old_pool = None

    def __enter__(self):
        m = self.model
        orig_ft = m._front_tail

        def _ft(*a, **k):
            x, x0 = orig_ft(*a, **k)
            (self.front_twin if k.get("twin") is not None else self.front_live).append(
                x.detach())
            return x, x0
        m.__dict__["_front_tail"] = _ft
        self._old_pool = self.T.pooled_span_states

        def _pool(x, gid, keep_tok, g_bins):
            z = self._old_pool(x, gid, keep_tok, g_bins)
            self.pooled.append((x.detach(), z.detach()))
            return z
        self.T.pooled_span_states = _pool
        if getattr(m, "tul", None) is not None:
            orig_pp = m.tul.prefix_project

            def _pp(h_slots, layout, l_total, cells=None):
                if cells is not None:
                    self.pp_cells.append(cells.detach())
                return orig_pp(h_slots, layout, l_total, cells=cells)
            m.tul.__dict__["prefix_project"] = _pp
        if getattr(m, "_lsel_mode", "off") != "off":
            m._lsel_capture = []
        return self

    def __exit__(self, *exc):
        m = self.model
        m.__dict__.pop("_front_tail", None)
        self.T.pooled_span_states = self._old_pool
        if getattr(m, "tul", None) is not None:
            m.tul.__dict__.pop("prefix_project", None)
        self.lsel = m._lsel_capture if getattr(m, "_lsel_mode", "off") != "off" else None
        if self.lsel is not None:
            m._lsel_capture = None
        return False


def _cells4(h: torch.Tensor, B: int, S: int, m: int) -> torch.Tensor:
    """carrier ``[B, S*M, (n,) C]`` or ``[B, S, M, (n,) C]`` -> ``[B, S, M, C]`` fp32."""
    # the slot axis decides the layout (the HC stream count n can equal M, so the third
    # axis cannot): S*M = the compact cell axis, S = already [B, S, M, ...]
    if h.shape[1] == S * m:
        h = h.view(B, S, m, *h.shape[2:])
    elif h.shape[1] != S or h.shape[2] != m:
        raise ValueError(f"cells {tuple(h.shape)} fit neither [B, S*M, ...] nor [B, S, M, ...]"
                         f" (S={S}, M={m})")
    while h.dim() > 4:
        h = h.mean(dim=-2)
    return h.float()


# ── one arm ──────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def collect(model, batches, device, kind: str, depth: int | None = None) -> dict:
    """One labelled eval forward per batch. Returns token-level channel accumulators and
    per-valid-slot tensors on the CPU: ``z_twin``, ``z_online`` (LN), ``raw_twin`` (pre-LN
    pooled), ``ok``, ``row``, and for loop arms ``cells[t]`` ``[N, M, C]`` (fp16),
    ``router[t]`` ``[N]``. ``kind``: "plain" | "fan" | "lsel"."""
    import morph.model.transformer as T
    from exploration_ledger import ledger_patch
    from morph.model.transformer import span_ce_index, span_token_counts

    acc_live, acc_twin = ChanAcc(), ChanAcc()
    out: dict = {"z_twin": [], "z_online": [], "raw_twin": [], "z_plain": [], "ok": [],
                 "row": [], "cells": {}, "router": {}, "post": {}}
    r0 = 0
    m = int(model.cfg.tul.fan_k) if kind != "plain" else 0
    for bi, (inp, labels, layout, _) in enumerate(batches):
        lay = layout.to(device)
        lab = labels.to(device)
        B, S = lay.slot_valid.shape
        gid, keep_tok, _, g_bins = span_ce_index(lab, lay)
        n_tok = span_token_counts(gid, keep_tok, g_bins)[:, 1:]
        ok = lay.slot_valid & (n_tok > 0)
        valid = lay.slot_valid
        if kind == "plain":
            # token-only rows: the same stream with the slot positions removed, right
            # padded (the plain model is causal, so a right pad never reaches a real
            # position). Its prelude output is scattered back to the packed positions and
            # pooled with the packed row's own span bins.
            tokpos = ~lay.slot_mask
            lens = tokpos.sum(1)
            Lp = int(lens.max())
            ids = torch.zeros(B, Lp, dtype=inp.dtype, device=device)
            for b in range(B):
                ids[b, :int(lens[b])] = inp[b].to(device)[tokpos[b]]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                x, _, _ = model._front_region(ids)
            xs = x.float()
            while xs.dim() > 3:
                xs = xs.mean(dim=-2)
            full = xs.new_zeros(B, inp.shape[1], xs.shape[-1])
            for b in range(B):
                full[b, tokpos[b]] = xs[b, :int(lens[b])]
                acc_live.add(xs[b, :int(lens[b])][(lab[b][tokpos[b]] >= 0)])
            out["z_plain"].append(_ln(pool_raw(full, gid, keep_tok, g_bins))[valid].cpu())
        else:
            kw = {}
            if depth is not None:
                # the depth table is CELL-level, [B, S*M]
                kw["slot_depths"] = torch.full((B, S * m), int(depth), dtype=torch.long)
            with ledger_patch(model), Capture(model, T) as cap:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    model.tul_forward_ablated(inp.to(device), lab, lay, plan_mode="normal",
                                              **kw)
            if len(cap.front_live) != 1:
                raise RuntimeError(f"{len(cap.front_live)} live front calls (expected 1)")
            xl = cap.front_live[0].float()
            while xl.dim() > 3:
                xl = xl.mean(dim=-2)
            acc_live.add(xl[keep_tok])
            if cap.front_twin:
                xt = cap.front_twin[0].float()
                while xt.dim() > 3:
                    xt = xt.mean(dim=-2)
                acc_twin.add(xt[keep_tok])
                if len(cap.pooled) < 1:
                    raise RuntimeError("the twin ran but nothing was pooled")
                x_t, z_t = cap.pooled[0]
                raw = pool_raw(x_t, gid, keep_tok, g_bins)
                dev = (_ln(raw) - z_t).abs().max()
                if float(dev) > 1e-3:
                    raise RuntimeError(f"pool_raw + LN does not reproduce the target: {dev}")
                out["z_twin"].append(z_t[valid].float().cpu())
                out["raw_twin"].append(raw[valid].float().cpu())
                if len(cap.pooled) > 1:
                    out["z_online"].append(cap.pooled[1][1][valid].float().cpu())
            if kind == "lsel":
                for e in cap.lsel:
                    t = int(e["t"])
                    out["cells"].setdefault(t, []).append(
                        _cells4(e["pre"], B, S, m)[valid].half().cpu())
                    out["post"].setdefault(t, []).append(
                        _cells4(e["post"], B, S, m)[valid].half().cpu())
                    out["router"].setdefault(t, []).append(e["router"][valid].cpu())
            elif kind == "fan":
                if len(cap.pp_cells) != 1:
                    raise RuntimeError(f"{len(cap.pp_cells)} cell writes (expected 1)")
                out["cells"].setdefault(0, []).append(
                    _cells4(cap.pp_cells[0], B, S, m)[valid].half().cpu())
        out["ok"].append(ok[valid].cpu())
        out["row"].append((torch.arange(B, device=device).view(B, 1).expand(B, S) + r0)[valid]
                          .cpu())
        r0 += B
    cat = lambda xs: torch.cat(xs) if xs else None  # noqa: E731
    res = {k: cat(out[k]) for k in ("z_twin", "z_online", "raw_twin", "z_plain", "ok", "row")}
    res["cells"] = {t: torch.cat(v) for t, v in out["cells"].items()}
    res["post"] = {t: torch.cat(v) for t, v in out["post"].items()}
    res["router"] = {t: torch.cat(v) for t, v in out["router"].items()}
    res["chan_live"] = acc_live.summary() if acc_live.n else None
    res["chan_twin"] = acc_twin.summary() if acc_twin.n else None
    return res


# ── the readings ─────────────────────────────────────────────────────────────────────


def cell_instruments(c: torch.Tensor, mu_global: torch.Tensor) -> dict:
    """H3 on ``c`` ``[N, M, C]``: within-slot cosine / rank and cell spread, raw vs with
    the GLOBAL per-coordinate mean ``mu_global`` (all cells of the eval rows) removed."""
    from morph.model.tul_fan import fan_stream_stats
    from morph.model.tul_fan_route import cell_spread
    res = {}
    for name, x in (("raw", c), ("centred", c - mu_global)):
        er, cos = fan_stream_stats(x.double())
        sp = cell_spread(x.unsqueeze(0), torch.ones(1, x.shape[0], dtype=torch.bool,
                                                    device=x.device))
        slot = x.mean(1)
        res[name] = {"within_slot_cos": float(cos.mean()), "within_slot_rank": float(er.mean()),
                     "cell_spread": float(sp), "across_slot_cos": mean_pairwise_cos(slot),
                     "across_slot_pr": pr_cov(slot)}
    return res


def cell_geometry(c: torch.Tensor, v: torch.Tensor) -> dict:
    """The split of cells ``[N, M, C]`` along the unit common direction ``v``."""
    cd = c.double()
    a = cd @ v.double()                                         # [N, M]
    perp = cd - a.unsqueeze(-1) * v.double()
    mu = cd.reshape(-1, cd.shape[-1]).mean(0)
    cen = cd - mu
    slot = cen.mean(1, keepdim=True)
    rms = lambda t: float(t.square().sum(-1).mean().sqrt())  # noqa: E731
    return {"cos_to_v": float(F.normalize(cd, dim=-1).matmul(v.double()).mean()),
            "rms_norm": rms(cd), "rms_along_v": float(a.square().mean().sqrt()),
            "rms_perp_v": rms(perp), "rms_centred": rms(cen),
            "rms_between_slot": rms(slot.expand_as(cen)), "rms_within_slot": rms(cen - slot)}


def coherence(x: torch.Tensor) -> float:
    """``|mean_n x_n|^2 / mean_n |x_n|^2`` over rows of ``x`` ``[N, C]``: 1 = every row is
    the same vector (one shared push), ~1/N = independent rows."""
    xd = x.double()
    return float(xd.mean(0).square().sum() / xd.square().sum(-1).mean())


def lsel_readings(model, d: dict, f: dict, device, eps: float) -> dict:
    """H1, H2 (offline picks) and the per-pass puzzle readings on a latent-selected arm.
    ``d`` = eval collection, ``f`` = fit collection."""
    head = model.tul_fan_lsel_head
    T = max(d["cells"]) + 1
    ok_e = d["ok"].to(device)
    ok_f = f["ok"].to(device)
    ze = d["z_twin"].to(device)
    zf = f["z_twin"].to(device)
    mu_z = zf[ok_f].mean(0)
    sd_z = zf[ok_f].std(0).clamp_min(1e-6)
    u = F.normalize(mu_z, dim=0)
    exit_f = f["cells"][T - 1].to(device).float()
    with torch.no_grad():
        gf = head(exit_f)[ok_f]
    mu_g = gf.reshape(-1, gf.shape[-1]).mean(0)
    sd_g = gf.reshape(-1, gf.shape[-1]).std(0).clamp_min(1e-6)

    def dists(gz, z):
        e = gz - z.unsqueeze(1)
        eu = e @ u
        ship = e.square().mean(-1)
        along = eu.square() / e.shape[-1]
        return {"ship": ship, "zstd": (e / sd_z).square().mean(-1),
                "own": ((gz - mu_g) / sd_g - ((z - mu_z) / sd_z).unsqueeze(1)
                        ).square().mean(-1),
                "perp": ship - along, "along_u": along}

    res: dict = {"passes": T, "per_pass": {}}
    # every fixed stat the CE readings need, kept for --follow
    res["_stats"] = {"mu_z": mu_z, "sd_z": sd_z, "u": u, "mu_g": mu_g, "sd_g": sd_g}
    # the cells' common direction (eval cells, all passes) and the ridge target
    allc = torch.cat([d["cells"][t].float().reshape(-1, ze.shape[-1]) for t in range(T)])
    v = F.normalize(allc.to(device).mean(0), dim=0)
    del allc
    ys_e = ((ze - mu_z) / sd_z)[ok_e]
    ys_f = ((zf - mu_z) / sd_z)[ok_f]
    prev = None
    for t in range(T):
        c = d["cells"][t].to(device).float()
        ce = c[ok_e]
        with torch.no_grad():
            gz = head(ce)
        ds = dists(gz, ze[ok_e])
        pk = {k: v_.argmin(-1) for k, v_ in ds.items()}
        rt = d["router"][t].to(device)[ok_e]
        dv = {k: ds[k] - ds[k].mean(-1, keepdim=True) for k in ("along_u", "perp")}
        va, vp = dv["along_u"].square().mean(-1), dv["perp"].square().mean(-1)
        win = rt
        gw = gz.gather(1, win.view(-1, 1, 1).expand(-1, 1, gz.shape[-1])).squeeze(1)
        zt = ze[ok_e]
        r2_ln = 1 - (gw - zt).square().sum() / (zt - zt.mean(0)).square().sum()
        gws, zts = (gw - mu_z) / sd_z, (zt - mu_z) / sd_z
        r2_std = 1 - (gws - zts).square().sum() / (zts - zts.mean(0)).square().sum()
        cf = f["cells"][t].to(device).float()[ok_f]
        rf = f["router"][t].to(device)[ok_f]
        cwf = cf.gather(1, rf.view(-1, 1, 1).expand(-1, 1, cf.shape[-1])).squeeze(1)
        cwe = ce.gather(1, win.view(-1, 1, 1).expand(-1, 1, ce.shape[-1])).squeeze(1)
        pp = {
            "agree_ship_vs": {k: float((pk[k] == pk["ship"]).float().mean())
                              for k in ("zstd", "own", "perp", "along_u")},
            "agree_router_vs": {k: float((pk[k] == rt).float().mean())
                                for k in ("ship", "zstd", "own", "perp", "along_u")},
            # per slot: share of the between-cell variance of the distance that sits
            # along u (the common mode's part of the ranking), median and mean
            "between_cell_var_share_along_u_median": float((va / (va + vp).clamp_min(1e-20))
                                                           .median()),
            "between_cell_var_share_along_u_mean": float((va / (va + vp).clamp_min(1e-20))
                                                         .mean()),
            "head_r2_ln_router_winner": float(r2_ln),
            "head_r2_std_router_winner": float(r2_std),
            "ridge_std_target_from_router_winner": ridge_r2(cwf, ys_f, cwe, ys_e),
            "geometry": cell_geometry(c, v),
        }
        if prev is not None:
            # the step the pass took: this pass's candidates minus the state it started
            # from (the previous pass's reset winner), along v and across slots
            step = (c - prev).reshape(-1, c.shape[-1]).double()
            base = prev.reshape(-1, c.shape[-1]).double()
            sa = step @ v.double()
            pp["step"] = {
                "rel_rms": float(step.square().sum(-1).mean().sqrt()
                                 / base.square().sum(-1).mean().sqrt()),
                "share_along_v": float(sa.square().mean() / step.square().sum(-1).mean()),
                "coherence_across_cells": coherence(step),
            }
        prev = d["post"][t].to(device).float() if t in d["post"] else c
        res["per_pass"][t] = pp
    # H1: the exit residual and the exit loss's gradient on the cells
    c = d["cells"][T - 1].to(device).float()[ok_e].requires_grad_(True)
    zt = ze[ok_e]
    with torch.enable_grad():
        gz = head(c)
        dist = (gz - zt.unsqueeze(1)).square().mean(-1)
        tex = dist.detach().argmin(-1)
        from morph.model.tul_fan_route import lsel_exit_loss
        loss = lsel_exit_loss(dist.unsqueeze(0), tex.unsqueeze(0),
                              torch.ones(1, c.shape[0], dtype=torch.bool, device=device), eps)
        (grad,) = torch.autograd.grad(loss, c)
    gw = gz.detach().gather(1, tex.view(-1, 1, 1).expand(-1, 1, gz.shape[-1])).squeeze(1)
    e = gw - zt
    gwin = grad.gather(1, tex.view(-1, 1, 1).expand(-1, 1, grad.shape[-1])).squeeze(1)
    res["h1"] = {
        "head_mean_vs_target_mean_rel": float((gw.mean(0) - zt.mean(0)).norm()
                                              / zt.mean(0).norm()),
        "head_mean_cos_target_mean": float(F.cosine_similarity(gw.mean(0), zt.mean(0), dim=0)),
        "residual_coherence": coherence(e),
        "residual_share_along_u": float((e @ u).square().mean() / e.square().sum(-1).mean()),
        "residual_mse": float(e.square().mean()),
        "target_var_per_coord": float(zt.var(0, unbiased=False).mean()),
        "grad_coherence_winner": coherence(gwin),
        "grad_share_along_v": float((gwin.double() @ v.double()).square().mean()
                                    / gwin.double().square().sum(-1).mean()),
        "grad_rms_winner": float(gwin.square().sum(-1).mean().sqrt()),
    }
    return res


def follow_readings(model, batches, device, st: dict, seed: int = 0) -> dict:
    """H2 on the CE: the loop follows the latent teacher under each distance at EVERY pass
    and the exit (``lsel_follow="teacher"``, with ``lsel_distance`` swapped in the
    transformer's namespace), paired token by token against the shipped router path."""
    import morph.model.transformer as T
    from exploration_ledger import ce_map, delta_ci, ledger_patch
    mu_z, sd_z, u, mu_g, sd_g = (st[k] for k in ("mu_z", "sd_z", "u", "mu_g", "sd_g"))
    orig = T.lsel_distance
    variants = {
        "ship": orig,
        "zstd": lambda gz, z: ((gz.float() - z.float().unsqueeze(2)) / sd_z).square().mean(-1),
        "own": lambda gz, z: (((gz.float() - mu_g) / sd_g)
                              - ((z.float() - mu_z) / sd_z).unsqueeze(2)).square().mean(-1),
        "perp": lambda gz, z: _perp(gz.float() - z.float().unsqueeze(2), u),
    }
    toks: dict = {}
    rows, tm = [], []
    r0 = 0
    for inp, labels, layout, _ in batches:
        tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
        rows.append((torch.arange(inp.shape[0]).view(-1, 1).expand_as(tokpos) + r0)[tokpos]
                    .numpy())
        tm.append(tokpos)
        r0 += inp.shape[0]
    row = np.concatenate(rows)
    plan = [("router", None, None)] + [(f"teacher_{k}", "teacher", k) for k in variants]
    for name, follow, var in plan:
        ces = []
        try:
            if var is not None:
                T.lsel_distance = variants[var]
            for (inp, labels, layout, _), tp in zip(batches, tm):
                with ledger_patch(model):
                    ce, _ = ce_map(model, inp, labels, layout.to(device), device,
                                   lsel_follow=follow)
                ces.append(ce[tp].numpy().astype(np.float64))
        finally:
            T.lsel_distance = orig
        toks[name] = np.concatenate(ces)
    res = {"rows": r0, "n_tokens": int(row.size),
           "ce": {k: float(v.mean()) for k, v in toks.items()}}
    res["delta_vs_router"] = {k: delta_ci(v, toks["router"], row, r0, seed=seed)
                              for k, v in toks.items() if k != "router"}
    res["delta_vs_teacher_ship"] = {k: delta_ci(v, toks["teacher_ship"], row, r0, seed=seed)
                                    for k, v in toks.items()
                                    if k not in ("router", "teacher_ship")}
    return res


def vablate_readings(model, batches, device, kind: str, depths: list[int],
                     seed: int = 0) -> dict:
    """Does the CODA read the cells' common direction as content or as a fixed bias?
    (Sun et al. 2024's test, applied to the written slot cells.)

    At each forced depth k: a shipped forward captures the WRITTEN cells (the
    ``prefix_project(cells=...)`` argument: the winner on a selecting arm, every cell on a
    write-all arm); ``v`` = the unit mean of the nonzero written cells and ``a_bar`` their
    mean amplitude along it. Then two forwards with the write patched: ``mean`` sets every
    nonzero cell's amplitude along ``v`` to ``a_bar`` (the per-input jitter of the common
    mode removed, the bias kept); ``zero`` removes the component along ``v``. Deltas are
    paired per token against the shipped forward at the same depth. ``mean`` ~ 0 and
    ``zero`` >> 0: the common mode is a bias the coda relies on, not content."""
    from exploration_ledger import ce_map, delta_ci, ledger_patch
    m = int(model.cfg.tul.fan_k)
    rows, tm = [], []
    r0 = 0
    for inp, labels, layout, _ in batches:
        tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
        rows.append((torch.arange(inp.shape[0]).view(-1, 1).expand_as(tokpos) + r0)[tokpos]
                    .numpy())
        tm.append(tokpos)
        r0 += inp.shape[0]
    row = np.concatenate(rows)
    orig_fwd = model.tul_forward_ablated
    orig_pp = model.tul.prefix_project
    out: dict = {"rows": r0, "n_tokens": int(row.size), "by_depth": {}}
    for k in depths:
        def _fwd(*a, _k=k, **kw):
            B, S = a[2].slot_valid.shape
            kw["slot_depths"] = torch.full((B, S * m), int(_k), dtype=torch.long)
            return orig_fwd(*a, **kw)
        stats = {"sum": None, "n": 0}
        mode = {"m": "capture"}

        def _pp(h_slots, layout, l_total, cells=None):
            if cells is None:
                raise RuntimeError("vablate: the write carried no cells")
            cr = cells.float()
            while cr.dim() > 4:      # the HC stream axis: v lives in the stream-mean space
                cr = cr.mean(dim=-2)
            nz = cr.abs().sum(-1) > 0                                   # [B, S, M]
            if mode["m"] == "capture":
                s = cr[nz].double().sum(0)
                stats["sum"] = s if stats["sum"] is None else stats["sum"] + s
                stats["n"] += int(nz.sum())
                stats.setdefault("amp", []).append(cr[nz].double())
                return orig_pp(h_slots, layout, l_total, cells=cells)
            v, a_bar = stats["v"], stats["a_bar"]
            a = cr @ v                                                  # [B, S, M]
            new_a = torch.full_like(a, a_bar) if mode["m"] == "mean" else torch.zeros_like(a)
            shift = ((new_a - a) * nz.to(a.dtype)).unsqueeze(-1) * v    # [B, S, M, C]
            while shift.dim() < cells.dim():
                shift = shift.unsqueeze(-2)          # the same shift on every HC stream
            return orig_pp(h_slots, layout, l_total, cells=(cells.float() + shift).to(cells.dtype))
        model.__dict__["tul_forward_ablated"] = _fwd
        model.tul.__dict__["prefix_project"] = _pp
        toks: dict = {}
        try:
            for name in ("shipped", "mean", "zero"):
                mode["m"] = "capture" if name == "shipped" else name
                if name == "mean":
                    allc = torch.cat(stats.pop("amp"))
                    stats["v"] = F.normalize(stats["sum"] / stats["n"], dim=0).float()
                    stats["a_bar"] = float((allc @ stats["v"].double()).mean())
                    stats["cos_to_v"] = float(F.normalize(allc, dim=-1).mv(
                        stats["v"].double()).mean())
                    stats["amp_std"] = float((allc @ stats["v"].double()).std())
                    stats["rms_norm"] = float(allc.square().sum(-1).mean().sqrt())
                    del allc
                ces = []
                for (inp, labels, layout, _), tp in zip(batches, tm):
                    with ledger_patch(model):
                        ce, _ = ce_map(model, inp, labels, layout.to(device), device)
                    ces.append(ce[tp].numpy().astype(np.float64))
                toks[name] = np.concatenate(ces)
        finally:
            model.__dict__.pop("tul_forward_ablated", None)
            model.tul.__dict__.pop("prefix_project", None)
        out["by_depth"][k] = {
            "ce_shipped": float(toks["shipped"].mean()),
            "written_cells": {x: stats[x] for x in ("a_bar", "amp_std", "cos_to_v", "rms_norm")},
            "mean_ablate": delta_ci(toks["mean"], toks["shipped"], row, r0, seed=seed),
            "zero_ablate": delta_ci(toks["zero"], toks["shipped"], row, r0, seed=seed),
        }
        if k == depths[0]:
            ref = toks
        else:
            out["by_depth"][k]["k_minus_first"] = {
                nm: delta_ci(toks[nm], ref[nm], row, r0, seed=seed) for nm in toks}
    return out


@torch.no_grad()
def _h3_readout(model, c: torch.Tensor) -> dict:
    """H3 in the space the logged ``slot_cell_*`` instruments read: each cell through
    ``lm_mixer`` then ``final_norm`` (``_readout`` after its HC stream mean, which the
    cells here have already had), fp32."""
    r = model.final_norm(model.lm_mixer(c.float())).float()
    return cell_instruments(r, r.reshape(-1, r.shape[-1]).mean(0))


def _perp(e: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    return (e - (e @ u).unsqueeze(-1) * u).square().mean(-1)


# ── main ─────────────────────────────────────────────────────────────────────────────


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if not str(k).startswith("_")}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if torch.is_tensor(x):
        return x.tolist()
    return x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR]")
    ap.add_argument("--pack_config", default="tul_slot_spandec_strict_fan4_all_fp01_nowta")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--fit_rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--follow", action="store_true", help="H2 CE readings (lsel arms)")
    ap.add_argument("--fan_depths", default="1,2,3,4,5,6",
                    help="write-all fan: forced depths whose exit cells are read")
    ap.add_argument("--vablate", default="",
                    help="ONLY the common-direction ablation of the written cells, at these "
                         "forced depths (e.g. 1,2,3,4,5,6); every other reading is skipped")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    t0 = time.time()
    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    pcfg = build_cfg(a.pack_config, ["model.use_kernels=false"])
    prt = build_tul_runtime(pcfg)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    rt = build_tul_runtime(cfg)
    if rt is not None and (rt.data_cfg.spec_for(cfg.data.seq_len).l_total
                           != prt.data_cfg.spec_for(pcfg.data.seq_len).l_total):
        raise SystemExit("the arm packs rows differently from --pack_config")
    model, step = load_ckpt(cfg, path, a.device, rt.model_cfg if rt is not None else None)
    model.eval()
    loader = create_dataloader(pcfg.data.tokenizer, pcfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    n_rows = a.rows + a.fit_rows
    row_tokens = prt.data_cfg.spec_for(pcfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, n_rows * row_tokens)
    batches = pack_rows(stream, prt, pcfg, a.batch, False)[:-(-n_rows // a.batch)]
    ne = -(-a.rows // a.batch)
    ev, fit = batches[:ne], batches[ne:]
    if rt is None:
        kind = "plain"
    elif getattr(model, "_lsel_mode", "off") != "off":
        kind = "lsel"
    else:
        kind = "fan"
    print(f"[{label}] step {step} kind {kind} eval batches {len(ev)} fit batches {len(fit)}",
          flush=True)
    res: dict = {"label": label, "config": config, "path": path, "step": step, "kind": kind,
                 "rows_eval": a.rows, "rows_fit": a.fit_rows}
    dev = a.device
    if a.vablate:
        if kind == "plain":
            raise SystemExit("--vablate needs a slot-loop arm")
        res["vablate"] = vablate_readings(model, ev, dev, kind,
                                          [int(x) for x in a.vablate.split(",")])
    elif kind == "fan":
        per_depth = {}
        for dpt in [int(x) for x in a.fan_depths.split(",")]:
            per_depth[dpt] = collect(model, ev, dev, kind, depth=dpt)
            print(f"  fan depth {dpt} collected ({time.time() - t0:.0f}s)", flush=True)
        d = per_depth[max(per_depth)]
        res["chan_live"] = d["chan_live"]
        c_all = torch.cat([per_depth[k]["cells"][0].float().reshape(-1, d["cells"][0].shape[-1])
                           for k in per_depth]).to(dev)
        v = F.normalize(c_all.mean(0), dim=0)
        del c_all
        cex = d["cells"][0].to(dev).float()
        res["h3_exit"] = cell_instruments(cex, cex.reshape(-1, cex.shape[-1]).mean(0))
        res["h3_exit_readout"] = _h3_readout(model, cex)
        geo, prev = {}, None
        for k in sorted(per_depth):
            c = per_depth[k]["cells"][0].to(dev).float()
            g = cell_geometry(c, v)
            if prev is not None:
                step_ = (c - prev).reshape(-1, c.shape[-1]).double()
                base = prev.reshape(-1, c.shape[-1]).double()
                g["step"] = {"rel_rms": float(step_.square().sum(-1).mean().sqrt()
                                              / base.square().sum(-1).mean().sqrt()),
                             "share_along_v": float((step_ @ v.double()).square().mean()
                                                    / step_.square().sum(-1).mean()),
                             "coherence_across_cells": coherence(step_)}
            geo[k] = g
            prev = c
        res["geometry_by_depth"] = geo
    else:
        d = collect(model, ev, dev, kind)
        print(f"  eval collected ({time.time() - t0:.0f}s)", flush=True)
        f = collect(model, fit, dev, kind)
        print(f"  fit collected ({time.time() - t0:.0f}s)", flush=True)
        res["chan_live"] = d["chan_live"]
        res["chan_twin"] = d["chan_twin"]
        if kind == "plain":
            ok_f, ok_e = f["ok"], d["ok"]
            zf = f["z_plain"][ok_f]
            res["target_plain_prelude_ln"] = target_stats(d["z_plain"][ok_e], zf.mean(0),
                                                          zf.std(0).clamp_min(1e-6))
        else:
            ok_f, ok_e = f["ok"], d["ok"]
            zf = f["z_twin"][ok_f]
            mu, sd = zf.mean(0), zf.std(0).clamp_min(1e-6)
            res["target_twin_ln"] = target_stats(d["z_twin"][ok_e], mu, sd)
            if d["z_online"] is not None:
                zof = f["z_online"][ok_f]
                res["target_online_ln"] = target_stats(d["z_online"][ok_e], zof.mean(0),
                                                       zof.std(0).clamp_min(1e-6))
            raw = d["raw_twin"][ok_e].double()
            res["target_twin_pre_ln"] = {
                "mean_pairwise_cos": mean_pairwise_cos(raw),
                "pr_centred": pr_cov(raw),
                "rms_norm": float(raw.square().sum(-1).mean().sqrt()),
            }
            c = d["cells"][max(d["cells"])].to(dev).float()
            res["h3_exit"] = cell_instruments(c, c.reshape(-1, c.shape[-1]).mean(0))
            res["h3_exit_readout"] = _h3_readout(model, c)
            res["lsel"] = lsel_readings(model, d, f, dev, float(model.cfg.tul.fan_lsel_eps))
            print(f"  readings done ({time.time() - t0:.0f}s)", flush=True)
            if a.follow:
                res["follow"] = follow_readings(model, ev, dev, res["lsel"]["_stats"])
                print(f"  follow done ({time.time() - t0:.0f}s)", flush=True)
    res["wall_s"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(_jsonable(res), fh, indent=1)
    print(f"wrote {a.out} ({res['wall_s']}s)")


if __name__ == "__main__":
    main()
