"""The slot loop's SHARED CARRIER CONSTANT: where it is made, what it hides, what it carries.

Follow-up to ``dataflow_probe.py`` (lab/experiments/successes/2026-10-05-snap-dataflow-probe.md):
in snap and LXTUL the loop's FIRST pass turns the Hyper-Connection cell carrier (4 streams x
1024) from ~55 % slot-specific energy into a vector that is 99.6-99.8 % shared by every slot
of every row; it lives in the DIFFERENCES between the streams and cancels in their mean.

Four readings, same 96 rows and the same packer / seed as the data-flow probe:

  Q1 SOURCE   inside a pass (passes 1, 2 and the last), the carrier is followed stage by
              stage and every update is split by source:
                inject      DiagonalInjection (context channels: A*h + dt*e - h)
                x0_l{i}     the per-layer x0/bigram term before core block i (broadcast)
                attn_mix_l{i} / mlp_mix_l{i}     (Hres - I) h, the Cayley stream mixer
                attn_write_l{i} / mlp_write_l{i} Hpost_row (x) y, the branch's write
                norm        RMSNorm(f) * g - f (tul.slot_cell_pass_norm: rms, per stream)
                reset       reset_to_winner (the latent-selected reset)
              Per source: RMS, the share of its energy that every slot shares, the cosine of
              its slot-mean to the pass's final shared vector c*, and its PROJECTION share
              <mean_s, c*> / |c*|^2 (the entry's share plus every source's sums to 1).
              The states between stages carry the same shared-share reading (the trace).
  Q2 IDENTITY the shared vector per pass (pre- and post-norm) is written out; cosines between
              passes, its split over the 4 streams (stream-mean part vs difference part),
              its channel concentration (top-k channel share, max element / RMS), its
              overlap with the norm gain g and with the injection's context channels.
              The cross-arm cosine is computed by the plot script from the two JSONs.
  Q3 CONTENT  closed-form ridge (float64, row-grouped 5-fold CV, ridge strength picked on an
              inner split of the training rows) from the winner cell's seed, its carrier
              after pass 1, its exit carrier and its written prefix vector (all streams;
              plus the stream mean of the last three) to (a) the mean input embedding of the
              slot's OWN span and of the NEXT span (held-out R^2) and (b) the multi-hot bag
              over the most frequent ``--vocab`` token ids (R-precision; frequency baseline).
  Q4 LOAD     at eval, the pass output f is edited BEFORE the model's own per-pass norm
              (hooks on ``tul_cell_norm``), with c_pre / c_post the pre- / post-norm shared
              vector per (pass, cell index) estimated LEAVE-ROW-OUT:
                ctrl          N((f - c_pre) + c_pre)           (must equal the plain forward)
                remove        N(f - c_pre)                     (constant gone, norm re-applied)
                remove_post   N(f) - c_post                    (constant gone, NO renorm)
                budget        N(f - c_pre) + c_post            (norm never sees the constant)
                perm          N(f - c_pre + P c_pre)           (same-size constant, channels
                                                                permuted within each stream)
              at every pass of depth 6 (``all``), every pass but the last (``butlast``) and at
              depth 1; CE change on the scored positions (paired per row, bootstrap CI) and
              the K1-K6 change.

NOTHING in ``morph/`` is edited. Hooks: forward hooks on ``model.injection``, the core
blocks and ``tul_cell_norm``; the instance attribute ``_hc_post`` of every core block's two
Hyper-Connection residuals is wrapped (it receives Hres, Hpost_row, h and y exactly as the
model computed them); the model's own ``_jac_capture`` / ``_lsel_capture`` lists. Self-tests:
the instrumented forward's CE is bit-equal to the plain one; every stage's input equals the
previous stage's output; the HC split reproduces the residual's output; the sources sum to
the final carrier; the Q4 control reproduces the plain CE.

Usage (3070: morph-scratch/dataflow/run_carrier_constant.sh):
  PYTHONPATH=. python lab/divergence/carrier_constant_probe.py \\
      --ckpt snap=lxtul_snap=/path/step_5000.pt --rows 96 --out carrier_constant_snap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg, parse_ckpt_spec  # noqa: E402
from _earning import offsets_from_layout  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from dataflow_probe import RowAcc, VecAcc, ce_of, forward_ce, rel_err  # noqa: E402

import morph.model.attention as _attn  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

VARIANTS = ("ctrl", "remove", "remove_post", "budget", "perm")
SCOPES = (("all", 6), ("butlast", 6), ("all", 1))


def flat_valid(x: torch.Tensor, vcell: torch.Tensor) -> torch.Tensor:
    """``[B, P, n, C]`` (or ``[B, P, C]``) -> ``[N_valid, n*C]`` fp32."""
    return x.float().flatten(2)[vcell] if x.dim() == 4 else x.float()[vcell]


def vec_means(vacc: VecAcc, names) -> dict[str, np.ndarray]:
    """Overall mean vector of every named stream of vectors (sum over rows / count)."""
    out = {}
    for nm in names:
        rows = vacc.d.get(nm)
        if not rows:
            continue
        s = sum(v[0] for v in rows.values())
        c = sum(v[2] for v in rows.values())
        out[nm] = s / max(c, 1e-12)
    return out


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


# ── Q1: the stage-by-stage capture ───────────────────────────────────────────────────────

class Cap:
    """Hooks that follow the carrier through one pass and split every update by source."""

    def __init__(self, model, vacc: VecAcc, acc: RowAcc):
        self.m = model
        self.vacc = vacc
        self.acc = acc
        self.M = int(model.cfg.tul.fan_k)
        self.n_core = len(model.core)
        self.on = False
        self.rec: set[int] = set()
        self.selftest: dict[str, float] = defaultdict(float)
        self.handles = []
        self.orig_post = {}
        self.vcell = None
        self.last = None          # the carrier after the last stage, fp32 [B, P, n, C]
        self.k = 0                # stage counter within a recorded pass
        self.calls = defaultdict(int)
        self.q4_sums = None       # [T, M, 2, D] per row (pre, post) and counts [T, M]
        self.order: dict[int, list[str]] = defaultdict(list)   # pass -> state names in order
        self.cur = None           # the core block running now (None outside the core)
        self.o_gcu = None

    # -- helpers -----------------------------------------------------------------------
    def t(self) -> int:
        return len(self.m._jac_capture) - 1

    def _state(self, t, name, x):
        nm = f"t{t}/s{self.k:02d}_{name}"
        if nm not in self.order[t]:
            self.order[t].append(nm)
        self.vacc.add(nm, flat_valid(x, self.vcell))
        self.k += 1

    def _contrib(self, t, name, d):
        self.vacc.add(f"t{t}/c/{name}", flat_valid(d, self.vcell))

    def _cont(self, t, name, x_in):
        """The stage's input must be the previous stage's output (nothing unseen ran)."""
        if self.last is not None:
            self.selftest[f"continuity/{name}"] = max(
                self.selftest[f"continuity/{name}"], rel_err(x_in, self.last))

    # -- hooks -------------------------------------------------------------------------
    def attach(self):
        m = self.m
        for blk in m.core:
            if blk.retention is not None:
                raise NotImplementedError("a core block carries a retention branch: its "
                                          "write is not split from attention here")
        self.handles.append(m.injection.register_forward_hook(self._inj_hook))
        for i, blk in enumerate(m.core):
            self.handles.append(blk.register_forward_pre_hook(self._blk_pre(i)))
            self.handles.append(blk.register_forward_hook(self._blk_post(i)))
            for br, res in (("attn", blk.mrr_attn), ("mlp", blk.mrr_mlp)):
                self.orig_post[(i, br)] = res._hc_post
                res._hc_post = self._wrap_post(i, br, res._hc_post)
        self.handles.append(m.tul_cell_norm.register_forward_hook(self._norm_hook))
        for i, blk in enumerate(m.core):
            for br, nrm in (("attn", blk.norm_attn), ("mlp", blk.norm_mlp)):
                self.handles.append(nrm.register_forward_hook(self._sub_in_hook(i, br)))
        self.o_gcu = _attn._CCABase._gate_combine_up
        _attn._CCABase._gate_combine_up = self._wrap_gcu(self.o_gcu)

    def detach(self):
        for h in self.handles:
            h.remove()
        for i, blk in enumerate(self.m.core):
            blk.mrr_attn._hc_post = self.orig_post[(i, "attn")]
            blk.mrr_mlp._hc_post = self.orig_post[(i, "mlp")]
        _attn._CCABase._gate_combine_up = self.o_gcu

    def _sub_in_hook(self, i, br):
        """The sublayer's input: x_bar (the HC pre-map's stream read) and norm(x_bar)."""
        def f(mod, args, out):
            if not self.on:
                return
            t = self.t()
            if t in self.rec:
                self.vacc.add(f"t{t}/xin/{br}_l{i}/xbar", flat_valid(args[0], self.vcell))
                self.vacc.add(f"t{t}/xin/{br}_l{i}/normed", flat_valid(out, self.vcell))
        return f

    def _wrap_gcu(self, orig):
        """Split a CORE attention's output y = W_up(g_c out_comp + g_w out_win + alpha q_lat)
        into its three parts (W_up is bias-free, so the parts sum to y)."""
        cap = self

        def w_gcu(mod, x, out_comp, out_win, q_lat=None, gate_pre=None, win_capture=None):
            res = orig(mod, x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre,
                       win_capture=win_capture)
            if not cap.on or cap.cur is None:
                return res
            t = cap.t()
            if t not in cap.rec:
                return res
            i = cap.cur
            with torch.no_grad():
                B, S_, _ = x.shape
                H, Dh = mod.n_heads, mod.d_head
                g_lin = mod.gate(x) if gate_pre is None else mod.gate[2](mod.gate[1](gate_pre))
                g = torch.sigmoid(g_lin).reshape(B, S_, H, 2).permute(0, 2, 1, 3)
                ql = mod.W_down_q(x) if q_lat is None else q_lat
                x_res = ql.reshape(B, S_, H, Dh).transpose(1, 2)

                def up(z):
                    return mod.W_up(z.transpose(1, 2).reshape(B, S_, mod.latent_q_dim)).float()

                parts = {"comp": up(g[..., 0:1] * out_comp), "win": up(g[..., 1:2] * out_win),
                         "res": up(mod.alpha * x_res)}
                cap.selftest["attn_parts_sum"] = max(cap.selftest["attn_parts_sum"],
                                                     rel_err(sum(parts.values()), res))
                for nm, v in parts.items():
                    cap.vacc.add(f"t{t}/attn_part/l{i}/{nm}", flat_valid(v, cap.vcell))
                cap.vacc.add(f"t{t}/attn_part/l{i}/x", flat_valid(x, cap.vcell))
                gv = g.float().mean(1)[cap.vcell]                       # [N, 2] head-mean
                cap.acc.add(f"t{t}/attn_gate/l{i}/comp", float(gv[:, 0].sum()), gv.shape[0])
                cap.acc.add(f"t{t}/attn_gate/l{i}/win", float(gv[:, 1].sum()), gv.shape[0])
            return res
        return w_gcu

    def _inj_hook(self, mod, args, out):
        if not self.on:
            return
        t = self.t()
        h = args[0]
        self.calls["inject"] += 1
        self.selftest["entry_eq_injection_input"] = max(
            self.selftest["entry_eq_injection_input"],
            rel_err(h, self.m._jac_capture[-1]["h"]))
        self.last = h.detach().float()
        self.k = 0
        if t in self.rec:
            self._state(t, "entry", h)
            self._contrib(t, "inject", out.float() - h.float())
            self._state(t, "inject", out)
        self.last = out.detach().float()

    def _blk_pre(self, i):
        def f(mod, args, kwargs=None):
            if not self.on:
                return
            t = self.t()
            self.cur = i
            x = args[0].detach().float()
            if t in self.rec:
                self._contrib(t, f"x0_l{i}", x - self.last)
                self._state(t, f"x0_l{i}", x)
            self.last = x
        return f

    def _blk_post(self, i):
        def f(mod, args, out):
            if not self.on:
                return
            self.calls["core"] += 1
            self.cur = None
            self._cont(self.t(), "block_out", out)
            self.last = out.detach().float()
        return f

    def _wrap_post(self, i, br, orig):
        cap = self

        def post(hres, hpost_row, h, y, term=None):
            out = orig(hres, hpost_row, h, y, term)
            if not cap.on:
                return out
            t = cap.t()
            cap._cont(t, f"{br}_in", h)
            hf = h.detach().float()
            mix = torch.einsum("bsij,bsjc->bsic", hres.detach().float(), hf) - hf
            write = hpost_row.detach().float().unsqueeze(-1) * y.detach().float().unsqueeze(2)
            if term is not None:
                write = write + term.detach().float().unsqueeze(2)
            of = out.detach().float()
            cap.selftest[f"hc_split/{br}"] = max(cap.selftest[f"hc_split/{br}"],
                                                 rel_err(hf + mix + write, of))
            if t in cap.rec:
                # the write absorbs the bf16 rounding of the residual's own add, so the
                # sources telescope EXACTLY to the carrier the model carried
                cap._contrib(t, f"{br}_mix_l{i}", mix)
                cap._contrib(t, f"{br}_write_l{i}", of - hf - mix)
                cap._state(t, f"{br}_l{i}", of)
                vy = flat_valid(y, cap.vcell)                          # [N, C] single stream
                cap.vacc.add(f"t{t}/y/{br}_l{i}", vy)
                hp = hpost_row.detach().float()[cap.vcell]             # [N, n]
                for j in range(hp.shape[-1]):
                    cap.acc.add(f"t{t}/hpost/{br}_l{i}/s{j}", float(hp[:, j].sum()),
                                hp.shape[0])
                    cap.acc.add(f"t{t}/hpost_sd/{br}_l{i}/s{j}",
                                float(hp[:, j].std()) * hp.shape[0], hp.shape[0])
                eye = torch.eye(hres.shape[-1], device=hres.device)
                dres = (hres.detach().float() - eye).flatten(2).norm(dim=-1)[cap.vcell]
                cap.acc.add(f"t{t}/hres_dev/{br}_l{i}", float(dres.sum()), dres.shape[0])
            cap.last = of
            return out
        return post

    def _norm_hook(self, mod, args, out):
        if not self.on:
            return
        t = self.t()
        f_ = args[0].detach().float()
        of = out.detach().float()
        self.calls["norm"] += 1
        self._cont(t, "norm_in", f_)
        if t in self.rec:
            u = f_ * f_.pow(2).mean(-1, keepdim=True).add(mod.eps).rsqrt()
            self.selftest["norm_formula"] = max(
                self.selftest["norm_formula"], rel_err(u * mod.weight.float(), of))
            self.vacc.add(f"t{t}/x/prenorm_unit", flat_valid(u, self.vcell))
            self._contrib(t, "norm", of - f_)
            self._state(t, "norm", of)
        # Q4's leave-row-out sums, every pass, per cell index
        B, P = f_.shape[:2]
        mi = (torch.arange(P, device=f_.device) % self.M).view(1, P).expand(B, P)
        for mm in range(self.M):
            sel = self.vcell & (mi == mm)
            n = int(sel.sum())
            if n:
                self.q4_sums[t, mm, 0] += f_.flatten(2)[sel].double().sum(0).cpu().numpy()
                self.q4_sums[t, mm, 1] += of.flatten(2)[sel].double().sum(0).cpu().numpy()
                self.q4_cnt[t, mm] += n
        self.last = of


# ── Q4: the intervention on the per-pass norm ────────────────────────────────────────────

class Intervene:
    """Edits the pass output ``f`` before ``tul_cell_norm`` and / or its output after it."""

    def __init__(self, model):
        self.m = model
        self.M = int(model.cfg.tul.fan_k)
        self.variant = None
        self.k = 0
        self.vacc = None
        self.h1 = model.tul_cell_norm.register_forward_pre_hook(self._pre)
        self.h2 = model.tul_cell_norm.register_forward_hook(self._post)

    def set(self, variant, last_active, c_pre, c_post, c_perm, vcell, vacc=None):
        """``c_*`` ``[T, M, D]`` fp32 on device; ``last_active`` the last pass edited."""
        self.variant, self.last_active = variant, last_active
        self.c_pre, self.c_post, self.c_perm, self.vcell = c_pre, c_post, c_perm, vcell
        self.vacc = vacc
        self.k = 0

    def clear(self):
        self.variant = None

    def _cells(self, c, t, shape):
        B, P = shape[:2]
        cc = c[t][torch.arange(P, device=c.device) % self.M]               # [P, D]
        return cc.view(1, P, *shape[2:]) * self.vcell.view(B, P, 1, 1).float()

    def _pre(self, mod, args):
        if self.variant is None or self.k > self.last_active:
            return None
        f = args[0]
        ff = f.float()
        v = self.variant
        t = self.k
        if v == "ctrl":
            cp = self._cells(self.c_pre, t, f.shape)
            ff = (ff - cp) + cp
        elif v in ("remove", "budget"):
            ff = ff - self._cells(self.c_pre, t, f.shape)
        elif v == "perm":
            ff = ff - self._cells(self.c_pre, t, f.shape) + self._cells(self.c_perm, t, f.shape)
        else:
            return None
        return (ff.to(f.dtype),)

    def _post(self, mod, args, out):
        if self.variant is None:
            return None
        t = self.k
        self.k += 1
        res = None
        if t <= self.last_active:
            if self.variant == "remove_post":
                res = (out.float() - self._cells(self.c_post, t, out.shape)).to(out.dtype)
            elif self.variant == "budget":
                res = (out.float() + self._cells(self.c_post, t, out.shape)).to(out.dtype)
        if self.vacc is not None:
            self.vacc.add(f"q4/{self.variant}/t{t}/post",
                          flat_valid(out if res is None else res, self.vcell))
        return res


# ── Q3: closed-form ridge ────────────────────────────────────────────────────────────────

def ridge_eval(X, rows, targets, folds, seed, device, scales=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10)):
    """Row-grouped K-fold ridge in float64. ``targets``: name -> (Y [N, k], kind, mask [N]) with
    kind ``r2`` (pooled held-out R^2 against the train mean) or ``rprec`` (R-precision of a
    multi-hot bag against the train-frequency ranking). The ridge strength (a multiple of the
    mean eigenvalue of the centred Gram) is chosen per target on an inner split of the
    training rows, then the model is refit on all training rows."""
    rng = np.random.default_rng(seed)
    ur = np.unique(rows)
    perm = rng.permutation(ur)
    fold_of = {r: i % folds for i, r in enumerate(perm)}
    fid = np.array([fold_of[r] for r in rows])
    Xt = torch.from_numpy(X).to(device, torch.float64)

    def fit(tr):
        Xa = Xt[tr]
        mu = Xa.mean(0)
        Xc = Xa - mu
        lam, V = torch.linalg.eigh(Xc.T @ Xc)
        return mu, lam.clamp_min(0), V, Xc

    def predict(state, Ytr, te, alpha):
        mu, lam, V, Xc = state
        ym = Ytr.mean(0)
        XtY = Xc.T @ (Ytr - ym)
        W = V @ ((V.T @ XtY) / (lam + alpha).unsqueeze(1))
        return (Xt[te] - mu) @ W + ym, ym

    def score(kind, Y, P, ybar):
        if kind == "r2":
            return float(1 - ((Y - P) ** 2).sum() / ((Y - ybar) ** 2).sum())
        R = Y.sum(1).long()
        ok = R > 0
        if not bool(ok.any()):
            return float("nan")
        Y, P, R = Y[ok], P[ok], R[ok]
        order = torch.argsort(P, dim=1, descending=True)
        rk = torch.empty_like(order)
        rk.scatter_(1, order, torch.arange(P.shape[1], device=P.device).expand_as(order))
        hit = ((rk < R.unsqueeze(1)) & (Y > 0)).sum(1).double() / R.double()
        return float(hit.mean())

    out = {}
    Ys = {k: (torch.from_numpy(v[0]).to(device, torch.float64), v[1],
              torch.from_numpy(v[2]).to(device)) for k, v in targets.items()}
    acc = {k: {"num": 0.0, "den": 0.0, "hits": [], "base": [], "alpha": []} for k in Ys}
    for f in range(folds):
        tr_all = np.nonzero(fid != f)[0]
        te = np.nonzero(fid == f)[0]
        tr_rows = np.unique(rows[tr_all])
        inner_val_rows = set(rng.choice(tr_rows, size=max(1, len(tr_rows) // 5),
                                        replace=False).tolist())
        iv = np.array([r in inner_val_rows for r in rows[tr_all]])
        itr = torch.from_numpy(tr_all[~iv]).to(device)
        ival = torch.from_numpy(tr_all[iv]).to(device)
        tr = torch.from_numpy(tr_all).to(device)
        te_t = torch.from_numpy(te).to(device)
        fits: dict = {}          # one eigendecomposition per (target mask, train set)

        def get_fit(key, idx, mask):
            mk = (key, mask.cpu().numpy().tobytes())
            if mk not in fits:
                keep = idx[mask[idx]]
                fits[mk] = (fit(keep), keep)
            return fits[mk]

        for k, (Y, kind, mask) in Ys.items():
            # a slot without a target (a NEXT span the row cut off) drops out of the fit
            # and the scoring alike
            (st_in, keep_in) = get_fit("inner", itr, mask)
            ival_m = ival[mask[ival]]
            best, best_a = -np.inf, None
            for s in scales:
                P, ybar = predict(st_in, Y[keep_in], ival_m, s * float(st_in[1].mean()))
                sc = score(kind, Y[ival_m], P, ybar)
                if sc > best:
                    best, best_a = sc, s
            (st, keep) = get_fit("outer", tr, mask)
            te_m = te_t[mask[te_t]]
            if te_m.numel() == 0:
                continue
            P, ybar = predict(st, Y[keep], te_m, best_a * float(st[1].mean()))
            acc[k]["alpha"].append(best_a)
            if kind == "r2":
                acc[k]["num"] += float(((Y[te_m] - P) ** 2).sum())
                acc[k]["den"] += float(((Y[te_m] - ybar) ** 2).sum())
            else:
                acc[k]["hits"].append((score(kind, Y[te_m], P, ybar), int(te_m.numel())))
                acc[k]["base"].append((score(kind, Y[te_m], ybar.expand_as(P), ybar),
                                       int(te_m.numel())))
    for k, (Y, kind, mask) in Ys.items():
        a = acc[k]
        if kind == "r2":
            out[k] = {"r2": 1 - a["num"] / max(a["den"], 1e-30), "alpha_scale": a["alpha"]}
        else:
            w = np.array([n for _, n in a["hits"]], dtype=float)
            out[k] = {"rprec": float(np.average([h for h, _ in a["hits"]], weights=w)),
                      "rprec_freq_baseline": float(np.average([h for h, _ in a["base"]],
                                                              weights=w)),
                      "alpha_scale": a["alpha"]}
    return out


# ── main ─────────────────────────────────────────────────────────────────────────────────

def run_arm(spec: str, a) -> dict:
    t0 = time.time()
    label, config, path, ovr = parse_ckpt_spec(spec)
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    device = a.device
    torch.manual_seed(a.seed)
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    if model.tul_cell_norm is None or model._cell_norm_mode != "rms":
        raise RuntimeError("this probe models tul.slot_cell_pass_norm: rms only")
    T = int(tc.slot_mean_depth or model.cfg.mean_depth)
    M = int(tc.fan_k)
    K = int(tc.prefix_k)
    spec_ = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, a.rows * (spec_.l_total + 1))
    batches = pack_rows(stream, tul_rt, cfg, 1, False)[:a.rows]
    n_rows = len(batches)
    acc, vacc = RowAcc(), VecAcc()
    cap = Cap(model, vacc, acc)
    cap.attach()
    cap.rec = {0, 1, T - 1}
    D = None
    q4_sums = []
    feats = defaultdict(list)          # Q3 features, one entry per valid slot
    tgt_own, tgt_next, slot_rows = [], [], []
    sanity = {"hook_vs_plain_bit_equal": True, "hook_vs_plain_max_abs_ce_diff": 0.0,
              "winner_eq_written_nonzero": True}
    scores, ce_base = [], {}
    follows = []                       # the plain forward's router pick per pass, per row
    carrier_dtype = None

    # ---- phase 1: plain CE at depth 6 and 1, the instrumented forward ----
    for r, (inp, labels, layout, idx) in enumerate(batches):
        acc.row = vacc.row = r
        lay = layout.to(device)
        B, L = inp.shape
        S = layout.slot_valid.shape[1]
        off = offsets_from_layout(layout, 0)
        score = torch.from_numpy(off >= 0) & (labels[0] >= 0) & ~layout.slot_mask[0]
        scores.append(score)
        for d in (T, 1):
            tc.slot_mean_depth = d
            ce_base[(r, d)] = forward_ce(model, inp, labels, lay, "normal", device)[0].cpu()
        tc.slot_mean_depth = T
        cap.vcell = lay.slot_valid.repeat_interleave(M, 1)
        cap.q4_sums = None
        model._jac_capture, model._lsel_capture = [], []
        ctx = {}
        tul = model.tul
        orig_pp = tul.prefix_project

        def pp(*args, **kwargs):
            values, pos = orig_pp(*args, **kwargs)
            ctx["pp"] = (pos.detach(), kwargs.get("cells"))
            return values, pos

        def coda0(mod, args):
            ctx["coda_in"] = args[0].detach()

        hk = model.coda[0].register_forward_pre_hook(coda0)
        tul.prefix_project = pp
        cap.calls = defaultdict(int)
        cap.last = None
        if D is None:
            D = int(model.cfg.d_model) * int(model._n_streams)
        cap.q4_sums = np.zeros((T, M, 2, D))
        cap.q4_cnt = np.zeros((T, M))
        cap.on = True
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device.startswith("cuda")):
                res = model.tul_forward_ablated(inp.to(device), None, lay, plan_mode="normal")
        finally:
            cap.on = False
            hk.remove()
            del tul.prefix_project
        jac, lsel = model._jac_capture, model._lsel_capture
        model._jac_capture = model._lsel_capture = None
        if cap.calls["core"] != cap.n_core * T or cap.calls["norm"] != T \
                or cap.calls["inject"] != T or len(jac) != T or len(lsel) != T:
            raise RuntimeError(f"row {r}: calls {dict(cap.calls)}, {len(jac)} jac, "
                               f"{len(lsel)} lsel at depth {T}")
        ce_i = ce_of(res["logits"], labels, device)[0].cpu()
        tp = (~layout.slot_mask[0]) & (labels[0] >= 0)
        diff = float((ce_i - ce_base[(r, T)])[tp].abs().max())
        sanity["hook_vs_plain_max_abs_ce_diff"] = max(sanity["hook_vs_plain_max_abs_ce_diff"],
                                                      diff)
        if not torch.equal(ce_i[tp], ce_base[(r, T)][tp]):
            sanity["hook_vs_plain_bit_equal"] = False
        q4_sums.append((cap.q4_sums, cap.q4_cnt))
        follows.append([lsel[t]["follow"].detach() for t in range(T)])
        carrier_dtype = str(jac[0]["h"].dtype)
        vb = lay.slot_valid
        vcell = cap.vcell
        # reset contribution and the post-reset state, recorded passes
        for t in sorted(cap.rec):
            if t < T - 1:
                pre, post = lsel[t]["pre"].float(), lsel[t]["post"].float()
                cap.k = 99
                vacc.add(f"t{t}/c/reset", flat_valid(post - pre, vcell))
                cap._state(t, "reset", post)

        def pick(x, w):                    # [B, S*M, n, C] cells, [B, S] -> [B, S, n*C]
            xc = x.float().view(B, S, M, -1)
            return torch.gather(xc, 2, w.view(B, S, 1, 1).expand(B, S, 1, xc.shape[-1])
                                ).squeeze(2)

        w0 = lsel[0]["follow"]
        wT = lsel[T - 1]["follow"]
        # reproduce the data-flow probe's per-slot shares (its cm_winner, all streams)
        for t in range(T):
            vacc.add(f"repro/t{t}/cm_winner", pick(lsel[t]["pre"], lsel[t]["follow"])[vb])
        vacc.add("repro/cm_entry", pick(jac[0]["h"], w0)[vb])
        # Q3 features: the winner cell at each stage, all streams
        f_seed = pick(jac[0]["h"], w0)
        f_p1 = pick(lsel[0]["pre"], w0)
        f_exit = pick(lsel[T - 1]["pre"], wT)
        pos, cells = ctx["pp"]
        pv = pos.reshape(B, S, K)
        if cells is not None:
            mag = cells.float().flatten(3).abs().sum(-1)
            wz = mag.argmax(-1)
            if not torch.equal(wz[vb], wT[vb]):
                sanity["winner_eq_written_nonzero"] = False
        x0 = ctx["coda_in"].float().flatten(2)                    # [B, L, n*C]
        wpos = torch.gather(pv, 2, wT.view(B, S, 1)).squeeze(2).clamp(max=L - 1)
        f_pref = torch.gather(x0, 1, wpos.unsqueeze(-1).expand(B, S, x0.shape[-1]))
        n_s = int(model._n_streams)
        for nm, fv in (("seed", f_seed), ("p1", f_p1), ("exit", f_exit), ("prefix", f_pref)):
            feats[nm].append(fv[vb].cpu().numpy().astype(np.float32))
            if nm != "seed":
                feats[nm + "_sm"].append(fv.view(B, S, n_s, -1).mean(2)[vb].cpu().numpy()
                                         .astype(np.float32))
        bag = layout.bag_id[0]
        tok = ~layout.slot_mask[0]
        for s in range(S):
            if not bool(layout.slot_valid[0, s]):
                continue
            tgt_own.append(inp[0, (bag == s) & tok].tolist())
            tgt_next.append(inp[0, (bag == s + 1) & tok].tolist())
            slot_rows.append(r)
        if r % 8 == 0 or r == n_rows - 1:
            print(f"[{label}] phase1 row {r + 1}/{n_rows}  {time.time() - t0:.0f}s", flush=True)
    cap.detach()

    # ---- the leave-row-out constants ----
    tot = sum(s for s, _ in q4_sums)
    cnt = sum(c for _, c in q4_sums)
    gen = torch.Generator().manual_seed(a.seed)
    n_s = int(model._n_streams)
    C = D // n_s
    perm_idx = torch.stack([torch.randperm(C, generator=gen) for _ in range(n_s)])  # [n, C]

    def lro(r):
        s, c = q4_sums[r]
        cc = (tot - s) / np.maximum(cnt - c, 1)[..., None, None]           # [T, M, 2, D]
        c_pre = torch.from_numpy(cc[:, :, 0]).float().to(device)
        c_post = torch.from_numpy(cc[:, :, 1]).float().to(device)
        cp = c_pre.view(T, M, n_s, C)
        c_perm = torch.gather(cp, 3, perm_idx.to(device).view(1, 1, n_s, C).expand_as(cp)
                              ).view(T, M, D)
        return c_pre, c_post, c_perm

    # ---- phase 2: the interventions ----
    iv = Intervene(model)
    for r, (inp, labels, layout, idx) in enumerate(batches):
        acc.row = vacc.row = r
        lay = layout.to(device)
        vcell = lay.slot_valid.repeat_interleave(M, 1)
        c_pre, c_post, c_perm = lro(r)
        sc = scores[r]
        n_sc = float(sc.sum())
        k_base = ce_base[(r, 1)] - ce_base[(r, T)]
        acc.add("ce/d6", float(ce_base[(r, T)][sc].sum()), n_sc)
        acc.add("ce/d1", float(ce_base[(r, 1)][sc].sum()), n_sc)
        acc.add("ce/K1-K6", float(k_base[sc].sum()), n_sc)
        for v in VARIANTS:
            ce_v = {}
            for scope, d in SCOPES:
                tc.slot_mean_depth = d
                last_active = d - 1 if scope == "all" else d - 2
                iv.set(v, last_active, c_pre, c_post, c_perm, vcell,
                       vacc=vacc if (scope == "all" and d == T) else None)
                watch = scope == "all" and d == T
                if watch:
                    model._lsel_capture = []
                try:
                    ce = forward_ce(model, inp, labels, lay, "normal", device)[0].cpu()
                finally:
                    iv.clear()
                    cap_l, model._lsel_capture = model._lsel_capture, None
                if watch:
                    # does the edit change the router's pick? (valid slots, every pass)
                    vb = lay.slot_valid
                    for t in range(T):
                        same = (cap_l[t]["follow"] == follows[r][t])[vb].float()
                        acc.add(f"q4/{v}/winner_agree_t{t}", float(same.sum()), same.numel())
                if iv.k != d:
                    raise RuntimeError(f"{v}/{scope}/d{d}: the norm ran {iv.k} times")
                ce_v[(scope, d)] = ce
                dl = ce - ce_base[(r, d)]
                acc.add(f"q4/{v}/{scope}_d{d}/dCE", float(dl[sc].sum()), n_sc)
                if v == "ctrl":
                    acc.add(f"q4/ctrl/{scope}_d{d}/frac_abs_gt_1e-3",
                            float((dl[sc].abs() > 1e-3).sum()), n_sc)
                    sanity.setdefault("ctrl_max_abs_ce_diff", 0.0)
                    sanity["ctrl_max_abs_ce_diff"] = max(sanity["ctrl_max_abs_ce_diff"],
                                                         float(dl[sc].abs().max()))
            tc.slot_mean_depth = T
            kv = ce_v[("all", 1)] - ce_v[("all", T)]
            acc.add(f"q4/{v}/K1-K6", float(kv[sc].sum()), n_sc)
            acc.add(f"q4/{v}/dK1-K6", float((kv - k_base)[sc].sum()), n_sc)
        if r % 8 == 0 or r == n_rows - 1:
            print(f"[{label}] phase2 row {r + 1}/{n_rows}  {time.time() - t0:.0f}s", flush=True)
    iv.h1.remove()
    iv.h2.remove()
    tc.slot_mean_depth = T

    # ---- summaries ----
    print(f"[{label}] summarising  {time.time() - t0:.0f}s", flush=True)
    cm = vacc.summary(n_rows, a.boot, a.seed)
    means = vec_means(vacc, list(vacc.d))
    g = model.tul_cell_norm.weight.detach().float().cpu().numpy()
    with torch.no_grad():
        emb = model.embed.lm_weight().detach().float().cpu().numpy()      # [V, d] tied head
    inj = model.injection
    ctx_lo, ctx_hi = int(inj.start), int(inj.end)

    def shape_of(vec):
        """Q2 readings of one shared vector [n*C]."""
        v = vec.reshape(n_s, C)
        e = float((v ** 2).sum())
        sm = v.mean(0)
        ch = (v ** 2).sum(0)
        srt = np.sort(ch)[::-1]
        top = np.argsort(ch)[::-1][:8]
        return {
            "norm": float(np.sqrt(e)), "rms": float(np.sqrt(e / v.size)),
            "stream_energy_share": [float((v[i] ** 2).sum() / e) for i in range(n_s)],
            "stream_mean_share": float(n_s * (sm ** 2).sum() / e),
            "stream_pair_cos": [[cos(v[i], v[j]) for j in range(n_s)] for i in range(n_s)],
            "top_channel_share": {str(k): float(srt[:k].sum() / e) for k in (1, 8, 32, 128)},
            "top_channels": top.tolist(),
            "max_abs_over_rms": float(np.abs(v).max() / np.sqrt(e / v.size)),
            "ctx_channel_share": float(ch[ctx_lo:ctx_hi].sum() / e),
            "ctx_channel_fraction": (ctx_hi - ctx_lo) / C,
            "cos_channel_energy_vs_g2": cos(ch, g ** 2),
            "g_at_top8": [float(g[i]) for i in top],
            "channel_energy": ch.tolist(),
        }

    q1 = {}
    for t in sorted(cap.rec):
        order = cap.order[t]
        i_norm = [i for i, nm in enumerate(order) if nm.endswith("_norm")]
        if len(i_norm) != 1 or not order[0].endswith("_entry"):
            raise RuntimeError(f"pass {t}: unexpected stage order {order}")
        cstar = means[order[i_norm[0]]]
        fpre = means[order[i_norm[0] - 1]]
        entry = means[order[0]]
        src_names = sorted({nm.split("/c/")[1] for nm in vacc.d if nm.startswith(f"t{t}/c/")},
                           key=lambda s: _src_order(s))
        tot_share = cos_den = float(cstar @ cstar)
        rows_ = {}
        share_sum = float(entry @ cstar) / tot_share
        pre_den = float(fpre @ fpre)
        pre_sum = float(entry @ fpre) / pre_den
        for s in src_names:
            nm = f"t{t}/c/{s}"
            mv = means[nm]
            ms = mv.reshape(n_s, C)
            e = float(mv @ mv)
            sh = float(mv @ cstar) / cos_den
            rows_[s] = {
                "rms": cm[nm]["rms"], "shared_share": cm[nm]["mean"],
                "shared_share_ci": [cm[nm]["lo"], cm[nm]["hi"]],
                "per_slot_rms": cm[nm]["dev_rms"],
                "mean_rms": float(np.sqrt(e / mv.size)),
                "cos_to_cstar": cos(mv, cstar), "proj_share_cstar": sh,
                "proj_share_prenorm": (float(mv @ fpre) / pre_den
                                       if s not in ("norm", "reset") else None),
                "mean_stream_diff_share": (1 - n_s * float((ms.mean(0) ** 2).sum()) / e
                                           if e > 0 else None),
            }
            if s not in ("reset",):
                share_sum += sh
            if s not in ("norm", "reset"):
                pre_sum += rows_[s]["proj_share_prenorm"]
        states = []
        for nm in order:
            r_ = cm[nm]
            mv = means[nm]
            ms = mv.reshape(n_s, C)
            states.append({"stage": nm.split("_", 1)[1], "shared_share": r_["mean"],
                           "ci": [r_["lo"], r_["hi"]], "rms": r_["rms"],
                           "per_slot_rms": r_["dev_rms"],
                           "cos_to_cstar": cos(mv, cstar),
                           "mean_stream_diff_share": 1 - n_s * float((ms.mean(0) ** 2).sum())
                           / max(float(mv @ mv), 1e-30)})
        unit = cm.get(f"t{t}/x/prenorm_unit")
        ys = {nm.split("/y/")[1]: {"shared_share": cm[nm]["mean"], "rms": cm[nm]["rms"]}
              for nm in vacc.d if nm.startswith(f"t{t}/y/")}
        parts = {}
        for nm in vacc.d:
            for key in ("/attn_part/", "/xin/"):
                if nm.startswith(f"t{t}{key}"):
                    mv = means[nm]
                    parts[nm.split("/", 1)[1]] = {
                        "shared_share": cm[nm]["mean"], "rms": cm[nm]["rms"],
                        "per_slot_rms": cm[nm]["dev_rms"],
                        "mean_rms": float(np.sqrt(float(mv @ mv) / mv.size))}
        q1[f"t{t}"] = {
            "entry_proj_share_cstar": float(entry @ cstar) / tot_share,
            "sum_of_proj_shares_excl_reset": share_sum,
            "entry_proj_share_prenorm": float(entry @ fpre) / pre_den,
            "sum_of_proj_shares_prenorm": pre_sum,
            "sources": rows_, "states": states,
            "prenorm_unit_no_g": None if unit is None else {
                "shared_share": unit["mean"], "ci": [unit["lo"], unit["hi"]]},
            "y": ys, "parts": parts,
        }
    hp = acc.summary(n_rows, a.boot, a.seed)
    # Q2: the shared vectors per pass (pre- and post-norm, all valid cells pooled)
    cpre = (tot[:, :, 0].sum(1) / cnt.sum(1)[:, None])                     # [T, D]
    cpost = (tot[:, :, 1].sum(1) / cnt.sum(1)[:, None])
    q2 = {
        "pass_cos_post": [[cos(cpost[i], cpost[j]) for j in range(T)] for i in range(T)],
        "pass_cos_pre": [[cos(cpre[i], cpre[j]) for j in range(T)] for i in range(T)],
        "cell_cos_post_t0": [[cos(tot[0, i, 1] / cnt[0, i], tot[0, j, 1] / cnt[0, j])
                              for j in range(M)] for i in range(M)],
        "shape_post": {f"t{t}": shape_of(cpost[t]) for t in range(T)},
        "shape_pre": {f"t{t}": shape_of(cpre[t]) for t in range(T)},
        "c_post": cpost.astype(np.float32).tolist(),
        "c_pre": cpre.astype(np.float32).tolist(),
        "g": g.tolist(), "ctx_channels": [ctx_lo, ctx_hi],
        "injection_A_mean": float(inj.log_A.detach().float().exp().mean()),
        "injection_dt_mean": float(inj.log_dt.detach().float().exp().mean()),
    }
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Q3: ridge
    print(f"[{label}] ridge  {time.time() - t0:.0f}s", flush=True)
    # float64 eigh: a consumer GPU runs fp64 at a small fraction of its fp32 rate, so the
    # ridge runs on the host (3070 box: 3.8 s per 4096 eigh at 8 threads)
    torch.set_num_threads(a.ridge_threads)
    rows_arr = np.array(slot_rows)
    freq = defaultdict(int)
    for lst in tgt_own:
        for tk in lst:
            freq[tk] += 1
    for lst in tgt_next:
        for tk in lst:
            freq[tk] += 1
    vocab = [tk for tk, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:a.vocab]]
    vid = {tk: i for i, tk in enumerate(vocab)}
    N = len(slot_rows)
    targets = {}
    cover = {}
    for nm, lst in (("own", tgt_own), ("next", tgt_next)):
        Ye = np.zeros((N, emb.shape[1]))
        Ym = np.zeros((N, len(vocab)))
        mask = np.zeros(N, dtype=bool)
        n_in = n_all = 0
        for i, toks in enumerate(lst):
            if not toks:
                continue
            mask[i] = True
            Ye[i] = emb[toks].mean(0)
            for tk in set(toks):
                n_all += 1
                if tk in vid:
                    Ym[i, vid[tk]] = 1.0
                    n_in += 1
        targets[f"{nm}_emb"] = (Ye, "r2", mask)
        targets[f"{nm}_bag"] = (Ym, "rprec", mask)
        cover[nm] = {"slots": int(mask.sum()), "distinct_token_coverage": n_in / max(n_all, 1)}
    q3 = {"n_slots": N, "vocab": len(vocab), "coverage": cover, "sources": {}}
    for nm in ("seed", "p1", "exit", "prefix", "p1_sm", "exit_sm", "prefix_sm"):
        X = np.concatenate(feats[nm]).astype(np.float64)
        q3["sources"][nm] = ridge_eval(X, rows_arr, targets, a.folds, a.seed, "cpu")
        print(f"[{label}] ridge {nm} ({time.time() - t0:.0f}s): " + ", ".join(
            f"{k} {v.get('r2', v.get('rprec')):.3f}" for k, v in q3["sources"][nm].items()),
            flush=True)

    out = {
        "label": label, "config": config, "ckpt": path, "step": step, "rows": n_rows,
        "batch": 1, "depth": T, "fan_k": M, "prefix_k": K, "n_streams": n_s, "d_model": C,
        "seed": a.seed, "carrier_dtype": carrier_dtype,
        "sanity": {**sanity, "selftest_max_rel_err": dict(cap.selftest)},
        "repro": {k: {"per_slot_share": 1 - cm[k]["mean"], "rms": cm[k]["rms"],
                      "per_slot_rms": cm[k]["dev_rms"]}
                  for k in cm if k.startswith("repro/")},
        "q1": q1, "q2": q2, "q3": q3,
        "q4": {k: v for k, v in hp.items() if k.startswith(("q4/", "ce/"))},
        "q4_regrowth": {k: {"per_slot_share": 1 - v["mean"], "rms": v["rms"]}
                        for k, v in cm.items() if k.startswith("q4/")},
        "hc": {k: v for k, v in hp.items() if k.startswith(("t0/", "t1/", f"t{T - 1}/"))},
        "stream_pair_cos_note": "q2.shape_*.stream_pair_cos: +-1 entries mean one channel "
                                "direction with a per-stream sign / scale",
        "wall_s": time.time() - t0,
    }
    print(f"[{label}] sanity {json.dumps(out['sanity'])}", flush=True)
    return out


def _src_order(s: str) -> tuple:
    if s == "inject":
        return (0, 0, 0)
    if s in ("norm", "reset"):
        return (9, 0 if s == "norm" else 1, 0)
    kind, li = s.rsplit("_l", 1)
    return (1, int(li), ("x0", "attn_mix", "attn_write", "mlp_mix", "mlp_write").index(kind))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr,...]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ridge-threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = run_arm(a.ckpt, a)
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
