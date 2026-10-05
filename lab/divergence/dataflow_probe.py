"""Stage-by-stage DATA-FLOW probe of an LXTUL slot-loop model (snap vs LXTUL).

Prereg (authoritative, not edited here): lab/experiments/planned/2026-10-05-snap-dataflow-probe.md.
Wolfe (2026-10-05): "the loop should be driving almost all the contribution ... I feel like we
are missing something that will be obvious when looking at the data moving through the model."

ONE checkpoint per ``--ckpt``; the same packed validation rows for every arm (the sweep's own
packer, ``_rows.pack_rows``, batch 1 so every row is one bootstrap unit). Eval mode, so the slot
depth is the deterministic ``tul.slot_mean_depth or model.mean_depth`` (6 on both arms) and the
loop follows its ROUTER (labels are not passed, so the latent teacher never drives): the deployed
forward. Stage 6 forces depth 1 through ``tul.slot_mean_depth`` (core_depth_sweep's lever).

NOTHING in ``morph/`` is edited. The probe reads the model through:
  * forward pre/post hooks on every prelude / core / coda block (they set a region tag and
    count calls, so a core call is (layer, pass) and a coda call is (layer));
  * wrappers around ``attention._tg_slot_attention`` (the compressed branch under tg_restrict),
    ``attention._window_fallback`` (the window branch; strict always routes here) and
    ``attention._CCABase._gate_combine_up`` (the per-head sigmoid gate that mixes the two).
    MORPH never materialises an attention matrix, so the wrapper RECOMPUTES the softmax from
    the same q, k and masks the call received and SELF-TESTS that ``w @ v`` reproduces the
    branch output the model actually used (``attn_sink_probe.py``'s rule);
  * a forward hook on ``tul_register`` (stage 1: the seed pool is recomputed and checked
    against the hook output), and on ``tul_pseudo_snap`` (stage 4: the snap's mixture ``p``);
  * the model's own capture lists ``_jac_capture`` (the carrier entering each pass),
    ``_lsel_capture`` (each pass's cells before / after the reset, the followed winner) and
    ``_pseudo_snap_capture``;
  * an instance wrapper on ``tul.prefix_project`` (the write's values and positions; the
    worth_profile.py pattern) and, for the stage-7 ablations, on ``_tul_pseudo_snap_write``.

STAGES (prereg numbering):
  1 seed pool   TULSlotRegister's per-cell attention over its span's prelude tokens.
  2 loop        per pass: ||out - in|| / ||out|| per cell, the followed winner's change and
                cosine to pass 1, router switches, cell spread before the reset.
  3 core reads  per core layer and pass: a cell's attention mass on its own slot's cells vs
                earlier slots' cells (vs the sink), per branch and gate-combined, and the
                value-weighted output norm from each.
  4 write       RMS of the written winner / pseudo / zero prefix positions against the coda's
                token-position states; the snap's gate g, beta, vertex mass, picks, copy hits.
  5 coda reads  per coda layer and head: a token's attention mass on slot positions (winner /
                pseudo / zero) vs its own span's tokens, by slot distance and by offset in
                span; the value-weighted output norm from slots vs tokens vs the residual-alpha
                term.
  6 depth       stages 2-5 at forced depth 1 vs the eval depth 6, plus K1-K6 on these rows.
  7 worth       zero the winner only / the pseudo tokens only / both; per offset bin and per
                exact-recall bucket (exact_recall_gap.row_buckets: far bigram / far token /
                novel, far gap 33, i >= 64).

GATE-COMBINED MASS (a definition, stated): the layer's attention output per head is
``g_c * out_comp + g_w * out_win`` with ``g`` two independent sigmoids, each ``out`` a convex
combination of values. The combined mass on a key set is ``(g_c * m_comp + g_w * m_win) /
(g_c + g_w)`` per query and head; the compressed branch's sink (a zero value) counts as mass on
nothing. The residual-attention term ``alpha * q_lat`` reads no key and is reported only in the
value-weighted norms.

Usage (one arm per call keeps the GPU hold short; 3070: see morph-scratch/arc/probe3070.sh):
  PYTHONPATH=. python lab/divergence/dataflow_probe.py \\
      --ckpt snap=lxtul_snap=/path/step_5000.pt --rows 48 --out dataflow_snap.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import string
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg, parse_ckpt_spec  # noqa: E402
from _earning import BINS, bin_of, offsets_from_layout  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from exact_recall_gap import FAR_GAP, MIN_I, row_buckets  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

import morph.model.attention as _attn  # noqa: E402

RECALL = {0: "novel", 1: "far_token", 2: "far_bigram"}
CORE_DIST = (("d1", 1, 1), ("d2", 2, 2), ("d3-4", 3, 4), ("d5-8", 5, 8), ("d9+", 9, 10 ** 9))
ROLES = ("winner", "pseudo", "zero")
CATS = ("newline", "space", "punct", "number", "function_word", "content")


# ── per-row accumulation (bootstrap unit = row) ──────────────────────────────────────────

class RowAcc:
    """``name -> row -> [sum, count]``; a reading is the token-weighted ratio of sums, its
    CI a percentile bootstrap over rows (``_stats.paired_bootstrap_ci``'s convention)."""

    def __init__(self):
        self.d: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(
            lambda: [0.0, 0.0]))
        self.row = 0

    def add(self, name: str, s, c) -> None:
        e = self.d[name][self.row]
        e[0] += float(s)
        e[1] += float(c)

    def summary(self, n_rows: int, boot: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_rows, size=(boot, n_rows))
        out = {}
        for name in sorted(self.d):
            s = np.zeros(n_rows)
            c = np.zeros(n_rows)
            for r, (a, b) in self.d[name].items():
                s[r], c[r] = a, b
            if c.sum() <= 0:
                continue
            bs = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1e-12)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            out[name] = {"mean": float(s.sum() / c.sum()), "lo": float(lo), "hi": float(hi),
                         "n": float(c.sum())}
        return out


class VecAcc:
    """COMMON-MODE share of a set of vectors: ``|mean v|^2 / mean |v|^2`` over every vector
    of every row (1 = every query receives the same vector, i.e. a bias; 0 = zero-mean
    content). Per row: the vector sum, the squared-norm sum, the count; the CI resamples
    rows. Vectors are fp32 on the host, ``d`` = 1024, so one key is ``rows x d`` floats."""

    def __init__(self):
        self.d: dict[str, dict[int, list]] = defaultdict(dict)
        self.row = 0

    def add(self, name: str, v: torch.Tensor) -> None:
        """``v`` ``[N, d]``."""
        if v.shape[0] == 0:
            return
        v = v.float()
        e = self.d[name].get(self.row)
        vs = v.sum(0).cpu().double().numpy()
        q = float(v.pow(2).sum())
        if e is None:
            self.d[name][self.row] = [vs, q, float(v.shape[0])]
        else:
            e[0] += vs
            e[1] += q
            e[2] += float(v.shape[0])

    def summary(self, n_rows: int, boot: int, seed: int) -> dict:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n_rows, size=(boot, n_rows))
        w = np.stack([np.bincount(i, minlength=n_rows) for i in idx]).astype(np.float64)
        out = {}
        for name in sorted(self.d):
            rows = self.d[name]
            dim = next(iter(rows.values()))[0].shape[0]
            V = np.zeros((n_rows, dim))
            Q = np.zeros(n_rows)
            C = np.zeros(n_rows)
            for r, (vs, q, c) in rows.items():
                V[r], Q[r], C[r] = vs, q, c

            def share(Vs, Qs, Cs):
                m = Vs / np.maximum(Cs, 1e-12)[..., None]
                return (m ** 2).sum(-1) / np.maximum(Qs / np.maximum(Cs, 1e-12), 1e-30)

            pt = float(share(V.sum(0), Q.sum(), C.sum()))
            bs = share(w @ V, w @ Q, w @ C)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            # absolute sizes, per coordinate: the RMS of the vectors, and the RMS of their
            # deviation from the shared mean (the part that differs between slots / queries)
            msq = float(Q.sum() / max(C.sum(), 1e-12))
            out[name] = {"mean": pt, "lo": float(lo), "hi": float(hi), "n": float(C.sum()),
                         "rms": float(np.sqrt(msq / dim)),
                         "dev_rms": float(np.sqrt(max(1.0 - pt, 0.0) * msq / dim))}
        return out


# ── token categories (stage 1 and 4) ─────────────────────────────────────────────────────

_FUNC = set("""the a an of to in and or for on at by with from as is are was were be been it its
that this these those he she they we you i his her their our your not but if than then so
which who whom what when where how there here do does did has have had will would can could
may might must shall should about into over after before up down out off""".split())


class TokCat:
    def __init__(self, tok):
        self.tok = tok
        self.cache: dict[int, tuple[str, str]] = {}

    def __call__(self, tid: int) -> tuple[str, str]:
        if tid not in self.cache:
            s = self.tok.decode([int(tid)])
            st = s.strip()
            if "\n" in s:
                cat = "newline"
            elif st == "":
                cat = "space"
            elif all(ch in string.punctuation or ch in "—–“”’"
                     for ch in st):
                cat = "punct"
            elif st.replace(",", "").replace(".", "").isdigit():
                cat = "number"
            elif st.lower() in _FUNC:
                cat = "function_word"
            else:
                cat = "content"
            self.cache[tid] = (s, cat)
        return self.cache[tid]


# ── attention recomputation (exact copies of the shipped masks) ─────────────────────────

def window_weights(q, k, window_size, scale, n_skip_rope, extra_mask, relation):
    """[B, H, S, S] fp32, rows with no allowed key all zero (SDPA's -inf row)."""
    if n_skip_rope:
        raise RuntimeError("dataflow_probe: n_skip_rope > 0 is not modelled")
    S = q.shape[2]
    dev = q.device
    row = torch.arange(S, device=dev).unsqueeze(1)
    col = torch.arange(S, device=dev).unsqueeze(0)
    dist = row - col
    if relation is not None:
        mask = (dist.abs() < window_size) & (dist != 0)
    else:
        mask = (dist >= 0) & (dist < window_size) & (dist != 0)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if relation is not None:
        mask = mask & relation
    if extra_mask is not None:
        mask = mask & extra_mask
    sc = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale
    sc = sc.masked_fill(~mask, float("-inf"))
    w = torch.softmax(sc, dim=-1)
    return torch.nan_to_num(w, nan=0.0), mask.expand(q.shape[0], 1, S, S)[:, 0]


def comp_weights(q, k, slot_mask, sink_logits, scale, extra_mask, relation):
    """Compressed branch under tg_restrict. Returns ``(w [B,H,S,Mc+1] incl. sink, cols
    [B, Mc] key positions or None for the dense form, colvalid [B, Mc])``."""
    B, H, S, D = q.shape
    dev = q.device
    if slot_mask is None:
        if relation is not None:
            allow = relation.squeeze(1)
        else:
            r = torch.arange(S, device=dev).unsqueeze(1)
            c = torch.arange(S, device=dev).unsqueeze(0)
            allow = (c <= r).unsqueeze(0)
            if extra_mask is not None:
                allow = allow & extra_mask.squeeze(1)
        allow = allow.expand(B, S, S)
        sc = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale
        sc = sc.masked_fill(~allow.unsqueeze(1), float("-inf"))
        sink = sink_logits.float().view(1, H, 1, 1).expand(B, H, S, 1)
        w = torch.softmax(torch.cat([sc, sink], -1), dim=-1)
        return w, None, torch.ones(B, S, dtype=torch.bool, device=dev)
    M = int(slot_mask.sum(-1).max())
    idx = torch.argsort((~slot_mask).to(torch.int8), dim=-1, stable=True)[:, :M]
    valid = torch.gather(slot_mask, 1, idx)
    gidx = idx[:, None, :, None].expand(B, H, M, D)
    k_s = torch.gather(k, 2, gidx)
    sc = torch.einsum("bhid,bhjd->bhij", q.float(), k_s.float()) * scale
    row = torch.arange(S, device=dev).view(1, 1, S, 1)
    allow = (idx[:, None, None, :] <= row) & valid[:, None, None, :]
    if extra_mask is not None:
        em = extra_mask.squeeze(1)
        allow = allow & torch.gather(em, 2, idx[:, None, :].expand(B, S, M)).unsqueeze(1)
    sc = sc.masked_fill(~allow, float("-inf"))
    sink = sink_logits.float().view(1, H, 1, 1).expand(B, H, S, 1)
    w = torch.softmax(torch.cat([sc, sink], -1), dim=-1)
    return w, idx, valid


def rel_err(a, b, rows=None):
    a = a.float()
    b = b.float()
    if rows is not None:
        a = a[rows]
        b = b[rows]
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


# ── the probe ────────────────────────────────────────────────────────────────────────────

class Probe:
    """Hooks, wrappers and the per-forward analysis of one model."""

    def __init__(self, model, tokcat: TokCat, acc: RowAcc, vacc: VecAcc):
        self.m = model
        self.vacc = vacc
        self.tc = model.cfg.tul
        self.tokcat = tokcat
        self.acc = acc
        self.K = int(self.tc.prefix_k)
        self.Mf = int(self.tc.fan_k)
        self.n_core = len(model.core)
        self.n_coda = len(model.coda)
        self.snap = getattr(model, "tul_pseudo_snap", None) is not None
        self.tag = None
        self.pending: dict = {}
        self.selftest: dict[str, float] = defaultdict(float)   # max rel err per kind
        self.rowsum_err: dict[str, float] = defaultdict(float)  # max |sum w - 1|
        self.handles = []
        self.on = False
        self.counts = defaultdict(int)
        self.picks = defaultdict(lambda: defaultdict(int))     # name -> token string -> n
        self.ctx: dict = {}   # per-forward: layout, labels, depth, roles, queries ...

    # -- hooks ------------------------------------------------------------------------
    def _pre(self, region, li):
        def f(mod, args, kwargs=None):
            if not self.on:
                return
            if region == "core":
                c = self.counts["core"]
                self.tag = ("core", li, c // self.n_core)
                if c % self.n_core != li:
                    raise RuntimeError(f"core call order broke: call {c} is layer {li}")
                self.counts["core"] += 1
            elif region == "coda":
                c = self.counts["coda"]
                if c % self.n_coda != li:
                    raise RuntimeError(f"coda call order broke: call {c} is layer {li}")
                self.counts["coda"] += 1
                self.tag = ("coda", li)
                self.ctx.setdefault("coda_in", {})[li] = args[0].detach()
            else:
                self.tag = (region, li)
        return f

    def _post(self, region, li):
        def f(mod, args, out):
            if not self.on:
                return
            self.tag = None
            if region == "coda" and li == self.n_coda - 1:
                self.ctx["coda_out"] = (out[0] if isinstance(out, tuple) else out).detach()
        return f

    def attach(self):
        m = self.m
        for region, blocks in (("prelude", m.prelude), ("core", m.core), ("coda", m.coda)):
            for i, b in enumerate(blocks):
                self.handles.append(b.register_forward_pre_hook(self._pre(region, i)))
                self.handles.append(b.register_forward_hook(self._post(region, i)))
        for i, b in enumerate(m.coda):
            self.handles.append(b.norm_attn.register_forward_hook(self._norm_hook(i)))
        if m.tul_register is not None:
            self.handles.append(m.tul_register.register_forward_hook(self._reg_hook))
        if self.snap:
            self.handles.append(m.tul_pseudo_snap.register_forward_hook(self._snap_hook))

    def _norm_hook(self, li):
        """The coda attention's INPUT: ``x_bar`` (the Hyper-Connection mix of the streams)
        and ``norm_attn(x_bar)``. RMSNorm's eps is 1e-6, so an input of RMS << 1e-3 is
        NOT brought to unit RMS: that is the regime this reading checks."""
        def f(mod, args, out):
            if self.on and self.tag == ("coda", li):
                self.ctx.setdefault("norm_in", {})[li] = (args[0].detach(), out.detach())
        return f

    def _reg_hook(self, mod, args, out):
        if self.on:
            self.ctx["reg"] = (args[0].detach(), args[1], out.detach())

    def _snap_hook(self, mod, args, out):
        if self.on:
            c, ids, valid, _table = args
            self.ctx["snap"] = (ids.detach(), valid.detach(), out[1].detach(), out[2].detach())

    @contextlib.contextmanager
    def wrapped(self):
        """Wrap the three attention functions and ``prefix_project`` for one forward."""
        o_slot, o_win, o_gcu = (_attn._tg_slot_attention, _attn._window_fallback,
                                _attn._CCABase._gate_combine_up)
        probe = self

        def w_slot(q, k, v, slot_mask, sink_logits, scale, extra_mask=None, relation=None):
            out = o_slot(q, k, v, slot_mask, sink_logits, scale, extra_mask=extra_mask,
                         relation=relation)
            if probe.on and probe.tag and probe.tag[0] in ("core", "coda"):
                probe.pending["comp"] = (q, k, v, slot_mask, sink_logits, scale, extra_mask,
                                         relation, out)
            return out

        def w_win(q, k, v, window_size, device, scale, n_skip_rope=0, extra_mask=None,
                  relation=None):
            out = o_win(q, k, v, window_size, device, scale, n_skip_rope,
                        extra_mask=extra_mask, relation=relation)
            if probe.on and probe.tag and probe.tag[0] in ("core", "coda"):
                probe.pending["win"] = (q, k, v, window_size, scale, n_skip_rope, extra_mask,
                                        relation, out)
            return out

        def w_gcu(mod, x, out_comp, out_win, q_lat=None, gate_pre=None, win_capture=None):
            res = o_gcu(mod, x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre,
                        win_capture=win_capture)
            if probe.on and probe.tag and probe.tag[0] in ("core", "coda"):
                if "comp" not in probe.pending or "win" not in probe.pending:
                    raise RuntimeError(f"{probe.tag}: a branch call was not captured "
                                       f"({sorted(probe.pending)}); the fused path ran")
                with torch.no_grad():
                    probe._analyze(mod, x, q_lat, gate_pre, res)
            probe.pending = {}
            return res

        tul = self.m.tul
        orig_pp = tul.prefix_project

        def pp(*args, **kwargs):
            values, pos = orig_pp(*args, **kwargs)
            if probe.on:
                probe.ctx["pp"] = (values.detach(), pos.detach(),
                                   None if kwargs.get("cells") is None
                                   else kwargs["cells"].detach())
            return values, pos

        _attn._tg_slot_attention, _attn._window_fallback = w_slot, w_win
        _attn._CCABase._gate_combine_up = w_gcu
        tul.prefix_project = pp
        try:
            yield
        finally:
            _attn._tg_slot_attention, _attn._window_fallback = o_slot, o_win
            _attn._CCABase._gate_combine_up = o_gcu
            del tul.prefix_project

    # -- shared pieces -------------------------------------------------------------------
    def _gates(self, mod, x, gate_pre):
        B, S, _ = x.shape
        H = mod.n_heads
        g_lin = mod.gate(x) if gate_pre is None else mod.gate[2](mod.gate[1](gate_pre))
        return torch.sigmoid(g_lin.float()).reshape(B, S, H, 2).permute(0, 2, 1, 3)

    def _upv(self, mod, t):
        """[B, H, S, D] fp32 -> [B, S, d] W_up(concat heads), fp32."""
        B, H, S, D = t.shape
        z = t.transpose(1, 2).reshape(B, S, H * D).to(mod.W_up.weight.dtype)
        return mod.W_up(z).float()

    def _up(self, mod, t):
        """[B, H, S, D] fp32 -> [B, S] norm of W_up(concat heads)."""
        return self._upv(mod, t).norm(dim=-1)

    def _branch_checks(self, kind, ww, wmask, wc, cols, out_win, out_comp, v):
        has = wmask.any(-1)                                         # [B, S]
        rec_w = torch.einsum("bhij,bhjd->bhid", ww, v.float())
        rows = has.unsqueeze(1).expand_as(rec_w[..., 0])
        self.selftest[f"{kind}/win"] = max(self.selftest[f"{kind}/win"],
                                           rel_err(rec_w, out_win, rows))
        if cols is None:
            vs = v.float()
            rec_c = torch.einsum("bhij,bhjd->bhid", wc[..., :-1], vs)
        else:
            B, H, S, D = v.shape
            vs = torch.gather(v, 2, cols[:, None, :, None].expand(B, H, cols.shape[1], D))
            rec_c = torch.einsum("bhij,bhjd->bhid", wc[..., :-1], vs.float())
        self.selftest[f"{kind}/comp"] = max(self.selftest[f"{kind}/comp"],
                                            rel_err(rec_c, out_comp))
        sw = ww.sum(-1)
        self.rowsum_err[f"{kind}/win"] = max(self.rowsum_err[f"{kind}/win"],
                                             float((sw - 1).abs()[has.unsqueeze(1)
                                                                  .expand_as(sw)].max()))
        self.rowsum_err[f"{kind}/comp"] = max(self.rowsum_err[f"{kind}/comp"],
                                              float((wc.sum(-1) - 1).abs().max()))

    def _analyze(self, mod, x, q_lat, gate_pre, res):
        qc, kc, vc, slot_mask, sink, scale, ex_c, rel_c, out_comp = self.pending["comp"]
        qw, kw, vw, ws, scale_w, nskip, ex_w, rel_w, out_win = self.pending["win"]
        if not (qc is qw and kc is kw and vc is vw):
            raise RuntimeError("the two branches did not share q/k/v")
        ww, wmask = window_weights(qw, kw, ws, scale_w, nskip, ex_w, rel_w)
        wc, cols, colvalid = comp_weights(qc, kc, slot_mask, sink, scale, ex_c, rel_c)
        kind = self.tag[0]
        self._branch_checks(kind, ww, wmask, wc, cols, out_win, out_comp, vw)
        g = self._gates(mod, x, gate_pre)                            # [B, H, S, 2]
        gc, gw = g[..., 0], g[..., 1]
        gsum = gc + gw
        if q_lat is None:
            q_lat = mod.W_down_q(x)
        B, S, _ = x.shape
        H, D = mod.n_heads, mod.d_head
        x_res = q_lat.reshape(B, S, H, D).transpose(1, 2).float()
        res_part = mod.alpha.float() * x_res
        n_total = res.float().norm(dim=-1)                           # [B, S]
        n_res = self._up(mod, res_part)
        v = vw.float()
        if kind == "core":
            self._core_reads(ww, wc, gc, gw, gsum, v, n_total, n_res, mod)
        else:
            self._coda_reads(ww, wc, cols, colvalid, gc, gw, gsum, v, n_total, n_res, mod)

    # -- stage 3 -----------------------------------------------------------------------
    def _core_reads(self, ww, wc, gc, gw, gsum, v, n_total, n_res, mod):
        _, li, t = self.tag
        d = self.ctx["depth"]
        M = self.Mf
        B, H, P, _ = ww.shape
        valid = self.ctx["slot_valid"]                               # [B, S]
        S = valid.shape[1]
        if P != S * M:
            raise RuntimeError(f"core sequence {P} != S*M {S * M}")
        sl = torch.arange(P, device=ww.device) // M
        qv = valid.repeat_interleave(M, dim=1)                       # [B, P] query cells
        dsl = sl.view(P, 1) - sl.view(1, P)                          # slot distance i - j
        own = (dsl == 0)
        earl = (dsl > 0)
        wcd = wc[..., :-1]
        n_q = float(qv.sum())
        pre = f"core/d{d}"

        def put(name, val_bhp):          # head-averaged, query-masked, two keys
            s = float((val_bhp.mean(1) * qv).sum())
            self.acc.add(f"{pre}/{name}", s, n_q)
            self.acc.add(f"{pre}/t{t}/l{li}/{name}", s, n_q)

        m_own_w = (ww * own).sum(-1)
        m_earl_w = (ww * earl).sum(-1)
        m_own_c = (wcd * own).sum(-1)
        m_earl_c = (wcd * earl).sum(-1)
        m_sink = wc[..., -1]
        put("win_own", m_own_w)
        put("win_earlier", m_earl_w)
        put("comp_own", m_own_c)
        put("comp_earlier", m_earl_c)
        put("comp_sink", m_sink)
        put("comb_own", (gc * m_own_c + gw * m_own_w) / gsum)
        put("comb_earlier", (gc * m_earl_c + gw * m_earl_w) / gsum)
        put("comb_sink", gc * m_sink / gsum)
        put("gate_comp", gc)
        put("gate_win", gw)
        # how many earlier cells the read really averages: 1 / sum p^2 over the
        # gate-combined weights restricted to earlier cells (renormalised), against the
        # number of earlier valid cells the query can see
        wcomb = (gc.unsqueeze(-1) * wcd + gw.unsqueeze(-1) * ww) / gsum.unsqueeze(-1)
        pe = wcomb * earl
        pe = pe / pe.sum(-1, keepdim=True).clamp_min(1e-12)
        has_e = (earl.sum(-1) > 0).view(1, 1, P).float()
        put("neff_earlier", has_e / pe.pow(2).sum(-1).clamp_min(1e-12))
        put("n_earlier_cells", earl.sum(-1).float().view(1, 1, P).expand_as(gc))
        for nm, lo, hi in CORE_DIST:
            msk = (dsl >= lo) & (dsl <= hi)
            put(f"comb_earlier_{nm}",
                (gc * (wcd * msk).sum(-1) + gw * (ww * msk).sum(-1)) / gsum)
        o_own = gc.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", wcd * own, v) \
            + gw.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", ww * own, v)
        o_earl = gc.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", wcd * earl, v) \
            + gw.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", ww * earl, v)
        v_own, v_earl = self._upv(mod, o_own), self._upv(mod, o_earl)
        qb = qv.bool()
        self.vacc.add(f"{pre}/l{li}/cm_earlier", v_earl[qb])
        self.vacc.add(f"{pre}/l{li}/cm_own", v_own[qb])
        for nm, nrm in (("vnorm_own", v_own.norm(dim=-1)),
                        ("vnorm_earlier", v_earl.norm(dim=-1)),
                        ("vnorm_res", n_res), ("vnorm_total", n_total)):
            s = float((nrm * qv).sum())
            self.acc.add(f"{pre}/{nm}", s, n_q)
            self.acc.add(f"{pre}/t{t}/l{li}/{nm}", s, n_q)

    # -- stage 5 -----------------------------------------------------------------------
    def _coda_reads(self, ww, wc, cols, colvalid, gc, gw, gsum, v, n_total, n_res, mod):
        _, li = self.tag
        d = self.ctx["depth"]
        cx = self.ctx
        B, H, L, _ = ww.shape
        qm = cx["query"]                                             # [B, L] bool
        role = cx["role"]                                            # [B, L] int
        bag = cx["bag"]                                              # [B, L]
        n_q = float(qm.sum())
        pre = f"coda/d{d}"
        wcd = wc[..., :-1]
        crole = torch.gather(role, 1, cols)                          # [B, Mc]
        crole = torch.where(colvalid, crole, torch.full_like(crole, -1))
        cbag = torch.gather(bag, 1, cols)
        qbag = bag                                                   # query bag per row
        own_tok = (bag.unsqueeze(2) == bag.unsqueeze(1)) & (role == 0).unsqueeze(1)  # [B,L,L]

        def put(name, val_bhl, per_head=False):
            s_h = (val_bhl * qm.unsqueeze(1)).sum(dim=(0, 2))       # [H]
            s = float(s_h.mean())
            self.acc.add(f"{pre}/{name}", s, n_q)
            self.acc.add(f"{pre}/l{li}/{name}", s, n_q)
            if per_head:
                for h in range(H):
                    self.acc.add(f"{pre}/l{li}/h{h}/{name}", float(s_h[h]), n_q)
            return val_bhl

        def comb(mc, mw):
            return (gc * mc + gw * mw) / gsum

        # role ids: 0 token, 1 winner, 2 pseudo, 3 zero, 4 pad slot
        slot_w = (role >= 1).unsqueeze(1).unsqueeze(1)               # [B,1,1,L]
        m_slot_w = (ww * slot_w).sum(-1)
        m_tok_w = (ww * own_tok.unsqueeze(1)).sum(-1)
        m_slot_c = (wcd * (crole >= 1).unsqueeze(1).unsqueeze(1)).sum(-1)
        m_sink = wc[..., -1]
        put("win_slot", m_slot_w)
        put("win_owntok", m_tok_w)
        put("win_other", 1 - m_slot_w - m_tok_w)
        put("comp_slot", m_slot_c)
        put("comp_sink", m_sink)
        put("comb_slot", comb(m_slot_c, m_slot_w), per_head=True)
        put("comb_owntok", comb(torch.zeros_like(m_tok_w), m_tok_w))
        put("comb_sink", gc * m_sink / gsum)
        put("gate_comp", gc)
        put("gate_win", gw)
        # how many slot positions the compressed branch (which sees every earlier slot)
        # really averages: 1 / sum p^2 over its slot columns, renormalised
        ps = wcd * (crole >= 1).unsqueeze(1).unsqueeze(1)
        ps = ps / ps.sum(-1, keepdim=True).clamp_min(1e-12)
        put("comp_neff_slots", 1.0 / ps.pow(2).sum(-1).clamp_min(1e-12))
        allowed = (wcd > 0) & (crole >= 1).unsqueeze(1).unsqueeze(1)
        put("comp_n_slot_cols", allowed.float().sum(-1))
        rid = {"winner": 1, "pseudo": 2, "zero": 3, "pad": 4}
        o_parts = {}
        for nm, r in rid.items():
            mw_r = (role == r).unsqueeze(1).unsqueeze(1)
            mc_r = (crole == r).unsqueeze(1).unsqueeze(1)
            a_w = ww * mw_r
            a_c = wcd * mc_r
            put(f"comb_{nm}", comb(a_c.sum(-1), a_w.sum(-1)), per_head=(nm != "pad"))
            put(f"win_{nm}", a_w.sum(-1))
            put(f"comp_{nm}", a_c.sum(-1))
            vs = torch.gather(v, 2, cols[:, None, :, None].expand(B, H, cols.shape[1],
                                                                   v.shape[-1]))
            o_parts[nm] = gc.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", a_c, vs) \
                + gw.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", a_w, v)
        # slot distance (query span - slot's span), both branches
        dist_w = qbag.unsqueeze(2) - bag.unsqueeze(1)                # [B, L, L]
        dist_c = qbag.unsqueeze(2) - cbag.unsqueeze(1)               # [B, L, Mc]
        for nm, lo, hi in CORE_DIST:
            mw_d = ((dist_w >= lo) & (dist_w <= hi) & (role >= 1).unsqueeze(1)).unsqueeze(1)
            mc_d = ((dist_c >= lo) & (dist_c <= hi) & (crole >= 1).unsqueeze(1)).unsqueeze(1)
            put(f"comb_slot_{nm}", comb((wcd * mc_d).sum(-1), (ww * mw_d).sum(-1)))
            mc_dw = ((dist_c >= lo) & (dist_c <= hi) & (crole == 1).unsqueeze(1)).unsqueeze(1)
            mw_dw = ((dist_w >= lo) & (dist_w <= hi) & (role == 1).unsqueeze(1)).unsqueeze(1)
            put(f"comb_winner_{nm}", comb((wcd * mc_dw).sum(-1), (ww * mw_dw).sum(-1)))
        # by offset in span (the query's)
        comb_slot = comb(m_slot_c, m_slot_w).mean(1)                 # [B, L]
        for bi, (lo, hi) in enumerate(BINS):
            qb = qm & (cx["offbin"] == bi)
            nb = float(qb.sum())
            if nb:
                self.acc.add(f"{pre}/comb_slot_off{bi}", float((comb_slot * qb).sum()), nb)
                self.acc.add(f"{pre}/l{li}/comb_slot_off{bi}", float((comb_slot * qb).sum()),
                             nb)
        # value-weighted d_model norms
        o_slot = sum(o_parts.values())
        o_tok = gw.unsqueeze(-1) * torch.einsum("bhij,bhjd->bhid", ww * own_tok.unsqueeze(1), v)
        vecs = {"slot": self._upv(mod, o_slot), "tok": self._upv(mod, o_tok)}
        for nm in ROLES:
            vecs[nm] = self._upv(mod, o_parts[nm])
        norms = {"vnorm_res": n_res, "vnorm_total": n_total}
        for nm, vv in vecs.items():
            norms[f"vnorm_{nm}"] = vv.norm(dim=-1)
            self.vacc.add(f"{pre}/l{li}/cm_{nm}", vv[qm])
        # the coda attention's input at each role: x_bar RMS and norm_attn(x_bar) RMS
        xb, xo = cx["norm_in"][li]
        rid_all = {"token": 0, "winner": 1, "pseudo": 2, "zero": 3}
        for nm, r in rid_all.items():
            msk = qm if r == 0 else (role == r)
            n = float(msk.sum())
            if n:
                self.acc.add(f"{pre}/l{li}/xbar_rms_{nm}",
                             float((xb.float().pow(2).mean(-1).sqrt() * msk).sum()), n)
                self.acc.add(f"{pre}/l{li}/normed_rms_{nm}",
                             float((xo.float().pow(2).mean(-1).sqrt() * msk).sum()), n)
        for nm, nrm in norms.items():
            s = float((nrm * qm).sum())
            self.acc.add(f"{pre}/{nm}", s, n_q)
            self.acc.add(f"{pre}/l{li}/{nm}", s, n_q)
        # first-token queries (offset 0) separately, slot vs token output norm
        q0 = qm & (cx["offbin"] == 0)
        if float(q0.sum()):
            for nm in ("vnorm_slot", "vnorm_tok", "vnorm_total"):
                self.acc.add(f"{pre}/off0_{nm}", float((norms[nm] * q0).sum()),
                             float(q0.sum()))

    # -- per-forward context, run before the forward ---------------------------------------
    def begin(self, inp, labels, layout, depth):
        self.ctx = {"depth": depth, "slot_valid": layout.slot_valid,
                    "bag": layout.bag_id.long()}
        self.counts = defaultdict(int)
        L = inp.shape[1]
        B = inp.shape[0]
        off = np.stack([offsets_from_layout(layout, b) for b in range(B)])
        offbin = np.full(off.shape, -1)
        for b in range(B):
            for p in range(L):
                if off[b, p] >= 0:
                    offbin[b, p] = bin_of(int(off[b, p]))
        dev = layout.slot_valid.device
        self.ctx["offbin"] = torch.from_numpy(offbin).to(dev)
        self.ctx["query"] = (torch.from_numpy(off >= 0).to(dev)
                             & (labels.to(dev) >= 0) & ~layout.slot_mask)
        # roles need the winner: filled from the first coda call's context below. A coda
        # call reads `role` from ctx, so the forward is run in two steps: the role table is
        # built in `set_roles` from the write, which happens before the coda.
        self.ctx["role"] = torch.zeros(B, L, dtype=torch.long, device=dev)
        self.m._jac_capture = []
        self.m._lsel_capture = []
        if self.snap:
            self.m._pseudo_snap_capture = []
        self.on = True

    def end(self):
        self.on = False
        cap = {"jac": self.m._jac_capture, "lsel": self.m._lsel_capture,
               "snapcap": self.m._pseudo_snap_capture if self.snap else None}
        self.m._jac_capture = None
        self.m._lsel_capture = None
        self.m._pseudo_snap_capture = None
        return cap


def build_roles(probe: Probe, layout, B, L):
    """Role per position from the write: 0 token, 1 winner, 2 pseudo, 3 zero, 4 pad slot.
    Called from inside the forward (a hook on coda block 0's pre-call), after the write."""
    values, pos, cells = probe.ctx["pp"]
    K = probe.K
    S = layout.slot_valid.shape[1]
    if cells is None:
        raise RuntimeError("prefix_project got no cells: not a fan write")
    # the winner is the ONE non-zero cell (`_fan_route_cells` zeroes every loser exactly)
    mag = cells.float().flatten(3).abs().sum(-1)                     # [B, S, K]
    winner = mag.argmax(-1)
    nz = (mag > 0).sum(-1)
    valid = layout.slot_valid
    if bool(((nz != 1) & valid).any()):
        raise RuntimeError(f"a valid slot has {int(nz[valid].max())} non-zero cells; the "
                           "hard winner write is not what this probe assumes")
    role = torch.zeros(B, L, dtype=torch.long, device=pos.device)
    role[layout.slot_mask] = 4
    k_idx = torch.arange(K, device=pos.device).view(1, 1, K).expand(B, S, K)
    is_w = k_idx == winner.unsqueeze(-1)
    r = torch.where(is_w, torch.ones_like(k_idx),
                    torch.full_like(k_idx, 2 if probe.snap else 3))
    r = torch.where(valid.unsqueeze(-1), r, torch.full_like(r, 4)).reshape(B, S * K)
    pv = pos.reshape(B, S * K)
    inrow = pv < L
    for b in range(B):
        role[b, pv[b][inrow[b]]] = r[b][inrow[b]]
    if bool((role[layout.slot_mask] == 0).any()):
        raise RuntimeError("a slot position got no role")
    probe.ctx["role"] = role
    probe.ctx["winner"] = winner


# ── stage 1, 2, 4 readers (after the forward) ─────────────────────────────────────────────

def read_seed(probe: Probe, inp, layout, acc: RowAcc, depth: int):
    xn, lay, out = probe.ctx["reg"]
    reg = probe.m.tul_register
    B, L, C = xn.shape
    S = lay.slot_index.shape[1]
    M, H, Dh = reg.m, reg.n_heads, reg.d_head
    k = reg.W_k(xn).view(B, L, H, Dh)
    v = reg.W_v(xn).view(B, L, H, Dh)
    q = (reg.Q if reg.distinct else reg.Q.expand(M, C)).view(M, H, Dh)
    sc = torch.einsum("mhd,blhd->bhml", q.to(xn.dtype), k).float() * reg.scale
    tok = ~lay.slot_mask
    own = (lay.bag_id.unsqueeze(1) == torch.arange(S, device=xn.device).view(1, S, 1)) \
        & tok.unsqueeze(1)                                           # [B, S, L]
    has = own.any(-1)
    own_o = torch.where(has.unsqueeze(-1), own, torch.ones_like(own))
    logits = sc.unsqueeze(1).masked_fill(~own_o.view(B, S, 1, 1, L), float("-inf"))
    a = torch.softmax(logits, dim=-1)                                # [B, S, H, M, L]
    pooled = torch.einsum("bshml,blhd->bshmd", a.to(v.dtype), v)
    pooled = pooled.permute(0, 1, 3, 2, 4).reshape(B, S, M, C)
    rec = (reg.W_o(pooled) + reg.P_cell.to(xn.dtype).view(1, 1, M, C)) \
        * lay.slot_valid.view(B, S, 1, 1).to(xn.dtype)
    probe.selftest["seed/pool"] = max(probe.selftest["seed/pool"],
                                      rel_err(rec.reshape(B, S * M, C), out))
    probe.rowsum_err["seed/pool"] = max(
        probe.rowsum_err["seed/pool"],
        float((a.sum(-1) - 1).abs()[lay.slot_valid].max()))
    am = a.mean(2)                                                   # [B, S, M, L] head-avg
    pre = f"seed/d{depth}"
    for b in range(B):
        for s in range(S):
            if not bool(lay.slot_valid[b, s]) or not bool(has[b, s]):
                continue
            ps = own[b, s].nonzero().flatten()
            n = int(ps.numel())
            first, last = int(ps[0]), int(ps[-1])
            for i in range(M):
                w = am[b, s, i, ps].float()
                ent = float(-(w.clamp_min(1e-12) * w.clamp_min(1e-12).log()).sum())
                top = float(w.max())
                arg = int(ps[int(w.argmax())])
                wb, wf = float(am[b, s, i, last]), float(am[b, s, i, first])
                for key, val in (("entropy", ent), ("entropy_norm", ent / max(np.log(n), 1e-9)),
                                 ("top1", top), ("boundary_mass", wb), ("first_mass", wf),
                                 ("uniform_mass", 1.0 / n), ("span_len", n),
                                 ("top1_gt_0.3", float(top > 0.3)),
                                 ("bnd_or_first_gt_0.3", float(wb > 0.3 or wf > 0.3)),
                                 ("argmax_is_boundary", float(arg == last)),
                                 ("argmax_is_first", float(arg == first))):
                    acc.add(f"{pre}/{key}", val, 1)
                    acc.add(f"{pre}/cell{i}/{key}", val, 1)
                s_tok, cat = probe.tokcat(int(inp[b, arg]))
                for c2 in CATS:
                    acc.add(f"{pre}/argmax_cat_{c2}", float(c2 == cat), 1)
                probe.picks[f"seed_cell{i}"][s_tok] += 1
            span_cats = [probe.tokcat(int(inp[b, p]))[1] for p in ps.tolist()]
            for c2 in CATS:                                          # base rate of categories
                acc.add(f"{pre}/span_cat_{c2}", sum(c == c2 for c in span_cats), n)
    # seed magnitudes: the register term against the normed prelude state at the cells
    values, pos, _ = probe.ctx["pp"]
    K = probe.K
    pv = pos.reshape(B, S * K).clamp(max=L - 1)
    xhat = torch.gather(xn, 1, pv.unsqueeze(-1).expand(B, S * K, C)).float()
    rt = out.float()
    vmask = lay.slot_valid.repeat_interleave(M, 1).float()
    rms = lambda t: t.pow(2).mean(-1).sqrt()                         # noqa: E731
    acc.add(f"{pre}/xhat_rms", float((rms(xhat) * vmask).sum()), float(vmask.sum()))
    acc.add(f"{pre}/regterm_rms", float((rms(rt) * vmask).sum()), float(vmask.sum()))
    # how different are the 4 seeds of a slot (the register's job): pairwise cosine of e_i
    e = (xhat + rt).view(B, S, M, C)
    en = F.normalize(e, dim=-1)
    cos = torch.einsum("bsic,bsjc->bsij", en, en)
    off = ~torch.eye(M, dtype=torch.bool, device=cos.device)
    pc = cos[..., off].mean(-1)
    vs = lay.slot_valid.float()
    acc.add(f"{pre}/seed_pair_cos", float((pc * vs).sum()), float(vs.sum()))
    vb = lay.slot_valid
    probe.vacc.add(f"{pre}/cm_seed_e", e[vb].reshape(-1, C))
    probe.vacc.add(f"{pre}/cm_seed_xhat", xhat.view(B, S, M, C)[vb].reshape(-1, C))
    probe.vacc.add(f"{pre}/cm_seed_regterm", rt.view(B, S, M, C)[vb].reshape(-1, C))


def read_loop(probe: Probe, cap, layout, acc: RowAcc, depth: int):
    """Two VIEWS of every cell: ``full`` (all Hyper-Connection streams flattened, the
    carrier the loop carries) and ``sm_`` (the stream MEAN, what every per-cell reader in
    the tree reads, ``_cell_readout``). The full carrier holds a large per-stream constant
    that cancels in the mean (its common-mode share is ~0.99, the mean's ~0.02-0.16), so a
    relative change or cosine on the full view is dominated by that constant."""
    jac, lsel = cap["jac"], cap["lsel"]
    M = probe.Mf
    B, S = layout.slot_valid.shape
    if len(jac) != depth or len(lsel) != depth:
        raise RuntimeError(f"depth {depth}: {len(jac)} pass entries, {len(lsel)} lsel passes")
    pre = f"loop/d{depth}"
    vc = layout.slot_valid.repeat_interleave(M, 1).float()           # [B, S*M]
    vs = layout.slot_valid.float()
    vb = layout.slot_valid
    views = {"": lambda t: t.float().flatten(2), "sm_": lambda t: t.float().mean(2)}
    for t in range(depth):
        if t > 0:
            probe.selftest["loop/entry_eq_prev_post"] = max(
                probe.selftest["loop/entry_eq_prev_post"],
                rel_err(jac[t]["h"].float(), lsel[t - 1]["post"].float()))
        follow = lsel[t]["follow"]
        if t > 0:
            sw = (follow != lsel[t - 1]["follow"]).float()
            acc.add(f"{pre}/t{t}/switch", float((sw * vs).sum()), float(vs.sum()))
        for i in range(M):
            acc.add(f"{pre}/t{t}/winner_share_k{i}", float(((follow == i).float() * vs).sum()),
                    float(vs.sum()))

        def pick(cells, idx):
            return torch.gather(cells, 2, idx.view(B, S, 1, 1).expand(
                B, S, 1, cells.shape[-1])).squeeze(2)

        for vn, f in views.items():
            hin = f(jac[t]["h"])                                     # [B, S*M, D]
            out = f(lsel[t]["pre"])
            post = f(lsel[t]["post"])
            D = out.shape[-1]
            rc = (out - hin).norm(dim=-1) / out.norm(dim=-1).clamp_min(1e-12)
            acc.add(f"{pre}/t{t}/{vn}pass_rel_change", float((rc * vc).sum()), float(vc.sum()))
            acc.add(f"{pre}/t{t}/{vn}cell_rms", float((out.pow(2).mean(-1).sqrt() * vc).sum()),
                    float(vc.sum()))
            cells = out.view(B, S, M, D)
            w = pick(cells, follow)
            if t == 0:
                probe.ctx[f"loop_w0_{vn}"] = w
                r0 = (w - pick(hin.view(B, S, M, D), follow)).norm(dim=-1) \
                    / w.norm(dim=-1).clamp_min(1e-12)
                acc.add(f"{pre}/t0/{vn}winner_rel_change", float((r0 * vs).sum()),
                        float(vs.sum()))
                probe.vacc.add(f"{pre}/{vn}cm_entry", pick(hin.view(B, S, M, D), follow)[vb])
            else:
                wp = probe.ctx[f"loop_wprev_{vn}"]
                r = (w - wp).norm(dim=-1) / w.norm(dim=-1).clamp_min(1e-12)
                acc.add(f"{pre}/t{t}/{vn}winner_rel_change", float((r * vs).sum()),
                        float(vs.sum()))
            probe.ctx[f"loop_wprev_{vn}"] = w
            nm = "cm_winner" if vn == "" else "cm_winner_streammean"
            probe.vacc.add(f"{pre}/t{t}/{nm}", w[vb])
            cosw = F.cosine_similarity(w, probe.ctx[f"loop_w0_{vn}"], dim=-1)
            acc.add(f"{pre}/t{t}/{vn}winner_cos_to_pass1", float((cosw * vs).sum()),
                    float(vs.sum()))
            # spread among the M candidates BEFORE the reset
            cn = F.normalize(cells, dim=-1)
            pc = torch.einsum("bsic,bsjc->bsij", cn, cn)
            off = ~torch.eye(M, dtype=torch.bool, device=pc.device)
            acc.add(f"{pre}/t{t}/{vn}cell_pair_cos", float((pc[..., off].mean(-1) * vs).sum()),
                    float(vs.sum()))
            mu = cells.mean(2, keepdim=True)
            dev = (cells - mu).pow(2).mean(dim=(2, 3)).sqrt()
            base = mu.squeeze(2).pow(2).mean(-1).sqrt().clamp_min(1e-12)
            acc.add(f"{pre}/t{t}/{vn}cell_spread", float(((dev / base) * vs).sum()),
                    float(vs.sum()))
            # how far the reset moved the cells (0 at the last pass)
            rr = (post - out).norm(dim=-1) / out.norm(dim=-1).clamp_min(1e-12)
            acc.add(f"{pre}/t{t}/{vn}reset_rel_move", float((rr * vc).sum()), float(vc.sum()))
            # the change of the carried cell measured against its SLOT-SPECIFIC part only:
            # the deviation from the batch mean of the winners (removes the shared constant)
            if t > 0:
                wd = w - w[vb].mean(0)
                wpd = wp - wp[vb].mean(0) if t > 0 else None
                r = (wd - wpd).norm(dim=-1) / wd.norm(dim=-1).clamp_min(1e-12)
                acc.add(f"{pre}/t{t}/{vn}winner_rel_change_centred", float((r * vs).sum()),
                        float(vs.sum()))


def read_write(probe: Probe, inp, layout, acc: RowAcc, depth: int):
    cx = probe.ctx
    role = cx["role"]
    pre = f"write/d{depth}"
    rms = lambda t: t.float().mean(2).pow(2).mean(-1).sqrt()         # noqa: E731  stream mean
    rms_full = lambda t: t.float().pow(2).mean(dim=(-2, -1)).sqrt()  # noqa: E731
    names = {0: "token", 1: "winner", 2: "pseudo", 3: "zero"}
    tokq = cx["query"]
    for li in range(probe.n_coda):
        x = cx["coda_in"][li]
        r_mean, r_full = rms(x), rms_full(x)
        for rid, nm in names.items():
            msk = tokq if rid == 0 else (role == rid)
            n = float(msk.sum())
            if n:
                acc.add(f"{pre}/coda_in_l{li}/{nm}_rms", float((r_mean * msk).sum()), n)
                acc.add(f"{pre}/coda_in_l{li}/{nm}_rms_full", float((r_full * msk).sum()), n)
    x = cx["coda_out"]
    r_mean = rms(x)
    for rid, nm in names.items():
        msk = tokq if rid == 0 else (role == rid)
        n = float(msk.sum())
        if n:
            acc.add(f"{pre}/coda_out/{nm}_rms", float((r_mean * msk).sum()), n)
    # the written vectors themselves (prefix_project + the snap write), before the coda
    x0 = cx["coda_in"][0]
    if probe.snap:
        ids, valid, p, vertex = cx["snap"]
        B, S, k, J = p.shape
        vs = layout.slot_valid
        g = probe.m.tul_pseudo_snap.g.detach().float().cpu().tolist()
        beta = probe.m.tul_pseudo_snap.beta.detach().float().cpu().tolist()
        cx["snap_params"] = {"g": g, "beta": beta}
        ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)    # [B,S,k]
        nval = valid.sum(-1)                                              # [B,S]
        amax = p.argmax(-1)                                               # [B,S,k]
        picked = torch.gather(ids.unsqueeze(2).expand(B, S, k, J), -1, amax.unsqueeze(-1)
                              ).squeeze(-1)
        last = (nval - 1).clamp_min(0)
        L = inp.shape[1]
        later_tok = []
        for b in range(B):
            for s in range(S):
                if not bool(vs[b, s]):
                    continue
                n = int(nval[b, s])
                laterpos = ((layout.bag_id[b] > s) & ~layout.slot_mask[b]).nonzero().flatten()
                later = set(inp[b, laterpos.cpu()].tolist())
                own_ids = ids[b, s, :n].tolist()
                base_hit = np.mean([t_ in later for t_ in own_ids])
                acc.add(f"{pre}/pseudo_copy_hit_base", base_hit, 1)
                acc.add(f"{pre}/pseudo_distinct_picks", len(set(picked[b, s].tolist())), 1)
                for nn in range(k):
                    pk = int(picked[b, s, nn])
                    s_tok, cat = probe.tokcat(pk)
                    probe.picks[f"pseudo{nn}"][s_tok] += 1
                    a = int(amax[b, s, nn])
                    for key, val in (("vertex_mass", float(vertex[b, s, nn])),
                                     ("entropy", float(ent[b, s, nn])),
                                     ("entropy_norm", float(ent[b, s, nn]) / max(np.log(n), 1e-9)),
                                     ("boundary_mass", float(p[b, s, nn, int(last[b, s])])),
                                     ("first_mass", float(p[b, s, nn, 0])),
                                     ("uniform_mass", 1.0 / n),
                                     ("pick_is_boundary", float(a == int(last[b, s]))),
                                     ("pick_is_first", float(a == 0)),
                                     ("vertex_gt_0.9", float(vertex[b, s, nn] > 0.9)),
                                     ("copy_hit", float(pk in later))):
                        acc.add(f"{pre}/pseudo_{key}", val, 1)
                        acc.add(f"{pre}/pseudo{nn}_{key}", val, 1)
                    for c2 in CATS:
                        acc.add(f"{pre}/pseudo_cat_{c2}", float(c2 == cat), 1)
    # cosine of the winner written vector to the pseudo vectors of the same slot, and the
    # norm of each against the token positions (the same numbers coda_in_l0 holds, but
    # paired per slot here)
    values, pos, _ = cx["pp"]
    winner = cx["winner"]
    B, S = layout.slot_valid.shape
    K = probe.K
    L = x0.shape[1]
    pv = pos.reshape(B, S, K).clamp(max=L - 1)
    vs = layout.slot_valid.float()
    xs = torch.gather(x0.float().mean(2), 1,
                      pv.reshape(B, S * K, 1).expand(B, S * K, x0.shape[-1])
                      ).view(B, S, K, -1)
    wv = torch.gather(xs, 2, winner.view(B, S, 1, 1).expand(B, S, 1, xs.shape[-1])).squeeze(2)
    vb = layout.slot_valid
    probe.vacc.add(f"{pre}/cm_written_winner", wv[vb])
    probe.vacc.add(f"{pre}/cm_coda_in_tokens", x0.float().mean(2)[cx["query"]])
    probe.vacc.add(f"{pre}/cm_coda_in_tokens_allstreams", x0.float().flatten(2)[cx["query"]])
    xfull = torch.gather(x0.float().flatten(2), 1,
                         pv.reshape(B, S * K, 1).expand(B, S * K, x0[0, 0].numel())
                         ).view(B, S, K, -1)
    wfull = torch.gather(xfull, 2, winner.view(B, S, 1, 1).expand(B, S, 1, xfull.shape[-1])
                         ).squeeze(2)
    probe.vacc.add(f"{pre}/cm_written_winner_allstreams", wfull[vb])
    if probe.snap:
        for kk in range(1, K):
            idx = (winner + kk) % K
            ov = torch.gather(xs, 2, idx.view(B, S, 1, 1).expand(B, S, 1, xs.shape[-1])
                              ).squeeze(2)
            probe.vacc.add(f"{pre}/cm_written_pseudo", ov[vb])
    for kk in range(1, K):
        idx = (winner + kk) % K
        ov = torch.gather(xs, 2, idx.view(B, S, 1, 1).expand(B, S, 1, xs.shape[-1])).squeeze(2)
        cs = F.cosine_similarity(wv, ov, dim=-1)
        acc.add(f"{pre}/cos_winner_to_cell+{kk}", float((torch.nan_to_num(cs) * vs).sum()),
                float(vs.sum()))


# ── stage 7 forwards ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def forward_ce(model, inp, labels, layout, mode: str, device: str) -> torch.Tensor:
    """[B, L] fp32 CE. ``mode``: normal | zero | winner_only | pseudo_only | both_manual."""
    tul = model.tul
    patches = []
    if mode in ("winner_only", "both_manual"):
        orig = tul.prefix_project
        K = int(model.cfg.tul.prefix_k)

        def cut(*args, **kwargs):
            values, pos = orig(*args, **kwargs)
            cells = kwargs.get("cells")
            if cells is None:
                raise RuntimeError("winner_only needs the fan's cells")
            mag = cells.float().flatten(3).abs().sum(-1)            # [B, S, K]
            w = mag.argmax(-1)
            B, S = w.shape
            flat = (w + torch.arange(S, device=w.device).view(1, S) * K)   # [B, S]
            values = values.clone()
            keep = torch.ones(values.shape[:2], dtype=values.dtype, device=values.device)
            keep.scatter_(1, flat, 0.0)
            values = values * keep.view(*keep.shape, *([1] * (values.dim() - 2)))
            return values, pos

        tul.prefix_project = cut
        patches.append(lambda: delattr(tul, "prefix_project"))
    if mode in ("pseudo_only", "both_manual"):
        if getattr(model, "tul_pseudo_snap", None) is None:
            raise RuntimeError(f"{mode} on a model without the snap")

        def skip(values, *a, **k):
            return values

        model._tul_pseudo_snap_write = skip
        patches.append(lambda: delattr(model, "_tul_pseudo_snap_write"))
    plan_mode = "zero" if mode == "zero" else "normal"
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode=plan_mode)
    finally:
        for p in patches:
            p()
    return ce_of(res["logits"], labels, device)


def ce_of(logits, labels, device):
    lg = logits.float()
    B, L, V = lg.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(lg.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L)


# ── main ─────────────────────────────────────────────────────────────────────────────────

def run_arm(spec: str, a) -> dict:
    t0 = time.time()
    label, config, path, ovr = parse_ckpt_spec(spec)
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from transformers import AutoTokenizer

    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    device = a.device
    torch.manual_seed(a.seed)
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    tc = model.cfg.tul
    own_depth = int(tc.slot_mean_depth or model.cfg.mean_depth)
    spec_ = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, a.rows * (spec_.l_total + 1))
    batches = pack_rows(stream, tul_rt, cfg, 1, False)[:a.rows]
    tokcat = TokCat(AutoTokenizer.from_pretrained(cfg.data.tokenizer))
    acc = RowAcc()
    vacc = VecAcc()
    probe = Probe(model, tokcat, acc, vacc)
    probe.attach()
    sanity = {"hook_vs_plain_max_abs_ce_diff": 0.0, "hook_vs_plain_bit_equal": True}
    depths = [own_depth] + [d for d in a.depths if d != own_depth]
    modes = ["zero", "winner_only"] + (["pseudo_only", "both_manual"] if probe.snap else [])
    tok_first = tok_last = None
    for r, (inp, labels, layout, idx) in enumerate(batches):
        acc.row = r
        vacc.row = r
        lay = layout.to(device)
        B, L = inp.shape
        tokpos = (~layout.slot_mask) & (labels >= 0)
        ti = idx[~layout.slot_mask]
        tok_first = int(ti[0]) if tok_first is None else tok_first
        tok_last = int(ti[-1])
        # ---- scored set: the worth_profile strata (bag >= 1, not dump, label >= 0) ----
        off = offsets_from_layout(layout, 0)
        score = torch.from_numpy(off >= 0) & (labels[0] >= 0) & ~layout.slot_mask[0]
        offb = torch.tensor([bin_of(int(o)) if o >= 0 else -1 for o in off])
        # exact-recall buckets on the row's own token stream
        tok_positions = (~layout.slot_mask[0]).nonzero().flatten()
        first_si = int(idx[0, tok_positions[0]])
        last_si = int(idx[0, tok_positions[-1]])
        x_row = np.asarray(stream[first_si:last_si + 2], dtype=np.int64)
        loc, bkt, _dist = row_buckets(x_row, MIN_I, FAR_GAP)
        rb = torch.full((L,), -1, dtype=torch.long)
        rb[tok_positions[torch.from_numpy(loc)]] = torch.from_numpy(bkt.astype(np.int64))
        # ---- plain normal forward (no hooks active) ----
        tc.slot_mean_depth = own_depth
        ce_n = forward_ce(model, inp, labels, lay, "normal", device)[0].cpu()
        # ---- instrumented forwards ----
        ce_d = {}
        for d in depths:
            tc.slot_mean_depth = d
            probe.begin(inp, labels, lay, d)
            role_hook = model.coda[0].register_forward_pre_hook(
                lambda mod, args: build_roles(probe, lay, B, L) if probe.on else None)
            try:
                with probe.wrapped():
                    with torch.no_grad(), torch.autocast(
                            "cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                        res = model.tul_forward_ablated(inp.to(device), None, lay,
                                                        plan_mode="normal")
            finally:
                role_hook.remove()
                cap = probe.end()
            if probe.counts["core"] != probe.n_core * d or probe.counts["coda"] != probe.n_coda:
                raise RuntimeError(f"depth {d}: {probe.counts['core']} core calls, "
                                   f"{probe.counts['coda']} coda calls")
            ce_d[d] = ce_of(res["logits"], labels, device)[0].cpu()
            with torch.no_grad():
                if d == own_depth:
                    read_seed(probe, inp, lay, acc, d)
                read_loop(probe, cap, lay, acc, d)
                read_write(probe, inp, lay, acc, d)
            if d == own_depth:
                diff = float((ce_d[d] - ce_n)[tokpos[0]].abs().max())
                sanity["hook_vs_plain_max_abs_ce_diff"] = max(
                    sanity["hook_vs_plain_max_abs_ce_diff"], diff)
                if not torch.equal(ce_d[d][tokpos[0]], ce_n[tokpos[0]]):
                    sanity["hook_vs_plain_bit_equal"] = False
        tc.slot_mean_depth = own_depth
        # ---- K-curve on these rows (stage 6) ----
        n_sc = float(score.sum())
        for d in depths:
            acc.add(f"ce/d{d}", float(ce_d[d][score].sum()), n_sc)
            acc.add(f"ce/K{d}-K{own_depth}", float((ce_d[d] - ce_d[own_depth])[score].sum()),
                    n_sc)
        # ---- stage 7 ablations ----
        for m in modes:
            ce_m = forward_ce(model, inp, labels, lay, m, device)[0].cpu()
            dlt = ce_m - ce_n
            acc.add(f"worth/{m}/TOTAL", float(dlt[score].sum()), n_sc)
            for bi in range(len(BINS)):
                msk = score & (offb == bi)
                if bool(msk.any()):
                    acc.add(f"worth/{m}/off{bi}", float(dlt[msk].sum()), float(msk.sum()))
            for code, nm in RECALL.items():
                msk = score & (rb == code)
                if bool(msk.any()):
                    acc.add(f"worth/{m}/{nm}", float(dlt[msk].sum()), float(msk.sum()))
        for code, nm in RECALL.items():
            msk = score & (rb == code)
            acc.add(f"recall_share/{nm}", float(msk.sum()), n_sc)
        if r % 4 == 0 or r == len(batches) - 1:
            print(f"[{label}] row {r + 1}/{len(batches)}  {time.time() - t0:.0f}s", flush=True)
    tc.slot_mean_depth = own_depth
    for h in probe.handles:
        h.remove()
    n_rows = len(batches)
    out = {
        "label": label, "config": config, "ckpt": path, "step": step, "rows": n_rows,
        "batch": 1, "own_depth": own_depth, "depths": depths, "seed": a.seed,
        "stream_index_first": tok_first, "stream_index_last": tok_last,
        "snap": probe.snap, "modes": modes,
        "snap_params": probe.ctx.get("snap_params"),
        "injection_decay_A": None,
        "sanity": {**sanity,
                   "selftest_max_rel_err": dict(probe.selftest),
                   "attn_rowsum_max_abs_err": dict(probe.rowsum_err)},
        "top_picks": {k: sorted(v.items(), key=lambda kv: -kv[1])[:25]
                      for k, v in probe.picks.items()},
        "readings": acc.summary(n_rows, a.boot, a.seed),
        "common_mode": vacc.summary(n_rows, a.boot, a.seed),
        "wall_s": time.time() - t0,
    }
    inj = getattr(model, "injection", None)
    if inj is not None and hasattr(inj, "log_A"):
        out["injection_decay_A"] = float(inj.log_A.detach().float().exp().mean())
    print(f"[{label}] sanity {json.dumps(out['sanity'])}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr,...]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--depths", default="6,1",
                    help="forced depths; the arm's own eval depth is always run first")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.depths = [int(x) for x in a.depths.split(",") if x]
    res = run_arm(a.ckpt, a)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
