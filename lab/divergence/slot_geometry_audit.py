"""Is the shared core MUTED on the slot loop's compact sequence, and by what?

The narrow question. Not "why does the loop not earn depth" — the narrower one: why
does even ONE pass of a 6-block core change nothing the coda reads. The z-optimisation
probe (`slot_z_optimize.py`) says the coda's CE with the loop's EXIT state equals its CE
with the loop's ENTRY state to 0.003 nats, while a gradient-fitted z is worth 0.9-2.6
nats to the SAME frozen coda. So the reader is fine and the writer writes nothing.

Six leads, all measured on a real slot-arm checkpoint at the real shape:

1. **Attention geometry at S = 64.** Per core block: the compressor's `n_blocks`, the
   CSA top-k that actually fires, the gate's split between the compressed and window
   branches, the ABSOLUTE norm of each branch and of the CCA residual term, the window
   branch's entropy / participation ratio / mass on the immediately previous cell, and
   what cell 0 attends (XSA excludes self, so its softmax row can be empty). The SAME
   block on the SAME weights over the checkpoint's own TOKEN sequence (S = 1152) is the
   control.
2. **Branch outputs, absolute and relative.** `out/in` on a pre-norm block is a ratio: a
   large carrier shrinks it without shrinking the update. This reads the ABSOLUTE
   per-position norms of the attention and MLP branch outputs, of the carrier they land
   in, and of the carrier update each HC residual actually writes — on slots per pass and
   on tokens under one application of the same weights.
3. **The injection.** `DiagonalInjection` rewrites the ctx channel from `e` on EVERY
   pass, and each core layer adds an x0/bigram term. Per pass: the size of both against
   the carrier and against the blocks' own update. If the injection re-imposes `e` faster
   than the blocks move away from it, the exit state is a function of `e` and the loop is
   decorative.
4. **The readout.** The coda cell IS `W_prefix[k] · z` (scatter REPLACES). So
   `|W_k·(exit − entry)| / |W_k·entry|` is exactly the loop's share of what the coda
   receives. Also the stream-mean collapse (the MUX and `unpack` read `h.mean(dim=2)`)
   and the cosine of the loop's update with `W_prefix`'s top singular directions.
5. **K0 / K1 / K6 at the coda.** The write contribution per pass: the trainer's token CE
   with z = the state after exactly t passes, t = 0..depth, through the SHIPPED
   downstream code (`slot_z_optimize.ZSplit`). Plus one counterfactual: the same at
   depth T with the injection cut after pass 1.
6. **Norm growth.** The slot carrier's norm per pass against the token carrier's at the
   same entry, so a pre-norm reader's view is a stated number and not an inference.

Every reading is gated: the state this script captures after the last pass must be the
tensor `prefix_project` receives, bit for bit, and substituting it must reproduce the
trained forward's token CE bit for bit.

Usage:
  python lab/divergence/slot_geometry_audit.py \
      --ckpt LABEL=CONFIG=PATH --rows 12 --batch 3 --depth 6 \
      --out ignored/experiment-artifacts/<slug>/LABEL.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from slot_z_optimize import ZSplit, guard_split_point  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")


# ── small helpers ─────────────────────────────────────────────────────────────


def _pp_norm(t: torch.Tensor, rows: torch.Tensor | None) -> float:
    """Mean over selected positions of the per-position L2 norm.

    `t` is [B, S, C] or [B, S, n, C]; `rows` is a [B, S] bool selection (None = all).
    Per POSITION, never over the whole tensor: S differs by 18x between the slot core
    (64) and the token sequence (1152), and a whole-tensor norm would carry that
    difference into every comparison.
    """
    z = t.float().flatten(2) if t.dim() == 4 else t.float()
    per = z.norm(dim=-1)                                          # [B, S]
    if rows is None:
        return float(per.mean())
    m = rows.to(per.dtype)
    return float((per * m).sum() / m.sum().clamp_min(1.0))


def window_weights(q: torch.Tensor, k: torch.Tensor, window_size: int, scale: float,
                   n_skip_rope: int = 0, extra_mask: torch.Tensor | None = None):
    """Explicit fp32 attention weights of the window branch, plus the per-row key count.

    The shipped path never materialises them: `_CCABase._window_attn` calls SDPA (or a
    fused kernel), so entropy and per-key mass are unreadable without rebuilding the
    weights. The mask rule is `attention._window_fallback`'s, line for line — causal,
    strictly inside `window_size`, self EXCLUDED (XSA), the skip-rope suffix, and the
    optional TG extra mask. `tests/test_slot_geometry_audit.py` checks that `w @ v`
    reproduces `_window_fallback`'s own output, so "the same mask" is a verified claim
    and not a reading of the source.

    A row with no visible key (query 0 under XSA) softmaxes to NaN; it is zeroed here and
    reported separately, never averaged into a statistic.
    """
    S = q.shape[2]
    dev = q.device
    row = torch.arange(S, device=dev).unsqueeze(1)
    col = torch.arange(S, device=dev).unsqueeze(0)
    dist = row - col
    mask = (dist >= 0) & (dist < window_size) & (dist != 0)
    if n_skip_rope > 0:
        mask = mask | (col >= S - n_skip_rope) | (row >= S - n_skip_rope)
    mask = mask.unsqueeze(0).unsqueeze(0)
    if extra_mask is not None:
        mask = mask & extra_mask
    scores = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    n_keys = mask.expand(scores.shape[0], scores.shape[1], -1, -1).sum(-1)   # [B, H, S]
    w = torch.softmax(scores, dim=-1)
    return torch.where((n_keys > 0).unsqueeze(-1), w, torch.zeros_like(w)), n_keys


def _detach_any(t):
    """Detach a `_tul_core` return slot whatever its shape. Raises on anything it has no
    rule for, rather than caching a graph-carrying object and replaying it."""
    if t is None:
        return None
    if torch.is_tensor(t):
        return t.detach()
    if isinstance(t, (list, tuple)):
        return type(t)(_detach_any(u) for u in t)
    raise RuntimeError(f"_tul_core returned a {type(t).__name__} this probe has no rule "
                       "for; caching and replaying it would be a guess")


class Split(ZSplit):
    """`slot_z_optimize.ZSplit` with an ARITY-TOLERANT `_tul_core` patch.

    `_tul_core`'s return tuple grows as arms are added (6 entries, then 7 with
    `tul.mux_every_pass`), and the parent's patch unpacked exactly six — it raised on the
    longer tuple, which is how this probe first died. Everything that makes the parent's
    split trustworthy is inherited untouched: the `prefix_project` identity assertion, the
    "no core block ran during replay" check, the front cache, the group-loss spy. Only the
    cache shape changes: entry 1 is z and every other entry is stashed and replayed
    verbatim.

    DELETE THIS CLASS when `ZSplit` itself is arity-tolerant on master and use the parent
    directly. A second implementation of one rule is drift. It is here because the parent
    was 6-arity when this audit ran; another agent's UNCOMMITTED edit made the parent
    tolerant the same afternoon (its cache stores an `n_extra` count and replays the extra
    entries as None; this one replays them detached, which is the same thing on the eval
    forward both take). Whoever lands that edit owns removing this.
    """

    def __enter__(self):
        super().__enter__()
        m, st = self.model, self.st
        real_core = self._real["core"]

        def core(*a, **k):
            if st["mode"] == "replay":
                rest = st["core"]
                return (rest[0], st["z"]) + tuple(rest[2:])
            out = real_core(*a, **k)
            st["core"] = tuple(_detach_any(t) for t in out)
            return out
        m._tul_core = core
        return self


class Acc:
    """Weighted running means, keyed by a string. One place, so every table row is
    accumulated the same way over batches."""

    def __init__(self) -> None:
        self.d: dict[str, list[float]] = {}

    def add(self, key: str, value: float, weight: float = 1.0) -> None:
        s, c = self.d.get(key, [0.0, 0.0])
        self.d[key] = [s + value * weight, c + weight]

    def out(self) -> dict[str, float]:
        return {k: s / c for k, (s, c) in self.d.items() if c > 0}


# ── the instrument ────────────────────────────────────────────────────────────


class GeometryHooks:
    """Forward hooks + three method patches that read one core application.

    Everything is keyed by ``ctx["sec"]`` (``"slot"`` / ``"token"``) and ``ctx["pass"]``,
    so the SAME instrument reads the slot loop's passes and the token twin's single
    application. ``ctx["rows"]`` is the [B, S] bool row selection used by every norm.

    The three patches are needed because the quantities they read are LOCALS:

    * ``cca._gate_combine_up`` — ``out_comp`` / ``out_win`` / the gate never leave the
      attention implementation's frame. This is the only way to see that a branch is
      empty.
    * ``cca._window_attn`` — the window branch's attention weights are never
      materialised (SDPA / the fused kernel). The wrapper recomputes them explicitly in
      fp32 under the SAME mask rule for the geometry rows only.
    * ``model._apply_injection`` and ``model.injection.forward`` — the per-layer x0 term
      and the SSM ctx rewrite, and the pass counter the injection-off counterfactual
      needs. Instance attributes, so they shadow without touching the class.
    """

    def __init__(self, model, n_core: int, acc: Acc, geom: dict):
        self.m = model
        self.n_core = n_core
        self.acc = acc
        self.geom = geom                     # per-(sec, block) geometry, batch 0 only
        self.ctx = {"sec": None, "pass": 0, "rows": None, "geom": False,
                    "inj_cut_from": None}
        self.states: list[torch.Tensor] = []      # slot loop: state after each pass
        self.h0: torch.Tensor | None = None
        self._prev: torch.Tensor | None = None
        self._h = []
        self._patched = []

    # -- patches -----------------------------------------------------------
    def __enter__(self):
        m, ctx = self.m, self.ctx

        # (1) the SSM injection + the pass counter
        inj = m.injection
        real_inj = inj.forward

        def inj_fwd(h, e):
            ctx["pass"] += 1
            # The state this pass STARTS from. `model.injection` is called exactly once
            # per core application and nowhere else, so this is the only place the
            # pre-pass carrier is visible; `_exit_hook` differences against it.
            self._prev = h.detach()
            cut = ctx["inj_cut_from"]
            if cut is not None and ctx["pass"] >= cut:
                return h
            out = real_inj(h, e)
            if ctx["sec"] is not None:
                r = ctx["rows"]
                self.acc.add(f"{ctx['sec']}/inj_ssm_delta/p{ctx['pass']}",
                             _pp_norm(out - h, r))
                self.acc.add(f"{ctx['sec']}/carrier_in/p{ctx['pass']}", _pp_norm(h, r))
            return out
        inj.forward = inj_fwd
        self._patched.append((inj, "forward", real_inj))

        # (2) the per-core-layer x0/bigram term. `_apply_injection` is a staticmethod
        #     called as `self._apply_injection(h, term)`, so an instance attribute
        #     shadows it for the core, the prelude and the coda alike — hence the shape
        #     guard: only the compact/looped carrier of the current section is cut.
        real_apply = type(m)._apply_injection

        def apply_inj(h, term):
            in_core = ctx["sec"] is not None and h.shape[1] == ctx["S"]
            cut = ctx["inj_cut_from"]
            if in_core:
                self.acc.add(f"{ctx['sec']}/inj_layer_term/p{ctx['pass']}",
                             _pp_norm(term, ctx["rows"]))
                if cut is not None and ctx["pass"] >= cut:
                    return h
            return real_apply(h, term)
        m._apply_injection = apply_inj
        self._patched.append((m, "_apply_injection", None))

        # (3) per-block attention internals
        for i, blk in enumerate(m.core):
            cca = blk.attention._impl.cca
            self._patch_gate(cca, i)
            self._patch_window(cca, i)

        # -- hooks ---------------------------------------------------------
        self._hooks = [m.core_init.register_forward_hook(self._h0_hook)]
        for i, blk in enumerate(m.core):
            self._hooks.append(blk.mrr_attn.register_forward_hook(self._res_hook(i, "attn")))
            self._hooks.append(blk.mrr_mlp.register_forward_hook(self._res_hook(i, "mlp")))
            self._hooks.append(blk.attention.register_forward_hook(self._branch_hook(i, "attn")))
            self._hooks.append(blk.mlp.register_forward_hook(self._branch_hook(i, "mlp")))
            if i == self.n_core - 1:
                self._hooks.append(blk.register_forward_hook(self._exit_hook))
        return self

    def __exit__(self, *exc):
        for h in self._hooks:
            h.remove()
        for obj, name, real in reversed(self._patched):
            if real is None:
                delattr(obj, name)
            else:
                setattr(obj, name, real)
        self._patched = []
        return False

    def _patch_gate(self, cca, i: int):
        real = cca._gate_combine_up

        def gate(x, out_comp, out_win, q_lat=None, gate_pre=None):
            ctx = self.ctx
            if ctx["sec"] is not None:
                B, S, _ = x.shape
                H, D = cca.n_heads, cca.d_head
                g_lin = cca.gate(x) if gate_pre is None else cca.gate[2](cca.gate[1](gate_pre))
                g = torch.sigmoid(g_lin).reshape(B, S, H, 2).permute(0, 2, 1, 3)
                ql = cca.W_down_q(x) if q_lat is None else q_lat
                x_res = ql.reshape(B, S, H, D).transpose(1, 2)
                # back to [B, S, H*D] so `_pp_norm`'s per-position reduction applies
                def _pos(t):
                    return t.transpose(1, 2).reshape(B, S, H * D)
                r = ctx["rows"]
                p, sec = ctx["pass"], ctx["sec"]
                self.acc.add(f"{sec}/attn_comp/b{i}/p{p}", _pp_norm(_pos(g[..., 0:1] * out_comp), r))
                self.acc.add(f"{sec}/attn_win/b{i}/p{p}", _pp_norm(_pos(g[..., 1:2] * out_win), r))
                self.acc.add(f"{sec}/attn_res/b{i}/p{p}", _pp_norm(_pos(cca.alpha * x_res), r))
                self.acc.add(f"{sec}/attn_comp_raw/b{i}/p{p}", _pp_norm(_pos(out_comp), r))
                self.acc.add(f"{sec}/attn_win_raw/b{i}/p{p}", _pp_norm(_pos(out_win), r))
                self.acc.add(f"{sec}/gate_comp/b{i}/p{p}", float(g[..., 0].mean()))
                self.acc.add(f"{sec}/gate_win/b{i}/p{p}", float(g[..., 1].mean()))
                if ctx["geom"]:
                    self.geom.setdefault(f"{sec}/b{i}", {}).update(
                        comp_out_norm=float(out_comp.float().norm()),
                        comp_out_is_zero=bool(float(out_comp.float().abs().max()) == 0.0),
                        win_out_norm=float(out_win.float().norm()),
                        win_out_norm_cell0=float(out_win[:, :, 0].float().norm()),
                        gate_comp=float(g[..., 0].mean()), gate_win=float(g[..., 1].mean()))
            return real(x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)
        cca._gate_combine_up = gate
        self._patched.append((cca, "_gate_combine_up", real))

    def _patch_window(self, cca, i: int):
        real = cca._window_attn

        def win(q, k, v, device, scale, n_skip_rope=0, extra_mask=None):
            if self.ctx["geom"]:
                self._window_geometry(cca, i, q, k, scale, n_skip_rope, extra_mask)
            return real(q, k, v, device, scale, n_skip_rope, extra_mask=extra_mask)
        cca._window_attn = win
        self._patched.append((cca, "_window_attn", real))

    @torch.no_grad()
    def _window_geometry(self, cca, i, q, k, scale, n_skip_rope, extra_mask):
        """Read the window branch's attention weights and reduce them to five numbers."""
        sec = self.ctx["sec"]
        S = q.shape[2]
        w, n_keys = window_weights(q, k, cca.window_size, scale, n_skip_rope, extra_mask)
        rows = self.ctx["rows"]
        sel = (n_keys > 0)
        if rows is not None:
            sel = sel & rows.unsqueeze(1)
        selm = sel.float()
        n_sel = selm.sum().clamp_min(1.0)
        ent = -(w * (w + 1e-12).log()).sum(-1)                       # [B,H,S]
        part = 1.0 / (w.pow(2).sum(-1) + 1e-12)
        # w[b, h, i, i-1] for i >= 1: the diagonal one below the main one.
        prev = torch.zeros_like(ent)
        prev[:, :, 1:] = w.diagonal(offset=-1, dim1=2, dim2=3)
        top1 = w.max(-1).values
        self.geom.setdefault(f"{sec}/b{i}", {}).update(
            win_entropy=float((ent * selm).sum() / n_sel),
            win_participation=float((part * selm).sum() / n_sel),
            win_mass_prev=float((prev * selm).sum() / n_sel),
            win_mass_older=float(((1.0 - prev) * selm).sum() / n_sel),
            win_mass_top1=float((top1 * selm).sum() / n_sel),
            win_keys_mean=float((n_keys.float() * selm).sum() / n_sel),
            win_rows_with_no_key=int((n_keys == 0).sum()),
            win_keys_row0=int(n_keys[0, 0, 0]),
            S=int(S), window_size=int(cca.window_size),
            n_heads=int(cca.n_heads), d_head=int(cca.d_head))
        del w

    # -- hooks -------------------------------------------------------------
    def _h0_hook(self, _mod, _args, out):
        if self.ctx["sec"] == "slot":
            self.h0 = out.detach()

    def _exit_hook(self, _mod, _args, out):
        """End of one core application: the pass's state, and how far the pass moved it.

        `|h_t - h_{t-1}|` per position is the number lead 2 turns on — the ratio
        `out/in` on a pre-norm branch shrinks when the carrier grows, and this one does
        not."""
        ctx = self.ctx
        if ctx["sec"] is None:
            return
        o = out if torch.is_tensor(out) else out[0]
        if self._prev is not None:
            d = _pp_norm(o - self._prev, ctx["rows"])
            n_in = _pp_norm(self._prev, ctx["rows"])
            self.acc.add(f"{ctx['sec']}/state_delta/p{ctx['pass']}", d)
            self.acc.add(f"{ctx['sec']}/state_delta_rel/p{ctx['pass']}",
                         d / max(n_in, 1e-6))
        if ctx["sec"] == "slot":
            self.states.append(o.detach())

    def _res_hook(self, i: int, kind: str):
        """The HC residual's own carrier write: |h_out − h_in| per position, and |h_in|.
        This is the ABSOLUTE update the block makes to the carrier the coda later reads."""
        def f(_mod, args, out):
            ctx = self.ctx
            if ctx["sec"] is None:
                return
            h_in = args[0]
            r, p, sec = ctx["rows"], ctx["pass"], ctx["sec"]
            self.acc.add(f"{sec}/d_{kind}/b{i}/p{p}", _pp_norm(out - h_in, r))
            self.acc.add(f"{sec}/h_{kind}/b{i}/p{p}", _pp_norm(h_in, r))
        return f

    def _branch_hook(self, i: int, kind: str):
        """The sublayer's own input (the normed, stream-mixed x_bar) and output."""
        def f(_mod, args, out):
            ctx = self.ctx
            if ctx["sec"] is None:
                return
            o = out[0] if isinstance(out, tuple) else out
            r, p, sec = ctx["rows"], ctx["pass"], ctx["sec"]
            self.acc.add(f"{sec}/branch_in_{kind}/b{i}/p{p}", _pp_norm(args[0], r))
            self.acc.add(f"{sec}/branch_out_{kind}/b{i}/p{p}", _pp_norm(o, r))
        return f


# ── the token twin ────────────────────────────────────────────────────────────


@torch.no_grad()
def token_core_pass(model, x, x0, bigram_emb):
    """ONE application of the SAME core weights to the checkpoint's own TOKEN states.

    Mirrors `_tul_core`'s construction with the gather removed: `e = input_norm(x)` over
    ALL L positions, `h = core_init(e)`, the same per-core-layer injection stack (built
    with `input_ids=None`, exactly as `_tul_core` builds it — value embeds never fire in
    the core), then `_apply_core_step` once at `iter_idx=0`. Tokens never loop in
    training; the point is what the same weights DO at the long shape.
    """
    np_, n_core = model.cfg.n_prelude, model.cfg.n_core
    e = model.input_norm(x)
    h = model.core_init(e)
    inj = torch.stack(
        [model._build_injection_term(np_ + i, model.x0_injects[np_ + i].precompute(x0),
                                     None, bigram_emb, h.dtype)
         for i in range(n_core)], dim=0)
    out, _ = model._apply_core_step(h, e, None, None, None, ret_state=None,
                                    iter_idx=0, inj_terms=inj)
    return h, out


# ── readout statistics ────────────────────────────────────────────────────────


@torch.no_grad()
def readout_stats(model, z_entry, z_exit, valid, acc: Acc, svd: dict) -> None:
    """Lead 4: what the coda actually receives of the loop's work.

    The coda cell IS `W_prefix[k] · z` — `scatter_positions` REPLACES the base at those
    positions — so `|W_k·(exit − entry)| / |W_k·entry|` is the loop's share of the cell,
    exactly. `stream_mean_ratio` is the MUX / `unpack` reader's view: those read
    `h.mean(dim=2)`, which cancels anything the four Cayley streams disagree on.
    """
    d = (z_exit.float() - z_entry.float())
    v = valid
    n_valid = float(v.sum())

    def per_slot(t):                                    # [B, S, n, C] -> mean |·| over valid
        return float(t.flatten(2).norm(dim=2)[v].mean())

    acc.add("readout/entry_norm", per_slot(z_entry.float()), n_valid)
    acc.add("readout/exit_norm", per_slot(z_exit.float()), n_valid)
    acc.add("readout/delta_norm", per_slot(d), n_valid)
    acc.add("readout/delta_over_entry",
            float((d.flatten(2).norm(dim=2) / z_entry.float().flatten(2).norm(dim=2)
                   .clamp_min(1e-6))[v].mean()), n_valid)

    # stream-mean collapse: |mean_n u| / mean_n |u_n|
    for name, t in (("entry", z_entry.float()), ("delta", d)):
        num = t.mean(dim=2).norm(dim=2)                              # [B, S]
        den = t.norm(dim=3).mean(dim=2).clamp_min(1e-9)              # [B, S]
        acc.add(f"readout/stream_mean_ratio_{name}", float((num / den)[v].mean()), n_valid)

    W = model.tul.W_prefix
    if W is None:
        return
    K, C, _ = W.shape
    for k in range(K):
        Wk = W[k].float()
        pe = torch.matmul(z_entry.float(), Wk)
        pd = torch.matmul(d, Wk)
        acc.add(f"readout/Wp{k}_entry_norm", float(pe.flatten(2).norm(dim=2)[v].mean()), n_valid)
        acc.add(f"readout/Wp{k}_delta_norm", float(pd.flatten(2).norm(dim=2)[v].mean()), n_valid)
        acc.add(f"readout/Wp{k}_delta_share",
                float((pd.flatten(2).norm(dim=2) / pe.flatten(2).norm(dim=2).clamp_min(1e-9))[v]
                      .mean()), n_valid)
        if k not in svd:
            U, S, _V = torch.linalg.svd(Wk)
            svd[k] = (U, S)
        U, S = svd[k]
        dm = d.mean(dim=2)[v]                                        # [N, C] stream mean
        dm = dm / dm.norm(dim=1, keepdim=True).clamp_min(1e-9)
        cs = (dm @ U[:, :8]).abs().mean(0)                           # |cos| with top-8 U
        for j in range(8):
            acc.add(f"readout/Wp{k}_cos_u{j}", float(cs[j]), n_valid)
        acc.add(f"readout/Wp{k}_sigma_max", float(S[0]), 1.0)
        acc.add(f"readout/Wp{k}_sigma_min", float(S[-1]), 1.0)
        acc.add(f"readout/Wp{k}_sigma_ratio", float(S[0] / S[-1].clamp_min(1e-9)), 1.0)


# ── main ──────────────────────────────────────────────────────────────────────


def guard_model(model) -> None:
    """Refuse anything this instrument's bookkeeping cannot read honestly."""
    tc, mc = model.cfg.tul, model.cfg
    if getattr(model, "tul_reread", None) is not None:
        raise SystemExit("tul.reread adds a second `_apply_injection` call per pass; the "
                         "injection accounting below would mix the two. Not supported.")
    if getattr(model, "tul_recur_gate", None) is not None:
        raise SystemExit("tul.recur_gate blends h AFTER the last core block, so the "
                         "captured exit state is not the loop's state.")
    if model.scse is not None or bool(tc.db_loop):
        raise SystemExit("SCSE / db_loop carry a DEVIATION, not the slot state.")
    if bool(mc.slot_state_renorm) or float(mc.core_gain_clip) > 0.0:
        raise SystemExit("slot_state_renorm / core_gain_clip rescale h AFTER the last "
                         "core block; the captured exit state would not be the loop's.")
    if int(tc.prefix_k) < 1:
        raise SystemExit("prefix_k < 1: nothing is written to the coda.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_geometry_audit needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    guard_split_point(model)
    guard_model(model)

    tc = model.cfg.tul
    notes = {
        "slot_depth_fixed": [int(tc.slot_depth_fixed), a.depth],
        "slot_gain_lambda": [float(model.cfg.slot_gain_lambda), 0.0],
        "mode": "eval (dropout OFF, token-state dropout OFF)",
        "token_twin": "the SAME core weights applied ONCE to input_norm(prelude) over "
                      "ALL L positions of the SAME rows — tokens never loop in training",
        "autocast": "bf16, as the trainer",
    }
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, int(tc.slot_max_depth or model.cfg.max_depth))
    model.cfg.slot_gain_lambda = 0.0
    model.eval()
    model.requires_grad_(False)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    n_core = int(cfg.model.n_core)
    acc, geom, svd = Acc(), {}, {}
    static: dict = {}
    n_rows = 0
    repro_max = 0.0
    exit_gate_max = 0.0

    with Split(model) as split, GeometryHooks(model, n_core, acc, geom) as gh:
        for bi, (inp, labels, layout, _) in enumerate(batches):
            inp, labels = inp.cuda(), labels.cuda()
            layout = layout.to("cuda")
            valid = layout.slot_valid
            n_valid = float(valid.sum())
            n_rows += int(inp.shape[0])

            # ── the trained forward, instrumented ────────────────────────────
            gh.states.clear()
            gh.ctx.update(sec="slot", **{"pass": 0}, rows=valid,
                          S=int(layout.slot_index.shape[1]), geom=(bi == 0),
                          inj_cut_from=None)
            torch.manual_seed(a.seed + bi)
            torch.cuda.manual_seed_all(a.seed + bi)
            _out, z_loop, h0 = split.record(inp, labels, layout, want_groups=False)
            gh.ctx.update(sec=None, geom=False)
            ce_loop = float(split.token_ce)
            if split.st["core0"] != a.depth:
                raise RuntimeError(f"core[0] ran {split.st['core0']} times, "
                                   f"expected {a.depth}")
            if len(gh.states) != a.depth:
                raise RuntimeError(f"captured {len(gh.states)} pass states, "
                                   f"expected {a.depth}")
            # GATE 1: the captured last-pass state IS the tensor prefix_project receives.
            g1 = float((gh.states[-1].float() - z_loop.float())[valid].abs().max())
            exit_gate_max = max(exit_gate_max, g1)
            if g1 != 0.0:
                raise RuntimeError(
                    f"the captured pass-{a.depth} state differs from the loop's returned "
                    f"h_slots by {g1:.3e} on valid slots — something between the last core "
                    "block and the return moved it; every per-pass number below would be "
                    "measuring a different object")
            acc.add("ce_loop", ce_loop, 1.0)

            # ── K_t: the write contribution per pass, through the shipped coda ──
            # GATE 2: t = depth must reproduce the recorded forward's CE bit for bit.
            zs = [h0] + list(gh.states)
            for t, z in enumerate(zs):
                torch.manual_seed(a.seed + bi)
                zz = torch.where(valid.view(*valid.shape, *([1] * (z.dim() - 2))),
                                 z, z_loop)
                split.replay(inp, labels, layout, zz, want_groups=False)
                acc.add(f"ce_K{t}", float(split.token_ce), 1.0)
                if t == a.depth:
                    repro_max = max(repro_max, abs(float(split.token_ce) - ce_loop))

            # ── the counterfactual: the injection cut after pass 1 ───────────
            gh.states.clear()
            gh.ctx.update(sec="slot", **{"pass": 0}, rows=valid,
                          S=int(layout.slot_index.shape[1]), geom=False,
                          inj_cut_from=2)
            torch.manual_seed(a.seed + bi)
            torch.cuda.manual_seed_all(a.seed + bi)
            split.record(inp, labels, layout, want_groups=False)
            gh.ctx.update(sec=None, inj_cut_from=None)
            z_noinj = gh.states[-1]
            torch.manual_seed(a.seed + bi)
            split.replay(inp, labels, layout,
                         torch.where(valid.view(*valid.shape, *([1] * (z_noinj.dim() - 2))),
                                     z_noinj, z_loop), want_groups=False)
            acc.add("ce_K_noinj", float(split.token_ce), 1.0)
            acc.add("noinj_exit_delta_over_loop",
                    float(((z_noinj.float() - z_loop.float()).flatten(2).norm(dim=2)
                           / z_loop.float().flatten(2).norm(dim=2).clamp_min(1e-6))[valid]
                          .mean()), n_valid)

            # ── the readout ──────────────────────────────────────────────────
            readout_stats(model, h0, z_loop, valid, acc, svd)

            # ── the token twin: the same weights, S = L ──────────────────────
            x_f, x0_f, bg_f = split.st["front"][0], split.st["front"][1], split.st["front"][2]
            gh.ctx.update(sec="token", **{"pass": 0}, rows=None, S=int(x_f.shape[1]),
                          geom=(bi == 0), inj_cut_from=None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h_tok0, h_tok1 = token_core_pass(model, x_f, x0_f, bg_f)
            gh.ctx.update(sec=None, geom=False)
            acc.add("token/carrier_entry_norm", _pp_norm(h_tok0, None), 1.0)
            acc.add("token/carrier_exit_norm", _pp_norm(h_tok1, None), 1.0)
            acc.add("token/delta_over_entry",
                    float(((h_tok1.float() - h_tok0.float()).flatten(2).norm(dim=2)
                           / h_tok0.float().flatten(2).norm(dim=2).clamp_min(1e-6)).mean()), 1.0)
            del h_tok0, h_tok1

            if not static:
                static = {"S_slot": int(layout.slot_index.shape[1]),
                          "S_token": int(inp.shape[1]),
                          "n_valid_slots_per_batch": int(valid.sum()),
                          "d_model": int(cfg.model.d_model),
                          "hc_streams": int(cfg.model.hc_streams),
                          "prefix_k": int(tc.prefix_k)}
                for i, blk in enumerate(model.core):
                    impl = blk.attention._impl
                    cca = impl.cca
                    kind = "CSA" if type(impl).__name__ == "_CCACSAAttention" else "HCA"
                    m_ = int(impl.compress_ratio)
                    ent = {"kind": kind, "compress_ratio": m_,
                           "window_size": int(cca.window_size),
                           "n_heads": int(cca.n_heads), "d_head": int(cca.d_head),
                           "tg_restrict": bool(impl.tg_restrict),
                           "compressed_branch": ("tg_slot_dense_causal" if impl.tg_restrict
                                                 else "pooled_compressor"),
                           "n_blocks_slot": static["S_slot"] // m_,
                           "n_blocks_token": static["S_token"] // m_}
                    if kind == "CSA":
                        ent["top_k"] = int(impl.top_k)
                        ent["tk_slot"] = min(int(impl.top_k), ent["n_blocks_slot"])
                        ent["tk_token"] = min(int(impl.top_k), ent["n_blocks_token"])
                    static[f"core{i}"] = ent
            torch.cuda.empty_cache()
            print(f"  batch {bi + 1}/{len(batches)} done  ce_loop {ce_loop:.4f}", flush=True)

    res = acc.out()
    rec = {"label": label, "config": config, "step": step, "rows": n_rows,
           "batch": a.batch, "depth": a.depth, "n_batches": len(batches),
           "notes": notes, "static": static, "geometry": geom, "results": res,
           "gates": {"exit_state_is_z_max_abs": exit_gate_max,
                     "K_depth_reproduces_ce_max_abs": repro_max}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rec, open(a.out, "w"), indent=1)

    # ── console summary ───────────────────────────────────────────────────
    D = a.depth
    print(f"\n{label} step {step} — {n_rows} rows, batch {a.batch}, depth {D}")
    print(f"  gates: exit-state == z  |Δ| {exit_gate_max:.3e};  "
          f"K{D} reproduces ce_loop |Δ| {repro_max:.3e}")
    print(f"  S_slot {static['S_slot']}  S_token {static['S_token']}  "
          f"valid slots/batch {static['n_valid_slots_per_batch']}")
    print("\n  [1] attention geometry per core block")
    hdr = ("blk", "kind", "m", "nblk_s", "tk_s", "|comp|_s", "|win|_s",
           "g_comp", "H_win", "prev%", "keys0")
    print("      " + "".join(h.rjust(10) for h in hdr))
    for i in range(n_core):
        s = static[f"core{i}"]
        g = geom.get(f"slot/b{i}", {})
        print("      " + "".join(str(x).rjust(10) for x in (
            i, s["kind"], s["compress_ratio"], s["n_blocks_slot"],
            s.get("tk_slot", "-"),
            f"{g.get('comp_out_norm', float('nan')):.3f}",
            f"{g.get('win_out_norm', float('nan')):.3f}",
            f"{g.get('gate_comp', float('nan')):.3f}",
            f"{g.get('win_entropy', float('nan')):.3f}",
            f"{100 * g.get('win_mass_prev', float('nan')):.1f}",
            g.get("win_keys_row0", "-"))))
    print("      token twin (S = %d):" % static["S_token"])
    for i in range(n_core):
        s, g = static[f"core{i}"], geom.get(f"token/b{i}", {})
        print("      " + "".join(str(x).rjust(10) for x in (
            i, s["kind"], s["compress_ratio"], s["n_blocks_token"],
            s.get("tk_token", "-"),
            f"{g.get('comp_out_norm', float('nan')):.3f}",
            f"{g.get('win_out_norm', float('nan')):.3f}",
            f"{g.get('gate_comp', float('nan')):.3f}",
            f"{g.get('win_entropy', float('nan')):.3f}",
            f"{100 * g.get('win_mass_prev', float('nan')):.1f}",
            g.get("win_keys_row0", "-"))))

    print("\n  [2] ABSOLUTE per-position norms at pass 1, slot vs token, SAME weights")
    print("      (|h| = the 4-stream carrier entering the block; branch = the "
          "single-stream sublayer output)")
    print("      " + "".join(h.rjust(11) for h in
                             ("blk", "|h|_s", "attn_s", "mlp_s", "a/h_s",
                              "|h|_t", "attn_t", "mlp_t", "a/h_t")))
    for i in range(n_core):
        row = [i]
        for sec in ("slot", "token"):
            h_ = res.get(f"{sec}/h_attn/b{i}/p1", float("nan"))
            at = res.get(f"{sec}/branch_out_attn/b{i}/p1", float("nan"))
            ml = res.get(f"{sec}/branch_out_mlp/b{i}/p1", float("nan"))
            row += [f"{h_:.2f}", f"{at:.2f}", f"{ml:.2f}", f"{at / h_:.4f}"]
        print("      " + "".join(str(x).rjust(11) for x in row))

    print("\n  [3] injection vs the pass, per pass (slot); token twin on the last row")
    print("      " + "".join(h.rjust(14) for h in
                             ("pass", "|h_in|", "ssm_delta", "layer_term",
                              "|dh|", "|dh|/|h_in|")))
    for p in range(1, D + 1):
        print("      " + "".join(str(x).rjust(14) for x in (
            p, f"{res.get(f'slot/carrier_in/p{p}', float('nan')):.3f}",
            f"{res.get(f'slot/inj_ssm_delta/p{p}', float('nan')):.3f}",
            f"{res.get(f'slot/inj_layer_term/p{p}', float('nan')):.3f}",
            f"{res.get(f'slot/state_delta/p{p}', float('nan')):.3f}",
            f"{res.get(f'slot/state_delta_rel/p{p}', float('nan')):.4f}")))
    print("      " + "".join(str(x).rjust(14) for x in (
        "token", f"{res.get('token/carrier_in/p1', float('nan')):.3f}",
        f"{res.get('token/inj_ssm_delta/p1', float('nan')):.3f}",
        f"{res.get('token/inj_layer_term/p1', float('nan')):.3f}",
        f"{res.get('token/state_delta/p1', float('nan')):.3f}",
        f"{res.get('token/state_delta_rel/p1', float('nan')):.4f}")))

    print("\n  [4] readout")
    for k in sorted(res):
        if k.startswith("readout/"):
            print(f"      {k[8:]:28s} {res[k]:.4f}")

    print("\n  [5] K_t at the coda (token CE with z = the state after t passes)")
    base = res.get("ce_K0", float("nan"))
    for t in range(0, D + 1):
        v = res.get(f"ce_K{t}", float("nan"))
        print(f"      K{t}  {v:.4f}   ({v - base:+.4f} vs K0)")
    v = res.get("ce_K_noinj", float("nan"))
    print(f"      K{D} with the injection cut after pass 1: {v:.4f} "
          f"({v - res.get(f'ce_K{D}', float('nan')):+.4f} vs K{D}); exit state moved "
          f"{res.get('noinj_exit_delta_over_loop', float('nan')):.4f} of its norm")
    print(f"      ce_loop {res['ce_loop']:.4f}")

    print("\n  [6] carrier norm per pass (slot) vs the token carrier")
    print("      slot: " + "  ".join(
        f"p{p} {res.get(f'slot/carrier_in/p{p}', float('nan')):.2f}" for p in range(1, D + 1))
        + f"  exit {res.get('readout/exit_norm', float('nan')):.2f}")
    print(f"      token: entry {res.get('token/carrier_entry_norm', float('nan')):.2f} "
          f"exit {res.get('token/carrier_exit_norm', float('nan')):.2f} "
          f"(relative move {res.get('token/delta_over_entry', float('nan')):.4f}); "
          f"slot relative move over {D} passes "
          f"{res.get('readout/delta_over_entry', float('nan')):.4f}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
