"""H18 — is there a POSITIONAL attention sink in the looped core?

Working document: `lab/divergence/h18-attention-sink.md`.
Pre-registration: lab/experiments/planned/2026-08-25-h18-positional-attention-sink.md

The cotangent already sits on a stable sink — the same top-3 slots at every one of the six
core blocks, top slot's share rising 0.18 -> 0.54 across the onset ladder
(`lab/experiments/failures/2026-08-24-tul-takeover-cure.md`). The LEARNABLE sink is
refuted (H17: `sink_logits` never leave 0.005). The FORWARD attention has never been
measured. This probe measures it.

WHY A REIMPLEMENTATION AND NOT A HOOK. MORPH never materializes an attention weight
matrix: `fused_window_attention` and `_window_fallback` (which is `F.scaled_dot_product_
attention`) both go straight to the output, and the CSA/HCA compressed paths are flash
online-softmax by design. So the probe recomputes `A = softmax(q k^T * scale + bias)` from
the same `q, k` the model used, and SELF-TESTS that `A @ v` reproduces the shipped
`out_win`. A probe measuring a tensor the model never uses is worthless, so the self-test
raises rather than warns.

The call ORDER carries the loop iteration: the core runs blocks 0..n_core-1 once per
iteration, so recorded call `c` is `(t = c // n_core, block = c % n_core)`. The recorder
asserts the observed tag sequence really is `0,1,..,n-1,0,1,..` rather than trusting it.

Usage:
    PYTHONPATH=$PWD python lab/divergence/attn_sink_probe.py --geometry
    PYTHONPATH=$PWD python lab/divergence/attn_sink_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out attn_sink.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import morph.model.attention as _attn                            # noqa: E402
from lab.divergence.drift_probe import dropout_off                # noqa: E402
from lab.divergence.jac_ladder import build                      # noqa: E402
from morph.training.train import load_checkpoint                 # noqa: E402

__all__ = ["Recorder", "window_weights", "csa_weights", "mass_stats", "ladder"]


# ── the explicit window softmax ────────────────────────────────────────────────────
def window_weights(q: torch.Tensor, k: torch.Tensor, window_size: int,
                   scale: float, n_skip_rope: int) -> torch.Tensor:
    """`[B, H, S, S]` attention probabilities for MORPH's window branch.

    Transcribed from `_window_fallback`. XSA excludes the self token (`dist != 0`), so
    query 0 has NO valid key and its row is fully masked; `softmax` of an all `-inf` row
    is NaN, which is exactly what `F.scaled_dot_product_attention` produces there, so the
    self-test still passes and the caller drops that row explicitly rather than silently.
    """
    S = q.shape[2]
    dev = q.device
    row = torch.arange(S, device=dev).unsqueeze(1)
    col = torch.arange(S, device=dev).unsqueeze(0)
    dist = row - col
    mask = (dist >= 0) & (dist < window_size) & (dist != 0)
    if n_skip_rope > 0:
        mask = mask | (col >= S - n_skip_rope) | (row >= S - n_skip_rope)
    bias = torch.where(mask, 0.0, float("-inf"))
    logits = torch.einsum("bhid,bhjd->bhij", q.float(), k.float()) * scale + bias
    return torch.softmax(logits, dim=-1)


def csa_weights(q: torch.Tensor, C_comp: torch.Tensor, top_idx: torch.Tensor,
                invalid_mask: torch.Tensor, sink_logits: torch.Tensor,
                scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """`(attn_w, C_sel)` for the CSA compressed branch — `[B, H, S, tk]` and `[B, S, tk, D]`.

    Transcribed from `csa_attention_reference`, which IS the shipped path when
    `model.use_kernels=false` (how the onset ladder was produced). The self-test against
    `out_comp` is therefore exact by construction and is a WIRING check, not a numerical
    one: it proves the probe saw the same `q, C_comp, top_idx` the model used, nothing more.
    The window branch's self-test is the one that carries numerical weight.
    """
    B, H, S, D = q.shape
    bi = torch.arange(B, device=q.device)[:, None, None]
    C_sel = C_comp[bi, top_idx]
    sc = torch.einsum("bhsd,bstd->bhst", q, C_sel) * scale
    sc = sc.masked_fill(invalid_mask.unsqueeze(1), float("-inf"))
    sink = sink_logits.view(1, H, 1, 1).expand(B, -1, S, 1)
    w = F.softmax(torch.cat([sc, sink], dim=-1).float(), dim=-1)[..., :-1]
    return w, C_sel


def mass_stats(a: torch.Tensor, key_pos: torch.Tensor, q_valid: torch.Tensor,
               key_valid: torch.Tensor) -> dict:
    """Concentration of the attention mass RECEIVED by each key position.

    `a`         `[B, H, S, K]` probabilities over K key COLUMNS
    `key_pos`   `[B, S, K]` int — the key POSITION each column refers to. For the window
                branch a column IS the position; for CSA the columns are `top_idx`, a
                per-query permutation of the block indices, so the scatter is required.
    `q_valid`   `[B, S]` bool — which query rows carry a finite distribution
    `key_valid` `[B, P]` bool — which key positions are real (not padding)

    PER ROW, because the rows of a batch have different valid slot counts, then averaged
    over rows. Mass on invalid key positions is dropped and the distribution renormalised.

    Reported:
      top1/top3   the sink's size, on the row-mean distribution
      pr          participation ratio `(sum m)^2 / sum m^2`: 1 = one sink, P = uniform
      argmax      WHICH key position holds it
      row_agree   fraction of ROWS whose own argmax equals the batch argmax. The rows of a
                  batch carry different TEXT at the same POSITIONS, so this separates a
                  positional sink (agree -> 1) from a content-driven one (agree -> 1/P).
    """
    B, H, S, K = a.shape
    P = key_valid.shape[1]
    per_row = []
    for b in range(B):
        qs = q_valid[b]
        nq = int(qs.sum())
        if nq < 1:
            continue
        w = a[b][:, qs, :].mean(dim=0).double()            # [nq, K]
        kp = key_pos[b][qs].reshape(-1)                    # [nq*K]
        pos = torch.zeros(P, dtype=torch.float64, device=a.device)
        pos.index_add_(0, kp, w.reshape(-1))
        pos[~key_valid[b]] = 0.0
        tot = pos.sum()
        if float(tot) <= 0:
            continue
        per_row.append(pos / tot)
    if not per_row:
        raise RuntimeError("no usable rows — every row had zero valid queries")
    R = torch.stack(per_row)                               # [rows, P]
    m = R.mean(0)
    srt, idx = torch.sort(m, descending=True)
    amax = int(idx[0])
    return {
        "top1": float(srt[0]),
        "top3": float(srt[:3].sum()),
        "pr": float(m.sum().pow(2) / m.pow(2).sum().clamp_min(1e-300)),
        "argmax": amax,
        "top3_idx": [int(i) for i in idx[:3]],
        "row_agree": float((R.argmax(dim=1) == amax).double().mean()),
        "rows": int(R.shape[0]),
        "n_key_valid": float(key_valid.double().sum(1).mean()),
        "mass": [float(t) for t in m],
    }


# ── recorder ───────────────────────────────────────────────────────────────────────
class Recorder:
    """Patches `_CCABase._window_attn` and `_gate_combine_up` for the CORE blocks only.

    Prelude and coda blocks share the same classes, so the patched methods run for them
    too; they are ignored by the `_probe_tag` the installer writes onto the core `cca`
    modules alone.
    """

    def __init__(self, model, n_core: int, self_test_tol: float = 2e-2):
        self.root = getattr(model, "_orig_mod", model)
        self.n_core = n_core
        self.tol = self_test_tol
        self.win: list[dict] = []
        self.gate: list[dict] = []
        self.max_gate_err = 0.0
        self._orig_w = _attn._CCABase._window_attn
        self._orig_g = _attn._CCABase._gate_combine_up
        self._orig_csa = _attn.fused_csa_attention
        self.csa_calls: list[dict] = []
        self.csa: list[dict] = []
        self.cur_tag: int | None = None
        self._orig_fwd = {}
        self.on = False
        for li, blk in enumerate(self.root.core):
            blk.attention._impl.cca._probe_tag = li

    # ---- context manager ----
    def __enter__(self):
        rec = self

        def _window_attn(self, q, k, v, device, scale, n_skip_rope=0):
            out = rec._orig_w(self, q, k, v, device, scale, n_skip_rope)
            tag = getattr(self, "_probe_tag", None)
            if rec.on and tag is not None:
                a = window_weights(q, k, self.window_size, scale, n_skip_rope)
                # SELF-TEST: the recomputed weights must reproduce the shipped output.
                ref = torch.einsum("bhij,bhjd->bhid", a, v.float()).to(out.dtype)
                fin = torch.isfinite(out) & torch.isfinite(ref)
                den = ref[fin].abs().max().clamp_min(1e-6)
                err = float((out[fin] - ref[fin]).abs().max() / den)
                if err > rec.tol:
                    raise RuntimeError(
                        f"window self-test failed at core block {tag}: rel err {err:.3e} "
                        f"> {rec.tol:.1e}. The probe is not measuring the shipped path.")
                rec.win.append({"block": tag, "a": a.detach(), "err": err,
                                "nonfinite_out": int((~torch.isfinite(out)).sum())})
            return out

        def _gate_combine_up(self, x, out_comp, out_win, q_lat=None, gate_pre=None):
            tag = getattr(self, "_probe_tag", None)
            if rec.on and tag is not None:
                g_lin = self.gate(x) if gate_pre is None else self.gate[2](self.gate[1](gate_pre))
                B, S, _ = x.shape
                g = torch.sigmoid(g_lin).reshape(B, S, self.n_heads, 2).permute(0, 2, 1, 3)
                rec.gate.append({
                    "block": tag, "S": int(S),
                    "x_pos_norm": x.float().norm(dim=-1).mean(0).detach(),
                    "comp_norm": float(out_comp.float().norm()),
                    "win_norm": float(out_win.float().norm()),
                    "comp_numel": int(out_comp.numel()),
                    "g_comp": float(g[..., 0].float().mean()),
                    "g_win": float(g[..., 1].float().mean()),
                })
            return rec._orig_g(self, x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)

        def _csa(q, C_comp, top_idx, invalid_mask, sink, scale):
            out = rec._orig_csa(q, C_comp, top_idx, invalid_mask, sink, scale)
            if rec.on:
                nb = C_comp.shape[1]
                cov = int(torch.unique(top_idx).numel())
                rec.csa_calls.append({"n_blocks": int(nb), "tk": int(top_idx.shape[-1]),
                                      "distinct_idx": cov, "S": int(q.shape[2]),
                                      "tag": rec.cur_tag})
                if rec.cur_tag is not None:
                    w, C_sel = csa_weights(q, C_comp, top_idx, invalid_mask, sink, scale)
                    ref = torch.einsum("bhst,bstd->bhsd", w.to(q.dtype), C_sel)
                    den = ref.float().abs().max().clamp_min(1e-6)
                    err = float((out.float() - ref.float()).abs().max() / den)
                    if err > rec.tol:
                        raise RuntimeError(
                            f"CSA wiring check failed at core block {rec.cur_tag}: "
                            f"rel err {err:.3e} > {rec.tol:.1e}")
                    rec.csa.append({"block": rec.cur_tag, "w": w.detach(),
                                    "top_idx": top_idx.detach(), "n_blocks": int(nb),
                                    "m": int(q.shape[2] // nb) if nb else 0, "err": err})
            return out

        def _mk_fwd(cls):
            orig = cls.forward

            def fwd(self, x, n_skip_rope=0, cla_capture=None, cla_kv=None):
                prev, rec.cur_tag = rec.cur_tag, getattr(self.cca, "_probe_tag", None)
                try:
                    return orig(self, x, n_skip_rope, cla_capture=cla_capture,
                                cla_kv=cla_kv)
                finally:
                    rec.cur_tag = prev
            return orig, fwd

        for cls in (_attn._CCACSAAttention, _attn._CCAHCAAttention):
            orig, fwd = _mk_fwd(cls)
            self._orig_fwd[cls] = orig
            cls.forward = fwd

        _attn._CCABase._window_attn = _window_attn
        _attn._CCABase._gate_combine_up = _gate_combine_up
        _attn.fused_csa_attention = _csa
        return self

    def __exit__(self, *exc):
        _attn._CCABase._window_attn = self._orig_w
        _attn._CCABase._gate_combine_up = self._orig_g
        _attn.fused_csa_attention = self._orig_csa
        for cls, orig in self._orig_fwd.items():
            cls.forward = orig
        return False

    def reset(self):
        self.win.clear()
        self.gate.clear()
        self.csa.clear()
        self.csa_calls.clear()

    def iterations(self) -> int:
        """Loop-iteration count implied by the call log, with the ordering asserted."""
        tags = [r["block"] for r in self.win]
        if not tags:
            raise RuntimeError("no core window calls recorded — is the tag installed?")
        if len(tags) % self.n_core:
            raise RuntimeError(f"{len(tags)} core calls is not a multiple of n_core="
                               f"{self.n_core}; the (t, block) mapping would be wrong")
        want = list(range(self.n_core)) * (len(tags) // self.n_core)
        if tags != want:
            raise RuntimeError(f"core block call order is {tags[:12]}..., expected "
                               f"{want[:12]}...; the (t, block) mapping would be wrong")
        return len(tags) // self.n_core


# ── geometry audit (Phase 0) ───────────────────────────────────────────────────────
def geometry(model, x, y, layout, n_core: int, tag: str) -> dict:
    rec = Recorder(model, n_core)
    with rec:
        rec.on = True
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
        rec.on = False
        n_it = rec.iterations()
        out = {"path": tag, "iters": n_it, "blocks": []}
        for b in range(n_core):
            gs = [g for g in rec.gate if g["block"] == b]
            ws = [w for w in rec.win if w["block"] == b]
            impl = rec.root.core[b].attention._impl
            kind = type(impl).__name__
            m = impl.compress_ratio
            S = gs[0]["S"]
            out["blocks"].append({
                "block": b, "kind": kind, "m": m, "S": S, "n_blocks": S // m,
                "comp_numel": gs[0]["comp_numel"],
                "comp_norm": gs[0]["comp_norm"], "win_norm": gs[0]["win_norm"],
                "g_comp": gs[0]["g_comp"], "g_win": gs[0]["g_win"],
                "window_size": impl.cca.window_size,
                "self_test_relerr": max(w["err"] for w in ws),
                "nonfinite_out": max(w["nonfinite_out"] for w in ws),
            })
        # Deduped: prelude/coda CSA blocks run at the FULL sequence length and are
        # recorded too (they share the class). Both shapes are reported.
        seen, uniq = set(), []
        for c in rec.csa_calls:
            k = (c["S"], c["n_blocks"], c["tk"], c["distinct_idx"])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        out["csa_calls"] = uniq
    return out


def print_geometry(g: dict) -> None:
    print(f"\n=== GEOMETRY — {g['path']} path, {g['iters']} loop iterations ===")
    print(f"{'blk':>3} {'kind':<20} {'S':>5} {'m':>5} {'nblk':>5} {'win':>5} "
          f"{'compN':>8} {'|comp|':>10} {'|win|':>10} {'g_comp':>7} {'g_win':>7} "
          f"{'selftest':>9} {'nonfin':>7}")
    for b in g["blocks"]:
        print(f"{b['block']:>3} {b['kind']:<20} {b['S']:>5} {b['m']:>5} {b['n_blocks']:>5} "
              f"{b['window_size']:>5} {b['comp_numel']:>8} {b['comp_norm']:>10.4f} "
              f"{b['win_norm']:>10.4f} {b['g_comp']:>7.4f} {b['g_win']:>7.4f} "
              f"{b['self_test_relerr']:>9.2e} {b['nonfinite_out']:>7}")
    if g["csa_calls"]:
        print("CSA selection (first iteration):")
        for c in g["csa_calls"]:
            print(f"    S={c['S']} n_blocks={c['n_blocks']} tk={c['tk']} "
                  f"distinct_top_idx={c['distinct_idx']}")



# ── ladder measurement (Phase 3) ───────────────────────────────────────────────────
def _more_batches(cfg, n: int) -> list[tuple]:
    """`n` further validation batches, from DIFFERENT text at the same positions.

    The sink test is positional, so the batches must differ in content and not in shape.
    `skip_samples` is moved well past the one `jac_ladder.build` uses (50 000) so the
    rows share no document with it.
    """
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    tul_rt = build_tul_runtime(cfg)
    out = []
    for i in range(n):
        it = iter(create_dataloader(
            cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
            int(cfg.training.batch_size), split="validation",
            skip_samples=200_000 + 50_000 * i,
            tul=tul_rt.val_data_cfg if tul_rt else None))
        batch = next(it)
        if len(batch) == 3:
            bx, by, blay = batch
            blay = blay.to("cuda")
        else:
            (bx, by), blay = batch, None
        out.append((bx.cuda(), by.cuda(), blay))
    return out



def collect(rec: Recorder, valid: torch.Tensor, n_core: int) -> list[dict]:
    """One recorded forward -> one row per (loop iteration, core block, branch)."""
    n_it = rec.iterations()
    B, S = valid.shape
    dev = valid.device
    ar = torch.arange(S, device=dev)

    # Valid slots MUST be a prefix of the row: `mass_stats` compares key POSITION across
    # rows with different valid counts, which only means the same thing under packing.
    nv = valid.sum(1)
    if not torch.equal(valid, ar.view(1, -1) < nv.view(-1, 1)):
        raise RuntimeError("slot_valid is not a per-row prefix; the cross-row position "
                           "comparison in mass_stats would be meaningless")

    rows = []
    qv_win = valid.clone()
    qv_win[:, 0] = False            # XSA: query 0 has no key, its softmax row is NaN
    kp_win = ar.view(1, 1, S).expand(B, S, S)
    for c, w in enumerate(rec.win):
        t, b = divmod(c, n_core)
        a = w["a"]
        if not torch.isfinite(a[:, :, qv_win[0], :]).all():
            raise RuntimeError(f"non-finite window weights at t={t} block={b} on rows "
                               "the probe treats as valid")
        st = mass_stats(a, kp_win, qv_win, valid)
        st.update({"t": t, "block": b, "branch": "win", "self_test": w["err"]})
        rows.append(st)

    seen: dict[int, int] = {}
    for r in rec.csa:
        b = r["block"]
        t = seen.get(b, 0)
        seen[b] = t + 1
        wt, m, nb = r["w"], r["m"], r["n_blocks"]
        blk_valid = (torch.arange(nb, device=dev).view(1, -1) * m) < nv.view(-1, 1)
        qv = (wt.sum(-1).mean(1) > 0) & valid          # queries with any block in view
        st = mass_stats(wt, r["top_idx"], qv, blk_valid)
        st.update({"t": t, "block": b, "branch": "csa", "self_test": r["err"],
                   "n_blocks": nb})
        rows.append(st)

    xn = {}
    for c, g in enumerate(rec.gate):
        t, b = divmod(c, n_core)
        xn[(t, b)] = g["x_pos_norm"]
    for row in rows:
        v = xn.get((row["t"], row["block"]))
        if v is not None:
            vv = v[:S].double()
            row["x_norm_pr"] = float(vv.sum().pow(2) / vv.pow(2).sum().clamp_min(1e-300))
            row["x_norm_max"] = float(vv.max() / vv.sum().clamp_min(1e-30))
    return rows


def ladder(model, x, y, layout, n_core: int, ckpts: list[str], seed: int,
           n_batches: int, batches: list[tuple]) -> list[dict]:
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    rec = Recorder(model, n_core)
    out = []
    with rec, dropout_off(model):
        for path in ckpts:
            step = int(torch.load(path, map_location="cpu",
                                  weights_only=False).get("step", -1))
            load_checkpoint(path, model, scaler, torch.device("cuda"))
            per_batch = []
            for bi in range(n_batches):
                bx, by, blay = batches[bi]
                rec.reset()
                rec.on = True
                # SAME seed every rung and every batch: the Poisson depth draw must not
                # move, or a rung-to-rung change could be a depth change.
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(bx, labels=by, slot_layout=blay)
                rec.on = False
                per_batch.append(collect(rec, blay.slot_valid, n_core))
                rec.reset()
            out.append({"ckpt": os.path.basename(path), "step": step,
                        "batches": per_batch})
            print_rung(out[-1], n_core)
    return out


def print_rung(r: dict, n_core: int) -> None:
    b0 = r["batches"][0]
    n_it = max(x["t"] for x in b0 if x["branch"] == "win") + 1
    print(f"\n{r['ckpt']} (step {r['step']})", flush=True)
    for branch in ("win", "csa"):
        rows = [x for x in b0 if x["branch"] == branch]
        if not rows:
            continue
        print(f"  [{branch}] top1 by (block, iter):", flush=True)
        for b in range(n_core):
            rb = sorted([x for x in rows if x["block"] == b], key=lambda z: z["t"])
            if not rb:
                continue
            t1 = " ".join(f"{z['top1']:6.3f}" for z in rb)
            pr = " ".join(f"{z['pr']:6.2f}" for z in rb)
            am = " ".join(f"{z['argmax']:>3d}" for z in rb)
            ag = " ".join(f"{z['row_agree']:6.2f}" for z in rb)
            print(f"    blk{b} top1 {t1}", flush=True)
            print(f"         pr   {pr}", flush=True)
            print(f"         amax {am}   agree {ag}", flush=True)
    # cross-batch stability of the sink INDEX, the positional test
    if len(r["batches"]) > 1:
        b1 = r["batches"][1]
        key = lambda z: (z["branch"], z["block"], z["t"])
        m1 = {key(z): z["argmax"] for z in b1}
        same = [key(z) in m1 and m1[key(z)] == z["argmax"] for z in b0]
        print(f"  cross-batch argmax agreement: {sum(same)}/{len(same)} "
              f"= {sum(same)/max(1,len(same)):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a2")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--geometry", action="store_true", help="Phase 0 audit, no checkpoint")
    ap.add_argument("--smoke", action="store_true",
                    help="run the full collect path on the RANDOM-INIT model. Shakes out "
                         "crashes without showing any ladder number, so it can be run "
                         "before the predictions are committed.")
    ap.add_argument("--token-path", action="store_true")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--n-batches", type=int, default=2,
                    help="extra batches test whether the sink INDEX survives new text")
    a = ap.parse_args()

    ov = [s for s in a.overrides.split(",") if s]
    cfg, model, x, y, layout = build(a.config_name, ov)
    n_core = int(cfg.model.n_core)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    if a.geometry:
        res = [geometry(model, x, y, layout, n_core, "slot")]
        print_geometry(res[0])
        if a.token_path:
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            res.append(geometry(model, x, y, None, n_core, "token"))
            print_geometry(res[1])
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
            print(f"\nwrote {a.out}")
        return

    if a.smoke:
        model.train()
        rec = Recorder(model, n_core)
        with rec, dropout_off(model):
            rec.on = True
            torch.manual_seed(a.seed)
            torch.cuda.manual_seed_all(a.seed)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(x, labels=y, slot_layout=layout)
            rec.on = False
            rows = collect(rec, layout.slot_valid, n_core)
        nw = len([r for r in rows if r["branch"] == "win"])
        nc = len([r for r in rows if r["branch"] == "csa"])
        print(f"smoke OK: {nw} window rows, {nc} csa rows, "
              f"{max(r['t'] for r in rows) + 1} iterations")
        print("keys:", sorted(k for k in rows[0] if k != "mass"))
        print("max window self-test rel err:",
              f"{max(r['self_test'] for r in rows if r['branch'] == 'win'):.2e}")
        print("max csa    self-test rel err:",
              f"{max(r['self_test'] for r in rows if r['branch'] == 'csa'):.2e}")
        return

    if not (a.ckpt_dir and a.out):
        ap.error("--ckpt-dir and --out are required unless --geometry")

    model.train()                      # Poisson depths, as in training
    batches = [(x, y, layout)]
    if a.n_batches > 1:
        for extra in _more_batches(cfg, a.n_batches - 1):
            batches.append(extra)
    ckpts = sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")),
                   key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1)))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {a.ckpt_dir}")
    res = ladder(model, x, y, layout, n_core, ckpts, a.seed, len(batches), batches)
    json.dump(res, open(a.out, "w"))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
