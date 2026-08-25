"""H18 — is there a POSITIONAL attention sink in the looped core?

Working document: `lab/divergence/h18-attention-sink.md`.
Pre-registration: docs/experiments/planned/2026-08-25-h18-positional-attention-sink.md

The cotangent already sits on a stable sink — the same top-3 slots at every one of the six
core blocks, top slot's share rising 0.18 -> 0.54 across the onset ladder
(`docs/experiments/failures/2026-08-24-tul-takeover-cure.md`). The LEARNABLE sink is
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
from lab.divergence.jac_ladder import build                      # noqa: E402
from morph.training.train import load_checkpoint                 # noqa: E402

__all__ = ["Recorder", "window_weights", "mass_stats"]


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


def mass_stats(a: torch.Tensor, valid_q: torch.Tensor, n_valid: int) -> dict:
    """Concentration of the attention mass RECEIVED by each key position.

    `a` is `[B, H, S, S]` probabilities, `valid_q` a `[S]` bool over query rows that
    carry a finite distribution (row 0 is all `-inf` under XSA and is dropped).

    `mass_j` is the mean over valid queries, heads and batch of `a[..., j]`, restricted
    to the `n_valid` real slots. It sums to 1. Reported:

      top1/top3   the sink's size
      pr          participation ratio `(sum m)^2 / sum m^2`: 1 = one sink, n = uniform
      argmax      WHICH key position holds it
      top3_idx    the identity of the top three, for cross-block agreement
    """
    m = a[:, :, valid_q, :n_valid].mean(dim=(0, 1, 2)).double()
    m = m / m.sum().clamp_min(1e-30)
    srt, idx = torch.sort(m, descending=True)
    return {
        "top1": float(srt[0]),
        "top3": float(srt[:3].sum()),
        "pr": float(m.sum().pow(2) / m.pow(2).sum().clamp_min(1e-300)),
        "argmax": int(idx[0]),
        "top3_idx": [int(i) for i in idx[:3]],
        "n_valid": int(n_valid),
        "mass": [float(v) for v in m],
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
                    "comp_norm": float(out_comp.float().norm()),
                    "win_norm": float(out_win.float().norm()),
                    "comp_numel": int(out_comp.numel()),
                    "g_comp": float(g[..., 0].float().mean()),
                    "g_win": float(g[..., 1].float().mean()),
                })
            return rec._orig_g(self, x, out_comp, out_win, q_lat=q_lat, gate_pre=gate_pre)

        def _csa(q, C_comp, top_idx, invalid_mask, sink, scale):
            if rec.on:
                nb = C_comp.shape[1]
                cov = int(torch.unique(top_idx).numel())
                rec.csa_calls.append({"n_blocks": int(nb), "tk": int(top_idx.shape[-1]),
                                      "distinct_idx": cov, "S": int(q.shape[2])})
            return rec._orig_csa(q, C_comp, top_idx, invalid_mask, sink, scale)

        _attn._CCABase._window_attn = _window_attn
        _attn._CCABase._gate_combine_up = _gate_combine_up
        _attn.fused_csa_attention = _csa
        return self

    def __exit__(self, *exc):
        _attn._CCABase._window_attn = self._orig_w
        _attn._CCABase._gate_combine_up = self._orig_g
        _attn.fused_csa_attention = self._orig_csa
        return False

    def reset(self):
        self.win.clear()
        self.gate.clear()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a1")
    ap.add_argument("--overrides", default="training.batch_size=6,model.use_kernels=false")
    ap.add_argument("--geometry", action="store_true", help="Phase 0 audit, no checkpoint")
    ap.add_argument("--token-path", action="store_true")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
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

    raise SystemExit("ladder mode is Phase 3; only --geometry is implemented")


if __name__ == "__main__":
    main()
