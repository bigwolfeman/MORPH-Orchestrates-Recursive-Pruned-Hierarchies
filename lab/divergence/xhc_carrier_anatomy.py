"""Plan C1b diagnostic: the slot loop's carrier, per pass and per stream, on ONE val batch.

C1 (`lxtul-e4probe-fp01-xhc`) detonated at 3190 and C2 at 2711. The per-step probe says the
hinge's own reading (`loss/gain_est`, routing replayed) crossed its 0.98 target 250-550 steps
BEFORE the carrier grew, and the growth came as a per-pass compounding (x1.3-1.5 at 3000)
followed by a single jump at pass 1 (x10-16). The probe cannot say WHICH streams grew, whether
the jump is a common (slot-independent) write, or how the realised map compares with the
replayed one. This script reads those at a checkpoint, eval mode, one forward at a forced
per-slot depth:

* every call the loop makes to `_apply_core_step` is recorded (args and kwargs verbatim,
  `core_map_fd.StepRecorder`), with the Jacobian capture's `active & slot_valid` mask;
* per pass: the RMS of every stream of the carrier entering the pass and of the step's
  output, over valid slots, split into the entry streams (0 .. n-1, n = model.hc_streams)
  and the expanded ones (n .. N-1), and into the injection's context channels and the rest;
  the per-slot norm ratio |f(h)| / |h| (the trainer's `loop/core_gain` without the LXTUL-E
  code, which the loop adds after the step); the COMMON-MODE share of each stream group
  (energy of the across-slot mean over total energy: 1 = every slot carries the same
  vector, the "massive offset" reading);
* per pass, the hinge's finite difference of the step, g = |f(h+d) - f(h)| / |d| with d
  Gaussian, scaled per slot to `--eps` of the slot's norm, reported as the row aggregate
  (the hinge's number) and per slot, in THREE readings: `replay` (the router's choice at h
  reused at h + d, what `loss/gain_est` reads), `free` (the router re-chooses at h + d:
  the map the forward applies, jumps included) and `renorm` (the replayed difference of
  R(f), R the per-slot pin to |h|: what `model.slot_gain_renorm` would read at this
  operating point). A model without xHC gets `free` and `renorm` only;
* xHC only: per pass, per residual, the router's choice (share of positions whose active
  set holds each stream; the share whose routed pair differs from pass 0), the mean gate of
  the routed pair, the read's mass on the entry streams, |y| (the sublayer output RMS), and
  the write each stream group received (|X' - X| over |X| per group).

Precision: the forward runs under bf16 autocast (the operating point training reaches); the
finite differences replay the recorded step in `--precision` (fp32 by default: the hinge's
bf16 reading is biased high on a moving map, `morph-eager-hinge-reads-noise`).

Usage (from the worktree root, under the GPU lock; ~1-2 min per checkpoint):
  PYTHONPATH=.:lab/divergence python lab/divergence/xhc_carrier_anatomy.py \\
      CONFIG CKPT [--batch 2] [--depth 6] [--eps 0.02] [--precision fp32] [--out FILE.json]

Read C1 at step_2500 (pre-onset), C2 at step_2500 (inside its growth), C1 at step_5000
(after), and the N = 4 control `tul_slot_spandec_strict_e4probe_fp01_pk4` at step_2500 (the
arm that met the same hinge by growing its pass-0 write x21 while its reading fell back).
"""
from __future__ import annotations

import argparse
import json
import math

import torch

from _capture_lab import load_model_and_batch
from core_map_fd import StepRecorder, _cast_tree


def _stream_rms(h: torch.Tensor, valid: torch.Tensor) -> list[float]:
    """RMS of each stream over the valid slots and every channel: [N]."""
    x = h[valid].float()                                                 # [n, N, C]
    return x.pow(2).mean(dim=(0, 2)).sqrt().tolist()


def _group_rms(h: torch.Tensor, valid: torch.Tensor, n_entry: int, c0: int, c1: int) -> dict:
    x = h[valid].float()                                                 # [n, N, C]
    ctx = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
    ctx[c0:c1] = True
    out = {}
    for gname, gs in (("entry", slice(0, n_entry)), ("extra", slice(n_entry, x.shape[1]))):
        xs = x[:, gs]
        if xs.shape[1] == 0:
            continue
        out[f"rms_{gname}"] = float(xs.pow(2).mean().sqrt())
        out[f"rms_{gname}_ctx"] = float(xs[..., ctx].pow(2).mean().sqrt())
        out[f"rms_{gname}_nonctx"] = float(xs[..., ~ctx].pow(2).mean().sqrt())
    return out


def _common_mode(h: torch.Tensor, valid: torch.Tensor, n_entry: int) -> dict:
    """Per stream group: energy of the across-slot mean (per row) over the total energy."""
    out = {}
    for gname, gs in (("entry", slice(0, n_entry)), ("extra", slice(n_entry, h.shape[2]))):
        xs = h[:, :, gs].float()
        if xs.shape[2] == 0:
            continue
        num, den = 0.0, 0.0
        for b in range(h.shape[0]):
            v = valid[b]
            if int(v.sum()) < 2:
                continue
            z = xs[b, v]                                                 # [n, g, C]
            num += float(z.mean(0).pow(2).sum()) * z.shape[0]
            den += float(z.pow(2).sum())
        out[f"common_{gname}"] = num / max(den, 1e-30)
    return out


def _quant(x: torch.Tensor) -> dict:
    x = x.float().flatten()
    if x.numel() == 0:
        return {}
    q = torch.quantile(x, torch.tensor([0.5, 0.9], device=x.device))
    return {"p50": float(q[0]), "p90": float(q[1]), "max": float(x.max())}


def _renorm(x: torch.Tensor, n0: torch.Tensor) -> torch.Tensor:
    # `MORPHTransformer._renorm_to`, re-stated here so the script also runs on trees
    # built before it existed.
    hn = x.flatten(2).float().norm(dim=2)
    return x * (n0 / (hn + 1e-6)).to(x.dtype).view(*n0.shape, *([1] * (x.dim() - 2)))


def fd_pass(rec: StepRecorder, k: int, mask: torch.Tensor, eps: float, seed: int,
            precision: str, xhc: bool) -> dict:
    """The recorded step k at its operating point: f(h), and the three FD readings."""
    args, kwargs = rec.calls[k]
    h = args[0]
    if precision == "fp32":
        h = h.float()
        args = _cast_tree(args, torch.float32)
        kwargs = {kk: (_cast_tree(v, torch.float32) if kk in ("inj_terms", "ret_state")
                       else v) for kk, v in kwargs.items()}
    m = mask.view(*mask.shape, *([1] * (h.dim() - 2))).to(h.dtype)
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device, dtype=torch.float32).to(h.dtype) * m
    hn = h.flatten(2).float().norm(dim=2)                                     # [B, S]
    scale = (eps * hn / (v.flatten(2).float().norm(dim=2) + 1e-6)).to(h.dtype)
    d = v * scale.view(*scale.shape, *([1] * (h.dim() - 2)))

    def step(x, route=None):
        kw = dict(kwargs)
        if xhc:
            kw["xhc_route"] = route
        out, _ = rec.orig(x, *args[1:], **kw)
        return out

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=(precision == "bf16")):
        rec_route = {"replay": False, "idx": {}} if xhc else None
        f0 = step(h, rec_route)
        f_free = step(h + d)
        f_rep = step(h + d, {"replay": True, "idx": rec_route["idx"]}) if xhc else f_free
    den_row = d.float().flatten(1).norm(dim=1) + 1e-6
    den_pos = d.float().flatten(2).norm(dim=2) + 1e-12
    out = {"f0": f0}
    pairs = [("free", f0, f_free), ("replay", f0, f_rep),
             ("renorm", _renorm(f0, hn), _renorm(f_rep, hn))]
    for name, a, b in pairs:
        if name == "replay" and not xhc:
            continue
        df = ((b - a) * m).float()
        g_row = df.flatten(1).norm(dim=1) / den_row
        g_pos = df.flatten(2).norm(dim=2) / den_pos
        out[f"gain_{name}"] = float(g_row.mean())
        out[f"gain_{name}_slot"] = _quant(g_pos[mask])
    out["norm_ratio_slot"] = _quant((f0.float().flatten(2).norm(dim=2) / (hn + 1e-12))[mask])
    return out


class XHCLog:
    """Wraps every `XHCResidual.forward` of the core; logs only while `on` is True."""

    def __init__(self, mods, n_entry: int):
        self.mods, self.n_entry, self.on, self.rows = mods, n_entry, False, []
        for j, mm in enumerate(mods):
            orig = mm.forward

            def _w(h, sublayer_fn, *a, _o=orig, _mm=mm, _j=j, **kw):
                if not self.on:
                    return _o(h, sublayer_fn, *a, **kw)
                cap = {}

                def _sf(x, *aa, **kk):
                    y = sublayer_fn(x, *aa, **kk)
                    cap["y"] = y.detach()
                    return y
                idx, gate, h_pre, _ = _mm.route(h.detach(), kw.get("fixed_route"))
                out = _o(h, _sf, *a, **kw)
                self.rows.append({"j": _j, "h": h.detach(), "out": out.detach(),
                                  "idx": idx.detach(), "gate": gate.detach(),
                                  "h_pre": h_pre.detach(), "y": cap["y"]})
                return out
            mm.forward = _w

    def summarize(self, valid: torch.Tensor, depth: int) -> list[dict]:
        n_mod = len(self.mods)
        if len(self.rows) != depth * n_mod:
            raise RuntimeError(f"{len(self.rows)} residual calls logged, expected "
                               f"{depth} x {n_mod}")
        N = self.mods[0].n
        base = {r["j"]: r["idx"] for r in self.rows[:n_mod]}
        res = []
        for t in range(depth):
            rows = self.rows[t * n_mod:(t + 1) * n_mod]
            counts = torch.zeros(N)
            moved = 0
            per_res = []
            for r in rows:
                mm = self.mods[r["j"]]
                sel = r["idx"][valid]
                counts += torch.bincount(sel.flatten().cpu(), minlength=N).float()
                a_ = sel[:, mm.m:].sort(-1).values
                b_ = base[r["j"]][valid][:, mm.m:].sort(-1).values
                moved += int((a_ != b_).any(-1).sum())
                h, o = r["h"][valid].float(), r["out"][valid].float()
                dw = (o - h)
                hg = h.flatten(1).norm(dim=1) + 1e-12
                per_res.append({
                    "residual": mm.route_key,
                    "gate_routed_mean": float(r["gate"][valid][:, mm.m:].float().mean()),
                    "read_mass_entry": float(r["h_pre"][valid][:, :self.n_entry].sum(-1)
                                             .float().mean()),
                    "y_rms": float(r["y"][valid].float().pow(2).mean().sqrt()),
                    "write_rel_entry": float((dw[:, :self.n_entry].flatten(1).norm(dim=1)
                                              / hg).mean()),
                    "write_rel_extra": float((dw[:, self.n_entry:].flatten(1).norm(dim=1)
                                              / hg).mean()),
                })
            nv = int(valid.sum())
            res.append({"stream_share": (counts / (nv * n_mod)).tolist(),
                        "routed_moved_vs_pass0": moved / (nv * n_mod),
                        "residuals": per_res})
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("ckpt")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    model, x, _y, layout, step = load_model_and_batch(a.config, a.ckpt, a.batch, "cuda")
    model.eval()
    out = analyze(model, x, layout, a.depth, a.eps, a.precision, a.seed,
                  header=f"step {step}  config {a.config}")
    out.update({"config": a.config, "ckpt": a.ckpt, "step": step})
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {a.out}")


def analyze(model, x, layout, depth: int, eps: float, precision: str, seed: int,
            header: str = "") -> dict:
    """One forward at forced depth `depth`, then every per-pass reading; prints the tables
    and returns them as a dict. Split from `main` so a CPU test can drive it on a tiny model."""
    root = getattr(model, "_orig_mod", model)
    xhc = bool(getattr(root, "_xhc_streams", 0))
    n_entry = int(root.cfg.hc_streams)
    c0, c1 = int(root.injection.start), int(root.injection.end)
    for attr in ("tul_reread", "tul_carry"):
        if getattr(root, attr, None) is not None:
            raise SystemExit(f"{attr} is on: `_core_step` is not `_apply_core_step` here")
    mods = [mm for blk in root.core for mm in (blk.mrr_attn, blk.mrr_mlp)] if xhc else []
    xlog = XHCLog(mods, n_entry) if xhc else None

    depths = torch.full(layout.slot_index.shape, depth, dtype=torch.long, device=x.device)
    rec = StepRecorder(root)
    if xlog:
        xlog.on = True
    with rec, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(x, slot_layout=layout, slot_depths=depths)
    if xlog:
        xlog.on = False
    pts = rec.points
    if len(rec.calls) != depth or len(pts) != depth:
        raise RuntimeError(f"{len(rec.calls)} step calls / {len(pts)} capture points, "
                           f"forced depth {depth}")
    k_enum = int(getattr(root, "_code_enum_k", 0) or 1)
    valid = layout.slot_valid.repeat(k_enum, 1)
    print(f"{header}  xhc={xhc}  streams "
          f"{rec.calls[0][0][0].shape[2]} (entry {n_entry})  valid slots {int(valid.sum())}  "
          f"depth {depth}  precision {precision}  ctx channels [{c0}, {c1})")
    print(f"  injection A: mean {float(root.injection.log_A.exp().mean()):.4f}")
    out = {"xhc": xhc, "depth": depth, "eps": eps, "precision": precision, "passes": []}
    hdr = ("pass | rms entry/extra in -> out | common entry/extra | |f|/|h| p50/max | "
           "gain replay / free / renorm (row) | slot max replay / free / renorm")
    print(hdr)
    for k in range(depth):
        h = rec.calls[k][0][0]
        mask = pts[k]["active"]
        r = fd_pass(rec, k, mask, eps, seed + k, precision, xhc)
        f0 = r.pop("f0")
        row = {"pass": k, "stream_rms_in": _stream_rms(h, mask),
               "stream_rms_out": _stream_rms(f0, mask),
               **{f"in_{kk}": vv for kk, vv in _group_rms(h, mask, n_entry, c0, c1).items()},
               **{f"out_{kk}": vv for kk, vv in _group_rms(f0, mask, n_entry, c0, c1).items()},
               **_common_mode(h, mask, n_entry), **r}
        out["passes"].append(row)
        gr = row.get("gain_replay", math.nan)
        print(f"  t{k} | {row['in_rms_entry']:.3g}/{row.get('in_rms_extra', math.nan):.3g} -> "
              f"{row['out_rms_entry']:.3g}/{row.get('out_rms_extra', math.nan):.3g} | "
              f"{row['common_entry']:.3f}/{row.get('common_extra', math.nan):.3f} | "
              f"{row['norm_ratio_slot']['p50']:.3f}/{row['norm_ratio_slot']['max']:.3f} | "
              f"{gr:.3f} / {row['gain_free']:.3f} / {row['gain_renorm']:.3f} | "
              f"{row.get('gain_replay_slot', {}).get('max', math.nan):.3f} / "
              f"{row['gain_free_slot']['max']:.3f} / {row['gain_renorm_slot']['max']:.3f}")
    if xlog:
        rsum = xlog.summarize(valid, depth)
        out["router"] = rsum
        print("router | routed pair != pass 0 | per residual: gate(routed) read-mass(entry) "
              "|y| write(entry)/write(extra)")
        for t, rr in enumerate(rsum):
            print(f"  t{t} | moved {rr['routed_moved_vs_pass0']:.3f} | share "
                  + " ".join(f"{s:.2f}" for s in rr["stream_share"]))
            for pr in rr["residuals"]:
                print(f"       {pr['residual']:12s} gate {pr['gate_routed_mean']:.3f} "
                      f"read {pr['read_mass_entry']:.3f} |y| {pr['y_rms']:.3g} "
                      f"write {pr['write_rel_entry']:.3f}/{pr['write_rel_extra']:.3f}")
        for mm in mods:                           # unwrap: the model is usable afterwards
            del mm.forward
    return out


if __name__ == "__main__":
    main()
