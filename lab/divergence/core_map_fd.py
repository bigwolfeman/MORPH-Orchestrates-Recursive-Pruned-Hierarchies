"""The looped core's MAP, per pass and per position, by the slot hinge's own finite difference.

One instrument for every loop in the tree. The slot loop's gain hinge
(`MORPHTransformer._slot_gain_penalty`) reads

    g = ||f(h + d) - f(h)|| / ||d||,   d Gaussian, scaled PER POSITION to eps * ||h_i||,

at the detached operating point, same iteration index, same injection source and retention
state, same dropout masks, and aggregates it over every active position of a row. The plain
looped model has no such hinge, so its map has never been read in the same units. This
script reads both, through one code path:

* it runs the model's own eval forward at a FIXED depth (plain: `model.cfg.mean_depth`;
  slot loop: `tul.slot_mean_depth`, both deterministic at eval) and records every call the
  loop makes to `_apply_core_step` — the args and kwargs exactly as the loop passed them,
  so the map re-applied here is the map the forward ran, attention kwargs and all. The
  Jacobian probe's capture list (`model._jac_capture`) is attached at the same time: it is
  appended right before each step call and carries the active mask (`active & slot_valid`
  on the slot loop), and the two lists are checked to line up one to one;
* at every pass it applies that step at h and at h + d and reports, per position,
  `g_i = ||Δf_i|| / ||d_i||`, and per row the hinge's aggregate
  `||Δf * m||_F / ||d||_F`. Both include cross-position effects: d moves every position at
  once, so Δf_i is the response at i to the joint perturbation, exactly as in the hinge;
* it also reads an ISOTROPIC variant (d isotropic over the row's active set, one global
  scale), whose row reading estimates the Jacobian's typical gain `||J||_F / sqrt(n)` —
  the number `CoreJacobianProbe` reports as `rms` — so the two instruments can be bridged;
* and the one-step norm ratio `||f(h)_i|| / ||h_i||` per position and per row.

Modes: `bf16` is the hinge verbatim (the carrier's own dtype, autocast bf16, d cast to the
carrier dtype); `fp32` casts the operating point to fp32 and turns autocast off (the
Jacobian probe's precision). `--train-dropout` re-reads the map (in each precision named by
`--train-modes`) with the model in train mode at the SAME eval operating point, the RNG state put back before each of the two
applications (the hinge's discipline), so the two see identical dropout masks.

Sanity built in: `f(h_t)` recomputed here must equal the next pass's recorded input on the
active positions (reported as `replay_err`), or the replayed map is not the map the
forward ran and nothing below it means anything.

Usage:
  PYTHONPATH=$PWD python lab/divergence/core_map_fd.py \
    --ckpt plain=notul_panel_norm_match=/path/step_5000.pt \
    --ckpt ruler=tul_slot_spandec_strict=/path/step_5000.pt \
    --rows 96 --batch 4 --depth 6 --out /path/core_map_fd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _build import build_cfg, build_model, parse_ckpt_spec         # noqa: E402
from _rows import pack_rows, stream_from_loader                     # noqa: E402

QS = (10, 50, 90, 99)


# ── capture ────────────────────────────────────────────────────────────────────────────
class StepRecorder:
    """Records every `_apply_core_step` call of one forward, args and kwargs verbatim.

    Installed as an INSTANCE attribute, which shadows the bound method for every
    `self._apply_core_step(...)` lookup inside the loop closures. The replay in `fd_pass`
    calls `self.orig`, the original bound method, never the wrapper.
    """

    def __init__(self, root):
        self.root = root
        self.orig = type(root)._apply_core_step.__get__(root)
        self.calls: list[tuple[tuple, dict]] = []

    def __enter__(self):
        rec = self

        def wrapper(*args, **kwargs):
            rec.calls.append((args, dict(kwargs)))
            return rec.orig(*args, **kwargs)

        self.root._apply_core_step = wrapper
        self.root._jac_capture = []
        return self

    def __exit__(self, *exc):
        del self.root._apply_core_step
        self.points = self.root._jac_capture
        self.root._jac_capture = None
        return False


def _cast_tree(x, dtype):
    if torch.is_tensor(x) and x.is_floating_point():
        return x.to(dtype)
    if isinstance(x, tuple):
        return tuple(_cast_tree(v, dtype) for v in x)
    if isinstance(x, list):
        return [_cast_tree(v, dtype) for v in x]
    if isinstance(x, dict):
        return {k: _cast_tree(v, dtype) for k, v in x.items()}
    return x


# ── one pass ───────────────────────────────────────────────────────────────────────────
def fd_pass(rec: StepRecorder, k: int, h, mask, eps: float, seed: int, mode: str,
            train_dropout: bool = False) -> dict:
    """Finite-difference gain of pass k's step at its recorded operating point.

    Returns per-position tensors (flattened over the active positions, with their position
    index) and per-row aggregates. `mask` is `[B, S]` bool.
    """
    root = rec.root
    args, kwargs = rec.calls[k]
    if mode == "fp32":
        h = h.float()
        args = _cast_tree(args, torch.float32)
        kwargs = {kk: (_cast_tree(v, torch.float32) if kk in ("inj_terms", "ret_state")
                       else v) for kk, v in kwargs.items()}
    m = mask.view(*mask.shape, *([1] * (h.dim() - 2))).to(h.dtype)
    g = torch.Generator(device=h.device).manual_seed(seed)
    v = torch.randn(h.shape, generator=g, device=h.device, dtype=torch.float32).to(h.dtype) * m
    hn = h.flatten(2).float().norm(dim=2)                                         # [B, S]
    vn = v.flatten(2).float().norm(dim=2)
    scale = (eps * hn / (vn + 1e-6)).to(h.dtype)
    d = v * scale.view(*scale.shape, *([1] * (h.dim() - 2)))
    # isotropic direction over the row's active set, one scale per row
    vi = torch.randn(h.shape, generator=g, device=h.device, dtype=torch.float32).to(h.dtype) * m
    hrow = (h * m).float().flatten(1).norm(dim=1)
    virow = vi.float().flatten(1).norm(dim=1)
    di = vi * (eps * hrow / (virow + 1e-6)).to(h.dtype).view(-1, *([1] * (h.dim() - 1)))

    def step(x):
        out, _ = rec.orig(x, *args[1:], **kwargs)
        return out

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()

    def restore():
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)

    was_training = root.training
    if train_dropout:
        root.train()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=(mode == "bf16")):
            restore()
            f0 = step(h)
            restore()
            f1 = step(h + d)
            restore()
            f2 = step(h + di)
            restore()
    finally:
        root.train(was_training)

    diff = ((f1 - f0) * m).float()
    diffi = ((f2 - f0) * m).float()
    dnp = d.float().flatten(2).norm(dim=2)                                       # [B, S]
    g_pos = diff.flatten(2).norm(dim=2) / (dnp + 1e-12)
    ratio_pos = (f0 * m).float().flatten(2).norm(dim=2) / (hn + 1e-12)
    g_row = diff.flatten(1).norm(dim=1) / (d.float().flatten(1).norm(dim=1) + 1e-6)
    gi_row = diffi.flatten(1).norm(dim=1) / (di.float().flatten(1).norm(dim=1) + 1e-6)
    ratio_row = (f0 * m).float().flatten(1).norm(dim=1) / (hrow + 1e-12)
    return {"g_pos": g_pos, "ratio_pos": ratio_pos, "g_row": g_row, "gi_row": gi_row,
            "ratio_row": ratio_row, "f0": f0, "num_row": diff.flatten(1).norm(dim=1),
            "den_row": d.float().flatten(1).norm(dim=1),
            "numi_row": diffi.flatten(1).norm(dim=1),
            "deni_row": di.float().flatten(1).norm(dim=1)}


def operator_stats(rec: StepRecorder, k: int, h, mask, n_iter: int, seed: int) -> dict:
    """The operator of pass k's step, in fp32, WITHOUT double backward.

    * typical gain `||M J M||_F / sqrt(n)` — Hutchinson on the VJP: `||J||_F = ||J^T||_F`,
      so `E ||M J^T M u||^2 / ||u||^2` over isotropic u is exact and needs one backward
      per probe. This is the quantity `CoreJacobianProbe` reports as `rms`, read by an
      independent route (that probe needs a double backward, which does not exist through
      every kernel on the TG arms).
    * sigma_max over the whole batch's active set — power iteration `v <- J^T (J v)`, with
      `J^T` the exact VJP and `J v` a forward finite difference in fp32 at a small scale
      (the fp32 linearity rows show the map is linear at eps 0.005..0.08).
    * the isotropic FD reading on the SAME batch and scale convention, as the bridge.
    """
    args, kwargs = rec.calls[k]
    args = _cast_tree(args, torch.float32)
    kwargs = {kk: (_cast_tree(v, torch.float32) if kk in ("inj_terms", "ret_state") else v)
              for kk, v in kwargs.items()}
    m = mask.view(*mask.shape, *([1] * (h.dim() - 2))).float()
    x = h.detach().float().requires_grad_(True)
    g = torch.Generator(device=h.device).manual_seed(seed + 17)
    with torch.autocast("cuda", enabled=False):
        with torch.enable_grad():
            y, _ = rec.orig(x, *args[1:], **kwargs)
            y = y.float()
        n_probe, acc = 8, 0.0
        for _ in range(n_probe):
            u = torch.randn(h.shape, generator=g, device=h.device) * m
            (w,) = torch.autograd.grad(y, x, grad_outputs=u, retain_graph=True)
            acc += float(((w * m) ** 2).sum()) / float((u ** 2).sum())
        rms = (acc / n_probe) ** 0.5
        y0 = y.detach()
        x0 = x.detach()
        scale = 0.01 * float((x0 * m).norm())

        def jv(v):
            with torch.no_grad():
                f1, _ = rec.orig(x0 + scale * v, *args[1:], **kwargs)
            return (f1.float() - y0) * m / scale

        v = torch.randn(h.shape, generator=g, device=h.device) * m
        v = v / v.norm()
        fd_iso = float(jv(v).norm())
        sig, prev, rel = float("nan"), float("nan"), float("nan")
        for _ in range(n_iter):
            j = jv(v)
            prev, sig = sig, float(j.norm())
            (w,) = torch.autograd.grad(y, x, grad_outputs=j, retain_graph=True)
            w = w * m
            v = w / (w.norm() + 1e-30)
            if prev == prev:
                rel = abs(sig - prev) / max(sig, 1e-30)
    del y, x
    return {"rms_vjp": rms, "sigma": sig, "sigma_rel": rel, "fd_iso": fd_iso}


# ── statistics ─────────────────────────────────────────────────────────────────────────
def summarize(vals: np.ndarray) -> dict:
    if vals.size == 0:
        return {"n": 0}
    out = {"n": int(vals.size), "mean": float(vals.mean()),
           "max": float(vals.max()), "min": float(vals.min()),
           "frac_gt_1": float((vals > 1.0).mean()),
           "frac_gt_1.05": float((vals > 1.05).mean()),
           "frac_gt_0.95": float((vals > 0.95).mean())}
    for q, x in zip(QS, np.percentile(vals, QS)):
        out[f"p{q}"] = float(x)
    return out


def pos_bins(n_pos: int, slot: bool) -> list[tuple[str, int, int]]:
    if slot:
        q = max(1, n_pos // 4)
        return [(f"slot[{a}:{b}]", a, b) for a, b in
                [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n_pos)]]
    edges = [0, 1, 16, 128, 512, n_pos]
    return [(f"pos[{a}:{b}]", a, b) for a, b in zip(edges[:-1], edges[1:]) if a < b]


# ── one arm ────────────────────────────────────────────────────────────────────────────
def run_arm(spec: str, a) -> dict:
    from morph.training.data import create_dataloader
    from morph.training.train import load_checkpoint

    label, config, path, ovr = parse_ckpt_spec(spec)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    model, tul_rt = build_model(cfg, device="cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(path, model, scaler, torch.device("cuda"))
    step = int(torch.load(path, map_location="cpu", weights_only=False).get("step", -1))
    model.eval()
    root = getattr(model, "_orig_mod", model)
    plain = tul_rt is None
    if not plain:
        tc = root.cfg.tul
        if bool(getattr(tc, "tokens_through_core", False)) or bool(
                getattr(tc, "loop_reads_tokens", False)):
            raise SystemExit(f"{label}: not a slot loop; this script reads plain or slot "
                             "loop arms only")
        # The slot loop's step must be `_apply_core_step` and nothing else, or replaying
        # the recorded call is not replaying the map: `_core_step` adds the reread and the
        # carry BEFORE the call. Refuse rather than measure a different map.
        for attr in ("tul_reread", "tul_carry"):
            if getattr(root, attr, None) is not None:
                raise SystemExit(f"{label}: {attr} is on; `_core_step` is not "
                                 "`_apply_core_step` on this arm")
        tc.slot_mean_depth = int(a.depth)
        tc.slot_max_depth = max(int(a.depth), int(getattr(tc, "slot_max_depth", 0) or 0))
        if int(getattr(tc, "slot_depth_fixed", 0) or 0) > 0:
            tc.slot_depth_fixed = int(a.depth)
    else:
        root.cfg.mean_depth = int(a.depth)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = (int(cfg.data.seq_len) + 1 if plain
                  else tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1)
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, plain)[:n_batches]

    modes = list(a.modes)
    train_modes = [f"{m}_train" for m in a.train_modes] if a.train_dropout else []
    acc = {md: {} for md in modes + train_modes}
    peak: dict[int, list[np.ndarray]] = {}
    jac: dict[int, list[dict]] = {}
    replay_err: dict[int, list[float]] = {}
    lin: dict[str, list[float]] = {}
    rows_done = 0
    t0 = time.time()
    for bi, (inp, labels, layout, _idx) in enumerate(batches):
        inp = inp.cuda()
        lay = layout.to("cuda") if layout is not None else None
        rec = StepRecorder(root)
        with rec, torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if plain:
                model(inp, labels=None)
            else:
                model.tul_forward_ablated(inp, None, lay, plan_mode="normal")
        pts = rec.points
        if len(pts) != len(rec.calls):
            raise RuntimeError(f"{label}: {len(rec.calls)} core-step calls but "
                               f"{len(pts)} capture points; the lists do not line up")
        if len(rec.calls) != a.depth:
            raise RuntimeError(f"{label}: {len(rec.calls)} passes recorded, forced depth "
                               f"{a.depth}")
        for k, (p, (args, _)) in enumerate(zip(pts, rec.calls)):
            if int(p["iter_idx"]) != k or not torch.equal(p["h"], args[0].detach()):
                raise RuntimeError(f"{label}: pass {k}: capture point and recorded call "
                                   "disagree (iter_idx or h)")
        rows_done += int(inp.shape[0])
        carrier_dtype = str(rec.calls[0][0][0].dtype)
        do_train = a.train_dropout and bi < a.train_dropout_batches
        do_jac = bi < a.jac_batches
        for k in range(len(rec.calls)):
            h = rec.calls[k][0][0].detach()
            mask = pts[k]["active"].bool()
            seed = a.seed + 7919 * bi + 104729 * k
            # outlier structure of the carrier: per-position max|h| / rms(h)
            hf = h.float().flatten(2)
            pk = (hf.abs().amax(dim=2) / (hf.pow(2).mean(dim=2).sqrt() + 1e-12))
            peak.setdefault(k, []).append(pk.cpu()[mask.cpu()].numpy())
            if do_jac:
                jac.setdefault(k, []).append(
                    operator_stats(rec, k, h, mask, a.jac_power_iters, seed))
            for md in modes + (train_modes if do_train else []):
                r = fd_pass(rec, k, h, mask, a.eps, seed, md.split("_")[0],
                            train_dropout=md.endswith("_train"))
                slot = acc[md].setdefault(k, {"g": [], "ratio": [], "pos": [], "g_row": [],
                                              "gi_row": [], "ratio_row": [], "num2": 0.0,
                                              "den2": 0.0, "numi2": 0.0, "deni2": 0.0})
                mk = mask.cpu()
                posidx = torch.arange(mask.shape[1]).unsqueeze(0).expand_as(mk)[mk]
                slot["g"].append(r["g_pos"].cpu()[mk].numpy())
                slot["ratio"].append(r["ratio_pos"].cpu()[mk].numpy())
                slot["pos"].append(posidx.numpy())
                slot["g_row"].append(r["g_row"].cpu().numpy())
                slot["gi_row"].append(r["gi_row"].cpu().numpy())
                slot["ratio_row"].append(r["ratio_row"].cpu().numpy())
                slot["num2"] += float((r["num_row"] ** 2).sum())
                slot["den2"] += float((r["den_row"] ** 2).sum())
                slot["numi2"] += float((r["numi_row"] ** 2).sum())
                slot["deni2"] += float((r["deni_row"] ** 2).sum())
                if md == "bf16" and k + 1 < len(rec.calls):
                    # the replayed step must reproduce the next pass's input
                    nxt = rec.calls[k + 1][0][0].detach().float()
                    mm = mask.view(*mask.shape, *([1] * (h.dim() - 2))).float()
                    e = ((r["f0"].float() - nxt) * mm).norm() / ((nxt * mm).norm() + 1e-12)
                    replay_err.setdefault(k, []).append(float(e))
                del r
            if bi == 0 and a.lin_eps:
                # linearity: the fp32 row gain at smaller / larger eps, same direction seed
                for le in a.lin_eps:
                    for lm in ("fp32", "bf16"):
                        r = fd_pass(rec, k, h, mask, le, seed, lm)
                        lin.setdefault(f"{lm}_t{k}_eps{le}", []).extend(
                            r["g_row"].cpu().tolist())
                        del r
        del rec, pts
        torch.cuda.empty_cache()
        print(f"  [{label}] batch {bi + 1}/{len(batches)} rows={rows_done} "
              f"{time.time() - t0:.0f}s", flush=True)

    # ── reduce ──
    n_pos = int(max(np.concatenate(acc["bf16"][0]["pos"]).max() + 1, 1))
    bins = pos_bins(n_pos, slot=not plain)
    out_modes = {}
    for md, per in acc.items():
        rows_out = {}
        for k in sorted(per):
            s = per[k]
            gv = np.concatenate(s["g"]).astype(np.float64)
            rv = np.concatenate(s["ratio"]).astype(np.float64)
            pv = np.concatenate(s["pos"])
            grow = np.concatenate(s["g_row"])
            entry = {"g_pos": summarize(gv), "ratio_pos": summarize(rv),
                     "g_row": summarize(grow.astype(np.float64)),
                     "g_row_pooled": (s["num2"] / max(s["den2"], 1e-30)) ** 0.5,
                     "g_iso_row": summarize(np.concatenate(s["gi_row"]).astype(np.float64)),
                     "g_iso_pooled": (s["numi2"] / max(s["deni2"], 1e-30)) ** 0.5,
                     "ratio_row": summarize(np.concatenate(s["ratio_row"]).astype(np.float64)),
                     "by_position": {}}
            for name, lo, hi in bins:
                sel = (pv >= lo) & (pv < hi)
                entry["by_position"][name] = {"g_pos": summarize(gv[sel]),
                                              "ratio_pos": summarize(rv[sel])}
            rows_out[f"t{k}"] = entry
        allg = np.concatenate([np.concatenate(per[k]["g"]) for k in per]).astype(np.float64)
        allrow = np.concatenate([np.concatenate(per[k]["g_row"]) for k in per])
        rows_out["all_passes"] = {"g_pos": summarize(allg),
                                  "g_row": summarize(allrow.astype(np.float64))}
        out_modes[md] = rows_out
    if a.save_npz:
        npz = a.out.rsplit(".", 1)[0] + f".{label}.npz"
        np.savez_compressed(npz, **{f"{md}_t{k}_{f}": np.concatenate(acc[md][k][f])
                                    .astype(np.float32 if f != "pos" else np.int32)
                                    for md in acc for k in acc[md]
                                    for f in ("g", "ratio", "pos")})
    return {"label": label, "config": config, "ckpt": path, "step": step, "plain": plain,
            "rows": rows_done, "depth": a.depth, "carrier_dtype": carrier_dtype, "eps": a.eps, "n_positions": n_pos,
            "replay_err": {f"t{k}": float(np.max(v)) for k, v in replay_err.items()},
            "linearity_row_mean": {k: float(np.mean(v)) for k, v in lin.items()},
            "carrier_peak_to_rms": {f"t{k}": summarize(np.concatenate(v).astype(np.float64))
                                    for k, v in peak.items()},
            "operator": {f"t{k}": {kk: float(np.mean([x[kk] for x in v])) for kk in v[0]}
                         for k, v in jac.items()},
            "modes": out_modes}


def print_table(res: dict) -> None:
    print(f"\n=== {res['label']} (step {res['step']}, {res['rows']} rows, depth "
          f"{res['depth']}, eps {res['eps']}) replay_err={res['replay_err']}")
    for md, rows in res["modes"].items():
        print(f"  -- {md}")
        print(f"  {'pass':<5} {'mean':>6} {'p10':>6} {'p50':>6} {'p90':>6} {'p99':>6} "
              f"{'max':>6} {'>1':>7} {'>1.05':>7} | {'row':>6} {'pooled':>6} {'iso':>6} "
              f"| {'nrm_pos':>7} {'nrm_row':>7}")
        for k, e in rows.items():
            if k == "all_passes":
                continue
            gp = e["g_pos"]
            print(f"  {k:<5} {gp['mean']:6.3f} {gp['p10']:6.3f} {gp['p50']:6.3f} "
                  f"{gp['p90']:6.3f} {gp['p99']:6.3f} {gp['max']:6.3f} "
                  f"{gp['frac_gt_1']:7.4f} {gp['frac_gt_1.05']:7.4f} | "
                  f"{e['g_row']['mean']:6.3f} {e['g_row_pooled']:6.3f} "
                  f"{e['g_iso_pooled']:6.3f} | {e['ratio_pos']['mean']:7.3f} "
                  f"{e['ratio_row']['mean']:7.3f}")
        ap = rows["all_passes"]
        print(f"  all   {ap['g_pos']['mean']:6.3f} {ap['g_pos']['p10']:6.3f} "
              f"{ap['g_pos']['p50']:6.3f} {ap['g_pos']['p90']:6.3f} {ap['g_pos']['p99']:6.3f} "
              f"{ap['g_pos']['max']:6.3f} {ap['g_pos']['frac_gt_1']:7.4f} "
              f"{ap['g_pos']['frac_gt_1.05']:7.4f} | {ap['g_row']['mean']:6.3f}")
    if res["linearity_row_mean"]:
        print("  linearity (row-aggregate mean, batch 0):")
        for k, v in res["linearity_row_mean"].items():
            print(f"    {k:<18} {v:.4f}")
    print("  carrier peak/rms per position: " + " ".join(
        f"{k}:p50={v['p50']:.1f},max={v['max']:.1f}" for k, v in
        res["carrier_peak_to_rms"].items()))
    for k, v in res["operator"].items():
        print(f"  operator {k}: typical(VJP)={v['rms_vjp']:.4f} sigma_max={v['sigma']:.4f} "
              f"(rel_change {v['sigma_rel']:.1e}) fd_iso_same_batch={v['fd_iso']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-dropout", action="store_true")
    ap.add_argument("--train-dropout-batches", type=int, default=4)
    ap.add_argument("--lin-eps", default="0.005,0.08")
    ap.add_argument("--modes", default="bf16,fp32")
    ap.add_argument("--train-modes", default="fp32,bf16")
    ap.add_argument("--jac-batches", type=int, default=1)
    ap.add_argument("--jac-power-iters", type=int, default=60)
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.lin_eps = [float(x) for x in a.lin_eps.split(",") if x.strip()]
    a.modes = [m for m in a.modes.split(",") if m]
    a.train_modes = [m for m in a.train_modes.split(",") if m]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for spec in a.ckpt:
        res = run_arm(spec, a)
        print_table(res)
        results.append(res)
        torch.cuda.empty_cache()
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
