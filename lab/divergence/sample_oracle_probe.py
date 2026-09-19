"""Sample-oracle gate: does sampling around the slot loop find alternatives the point loses?

THE QUESTION. The latent-exploration literature (survey under
`docs/references/looping-depth/latent-exploration/`) says a loop earns depth by holding
several candidate continuations and narrowing them, and that one deterministic trajectory
collapses to one candidate (PLR Theorem 4.4, `D(T) = L^(2T) D(0)`). Parallel Test-Time
Scaling (You et al., ACL 2026) measures this with no training: draw N latent trajectories
by perturbing the state, score the best of N after seeing the answer (coverage@N, the
oracle-over-samples) and compare it to the one deterministic trajectory. This probe asks
that of the strict slot loop as it is, before any K-stream arm is built. Prereg:
`lab/experiments/planned/2026-09-18-sample-oracle-gate.md`.

TWO NOISE SITES, deliberately both.

* ENTRY (`--variant entry`): `model.core_init.forward` is wrapped (the same hook
  `basin_map.py` uses) so the loop's entry state `e0` at every VALID slot becomes
  `e0 + sigma * rms_slot(e0) * eps`. The noise then runs through the loop at the forced
  depth. This is the GRAM shape: sample the initial latent, let the recurrence act.
* EXIT (`--variant exit`): `model.tul.prefix_project` is wrapped so the loop's exit
  `h_slots` becomes `h_slots + sigma * rms_slot(h_slots) * eps` just before the write.
  The noise never touches the loop. This is the control that separates "the loop turns
  variation into alternatives" from "the reader benefits from any spread of exits".

Per sample the WHOLE forward is `core_depth_sweep.ce_maps` → `model.tul_forward_ablated`,
so every TG mask is the model's own (a bare `_tul_front` scores a strict arm from an
unrestricted prelude; measured 2026-09-13). The exit wrapper also SPIES in the entry runs
(no noise added) to read `cos_exit`, the mean pairwise cosine of the 16 exits per slot —
the collapse measure the theorem predicts.

AGGREGATION. The tokens of span j (j >= 1) are attributed to slot j−1, the slot that
precedes them; span 0 has no slot and is excluded. Per (row, span) the CE is summed per
sample; `oracle(N)` is the minimum over the first N samples; `gain(N) = det − oracle(N)`,
token-weighted; the paired bootstrap resamples ROWS.

Usage:
  PYTHONPATH=. python lab/divergence/sample_oracle_probe.py \
    --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
    --rows 96 --batch 4 --samples 16 --sigmas 0.3,1.0 --entry-depths 1,6 --exit-depth 6 \
    --out lab/experiments/results/2026-09-18-sample-oracle-gate/sample_oracle_strict_5000.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

__all__ = [
    "relative_noise",
    "span_sums",
    "oracle_curve",
    "pairwise_cos",
    "row_units",
]


# ── pure logic ─────────────────────────────────────────────────────────────────────

def relative_noise(x, sigma: float, valid, eps):
    """``x + sigma * rms_slot(x) * eps`` at valid slots, unchanged elsewhere.

    ``x`` is ``[B, S, ..., C]`` (a plain or an HC carrier); the RMS is per slot over every
    trailing dim, so ``sigma`` is a fraction of each slot's own scale. ``valid`` is
    ``[B, S]`` bool, ``eps`` has ``x``'s shape. Torch tensors in, torch tensor out.
    """
    import torch

    if sigma == 0.0:
        return x
    trailing = tuple(range(2, x.dim()))
    rms = x.float().pow(2).mean(dim=trailing, keepdim=True).sqrt()
    mask = valid.view(*valid.shape, *([1] * (x.dim() - 2))).to(x.dtype)
    noise = (sigma * rms * eps.to(x.device, torch.float32)).to(x.dtype) * mask
    return x + noise


def span_sums(ce_row: np.ndarray, span_idx: np.ndarray, istok: np.ndarray,
              n_spans: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-span ``(sum, count)`` of a row's per-token CE over scored token positions.

    Span 0 is kept in the arrays (index 0) so callers can drop it explicitly; only spans
    with index >= 1 have a preceding slot.
    """
    keep = np.asarray(istok, dtype=bool) & (np.asarray(span_idx) >= 0)
    sp = np.asarray(span_idx)[keep]
    s = np.bincount(sp, weights=np.asarray(ce_row, dtype=np.float64)[keep], minlength=n_spans)
    n = np.bincount(sp, minlength=n_spans).astype(np.float64)
    return s[:n_spans], n[:n_spans]


def oracle_curve(samp: np.ndarray, det: np.ndarray, count: np.ndarray,
                 ns: list[int]) -> dict[str, float]:
    """Token-weighted ``det``, mean-over-samples and ``oracle(N)`` over spans.

    ``samp`` is ``[N_samples, n_units]`` of per-unit CE SUMS, ``det`` ``[n_units]``,
    ``count`` ``[n_units]`` token counts. ``oracle(N)`` takes the per-unit minimum over the
    FIRST N samples. Units with count 0 contribute nothing.
    """
    samp = np.asarray(samp, dtype=np.float64)
    det = np.asarray(det, dtype=np.float64)
    count = np.asarray(count, dtype=np.float64)
    tot = count.sum()
    if tot <= 0:
        raise ValueError("no scored tokens")
    out = {"det": float(det.sum() / tot),
           "mean_samples": float(samp.sum(axis=0).mean(axis=0) / tot) if samp.shape[0] else float("nan")}
    out["mean_samples"] = float(samp.mean(axis=0).sum() / tot)
    for n in ns:
        if n > samp.shape[0]:
            continue
        o = samp[:n].min(axis=0)
        out[f"oracle_{n}"] = float(o.sum() / tot)
        out[f"gain_{n}"] = out["det"] - out[f"oracle_{n}"]
    return out


def pairwise_cos(states: np.ndarray, valid: np.ndarray) -> float:
    """Mean pairwise cosine across samples, averaged over valid slots.

    ``states`` is ``[N, B, S, D]`` (trailing dims already flattened), ``valid`` ``[B, S]``.
    """
    st = np.asarray(states, dtype=np.float64)
    N = st.shape[0]
    if N < 2:
        raise ValueError("pairwise cosine needs >= 2 samples")
    norm = np.linalg.norm(st, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    u = st / norm
    g = np.einsum("nbsd,mbsd->bsnm", u, u)                    # [B, S, N, N]
    off = (g.sum(axis=(-1, -2)) - np.trace(g, axis1=-2, axis2=-1)) / (N * (N - 1))
    v = np.asarray(valid, dtype=bool)
    return float(off[v].mean()) if v.any() else float("nan")


def row_units(unit_vals: np.ndarray, unit_row: np.ndarray, n_rows: int) -> np.ndarray:
    """Sum per-unit values into per-row sums (the bootstrap's unit is the ROW)."""
    return np.bincount(np.asarray(unit_row, dtype=np.int64),
                       weights=np.asarray(unit_vals, dtype=np.float64),
                       minlength=n_rows)[:n_rows]


# ── the run ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--sigmas", default="0.3,1.0")
    ap.add_argument("--entry-depths", default="1,6")
    ap.add_argument("--exit-depth", type=int, default=6)
    ap.add_argument("--variants", default="entry,exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch

    from _build import ROOT, build_cfg, uses_sample_depth
    from _rows import pack_rows, stream_from_loader
    from _stats import paired_bootstrap_ci
    from core_depth_sweep import ce_maps, warn_if_frozen_reader
    from hop_distance_probe import _depth_forcer, _row_spans

    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    device = a.device
    N = int(a.samples)
    sigmas = [float(s) for s in a.sigmas.split(",")]
    entry_depths = sorted({int(x) for x in a.entry_depths.split(",")})
    exit_depth = int(a.exit_depth)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    ns = [n for n in (1, 2, 4, 8, 16, 32) if n <= N]
    if min(entry_depths + [exit_depth]) < 1:
        raise SystemExit("depth 0 is not valid on a slot arm (slot_mean_depth=0 reads as the mean)")

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit("this probe needs a slot-loop TUL arm")
    if uses_sample_depth(tul_rt.model_cfg):
        raise SystemExit("this arm runs tokens through the core; its loop is not the slot chain")
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    warn_if_frozen_reader(cfg, label)
    tc = model.cfg.tul
    set_depth, restore = _depth_forcer(model, tc)

    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    stream = stream_from_loader(loader, a.rows * (spec.l_total + 1))
    n_batches = -(-a.rows // a.batch)
    packed = pack_rows(stream, tul_rt, cfg, a.batch, plain=False)[:n_batches]

    # per-row span maps, once
    meta = []
    row0 = 0
    for inp, labels, layout, idx in packed:
        B, L = inp.shape
        spans = np.stack([_row_spans(layout, b, spec.max_slots) for b in range(B)])
        istok = (~layout.slot_mask).numpy() & (labels.numpy() >= 0)
        n_sp = int(spans.max()) + 1
        meta.append({"spans": spans, "istok": istok, "row0": row0, "n_spans": n_sp,
                     "valid": layout.slot_valid.numpy().copy()})
        row0 += B
    n_rows = row0
    packed = [(inp, labels, layout.to(device), idx) for inp, labels, layout, idx in packed]

    arm = {"label": label, "config": config, "ckpt": path, "step": step, "rows": n_rows,
           "batch": a.batch, "samples": N, "sigmas": sigmas, "entry_depths": entry_depths,
           "exit_depth": exit_depth, "variants": variants, "seed": a.seed,
           "tg_geometry": str(getattr(tc, "tg_geometry", "none")),
           "tg_coda_prefix_reach": str(getattr(tc, "tg_coda_prefix_reach", "all")),
           "loop_reach": int(getattr(tc, "loop_reach", 0))}
    print(f"[arm] {label} step={step} rows={n_rows} geometry={arm['tg_geometry']} "
          f"coda_reach={arm['tg_coda_prefix_reach']} samples={N}", flush=True)

    # ── hooks ──────────────────────────────────────────────────────────────────────
    real_init = model.core_init.forward
    real_proj = model.tul.prefix_project
    state = {"sigma_entry": 0.0, "sigma_exit": 0.0, "valid": None, "gen": None,
             "spy": None}

    def eps_like(x):
        return torch.randn(x.shape, generator=state["gen"], dtype=torch.float32)

    def init_hook(e):
        out = real_init(e)
        if state["sigma_entry"] > 0.0:
            out = relative_noise(out, state["sigma_entry"], state["valid"], eps_like(out))
        return out

    def proj_hook(h_slots, layout, l_total, cells=None):
        if state["spy"] is not None:
            state["spy"].append(h_slots.detach().float().flatten(2).cpu().numpy())
        if state["sigma_exit"] > 0.0:
            h_slots = relative_noise(h_slots, state["sigma_exit"], state["valid"],
                                     eps_like(h_slots))
        return real_proj(h_slots, layout, l_total, cells=cells)

    model.core_init.forward = init_hook
    model.tul.prefix_project = proj_hook

    def forward_ce(inp, labels, layout, d):
        set_depth(d)
        ce, _ = ce_maps(model, inp, layout, labels, device, want_mux=False)
        return ce.float().cpu().numpy()

    def gen_for(variant: str, sigma: float, d: int, s: int, bi: int):
        key = (hash((variant, sigma, d, s, bi, a.seed)) & 0x7FFFFFFF)
        return torch.Generator(device="cpu").manual_seed(key)

    # ── per-(row, span) bookkeeping ────────────────────────────────────────────────
    unit_row: list[int] = []
    unit_cnt: list[float] = []
    for m in meta:
        B = m["spans"].shape[0]
        for b in range(B):
            _, n = span_sums(np.zeros(m["spans"].shape[1]), m["spans"][b], m["istok"][b],
                             m["n_spans"])
            for j in range(1, m["n_spans"]):          # span 0 has no preceding slot
                unit_row.append(m["row0"] + b)
                unit_cnt.append(n[j])
    unit_row_a = np.asarray(unit_row, dtype=np.int64)
    unit_cnt_a = np.asarray(unit_cnt, dtype=np.float64)

    def collect(ce_batches: list[np.ndarray]) -> np.ndarray:
        """Per-unit CE sums for one forward over all batches, in unit order."""
        vals = []
        for ce, m in zip(ce_batches, meta):
            B = ce.shape[0]
            for b in range(B):
                s, _ = span_sums(ce[b], m["spans"][b], m["istok"][b], m["n_spans"])
                vals.extend(s[1:].tolist())
        return np.asarray(vals, dtype=np.float64)

    results: dict = {label: arm}
    npz: dict[str, np.ndarray] = {"unit_row": unit_row_a, "unit_cnt": unit_cnt_a}
    try:
        det: dict[int, np.ndarray] = {}
        for d in sorted(set(entry_depths + [exit_depth])):
            state.update(sigma_entry=0.0, sigma_exit=0.0, spy=None)
            det[d] = collect([forward_ce(inp, labels, layout, d)
                              for inp, labels, layout, _ in packed])
            npz[f"det_d{d}"] = det[d]
            print(f"  det depth={d}  ce={det[d].sum() / unit_cnt_a.sum():.4f}", flush=True)

        arm["cells"] = {}
        plan = []
        if "entry" in variants:
            plan += [("entry", sg, d) for sg in sigmas for d in entry_depths]
        if "exit" in variants:
            plan += [("exit", sg, exit_depth) for sg in sigmas]
        for variant, sigma, d in plan:
            samp = np.zeros((N, unit_row_a.shape[0]), dtype=np.float64)
            cos_vals: list[float] = []
            for s in range(N):
                ce_batches = []
                spies: list[list[np.ndarray]] = []
                for bi, (inp, labels, layout, _) in enumerate(packed):
                    state["valid"] = torch.from_numpy(meta[bi]["valid"]).to(device)
                    state["gen"] = gen_for(variant, sigma, d, s, bi)
                    state["sigma_entry"] = sigma if variant == "entry" else 0.0
                    state["sigma_exit"] = sigma if variant == "exit" else 0.0
                    state["spy"] = [] if variant == "entry" else None
                    ce_batches.append(forward_ce(inp, labels, layout, d))
                    spies.append(state["spy"] or [])
                samp[s] = collect(ce_batches)
                if variant == "entry":
                    # one spy entry per batch: the exit states of this sample
                    exits_by_batch = [sp[0] for sp in spies if sp]
                    if s == 0:
                        exit_store = [[] for _ in exits_by_batch]
                    for bi, ex in enumerate(exits_by_batch):
                        exit_store[bi].append(ex)
                print(f"  {variant} sigma={sigma} depth={d} sample={s + 1}/{N}  "
                      f"ce={samp[s].sum() / unit_cnt_a.sum():.4f}", flush=True)
            curve = oracle_curve(samp, det[d], unit_cnt_a, ns)
            key = f"{variant}_s{sigma}_d{d}"
            cell = dict(curve)
            for n in ns:
                if f"oracle_{n}" in curve:
                    o = samp[:n].min(axis=0)
                    cell[f"gain_{n}_ci"] = paired_bootstrap_ci(
                        row_units(det[d], unit_row_a, n_rows),
                        row_units(o, unit_row_a, n_rows),
                        row_units(unit_cnt_a, unit_row_a, n_rows))
            if variant == "entry":
                cos_vals = [pairwise_cos(np.stack(exit_store[bi]), meta[bi]["valid"])
                            for bi in range(len(exit_store))]
                cell["cos_exit"] = float(np.mean(cos_vals))
            arm["cells"][key] = cell
            npz[f"samp_{key}"] = samp.astype(np.float32)
            print(f"[cell] {key}  det={cell['det']:.4f}  mean={cell['mean_samples']:.4f}  "
                  + "  ".join(f"g{n}={cell[f'gain_{n}']:+.4f}" for n in ns if f"gain_{n}" in cell)
                  + (f"  cos_exit={cell['cos_exit']:.4f}" if "cos_exit" in cell else ""),
                  flush=True)
    finally:
        model.core_init.forward = real_init
        model.tul.prefix_project = real_proj
        restore()

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez_compressed(a.out.rsplit(".", 1)[0] + ".units.npz", **npz)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    _print_table(arm, ns)
    print(f"wrote {a.out}", flush=True)


def _print_table(arm: dict, ns: list[int]) -> None:
    print()
    print(f"== {arm['label']} step {arm['step']} rows {arm['rows']} samples {arm['samples']} ==")
    head = f"{'cell':<20}{'det':>8}{'mean':>8}" + "".join(f"{'g' + str(n):>9}" for n in ns) \
        + f"{'g16 CI':>22}{'cos_exit':>10}"
    print(head)
    for key, c in arm["cells"].items():
        ci = c.get("gain_16_ci") or c.get(f"gain_{ns[-1]}_ci")
        cis = f"[{ci['lo']:+.4f},{ci['hi']:+.4f}]" if ci else ""
        print(f"{key:<20}{c['det']:>8.4f}{c['mean_samples']:>8.4f}"
              + "".join(f"{c.get(f'gain_{n}', float('nan')):>+9.4f}" for n in ns)
              + f"{cis:>22}" + (f"{c['cos_exit']:>10.4f}" if "cos_exit" in c else ""))


if __name__ == "__main__":
    main()
