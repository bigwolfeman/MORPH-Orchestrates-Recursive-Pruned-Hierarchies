"""Fractal-basin map — Lai, Bao, Quinn, Gilpin, "Fractal basins trap latent reasoning"
(arXiv 2609.04963), Appendix B (docs/references/looping-depth/2026-09-13-lit-mining/
A_theory.md, paper 2): fix a model, a prompt and a loop schedule; vary ONLY the initial
latent state along a random 2D slice of the full latent space; decode at every loop
iteration; record the number of iterations until the decoded output stops changing (the
"settling time"). Their concrete arm for us: run this on the MORPH slot loop, perturbing a
slot's entry state, decoding the coda's next-span prediction at each iteration.

Per (ROW, SLOT) pair: build a ``--grid`` x ``--grid`` grid over two random orthonormal
directions in the slot's own ENTRY-state space (``core_init(e)``'s output at that slot),
scaled by the entry's own norm times ``--radius``. For every grid point, force the loop to
depth ``d = 1 .. T`` (``_tul_core(..., slot_depths=...)``, the SAME per-forced-depth
recompute ``slot_state_probe.py``/``slot_rank_anatomy.py`` use — no per-pass trajectory
exists on an eval forward) and DECODE:

* slot-loop model — the span decoder's first-token argmax from the slot's exit state at
  depth d (``tul_spandec.decode`` with ``J=1``, which reads ``z`` alone: position 0 takes
  no token input, so this is well-defined with no leakage);
* plain model (no spandec, no slots) — there is no span to decode, so this instrument
  perturbs a TOKEN position's entry instead and reads the ordinary LM head's argmax at
  that position. This is NOT the paper's setup verbatim; it is the closest analogue this
  model family has, and is reported as such, never conflated with the slot-loop reading.

Settling time = the first depth after which the argmax token never changes again through
depth T (1 if it is already fixed at depth 1). Reports the settling-time field's Shannon
entropy and the fraction of grid points whose settling time differs from the CENTRE
point's (perturbation = 0). A flat field (entropy near 0, near-0 differing fraction) is
the paper's own "pre-bifurcation, purely contractive" signature — an informative negative,
not a null result.

Usage:
  python lab/divergence/basin_map.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --pairs 6 --grid 41 --radius 1.0 --device cuda --out results/basin_strict.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diag_common import Arm  # noqa: E402


def _orthonormal_pair(shape: tuple[int, ...], device, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two random orthonormal directions of ``shape``, via Gram-Schmidt in flattened space."""
    n = 1
    for s in shape:
        n *= s
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d1 = torch.randn(n, generator=gen).to(device=device, dtype=torch.float32)
    d1 = d1 / d1.norm()
    gen2 = torch.Generator(device="cpu").manual_seed(seed + 1)
    d2 = torch.randn(n, generator=gen2).to(device=device, dtype=torch.float32)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / d2.norm()
    return d1.view(*shape), d2.view(*shape)


def _settling_time(argmax_by_depth: list[int]) -> int:
    """First depth (1-indexed) after which the argmax never changes again."""
    T = len(argmax_by_depth)
    last = argmax_by_depth[-1]
    for d in range(T - 1, -1, -1):
        if argmax_by_depth[d] != last:
            return d + 2 if d + 1 < T else T
    return 1


def _entropy(counts: dict[int, int], n: int) -> float:
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return h


@torch.no_grad()
def _decode_argmax_slot(model, h_slot_col: torch.Tensor) -> torch.Tensor:
    """``h_slot_col`` ``[G, n, C]`` or ``[G, C]`` (one slot's exit state, G grid points).
    Returns ``[G]`` int64 argmax token ids via the span decoder's J=1 (z-only) decode."""
    dec = model.tul_spandec
    z = model._readout(h_slot_col.unsqueeze(1))              # [G, 1, C]
    G = z.shape[0]
    ids = torch.zeros(G, 1, 1, dtype=torch.long, device=z.device)
    valid = torch.zeros(G, 1, 1, dtype=torch.bool, device=z.device)
    mem = None
    if dec.reads_cells:
        raise RuntimeError("basin_map does not support tul.spandec_reads_cells")
    w_tied = model.embed.lm_weight().detach()
    st = dec.decode(z, ids, valid, w_tied, mem=mem)           # [G, 1, 1, C]
    logits = model.embed.attend(st[:, 0, 0])                  # [G, V]
    return logits.argmax(dim=-1)


@torch.no_grad()
def _decode_argmax_plain(model, h_pos: torch.Tensor) -> torch.Tensor:
    """``h_pos`` ``[G, n, C]`` or ``[G, C]`` (one token position's exit state).
    Returns ``[G]`` int64 argmax via the ordinary tied LM head."""
    z = model._readout(h_pos.unsqueeze(1))                    # [G, 1, C]
    logits = model.embed.attend(z[:, 0])                      # [G, V]
    return logits.argmax(dim=-1)


def run_one_pair(arm: Arm, inp, labels, layout, target_idx: int, grid: int, radius: float,
                 depths: list[int], seed: int, chunk: int, device: str) -> dict:
    model = arm.model
    B0, L = inp.shape
    assert B0 == 1

    if arm.is_slot_loop:
        fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout)
        x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
        real_init = model.core_init.forward

        captured = {}

        def spy(e):
            out = real_init(e)
            captured["e0"] = out.detach().clone()
            return out

        model.core_init.forward = spy
        try:
            model._tul_core(x, x0, bigram, layout, input_ids=inp,
                            slot_depths=torch.ones_like(layout.slot_index))
        finally:
            model.core_init.forward = real_init
        e0_full = captured["e0"]                              # [1, S, n, C] (or [1,S,C])
        e0_slot = e0_full[0, target_idx]                       # [n, C] or [C]
    else:
        x, x0, bigram = model._front_region(inp)
        real_init = model.core_init.forward
        captured = {}

        def spy(e):
            out = real_init(e)
            captured["e0"] = out.detach().clone()
            return out

        model.core_init.forward = spy
        try:
            model._core_region(x, x0, bigram)
        finally:
            model.core_init.forward = real_init
        e0_full = captured["e0"]                              # [1, L, n, C]
        e0_slot = e0_full[0, target_idx]

    shape = tuple(e0_slot.shape)
    d1, d2 = _orthonormal_pair(shape, device, seed)
    scale = float(e0_slot.norm())
    axis = torch.linspace(-radius, radius, grid, device=device)
    grid_ab = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1).reshape(-1, 2)  # [G,2]
    G_total = grid_ab.shape[0]
    center_idx = (G_total - 1) // 2  # linspace with odd `grid` puts (0,0) at the exact centre

    argmax_by_depth: dict[int, list[int]] = {d: [] for d in depths}
    for start in range(0, G_total, chunk):
        ab = grid_ab[start:start + chunk]                      # [g,2]
        g = ab.shape[0]
        pert = (ab[:, 0:1] * scale).view(g, *([1] * len(shape))) * d1 \
             + (ab[:, 1:2] * scale).view(g, *([1] * len(shape))) * d2
        pert_batch = e0_slot.unsqueeze(0) + pert               # [g, *shape]

        if arm.is_slot_loop:
            inp_g = inp.repeat(g, 1)
            layout_g = layout.__class__(
                slot_mask=layout.slot_mask.repeat(g, 1),
                bag_id=layout.bag_id.repeat(g, 1),
                slot_index=layout.slot_index.repeat(g, 1),
                slot_valid=layout.slot_valid.repeat(g, 1),
                prefix_k=layout.prefix_k)
            fkw_g, freset_g, _ckw_g, _creset_g = model._tul_tg_kwargs(layout_g)
            x_g, x0_g, bigram_g = model._tul_front(inp_g, layout_g, attn_kwargs=fkw_g,
                                                   ret_reset_mask=freset_g)

            def hook(e, _pert=pert_batch):
                out = real_init(e).clone()
                out[:, target_idx] = _pert.to(out.dtype)
                return out

            for d in depths:
                model.core_init.forward = hook
                try:
                    table = torch.full_like(layout_g.slot_index, d)
                    out = model._tul_core(x_g, x0_g, bigram_g, layout_g, input_ids=inp_g,
                                          slot_depths=table)
                finally:
                    model.core_init.forward = real_init
                h_slots = out[1]
                am = _decode_argmax_slot(model, h_slots[:, target_idx])
                argmax_by_depth[d].extend(am.tolist())
        else:
            def _rep(t):
                return None if t is None else t.repeat(g, *([1] * (t.dim() - 1)))
            x_g, x0_g, bigram_g = _rep(x), _rep(x0), _rep(bigram)

            def hook(e, _pert=pert_batch):
                out = real_init(e).clone()
                out[:, target_idx] = _pert.to(out.dtype)
                return out

            orig_mean = int(model.cfg.mean_depth)
            for d in depths:
                model.cfg.mean_depth = d
                model.core_init.forward = hook
                try:
                    h = model._core_region(x_g, x0_g, bigram_g)
                finally:
                    model.core_init.forward = real_init
                    model.cfg.mean_depth = orig_mean
                am = _decode_argmax_plain(model, h[:, target_idx])
                argmax_by_depth[d].extend(am.tolist())

    settling = []
    for i in range(G_total):
        seq = [argmax_by_depth[d][i] for d in depths]
        settling.append(_settling_time(seq))

    center_settling = settling[center_idx]
    counts: dict[int, int] = {}
    for s in settling:
        counts[s] = counts.get(s, 0) + 1
    frac_diff = sum(1 for s in settling if s != center_settling) / G_total
    ent = _entropy(counts, G_total)

    return {
        "target_idx": int(target_idx), "grid": grid, "radius": radius,
        "depths": depths, "n_points": G_total, "scale": scale,
        "settling_time_counts": {str(k): v for k, v in sorted(counts.items())},
        "settling_time_entropy_nats": ent,
        "center_settling_time": center_settling,
        "fraction_differing_from_centre": frac_diff,
    }


def run(label: str, config: str, path: str, device: str, pairs: int, grid: int,
       radius: float, chunk: int) -> dict:
    t0 = time.time()
    arm = Arm(label, config, path, device, rows=pairs, row_batch=1)
    model = arm.model
    T = arm.own_eval_depth()
    depths = list(range(1, max(T, 2) + 1))

    pair_results = []
    for i, (inp, labels, layout) in enumerate(arm.rows()):
        if i >= pairs:
            break
        if arm.is_slot_loop:
            if layout.slot_valid[0].sum() < 2:
                continue
            valid_idx = layout.slot_valid[0].nonzero().flatten()
            target_idx = int(valid_idx[min(1, valid_idx.numel() - 1)])
        else:
            L = inp.shape[1]
            target_idx = min(8, L - 1)
        res = run_one_pair(arm, inp, labels, layout, target_idx, grid, radius, depths,
                           seed=1000 + i, chunk=chunk, device=device)
        res["row"] = i
        pair_results.append(res)
        print(f"{label} pair {i} (row={i}, slot/pos={target_idx}): "
              f"entropy={res['settling_time_entropy_nats']:.4f} "
              f"frac_diff_from_centre={res['fraction_differing_from_centre']:.4f} "
              f"counts={res['settling_time_counts']}", flush=True)

    ents = [r["settling_time_entropy_nats"] for r in pair_results]
    fracs = [r["fraction_differing_from_centre"] for r in pair_results]
    out = {
        "label": label, "config": config, "ckpt": path, "step": arm.step,
        "is_slot_loop": arm.is_slot_loop, "n_pairs": len(pair_results),
        "grid": grid, "radius": radius, "trained_depth": T, "wall_s": time.time() - t0,
        "pairs": pair_results,
        "entropy_mean": sum(ents) / len(ents) if ents else float("nan"),
        "fraction_differing_mean": sum(fracs) / len(fracs) if fracs else float("nan"),
    }
    print(f"{label}: mean entropy={out['entropy_mean']:.4f} "
          f"mean frac differing from centre={out['fraction_differing_mean']:.4f} "
          f"over {len(pair_results)} (row,slot) pairs", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--pairs", type=int, default=6, help="(row, slot) pairs, one per row")
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=64, help="grid points per forward")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        out[label] = run(label, config, path, a.device, a.pairs, a.grid, a.radius, a.chunk)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        for label, e in out.items():
            f.write(f"{label}: config={e['config']} step={e['step']} slot_loop={e['is_slot_loop']} "
                    f"n_pairs={e['n_pairs']} grid={e['grid']} radius={e['radius']}\n")
            f.write(f"  mean entropy={e['entropy_mean']:.4f} "
                    f"mean frac differing from centre={e['fraction_differing_mean']:.4f}\n")
            for r in e["pairs"]:
                f.write(f"  row={r['row']} target_idx={r['target_idx']} "
                        f"entropy={r['settling_time_entropy_nats']:.4f} "
                        f"frac_diff={r['fraction_differing_from_centre']:.4f} "
                        f"counts={r['settling_time_counts']}\n")
    print("wrote", a.out, "and", txt, flush=True)


if __name__ == "__main__":
    main()
