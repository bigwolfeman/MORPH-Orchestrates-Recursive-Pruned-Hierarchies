"""Asymptotic Alignment (AA) score — Anil, Pokle, Liang, Treutlein, Wu, Bai, Kolter,
Grosse, "Path Independent Equilibrium Models Can Better Exploit Test-Time Computation"
(arXiv 2211.09961), Algorithm 1.

    docs/references/looping-depth/2026-09-13-lit-mining/A_theory.md, paper 4.

Run the loop from its real (trained) entry to the model's own eval depth: this is the
"once-converged" state z_once. Then re-initialise the SAME context's loop from a
DIFFERENT starting point and run the SAME number of passes: this is "twice-converged"
z_twice. AA = cosine(z_twice, z_once). AA near 1 means the loop reaches the same place
regardless of where it started (path independent, consistent with a fixed point that adds
no information beyond what the entry already carried); AA well below 1 means the extra
iterations are doing real, entry-dependent work.

Two twice-converged starting points, both derived from tensors already in play so no
extra forward is needed to build them:

* **swap** — another ROW's own once-converged state (batch-rolled by one), the paper's
  "swap with the other example's converged state";
* **noise** — ``N(0, e.std()^2)`` matched to the loop entry's OWN empirical scale, the
  paper's Gaussian-noise re-init.

BOTH the slot loop (per SLOT, over ``layout.slot_valid``) and the plain looped core (per
TOKEN POSITION — the plain model's entry is Parcae's ``_NoiseInit``, already an
independent draw per position, so "per token position" is the natural granularity here)
are run through the SAME mechanism: hook ``model.core_init.forward`` to return the
substitute start point instead of calling the real module, run the SAME forward path a
second time on the SAME context, and undo the hook. No forced-depth override is used —
every run is the checkpoint's own eval-time depth, exactly what the trainer logs.

Usage:
  python lab/divergence/aa_score.py \
      --ckpt strict=tul_slot_spandec_strict=/path/step_5000.pt \
      --rows 96 --batch 8 --device cuda --out results/aa_strict.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _diag_common import Arm  # noqa: E402


def _cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def _readout_vec(model, h: torch.Tensor) -> torch.Tensor:
    """``model._readout`` on a carrier stage: ``[B, n_pos, (n,) C] -> [B, n_pos, C]``."""
    return model._readout(h)


@torch.no_grad()
def _run_once_and_twice_slot(model, inp, layout, fkw, freset):
    """One batch: ``(h_once, h_swap, h_noise, valid[B,S])`` for the slot loop."""
    x, x0, bigram = model._tul_front(inp, layout, attn_kwargs=fkw, ret_reset_mask=freset)
    out = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    h_once = out[1]

    real_init = model.core_init.forward

    def swap_hook(e):
        return torch.roll(h_once, shifts=1, dims=0).to(e.dtype)

    def noise_hook(e):
        return torch.randn_like(e) * float(e.float().std())

    model.core_init.forward = swap_hook
    try:
        out_s = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    finally:
        model.core_init.forward = real_init
    h_swap = out_s[1]

    model.core_init.forward = noise_hook
    try:
        out_n = model._tul_core(x, x0, bigram, layout, input_ids=inp)
    finally:
        model.core_init.forward = real_init
    h_noise = out_n[1]

    return h_once, h_swap, h_noise, layout.slot_valid


@torch.no_grad()
def _run_once_and_twice_plain(model, inp):
    """One batch: ``(h_once, h_swap, h_noise, valid[B,L])`` for the plain core."""
    x, x0, bigram = model._front_region(inp)
    h_once = model._core_region(x, x0, bigram)

    real_init = model.core_init.forward

    def swap_hook(e):
        return torch.roll(h_once, shifts=1, dims=0).to(e.dtype)

    def noise_hook(e):
        return torch.randn_like(e) * float(e.float().std())

    model.core_init.forward = swap_hook
    try:
        h_swap = model._core_region(x, x0, bigram)
    finally:
        model.core_init.forward = real_init

    model.core_init.forward = noise_hook
    try:
        h_noise = model._core_region(x, x0, bigram)
    finally:
        model.core_init.forward = real_init

    B, L = inp.shape
    valid = torch.ones(B, L, dtype=torch.bool, device=inp.device)
    return h_once, h_swap, h_noise, valid


def summarize(cos: list[float]) -> dict:
    if not cos:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "frac_above_0.9": float("nan")}
    t = torch.tensor(cos, dtype=torch.float64)
    return {"n": int(t.numel()), "mean": float(t.mean()), "median": float(t.median()),
            "frac_above_0.9": float((t > 0.9).float().mean())}


def run(label: str, config: str, path: str, device: str, rows: int, batch: int) -> dict:
    t0 = time.time()
    arm = Arm(label, config, path, device, rows, row_batch=batch)
    model = arm.model
    cos_swap: list[float] = []
    cos_noise: list[float] = []
    n_rows = 0
    for i, (inp, labels, layout, _idx) in enumerate(arm.batches):
        inp = inp.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            if arm.is_slot_loop:
                layout_dev = layout.to(device)
                fkw, freset, _ckw, _creset = model._tul_tg_kwargs(layout_dev)
                h_once, h_swap, h_noise, valid = _run_once_and_twice_slot(
                    model, inp, layout_dev, fkw, freset)
            else:
                h_once, h_swap, h_noise, valid = _run_once_and_twice_plain(model, inp)
            ro_once = _readout_vec(model, h_once)
            ro_swap = _readout_vec(model, h_swap)
            ro_noise = _readout_vec(model, h_noise)
            c_swap = _cosine(ro_once, ro_swap)[valid].float().cpu()
            c_noise = _cosine(ro_once, ro_noise)[valid].float().cpu()
        cos_swap.extend(c_swap.tolist())
        cos_noise.extend(c_noise.tolist())
        n_rows += inp.shape[0]
        if i % 5 == 0:
            print(f"{label} batch {i}/{len(arm.batches)} rows={n_rows} "
                  f"swap_mean={sum(c_swap.tolist())/max(1,len(c_swap)):.4f} "
                  f"noise_mean={sum(c_noise.tolist())/max(1,len(c_noise)):.4f}", flush=True)

    out = {
        "label": label, "config": config, "ckpt": path, "step": arm.step,
        "is_slot_loop": arm.is_slot_loop, "rows": n_rows, "wall_s": time.time() - t0,
        "aa_swap": summarize(cos_swap), "aa_noise": summarize(cos_noise),
    }
    print(f"{label}: AA(swap) mean={out['aa_swap']['mean']:.4f} median={out['aa_swap']['median']:.4f} "
          f"frac>0.9={out['aa_swap']['frac_above_0.9']:.3f} | "
          f"AA(noise) mean={out['aa_noise']['mean']:.4f} median={out['aa_noise']['median']:.4f} "
          f"frac>0.9={out['aa_noise']['frac_above_0.9']:.3f}", flush=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        out[label] = run(label, config, path, a.device, a.rows, a.batch)

    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    with open(txt, "w") as f:
        for label, e in out.items():
            f.write(f"{label}: config={e['config']} step={e['step']} slot_loop={e['is_slot_loop']} "
                    f"rows={e['rows']}\n")
            f.write(f"  AA(swap):  mean={e['aa_swap']['mean']:.4f} median={e['aa_swap']['median']:.4f} "
                    f"frac>0.9={e['aa_swap']['frac_above_0.9']:.3f} n={e['aa_swap']['n']}\n")
            f.write(f"  AA(noise): mean={e['aa_noise']['mean']:.4f} median={e['aa_noise']['median']:.4f} "
                    f"frac>0.9={e['aa_noise']['frac_above_0.9']:.3f} n={e['aa_noise']['n']}\n")
    print("wrote", a.out, "and", txt, flush=True)


if __name__ == "__main__":
    main()
