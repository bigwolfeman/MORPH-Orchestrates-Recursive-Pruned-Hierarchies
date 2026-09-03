"""E1 — mask-surgery decomposition: TUL's visibility imposed on the trained noTUL model.

Prereg: lab/experiments/planned/2026-09-01-mask-surgery-decomposition.md.
Eval-only. Replaces every attention module's forward with the tg_restrict
branch VERBATIM (project -> _tg_slot_attention over CARRIER positions +
_window_attn(extra_mask) -> _gate_combine_up), where the carrier of span j is
its boundary token (the position a slot would sit after). Spans come from the
SAME BoundaryRule TUL trains with, applied to the raw token stream (no slot
insertion). Variants share the carrier compressed branch:

  base  unpatched forward (sanity anchor vs the token depth sweep)
  e1c   window UNRESTRICTED           — prices the compressor->carrier swap
  e1b   window same-span-or-carrier   — the full TUL-visibility analogue
  e1a   window same-span-only         — harsh floor

Gate (runs first): _window_attn with an all-True extra_mask must match
extra_mask=None bitwise on the same q,k,v — validates the mask plumbing.

Usage:
  python lab/divergence/mask_surgery.py \
    --ckpt notul=notul_bg0c0=checkpoints/morph/notul-20k/step_20000.pt \
    --rule-config tul_g0c0 --rows 48 \
    --out /home/wolfe/morph-scratch/tulfm/mask_surgery.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.attention import (_CCACSAAttention, _CCAHCAAttention,
                                   _tg_slot_attention)

# Per-batch masks the patched forwards read (set in the eval loop).
MASKS: dict = {"allow": None, "carrier": None}


def make_patched(mod):
    cca = mod.cca

    def fwd(x, n_skip_rope: int = 0, cla_capture=None, cla_kv=None,
            tg_allow=None, tg_slot_mask=None):
        assert cla_kv is None and cla_capture is None
        B, S, _ = x.shape
        D = cca.d_head
        scale = D ** -0.5
        q, k, v, q_lat, k_lat = cca._cca_project(x, n_skip_rope,
                                                 return_klat=True, pre=None)
        out_comp = _tg_slot_attention(q, k, v, MASKS["carrier"],
                                      cca.sink_logits, scale)
        out_win = cca._window_attn(q, k, v, x.device, scale, n_skip_rope,
                                   extra_mask=MASKS["allow"])
        return cca._gate_combine_up(x, out_comp, out_win, q_lat=q_lat,
                                    gate_pre=None)

    return fwd


def span_ids(ids_row: np.ndarray, rule) -> tuple[np.ndarray, np.ndarray]:
    """(bag_id [L] int64, carrier [L] bool) for one raw row. A slot goes AFTER
    ids[p] for each boundary p, so position i's span = #boundaries strictly
    before i, and the carriers are the boundary tokens themselves."""
    pos, _ = rule.cut(ids_row)
    bag = np.searchsorted(pos, np.arange(ids_row.shape[0]), side="left")
    car = np.zeros(ids_row.shape[0], dtype=bool)
    car[pos] = True
    return bag.astype(np.int64), car


@torch.no_grad()
def ce_map(model, x, y, device) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(x.to(device), labels=y.to(device))
    assert "logits" in out, f"forward returned {list(out)} — need logits for strata"
    logits = out["logits"].float()
    B, L, V = logits.shape
    lab = y.to(device).clone()
    lab[lab < 0] = 0
    return F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                           reduction="none").reshape(B, L).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rule-config", default="tul_g0c0",
                    help="TUL config whose BoundaryRule defines the spans")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path = a.ckpt.split("=", 2)
    cfg = build_cfg(config, ["model.use_kernels=false"])
    assert build_tul_runtime(cfg) is None, "E1 wants the noTUL model"
    rule = build_tul_runtime(
        build_cfg(a.rule_config, ["model.use_kernels=false"])).data_cfg.rule

    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, None)
    model.eval()

    # ── rows + masks, drawn once (paired across variants) ────────────────────
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                               cfg.data.seq_len, a.batch, split="validation",
                               skip_samples=0, bag_size=0, tul=None)
    batches = []
    while len(batches) * a.batch < a.rows:
        x, y = next(loader)[:2]
        B, L = x.shape
        bag = np.zeros((B, L), dtype=np.int64)
        car = np.zeros((B, L), dtype=bool)
        for b in range(B):
            bag[b], car[b] = span_ids(x[b].numpy(), rule)
        bag_t = torch.from_numpy(bag)
        car_t = torch.from_numpy(car)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        same = bag_t.unsqueeze(2) == bag_t.unsqueeze(1)              # [B, L, L]
        allow_a = (same & causal).unsqueeze(1)                        # [B,1,L,L]
        allow_b = ((same | car_t.unsqueeze(1)) & causal).unsqueeze(1)
        # span-first stratum: first token AFTER a boundary, excluding span 0
        first = torch.zeros(B, L, dtype=torch.bool)
        first[:, 1:] = (bag_t[:, 1:] != bag_t[:, :-1]) & (bag_t[:, 1:] > 0)
        first &= (y >= 0)
        tokv = (y >= 0)
        batches.append((x, y, allow_a.to(device), allow_b.to(device),
                        car_t.to(device), tokv, first))
    n_car = float(np.mean([float(c[4].float().sum(-1).mean()) for c in batches]))
    print(f"rows={len(batches) * a.batch} seq={cfg.data.seq_len} "
          f"carriers/row={n_car:.1f}", flush=True)

    # ── gate: all-True extra_mask must equal extra_mask=None bitwise ─────────
    mods = [m for m in model.modules()
            if isinstance(m, (_CCACSAAttention, _CCAHCAAttention))]
    assert mods, "no attention modules found"
    x0 = batches[0][0][:1].to(device)
    with torch.no_grad():
        h = model.embed(x0).float()
        cca = mods[0].cca
        q, k, v, *_ = cca._cca_project(h, 0, return_klat=True, pre=None)
        Lg = h.shape[1]
        full = torch.ones(1, 1, Lg, Lg, dtype=torch.bool, device=device)
        w_none = cca._window_attn(q, k, v, device, cca.d_head ** -0.5, 0)
        w_full = cca._window_attn(q, k, v, device, cca.d_head ** -0.5, 0,
                                  extra_mask=full)
        gate_max = float((w_none - w_full).abs().max())
    print(f"GATE extra_mask plumbing: max|Δ|={gate_max:.3e} "
          f"({'PASS' if gate_max < 1e-5 else 'FAIL'})", flush=True)
    assert gate_max < 1e-5, "extra_mask=all-True must be a no-op"

    originals = {id(m): m.forward for m in mods}
    variants = {
        "base": None,
        "e1c": ("none",),
        "e1b": ("allow_b",),
        "e1a": ("allow_a",),
    }
    results: dict[str, dict] = {"step": step, "rows": len(batches) * a.batch,
                                "carriers_per_row": n_car, "variants": {}}
    try:
        for name, spec in variants.items():
            if spec is None:
                for m in mods:
                    m.forward = originals[id(m)]
            else:
                for m in mods:
                    m.forward = make_patched(m)
            tot = tot_n = fst = fst_n = 0.0
            for (x, y, aa, ab, car, tokv, first) in batches:
                if spec is not None:
                    MASKS["carrier"] = car
                    MASKS["allow"] = (None if spec[0] == "none"
                                      else (ab if spec[0] == "allow_b" else aa))
                ce = ce_map(model, x, y, device)
                tot += float(ce[tokv].sum()); tot_n += int(tokv.sum())
                fst += float(ce[first].sum()); fst_n += int(first.sum())
            r = {"ce": tot / tot_n, "ce_span_first": fst / fst_n,
                 "n_tokens": tot_n, "n_first": fst_n}
            results["variants"][name] = r
            print(f"{name:5s} ce={r['ce']:.4f}  span_first={r['ce_span_first']:.4f}",
                  flush=True)
    finally:
        for m in mods:
            m.forward = originals[id(m)]
    b = results["variants"]["base"]
    for name in ("e1c", "e1b", "e1a"):
        v = results["variants"][name]
        v["delta_ce"] = v["ce"] - b["ce"]
        v["delta_span_first"] = v["ce_span_first"] - b["ce_span_first"]
        print(f"Δ({name}) = {v['delta_ce']:+.4f}  span_first "
              f"{v['delta_span_first']:+.4f}", flush=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
