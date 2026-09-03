"""Is the coda's readout of the plan DEAD at the gradient level?

`region_shapley.py` and `slot_path_worth.py` measured the plan's WORTH to the main
token CE (0.0007-0.02 nats). Worth near zero is consistent with two worlds:

  1. the plan carries nothing (an upstream problem), or
  2. the coda has learned to IGNORE the slot positions — the readout is dead, so the
     gradient of ce_main w.r.t. the slot states at the coda input is ~0, and any
     plan-side objective is starved of useful gradient regardless of design.

This measures world 2 directly. It patches `morph.model.transformer.scatter_positions`
(the call that writes the projected plan into the coda's input sequence,
`_forward_tul`) to retain the gradient of its output `x_coda`, runs the eval forward
with grads enabled, backwards ce_main ONLY (never the total loss, never ce_emit /
ce_plast — ce_emit is the slot's private one-token loss and would contaminate the
channel), and compares per-position grad norms at SLOT positions vs TOKEN positions.

    PYTHONPATH=. python lab/divergence/readout_jacobian.py \
        --ckpts checkpoints/morph/onset-capture/ROLL_step_1750.pt

NOT pre-registered — diagnostic probe.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402
from morph.training.train import load_checkpoint            # noqa: E402
import morph.model.transformer as _tf                        # noqa: E402

# ── scatter_positions capture ───────────────────────────────────────────────
# transformer.py imports scatter_positions into its own namespace, so patching
# morph.model.transformer.scatter_positions intercepts exactly the transformer.py
# call sites: line ~2059 (`_forward_tul`, the loss path) and line ~2222
# (`_tul_coda_prep`, used only by tul_forward_cw_arms — never fired by an ordinary
# forward, which the per-batch assertion below verifies).
_ORIG_SCATTER = _tf.scatter_positions
CAPTURED: list[tuple[torch.Tensor, torch.Tensor]] = []


def _capturing_scatter(x, index, values):
    out = _ORIG_SCATTER(x, index, values)
    if torch.is_grad_enabled() and out.requires_grad:
        out.retain_grad()
    CAPTURED.append((out, index))
    return out


def _stats(v: torch.Tensor) -> dict:
    v = v.float()
    return {
        "n": int(v.numel()),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "p90": float(torch.quantile(v, 0.90)),
        "max": float(v.max()),
    }


def probe_checkpoint(ckpt_path: str, cfg, tul_data_cfg, batches) -> dict:
    """Build the model, load `ckpt_path`, backward ce_main per batch, return stats."""
    model, _ = build_model(cfg, device="cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    load_checkpoint(ckpt_path, model, scaler, torch.device("cuda"))
    model.eval()   # want_groups=True path (ce_main is computed), dropout off,
                   # deterministic mean slot depth — the forward is fully paired
                   # across checkpoints on the fixed batch list.

    acc = {grp: {"g": [], "h": []} for grp in ("slot", "token", "pad")}
    ce_mains: list[float] = []
    captures_per_forward: list[int] = []
    ambiguous_captures = 0

    for x, y, layout in batches:
        CAPTURED.clear()
        model.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y, slot_layout=layout)
        captures_per_forward.append(len(CAPTURED))
        if len(CAPTURED) == 1:
            x_coda, pos = CAPTURED[0]
        else:
            # Keep the capture whose shape matches the loss path ([B, L_total, ...])
            # and record the fact — the assertion in main() reports it.
            ambiguous_captures += 1
            L = layout.l_total
            matches = [(t, p) for (t, p) in CAPTURED if t.shape[1] == L]
            if len(matches) != 1:
                raise RuntimeError(
                    f"{len(CAPTURED)} scatter captures, {len(matches)} match "
                    f"L_total={L}; cannot identify the loss-path tensor.")
            x_coda, pos = matches[0]

        ce_main = out["ce_main"]
        if not math.isfinite(float(ce_main)):
            raise RuntimeError(f"ce_main is not finite: {float(ce_main)}")
        ce_mains.append(float(ce_main))

        ce_main.backward()

        if x_coda.grad is None:
            raise RuntimeError("x_coda.grad is None — retain_grad/backward broken.")

        B, L = x_coda.shape[0], x_coda.shape[1]
        # Per-position L2 over EVERY trailing dim (the carrier is [B, L, n, C] with
        # n=4 HyperConnection streams here; a plain [B, L, C] carrier works the same).
        g = x_coda.grad.detach().float().reshape(B, L, -1).norm(dim=-1)   # [B, L]
        h = x_coda.detach().float().reshape(B, L, -1).norm(dim=-1)        # [B, L]

        # Group masks. `pos` is prefix_project's index: entries of INVALID slots point
        # at the dump row L, so `pos < L` selects exactly the real scattered slot
        # positions. slot_mask is True at every slot position INCLUDING tail pads, so
        # ~slot_mask is exactly the real-token set and slot_mask & ~scattered is pads.
        slot_scatter = torch.zeros(B, L + 1, dtype=torch.bool, device=x_coda.device)
        slot_scatter.scatter_(1, pos, True)
        slot_scatter = slot_scatter[:, :L]
        masks = {
            "slot": slot_scatter,
            "token": ~layout.slot_mask,
            "pad": layout.slot_mask & ~slot_scatter,
        }
        assert not (masks["slot"] & masks["token"]).any(), "slot/token masks overlap"
        for grp, m in masks.items():
            acc[grp]["g"].append(g[m].cpu())
            acc[grp]["h"].append(h[m].cpu())

        del out, x_coda, g, h
        CAPTURED.clear()

    model.zero_grad(set_to_none=True)
    del model
    torch.cuda.empty_cache()

    res = {
        "ckpt": ckpt_path,
        "ce_main_per_batch": ce_mains,
        "ce_main_mean": sum(ce_mains) / len(ce_mains),
        "captures_per_forward": captures_per_forward,
        "ambiguous_capture_forwards": ambiguous_captures,
        "groups": {},
    }
    for grp in ("slot", "token", "pad"):
        g = torch.cat(acc[grp]["g"])
        h = torch.cat(acc[grp]["h"])
        res["groups"][grp] = {"g": _stats(g), "h": _stats(h), "gh": _stats(g * h)}

    med = lambda grp, k: res["groups"][grp][k]["median"]
    res["ratio_median_g_slot_over_token"] = med("slot", "g") / med("token", "g")
    res["ratio_median_gh_slot_over_token"] = med("slot", "gh") / med("token", "gh")

    # Sanity checks (each reported, all must hold before the numbers are trusted).
    res["sanity"] = {
        "capture_fired_once_per_forward": all(c == 1 for c in captures_per_forward),
        "grad_not_none": True,   # a None grad raised above
        "ce_main_finite_and_plausible": all(math.isfinite(c) and 3.0 <= c <= 8.0
                                            for c in ce_mains),
        "slot_input_norms_nonzero": med("slot", "h") > 0.0,
        "token_grad_norms_nonzero": med("token", "g") > 0.0,
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--config", default="tul_a2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    _tf.scatter_positions = _capturing_scatter

    cfg = build_cfg(a.config, ["training.batch_size=6", "model.use_kernels=false"])
    # The SAME fixed eval set for every checkpoint: same loader args as
    # region_shapley.py / slot_path_worth.py, batches materialised ONCE.
    from morph.training.tul_setup import build_tul_runtime
    tul_rt = build_tul_runtime(cfg)
    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len),
        int(cfg.training.batch_size), split="validation", skip_samples=60_000,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    print(f"fixed eval set: {len(batches)} batches of {batches[0][0].shape[0]}\n")

    all_res = []
    for ckpt in a.ckpts:
        print(f"── {ckpt}")
        r = probe_checkpoint(ckpt, cfg, tul_rt, batches)
        all_res.append(r)
        print(f"  ce_main mean {r['ce_main_mean']:.4f}   "
              f"captures/forward {r['captures_per_forward']}")
        hdr = f"  {'group':<7} {'stat':<5} {'mean':>12} {'median':>12} {'p90':>12} {'max':>12}   n"
        print(hdr)
        for grp in ("slot", "token", "pad"):
            for k in ("g", "h", "gh"):
                s = r["groups"][grp][k]
                print(f"  {grp:<7} {k:<5} {s['mean']:>12.4e} {s['median']:>12.4e} "
                      f"{s['p90']:>12.4e} {s['max']:>12.4e}   {s['n']}")
        print(f"  HEADLINE median_g slot/token   = "
              f"{r['ratio_median_g_slot_over_token']:.4f}")
        print(f"  HEADLINE median_gh slot/token  = "
              f"{r['ratio_median_gh_slot_over_token']:.4f}")
        print(f"  sanity: {r['sanity']}\n")

    if a.out:
        with open(a.out, "w") as fh:
            json.dump(all_res, fh, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
