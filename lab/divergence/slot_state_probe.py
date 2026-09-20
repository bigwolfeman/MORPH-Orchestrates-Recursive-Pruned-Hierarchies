"""Does the slot loop MOVE its state with depth? (2026-09-09, slot-loop panel.)

The depth sweep on an A1-style slot-loop arm reads a token curve flat to 1e-5 over forced
slot depths 1..16. Two mechanisms give that reading: the loop reaches its fixed point by
iteration 1 (the state at depth 16 IS the state at depth 1), or the state moves and the
tokens do not read the difference (the arc's F3). This probe separates them: it captures
the looped slot state handed to `prefix_project` at each forced depth and reports, over the
valid slots, the relative distance from the depth-1 state and the per-iteration movement.

Usage: python lab/divergence/slot_state_probe.py --ckpt LABEL=CONFIG=PATH --depths 1,2,3,6,16 --rows 12
"""
from __future__ import annotations
import argparse
import json
import torch
from _build import ROOT, build_cfg, parse_ckpt_spec
from _rows import pack_rows, stream_from_loader
import sys
sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--depths", default="1,2,3,6,16")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    depths = [int(x) for x in a.depths.split(",")]
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_state_probe needs a SLOT-LOOP model (tokens_through_core false)")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    model.eval()
    if getattr(model, "tul_code_enc", None) is not None:
        # TUL-Code (docs/tul-code-spec.md): no slot loop runs and `prefix_project` is never
        # called, so there is no per-depth written state to capture. The code's rank and
        # cosine are `val/code_eff_rank` / `val/code_pairwise_cos` in the trainer's eval
        # and the K-curve is core_depth_sweep.py (depth = sampler steps, 0 = the encoder).
        print(f"{label} is a TUL-Code model: slot_state_probe does not apply "
              f"(no slot loop; see val/code_eff_rank and the sweep's ce_0)", flush=True)
        if a.out:
            with open(a.out, "w") as f:
                json.dump({"label": label, "step": step, "skipped": "tul.code model: no "
                           "slot loop, no per-depth state; see val/code_eff_rank"}, f)
        return
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    tc = model.cfg.tul
    orig_mean, orig_max = int(tc.slot_mean_depth), int(tc.slot_max_depth)
    # k-fixed arms (tul.slot_depth_fixed > 0) IGNORE slot_mean_depth at eval, so forcing a
    # depth must go through whichever knob the model reads — core_depth_sweep.py's rule.
    # Before 2026-09-14 this probe set the mean only and read the SAME state at every
    # depth on the fixed-6 horizon control (rel_dist 0.0, cos 1.0 at 1/2/3/6/16).
    orig_fixed = int(getattr(tc, "slot_depth_fixed", 0))
    captured: dict[int, list[torch.Tensor]] = {d: [] for d in depths}
    valids: list[torch.Tensor] = []
    real_project = model.tul.prefix_project
    cur = {"d": None}

    def spy(h_slots, layout, l_total, cells=None):
        captured[cur["d"]].append(h_slots.detach().float().cpu())
        return real_project(h_slots, layout, l_total, cells=cells)

    model.tul.prefix_project = spy
    try:
        for d in depths:
            cur["d"] = d
            if orig_fixed > 0:
                tc.slot_depth_fixed = d
            else:
                tc.slot_mean_depth = d
            tc.slot_max_depth = max(d, orig_max or int(cfg.model.max_depth))
            for inp, labels, layout, _ in batches:
                if d == depths[0]:
                    valids.append(layout.slot_valid.cpu())
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    model.tul_forward_ablated(inp.cuda(), None, layout.to("cuda"), plan_mode="normal")
    finally:
        model.tul.prefix_project = real_project
        tc.slot_mean_depth, tc.slot_max_depth = orig_mean, orig_max
        tc.slot_depth_fixed = orig_fixed
    valid = torch.cat(valids)                                        # [N, S]
    states = {d: torch.cat(captured[d]) for d in depths}             # [N, S, ...]
    flat = {d: states[d].reshape(states[d].shape[0], states[d].shape[1], -1)[valid] for d in depths}
    base = flat[depths[0]]
    if len(depths) > 1 and all(torch.equal(flat[d], base) for d in depths[1:]):
        raise RuntimeError("slot_state_probe: the state is bit-identical at every forced depth, "
                           "so the depth knob this probe set is not the one the model reads")
    out = {"label": label, "step": step, "rows": int(valid.shape[0]), "valid_slots": int(valid.sum()),
           "state_dim": int(base.shape[1]), "ref_depth": depths[0], "per_depth": {}}
    prev = None
    for d in depths:
        h = flat[d]
        rel = ((h - base).norm(dim=1) / base.norm(dim=1).clamp_min(1e-6))
        rec = {"norm_mean": float(h.norm(dim=1).mean()),
               "rel_dist_from_ref_mean": float(rel.mean()), "rel_dist_from_ref_p90": float(rel.quantile(0.9)),
               "cos_to_ref_mean": float(torch.nn.functional.cosine_similarity(h, base, dim=1).mean())}
        if prev is not None:
            rec["rel_dist_from_prev_depth_mean"] = float(((h - prev).norm(dim=1) / prev.norm(dim=1).clamp_min(1e-6)).mean())
        prev = h
        out["per_depth"][d] = rec
        print(f"{label} depth={d:2d} |h|={rec['norm_mean']:.3f} rel_dist_from_d{depths[0]}={rec['rel_dist_from_ref_mean']:.4f} "
              f"(p90 {rec['rel_dist_from_ref_p90']:.4f}) cos={rec['cos_to_ref_mean']:.5f}"
              + (f" rel_from_prev={rec['rel_dist_from_prev_depth_mean']:.4f}" if 'rel_dist_from_prev_depth_mean' in rec else ""), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
