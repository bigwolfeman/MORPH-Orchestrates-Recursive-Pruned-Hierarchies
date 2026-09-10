"""Core-loop anatomy of a SLOT-LOOP checkpoint: what each pass does to the slot states.

The twin of core_anatomy.py for `_tul_core` (tokens_through_core false). The compact
sequence is the valid slots of the packed rows; per forced pass: the slot-state norm, its
participation-ratio rank across valid slots, distance from and cosine to h_0 (the loop's
entry state, `core_init(e)`), the consecutive movement, and every core block's attention /
MLP branch out-in norm ratio over the valid slots. Plus the injection decay / dt.
Usage:
  python lab/divergence/slot_anatomy.py --ckpt LABEL=CONFIG=PATH --rows 12 --depth 8 --out .../slot_anatomy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from core_anatomy import _flat, eff_rank  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    label, config, path = a.ckpt.split("=", 2)
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    cfg = build_cfg(config, ["model.use_kernels=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("slot_anatomy needs a SLOT-LOOP model (tokens_through_core false)")
    model, step = load_ckpt(cfg, path, "cuda", tul_rt.model_cfg)
    model.eval()
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    n_core = int(cfg.model.n_core)
    tc = model.cfg.tul
    orig = (int(tc.slot_mean_depth), int(tc.slot_max_depth), int(tc.slot_depth_fixed))
    tc.slot_depth_fixed = a.depth
    tc.slot_max_depth = max(a.depth, orig[1] or int(cfg.model.max_depth))

    cur: dict = {"valid": None}
    branch: dict[str, dict[int, list[list[float]]]] = {"attn": {}, "mlp": {}}
    branch_full: dict[str, dict[int, list[list[float]]]] = {"attn": {}, "mlp": {}}
    inj_span = (int(model.injection.start), int(model.injection.end))
    full_norms: dict[str, list[float]] = {}
    ctx_shares: dict[str, list[float]] = {}
    calls = {"attn": 0, "mlp": 0}
    states: list[torch.Tensor] = []          # per pass, [N_valid, C] of this batch
    states_full: list[torch.Tensor] = []     # per pass, [N_valid, n*C]: every stream
    h0s_full: list[torch.Tensor] = []
    h0s: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []

    def _valid_rows(t: torch.Tensor) -> torch.Tensor:
        z = t.mean(dim=2) if t.dim() == 4 else t                # [B, S, C]
        return z.float()[cur["valid"]]

    def _valid_full(t: torch.Tensor) -> torch.Tensor:
        """[N_valid, n*C]: every stream, no averaging (a stream-mean can cancel)."""
        z = t.flatten(2) if t.dim() == 4 else t
        return z.float()[cur["valid"]]

    def _ctx_share(t: torch.Tensor) -> float:
        """Norm share of the injection's ctx channel in the stream-mean state."""
        z = _valid_rows(t)
        lo, hi = inj_span
        return float(z[:, lo:hi].norm() / (z.norm() + 1e-6))

    def branch_hook(kind: str):
        def f(_mod, args, out):
            t = calls[kind] // n_core
            i = calls[kind] % n_core
            calls[kind] += 1
            o = out[0] if isinstance(out, tuple) else out
            r = float(_valid_rows(o).norm() / (_valid_rows(args[0]).norm() + 1e-6))
            rf = float(_valid_full(o).norm() / (_valid_full(args[0]).norm() + 1e-6))
            branch[kind].setdefault(t, [[] for _ in range(n_core)])[i].append(r)
            branch_full[kind].setdefault(t, [[] for _ in range(n_core)])[i].append(rf)
        return f

    def block_hook(i: int):
        def f(_mod, _args, out):
            if i == n_core - 1:
                o = out if torch.is_tensor(out) else out[0]
                states.append(_valid_rows(o).cpu())
                states_full.append(_valid_full(o).cpu())
                full_norms.setdefault(f"pass{len(states)}", []).append(float(_valid_full(o).norm(dim=1).mean()))
                ctx_shares.setdefault(f"pass{len(states)}", []).append(_ctx_share(o))
        return f

    def h0_hook(_mod, _args, out):
        h0s.append(_valid_rows(out).cpu())
        h0s_full.append(_valid_full(out).cpu())
        full_norms.setdefault("h0", []).append(float(_valid_full(out).norm(dim=1).mean()))
        ctx_shares.setdefault("h0", []).append(_ctx_share(out))

    hooks = [model.core_init.register_forward_hook(h0_hook)]
    for i, blk in enumerate(model.core):
        hooks += [blk.attention.register_forward_hook(branch_hook("attn")),
                  blk.mlp.register_forward_hook(branch_hook("mlp")),
                  blk.register_forward_hook(block_hook(i))]
    per_batch: list[list[torch.Tensor]] = []
    # tul.reread: the read term's norm relative to the state it is added to, per pass
    read_ratio: dict[int, list[float]] = {}
    rr = getattr(model, "tul_reread", None)
    if rr is not None:
        _real_read = rr.read

        def _spy_read(h, k, v, allow, slot_valid):
            term = _real_read(h, k, v, allow, slot_valid)
            t = len(states)
            z = (h.mean(dim=2) if h.dim() == 4 else h).float()[cur["valid"]]
            read_ratio.setdefault(t, []).append(float(term.float()[cur["valid"]].norm() / (z.norm() + 1e-6)))
            return term
        rr.read = _spy_read
    per_batch_full: list[list[torch.Tensor]] = []
    try:
        for inp, labels, layout, _ in batches:
            layout = layout.to("cuda")
            cur["valid"] = layout.slot_valid
            calls["attn"] = calls["mlp"] = 0
            states.clear(); states_full.clear()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                model.tul_forward_ablated(inp.cuda(), None, layout, plan_mode="normal")
            assert len(states) == a.depth, (len(states), a.depth)
            per_batch.append(list(states))
            per_batch_full.append(list(states_full))
    finally:
        for h in hooks:
            h.remove()
        if rr is not None:
            rr.read = _real_read
        tc.slot_mean_depth, tc.slot_max_depth, tc.slot_depth_fixed = orig
    h0 = torch.cat(h0s)                                            # [N, C]
    traj = [torch.cat([pb[t] for pb in per_batch]) for t in range(a.depth)]
    h0f = torch.cat(h0s_full)
    trajf = [torch.cat([pb[t] for pb in per_batch_full]) for t in range(a.depth)]
    rec = {
        "label": label, "step": step, "rows": a.rows, "depth": a.depth, "slots": int(h0.shape[0]),
        "h0": {"norm_per_slot": float(h0.norm(dim=1).mean()), "rank": eff_rank(h0)},
        "passes": [{"t": t + 1, "norm_per_slot": float(s.norm(dim=1).mean()), "rank": eff_rank(s),
                    "dist_from_h0": float(((s - h0).norm(dim=1) / h0.norm(dim=1).clamp_min(1e-6)).mean()),
                    "cos_to_h0": float(F.cosine_similarity(s, h0, dim=1).mean())}
                   for t, s in enumerate(traj)],
        "consecutive": [float(((traj[k] - traj[k - 1]).norm(dim=1)
                               / traj[k - 1].norm(dim=1).clamp_min(1e-6)).mean())
                        for k in range(1, len(traj))],
        "first_pass_from_h0": float(((traj[0] - h0).norm(dim=1) / h0.norm(dim=1).clamp_min(1e-6)).mean()),
        # the same movement on the FULL carrier (every HC stream): the reading the coda's
        # W_prefix sees; the stream mean can cancel across streams and inflate the ratio
        "consecutive_full": [float(((trajf[k] - trajf[k - 1]).norm(dim=1)
                                    / trajf[k - 1].norm(dim=1).clamp_min(1e-6)).mean())
                             for k in range(1, len(trajf))],
        "first_pass_from_h0_full": float(((trajf[0] - h0f).norm(dim=1) / h0f.norm(dim=1).clamp_min(1e-6)).mean()),
        "branch_out_in": {k: {str(t): [sum(v) / len(v) for v in per_block]
                              for t, per_block in d.items()} for k, d in branch.items()},
        "branch_out_in_full": {k: {str(t): [sum(v) / len(v) for v in per_block]
                                   for t, per_block in d.items()} for k, d in branch_full.items()},
        "full_carrier_norm_per_slot": {k: sum(v) / len(v) for k, v in full_norms.items()},
        "reread_term_over_state": {str(t + 1): sum(v) / len(v) for t, v in read_ratio.items()},
        "ctx_channel_norm_share": {k: sum(v) / len(v) for k, v in ctx_shares.items()},
    }
    inj = model.injection
    rec["injection"] = {"A_mean": float(inj.log_A.exp().mean()), "dt_mean": float(inj.log_dt.exp().mean()),
                        "span": [inj.start, inj.end],
                        "B_diag_mean": float(inj.B.diag().mean()) if inj.B is not None else None}
    json.dump(rec, open(a.out, "w"), indent=1)
    print(f"{label} step {step}: {rec['slots']} valid slots, h0 |h| {rec['h0']['norm_per_slot']:.2f} rank {rec['h0']['rank']:.1f}")
    for c in rec["passes"]:
        print(f"{label} pass {c['t']}: |h| {c['norm_per_slot']:.2f} rank {c['rank']:.1f} |h-h0|/|h0| {c['dist_from_h0']:.3f} cos {c['cos_to_h0']:.4f}")
    print(f"{label} movement: first {rec['first_pass_from_h0']:.4f} then", [round(x, 4) for x in rec["consecutive"]])
    print(f"{label} movement FULL carrier: first {rec['first_pass_from_h0_full']:.4f} then", [round(x, 4) for x in rec["consecutive_full"]])
    for kind in ("attn", "mlp"):
        last = max(branch[kind])
        print(f"{label} {kind} out/in per block, pass 1 and {last + 1}:",
              [round(x, 3) for x in rec["branch_out_in"][kind]["0"]],
              [round(x, 3) for x in rec["branch_out_in"][kind][str(last)]])
    for kind in ("attn", "mlp"):
        last = max(branch_full[kind])
        print(f"{label} {kind} out/in FULL carrier, pass 1 and {last + 1}:",
              [round(x, 3) for x in rec["branch_out_in_full"][kind]["0"]],
              [round(x, 3) for x in rec["branch_out_in_full"][kind][str(last)]])
    print(f"{label} full-carrier |h| per slot:", {k: round(v, 2) for k, v in rec["full_carrier_norm_per_slot"].items()})
    print(f"{label} ctx-channel norm share:", {k: round(v, 3) for k, v in rec["ctx_channel_norm_share"].items()})
    if rec["reread_term_over_state"]:
        print(f"{label} reread term / state per pass:", {k: round(v, 3) for k, v in rec["reread_term_over_state"].items()})
    print(f"{label} injection", rec["injection"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
