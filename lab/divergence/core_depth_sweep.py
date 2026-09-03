"""Core-contribution depth sweep: token CE AND the slot's own loss vs forced loop depth.

The per-iteration probe (probe.jsonl loop/delta_ratio_t*) shows every ladder arm's
loop keeps MOVING the slot state after t0 (~0.2-0.5 of its norm per iteration), but
norm motion is not nats. This measures what the iterations are WORTH: eval the same
packed rows at slot depth d = 1..max and read

  * per-token CE (all token positions, and span-first tokens separately — the
    positions the plan serves), and
  * `mux_local` — the slot's OWN local loss (arXiv 2607.18264) at that depth, when the
    arm trains one. This is the think-once panel's decisive column: it measures the
    loop's earning on the slot's job WITHOUT the coda in the way.

Depth forcing: in eval, `_sample_slot_depths` is the deterministic
`tul.slot_mean_depth or model.mean_depth` for every valid slot, so setting
`model.cfg.tul.slot_mean_depth = d` between evals forces depth d exactly (the
runtime-dispatch pattern tul_forward_ablated already uses for wrong_seed).

Rows are packed ONCE and reused for every depth -> the CE(d) curve is exactly paired,
and per-row sums are kept so the JSON carries a paired bootstrap CI over rows for
K1−K6, K3−K6 and K1−Kmax (`_stats.paired_bootstrap_ci`). `mux_local` is a batch
mean inside the labelled forward (which returns loss groups, not logits), so on a mux
arm every batch is run TWICE per depth — once for the CE map, once for the mux stats,
both at the same forced depth — and its interval resamples BATCHES; run with
`--batch 1` when that interval is the headline.

Usage:
  python lab/divergence/core_depth_sweep.py \
    --ckpt l3=tul_l3=checkpoints/morph/tul-l3/step_4500.pt \
    --depths 1,2,3,4,5,6,7,8 --rows 48 --out .../depth_sweep.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg
from _stats import paired_bootstrap_ci

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

MUX_KEYS = ("mux_local", "mux_n_supervised", "mux_rel", "mux_kl")


@torch.no_grad()
def ce_maps(model, inp, layout, labels, device, step_mode=None,
            want_mux: bool = False) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-position CE map ``[B, L]`` and, when asked, the forward's mux stats.

    The CE map comes from the label-free forward (full logits). The mux stats only
    exist on the LABELLED forward, which returns loss groups and no logits, so a mux
    arm pays a second forward per batch. Eval is deterministic at a forced depth, so
    both forwards see the same slot depths and the same (dropout-free) graph.
    """
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        tul_step_mode=step_mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    stats: dict[str, float] = {}
    if want_mux:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            res_l = model.tul_forward_ablated(inp.to(device), labels.to(device), layout,
                                              plan_mode="normal", tul_step_mode=step_mode)
        stats = {k: float(res_l[k]) for k in MUX_KEYS if k in res_l}
    return ce, stats


def _bootstrap_pairs(depths: list[int], unit_sum: dict[int, np.ndarray],
                     unit_cnt: np.ndarray) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for a, b in [(1, 6), (3, 6), (1, max(depths))]:
        if a in unit_sum and b in unit_sum and a != b:
            out[f"K{a}-K{b}"] = paired_bootstrap_ci(unit_sum[a], unit_sum[b], unit_cnt)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--depths", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-mode", default="auto", choices=["auto", "force-loop"],
                    help="auto: sigma-conditioned models use the Euler ladder "
                         "(db1_ladder_steps=d). force-loop: run the plain _tul_core "
                         "loop at forced depth d even on a sigma model (measures the "
                         "bptt-trained loop of a step_mix arm; tul_step_mode='bptt' "
                         "at eval opts out of the auto-ladder).")
    a = ap.parse_args()
    device = a.device
    depths = [int(x) for x in a.depths.split(",")]

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        # pack ALL rows once — every depth sees identical batches (paired curve)
        buf: list[int] = []
        need = a.batch * (spec.l_total + 1)
        batches = []
        rows_done = 0
        while rows_done < a.rows:
            while len(buf) < need:
                buf.extend(next(loader)[0].reshape(-1).tolist())
            inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
            batches.append((inp, labels, layout.to(device)))
            rows_done += a.batch
        # scoreable masks (token pos, label valid) + span-first flags, once
        masks = []
        dump = spec.max_slots
        for inp, labels, layout in batches:
            tokpos = (~layout.slot_mask.cpu()) & (labels >= 0)
            first = torch.zeros_like(tokpos)
            for b in range(inp.shape[0]):
                seen: set[int] = set()
                for p in range(inp.shape[1]):
                    if not bool(tokpos[b, p]):
                        continue
                    bag = int(layout.bag_id[b, p])
                    if bag not in seen:
                        seen.add(bag)
                        if 0 < bag < dump:
                            first[b, p] = True
            masks.append((tokpos, first))
        orig_mean = int(model.cfg.tul.slot_mean_depth)
        orig_max = int(model.cfg.tul.slot_max_depth)
        has_mux = float(model.cfg.tul.mux_beta) > 0.0
        arm = {"step": step, "rows": rows_done, "batch": a.batch, "eval_mode": a.eval_mode,
               "train_eval_depth":
               orig_mean or int(cfg.model.mean_depth), "depths": {},
               "mux_target": str(model.cfg.tul.mux_target) if has_mux else None,
               "cond_layers": int(model.cfg.tul.cond_layers),
               "detach_z": bool(model.cfg.tul.detach_z)}
        _sigma = (getattr(model.cfg.tul, "core_stage_cond", "none") == "sigma"
                  and a.eval_mode == "auto")
        _step_mode = "bptt" if a.eval_mode == "force-loop" else None
        orig_ladder = int(getattr(model.cfg.tul, "db1_ladder_steps", 0))
        # per-ROW bookkeeping (token CE) and per-BATCH bookkeeping (mux) for the CIs
        row_sum: dict[int, np.ndarray] = {}
        row_cnt: np.ndarray | None = None
        mux_sum: dict[int, np.ndarray] = {}
        mux_cnt: np.ndarray | None = None
        try:
            for d in depths:
                if _sigma:
                    # sigma-conditioned (db1) models: eval depth = Euler-ladder steps K
                    # (transformer.py: K = k_steps or cfg.tul.db1_ladder_steps or mean_depth);
                    # slot_mean_depth is ignored by that path.
                    model.cfg.tul.db1_ladder_steps = d
                model.cfg.tul.slot_mean_depth = d
                model.cfg.tul.slot_max_depth = max(d, orig_max or int(cfg.model.max_depth))
                tot = tot_n = fst = fst_n = 0.0
                rs: list[float] = []
                rc: list[float] = []
                ms: list[float] = []
                mc: list[float] = []
                for (inp, labels, layout), (tokpos, first) in zip(batches, masks):
                    ce, stats = ce_maps(model, inp, layout, labels, device,
                                        step_mode=_step_mode, want_mux=has_mux)
                    ce = ce.cpu()
                    tot += float(ce[tokpos].sum()); tot_n += int(tokpos.sum())
                    fst += float(ce[first].sum()); fst_n += int(first.sum())
                    rs.extend((ce * tokpos).sum(dim=1).tolist())
                    rc.extend(tokpos.sum(dim=1).tolist())
                    if "mux_local" in stats:
                        n_sup = stats.get("mux_n_supervised", 1.0)
                        ms.append(stats["mux_local"] * n_sup)
                        mc.append(n_sup)
                row_sum[d] = np.asarray(rs)
                if row_cnt is None:
                    row_cnt = np.asarray(rc)
                entry = {"ce_tokens": tot / tot_n, "ce_span_first": fst / fst_n,
                         "n_tokens": tot_n, "n_first": fst_n}
                if ms:
                    mux_sum[d] = np.asarray(ms)
                    if mux_cnt is None:
                        mux_cnt = np.asarray(mc)
                    entry["mux_local"] = float(np.sum(ms) / np.sum(mc))
                    entry["mux_n_supervised"] = float(np.sum(mc))
                arm["depths"][d] = entry
                print(f"{label:10s} depth={d}  ce={tot/tot_n:.4f}  "
                      f"span_first={fst/fst_n:.4f}"
                      + (f"  mux_local={entry['mux_local']:.4f}" if ms else ""), flush=True)
        finally:
            model.cfg.tul.slot_mean_depth = orig_mean
            model.cfg.tul.slot_max_depth = orig_max
            if _sigma:
                model.cfg.tul.db1_ladder_steps = orig_ladder
        assert row_cnt is not None
        arm["ci_ce_tokens"] = _bootstrap_pairs(depths, row_sum, row_cnt)
        if mux_sum and mux_cnt is not None:
            arm["ci_mux_local"] = _bootstrap_pairs(depths, mux_sum, mux_cnt)
        for k, v in arm["ci_ce_tokens"].items():
            print(f"{label:10s} ce_tokens {k}: {v['point']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} rows", flush=True)
        for k, v in arm.get("ci_mux_local", {}).items():
            print(f"{label:10s} mux_local {k}: {v['point']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} batches", flush=True)
        # per-row sums travel with the JSON so arm-vs-arm paired readouts on the same
        # rows can be computed offline without re-running the sweep
        arm["row_ce_sum"] = {str(d): row_sum[d].tolist() for d in depths}
        arm["row_n_tokens"] = row_cnt.tolist()
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
