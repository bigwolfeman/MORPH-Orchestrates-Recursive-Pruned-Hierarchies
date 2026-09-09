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
from _rows import pack_rows, stream_from_loader
from _stats import paired_bootstrap_ci

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

MUX_KEYS = ("mux_local", "mux_n_supervised", "mux_rel", "mux_kl",
            "mux_local_own_final", "mux_n_supervised_own",
            "mux_local_next_final", "mux_n_supervised_next")
# metric name -> (value key, count key) for the batch-weighted means and their CIs.
# The last two exist only on a staged-target arm (tul.mux_stage_own_iters > 0).
MUX_METRICS = {"mux_local": ("mux_local", "mux_n_supervised"),
               "mux_local_own": ("mux_local_own_final", "mux_n_supervised_own"),
               "mux_local_next": ("mux_local_next_final", "mux_n_supervised_next")}


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
        if layout is None:  # the plain control: the ordinary forward at cfg.mean_depth
            res = model(inp.to(device), labels=None)
        else:
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
        plain = tul_rt is None  # the plain control (tul.activate_at: never)
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        # The SAME validation stream for every arm, packed by the arm's own cut (the
        # trainer's packer for a TUL arm, non-overlapping seq_len+1 rows for the plain
        # control), with the stream index of every scored position kept: two arms that
        # cut the stream differently pair at the TOKEN level offline. Every depth sees
        # identical batches (the paired curve).
        row_tokens = (int(cfg.data.seq_len) + 1 if plain
                      else tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1)
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, plain)[:n_batches]
        rows_done = sum(inp.shape[0] for inp, _, _, _ in batches)
        # scoreable masks (token pos, label valid) + span-first flags, once
        masks = []
        dump = 0 if plain else tul_rt.data_cfg.spec_for(cfg.data.seq_len).max_slots
        for inp, labels, layout, idx in batches:
            tokpos = labels >= 0 if layout is None else (~layout.slot_mask) & (labels >= 0)
            first = torch.zeros_like(tokpos)
            if layout is not None:
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
            masks.append((tokpos, first, idx))
        batches = [(inp, labels, (lay.to(device) if lay is not None else None))
                   for inp, labels, lay, _ in batches]
        tc = None if plain else model.cfg.tul
        orig_mean = int(model.cfg.mean_depth if plain else tc.slot_mean_depth)
        orig_max = 0 if plain else int(tc.slot_max_depth)
        # k-fixed arms (tul.slot_depth_fixed > 0, the 2026-09-07 k=12 panel) ignore the mean
        # at eval, so the forced depth must go through the fixed knob as well.
        orig_fixed = 0 if plain else int(getattr(tc, "slot_depth_fixed", 0))
        has_mux = (not plain) and float(tc.mux_beta) > 0.0
        arm = {"step": step, "rows": rows_done, "batch": a.batch, "eval_mode": a.eval_mode,
               "plain": plain,
               "train_eval_depth":
               orig_fixed or orig_mean or int(cfg.model.mean_depth), "depths": {},
               "mux_target": str(tc.mux_target) if has_mux else None,
               "cond_layers": 0 if plain else int(tc.cond_layers),
               "detach_z": False if plain else bool(tc.detach_z)}
        _sigma = ((not plain) and getattr(tc, "core_stage_cond", "none") == "sigma"
                  and a.eval_mode == "auto")
        _step_mode = "bptt" if a.eval_mode == "force-loop" else None
        orig_ladder = 0 if plain else int(getattr(tc, "db1_ladder_steps", 0))
        # the stream index of every scored position, in row order (one array; the same
        # order the per-token CE arrays below use)
        tok_index = np.concatenate([idx[tokpos].numpy() for tokpos, _, idx in masks]).astype(np.int32)
        tok_ce: dict[int, np.ndarray] = {}
        # per-ROW bookkeeping (token CE) and per-BATCH bookkeeping (mux) for the CIs
        row_sum: dict[int, np.ndarray] = {}
        row_cnt: np.ndarray | None = None
        mux_sum: dict[str, dict[int, np.ndarray]] = {m: {} for m in MUX_METRICS}
        mux_cnt: dict[str, np.ndarray | None] = {m: None for m in MUX_METRICS}
        try:
            for d in depths:
                if plain:
                    # the plain forward's eval depth is a uniform cfg.mean_depth fill with
                    # no clamp at eval (transformer.py, the `else` of `if self.training`)
                    model.cfg.mean_depth = d
                else:
                    if _sigma:
                        # sigma-conditioned (db1) models: eval depth = Euler-ladder steps K
                        # (transformer.py: K = k_steps or cfg.tul.db1_ladder_steps or
                        # mean_depth); slot_mean_depth is ignored by that path.
                        tc.db1_ladder_steps = d
                    tc.slot_mean_depth = d
                    tc.slot_max_depth = max(d, orig_max or int(cfg.model.max_depth))
                    if orig_fixed > 0:
                        tc.slot_depth_fixed = d
                tot = tot_n = fst = fst_n = 0.0
                rs: list[float] = []
                rc: list[float] = []
                ms: dict[str, list[float]] = {m: [] for m in MUX_METRICS}
                mc: dict[str, list[float]] = {m: [] for m in MUX_METRICS}
                ces: list[np.ndarray] = []
                for (inp, labels, layout), (tokpos, first, _) in zip(batches, masks):
                    ce, stats = ce_maps(model, inp, layout, labels, device,
                                        step_mode=_step_mode, want_mux=has_mux)
                    ce = ce.cpu()
                    ces.append(ce[tokpos].numpy().astype(np.float32))
                    tot += float(ce[tokpos].sum())
                    tot_n += int(tokpos.sum())
                    fst += float(ce[first].sum())
                    fst_n += int(first.sum())
                    rs.extend((ce * tokpos).sum(dim=1).tolist())
                    rc.extend(tokpos.sum(dim=1).tolist())
                    for m, (vk, ck) in MUX_METRICS.items():
                        if vk in stats:
                            n_sup = stats.get(ck, 1.0)
                            ms[m].append(stats[vk] * n_sup)
                            mc[m].append(n_sup)
                row_sum[d] = np.asarray(rs)
                tok_ce[d] = np.concatenate(ces)
                if row_cnt is None:
                    row_cnt = np.asarray(rc)
                entry = {"ce_tokens": tot / tot_n,
                         "ce_span_first": fst / fst_n if fst_n else float("nan"),
                         "n_tokens": tot_n, "n_first": fst_n}
                for m in MUX_METRICS:
                    if ms[m]:
                        mux_sum[m][d] = np.asarray(ms[m])
                        if mux_cnt[m] is None:
                            mux_cnt[m] = np.asarray(mc[m])
                        entry[m] = float(np.sum(ms[m]) / np.sum(mc[m]))
                        entry[f"{m}_n_supervised"] = float(np.sum(mc[m]))
                arm["depths"][d] = entry
                print(f"{label:10s} depth={d}  ce={tot/tot_n:.4f}  "
                      f"span_first={entry['ce_span_first']:.4f}"
                      + "".join(f"  {m}={entry[m]:.4f}" for m in MUX_METRICS if m in entry),
                      flush=True)
        finally:
            if plain:
                model.cfg.mean_depth = orig_mean
            else:
                tc.slot_mean_depth = orig_mean
                tc.slot_max_depth = orig_max
                if orig_fixed > 0:
                    tc.slot_depth_fixed = orig_fixed
                if _sigma:
                    tc.db1_ladder_steps = orig_ladder
        assert row_cnt is not None
        arm["ci_ce_tokens"] = _bootstrap_pairs(depths, row_sum, row_cnt)
        for m in MUX_METRICS:
            if mux_sum[m] and mux_cnt[m] is not None:
                arm[f"ci_{m}"] = _bootstrap_pairs(depths, mux_sum[m], mux_cnt[m])
        for k, v in arm["ci_ce_tokens"].items():
            print(f"{label:10s} ce_tokens {k}: {v['point']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} rows", flush=True)
        for m in MUX_METRICS:
            for k, v in arm.get(f"ci_{m}", {}).items():
                print(f"{label:10s} {m} {k}: {v['point']:+.4f} "
                      f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} batches",
                      flush=True)
        # per-row sums travel with the JSON so arm-vs-arm paired readouts on the same
        # rows can be computed offline without re-running the sweep
        arm["row_ce_sum"] = {str(d): row_sum[d].tolist() for d in depths}
        arm["row_n_tokens"] = row_cnt.tolist()
        # per-token CE keyed by stream index (float32, ~14 MB per arm at 480 rows x 7
        # depths) beside the JSON: arms that cut the stream differently pair at the token
        # level offline (block-bootstrap over stream blocks, score_e18.py)
        npz = a.out.rsplit(".", 1)[0] + f".{label}.tokens.npz"
        np.savez_compressed(npz, tok_index=tok_index,
                            **{f"ce_{d}": tok_ce[d] for d in depths})
        arm["tokens_npz"] = npz
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
