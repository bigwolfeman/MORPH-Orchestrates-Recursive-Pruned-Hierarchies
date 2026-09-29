"""Offline probe, Part 1 of the LXTUL latent-WTA build
(.agents/notes/proposed/architecture/2026-09-29-latent-wta.md): does a CHEAP LATENT-SPACE
score pick the same WTA winner `_tul_fan_all` picks with M no-grad coda passes, and at
what REGRET (nats/token) if it disagrees?

WHY THIS EXISTS. `_tul_fan_all` (morph/model/transformer.py) spends M no-grad
`_back_region` passes per step just to find, for every (rollout, slot), which of the M
register cells the coda would score best -- the expensive part of LX-Fan+WTA's step
(lab/theory/efficient-latent-exploration-lit-2026-09-29.md, shortlist item 1). This probe
answers, on an ALREADY-TRAINED checkpoint and with NO new training: if we replace those
M passes with a cheap distance in latent space, evaluated against a target built from the
TRUE next span, how often does it agree with the coda's own pick, and how much worse (in
nats/token) is the span when it disagrees?

THREE CANDIDATE TARGETS, all "free" (already computed in the SAME forward, no new
network, no coda read):

  (a) nearest cell (cosine AND L2) to the NEXT slot's own pooled PRE-LOOP cell -- slot
      s+1's seed (`_tul_core`'s trajectory index 0, the state the register's trigger
      built before any pass), mean-pooled over its M cells. Slot s+1's seed is built from
      span s+1's (== the span slot s is scored on) LAST token
      (`E_slot + W_sent . embed(t_last)`), so it is informative about that span's content
      and it already exists in the SAME forward -- no extra compute. Undefined for the
      LAST slot of a row (no slot s+1); those rows are excluded from candidate (a) only.
  (b) nearest cell to the mean-pool of the PRELUDE states (pre-loop, `xn`/`base` at TOKEN
      positions) of span s+1's OWN tokens -- the direct "read the true span" baseline.
  (c) an InfoNCE-style pick: for cell (s, m), score = cos(cell, target_b[s]) / T -
      logsumexp_{s' in batch} cos(cell, target_b[s']) / T (in-batch negatives = every
      OTHER valid slot's candidate-(b) target in the probe's row set); argmax over m.
      This is NOT the same ranking as plain cosine to the own target: a cell that reads
      close to MANY spans (a generic / collapsing cell) is penalised relative to one that
      reads close to its own span alone, which is the point of using InfoNCE instead of a
      raw distance (JEPA-paradox / centroid-collapse guard named in the lit report).

TARGETS ARE NEVER WRITTEN ANYWHERE. This is a read-only probe: nothing computed here
re-enters the model. The forward runs exactly the checkpoint's own training step (model
in train() mode, real dropout, a fixed seed) via TWO read-only capture hooks:

  * `model._fan_all_ce_capture` (already built into `_tul_fan_all` for exactly this,
    docs at morph/model/transformer.py::_tul_fan_all) -- the coda's own per-(rollout,
    slot, cell) span CE table (`ce`, `[B, S, K]`) and the exact tensors it was built
    from (`cells_d`, the detached M cells; `base_d`, the detached prelude carrier at
    every position; `gid`/`keep_tok`/`lab`/`g_bins`, the span<-token index).
  * `morph.model.transformer.fan_repel_term` is monkey-patched (the `fan_stream_probe.py`
    precedent: wrap a module-level NAME transformer.py calls, not a method) to also stash
    its first three args: `db_traj` (the loop's per-pass trajectory list, `db_traj[0]` is
    the SEED), `valid` (`layout.slot_valid`, rollout-expanded), `m` (fan_k). This
    function is unconditionally called once per forward whenever the model has a
    register (`_m > 1`), regardless of `fan_repel_lambda`, so it is a reliable hook.

Usage (a probe host, never the trainer's GPU; wrap in `flock` per the machine rules):

    python lab/divergence/latent_wta_probe.py \\
        --ckpt b5k=tul_slot_spandec_strict_lxfan4_wta_fp01=/path/step_5000.pt \\
        --ckpt ev01_3k=tul_slot_spandec_strict_lxfan4_wta_fp01_ev01=/path/step_3000.pt \\
        --rows 96 --batch 6 --seed 0 --temp 0.1 \\
        --out /home/wolfe/morph-scratch/latwta/probe_<tag>.json
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


def regret_and_agreement(ce: torch.Tensor, ok: torch.Tensor, pick: torch.Tensor,
                         n_tok: torch.Tensor) -> dict:
    """``ce`` ``[B,S,K]`` SUMMED span CE per cell (``accumulate_span_ce[_fused]``'s
    contract), ``ok``/``pick`` ``[B,S]`` (``pick`` the LATENT candidate's chosen cell
    index), ``n_tok`` ``[B,S]`` token counts of the scored span.

    Returns agreement with the coda's own argmin and the SUMS a token-weighted regret
    needs (``regret_sum`` nats, ``tok_sum`` tokens), so batches pool by tokens, not by
    slots. ``ce`` is already a per-span SUM, so regret per token is
    ``sum(picked - best) / sum(n_tok)``: multiplying by ``n_tok`` again (the pre-2026-09-29
    version) counted every span's regret ``n_tok`` times and inflated the reading by
    about one span length. ``random_regret_sum`` is the expected regret of a uniformly
    random cell (``mean_k ce - best``), the chance floor the latent pick must beat.
    """
    if not bool(ok.any()):
        return {"n": 0, "agree_sum": 0.0, "regret_sum": 0.0, "random_regret_sum": 0.0,
                "tok_sum": 0.0}
    best, arg = ce.min(dim=-1)
    picked_ce = ce.gather(-1, pick.unsqueeze(-1)).squeeze(-1)
    return {"n": int(ok.sum()),
            "agree_sum": float((pick[ok] == arg[ok]).float().sum()),
            "regret_sum": float((picked_ce - best)[ok].sum()),
            "random_regret_sum": float((ce.mean(dim=-1) - best)[ok].sum()),
            "tok_sum": float(n_tok[ok].sum())}


def candidate_a_pick(cells: torch.Tensor, seed_pool: torch.Tensor, ok: torch.Tensor
                     ) -> dict:
    """(a): nearest cell to slot s+1's pooled pre-loop seed. ``cells`` ``[B,S,M,C]``,
    ``seed_pool`` ``[B,S,C]`` (pre-loop, EVERY slot's own pooled seed -- the caller
    shifts). Drops the last slot column (no s+1). Returns cosine and L2 picks + masks.
    """
    B, S, M, C = cells.shape
    cells_a = cells[:, :-1]                      # [B, S-1, M, C]
    target = seed_pool[:, 1:]                     # [B, S-1, C]  (slot s+1's seed)
    ok_a = ok[:, :-1]
    cos = F.cosine_similarity(cells_a, target.unsqueeze(2), dim=-1)      # [B, S-1, M]
    pick_cos = cos.argmax(dim=-1)
    l2 = (cells_a - target.unsqueeze(2)).norm(dim=-1)
    pick_l2 = l2.argmin(dim=-1)
    return {"pick_cos": pick_cos, "pick_l2": pick_l2, "ok": ok_a, "ce_slice": slice(0, S - 1)}


def candidate_b_pool(base: torch.Tensor, gid: torch.Tensor, keep_tok: torch.Tensor,
                     g_bins: int) -> torch.Tensor:
    """Mean-pool ``base`` (prelude carrier -- ``[B,L,C]`` plain, ``[B,L,n,C]`` on a
    Hyper-Connection model, ``n`` the residual stream count, reduced by its mean first)
    per span bin -> ``[B, S, C]``, the SAME ``gid``/``keep_tok``/``g_bins`` the coda's
    own CE table used, so bin ``s`` here is exactly the span `ce[:, s, :]` scores. Bin 0
    (before the first slot) is dropped, matching `ce`'s own convention.

    THIS IS `morph.model.transformer.latent_wta_target`, called directly (not a second
    copy): the shipped `tul.fan_all_wta_winner="latent"` mode scores its picks against
    this exact function, so the probe that chose it and the mode that runs it can never
    drift apart. Kept as a thin re-export (rather than a bare import at every call site)
    so this module's public candidate-(a)/(b)/(c) API stays one flat set of functions.
    """
    from morph.model.transformer import latent_wta_target
    return latent_wta_target(base, gid, keep_tok, g_bins)


def candidate_b_pick(cells: torch.Tensor, target: torch.Tensor) -> dict:
    """(b): nearest cell to the true next span's own prelude mean-pool."""
    cos = F.cosine_similarity(cells, target.unsqueeze(2), dim=-1)
    pick_cos = cos.argmax(dim=-1)
    l2 = (cells - target.unsqueeze(2)).norm(dim=-1)
    pick_l2 = l2.argmin(dim=-1)
    return {"pick_cos": pick_cos, "pick_l2": pick_l2}


def candidate_c_pick(cells: torch.Tensor, target_b: torch.Tensor, ok: torch.Tensor,
                     temp: float) -> torch.Tensor:
    """(c): InfoNCE argmax over M, in-batch negatives = every OTHER valid (row, slot) of
    ``target_b``.

    THIS IS `morph.model.transformer.latent_wta_infonce_score`, called directly (not a
    second copy) and reduced to an argmax over M -- this candidate is what
    `tul.fan_all_wta_winner="latent"` ships with (chosen 2026-09-29 for the lowest
    REGRET of the three candidates on both probed checkpoints), so the probe that
    measured it and the mode that runs it can never drift apart. This function used to
    be its own per-row Python loop (correct, O(N) forward passes over a dict lookup);
    kept as a thin wrapper now for the SAME reason `candidate_b_pool` is a thin wrapper
    over `latent_wta_target` -- one flat candidate-(a)/(b)/(c) API for this module's
    callers, one implementation for the score itself.
    """
    from morph.model.transformer import latent_wta_infonce_score
    score = latent_wta_infonce_score(cells, target_b, ok, temp)        # [B, S, M]
    return score.argmax(dim=-1)


def run_one(model, ce_capture: list, traj_capture: dict, tr_mod, seed: int, temp: float
           ) -> dict:
    from morph.model.tul_fan import _cell_readout

    entry = ce_capture[-1]
    ce = entry["ce"]                                               # [B, S, K] fp32
    cells = entry["cells_d"]                                       # [B, S, M, C] (or +carrier)
    while cells.dim() > 4:
        cells = _cell_readout(cells)
    base = entry["base_d"]
    gid, keep_tok, g_bins = entry["gid"], entry["keep_tok"], entry["g_bins"]
    n_tok = tr_mod.span_token_counts(gid, keep_tok, g_bins)[:, 1:]
    layout = entry["layout"]
    ok = layout.slot_valid & (n_tok > 0)

    traj = traj_capture["traj"]
    m = traj_capture["m"]
    B_r, S = ok.shape
    traj0 = traj[0].reshape(B_r, S, m, *traj[0].shape[2:])
    seed_state = _cell_readout(traj0)
    while seed_state.dim() > 4:
        seed_state = _cell_readout(seed_state)
    seed_pool = seed_state.mean(dim=2)                              # [B, S, C]

    target_b = candidate_b_pool(base, gid, keep_tok, g_bins)        # [B, S, C]

    out = {}
    a = candidate_a_pick(cells, seed_pool, ok)
    for kind in ("cos", "l2"):
        r = regret_and_agreement(ce[:, a["ce_slice"]], a["ok"], a[f"pick_{kind}"],
                                 n_tok[:, a["ce_slice"]])
        out[f"a_{kind}"] = r

    b = candidate_b_pick(cells, target_b)
    for kind in ("cos", "l2"):
        r = regret_and_agreement(ce, ok, b[f"pick_{kind}"], n_tok)
        out[f"b_{kind}"] = r

    c_pick = candidate_c_pick(cells, target_b, ok, temp)
    out["c_infonce"] = regret_and_agreement(ce, ok, c_pick, n_tok)
    out["n_ok_slots"] = int(ok.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=0.1, help="candidate (c) temperature")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    import morph.model.transformer as tr_mod

    results = {}
    for spec in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(spec)
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        if tul_rt is None or int(getattr(tul_rt.model_cfg, "fan_k", 0)) < 2:
            raise SystemExit(f"{label}: not a fan arm (tul.fan_k < 2)")
        if float(getattr(tul_rt.model_cfg, "fan_all_wta_lambda", 0.0)) == 0.0:
            raise SystemExit(f"{label}: fan_all_wta_lambda == 0, _tul_fan_all builds no "
                             f"table -- nothing to probe")
        print(f"[{label}] loading checkpoint {path} ...", flush=True)
        t0 = time.time()
        model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
        print(f"[{label}] checkpoint loaded in {time.time() - t0:.1f}s (step {step})",
             flush=True)
        model.train()          # the task: "train mode" -- real dropout, the real fwd path

        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
        stream = stream_from_loader(loader, a.rows * row_tokens)
        n_batches = -(-a.rows // a.batch)
        batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
        print(f"[{label}] {len(batches)} batch(es) packed, starting forward passes ...",
             flush=True)

        real_repel = tr_mod.fan_repel_term
        traj_capture: dict = {}

        def patched_repel(traj, valid, m, n_passes, stats=None, _rr=real_repel,
                          _tc=traj_capture):
            _tc["traj"], _tc["valid"], _tc["m"] = traj, valid, m
            return _rr(traj, valid, m, n_passes, stats=stats)

        tr_mod.fan_repel_term = patched_repel
        per_batch = []
        try:
            for bi, (inp, labels, layout, _idx) in enumerate(batches):
                tb0 = time.time()
                torch.manual_seed(a.seed + bi)
                torch.cuda.manual_seed_all(a.seed + bi)
                model._fan_all_ce_capture = []
                traj_capture.clear()
                with torch.no_grad(), torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=a.device == "cuda"):
                    model(inp.to(a.device), labels.to(a.device),
                         slot_layout=layout.to(a.device))
                if not model._fan_all_ce_capture or "traj" not in traj_capture:
                    raise RuntimeError(
                        f"{label} batch {bi}: no fan/traj capture -- the training branch "
                        f"of _tul_fan_all did not run (check model.training / labels / "
                        f"plan_mode)")
                per_batch.append(run_one(model, model._fan_all_ce_capture, traj_capture,
                                         tr_mod, a.seed + bi, a.temp))
                print(f"[{label}] batch {bi + 1}/{len(batches)} done in "
                     f"{time.time() - tb0:.1f}s", flush=True)
        finally:
            tr_mod.fan_repel_term = real_repel
            model._fan_all_ce_capture = None

        agg = {}
        for key in per_batch[0]:
            if key == "n_ok_slots":
                agg[key] = int(sum(p[key] for p in per_batch))
                continue
            n = sum(p[key]["n"] for p in per_batch)
            tok = sum(p[key]["tok_sum"] for p in per_batch)
            if n == 0 or tok == 0:
                agg[key] = {"n": 0, "agree": None, "regret_nats_per_tok": None,
                            "random_regret_nats_per_tok": None}
                continue
            agg[key] = {
                "n": n, "tokens": tok,
                "agree": sum(p[key]["agree_sum"] for p in per_batch) / n,
                "regret_nats_per_tok": sum(p[key]["regret_sum"] for p in per_batch) / tok,
                "random_regret_nats_per_tok":
                    sum(p[key]["random_regret_sum"] for p in per_batch) / tok}

        results[label] = {"config": config, "ckpt": path, "step": step,
                          "rows": a.rows, "batch": a.batch, "n_batches": len(batches),
                          "candidates": agg}
        print(f"== {label} (step {step}) ==  chance agree = {1.0 / int(tul_rt.model_cfg.fan_k):.3f}")
        for k, v in agg.items():
            if k == "n_ok_slots":
                continue
            print(f"  {k:12s} n={v['n']:6d} agree={v['agree']}"
                 f" regret={v['regret_nats_per_tok']}"
                 f" random_pick_regret={v['random_regret_nats_per_tok']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
