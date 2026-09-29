"""WTA winner agreement across the K = ``tul.code_enum_k`` rollouts (2026-09-29,
"onewinner" Part 1): does a slot's WINNING fan stream (arm b's per-rollout table,
``tests/test_tul_lx_credit.py``) agree across rollouts, and does the rollout the exact
mixture posterior favours (the MAP rollout) predict that winner any better than the
others? This decides whether ``tul.fan_all_wta_winner="map"``
(``morph/model/transformer.py::_tul_fan_all``) throws away real per-rollout signal or
merely a redundant copy of it.

WHAT IS MEASURED, on a TRAINED checkpoint, in ``model.train()`` mode (the WTA table is
built ONLY at train — ``if not self.training: return state, w, None`` in
``_tul_fan_all``), at a FIXED seed for the dropout / depth draw:

  * the per-(rollout, slot) WTA table itself (``_fan_all_ce_capture``'s ``ce``
    ``[R*B0, S, M]``; ``choice = ce.argmin(-1)`` is EXACTLY arm (b)'s own per-rollout
    winner, `tests/test_tul_lx_credit.py`'s own reading of it);
  * the EXACT per-span mixture posterior over the R rollouts (``softmax_k S_k(span)``,
    ``S`` from ``morph.model.tul_gram.iw_span_bound``, spied where
    ``_enum_mix_losses`` calls it — the SAME group indexing ``span_ce_index`` uses, so
    ``S``'s slot axis lines up with ``ce``'s 1:1, checked at runtime).

Per SCORED, VALID slot:
  (a) fraction where ALL R rollouts pick the same winning stream;
  (b) mean pairwise agreement over rollout PAIRS (R choose 2), the softer version of (a);
  (c) agreement between the MAP rollout's (highest posterior weight) winner and each
      OTHER rollout's winner — reported per rollout-offset AND pooled;
  (d) chance level 1/M (M = fan_k, always 4 on this arm) for a naive comparison.

Usage:
  python lab/divergence/wta_winner_agreement.py \\
    --ckpt lxtul-lxfan4-wta-fp01=tul_slot_spandec_strict_lxfan4_wta_fp01=\\
checkpoints/morph/lxtul-lxfan4-wta-fp01/step_5000.pt \\
    --rows 96 --batch 6 --seed 0 \\
    --out /home/wolfe/morph-scratch/onewinner/agreement_lxtul-lxfan4-wta-fp01.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys

import torch

from _build import ROOT, build_cfg, parse_ckpt_spec

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402  (handles the _orig_mod. compile prefix)

from _rows import pack_rows, stream_from_loader  # noqa: E402


def _agreement(model, inp, labels, layout, device: str, seed: int) -> dict | None:
    """One train forward at a fixed seed; returns the per-batch agreement counts, or
    None if the batch has no valid scored slot at all (skipped)."""
    import morph.model.tul_gram as gram_mod

    real_iwb = gram_mod.iw_span_bound
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _spy(lp, w, grp, n_groups):
        bound_sum, S, scored = real_iwb(lp, w, grp, n_groups)
        captured.append((S.detach().clone(), scored.detach().clone()))
        return bound_sum, S, scored

    gram_mod.iw_span_bound = _spy
    # `_enum_mix_losses` reads the module-level name it imported at definition time
    # (`from .tul_gram import iw_span_bound` style, or a direct module attribute read —
    # patched on `tul_gram` itself so either binding sees it); `transformer.py`'s own
    # copy of the name, if it imported one directly, is patched too.
    import morph.model.transformer as tmod
    had_direct = hasattr(tmod, "iw_span_bound")
    real_direct = tmod.iw_span_bound if had_direct else None
    if had_direct:
        tmod.iw_span_bound = _spy
    model._fan_all_ce_capture = []
    try:
        model.train()
        torch.manual_seed(seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=device == "cuda"):
            model(inp.to(device), labels=labels.to(device), slot_layout=layout)
    finally:
        gram_mod.iw_span_bound = real_iwb
        if had_direct:
            tmod.iw_span_bound = real_direct
    cap_list = model._fan_all_ce_capture
    model._fan_all_ce_capture = None
    if not cap_list:
        return None
    cap = cap_list[0]
    ce = cap["ce"]                                              # [R*B0, S, M]
    if not captured:
        raise RuntimeError("iw_span_bound never ran: is tul.code_enum_k > 1 on this arm?")
    S_mix, scored = captured[-1]                                 # the LAST call this
    # forward made — `_enum_mix_losses` is called once per forward (want_groups False
    # at train), so with the code active there is exactly one call and "last" is "only".
    R = int(model.cfg.tul.code_enum_k)
    B0 = layout.slot_valid.shape[0]
    max_slots = layout.slot_valid.shape[1]
    n_groups = S_mix.shape[1]
    if n_groups != B0 * (max_slots + 1):
        raise RuntimeError(f"group count {n_groups} != B0*(max_slots+1) "
                           f"({B0}*{max_slots + 1}); the offline group indexing "
                           "assumption (span_ce_index's own) does not hold here")
    S_slots = S_mix.view(R, B0, max_slots + 1)[:, :, 1:]           # [R, B0, S]
    scored_slots = scored.view(B0, max_slots + 1)[:, 1:]           # [B0, S] bool

    n_tok_ok = ce.sum(-1).view(R, B0, -1)[0] != 0.0                # WTA's own "scored" read
    ok = layout.slot_valid & n_tok_ok & scored_slots
    if not bool(ok.any()):
        return None

    choice = ce.argmin(-1).view(R, B0, -1)                          # [R, B0, S]
    M = ce.shape[-1]

    all_same = (choice == choice[0:1]).all(dim=0)                   # [B0, S]
    n_valid = int(ok.sum())
    n_all_agree = int((all_same & ok).sum())

    pair_agree_sum = 0
    pair_n = 0
    for r1, r2 in itertools.combinations(range(R), 2):
        same = choice[r1] == choice[r2]
        pair_agree_sum += int((same & ok).sum())
        pair_n += n_valid

    map_r = S_slots.argmax(dim=0)                                    # [B0, S]
    b0_idx = torch.arange(B0, device=choice.device).view(B0, 1).expand(B0, choice.shape[-1])
    s_idx = torch.arange(choice.shape[-1], device=choice.device).view(
        1, choice.shape[-1]).expand(B0, choice.shape[-1])
    map_choice = choice[map_r, b0_idx, s_idx]                        # [B0, S]
    map_agree_sum, map_agree_n = 0, 0
    map_agree_per_offset: dict[int, list[int]] = {}
    for r in range(R):
        is_map_here = map_r == r
        for r2 in range(R):
            if r2 == r:
                continue
            same = (map_choice == choice[r2]) & is_map_here & ok
            off = (r2 - r) % R
            cnt_same = int(same.sum())
            cnt_n = int((is_map_here & ok).sum())
            map_agree_sum += cnt_same
            map_agree_n += cnt_n
            e = map_agree_per_offset.setdefault(off, [0, 0])
            e[0] += cnt_same
            e[1] += cnt_n

    return {
        "n_valid_slots": n_valid,
        "n_all_agree": n_all_agree,
        "pair_agree_sum": pair_agree_sum, "pair_n": pair_n,
        "map_agree_sum": map_agree_sum, "map_agree_n": map_agree_n,
        "map_agree_per_offset": {str(k): v for k, v in map_agree_per_offset.items()},
        "R": R, "M": M,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise RuntimeError(f"{label}: not a TUL config")
    if int(tul_rt.model_cfg.code_enum_k) < 2:
        raise RuntimeError(f"{label}: tul.code_enum_k={tul_rt.model_cfg.code_enum_k} < 2; "
                           "nothing to measure agreement across")
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, plain=False)
    rows_done = 0
    agg = {"n_valid_slots": 0, "n_all_agree": 0, "pair_agree_sum": 0, "pair_n": 0,
          "map_agree_sum": 0, "map_agree_n": 0, "map_agree_per_offset": {}}
    n_batches_used = 0
    for inp, labels, layout, _idx in batches:
        if rows_done >= a.rows:
            break
        layout = layout.to(device)
        res = _agreement(model, inp, labels, layout, device, a.seed)
        rows_done += inp.shape[0]
        if res is None:
            continue
        n_batches_used += 1
        for k in ("n_valid_slots", "n_all_agree", "pair_agree_sum", "pair_n",
                  "map_agree_sum", "map_agree_n"):
            agg[k] += res[k]
        for off, (num, den) in res["map_agree_per_offset"].items():
            e = agg["map_agree_per_offset"].setdefault(off, [0, 0])
            e[0] += num
            e[1] += den
        agg.setdefault("R", res["R"])
        agg.setdefault("M", res["M"])

    def _rate(num, den):
        return (num / den) if den else None

    out = {
        "label": label, "config": config, "ckpt": path, "step": step,
        "rows_requested": a.rows, "rows_seen": rows_done, "batches_used": n_batches_used,
        "batch": a.batch, "seed": a.seed,
        "R": agg.get("R"), "M": agg.get("M"),
        "n_valid_slots": agg["n_valid_slots"],
        "frac_all_rollouts_agree": _rate(agg["n_all_agree"], agg["n_valid_slots"]),
        "mean_pairwise_agreement": _rate(agg["pair_agree_sum"], agg["pair_n"]),
        "map_rollout_agreement": _rate(agg["map_agree_sum"], agg["map_agree_n"]),
        "map_agreement_by_rollout_offset": {
            k: _rate(v[0], v[1]) for k, v in agg["map_agree_per_offset"].items()},
        "chance_level": (1.0 / agg["M"]) if agg.get("M") else None,
    }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
