"""Depth x horizon: does a deeper pass make a plan that reaches FURTHER?

`tul.spandec_per_pass` (commit 397bdf3) asks the loop for a target that grows by one span
a pass — pass t is graded on spans s+1 .. s+min(t, `spandec_pass_horizon_max`). Its own
training readout `tul/spandec_pass_t{t}` cannot answer whether the DEPTH bought the
horizon, because it changes two things at once: pass 1 is graded on one span and pass 6 on
six, so a falling series is consistent with "the deeper state plans further" AND with "six
spans are easier to average than one".

This holds the target FIXED and moves the depth:

    column pass_h1   the H = 1 target at `spandec_pass_tokens` tokens, `pos_pass` table
    column pass_h6   the H = 6 target (six spans, `spandec_pass_tokens` tokens each),
                     the SAME `pos_pass` table
    column exit      the shipped exit target: H = 1 at `spandec_max_tokens` tokens
                     through the `pos` table — the `tul/spandec_ce` a training run logs

and reads each column at forced depths 1, 2, 3, 6. Within a column every depth sees the
SAME target tokens, the SAME position table and the SAME eligibility mask, so the
difference between two rows of a column is the depth and nothing else. K1-K6 and K3-K6 per
column are what "a deeper pass plans further" would have to move; if `pass_h6` shows a
K-curve and `pass_h1` does not, depth is buying HORIZON rather than accuracy — the one
reading `spandec_pass_t{t}` cannot give.

THE STATE SCORED is the loop's exit state at the forced depth, read at exactly the seam
`_tul_spandec_loss` reads (`_readout(h_slots)`, after `_tul_cond_apply` where an arm has
one, before the gate budget and before `prefix_project`). Scoring goes through the model's
own `SpanDecoder.decode` and its own `fused_linear_cross_entropy`, with the same
`mux_detach_head` asymmetry the shipped loss uses (output head per the knob, decoder input
embeddings always detached).

IT ALSO RUNS ON A NON-PER-PASS CHECKPOINT. A model with no `pos_pass` table has no
defined position geometry for the per-pass blocks, so columns `pass_h*` are SKIPPED (not
faked with `pos`) and only `exit` is reported — which is what makes the strict ruler
(`slot-spandec-strict`) a usable control for a per-pass arm.

WHAT IT CANNOT SAY. It is an EVAL-time depth intervention on a trained model: a column
that is flat says this checkpoint's exit state does not vary usefully with depth, NOT that
a model trained at another depth would be flat (2026-09-12: "training depth moves CE, eval
depth does not" — pair it with a depth-1-TRAINED control). It grades `z` through the span
DECODER, which is a training-only reader that is never in the deployed forward, so a
moving column is not by itself a token-CE result: read it beside `core_depth_sweep.py`'s
`ce_tokens`. It says nothing about the row's open tail, about span 0, or about slots whose
downstream spans are incomplete — those are masked out of every column, identically.

Usage:
  python lab/divergence/spandec_horizon_grid.py \
      --ckpt perpass=tul_slot_spandec_strict_perpass=/path/step_5000.pt \
      --rows 48 --batch 3 --depths 1,2,3,6 --pass-horizons 1,6 \
      --out .../horizon_grid.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg  # noqa: E402
from _rows import pack_rows, stream_from_loader  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")


def exit_state(model, inp, layout, depth: int) -> torch.Tensor:
    """The loop's exit state at a FORCED depth, at the seam `_tul_spandec_loss` reads.

    `_tul_front` -> `_tul_core(slot_depths=depth)` -> `_tul_cond_apply` (only where the
    arm has a think-once conditioning stack, which `_forward_tul` applies BEFORE the
    span-decoder term). Everything the shipped forward does AFTER that seam — the gate's
    budget, `detach_z`, the eval plan ablation, `prefix_project` — is downstream of the
    decoder and is deliberately not run here.
    """
    tab = torch.full(layout.slot_index.shape, int(depth), dtype=torch.long,
                     device=inp.device)
    x, x0, bigram = model._tul_front(inp, layout)
    out = model._tul_core(x, x0, bigram, layout, input_ids=inp, slot_depths=tab)
    h_slots, depths = out[1], out[2]
    if not torch.equal(depths[layout.slot_valid],
                       torch.full_like(depths[layout.slot_valid], int(depth))):
        raise RuntimeError(f"the loop did not run every valid slot at depth {depth}")
    if getattr(model, "tul_cond", None) is not None:
        h_slots = model._tul_cond_apply(h_slots)
    return h_slots


def column_targets(model, inp, layout, kind: str, horizon: int):
    """`(ids, valid, pos_table, J)` for one column. Depth-independent BY CONSTRUCTION."""
    from morph.model.tul_spandec import horizon_span_slots

    dec = model.tul_spandec
    tc = model.cfg.tul
    if kind == "exit":
        ids, valid = horizon_span_slots(inp, layout, dec.per_span_tokens, 1)
        return ids, valid, dec.pos, dec.per_span_tokens
    j1 = int(tc.spandec_pass_tokens)
    ids, valid = horizon_span_slots(inp, layout, j1, horizon)
    return ids, valid, dec.pos_pass, j1 * horizon


def choose_columns(model, pass_horizons: list[int]
                   ) -> tuple[list[tuple[str, str, int]], dict[str, str]]:
    """`([(name, kind, H)], {skipped column: why})` for this checkpoint.

    The exit column always runs. The per-pass columns run only where the checkpoint has a
    `pos_pass` table, because the per-pass block geometry is a DIFFERENT meaning of the
    same row index — row 8 is "span s+2, token 0" there and "span s+1, token 8" in `pos`
    (`SpanDecoder.__init__`). Reading the per-pass target through `pos` would be a
    number, and it would be a number about a table that was never trained for it, so the
    column is skipped WITH A REASON in the JSON instead.
    """
    dec, tc = model.tul_spandec, model.cfg.tul
    cols: list[tuple[str, str, int]] = []
    skipped: dict[str, str] = {}
    if dec.pos_pass is None:
        skipped["pass_h*"] = ("no pos_pass table (tul.spandec_per_pass false): the "
                              "per-pass block geometry is undefined on this checkpoint, "
                              "so the columns are SKIPPED rather than faked with `pos`")
    else:
        cap = int(tc.spandec_pass_horizon_max)
        for h in pass_horizons:
            if h < 1 or h > cap:
                skipped[f"pass_h{h}"] = (f"H={h} outside [1, spandec_pass_horizon_max="
                                         f"{cap}]")
            elif h * int(tc.spandec_pass_tokens) > int(dec.pos_pass.shape[0]):
                skipped[f"pass_h{h}"] = "past the pos_pass table's rows"
            else:
                cols.append((f"pass_h{h}", "pass", h))
    cols.append(("exit", "exit", 1))
    return cols, skipped


@torch.no_grad()
def row_ce(model, z, ids, valid, table) -> tuple[np.ndarray, np.ndarray]:
    """Per-ROW (sum of CE, number of scored tokens) through the model's OWN readers.

    One `fused_linear_cross_entropy` call per row rather than one for the batch: the
    kernel returns a mean, and a per-row mean times its count is the sum the bootstrap
    resamples. Same total arithmetic (each call sees 1/B of the rows), and the batch
    aggregate reproduces the single-call value to float-summation order — which is what
    `tests/test_spandec_horizon_grid.py` pins against the model's own `spandec_ce`.
    """
    from morph.model.fused_ce import fused_linear_cross_entropy

    dec = model.tul_spandec
    tc = model.cfg.tul
    w_tied = model.embed.lm_weight()
    w_head = w_tied.detach() if tc.mux_detach_head else w_tied
    st = dec.decode(z, ids, valid, w_tied.detach(), pos=table)      # [B, S, J, C]
    C = st.shape[-1]
    lab = torch.where(valid, ids, torch.full_like(ids, -100))
    sums, cnts = [], []
    for b in range(st.shape[0]):
        n = float(valid[b].sum())
        if n == 0.0:
            sums.append(0.0)
            cnts.append(0.0)
            continue
        ce = fused_linear_cross_entropy(
            st[b].reshape(-1, C), w_head, lab[b].reshape(-1), ignore_index=-100,
            chunk_size=model.cfg.ce_chunk_size, mask_token_id=tc.slot_id)
        sums.append(float(ce) * n)
        cnts.append(n)
    return np.asarray(sums), np.asarray(cnts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=OVR1,OVR2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--depths", default="1,2,3,6")
    ap.add_argument("--pass-horizons", default="1,6",
                    help="the H values of the per-pass columns; each is capped at "
                         "tul.spandec_pass_horizon_max. Ignored on a checkpoint with no "
                         "pos_pass table (those report the exit column alone).")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    depths = [int(x) for x in a.depths.split(",")]

    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or bool(tul_rt.model_cfg.tokens_through_core):
        raise SystemExit("spandec_horizon_grid needs a SLOT-LOOP model "
                         "(tul.tokens_through_core false)")
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            a.device, tul_rt.model_cfg)
    model.eval()
    if model.tul_spandec is None:
        raise SystemExit("this checkpoint has no span decoder (tul.spandec false): there "
                         "is no reader to grade z through.")
    dec, tc = model.tul_spandec, model.cfg.tul
    max_d = int(tc.slot_max_depth or model.cfg.max_depth)
    for d in depths:
        if not (1 <= d <= max_d):
            raise SystemExit(f"depth {d} is outside [1, slot_max_depth={max_d}] — the "
                             "model has never run it")

    cols, skipped = choose_columns(model, [int(x) for x in a.pass_horizons.split(",")])

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]

    sums: dict[tuple[str, int], list[float]] = {}
    cnts: dict[str, list[float]] = {}
    n_rows = 0
    for bi, (inp, labels, layout, _idx) in enumerate(batches):
        inp = inp.to(a.device)
        layout = layout.to(a.device)
        n_rows += inp.shape[0]
        tgt = {name: column_targets(model, inp, layout, kind, h)
               for name, kind, h in cols}
        for name, _kind, _h in cols:
            cnts.setdefault(name, []).extend(
                tgt[name][1].flatten(1).sum(dim=1).double().cpu().numpy().tolist())
        for d in depths:
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=a.device == "cuda"):
                h_slots = exit_state(model, inp, layout, d)
            z = model._readout(h_slots)
            for name, kind, h in cols:
                # THE MASK IDENTITY, rebuilt at every depth rather than assumed. The
                # targets are depth-independent by construction today; a column whose
                # population moved with the depth would make its K-curve partly a change
                # of what is being scored, and that must fail loudly, not silently.
                ids, valid, table, _J = column_targets(model, inp, layout, kind, h)
                if not (torch.equal(ids, tgt[name][0])
                        and torch.equal(valid, tgt[name][1])):
                    raise RuntimeError(
                        f"column {name!r}: the target at depth {d} differs from the one "
                        "built before the depth loop — the eligibility mask is not "
                        "depth-independent and the column is unreadable")
                s, _c = row_ce(model, z, ids, valid, table)
                sums.setdefault((name, d), []).extend(s.tolist())
            del h_slots, z
        print(f"  batch {bi + 1}/{len(batches)}: {inp.shape[0]} rows", flush=True)

    res: dict = {}
    for name, _kind, _h in cols:
        n = np.asarray(cnts[name])
        col: dict = {"n_tokens": float(n.sum()), "depths": {}, "ci": {}}
        for d in depths:
            s = np.asarray(sums[(name, d)])
            col["depths"][str(d)] = float(s.sum() / max(n.sum(), 1.0))
        for lo, hi in ((1, 6), (3, 6)):
            if lo in depths and hi in depths:
                col["ci"][f"K{lo}-K{hi}"] = paired_bootstrap_ci(
                    np.asarray(sums[(name, lo)]), np.asarray(sums[(name, hi)]), n,
                    n_boot=a.boot)
        res[name] = col

    rec = {
        "label": label, "config": config, "step": step, "rows": n_rows,
        "batch": a.batch, "depths": depths, "columns": [c[0] for c in cols],
        "skipped_columns": skipped,
        "spandec": {"per_span_tokens": int(dec.per_span_tokens),
                    "horizon": int(dec.horizon),
                    "pass_tokens": int(tc.spandec_pass_tokens),
                    "pass_horizon_max": int(tc.spandec_pass_horizon_max),
                    "per_pass": bool(tc.spandec_per_pass),
                    "pos_pass_rows": (0 if dec.pos_pass is None
                                      else int(dec.pos_pass.shape[0]))},
        "notes": {
            "state": "the loop's EXIT state at the forced depth, read at the seam "
                     "_tul_spandec_loss reads (_readout(h_slots), after _tul_cond_apply)",
            "exit_column": "H=1 at spandec_max_tokens through `pos` — the tul/spandec_ce "
                           "a training run logs",
            "pass_columns": "H spans of spandec_pass_tokens each through `pos_pass`",
            "mask": "horizon_span_slots' rule, depth-independent by construction and "
                    "CHECKED per batch: a slot is scored in a column only when every "
                    "span that column needs exists and is complete",
            "K": "K{a}-K{b} = CE at depth a minus CE at depth b; POSITIVE means the "
                 "deeper state is better. Paired bootstrap over rows.",
            "eval_only": "an eval-time depth intervention; pair a moving column with a "
                         "depth-1-TRAINED control before reading it as 'depth earns'",
        },
        "results": res,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=1)

    print(f"\n{label} step {step} — {n_rows} rows")
    for k, v in skipped.items():
        print(f"  SKIPPED {k}: {v}")
    hdr = "".join(f"d{d}".rjust(10) for d in depths)
    print("  " + "column".ljust(12) + hdr + "K1-K6".rjust(12) + "K3-K6".rjust(12))
    for name, _kind, _h in cols:
        col = res[name]
        cells = "".join(f"{col['depths'][str(d)]:10.4f}" for d in depths)
        k16 = col["ci"].get("K1-K6", {}).get("point", float("nan"))
        k36 = col["ci"].get("K3-K6", {}).get("point", float("nan"))
        print(f"  {name:12s}{cells}{k16:+12.4f}{k36:+12.4f}")
    for name, _kind, _h in cols:
        for k, v in res[name]["ci"].items():
            print(f"    {name:12s} {k}: {v['point']:+.4f} "
                  f"[{v['lo']:+.4f}, {v['hi']:+.4f}] over {v['n_units']} rows")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
