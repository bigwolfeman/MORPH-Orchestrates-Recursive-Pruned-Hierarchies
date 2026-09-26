"""Stratified plan-worth profile — the decision-grade replacement for the all-token
worth_shuffle scalar (2026-08-29, "tea leaves" verdict).

Why: `val/plan_worth_shuffle` is a paired counterfactual (good) averaged over ALL
~1044 token positions per row (bad). The mechanism it hunts concentrates on the
tokens right after a slot; span-first tokens are ~5% of positions, so a real
0.4-nat effect there reads as ~0.02 in the mean — exactly the measured noise floor
(l1 vs l1-rep at step 1000: 0.012 vs 0.051; adjacent-eval bounce ~0.02).

What this does instead, offline on a saved checkpoint:
  1. per-token PAIRED CE deltas (ablated - intact) on the same packed val rows;
  2. stratified by the token's offset within its span (bag_id cumcount) — a real
     write effect must DECAY with offset (the plan is a prefix the coda refines);
     noise is flat and cannot fake the shape;
  3. bootstrap CI over rows, so an arm's profile carries its own error bars;
  4. all three ablations (zero / shuffle / wrong_seed) → dose-response ordering.

Tokens in span 0 (no preceding slot: nothing to ablate FOR them) and dump-bin
tokens are excluded. Slot positions are never scored (GL arms train them at
weight 0 — the emit_source lesson).

Usage:
  python lab/divergence/worth_profile.py \
    --ckpt l3=tul_l3=checkpoints/morph/tul-l3/step_4500.pt --rows 96 \
    --out /home/wolfe/morph-scratch/tulfm/worth_profile.json
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

from _build import ROOT, build_cfg, parse_ckpt_spec

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402  (handles the _orig_mod. compile prefix)

from _earning import BINS, bin_of  # noqa: E402  (ONE home for the offset bins)
from _rows import pack_rows, stream_from_loader  # noqa: E402  (the sweep's own packer)

MODES = ("zero", "shuffle", "wrong_seed")
# THE ROUTE SPLIT (2026-09-11). `zero` ablates ONE route — the prefix write — and the
# cross-span budget priced that at 0.093 nats of a 0.399-nat budget. `all_slots` also
# zeroes the coda's per-layer injections at the slot cells and cuts the cells' own
# attention over their span, so the slot channel carries nothing at all
# (`transformer._tul_all_slots_coda`). It is DEFINED only on a tg_restrict arm at scope
# "all", so `--modes auto` adds it where it is legal and prints why it did not elsewhere.
ALL_SLOTS = "all_slots"


def _all_slots_supported(model) -> str | None:
    """None if the arm supports ``all_slots``; otherwise the reason it does not."""
    tc = getattr(model.cfg, "tul", None)
    if tc is None:
        return "no TUL config"
    if not tc.tg_restrict:
        return "tul.tg_restrict is false (a coda slot cell attends every earlier position)"
    if tc.tg_restrict_scope != "all":
        return f"tul.tg_restrict_scope={tc.tg_restrict_scope!r}, not 'all'"
    if tc.tokens_through_core:
        return "the paid loop has no separate slot write to ablate"
    if not (tc.coda_sees_slots and tc.coda_token_cut == 0):
        return "the coda runs on a gathered subset of positions"
    if tc.bcast:
        return "tul.bcast adds z to the next span's TOKEN inputs, a route this does not cut"
    return None


def token_strata(layout, labels_row, b: int, spec) -> list[tuple[int, int]]:
    """[(position, bin)] for scoreable token positions of row b: real token, label
    present, in a span with a PRECEDING slot (bag_id in 1..n_valid-ish, not dump bin)."""
    L = layout.slot_mask.shape[1]
    out = []
    counts: dict[int, int] = {}
    dump = int(layout.slot_valid.shape[1])  # max_slots = the dump bin id
    for p in range(L):
        if bool(layout.slot_mask[b, p]):
            continue
        bag = int(layout.bag_id[b, p])
        off = counts.get(bag, 0)
        counts[bag] = off + 1
        if bag == 0 or bag >= dump:
            continue  # no preceding plan / dump bin
        if int(labels_row[p]) < 0:
            continue
        out.append((p, bin_of(off)))
    return out


@torch.no_grad()
def _cell_mode(mode: str) -> int | None:
    """``cell<j>`` -> j; ``cellall`` -> -1; any other mode -> None."""
    if not mode.startswith("cell"):
        return None
    return -1 if mode == "cellall" else int(mode[4:])


def _forward(model, inp, layout, mode: str) -> dict:
    """``tul_forward_ablated`` under ``mode``. THE PER-CELL SPLIT (2026-09-26, plan C):
    ``cell<j>`` zeroes prefix cell j of every slot AFTER ``TULSlots.prefix_project`` (the
    projected value, ``E_pass`` included) and leaves the other cells intact; ``cellall``
    zeroes every cell there. ``W_prefix`` has no bias, so with ``E_pass`` absent
    ``cellall`` equals ``zero`` (the source state zeroed before the projection): that
    equality is the self-check a hand run reads off the two TOTAL lines."""
    j = _cell_mode(mode)
    if j is None:
        return model.tul_forward_ablated(inp, None, layout, plan_mode=mode)
    tul = model.tul
    K = int(tul.tul.prefix_k)
    if j >= K:
        raise ValueError(f"{mode}: the model has prefix_k={K} cells")
    orig = tul.prefix_project

    def _cut(*args, **kwargs):
        values, pos = orig(*args, **kwargs)
        values = values.clone()
        if j < 0:
            values.zero_()
        else:
            values[:, j::K] = 0                        # slot-major: index s*K + k
        return values, pos

    tul.prefix_project = _cut
    try:
        return model.tul_forward_ablated(inp, None, layout, plan_mode="normal")
    finally:
        del tul.prefix_project                         # back to the bound method


def per_token_ce(model, inp, layout, labels, device, mode: str) -> torch.Tensor:
    """[B, L] CE at token positions (nan elsewhere), plan ablated per `mode`."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = _forward(model, inp.to(device), layout, mode)
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    return ce


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True,
                    help="LABEL=CONFIG=PATH[=OVR1,OVR2] (extra Hydra overrides, comma-split)")
    ap.add_argument("--rows", type=int, default=96)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--modes", default="auto",
                    help="comma-separated ablations, or 'auto' = zero,shuffle,wrong_seed "
                         "plus all_slots wherever the arm supports it. cell<j> zeroes prefix "
                         "cell j only, cellall every cell (both after the projection)")
    ap.add_argument("--paired-rows", action="store_true",
                    help="pack the val rows with lab/divergence/_rows.py::pack_rows — the "
                         "SAME packer and the same stream core_depth_sweep.py uses — and "
                         "record the stream index of every scored token, so the worth "
                         "profile and the depth sweep sit on the same tokens.")
    ap.add_argument("--verify-tok-index", default=None,
                    help="a core_depth_sweep .tokens.npz; assert this profile's tok_index "
                         "is a PREFIX of the sweep's (it is shorter when --rows is). "
                         "Implies --paired-rows.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    if a.verify_tok_index:
        a.paired_rows = True

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path, ovr = parse_ckpt_spec(triple)
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path,
                                device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        # Same val stream, same seed for every arm -> profiles are comparable.
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        torch.manual_seed(a.seed)
        # WHICH ablations. `auto` = today's three plus the route split where it is legal;
        # an explicit list is taken as given and RAISES inside the forward if the arm does
        # not support a mode, which is what a hand-run comparison wants.
        if a.modes == "auto":
            modes = list(MODES)
            if getattr(model, "tul_code_enc", None) is not None:
                # A TUL-Code model has no seed->cells path: the seed feeds the velocity
                # field, the cells are the encoder's code (spec §8). The forward refuses
                # wrong_seed there; zero/shuffle act on the cells and are the profile.
                modes.remove("wrong_seed")
                print(f"  {label}: wrong_seed SKIPPED — TUL-Code model (cells come from "
                      "the encoder, not the seed)", flush=True)
            why = _all_slots_supported(model)
            if why is None:
                modes.append(ALL_SLOTS)
                print(f"  {label}: all_slots ON (the route split)", flush=True)
            else:
                print(f"  {label}: all_slots SKIPPED — {why}", flush=True)
        else:
            modes = [m.strip() for m in a.modes.split(",") if m.strip()]
        # per ROW: sums[mode][bin], counts[bin]  (bootstrap unit = row)
        row_sums = {m: [] for m in modes}
        row_counts = []
        tok_index: list[np.ndarray] = []
        rows_done = 0
        if a.paired_rows:
            # THE SWEEP'S OWN ROWS. `pack_rows` packs the same stream with the same
            # `pack_tul_batch`, checks that the packed token positions reproduce the
            # stream in order, and carries the stream index of every token position — so
            # the worth profile, the depth sweep and the budget profile can be read on the
            # same tokens instead of on three row sets that merely look alike.
            row_tokens = spec.l_total + 1
            stream = stream_from_loader(loader, a.rows * row_tokens)
            n_batches = -(-a.rows // a.batch)
            batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]
        else:
            batches = None
            buf: list[int] = []
            need = a.batch * (spec.l_total + 1)
        bi_iter = 0
        while rows_done < a.rows:
            if batches is not None:
                if bi_iter >= len(batches):
                    break
                inp, labels, layout, idx = batches[bi_iter]
                bi_iter += 1
            else:
                while len(buf) < need:
                    ids = next(loader)[0]
                    buf.extend(ids.reshape(-1).tolist())
                inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
                idx = None
            layout = layout.to(device)
            ce = {"normal": per_token_ce(model, inp, layout, labels, device, "normal")}
            for m in modes:
                ce[m] = per_token_ce(model, inp, layout, labels, device, m)
            for b in range(inp.shape[0]):
                strata = token_strata(layout, labels[b], b, spec)
                cnt = np.zeros(len(BINS))
                sums = {m: np.zeros(len(BINS)) for m in modes}
                for p, bi in strata:
                    cnt[bi] += 1
                    for m in modes:
                        sums[m][bi] += float(ce[m][b, p] - ce["normal"][b, p])
                row_counts.append(cnt)
                for m in modes:
                    row_sums[m].append(sums[m])
            if idx is not None:
                # the SWEEP's scoreable set: every token position, in row order. Kept
                # whole (not filtered to the strata) so it is the same array the sweep
                # writes and the two can be compared element for element.
                tokpos = (~layout.slot_mask).cpu()
                tok_index.append(idx[tokpos].numpy().astype(np.int32))
            rows_done += inp.shape[0]
            print(f"  {label}: {rows_done}/{a.rows} rows", flush=True)
        cnts = np.stack(row_counts)                      # [R, bins]
        rng = np.random.default_rng(a.seed)
        arm = {"step": step, "rows": rows_done, "bins": [list(x) for x in BINS],
               "n_tokens_per_bin": cnts.sum(0).tolist(), "paired_rows": bool(a.paired_rows),
               "modes": {}}
        if tok_index:
            ti = np.concatenate(tok_index)
            arm["tok_index_n"] = int(ti.shape[0])
            arm["tok_index_first"] = int(ti[0])
            arm["tok_index_last"] = int(ti[-1])
            npz = a.out.rsplit(".", 1)[0] + f".{label}.rows.npz"
            np.savez_compressed(npz, tok_index=ti)
            arm["tok_index_npz"] = npz
            if a.verify_tok_index:
                ref = np.load(a.verify_tok_index)["tok_index"]
                n = int(ti.shape[0])
                if n > ref.shape[0] or not np.array_equal(ref[:n], ti):
                    raise SystemExit(
                        f"tok_index MISMATCH against {a.verify_tok_index}: this profile "
                        f"scored {n} token positions, the sweep {ref.shape[0]}; "
                        f"first differing index "
                        f"{int(np.argmax(ref[:n] != ti)) if n <= ref.shape[0] else 'n/a'}. "
                        "The two readouts are NOT on the same rows.")
                print(f"  {label}: tok_index verified against "
                      f"{a.verify_tok_index} ({n} of {ref.shape[0]} positions)", flush=True)
        for m in modes:
            sums = np.stack(row_sums[m])                 # [R, bins]
            mean = sums.sum(0) / np.maximum(cnts.sum(0), 1)
            idx = rng.integers(0, len(sums), size=(a.boot, len(sums)))
            bmeans = sums[idx].sum(1) / np.maximum(cnts[idx].sum(1), 1)  # [boot, bins]
            lo, hi = np.percentile(bmeans, [2.5, 97.5], axis=0)
            arm["modes"][m] = {"mean": mean.tolist(), "ci_lo": lo.tolist(),
                               "ci_hi": hi.tolist()}
            cells = "  ".join(f"{mu:+.3f}[{l:+.3f},{h:+.3f}]"
                              for mu, l, h in zip(mean, lo, hi))
            print(f"{label:10s} {m:10s} {cells}", flush=True)
            # the token-weighted total, the one number the budget profile is compared with
            tot = float(sums.sum() / max(cnts.sum(), 1))
            arm["modes"][m]["total"] = tot
            print(f"{label:10s} {m:10s} TOTAL {tot:+.4f}", flush=True)
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
