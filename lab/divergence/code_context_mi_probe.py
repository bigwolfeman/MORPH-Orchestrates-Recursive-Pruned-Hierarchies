"""LCM's MI, built on the model's OWN reach knob: does a decoded span use its context?

LCM's instrument (arXiv 2412.08821 §2.4.1; the reading note §2 and §8 bullet 2):

    MI = (1/|s_hat|) ( log p_GPT2(s_hat) − log p_GPT2(s_hat | previous 10 sentences) )

with the sign written so a POSITIVE value means the generated sentence gets cheaper once
the scorer sees the context. Base-LCM, the pure MSE regression, posts 0.062 / −0.105 /
0.071 / −0.184 on ROC / C4 / Wikipedia-en / Gutenberg — at or below zero — while the
diffusion LCMs post 0.7 to 1.3. A conditional mean is a sentence that could follow
anything, and MI is the instrument that says so.

WHICH FORM THIS FILE BUILDS. The model-internal one, not the GPT-2 one. Under the strict
geometry a coda token reads its own span's tokens plus the earlier slots' PREFIX CELLS
and nothing else, so the context of span ``s+1`` IS cells ``0..s`` and it can be cut by
ZEROING those cells — no retraining, no outside LM, and the scorer is the model whose
span is being scored:

    mi_ctx = CE(span | slot s's cell alone) − CE(span | cells 0..s)          per token

Cell ``s`` itself is held FIXED between the two passes (it is the code the span was
written from, not context), so the difference isolates the earlier cells. A third pass
reports the ``shuffle`` control beside the ``zero`` one, because a zeroed cell is off
distribution and "cells present at all" is not the same claim as "these cells":

    mi_shuf = CE(span | cells 0..s PERMUTED across the row) − CE(span | cells 0..s)

Both are reported for TWO spans: the DECODED one (the model's own greedy continuation
from its sampled code) and the row's TRUE span, which is the reference. A decoded span
that is a generic conditional mean reads ``mi_ctx`` near zero while the true span reads
positive.

ONE PASS PER SCORED SLOT, and why. Span ``s+1`` sees cells ``0..s``, so a pass that keeps
only cell ``s`` isolates exactly one slot; keeping a set of cells isolates only the
smallest of them. The probe therefore scores ``--mi-slots`` slots per row, taken evenly
spaced through the row's eligible slots, one forward per (slot ordinal, context mode,
span source). The scored span is written into a copy of the TRUE row on its own, so no
reading here leans on spans being invisible to each other.

ABSOLUTE CE FIRST. Every CE in the table is printed beside ``ln(V)``, the uniform
baseline. Three instruments once agreed confidently and contradicted each other because
the reader was 1.84 nats worse than uniform (memory: check-ce-against-uniform-first).

Usage:

    python lab/divergence/code_context_mi_probe.py \
        --ckpt tul-code-20k=tul_code=/path/step_20000.pt \
        --rows 48 --batch 3 --mi-slots 6 --out code_context_mi_tul-code-20k.json

The pure functions (``context_mi``, ``permute_valid``, ``pick_ordinals``) are tested in
``tests/test_lcm_instruments.py``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fan_mixture_probe import block_bootstrap  # noqa: E402

__all__ = ["context_mi", "permute_valid", "pick_ordinals", "block_bootstrap"]


# ────────────────────────── pure readings ──────────────────────────

def context_mi(nll_cut: np.ndarray, nll_full: np.ndarray,
               n_tok: np.ndarray) -> dict[str, np.ndarray]:
    """LCM's MI from two SUMMED span NLLs and the span's token count.

    ``mi`` is ``(nll_cut − nll_full) / n_tok``: positive when the context makes the span
    cheaper. ``ce_cut`` and ``ce_full`` are the same two numbers per token, so a caller
    can print the level beside the difference — an MI of +0.05 means something different
    at CE 4.0 than at CE 11.0.
    """
    cut = np.asarray(nll_cut, dtype=np.float64)
    full = np.asarray(nll_full, dtype=np.float64)
    n = np.asarray(n_tok, dtype=np.float64)
    if cut.shape != full.shape or cut.shape != n.shape:
        raise ValueError("nll_cut, nll_full and n_tok must have the same shape")
    if np.any(n <= 0):
        raise ValueError("every scored span needs at least one token")
    return {"mi": (cut - full) / n, "ce_cut": cut / n, "ce_full": full / n}


def permute_valid(z: torch.Tensor, valid: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Shuffle a row's cells across its slots: ``out[b, s] = z[b, perm[b, s]]``.

    ``z`` ``[B, S, ...]``, ``valid`` ``[B, S]``, ``perm`` ``[B, S]`` a per-row
    permutation of ``0..S-1``. Slots that are not valid keep exactly what they held (a
    pad slot's cells are zero and must stay zero — a permutation that moved a real code
    into a pad would change which slots exist, not which codes they hold).
    """
    if perm.shape != valid.shape or z.shape[:2] != valid.shape:
        raise ValueError(f"z {tuple(z.shape)}, valid {tuple(valid.shape)} and perm "
                         f"{tuple(perm.shape)} must agree on [B, S]")
    B, S = valid.shape
    idx = perm.clamp(0, S - 1)
    out = z[torch.arange(B, device=z.device).view(B, 1), idx]
    keep = valid.view(B, S, *([1] * (z.dim() - 2)))
    return torch.where(keep, out, z)


def pick_ordinals(elig: torch.Tensor, n: int) -> torch.Tensor:
    """``[B, n]`` slot indices, evenly spaced through each row's eligible slots, −1 padded.

    Evenly spaced rather than the first ``n``: a row's first spans sit at the top of the
    document where there is no context to use, and reading MI only there would report the
    absence of context as the absence of context USE.
    """
    B, S = elig.shape
    out = torch.full((B, int(n)), -1, dtype=torch.long, device=elig.device)
    for b in range(B):
        ids = elig[b].nonzero().flatten()
        k = int(ids.numel())
        if k == 0:
            continue
        take = min(int(n), k)
        pos = torch.linspace(0, k - 1, take, device=elig.device).round().long()
        out[b, :take] = ids[pos]
    return out


# ────────────────────────── the run ──────────────────────────

def _rowsum(vals: np.ndarray, row_ix: np.ndarray, n_rows: int) -> np.ndarray:
    out = np.zeros(n_rows, dtype=np.float64)
    np.add.at(out, row_ix, vals.astype(np.float64))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=ovr1,ovr2]")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--mi-slots", type=int, default=6, help="slots scored per row")
    ap.add_argument("--steps", type=int, default=0, help="sampler k; 0 = tul.code_infer_steps")
    ap.add_argument("--max-span-tokens", type=int, default=32)
    ap.add_argument("--code-seed", type=int, default=0)
    ap.add_argument("--shuffle-seed", type=int, default=7)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed; it does NOT draw rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    from _build import ROOT, build_cfg, parse_ckpt_spec
    from _rows import pack_rows, stream_from_loader
    from _span_decode import (PinnedCells, check_span_layout, greedy_decode, row_logits,
                              span_nll, span_slots)
    sys.path.insert(0, f"{ROOT}/scripts")
    from tul_samples import load_ckpt  # noqa: E402
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime
    import morph

    label, config, path, ovr = parse_ckpt_spec(a.ckpt)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None or not bool(getattr(tul_rt.model_cfg, "code", False)):
        raise SystemExit(f"{label}: not an LCTUL arm (tul.code is false)")
    if str(getattr(tul_rt.model_cfg, "tg_geometry", "")) != "strict":
        raise SystemExit(f"{label}: the cell-zero reach cut needs tul.tg_geometry=strict")
    model, step = load_ckpt(cfg, path, a.device, tul_rt.model_cfg)
    model.eval()
    k_main = int(a.steps or model.cfg.tul.code_infer_steps)
    J = int(a.max_span_tokens)
    ln_v = math.log(float(model.embed.lm_weight().shape[0]))

    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    n_batches = -(-a.rows // a.batch)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[:n_batches]

    MODES = ("full", "own", "shuffle")
    SRC = ("decoded", "true")
    recs: list[dict] = []
    n_rows_seen = 0

    with PinnedCells(model, "code") as pin:
        for bi, (inp, labels, layout, _idx) in enumerate(batches):
            ids = inp.to(a.device)
            lay = layout.to(a.device)
            elig, lens, first, src0 = span_slots(lay, J)
            check_span_layout(lay, elig, first, src0)
            if not bool(elig.any()):
                n_rows_seen += ids.shape[0]
                continue
            B, S = elig.shape
            ac = torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.device == "cuda")
            with ac:
                pin.pin = None
                row_logits(model, ids, lay, "encoder")
                z_true = pin.last.float().clone()
                model(ids, slot_layout=lay, code_mode="sampled", code_steps=k_main,
                      code_seed=a.code_seed)
                z_hat = pin.last.float().clone()
                pin.pin = z_hat
                ids_dec, _cand = greedy_decode(model, ids, lay, elig, lens, first, src0,
                                               J, code_mode="encoder")
            g = torch.Generator(device="cpu").manual_seed(a.shuffle_seed + bi)
            perm = torch.stack([torch.randperm(S, generator=g) for _ in range(B)]).to(ids.device)
            z_shuf = permute_valid(z_true, lay.slot_valid, perm)
            sel = pick_ordinals(elig, a.mi_slots)                       # [B, P]
            rb = torch.arange(B, device=ids.device)

            for p in range(sel.shape[1]):
                s_p = sel[:, p]
                live_rows = s_p >= 0
                if not bool(live_rows.any()):
                    break
                s_c = s_p.clamp_min(0)
                live = torch.zeros_like(elig)
                live[rb[live_rows], s_c[live_rows]] = True
                for src_name in SRC:
                    ids_s = ids.clone()
                    if src_name == "decoded":
                        r_i, s_i = live.nonzero(as_tuple=True)
                        for j in range(J):
                            m = lens[r_i, s_i] > j
                            if not bool(m.any()):
                                break
                            rr, ss = r_i[m], s_i[m]
                            ids_s[rr, first[rr, ss] + j] = ids_dec[rr, first[rr, ss] + j]
                    # the code slot s was written from stays FIXED across the three modes
                    z_used = z_hat if src_name == "decoded" else z_true
                    own_cell = z_used[rb, s_c]                            # [B, M, C]
                    for mode in MODES:
                        if mode == "full":
                            cells = z_true.clone()
                        elif mode == "own":
                            cells = torch.zeros_like(z_true)
                        else:
                            cells = z_shuf.clone()
                        cells[rb, s_c] = own_cell
                        with ac:
                            pin.pin = cells
                            nll = span_nll(model, ids_s, lay, live, lens, first, src0,
                                           J, code_mode="encoder")
                        r_i, s_i = live.nonzero(as_tuple=True)
                        for r_, s_ in zip(r_i.tolist(), s_i.tolist()):
                            recs.append({"row": n_rows_seen + r_, "slot": s_,
                                         "src": src_name, "mode": mode,
                                         "nll": float(nll[r_, s_]),
                                         "ntok": float(lens[r_, s_])})
            pin.pin = None
            n_rows_seen += ids.shape[0]
            print(f"  batch {bi + 1}/{len(batches)} done ({time.time() - t0:.0f}s)",
                  flush=True)

    # ── assemble: one record per (row, slot, src, mode) ──────────────────────
    key = {}
    for r in recs:
        key[(r["row"], r["slot"], r["src"], r["mode"])] = r
    units = sorted({(r["row"], r["slot"], r["src"]) for r in recs})
    res = {"arm": label, "config": config, "ckpt": path, "step": step, "rows": a.rows,
           "batch": a.batch, "mi_slots": a.mi_slots, "k": k_main,
           "form": "model-internal (cells zeroed / shuffled); no outside LM",
           "ln_vocab": ln_v, "morph_module": os.path.dirname(morph.__file__),
           "sources": {}}
    for src_name in SRC:
        us = [u for u in units if u[2] == src_name]
        if not us:
            continue
        rows_a = np.array([u[0] for u in us])
        ntok = np.array([key[(u[0], u[1], src_name, "full")]["ntok"] for u in us])
        full = np.array([key[(u[0], u[1], src_name, "full")]["nll"] for u in us])
        own = np.array([key[(u[0], u[1], src_name, "own")]["nll"] for u in us])
        shuf = np.array([key[(u[0], u[1], src_name, "shuffle")]["nll"] for u in us])
        uniq, ix = np.unique(rows_a, return_inverse=True)
        n_rows = int(uniq.shape[0])
        row_sum = {"ce_full": _rowsum(full, ix, n_rows),
                   "ce_own": _rowsum(own, ix, n_rows),
                   "ce_shuffle": _rowsum(shuf, ix, n_rows)}
        row_tok = _rowsum(ntok, ix, n_rows)
        boot = block_bootstrap(row_sum, row_tok,
                               [("ce_own", "ce_full"), ("ce_shuffle", "ce_full")],
                               n_boot=a.boot, seed=a.seed)
        per = context_mi(own, full, ntok)
        res["sources"][src_name] = {
            "n_spans": len(us), "n_tokens": float(ntok.sum()), "n_rows": n_rows,
            "bootstrap": boot,
            "mi_ctx": boot["ce_own-ce_full"], "mi_shuf": boot["ce_shuffle-ce_full"],
            "mi_median_per_span": float(np.median(per["mi"]))}
    res["wall_clock_s"] = time.time() - t0

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    txt = os.path.splitext(a.out)[0] + ".txt"
    L = [f"{label} step {step}  rows={a.rows} batch={a.batch} mi_slots={a.mi_slots} k={k_main}",
         f"form: {res['form']}", f"morph from {res['morph_module']}",
         f"ln(V) = {ln_v:.4f}  (every CE below is nats/token against this)", "",
         "span source   spans  tokens    CE|full    CE|own    CE|shuffle   "
         "mi_ctx = CE|own - CE|full        mi_shuf"]
    for src_name in SRC:
        s = res["sources"].get(src_name)
        if s is None:
            continue
        b = s["bootstrap"]
        L.append(
            f"{src_name:<13} {s['n_spans']:>5} {s['n_tokens']:>7.0f}  "
            f"{b['ce_full']['point']:>9.4f} {b['ce_own']['point']:>9.4f} "
            f"{b['ce_shuffle']['point']:>11.4f}   "
            f"{b['ce_own-ce_full']['point']:+.4f} "
            f"[{b['ce_own-ce_full']['lo']:+.4f}, {b['ce_own-ce_full']['hi']:+.4f}]  "
            f"{b['ce_shuffle-ce_full']['point']:+.4f} "
            f"[{b['ce_shuffle-ce_full']['lo']:+.4f}, {b['ce_shuffle-ce_full']['hi']:+.4f}]")
    L += ["", "95 % row-block bootstrap, every column:"]
    for src_name in SRC:
        s = res["sources"].get(src_name)
        if s is None:
            continue
        L.append(f"  {src_name}")
        for k in ("ce_full", "ce_own", "ce_shuffle", "ce_own-ce_full", "ce_shuffle-ce_full"):
            v = s["bootstrap"][k]
            L.append(f"    {k:<20} {v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]")
    L += ["", f"wall clock {res['wall_clock_s']:.0f} s"]
    with open(txt, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {a.out} and {txt}")


if __name__ == "__main__":
    main()
