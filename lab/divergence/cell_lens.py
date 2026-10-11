"""Cell lens: what does each slot-loop cell contain, per pass? Offline, frozen checkpoint.

Two readings of a fan / LXTUL checkpoint (2026-10-10, survey recommendations 1 and 2 of
/home/wolfe/morph-scratch/abq/lens/lens-survey.md):

``--part data``   CPU only. Delegates to `credit_matrix.part_data`: packs TRAIN batches
                  (the trainer's stream, fast-forwarded ``--train-skip`` batches, so a step-5000
                  checkpoint never saw them) and VAL batches (the trainer's val stream, skip
                  50,000 docs; memory owt-val-overlaps-training) into ONE file that the other
                  parts read. Same file format as credit_matrix, so a file from there works.

``--part patch``  CAUSAL PATCH THROUGH THE FROZEN CODA. No training. For each forced loop depth
                  d and each cell i, the coda's next-span CE when ONLY cell i is written to
                  it. "Cell i alone" is NOT re-implemented here. It is the model's own val
                  oracle (`MORPHTransformer._tul_fan_oracle`, whose per-stream write is
                  `_tul_fan_stream_write`), tapped:
                    * ungraded fan (`fan_mix: all`, no selection): the other M-1 prefix
                      cells are BLANK (zeros through `prefix_project(cells=)`);
                    * latent-selected loop (LXTUL, `fan_loop_select`): the router's write
                      with the winner FORCED to i (`_fan_route_cells`, losers exactly 0).
                  The tap records the oracle's [B, S, M] span-CE table. Baselines on the
                  same rows and the same tokens: ALL cells as trained (the shipped write),
                  NO cells (`plan_mode: zero`; the coda's output is tapped and scored with
                  the oracle's own accumulator), and the per-span BEST single cell.
                  Units are the oracle's: summed coda CE of bin s+1 (tokens 2..n of span s+1
                  plus the first token of span s+2) over the model's token count, nats per
                  token, tied head, NO pointer head. They equal the val `fan/stream_ce_k{i}`,
                  `fan/oracle_ce` and `fan/mixed_ce`, and the script refuses to continue if
                  its table disagrees with those keys on any batch.

``--part probe``  FUTURE-LENS STYLE TRAINED READOUT (Pal et al. 2311.04897, the linear
                  variant). The model is frozen and only makes features. For each valid slot
                  s, cell i and pass t (t = 0 the loop's entry state, t >= 1 the cell state
                  AFTER pass t, before the latent-selected loop's reset) the feature is the
                  cell's readout `_cell_readout` (the vector the latent head, teacher and
                  router read). Target: tokens 1..N of span s+1 (the span decoder's own
                  target). Per (pass, offset) ONE probe shared by all cells:
                  `logits = (RMSNorm(x) W) E^T`, W = I + dW (C x C, dW = 0 at init), E the
                  model's tied embedding table (detached, frozen). Trained on TRAIN rows,
                  scored on VAL rows against: unigram (train frequency), shuffled cell (the
                  same probe on the same cell index of another slot, mostly another row) and
                  the all-cells MEAN as input. Also prints readable examples.

COPIED vs NOVEL (patch and probe, 2026-10-11). A model with a pointer / copy head can predict a
token that already occurred in the row without the cell. A target token is "copied" when the
SAME token id occurs at a non-slot position of the row's token stream BEFORE the start of the
next span (`copied_tokens`; for slot s that start is `slot_index[s] + prefix_k`), else "novel".
The probe reports every metric split by it; the patch splits every per-cell CE the same way
(the split re-uses the oracle's own head, row by row, and the script raises unless
copied + novel reproduces the oracle's table).

``--part special``  CPU. Reads the patch JSON (``--patch-json``; its per-span table: batch, row,
                  slot, winning cell, per-cell CE, all-cells CE) and the batches file, adds span
                  properties (next-span length, copy ratio, first-token class, all-cells CE =
                  difficulty), and reports per property the contingency table (winning cell x
                  bin), NMI(winner; property) and a 200-permutation null (mean, 95th percentile).

``--part spandec`` Free-running span-decoder readout per cell (models with ``tul.spandec``).
                  Greedy decode, no teacher forcing, from {cell i alone, mean of cells, the
                  decoder's own trained input, RMS-matched noise, a cell of another slot}.

``--part geom``   Geometry of the cell readouts per pass: linear CKA between cells and between
                  passes, effective rank, within-slot cosine, with shuffled and Gaussian
                  controls. Two ``--ckpt`` also give the CKA between the checkpoints' cells.

Usage (repo root, the MORPH venv; GPU parts under `flock /home/wolfe/morph-scratch/gpu.lock`):
  python lab/divergence/cell_lens.py --part data --config nowta_pointer_fast \
      --batches-file ignore/lens/batches.pt --train-batches 64 --val-batches 40
  python lab/divergence/cell_lens.py --part patch \
      --ckpt fan=nowta_pointer_fast=/home/wolfe/morph-scratch/abq/tree-e/checkpoints/morph/ab-nowta-ptr/step_5000.pt \
      --batches-file ignore/lens/batches.pt --depths 1,2,3,6 --out patch_fan.json
  python lab/divergence/cell_lens.py --part probe \
      --ckpt lx=lxtul_pointer_fast_spdwin=/home/wolfe/morph-scratch/spdwin/tree/checkpoints/morph/ab-spdwin-r2/step_5000.pt \
      --batches-file ignore/lens/batches.pt --depths 1,2,3,6 --offsets 4 --out probe_lx.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import unicodedata
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _build import ROOT, build_cfg, parse_ckpt_spec  # noqa: E402
from _next_span import next_span_positions  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

sys.path.insert(0, f"{ROOT}/scripts")

import morph.model.transformer as T  # noqa: E402
from morph.model.tul_fan import _cell_readout  # noqa: E402
from morph.model.tul_spandec import next_span_slots  # noqa: E402

IGNORE = -100


# ── shared helpers ───────────────────────────────────────────────────────────────────────
def autocast_ctx(device: str):
    return torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu",
                          dtype=torch.bfloat16, enabled=str(device).startswith("cuda"))


def forced_depths(layout, d: int, m_cells: int) -> torch.Tensor:
    """The eval-only depth table, every slot at depth ``d``. A register / fan model takes it
    on the CELL axis, `[B, max_slots * M]` (constant within a slot,
    `_slot_depth_override`); the model validates the range (1..slot_max_depth) and RAISES
    outside it."""
    b, s = layout.slot_index.shape
    return torch.full((b, s * int(m_cells)), int(d), dtype=torch.long,
                      device=layout.slot_index.device)


def check_depths(model, depths: list[int]) -> None:
    tc = model.cfg.tul
    max_d = int(tc.slot_max_depth or model.cfg.max_depth)
    if min(depths) < 1 or max(depths) > max_d:
        raise ValueError(f"--depths {depths} outside the trained range [1, {max_d}]")
    if getattr(tc, "tokens_through_core", False) or getattr(tc, "loop_reads_tokens", False):
        raise ValueError("cell_lens reads the SLOT loop; this model runs the paid loop")
    if int(tc.slot_cells) < 2:
        raise ValueError("cell_lens needs a register / fan model (tul.slot_cells >= 2)")


def span_table(model, xh: torch.Tensor, labels: torch.Tensor, layout) -> torch.Tensor:
    """`[B, S]` summed coda CE of bin s+1 from a coda output `xh`: the val oracle's own
    accumulator, head padding and slicing (`_tul_fan_oracle`)."""
    gid, keep_tok, lab, g_bins = T.span_ce_index(labels, layout)
    w = model.embed.lm_weight()
    return T.accumulate_span_ce(xh, T.pad_vocab_align8(w), gid, keep_tok, lab, g_bins,
                                vocab_size=int(w.shape[0]))[:, 1:].float()


def span_tokens(labels: torch.Tensor, layout) -> torch.Tensor:
    """`[B, S]` scored-token count of bin s+1: the oracle's `n_tok`."""
    gid, keep_tok, _lab, g_bins = T.span_ce_index(labels, layout)
    return T.span_token_counts(gid, keep_tok, g_bins)[:, 1:]


# ── copied vs novel ──────────────────────────────────────────────────────────────────────
def copied_tokens(x: torch.Tensor, slot_mask: torch.Tensor, tok: torch.Tensor,
                  start: torch.Tensor) -> torch.Tensor:
    """THE copied rule, one home. ``x [B, L]`` the packed input ids, ``slot_mask [B, L]`` True at
    slot positions, ``tok [B, N]`` target token ids (``< 0`` = no token), ``start [B, N]`` the row
    position where each target's next span starts. Returns ``[B, N]`` bool: True when ``tok``
    occurs at a NON-slot position strictly before ``start`` (slot / prefix positions hold
    ``slot_id`` and cells, not text). Within-span repeats do not count: the rule is "earlier in the
    row, before the next span starts"."""
    b, ln = x.shape
    v = max(int(x.max()), int(tok.max())) + 1
    pos = torch.arange(ln, device=x.device).expand(b, ln)
    stream = ~slot_mask
    first = torch.full((b, v), ln, dtype=torch.long, device=x.device)
    first.scatter_reduce_(1, torch.where(stream, x, torch.zeros_like(x)),
                          torch.where(stream, pos, torch.full_like(pos, ln)), "amin",
                          include_self=True)
    seen = first.gather(1, tok.clamp(min=0))
    return (tok >= 0) & (seen < start)


def copied_scored(x: torch.Tensor, labels: torch.Tensor, layout) -> torch.Tensor:
    """``[B, L]`` copied flag of the LABEL at every position, against the start of the span of
    the bin the position is scored in (bin g belongs to slot g - 1: `span_ce_index`). Only
    meaningful where `span_ce_index`'s ``keep_tok`` holds."""
    s = int(layout.slot_valid.shape[1])
    slot = (layout.bag_id.clamp(0, s) - 1).clamp(min=0)
    start = layout.slot_index.gather(1, slot) + int(layout.prefix_k)
    return copied_tokens(x, layout.slot_mask, labels, start)


def copied_next_span(x: torch.Tensor, layout, tok: torch.Tensor) -> torch.Tensor:
    """``tok [B, S, J]`` next-span token ids (``IGNORE`` where absent) -> ``[B, S, J]`` copied flags.
    The next span of slot s starts at ``slot_index[s] + prefix_k``."""
    b, s, j = tok.shape
    start = (layout.slot_index + int(layout.prefix_k)).unsqueeze(-1).expand(b, s, j)
    return copied_tokens(x, layout.slot_mask, tok.reshape(b, -1),
                         start.reshape(b, -1)).reshape(b, s, j)


def split_span_ce(xh, w_head, gid, keep_tok, lab, n_groups, copied, vocab_size=None):
    """`accumulate_span_ce`, one head pass per row, summed twice: ``(copied, novel)`` each
    `[B, n_groups - 1]` (bin 0 dropped like the oracle's ``[:, 1:]``). Their sum is the oracle's
    table; `check_split` raises if it is not."""
    b = lab.shape[0]
    cp = torch.zeros(b * n_groups, device=lab.device, dtype=torch.float32)
    nv = torch.zeros_like(cp)
    for i in range(b):
        logits = (xh[i].to(w_head.dtype) @ w_head.t()).float()
        if vocab_size is not None:
            logits = logits[:, :vocab_size]
        ce = F.cross_entropy(logits, lab[i], reduction="none") * keep_tok[i]
        cp.index_add_(0, gid[i], ce * copied[i])
        nv.index_add_(0, gid[i], ce * (~copied[i]))
    return cp.view(b, n_groups)[:, 1:], nv.view(b, n_groups)[:, 1:]


def count_split(keep_tok, gid, copied, n_groups):
    """``(n_copied, n_novel)`` scored tokens per bin, `[B, n_groups - 1]`."""
    b = keep_tok.shape[0]
    out = []
    for flag in (copied, ~copied):
        c = torch.zeros(b * n_groups, device=gid.device, dtype=torch.float32)
        for i in range(b):
            c.index_add_(0, gid[i], (keep_tok[i] & flag[i]).float())
        out.append(c.view(b, n_groups)[:, 1:])
    return out[0], out[1]


def span_table_split(model, xh, x, labels, layout):
    """`span_table` split into ``(copied, novel)`` `[B, S]` tables from a coda output ``xh``."""
    gid, keep_tok, lab, g_bins = T.span_ce_index(labels, layout)
    w = model.embed.lm_weight()
    return split_span_ce(xh, T.pad_vocab_align8(w), gid, keep_tok, lab, g_bins,
                         copied_scored(x, labels, layout), vocab_size=int(w.shape[0]))


# ── part: patch ──────────────────────────────────────────────────────────────────────────
class _Done(Exception):
    pass


class OracleTap:
    """Instance-level taps (restored on exit) on ONE model:

    * `_tul_fan_oracle`: runs the original, and records the per-stream span-CE table
      `[B, S, K]` and the shipped-write table `[B, S]` by watching its calls of
      `accumulate_span_ce` (K stream calls, then the mixed one), plus the scored-slot mask.
    * `_back_region`: records the FIRST coda output of the forward (the shipped coda),
      which is all a `plan_mode != normal` forward gives (the oracle does not run there).
    """

    def __init__(self, model):
        self.m = model
        self.ce = self.mixed = self.ok = self.n_tok = self.xh0 = None
        self.ce_cp = self.ce_nv = self.mixed_cp = self.mixed_nv = None
        self.n_cp = self.n_nv = None

    def __enter__(self):
        m = self.m
        o_oracle, o_back = m._tul_fan_oracle, m._back_region
        import inspect
        sig = inspect.signature(o_oracle)

        def oracle(*a, **k):
            bound = sig.bind(*a, **k)
            bound.apply_defaults()
            if int(bound.arguments["n_rollouts"]) != 1:
                raise NotImplementedError("cell_lens: code rollouts (tul.code_enum_k > 1)")
            rec, split = [], []
            orig_acc = T.accumulate_span_ce
            cp_tok = copied_scored(bound.arguments["input_ids"], bound.arguments["labels"],
                                   bound.arguments["layout"])

            def acc(*aa, **kk):
                r = orig_acc(*aa, **kk)
                rec.append(r[:, 1:].detach().float())
                split.append(split_span_ce(*aa, copied=cp_tok, **kk))
                return r
            T.accumulate_span_ce = acc
            try:
                res = o_oracle(*a, **k)
            finally:
                T.accumulate_span_ce = orig_acc
            kk_ = int(bound.arguments["cells"].shape[2])
            if len(rec) != kk_ + 1:
                raise RuntimeError(f"oracle made {len(rec)} span-CE calls, expected "
                                   f"{kk_} streams + 1 mixed; the tap no longer matches it")
            labels, layout = bound.arguments["labels"], bound.arguments["layout"]
            n_tok = span_tokens(labels, layout)
            self.ce = torch.stack(rec[:kk_], dim=-1)
            self.mixed = rec[kk_]
            self.ce_cp = torch.stack([c for c, _ in split[:kk_]], dim=-1)
            self.ce_nv = torch.stack([n for _, n in split[:kk_]], dim=-1)
            self.mixed_cp, self.mixed_nv = split[kk_]
            gid, keep_tok, _lab, g_bins = T.span_ce_index(labels, layout)
            self.n_cp, self.n_nv = count_split(keep_tok, gid, cp_tok, g_bins)
            self.n_tok = n_tok
            self.ok = layout.slot_valid & (n_tok > 0)
            return res

        def back(*a, **k):
            r = o_back(*a, **k)
            if self.xh0 is None:
                self.xh0 = r
            return r

        m._tul_fan_oracle = oracle
        m._back_region = back
        return self

    def __exit__(self, *exc):
        for n in ("_tul_fan_oracle", "_back_region"):
            self.m.__dict__.pop(n, None)
        return False


ROUTER_KEYS = ("router_coda_agree", "router_pick_regret", "rand_pick_regret",
               "teacher_coda_agree")


@torch.no_grad()
def patch_batch(model, x, y, layout, d: int, device: str) -> dict:
    """One batch at forced depth ``d``: the oracle's tables, the zero-write table and the
    forward's own `fan_*` stats (for the anchor). Tensors stay on ``device``."""
    sd = forced_depths(layout, d, int(model.cfg.tul.slot_cells))
    with autocast_ctx(device):
        with OracleTap(model) as tap:
            out = model.tul_forward_ablated(x, y, layout, plan_mode="normal", slot_depths=sd)
        if tap.ce is None:
            raise RuntimeError("the val oracle did not run: this model has no fan "
                               "(tul.fan_k) or the forward was not eval + labelled")
        with OracleTap(model) as ztap:
            model.tul_forward_ablated(x, y, layout, plan_mode="zero", slot_depths=sd)
        if ztap.xh0 is None:
            raise RuntimeError("plan_mode=zero forward ran no coda")
        zero = span_table(model, ztap.xh0, y, layout)
        zero_cp, zero_nv = span_table_split(model, ztap.xh0, x, y, layout)
    return {"ce": tap.ce, "mixed": tap.mixed, "zero": zero, "ok": tap.ok,
            "n_tok": tap.n_tok, "out": out, "xh_all": tap.xh0, "xh_zero": ztap.xh0,
            "ce_cp": tap.ce_cp, "ce_nv": tap.ce_nv, "mixed_cp": tap.mixed_cp,
            "mixed_nv": tap.mixed_nv, "zero_cp": zero_cp, "zero_nv": zero_nv,
            "n_cp": tap.n_cp, "n_nv": tap.n_nv}


def summarise_batch(tb: dict) -> dict:
    """Per-row sums over the scored spans of one batch (the paired-bootstrap units)."""
    ok = tb["ok"]
    okf = ok.float()
    ce, mixed, zero = tb["ce"], tb["mixed"], tb["zero"]
    best, arg = ce.min(dim=-1)
    cnt = (tb["n_tok"].float() * okf).sum(1)
    kk = ce.shape[-1]
    wins = torch.stack([((arg == i) & ok).sum() for i in range(kk)]).double()
    okk = okf.unsqueeze(-1)
    best_cp = tb["ce_cp"].gather(-1, arg.unsqueeze(-1)).squeeze(-1)
    best_nv = tb["ce_nv"].gather(-1, arg.unsqueeze(-1)).squeeze(-1)

    def rs(t):                                                   # per-row sum over scored spans
        return t.sum(1).double().cpu().numpy()
    return {"cell_cp": rs(tb["ce_cp"] * okk), "cell_nv": rs(tb["ce_nv"] * okk),
            "all_cp": rs(tb["mixed_cp"] * okf), "all_nv": rs(tb["mixed_nv"] * okf),
            "zero_cp": rs(tb["zero_cp"] * okf), "zero_nv": rs(tb["zero_nv"] * okf),
            "best_cp": rs(best_cp * okf), "best_nv": rs(best_nv * okf),
            "cnt_cp": rs(tb["n_cp"] * okf), "cnt_nv": rs(tb["n_nv"] * okf),
            "cell": (ce * okf.unsqueeze(-1)).sum(1).double().cpu().numpy(),       # [B, K]
            "all": (mixed * okf).sum(1).double().cpu().numpy(),
            "zero": (zero * okf).sum(1).double().cpu().numpy(),
            "best": (best * okf).sum(1).double().cpu().numpy(),
            "cnt": cnt.double().cpu().numpy(), "wins": wins.cpu().numpy(),
            "n_spans": float(ok.sum())}


def check_anchor(tb: dict, sm: dict, tag: str) -> None:
    """The tapped table, aggregated here, must reproduce the forward's own `fan_*` stats.
    Same rows, same tokens: a drift here means the tap or the masks are wrong."""
    out, cnt = tb["out"], float(sm["cnt"].sum())
    pairs = [(f"stream_ce_k{i}", float(sm["cell"][:, i].sum()) / cnt)
             for i in range(sm["cell"].shape[1])]
    pairs += [("oracle_ce", float(sm["best"].sum()) / cnt),
              ("mixed_ce", float(sm["all"].sum()) / cnt)]
    for k, mine in pairs:
        theirs = float(out[f"fan_{k}"])
        if abs(mine - theirs) > 1e-4 * max(1.0, abs(theirs)):
            raise RuntimeError(f"{tag}: fan_{k} from the model {theirs:.6f} vs the tapped "
                               f"table {mine:.6f}; refusing to report")


def check_split(tb: dict, tag: str) -> None:
    """copied + novel must reproduce the oracle's own tables and the scored-token count."""
    ok = tb["ok"]
    pairs = [("cells", tb["ce_cp"] + tb["ce_nv"], tb["ce"]),
             ("all", tb["mixed_cp"] + tb["mixed_nv"], tb["mixed"]),
             ("zero", tb["zero_cp"] + tb["zero_nv"], tb["zero"])]
    for nm, got, ref in pairs:
        m = ok.unsqueeze(-1) if got.dim() == 3 else ok
        if not torch.allclose(got * m, ref * m, atol=1e-3, rtol=1e-4):
            raise RuntimeError(f"{tag}: copied + novel {nm} table != the oracle's table; "
                               "refusing to report the split")
    if not torch.equal((tb["n_cp"] + tb["n_nv"])[ok], tb["n_tok"][ok].float()):
        raise RuntimeError(f"{tag}: copied + novel token counts != the oracle's n_tok")


def load_model(spec: str, device: str):
    from morph.training.tul_setup import build_tul_runtime
    from tul_samples import load_ckpt
    label, config, path, ovr = parse_ckpt_spec(spec)
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    model, step = load_ckpt(cfg, path, device, tul_rt.model_cfg)
    model.eval()
    return label, cfg, model, step


def load_batches(path: str, key: str, n: int, device: str, model):
    data = torch.load(path, weights_only=False)
    raw = data[key][:n]
    if not raw:
        raise ValueError(f"{path} holds no '{key}' batches")
    out = [(x.to(device), y.to(device), lay.to(device)) for x, y, lay in raw]
    pk = int(model.cfg.tul.prefix_k)
    if int(out[0][2].prefix_k) != pk:
        raise ValueError(f"batches were packed at prefix_k {int(out[0][2].prefix_k)}, the "
                         f"model has {pk}; pack with --config of this model")
    return out, data


def _ratio(num: float, den: float):
    return float(num / den) if den > 0 else None


def span_rows(tb: dict, batch: int, row0: int) -> dict:
    """The per-span table of one batch (scored spans only): batch index, row inside the batch,
    global row, slot, winning cell (argmin of the single-cell CE), per-cell CE, all-cells CE,
    scored-token count. Columns, so the JSON stays small."""
    ok = tb["ok"]
    bi, si = ok.nonzero(as_tuple=True)
    ce = tb["ce"][bi, si]
    return {"batch": [batch] * len(bi), "b": bi.tolist(), "row": (bi + row0).tolist(),
            "slot": si.tolist(), "win": ce.argmin(-1).tolist(),
            "ce": ce.double().cpu().tolist(), "all": tb["mixed"][bi, si].double().cpu().tolist(),
            "n_tok": tb["n_tok"][bi, si].long().cpu().tolist()}


def run_patch(model, val, depths: list[int], device: str, log=print) -> dict:
    """The patch table per depth over ``val`` ([(x, y, layout)])."""
    check_depths(model, depths)
    res = {}
    keys = ("cell", "all", "zero", "best", "cnt", "cell_cp", "cell_nv", "all_cp", "all_nv",
            "zero_cp", "zero_nv", "best_cp", "best_nv", "cnt_cp", "cnt_nv")
    for d in depths:
        t0 = time.time()
        sums: list[dict] = []
        rt = {k: [0.0, 0.0] for k in ROUTER_KEYS}
        spans: dict[str, list] = {}
        row0 = 0
        for bi, (x, y, lay) in enumerate(val):
            tb = patch_batch(model, x, y, lay, d, device)
            sm = summarise_batch(tb)
            check_anchor(tb, sm, f"depth {d} batch {bi}")
            check_split(tb, f"depth {d} batch {bi}")
            sums.append(sm)
            for k_, v_ in span_rows(tb, bi, row0).items():
                spans.setdefault(k_, []).extend(v_)
            row0 += int(x.shape[0])
            for k in ROUTER_KEYS:
                if f"fan_{k}" in tb["out"]:
                    rt[k][0] += float(tb["out"][f"fan_{k}"]) * float(sm["cnt"].sum())
                    rt[k][1] += float(sm["cnt"].sum())
        cat = {k: np.concatenate([s[k] for s in sums]) for k in keys}
        cnt = cat["cnt"]
        wins = np.sum([s["wins"] for s in sums], axis=0)
        n_sp = float(sum(s["n_spans"] for s in sums))
        kk = cat["cell"].shape[1]
        e = {"n_tokens": float(cnt.sum()), "n_spans": n_sp,
             "cells": [float(cat["cell"][:, i].sum() / cnt.sum()) for i in range(kk)],
             "all": float(cat["all"].sum() / cnt.sum()),
             "oracle_best": float(cat["best"].sum() / cnt.sum()),
             "zero": float(cat["zero"].sum() / cnt.sum()),
             "win_frac": [float(w / n_sp) for w in wins],
             "ci_vs_all": {f"cell{i}": paired_bootstrap_ci(cat["cell"][:, i], cat["all"], cnt)
                           for i in range(kk)},
             "ci_best_vs_all": paired_bootstrap_ci(cat["best"], cat["all"], cnt),
             "ci_zero_vs_all": paired_bootstrap_ci(cat["zero"], cat["all"], cnt),
             "router": {k: rt[k][0] / rt[k][1] for k in ROUTER_KEYS if rt[k][1] > 0}}
        copy = {"frac_tokens_copied": _ratio(cat["cnt_cp"].sum(), cnt.sum())}
        for nm, suf in (("copied", "cp"), ("novel", "nv")):
            den = float(cat[f"cnt_{suf}"].sum())
            copy[nm] = {"n_tokens": den,
                        "cells": [_ratio(cat[f"cell_{suf}"][:, i].sum(), den) for i in range(kk)],
                        "all": _ratio(cat[f"all_{suf}"].sum(), den),
                        "oracle_best": _ratio(cat[f"best_{suf}"].sum(), den),
                        "zero": _ratio(cat[f"zero_{suf}"].sum(), den)}
        e["copy"] = copy
        e["spans"] = spans
        res[d] = e
        log(f"[patch] depth {d}: {len(val)} batches, {n_sp:.0f} spans ({time.time() - t0:.0f}s)")
    return res


def patch_markdown(label: str, res: dict) -> str:
    lines = [f"### {label}: CE (nats/token, coda only, no pointer head) with ONE cell written", ""]
    kk = len(next(iter(res.values()))["cells"])
    hdr = ["depth"] + [f"cell{i}" for i in range(kk)] + ["all", "best-of-cells", "zero"] + \
          [f"win{i}" for i in range(kk)]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "---|" * len(hdr))
    for d, e in res.items():
        row = [str(d)] + [f"{c:.4f}" for c in e["cells"]] + \
              [f"{e['all']:.4f}", f"{e['oracle_best']:.4f}", f"{e['zero']:.4f}"] + \
              [f"{w:.2f}" for w in e["win_frac"]]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    if all("copy" in e for e in res.values()):
        lines.append(f"### {label}: the same CE split by copied / novel target tokens "
                     "(copied = id seen earlier in the row, before the next span)")
        lines.append("")
        hdr2 = ["depth", "frac copied"] + [f"cell{i}" for i in range(kk)] + \
               ["all", "best-of-cells", "zero"]
        for nm in ("copied", "novel"):
            lines.append(f"**{nm}**")
            lines.append("")
            lines.append("| " + " | ".join(hdr2) + " |")
            lines.append("|" + "---|" * len(hdr2))
            for d, e in res.items():
                c = e["copy"]
                f_ = lambda v: "n/a" if v is None else f"{v:.4f}"  # noqa: E731
                lines.append("| " + " | ".join(
                    [str(d), f_(c["frac_tokens_copied"])] + [f_(v) for v in c[nm]["cells"]] +
                    [f_(c[nm]["all"]), f_(c[nm]["oracle_best"]), f_(c[nm]["zero"])]) + " |")
            lines.append("")
    for d, e in res.items():
        if e["router"]:
            lines.append(f"depth {d} router/pick readings: " +
                         ", ".join(f"{k}={v:.4f}" for k, v in e["router"].items()))
    return "\n".join(lines)


def part_patch(a) -> None:
    device = a.device
    results = {}
    for spec in a.ckpt:
        label, cfg, model, step = load_model(spec, device)
        val, data = load_batches(a.batches_file, "val", a.n_val, device, model)
        rows = sum(int(x.shape[0]) for x, _, _ in val)
        print(f"[patch] {label}: step {step}, {len(val)} val batches, {rows} rows", flush=True)
        res = run_patch(model, val, a.depths, device)
        print(patch_markdown(label, res), flush=True)
        results[label] = {"spec": spec, "step": step, "rows": rows, "n_val_batches": len(val),
                          "val_skip_docs": data.get("val_skip_docs"), "patch": res}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[patch] wrote {a.out}", flush=True)


# ── part: probe ──────────────────────────────────────────────────────────────────────────
@dataclasses.dataclass
class Features:
    """`feats [N, P, M, C]` cell readouts of N slots at the P passes of ``passes`` (0 = the
    loop's entry); `tgt [N, n_off]` tokens 1..n_off of the slot's next span (-100 where the
    span is shorter); `text [N, Tt]` the same with more tokens, for printing; `row [N]` the
    global row of each slot; ``copied [N, Tt]`` bool, the `copied_tokens` flag of every ``text``
    token (None = not computed)."""
    feats: torch.Tensor
    tgt: torch.Tensor
    text: torch.Tensor
    row: torch.Tensor
    passes: list[int]
    copied: torch.Tensor | None = None


@torch.no_grad()
def collect_features(model, batches, depths: list[int], n_off: int, device: str,
                     text_tokens: int = 24, store_dtype=torch.bfloat16) -> Features:
    """Run the loop to the deepest requested pass on each batch and stop there (no coda,
    no loss heads): the eval-only `_slot_pass_hook` hands over the carrier at each pass,
    and an exception ends the forward right after the last one that is wanted."""
    check_depths(model, depths)
    m_cells = int(model.cfg.tul.slot_cells)
    want = sorted(set(depths))
    big_d = want[-1]
    passes = [0] + want
    tt = max(text_tokens, n_off)
    fs, tg, tx, rw, cps = [], [], [], [], []
    row0 = 0
    for x, y, layout in batches:
        B, S = layout.slot_valid.shape
        cap: dict[int, torch.Tensor] = {}

        def hook(t, h, h_new, lay, cap=cap):
            if t == 0:
                cap[0] = h
            if (t + 1) in want:
                cap[t + 1] = h_new
            if t + 1 >= big_d:
                raise _Done()
            return h_new

        prev = model.__dict__.get("_slot_pass_hook")
        model._slot_pass_hook = hook
        try:
            with autocast_ctx(device):
                model.tul_forward_ablated(x, None, layout, plan_mode="normal",
                                          slot_depths=forced_depths(layout, big_d, m_cells))
        except _Done:
            pass
        finally:
            if prev is None:
                model.__dict__.pop("_slot_pass_hook", None)
            else:
                model._slot_pass_hook = prev
        if sorted(cap) != passes:
            raise RuntimeError(f"hook saw passes {sorted(cap)}, wanted {passes}: no slot loop?")
        cr = []
        for p in passes:
            h = cap[p]
            if h.shape[0] != B or h.shape[1] != S * m_cells:
                raise RuntimeError(f"carrier {tuple(h.shape)} is not [B={B}, S*M={S * m_cells}, ..]")
            cr.append(_cell_readout(h.reshape(B, S, m_cells, *h.shape[2:])))
        cr = torch.stack(cr, dim=2)                                   # [B, S, P, M, C]
        pos, valid = next_span_positions(layout, y, x, max_tokens=tt)  # [B, S, tt]
        tok = y.gather(1, pos.reshape(B, -1)).reshape(B, S, tt)
        tok = torch.where(valid, tok, torch.full_like(tok, IGNORE))
        ok = valid[:, :, 0]
        rid = (torch.arange(B, device=y.device) + row0).view(B, 1).expand(B, S)
        cps.append(copied_next_span(x, layout, tok)[ok])
        fs.append(cr[ok].to(store_dtype))
        tg.append(tok[ok][:, :n_off])
        tx.append(tok[ok])
        rw.append(rid[ok])
        row0 += B
    return Features(torch.cat(fs), torch.cat(tg), torch.cat(tx), torch.cat(rw), passes,
                    torch.cat(cps))


class LensProbes(nn.Module):
    """One probe per (pass, offset), shared by all cells:
    `logits = (RMSNorm(x) (I + dW)^T) E^T * exp(s)`; E is the model's tied table, frozen."""

    def __init__(self, n_pass: int, n_off: int, c: int, emb: torch.Tensor):
        super().__init__()
        self.register_buffer("emb", emb.detach().float().clone())
        self.dw = nn.Parameter(torch.zeros(n_pass, n_off, c, c))
        self.log_s = nn.Parameter(torch.zeros(n_pass, n_off))
        self.c = c

    def logits(self, x: torch.Tensor, p: int, k: int) -> torch.Tensor:
        z = F.rms_norm(x.float(), (self.c,))
        z = z + z @ self.dw[p, k].t()
        return (z @ self.emb.t()) * self.log_s[p, k].exp()


def _rows(f: Features, idx: torch.Tensor, p: int, device: str):
    """Cell vectors `[len(idx) * M, C]` of pass index ``p`` (cell-fastest order)."""
    return f.feats[idx.to(f.feats.device), p].to(device).flatten(0, 1)


def train_probes(f: Features, emb: torch.Tensor, steps: int, batch: int, lr: float,
                 wd: float, device: str, seed: int = 0, log=print) -> LensProbes:
    n, p_n, m, c = f.feats.shape
    n_off = f.tgt.shape[1]
    g = torch.Generator().manual_seed(seed)
    probes = LensProbes(p_n, n_off, c, emb.to(device)).to(device)
    opt = torch.optim.AdamW([{"params": [probes.dw], "lr": lr, "weight_decay": wd},
                             {"params": [probes.log_s], "lr": 10 * lr, "weight_decay": 0.0}])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, n, (min(batch, n),), generator=g)
        tgt = f.tgt[idx.to(f.tgt.device)].to(device)                 # [b, n_off]
        tot = 0.0
        opt.zero_grad(set_to_none=True)
        for p in range(p_n):
            xr = _rows(f, idx, p, device)
            for k in range(n_off):
                lab = tgt[:, k].repeat_interleave(m)
                if not bool((lab != IGNORE).any()):
                    continue
                with autocast_ctx(device):
                    lg = probes.logits(xr, p, k)
                loss = F.cross_entropy(lg.float(), lab, ignore_index=IGNORE) / (p_n * n_off)
                loss.backward()          # disjoint parameters per (p, k): one graph at a time
                tot += loss.item()
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log(f"[probe] step {step:5d}/{steps} mean CE {tot:.4f} ({time.time() - t0:.0f}s)")
    return probes


def unigram_stats(train_tgt: torch.Tensor, vocab: int) -> dict:
    """Train-frequency unigram per offset: smoothed (+1) probabilities."""
    n_off = train_tgt.shape[1]
    out = []
    for k in range(n_off):
        t = train_tgt[:, k]
        t = t[t != IGNORE]
        cnt = torch.bincount(t, minlength=vocab).double() + 1.0
        out.append(cnt / cnt.sum())
    return {"p": out}


@torch.no_grad()
def eval_probes(probes: LensProbes, f: Features, uni: dict, device: str, chunk: int = 128
                ) -> dict:
    """CE / top-1 / top-5 per (pass, offset): per cell, all-cells-mean input, shuffled cell,
    and the unigram. Shuffled = the SAME cell index of the slot half the set away (another
    row whenever the set spans >= 2 rows); its same-row fraction is reported.

    With ``f.copied`` every number is also kept split by copied / novel target token
    (``probe_split``, ``unigram_split``, ``copy``); the unsplit numbers are the sums of the two
    parts and are unchanged."""
    n, p_n, m, c = f.feats.shape
    n_off = f.tgt.shape[1]
    perm = (torch.arange(n) + max(1, n // 2)) % n
    same_row = float((f.row == f.row[perm]).float().mean())
    names = [f"cell{i}" for i in range(m)] + ["mean", "shuffled"]
    have_cp = f.copied is not None
    acc = np.zeros((p_n, n_off, len(names), 3, 2))                   # ce, top1, top5 | novel, copied
    cnt = np.zeros((n_off, 2))
    uni_ce = np.zeros((n_off, 3, 2))
    for s in range(0, n, chunk):
        idx = torch.arange(s, min(n, s + chunk))
        tgt = f.tgt[idx].to(device)
        cpf = (f.copied[idx][:, :n_off].to(device) if have_cp
               else torch.zeros_like(tgt, dtype=torch.bool))
        x = f.feats[idx].to(device).float()                           # [b, P, M, C]
        xs = f.feats[perm[idx]].to(device).float()
        for k in range(n_off):
            valid = tgt[:, k] != IGNORE
            nv = int(valid.sum())
            if nv == 0:
                continue
            lab = tgt[:, k][valid]
            cp = cpf[:, k][valid]
            sides = (~cp, cp)
            for q, sd in enumerate(sides):
                cnt[k, q] += float(sd.sum())
            pu = uni["p"][k].to(device)
            top5u = pu.topk(5).indices
            u_stats = (-pu.log()[lab], (lab == top5u[0]).float(),
                       (lab.unsqueeze(1) == top5u.unsqueeze(0)).any(1).float())
            for st_i, st in enumerate(u_stats):
                for q, sd in enumerate(sides):
                    uni_ce[k, st_i, q] += float(st[sd].sum())
            for p in range(p_n):
                variants = [x[valid][:, p, i] for i in range(m)]
                variants.append(x[valid][:, p].mean(dim=1))
                variants.append(xs[valid][:, p, :].reshape(nv, m, c))   # shuffled: per cell
                for j, v in enumerate(variants):
                    shuf = j == len(names) - 1                          # shuffled, all cells
                    vv = v.reshape(nv * m, c) if shuf else v
                    ll = lab.repeat_interleave(m) if shuf else lab
                    lg = probes.logits(vv, p, k).float()
                    ce = F.cross_entropy(lg, ll, reduction="none")
                    t5 = lg.topk(5, dim=-1).indices
                    st = (ce, (t5[:, 0] == ll).float(), (t5 == ll.unsqueeze(1)).any(1).float())
                    if shuf:                                            # mean over the M cells
                        st = tuple(t.view(nv, m).mean(1) for t in st)
                    for st_i, t in enumerate(st):
                        for q, sd in enumerate(sides):
                            acc[p, k, j, st_i, q] += float(t[sd].sum())
    tot = cnt.sum(1)
    res = {"names": names, "passes": f.passes, "n_scored": tot.tolist(), "same_row_frac": same_row,
           "probe": {}, "unigram": {}}
    if have_cp:
        res["copy"] = {"n_copied": cnt[:, 1].tolist(), "n_novel": cnt[:, 0].tolist(),
                       "frac_copied": [float(cnt[k, 1] / tot[k]) if tot[k] else None
                                       for k in range(n_off)]}
        res["probe_split"], res["unigram_split"] = {}, {}

    def triple(a3, den):
        return {"ce": a3[0] / den, "top1": a3[1] / den, "top5": a3[2] / den} if den else None
    for k in range(n_off):
        res["unigram"][f"off{k + 1}"] = triple(uni_ce[k].sum(1), tot[k])
        if have_cp:
            res["unigram_split"][f"off{k + 1}"] = {
                "copied": triple(uni_ce[k, :, 1], cnt[k, 1]),
                "novel": triple(uni_ce[k, :, 0], cnt[k, 0])}
        for p in range(p_n):
            key = f"pass{f.passes[p]}_off{k + 1}"
            res["probe"][key] = {nm: triple(acc[p, k, j].sum(1), tot[k])
                                 for j, nm in enumerate(names)}
            if have_cp:
                res["probe_split"][key] = {
                    nm: {"copied": triple(acc[p, k, j, :, 1], cnt[k, 1]),
                         "novel": triple(acc[p, k, j, :, 0], cnt[k, 0])}
                    for j, nm in enumerate(names)}
    return res


def probe_markdown(label: str, res: dict) -> str:
    names = res["names"]
    lines = [f"### {label}: Future-Lens probe, val CE | top1 | top5 (same-row frac of shuffled "
             f"control {res['same_row_frac']:.3f})", ""]
    n_off = len(res["unigram"])
    for k in range(n_off):
        lines.append(f"**offset {k + 1}** (n={res['n_scored'][k]:.0f}); unigram "
                     f"{res['unigram'][f'off{k + 1}']['ce']:.3f} | "
                     f"{res['unigram'][f'off{k + 1}']['top1']:.3f} | "
                     f"{res['unigram'][f'off{k + 1}']['top5']:.3f}")
        lines.append("")
        lines.append("| pass | " + " | ".join(names) + " |")
        lines.append("|" + "---|" * (len(names) + 1))
        for p in res["passes"]:
            r = res["probe"][f"pass{p}_off{k + 1}"]
            lines.append(f"| {p} | " + " | ".join(
                f"{r[nm]['ce']:.3f}/{r[nm]['top1']:.3f}/{r[nm]['top5']:.3f}" for nm in names) + " |")
        lines.append("")
        if "probe_split" in res:
            cp = res["copy"]
            lines.append(f"offset {k + 1} split: {cp['n_copied'][k]:.0f} copied "
                         f"({cp['frac_copied'][k]:.3f}), {cp['n_novel'][k]:.0f} novel")
            lines.append("")
            fmt = lambda t: "n/a" if t is None else \
                f"{t['ce']:.3f}/{t['top1']:.3f}/{t['top5']:.3f}"      # noqa: E731
            for part in ("copied", "novel"):
                u = res["unigram_split"][f"off{k + 1}"][part]
                lines.append(f"*{part}* (unigram {fmt(u)})")
                lines.append("")
                lines.append("| pass | " + " | ".join(names) + " |")
                lines.append("|" + "---|" * (len(names) + 1))
                for p in res["passes"]:
                    r = res["probe_split"][f"pass{p}_off{k + 1}"]
                    lines.append(f"| {p} | " + " | ".join(fmt(r[nm][part]) for nm in names) + " |")
                lines.append("")
    return "\n".join(lines)


@torch.no_grad()
def make_examples(probes: LensProbes, f: Features, decode: Callable[[list[int]], str],
                  device: str, n: int = 8, seed: int = 0) -> list[dict]:
    """``n`` val slots: the true next span, and per cell the probe's top-5 tokens at offsets
    1 and 2 on the final pass."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(f.feats.shape[0], generator=g)[:n]
    p = len(f.passes) - 1
    out = []
    for i in idx.tolist():
        txt = [t for t in f.text[i].tolist() if t != IGNORE]
        ex = {"slot": i, "true": decode(txt), "cells": []}
        x = f.feats[i, p].to(device).float()                          # [M, C]
        for ci in range(x.shape[0]):
            row = {"cell": ci}
            for k in range(min(2, probes.dw.shape[1])):
                top = probes.logits(x[ci:ci + 1], p, k).float().topk(5, dim=-1).indices[0].tolist()
                row[f"off{k + 1}"] = [decode([t]) for t in top]
            ex["cells"].append(row)
        out.append(ex)
    return out


def examples_markdown(label: str, exs: list[dict], last_pass: int) -> str:
    lines = [f"### {label}: examples (final pass {last_pass}; top-5 tokens per cell)", ""]
    for e in exs:
        lines.append(f"TRUE: {e['true']!r}")
        for r in e["cells"]:
            lines.append("  cell%d  off1 %s   off2 %s" % (r["cell"], r.get("off1"), r.get("off2")))
        lines.append("")
    return "\n".join(lines)


def part_probe(a) -> None:
    device = a.device
    results = {}
    for spec in a.ckpt:
        label, cfg, model, step = load_model(spec, device)
        train, _ = load_batches(a.batches_file, "train", a.n_train, device, model)
        val, data = load_batches(a.batches_file, "val", a.n_val, device, model)
        t0 = time.time()
        ftr = collect_features(model, train, a.depths, a.offsets, device)
        fva = collect_features(model, val, a.depths, a.offsets, device)
        print(f"[probe] {label}: step {step}; features train {tuple(ftr.feats.shape)} "
              f"val {tuple(fva.feats.shape)} ({time.time() - t0:.0f}s)", flush=True)
        emb = model.embed.lm_weight().detach().float()
        uni = unigram_stats(ftr.tgt, int(emb.shape[0]))
        probes = train_probes(ftr, emb, a.steps, a.probe_batch, a.lr, a.wd, device)
        res = eval_probes(probes, fva, uni, device)
        print(probe_markdown(label, res), flush=True)
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        exs = make_examples(probes, fva, lambda ids: tok.decode(ids, skip_special_tokens=True),
                            device, n=8)
        print(examples_markdown(label, exs, fva.passes[-1]), flush=True)
        results[label] = {"spec": spec, "step": step, "n_train_slots": int(ftr.feats.shape[0]),
                          "n_val_slots": int(fva.feats.shape[0]), "probe": res, "examples": exs,
                          "val_skip_docs": data.get("val_skip_docs"),
                          "args": {k: v for k, v in vars(a).items() if k != "ckpt"}}
        del model, probes
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f_:
        json.dump(results, f_, indent=1)
    print(f"[probe] wrote {a.out}", flush=True)


# ── part: special (specialization contingency) ─────────────────────────────────────────
FIRST_CLASSES = ["punctuation", "whitespace", "digit", "capitalised", "lowercase", "other"]


def first_token_class(text: str) -> str:
    """Class of a decoded first token: punctuation / whitespace (incl. newline) / digit /
    capitalised word / lowercase word / other. A leading space is stripped first, so " the" is a
    lowercase word and ".\\n" is punctuation. A broken byte sequence (U+FFFD) is "other"."""
    st = text.strip()
    if not st:
        return "whitespace"
    if "�" in st:
        return "other"
    if all(unicodedata.category(ch)[0] in "PS" for ch in st):
        return "punctuation"
    ch = st[0]
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "capitalised" if ch.isupper() else "lowercase"
    return "other"


def span_properties(batches) -> list[dict]:
    """Per batch, numpy `[B, S]` arrays for the NEXT span of every slot: its length in tokens
    ``n``, its copy ratio ``cr`` (fraction of its tokens that `copied_tokens` marks) and its
    first token id ``first`` (IGNORE where the slot has no next span)."""
    out = []
    for x, y, layout in batches:
        pos, valid = next_span_positions(layout, y, x)
        b, s, j = valid.shape
        tok = y.gather(1, pos.reshape(b, -1)).reshape(b, s, j)
        tok = torch.where(valid, tok, torch.full_like(tok, IGNORE))
        cp = copied_next_span(x, layout, tok)
        n = valid.sum(-1)
        out.append({"n": n.cpu().numpy(), "cr": (cp.sum(-1) / n.clamp(min=1)).cpu().numpy(),
                    "first": tok[:, :, 0].cpu().numpy()})
    return out


def _entropy(counts: np.ndarray) -> float:
    c = np.asarray(counts, dtype=np.float64)
    p = c[c > 0] / c.sum()
    return float(-(p * np.log(p)).sum())


def nmi_from_joint(joint: np.ndarray) -> float:
    """Normalised mutual information, I(X;Y) / mean(H(X), H(Y)) (arithmetic mean, the
    scikit-learn default). 0 when either variable is constant (nothing to explain)."""
    joint = np.asarray(joint, dtype=np.float64)
    n = joint.sum()
    px, py = joint.sum(1), joint.sum(0)
    hx, hy = _entropy(px), _entropy(py)
    if hx <= 0 or hy <= 0:
        return 0.0
    pxy = joint / n
    nz = pxy > 0
    mi = float((pxy[nz] * np.log(pxy[nz] / np.outer(px / n, py / n)[nz])).sum())
    return mi / (0.5 * (hx + hy))


def _joint(a: np.ndarray, b: np.ndarray, na: int, nb: int) -> np.ndarray:
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)


def nmi_with_null(a, b, n_perm: int = 200, seed: int = 0) -> dict:
    """NMI(a; b) for two label arrays and a permutation null: ``a`` is shuffled ``n_perm`` times
    (seeded). Reports the observed value, the null mean and 95th percentile and the one-sided
    permutation p-value ``(1 + #{null >= obs}) / (1 + n_perm)``."""
    ua, ia = np.unique(np.asarray(a), return_inverse=True)
    ub, ib = np.unique(np.asarray(b), return_inverse=True)
    obs = nmi_from_joint(_joint(ia, ib, len(ua), len(ub)))
    rng = np.random.default_rng(seed)
    null = np.array([nmi_from_joint(_joint(rng.permutation(ia), ib, len(ua), len(ub)))
                     for _ in range(n_perm)])
    return {"nmi": obs, "null_mean": float(null.mean()), "null_p95": float(np.percentile(null, 95)),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)), "n": int(len(ia))}


def quartile_bins(v) -> tuple[np.ndarray, list[str]]:
    """Quartile bin index per value (ties on an edge go to the upper bin, duplicate edges
    collapse, empty bins are dropped) and a readable name per kept bin."""
    v = np.asarray(v, dtype=np.float64)
    edges = np.unique(np.quantile(v, [0.25, 0.5, 0.75]))
    raw = np.searchsorted(edges, v, side="right")
    used = np.unique(raw)
    remap = {int(u): i for i, u in enumerate(used)}
    names = []
    for u in used:
        lo = "-inf" if u == 0 else f"{edges[u - 1]:.3g}"
        hi = "+inf" if u == len(edges) else f"{edges[u]:.3g}"
        names.append(f"[{lo},{hi})")
    return np.array([remap[int(r)] for r in raw]), names


def run_special(spans: dict, props: list[dict], decode: Callable[[list[int]], str],
                n_cells: int, n_perm: int = 200, seed: int = 0) -> dict:
    """Winning cell x span property contingency + NMI + permutation null, for one depth's
    per-span table (``spans``: `run_patch`'s columns). Difficulty = all-cells CE per scored
    token."""
    bidx, b, sl = (np.asarray(spans[k]) for k in ("batch", "b", "slot"))
    n = np.empty(len(bidx))
    cr = np.empty(len(bidx))
    first = np.empty(len(bidx), dtype=np.int64)
    for bi in np.unique(bidx):
        m = bidx == bi
        n[m], cr[m], first[m] = (props[bi][k][b[m], sl[m]] for k in ("n", "cr", "first"))
    # The patch scores the row's tail after the LAST slot of a full row (bin S, slot S - 1);
    # `next_span_positions` needs slot s + 1 to exist, so that one span has no properties.
    # Drop it, counted. Any other missing span means the files do not match: raise.
    tail = (n <= 0) & (sl == int(props[0]["n"].shape[1]) - 1)
    if ((n <= 0) & ~tail).any():
        raise RuntimeError("a patched span has no next-span length in the batches file: the "
                           "file does not match the patch JSON")
    keep = ~tail
    bidx, b, sl, n, cr, first = (v[keep] for v in (bidx, b, sl, n, cr, first))
    win = np.asarray(spans["win"])[keep]
    diff = (np.asarray(spans["all"]) / np.asarray(spans["n_tok"]))[keep]
    cls_of = {int(t): first_token_class(decode([int(t)])) for t in np.unique(first)}
    cls = np.array([FIRST_CLASSES.index(cls_of[int(t)]) for t in first])
    used = sorted(set(cls.tolist()))
    cls_bin = np.array([used.index(c) for c in cls])
    props_out = {"length": quartile_bins(n), "copy_ratio": quartile_bins(cr),
                 "difficulty": quartile_bins(diff),
                 "first_token_class": (cls_bin, [FIRST_CLASSES[c] for c in used])}
    res = {"n_spans": int(len(win)), "n_dropped_tail": int(tail.sum()), "n_cells": n_cells,
           "win_frac": [float((win == i).mean()) for i in range(n_cells)], "properties": {}}
    for name, (bins, names) in props_out.items():
        cont = _joint(win, bins, n_cells, len(names)).astype(int)
        res["properties"][name] = {"bins": names, "contingency": cont.tolist(),
                                   **nmi_with_null(win, bins, n_perm, seed)}
    return res


def special_markdown(label: str, depth, res: dict) -> str:
    lines = [f"### {label}: winning cell x span property, depth {depth} "
             f"({res['n_spans']} spans; winner shares " +
             " ".join(f"{w:.3f}" for w in res["win_frac"]) + ")", ""]
    for name, r in res["properties"].items():
        lines.append(f"**{name}**: NMI {r['nmi']:.4f}, permutation null mean {r['null_mean']:.4f}, "
                     f"p95 {r['null_p95']:.4f}, p {r['p_value']:.3f}")
        lines.append("")
        lines.append("| winner | " + " | ".join(r["bins"]) + " |")
        lines.append("|" + "---|" * (len(r["bins"]) + 1))
        for i, row in enumerate(r["contingency"]):
            lines.append(f"| cell{i} | " + " | ".join(str(c) for c in row) + " |")
        lines.append("")
    return "\n".join(lines)


def load_batches_cpu(path: str, key: str, n: int):
    data = torch.load(path, weights_only=False, map_location="cpu")
    return data[key][:n], data


def part_special(a) -> None:
    from transformers import AutoTokenizer
    with open(a.patch_json) as f_:
        patch = json.load(f_)
    results = {}
    for label, r in patch.items():
        n_val = int(r.get("n_val_batches", a.n_val))
        val, _ = load_batches_cpu(a.batches_file, "val", n_val)
        rows = sum(int(x.shape[0]) for x, _, _ in val)
        if rows != int(r["rows"]):
            raise ValueError(f"{label}: patch JSON scored {r['rows']} rows, the batches file "
                             f"gives {rows} (n_val_batches {n_val})")
        props = span_properties(val)
        _, config, _, _ = parse_ckpt_spec(r["spec"])
        tok = AutoTokenizer.from_pretrained(build_cfg(config, []).data.tokenizer)
        depths = sorted(int(d) for d in r["patch"])
        depth = a.special_depth if a.special_depth else depths[-1]
        e = r["patch"][str(depth)]
        res = run_special(e["spans"], props, lambda ids: tok.decode(ids), len(e["cells"]),
                          n_perm=a.perms, seed=a.seed)
        print(special_markdown(label, depth, res), flush=True)
        results[label] = {"spec": r["spec"], "depth": depth, "special": res}
    with open(a.out, "w") as f_:
        json.dump(results, f_, indent=1)
    print(f"[special] wrote {a.out}", flush=True)


# ── part: spandec (free-running span-decoder readout per cell) ───────────────────────────
class SpandecTap:
    """Instance-level tap on `_tul_spandec_loss`: records the arguments the forward hands it
    (the exit state `h_slots` the decoder was trained on, the register's `cells`, `input_ids`,
    `layout`, `fold_cells`) and ends the forward there with `_Done` (no coda, no loss)."""

    def __init__(self, model):
        self.m = model
        self.args = None

    def __enter__(self):
        import inspect
        orig = self.m._tul_spandec_loss
        sig = inspect.signature(orig)

        def tap(*a, **k):
            b = sig.bind(*a, **k)
            b.apply_defaults()
            self.args = dict(b.arguments)
            raise _Done()
        self.m._tul_spandec_loss = tap
        return self

    def __exit__(self, *exc):
        self.m.__dict__.pop("_tul_spandec_loss", None)
        return exc[0] is _Done


def check_spandec_model(model) -> None:
    dec = getattr(model, "tul_spandec", None)
    if dec is None:
        raise ValueError("cell_lens --part spandec: this checkpoint has no span decoder "
                         "(tul.spandec is false); there is nothing to read the cells with")
    if dec.reads_cells or dec.horizon != 1 or dec.target_offset != 1:
        raise ValueError("cell_lens --part spandec supports the next-span decoder on the mean / "
                         "winner / each readout; this one has reads_cells="
                         f"{dec.reads_cells}, horizon={dec.horizon}, target_offset={dec.target_offset}")


@dataclasses.dataclass
class SpandecInputs:
    """Per valid slot (one with a next span): `z_cells [N, M, C]` (the decoder input for cell i
    alone), `z_mean [N, C]` (readout of the mean of the cells), `z_trained [N, C]` or None (the
    exact state the model hands the decoder in training: the winner cell under
    `spandec_read: winner`; None under `each`, where the cells ARE the trained inputs), the next
    span's `ids`/`valid` `[N, J]` and the `row` of each slot."""
    z_cells: torch.Tensor
    z_mean: torch.Tensor
    z_trained: torch.Tensor | None
    ids: torch.Tensor
    valid: torch.Tensor
    row: torch.Tensor


@torch.no_grad()
def collect_spandec_inputs(model, batches, depth: int, device: str) -> SpandecInputs:
    check_spandec_model(model)
    check_depths(model, [depth])
    dec = model.tul_spandec
    m_cells = int(model.cfg.tul.slot_cells)
    zc, zm, zt, ids_l, val_l, rows = [], [], [], [], [], []
    row0 = 0
    for x, y, layout in batches:
        with SpandecTap(model) as tap:
            with autocast_ctx(device):
                model.tul_forward_ablated(x, None, layout, plan_mode="normal",
                                          slot_depths=forced_depths(layout, depth, m_cells))
        if tap.args is None:
            raise RuntimeError("the forward never reached the span decoder")
        a = tap.args
        cells = a["cells"]
        with autocast_ctx(device):
            z_i = torch.stack([model._readout(cells[:, :, i]) for i in range(m_cells)], dim=2)
            z_mean = model._readout(cells.mean(dim=2))
            z_tr = None if a["fold_cells"] else model._readout(a["h_slots"])
        if z_tr is not None and str(model.cfg.tul.spandec_read) == "mean" and \
                not torch.allclose(z_tr.float(), z_mean.float(), atol=1e-3, rtol=1e-3):
            raise RuntimeError("tap check: the decoder's trained input is not the readout of the "
                               "mean of the cells under spandec_read: mean")
        ids, valid = next_span_slots(x, layout, dec.per_span_tokens)
        ok = valid[:, :, 0]
        b, s = ok.shape
        zc.append(z_i[ok].float().cpu())
        zm.append(z_mean[ok].float().cpu())
        if z_tr is not None:
            zt.append(z_tr[ok].float().cpu())
        ids_l.append(ids[ok].cpu())
        val_l.append(valid[ok].cpu())
        rows.append((torch.arange(b, device=ok.device).view(b, 1).expand(b, s) + row0)[ok].cpu())
        row0 += b
    return SpandecInputs(torch.cat(zc), torch.cat(zm), torch.cat(zt) if zt else None,
                         torch.cat(ids_l), torch.cat(val_l), torch.cat(rows))


@torch.no_grad()
def greedy_free_run(dec, z: torch.Tensor, n_steps: int, emb: torch.Tensor, slot_id: int,
                    device: str, chunk: int = 256) -> torch.Tensor:
    """Greedy, no teacher forcing. ``z [N, C]`` -> ``[N, n_steps]`` token ids. Step j runs the
    decoder on the ids it produced so far (the decoder's input at position j is `z` for j = 0 and
    the embedding of its own token j-1 after that, exactly the training input without the true
    ids) and takes the argmax of the tied head at position j, the slot-id logit masked as in
    training. `decode` is causal, so a prefix of length j + 1 gives the same state as the full
    window."""
    out = []
    for s in range(0, z.shape[0], chunk):
        zc = z[s:s + chunk].to(device)
        ids = torch.zeros(zc.shape[0], 1, n_steps, dtype=torch.long, device=device)
        for j in range(n_steps):
            pre = ids[:, :, :j + 1]
            with autocast_ctx(device):
                st = dec.decode(zc.unsqueeze(1), pre, torch.ones_like(pre, dtype=torch.bool), emb)
            lg = (st[:, 0, j].float() @ emb.float().t())
            lg[:, slot_id] = float("-inf")
            ids[:, 0, j] = lg.argmax(-1)
        out.append(ids[:, 0].cpu())
    return torch.cat(out)


@torch.no_grad()
def teacher_forced(dec, z: torch.Tensor, ids: torch.Tensor, valid: torch.Tensor,
                   emb: torch.Tensor, slot_id: int, device: str, chunk: int = 64):
    """Teacher-forced CE and top-1 hit `[N, J]` (0 where invalid): the decoder's own training
    reading, from the same inputs."""
    ces, hits = [], []
    for s in range(0, z.shape[0], chunk):
        zc, ic, vc = (t[s:s + chunk].to(device) for t in (z, ids, valid))
        with autocast_ctx(device):
            st = dec.decode(zc.unsqueeze(1), ic.unsqueeze(1), vc.unsqueeze(1), emb)[:, 0]
        lg = st.float() @ emb.float().t()
        lg[..., slot_id] = float("-inf")
        ce = F.cross_entropy(lg.flatten(0, 1), ic.flatten(), reduction="none").view(ic.shape)
        ces.append((ce * vc).cpu())
        hits.append(((lg.argmax(-1) == ic) & vc).cpu())
    return torch.cat(ces), torch.cat(hits)


def score_decode(free: torch.Tensor, ce: torch.Tensor, hit: torch.Tensor, ids: torch.Tensor,
                 valid: torch.Tensor, n_pos: int = 8) -> dict:
    """Free-running token accuracy at positions 1..n_pos (over slots whose span is that long),
    exact-match rate over the whole span, overall free accuracy, and the teacher-forced CE and
    top-1 for the same inputs (per position 1..n_pos and overall)."""
    j = ids.shape[1]
    match = (free[:, :j] == ids) & valid
    pos = {}
    for p in range(1, min(n_pos, j) + 1):
        v = valid[:, p - 1]
        nv = int(v.sum())
        pos[str(p)] = {"n": nv,
                       "free_acc": float(match[:, p - 1][v].float().mean()) if nv else None,
                       "tf_ce": float(ce[:, p - 1][v].mean()) if nv else None,
                       "tf_top1": float(hit[:, p - 1][v].float().mean()) if nv else None}
    nt = float(valid.sum())
    return {"positions": pos, "exact_match": float((match | ~valid).all(1).float().mean()),
            "free_acc_all": float(match.sum() / nt), "tf_ce_all": float(ce.sum() / nt),
            "tf_top1_all": float(hit.sum() / nt)}


def spandec_variants(inp: SpandecInputs, seed: int = 0) -> dict[str, torch.Tensor]:
    """The decoder inputs: cell i alone, the mean of the cells, the decoder's trained input (if the
    model has one distinct from them), an RMS-matched Gaussian noise vector (per slot RMS of the
    mean-of-cells readout), and the readout of cell (n mod M) of the slot half the set away
    (another row whenever the set spans two rows)."""
    n, m, c = inp.z_cells.shape
    g = torch.Generator().manual_seed(seed)
    var = {f"cell{i}": inp.z_cells[:, i] for i in range(m)}
    var["mean"] = inp.z_mean
    if inp.z_trained is not None:
        var["trained"] = inp.z_trained
    rms = inp.z_mean.pow(2).mean(-1, keepdim=True).sqrt()
    var["noise"] = torch.randn(n, c, generator=g) * rms
    other = (torch.arange(n) + max(1, n // 2)) % n
    var["shuffled"] = inp.z_cells[other, torch.arange(n) % m]
    return var


def run_spandec(model, inp: SpandecInputs, device: str, max_slots: int = 4000, seed: int = 0,
                n_pos: int = 8, log=print) -> dict:
    """Free-running and teacher-forced scores of every variant on (at most ``max_slots``) slots.
    Returns ``{"n_slots", "variants": {name: score}, "free": {name: ids}, "sel": idx}``."""
    dec = model.tul_spandec
    emb = model.embed.lm_weight().detach().to(device)
    slot_id = int(model.cfg.tul.slot_id)
    n = inp.ids.shape[0]
    g = torch.Generator().manual_seed(seed)
    sel = torch.randperm(n, generator=g)[:max_slots].sort().values
    sub = SpandecInputs(inp.z_cells[sel], inp.z_mean[sel],
                        None if inp.z_trained is None else inp.z_trained[sel],
                        inp.ids[sel], inp.valid[sel], inp.row[sel])
    steps = int(sub.valid.sum(1).max())
    sub.ids, sub.valid = sub.ids[:, :steps], sub.valid[:, :steps]    # nothing is valid past it
    out, free_all = {}, {}
    for name, z in spandec_variants(sub, seed).items():
        t0 = time.time()
        free = greedy_free_run(dec, z, steps, emb, slot_id, device)
        ce, hit = teacher_forced(dec, z, sub.ids, sub.valid, emb, slot_id, device)
        out[name] = score_decode(free, ce, hit, sub.ids, sub.valid, n_pos)
        free_all[name] = free
        log(f"[spandec] {name}: {time.time() - t0:.0f}s")
    return {"n_slots": int(sub.ids.shape[0]), "variants": out, "free": free_all, "sel": sel,
            "inputs": sub}


def spandec_markdown(label: str, res: dict, n_pos: int = 8) -> str:
    names = list(res["variants"])
    lines = [f"### {label}: free-running span decoder (greedy), {res['n_slots']} val slots", "",
             "token accuracy at position p of the true next span (free-running; spans at least p "
             "long)", "",
             "| input | " + " | ".join(f"p{p}" for p in range(1, n_pos + 1)) +
             " | all | exact match | TF CE | TF top1 |",
             "|" + "---|" * (n_pos + 5)]
    for nm in names:
        r = res["variants"][nm]
        accs = [r["positions"].get(str(p), {}).get("free_acc") for p in range(1, n_pos + 1)]
        lines.append(f"| {nm} | " + " | ".join("n/a" if a is None else f"{a:.3f}" for a in accs) +
                     f" | {r['free_acc_all']:.3f} | {r['exact_match']:.4f} | "
                     f"{r['tf_ce_all']:.3f} | {r['tf_top1_all']:.3f} |")
    lines += ["", "teacher-forced CE per position (nats), same inputs", "",
              "| input | " + " | ".join(f"p{p}" for p in range(1, n_pos + 1)) + " |",
              "|" + "---|" * (n_pos + 1)]
    for nm in names:
        r = res["variants"][nm]
        ces = [r["positions"].get(str(p), {}).get("tf_ce") for p in range(1, n_pos + 1)]
        lines.append(f"| {nm} | " + " | ".join("n/a" if c is None else f"{c:.2f}" for c in ces) +
                     " |")
    return "\n".join(lines)


def spandec_examples(res: dict, decode: Callable[[list[int]], str], n: int = 8,
                     seed: int = 0) -> list[dict]:
    """``n`` slots with a next span of at least 4 tokens: the true span and each input's greedy
    decode, cut to the true length."""
    sub = res["inputs"]
    lens = sub.valid.sum(1)
    cand = (lens >= 4).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    pick = cand[torch.randperm(len(cand), generator=g)[:n]].tolist()
    out = []
    for i in pick:
        k = int(lens[i])
        ex = {"slot": i, "row": int(sub.row[i]), "true": decode(sub.ids[i, :k].tolist()),
              "decodes": {nm: decode(f[i, :k].tolist()) for nm, f in res["free"].items()}}
        out.append(ex)
    return out


def part_spandec(a) -> None:
    from transformers import AutoTokenizer
    device = a.device
    results = {}
    for spec in a.ckpt:
        label, cfg, model, step = load_model(spec, device)
        check_spandec_model(model)
        val, data = load_batches(a.batches_file, "val", a.n_val, device, model)
        depth = a.depths[-1]
        t0 = time.time()
        inp = collect_spandec_inputs(model, val, depth, device)
        print(f"[spandec] {label}: step {step}, depth {depth}, {inp.ids.shape[0]} slots with a "
              f"next span, read mode {model.cfg.tul.spandec_read} ({time.time() - t0:.0f}s)",
              flush=True)
        res = run_spandec(model, inp, device, a.max_slots, a.seed)
        print(spandec_markdown(label, res), flush=True)
        tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
        exs = spandec_examples(res, lambda ids: tok.decode(ids, skip_special_tokens=True), 8,
                               a.seed)
        for e in exs:
            print(f"TRUE: {e['true']!r}")
            for nm, t in e["decodes"].items():
                print(f"  {nm:8s}: {t!r}")
            print()
        results[label] = {"spec": spec, "step": step, "depth": depth,
                          "spandec_read": str(model.cfg.tul.spandec_read),
                          "n_slots": res["n_slots"], "variants": res["variants"],
                          "examples": exs, "val_skip_docs": data.get("val_skip_docs")}
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f_:
        json.dump(results, f_, indent=1)
    print(f"[spandec] wrote {a.out}", flush=True)


# ── part: geom (geometry of the cell readouts) ───────────────────────────────────────────
def _center(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return x - x.mean(0, keepdim=True)


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear CKA (Kornblith et al. 2019) of two `[N, C]` matrices over the same N rows, in the
    feature-space form ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F). 1 for X with itself or any
    rotation / isotropic scale of it."""
    xc, yc = _center(x), _center(y)
    num = (xc.t() @ yc).pow(2).sum()
    den = (xc.t() @ xc).pow(2).sum().sqrt() * (yc.t() @ yc).pow(2).sum().sqrt()
    return float(num / den)


def effective_rank(x: torch.Tensor) -> float:
    """exp(entropy of the normalised singular values) of the mean-centred `[N, C]` matrix
    (Roy and Vetterli 2007). The singular values come from the eigenvalues of the C x C Gram, so
    N may be large."""
    xc = _center(x)
    ev = torch.linalg.eigvalsh(xc.t() @ xc).clamp(min=0)
    sv = ev.sqrt()
    tot = sv.sum()
    if float(tot) <= 0:
        return 0.0
    p = sv / tot
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def within_slot_cosine(x: torch.Tensor) -> float:
    """Mean pairwise cosine between the M cells of a slot (raw vectors), averaged over slots.
    ``x [N, M, C]``."""
    xn = F.normalize(x.double(), dim=-1)
    g = xn @ xn.transpose(1, 2)
    m = g.shape[-1]
    off = (g.sum((1, 2)) - g.diagonal(dim1=1, dim2=2).sum(1)) / (m * (m - 1))
    return float(off.mean())


def geom_stats(feats: torch.Tensor) -> dict:
    """``feats [N, P, M, C]`` -> CKA between cells at each pass, CKA between passes for each
    cell, effective rank per (pass, cell) and pooled over cells per pass, within-slot cosine per
    pass. Matrices are plain nested lists."""
    n, p_n, m, c = feats.shape
    cells = {}                                   # centred matrices and self-norms, computed once
    for p in range(p_n):
        for i in range(m):
            xc = _center(feats[:, p, i])
            cells[p, i] = (xc, (xc.t() @ xc).pow(2).sum().sqrt())

    def cka(a, b):
        (xa, na), (xb, nb) = cells[a], cells[b]
        return float((xa.t() @ xb).pow(2).sum() / (na * nb))
    return {
        "cka_cells": [[[cka((p, i), (p, j)) for j in range(m)] for i in range(m)]
                      for p in range(p_n)],
        "cka_passes": [[[cka((p, i), (q, i)) for q in range(p_n)] for p in range(p_n)]
                       for i in range(m)],
        "erank_cells": [[effective_rank(feats[:, p, i]) for i in range(m)] for p in range(p_n)],
        "erank_pooled": [effective_rank(feats[:, p].reshape(n * m, c)) for p in range(p_n)],
        "cos_within": [within_slot_cosine(feats[:, p]) for p in range(p_n)]}


def shuffle_across_slots(feats: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Independent slot permutation for every (pass, cell): keeps each marginal matrix, breaks
    every pairing between cells and between passes. Effective rank of one (pass, cell) matrix is
    permutation invariant, so those entries cannot move; the pooled rank, the CKAs and the
    within-slot cosine can."""
    g = torch.Generator().manual_seed(seed)
    n, p_n, m, _ = feats.shape
    out = feats.clone()
    for p in range(p_n):
        for i in range(m):
            out[:, p, i] = feats[torch.randperm(n, generator=g), p, i]
    return out


def geom_report(feats: torch.Tensor, seed: int = 0) -> dict:
    """The real numbers and the two controls (shuffled across slots; iid Gaussian of the same
    shape)."""
    feats = feats.float()
    g = torch.Generator().manual_seed(seed + 1)
    return {"real": geom_stats(feats),
            "shuffled": geom_stats(shuffle_across_slots(feats, seed)),
            "noise": geom_stats(torch.randn(feats.shape, generator=g))}


def cka_between(fa: torch.Tensor, fb: torch.Tensor) -> list:
    """Per pass, the `[Ma, Mb]` linear CKA between checkpoint A's cells and checkpoint B's cells
    on the same slots. Cell identity is arbitrary across checkpoints, so the whole matrix is
    reported."""
    if fa.shape[0] != fb.shape[0] or fa.shape[1] != fb.shape[1] or fa.shape[3] != fb.shape[3]:
        raise ValueError(f"checkpoints disagree on slots / passes / width: {tuple(fa.shape)} vs "
                         f"{tuple(fb.shape)}; the batches file must be the same rows")
    return [[[linear_cka(fa[:, p, i], fb[:, p, j]) for j in range(fb.shape[2])]
             for i in range(fa.shape[2])] for p in range(fa.shape[1])]


def _mat(rows, fmt="{:.3f}") -> str:
    return "\n".join("  " + " ".join(fmt.format(v) for v in r) for r in rows)


def geom_markdown(label: str, passes: list[int], rep: dict) -> str:
    lines = [f"### {label}: geometry of the cell readouts (passes {passes}; 0 = loop entry)", ""]
    for src in ("real", "shuffled", "noise"):
        r = rep[src]
        lines.append(f"**{src}**")
        lines.append("")
        lines.append("| pass | within-slot cos | erank pooled | erank per cell | "
                     "CKA between cells (mean off-diagonal) |")
        lines.append("|---|---|---|---|---|")
        for pi, p in enumerate(passes):
            mat = r["cka_cells"][pi]
            m = len(mat)
            off = (sum(sum(row) for row in mat) - sum(mat[i][i] for i in range(m))) / (m * (m - 1))
            lines.append(f"| {p} | {r['cos_within'][pi]:.3f} | {r['erank_pooled'][pi]:.1f} | " +
                         " ".join(f"{v:.1f}" for v in r["erank_cells"][pi]) + f" | {off:.3f} |")
        lines.append("")
        if src == "real":
            for pi, p in enumerate(passes):
                lines.append(f"CKA between cells, pass {p}:")
                lines.append(_mat(r["cka_cells"][pi]))
            lines.append("")
            for ci in range(len(r["cka_passes"])):
                lines.append(f"CKA between passes {passes}, cell {ci}:")
                lines.append(_mat(r["cka_passes"][ci]))
            lines.append("")
    return "\n".join(lines)


def part_geom(a) -> None:
    device = a.device
    results, feats_by_label = {}, {}
    for spec in a.ckpt:
        label, cfg, model, step = load_model(spec, device)
        val, data = load_batches(a.batches_file, "val", a.n_val, device, model)
        t0 = time.time()
        f = collect_features(model, val, a.depths, 1, device, text_tokens=2,
                             store_dtype=torch.float32)
        n = f.feats.shape[0]
        sel = torch.randperm(n, generator=torch.Generator().manual_seed(a.seed))[:a.max_slots]
        feats = f.feats[sel.sort().values]
        print(f"[geom] {label}: step {step}; {n} slots, using {feats.shape[0]} "
              f"({time.time() - t0:.0f}s)", flush=True)
        rep = geom_report(feats, a.seed)
        print(geom_markdown(label, f.passes, rep), flush=True)
        results[label] = {"spec": spec, "step": step, "passes": f.passes,
                          "n_slots": int(feats.shape[0]), "geom": rep,
                          "val_skip_docs": data.get("val_skip_docs")}
        feats_by_label[label] = feats
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    labels = list(feats_by_label)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            key = f"{labels[i]}__vs__{labels[j]}"
            mats = cka_between(feats_by_label[labels[i]], feats_by_label[labels[j]])
            results.setdefault("_between", {})[key] = {"passes": results[labels[i]]["passes"],
                                                       "cka": mats}
            print(f"### CKA between checkpoints {key} (rows: {labels[i]} cells, columns: "
                  f"{labels[j]} cells)")
            for p, mat in zip(results[labels[i]]["passes"], mats):
                print(f"pass {p}:")
                print(_mat(mat))
    with open(a.out, "w") as f_:
        json.dump(results, f_, indent=1)
    print(f"[geom] wrote {a.out}", flush=True)


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["data", "patch", "probe", "special", "spandec", "geom"])
    ap.add_argument("--config", default="nowta_pointer_fast", help="data: the config to pack for")
    ap.add_argument("--ckpt", action="append",
                    help="patch/probe: LABEL=CONFIG=PATH[=ovr1,ovr2]; repeat for several")
    ap.add_argument("--batches-file", required=True)
    ap.add_argument("--train-skip", type=int, default=5000)
    ap.add_argument("--train-batches", type=int, default=64, help="data: train batches to pack")
    ap.add_argument("--val-batches", type=int, default=40, help="data: val batches to pack")
    ap.add_argument("--n-train", type=int, default=64, help="probe: train batches to use")
    ap.add_argument("--n-val", type=int, default=40, help="patch/probe: val batches to use")
    ap.add_argument("--depths", default="1,2,3,6")
    ap.add_argument("--offsets", type=int, default=4, help="probe: next-span tokens 1..N")
    ap.add_argument("--steps", type=int, default=3000, help="probe: AdamW steps")
    ap.add_argument("--probe-batch", type=int, default=128, help="probe: slots per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--patch-json", help="special: the JSON `--part patch` wrote")
    ap.add_argument("--special-depth", type=int, default=0,
                    help="special: the patch depth to read (default: the deepest in the JSON)")
    ap.add_argument("--perms", type=int, default=200, help="special: permutations for the NMI null")
    ap.add_argument("--max-slots", type=int, default=4000,
                    help="spandec / geom: random subset of val slots to score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.depths = [int(d) for d in a.depths.split(",")]
    if a.part == "data":
        import credit_matrix
        credit_matrix.part_data(a)
        return
    if a.part == "special":
        if not a.patch_json or not a.out:
            ap.error("--patch-json and --out are required for special")
        part_special(a)
        return
    if not a.ckpt or not a.out:
        ap.error(f"--ckpt and --out are required for {a.part}")
    {"patch": part_patch, "probe": part_probe, "spandec": part_spandec,
     "geom": part_geom}[a.part](a)


if __name__ == "__main__":
    main()
