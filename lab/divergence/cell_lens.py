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
            rec = []
            orig_acc = T.accumulate_span_ce

            def acc(*aa, **kk):
                r = orig_acc(*aa, **kk)
                rec.append(r[:, 1:].detach().float())
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
    return {"ce": tap.ce, "mixed": tap.mixed, "zero": zero, "ok": tap.ok,
            "n_tok": tap.n_tok, "out": out, "xh_all": tap.xh0, "xh_zero": ztap.xh0}


def summarise_batch(tb: dict) -> dict:
    """Per-row sums over the scored spans of one batch (the paired-bootstrap units)."""
    ok = tb["ok"]
    okf = ok.float()
    ce, mixed, zero = tb["ce"], tb["mixed"], tb["zero"]
    best, arg = ce.min(dim=-1)
    cnt = (tb["n_tok"].float() * okf).sum(1)
    kk = ce.shape[-1]
    wins = torch.stack([((arg == i) & ok).sum() for i in range(kk)]).double()
    return {"cell": (ce * okf.unsqueeze(-1)).sum(1).double().cpu().numpy(),       # [B, K]
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


def run_patch(model, val, depths: list[int], device: str, log=print) -> dict:
    """The patch table per depth over ``val`` ([(x, y, layout)])."""
    check_depths(model, depths)
    res = {}
    for d in depths:
        t0 = time.time()
        sums: list[dict] = []
        rt = {k: [0.0, 0.0] for k in ROUTER_KEYS}
        for bi, (x, y, lay) in enumerate(val):
            tb = patch_batch(model, x, y, lay, d, device)
            sm = summarise_batch(tb)
            check_anchor(tb, sm, f"depth {d} batch {bi}")
            sums.append(sm)
            for k in ROUTER_KEYS:
                if f"fan_{k}" in tb["out"]:
                    rt[k][0] += float(tb["out"][f"fan_{k}"]) * float(sm["cnt"].sum())
                    rt[k][1] += float(sm["cnt"].sum())
        cat = {k: np.concatenate([s[k] for s in sums]) for k in ("cell", "all", "zero", "best", "cnt")}
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
        results[label] = {"spec": spec, "step": step, "rows": rows,
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
    global row of each slot."""
    feats: torch.Tensor
    tgt: torch.Tensor
    text: torch.Tensor
    row: torch.Tensor
    passes: list[int]


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
    fs, tg, tx, rw = [], [], [], []
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
        fs.append(cr[ok].to(store_dtype))
        tg.append(tok[ok][:, :n_off])
        tx.append(tok[ok])
        rw.append(rid[ok])
        row0 += B
    return Features(torch.cat(fs), torch.cat(tg), torch.cat(tx), torch.cat(rw), passes)


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
    row whenever the set spans >= 2 rows); its same-row fraction is reported."""
    n, p_n, m, c = f.feats.shape
    n_off = f.tgt.shape[1]
    perm = (torch.arange(n) + max(1, n // 2)) % n
    same_row = float((f.row == f.row[perm]).float().mean())
    names = [f"cell{i}" for i in range(m)] + ["mean", "shuffled"]
    acc = np.zeros((p_n, n_off, len(names), 3))                      # ce, top1, top5
    cnt = np.zeros(n_off)
    uni_ce = np.zeros((n_off, 3))
    for s in range(0, n, chunk):
        idx = torch.arange(s, min(n, s + chunk))
        tgt = f.tgt[idx].to(device)
        x = f.feats[idx].to(device).float()                           # [b, P, M, C]
        xs = f.feats[perm[idx]].to(device).float()
        for k in range(n_off):
            valid = tgt[:, k] != IGNORE
            nv = int(valid.sum())
            if nv == 0:
                continue
            cnt[k] += nv
            lab = tgt[:, k][valid]
            pu = uni["p"][k].to(device)
            top5u = pu.topk(5).indices
            uni_ce[k, 0] += float(-pu.log()[lab].sum())
            uni_ce[k, 1] += float((lab == top5u[0]).sum())
            uni_ce[k, 2] += float((lab.unsqueeze(1) == top5u.unsqueeze(0)).any(1).sum())
            for p in range(p_n):
                variants = [x[valid][:, p, i] for i in range(m)]
                variants.append(x[valid][:, p].mean(dim=1))
                variants.append(xs[valid][:, p, :].reshape(nv, m, c))   # shuffled: per cell
                for j, v in enumerate(variants):
                    if j == len(names) - 1:                             # shuffled, all cells
                        vv = v.reshape(nv * m, c)
                        ll = lab.repeat_interleave(m)
                    else:
                        vv, ll = v, lab
                    lg = probes.logits(vv, p, k).float()
                    ce = F.cross_entropy(lg, ll, reduction="none")
                    t5 = lg.topk(5, dim=-1).indices
                    acc[p, k, j, 0] += float(ce.sum()) / (m if j == len(names) - 1 else 1)
                    acc[p, k, j, 1] += float((t5[:, 0] == ll).sum()) / (m if j == len(names) - 1 else 1)
                    acc[p, k, j, 2] += float((t5 == ll.unsqueeze(1)).any(1).sum()) / (
                        m if j == len(names) - 1 else 1)
    res = {"names": names, "passes": f.passes, "n_scored": cnt.tolist(), "same_row_frac": same_row,
           "probe": {}, "unigram": {}}
    for k in range(n_off):
        res["unigram"][f"off{k + 1}"] = {"ce": uni_ce[k, 0] / cnt[k], "top1": uni_ce[k, 1] / cnt[k],
                                         "top5": uni_ce[k, 2] / cnt[k]}
        for p in range(p_n):
            res["probe"][f"pass{f.passes[p]}_off{k + 1}"] = {
                nm: {"ce": acc[p, k, j, 0] / cnt[k], "top1": acc[p, k, j, 1] / cnt[k],
                     "top5": acc[p, k, j, 2] / cnt[k]} for j, nm in enumerate(names)}
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


# ── main ─────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["data", "patch", "probe"])
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()
    a.depths = [int(d) for d in a.depths.split(",")]
    if a.part == "data":
        import credit_matrix
        credit_matrix.part_data(a)
        return
    if not a.ckpt or not a.out:
        ap.error("--ckpt and --out are required for patch / probe")
    {"patch": part_patch, "probe": part_probe}[a.part](a)


if __name__ == "__main__":
    main()
