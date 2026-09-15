"""Does the boundary rule and the TUL-Code encoder treat an ellipsis like a period?

Wolfe's question: an ellipsis ("...", or the single Unicode character "…") is a
different speech act than a period — it elicits introspection, and recursive models are
said to lean on it more. The boundary rule (``morph/model/tul_layout.py``) cuts a span
after a token whose decoded text ends in one of ``.;!?`` or CONTAINS one of
``\\n — – --`` (``BOUNDARY_SUFFIX_CHARS`` / ``BOUNDARY_SUBSTRINGS``). Read literally, a
token that decodes to the single character "…" ends in "…", not in ".", so the
suffix rule does NOT fire for it, and it contains none of the substrings either — it
would never independently end a span. A token whose text is "..." (three ASCII periods)
DOES end in "." and fires the ordinary period rule. This script does not assume either
way; it reads the actual tokenizer's vocabulary and the actual boundary decisions on
packed validation rows, then asks whether the encoder's codes for ellipsis-terminated
spans are collapsed relative to a period-terminated control, and whether the coda reads
them as well.

Three measurements, read-only, no repo file is modified:

  1. TOKENIZATION AND CUTTING: how "..." / "…" tokenize, and a boundary-class
     frequency table (per 1000 real span boundaries) over the packed validation rows.
  2. COLLAPSE TEST: for each valid slot s (`code_target_valid`), group the encoder's
     code of span s+1 by the terminator class of span s+1 (grouping a) and of span s
     (grouping b, the PRECEDING span). Per group: within-group cosine, effective rank
     (participation ratio), cosine to centroid — each against a size-matched control
     drawn from the same axis's "." pool (5 draws averaged).
  3. READABILITY BY GROUP: the coda's token CE on span s+1 with the true code
     (`code_mode="encoder"`) and a sampled code (`code_mode="sampled"`), plus the
     sampler's residual energy, per group of both axes.

    python lab/divergence/code_ellipsis_probe.py \\
        --ckpt tul-code=tul_code=/path/step_5000.pt --rows 192 --batch 3 --out .../probe.json
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

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_code import code_target_valid  # noqa: E402

# ── boundary-class taxonomy ──────────────────────────────────────────────────
# Priority order matters: a composite token (e.g. one containing both "…" and "\n")
# is filed under the class this probe cares about isolating (ellipsis) first.
CLASSES = ["ellipsis_char", "ellipsis_dots", "newline", "dash", "question", "exclaim",
           "semicolon", "period", "eos", "other"]


def classify(s: str) -> str:
    if "…" in s:
        return "ellipsis_char"
    if "..." in s:
        return "ellipsis_dots"
    if "\n" in s:
        return "newline"
    if any(d in s for d in ("—", "–", "--")):
        return "dash"
    stripped = s.rstrip()
    if stripped.endswith("?"):
        return "question"
    if stripped.endswith("!"):
        return "exclaim"
    if stripped.endswith(";"):
        return "semicolon"
    if stripped.endswith("."):
        return "period"
    return "other"


def build_class_lut(tok, vocab_size: int, eos_id: int) -> tuple[np.ndarray, list[str]]:
    """``[vocab_size]`` array of class-index-into-CLASSES, and the raw decoded strings."""
    n_tok = min(int(len(tok)), vocab_size)
    strings = tok.batch_decode([[i] for i in range(n_tok)]) + [""] * (vocab_size - n_tok)
    cls_idx = np.full(vocab_size, CLASSES.index("other"), dtype=np.int64)
    for i, s in enumerate(strings[:n_tok]):
        cls_idx[i] = CLASSES.index(classify(s))
    if 0 <= eos_id < vocab_size:
        cls_idx[eos_id] = CLASSES.index("eos")
    return cls_idx, strings


@torch.no_grad()
def _forward(model, inp, layout, device, code_mode, code_steps=None):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal",
                                        code_mode=code_mode, code_steps=code_steps)
    assert "code_cells" in res, "the eval forward did not expose code_cells (not a code model?)"
    return res["logits"].float(), res["code_cells"].float()


def _token_ce_full(logits, labels, layout, device):
    """Per-position CE and the keep mask, both ``[B, L]`` — no reduction."""
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    keep = (lab >= 0) & (~layout.slot_mask)
    lab_safe = lab.clone()
    lab_safe[lab_safe < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab_safe.reshape(B * L),
                         reduction="none").reshape(B, L)
    return ce, keep


def row_terminators(bag_row: np.ndarray, id_row: np.ndarray, n_real_slots: int):
    """``(term_id[bag], length[bag])`` for ``bag`` in ``[0, n_real_slots)``.

    ``bag_row`` / ``id_row`` are the token positions of ONE row already restricted to
    non-slot positions (``~slot_mask``), in sequence order. The terminator of span
    ``bag`` is the LAST token (max position) carrying that bag id — the token whose text
    is what caused the boundary rule to cut after it (or EOS, or the length cap).
    """
    if bag_row.size == 0:
        return {}, {}
    rev = bag_row[::-1]
    uniq, first_idx_rev = np.unique(rev, return_index=True)
    last_pos = bag_row.size - 1 - first_idx_rev
    term_id = {int(b): int(id_row[p]) for b, p in zip(uniq.tolist(), last_pos.tolist())
              if 0 <= int(b) < n_real_slots}
    counts = np.bincount(bag_row, minlength=n_real_slots)
    length = {b: int(counts[b]) for b in range(n_real_slots) if b < counts.shape[0]}
    return term_id, length


def group_stats(Xg: torch.Tensor, seed: int = 1234, cap: int = 400):
    """Within-group cosine, participation-ratio effective rank, cosine-to-centroid.

    ``Xg`` is ``[n, C]``, already centered by the GLOBAL mean (shared across groups).
    Subsampled once to ``cap`` and every stat below is computed on that SAME draw.
    """
    n = Xg.shape[0]
    if n < 2:
        return {"n": n, "within_cos": float("nan"), "eff_rank": float("nan"),
                "centroid_cos": float("nan")}
    finite = torch.isfinite(Xg).all(dim=-1)
    if not bool(finite.all()):
        print(f"    [group_stats] dropping {int((~finite).sum())} of {n} rows with non-finite "
              "codes", flush=True)
        Xg = Xg[finite]
        n = Xg.shape[0]
        if n < 2:
            return {"n": n, "within_cos": float("nan"), "eff_rank": float("nan"),
                    "centroid_cos": float("nan")}
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:cap]
    sub = Xg[idx].double()
    normed = F.normalize(sub, dim=-1)
    sim = normed @ normed.T
    iu = torch.triu_indices(sub.shape[0], sub.shape[0], offset=1)
    within = float(sim[iu[0], iu[1]].mean())
    centroid = sub.mean(0)
    cn = F.normalize(centroid.unsqueeze(0), dim=-1)
    cent_cos = float((normed @ cn.T).mean())
    Xc = sub - centroid
    # eigenvalues of the n x n Gram matrix (n <= cap): the same spectrum as the SVD of Xc,
    # without LAPACK's DLASCL failure on ill-scaled inputs
    var = torch.linalg.eigvalsh(Xc @ Xc.T).clamp_min(0.0)
    vs = float(var.sum())
    er = float(var.sum() ** 2 / (var ** 2).sum()) if vs > 1e-12 else 0.0
    return {"n": n, "within_cos": within, "eff_rank": er, "centroid_cos": cent_cos}


def control_stats(Xg_pool: torch.Tensor, group_n: int, draws: int = 5, base_seed: int = 9000):
    """``draws`` size-matched random groups from ``Xg_pool`` (the axis's "." pool), averaged."""
    pool_n = Xg_pool.shape[0]
    if pool_n == 0 or group_n == 0:
        return {"n": group_n, "within_cos": float("nan"), "eff_rank": float("nan"),
                "centroid_cos": float("nan"), "pool_n": pool_n, "replacement": None}
    replace = pool_n < group_n
    accs = {"within_cos": [], "eff_rank": [], "centroid_cos": []}
    for d in range(draws):
        g = np.random.default_rng(base_seed + d)
        idx = g.choice(pool_n, size=group_n, replace=replace)
        s = group_stats(Xg_pool[idx], seed=base_seed + d)
        for k in accs:
            accs[k].append(s[k])
    return {"n": group_n, "pool_n": pool_n, "replacement": replace,
           **{k: float(np.nanmean(v)) for k, v in accs.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LABEL=CONFIG=PATH[=k=v,k=v]")
    ap.add_argument("--rows", type=int, default=192)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--steps", type=int, default=8, help="sampler steps for code_mode=sampled")
    ap.add_argument("--draws", type=int, default=5, help="control draws averaged")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from transformers import AutoTokenizer
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    parts = a.ckpt.split("=", 3)
    label, config, path = parts[0], parts[1], parts[2]
    ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
    cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
    tul_rt = build_tul_runtime(cfg)
    assert tul_rt is not None and bool(cfg.tul.code), "a TUL-Code checkpoint is required"
    model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                            device, tul_rt.model_cfg)
    model.eval()
    enc = model.tul_code_enc
    M = int(enc.m)
    C = int(model.tul_code_cell.shape[1])

    rule = tul_rt.data_cfg.rule
    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer)
    vocab_size = int(cfg.model.vocab_size)
    class_lut, strings = build_class_lut(tok, vocab_size, rule.eos_id)

    out = {"label": label, "step": step, "M": M, "C": C, "eos_id": int(rule.eos_id),
          "min_span": int(rule.min_span), "span_cap": int(rule.span_cap)}

    # ── §1a: tokenization facts ──────────────────────────────────────────────
    print("=== §1a tokenization of ellipsis-like strings ===", flush=True)
    tok_examples = {}
    for probe_str in ["...", " ...", "…", " …", "she said...", "wait…what?",
                      "well...", "...\n", "Really...I", "etc."]:
        ids = tok.encode(probe_str, add_special_tokens=False)
        pieces = [(i, repr(tok.decode([i])), CLASSES[int(class_lut[i])],
                  bool(rule.is_boundary[i]) if i < len(rule.is_boundary) else None)
                 for i in ids]
        tok_examples[probe_str] = pieces
        print(f"  {probe_str!r:20s} -> {pieces}", flush=True)
    out["tokenization_examples"] = {k: [[int(i), s, c, bnd] for i, s, c, bnd in v]
                                    for k, v in tok_examples.items()}

    ell_ids = [i for i in range(min(len(tok), vocab_size))
              if CLASSES[int(class_lut[i])] in ("ellipsis_char", "ellipsis_dots")]
    print(f"  {len(ell_ids)} vocab ids classify as ellipsis_char/ellipsis_dots; "
         f"first 15: {[(i, repr(strings[i]), bool(rule.is_boundary[i])) for i in ell_ids[:15]]}",
         flush=True)
    out["ellipsis_vocab_ids"] = [[i, strings[i], bool(rule.is_boundary[i])] for i in ell_ids]

    # ── pack validation rows ─────────────────────────────────────────────────
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    row_tokens = tul_rt.data_cfg.spec_for(cfg.data.seq_len).l_total + 1
    stream = stream_from_loader(loader, a.rows * row_tokens)
    batches = pack_rows(stream, tul_rt, cfg, a.batch, False)[: -(-a.rows // a.batch)]
    batches = [(inp, labels, lay.to(device)) for inp, labels, lay, _ in batches]
    n_rows = sum(inp.shape[0] for inp, _, _ in batches)
    print(f"{label}: step {step}, {n_rows} rows, M={M}, C={C}", flush=True)

    # ── §1b boundary-class frequency, §2/§3 per-valid-slot data ────────────────
    boundary_counts = {c: 0 for c in CLASSES}
    Z_list, Zs_list, cls_a_list, cls_b_list, len_list = [], [], [], [], []
    ce_e_sum_list, ce_e_n_list, ce_s_sum_list, ce_s_n_list = [], [], [], []
    n_boundaries_total = 0

    for bi, (inp, labels, lay) in enumerate(batches):
        lg_e, cells_e = _forward(model, inp, lay, device, "encoder")
        lg_s, cells_s = _forward(model, inp, lay, device, "sampled", a.steps)
        ok = code_target_valid(lay).cpu().numpy()  # [B, S]
        ce_e, keep_e = _token_ce_full(lg_e, labels, lay, device)
        ce_s, keep_s = _token_ce_full(lg_s, labels, lay, device)
        ce_e_np, keep_np = ce_e.cpu().numpy(), keep_e.cpu().numpy()
        ce_s_np = ce_s.cpu().numpy()
        bag_full = lay.bag_id.cpu().numpy()
        id_full = inp.cpu().numpy()
        slot_mask_np = lay.slot_mask.cpu().numpy()
        slot_valid_np = lay.slot_valid.cpu().numpy()
        cells_e_cpu = cells_e.cpu()
        cells_s_cpu = cells_s.cpu()

        B = inp.shape[0]
        for b in range(B):
            n_real = int(slot_valid_np[b].sum())
            tokpos = ~slot_mask_np[b]
            bag_row = bag_full[b][tokpos]
            id_row = id_full[b][tokpos]
            term_id, length = row_terminators(bag_row, id_row, n_real)
            for bag in range(n_real):
                if bag in term_id:
                    boundary_counts[CLASSES[int(class_lut[term_id[bag]])]] += 1
                    n_boundaries_total += 1
            S = ok.shape[1]
            for s in range(min(S - 1, n_real - 1)):
                if not ok[b, s]:
                    continue
                if s not in term_id or (s + 1) not in term_id:
                    continue
                cls_a = CLASSES[int(class_lut[term_id[s + 1]])]   # terminator of span s+1 (encoded)
                cls_b = CLASSES[int(class_lut[term_id[s]])]       # terminator of span s (preceding)
                pos_mask = (bag_full[b] == (s + 1)) & keep_np[b]
                n_tok_s1 = int(pos_mask.sum())
                if n_tok_s1 == 0:
                    continue
                Z_list.append(cells_e_cpu[b, s])       # [M, C]
                Zs_list.append(cells_s_cpu[b, s])
                cls_a_list.append(cls_a)
                cls_b_list.append(cls_b)
                len_list.append(length.get(s + 1, n_tok_s1))
                ce_e_sum_list.append(float(ce_e_np[b][pos_mask].sum()))
                ce_e_n_list.append(n_tok_s1)
                ce_s_sum_list.append(float(ce_s_np[b][pos_mask].sum()))
                ce_s_n_list.append(n_tok_s1)
        if (bi + 1) % 8 == 0 or bi == len(batches) - 1:
            print(f"  batch {bi + 1}/{len(batches)}: {len(Z_list)} valid-slot entries so far",
                 flush=True)

    N = len(Z_list)
    print(f"N={N} valid slots with a non-empty span s+1; {n_boundaries_total} total boundaries",
         flush=True)
    Z = torch.stack(Z_list)    # [N, M, C]
    Zs = torch.stack(Zs_list)
    cls_a_arr = np.array(cls_a_list)
    cls_b_arr = np.array(cls_b_list)
    len_arr = np.array(len_list)
    ce_e_sum = np.array(ce_e_sum_list)
    ce_e_n = np.array(ce_e_n_list)
    ce_s_sum = np.array(ce_s_sum_list)
    ce_s_n = np.array(ce_s_n_list)

    print("\n=== §1b boundary-class frequency (per 1000 real span boundaries) ===", flush=True)
    freq_table = {}
    for c in CLASSES:
        cnt = boundary_counts[c]
        per1000 = 1000.0 * cnt / max(n_boundaries_total, 1)
        freq_table[c] = {"count": cnt, "per_1000": per1000}
        print(f"  {c:15s} count={cnt:6d}  per_1000={per1000:8.2f}", flush=True)
    out["boundary_class_frequency"] = freq_table
    out["n_boundaries_total"] = n_boundaries_total
    out["n_valid_slot_pairs"] = N

    # ── §2 collapse test + §3 readability, per axis ─────────────────────────
    global_mean = [Z[:, m].mean(0) for m in range(M)]  # [C] each
    Zc = torch.stack([Z[:, m] - global_mean[m] for m in range(M)], dim=1)  # [N, M, C]

    collapse_out = {"a": {}, "b": {}}
    readability_out = {"a": {}, "b": {}}
    axes = {"a": cls_a_arr, "b": cls_b_arr}
    axis_desc = {"a": "terminator of span s+1 (the span the code ENCODES)",
                "b": "terminator of span s (the PRECEDING span)"}

    for axis, arr in axes.items():
        print(f"\n=== §2/§3 grouping ({axis}): {axis_desc[axis]} ===", flush=True)
        present = sorted(set(arr.tolist()), key=lambda c: -int((arr == c).sum()))
        period_idx = np.flatnonzero(arr == "period")
        for cls in present:
            idx = np.flatnonzero(arr == cls)
            n = idx.shape[0]
            per_cell = {}
            ctrl_per_cell = {}
            for m in range(M):
                Xg = Zc[idx, m]
                stats = group_stats(Xg)
                per_cell[m] = stats
                pool = Zc[period_idx, m]
                ctrl = control_stats(pool, n, draws=a.draws)
                ctrl_per_cell[m] = ctrl
                print(f"  [{axis}] {cls:15s} cell{m} n={n:5d}  "
                     f"within_cos={stats['within_cos']:+.4f} eff_rank={stats['eff_rank']:7.2f} "
                     f"centroid_cos={stats['centroid_cos']:+.4f}  || control(.): "
                     f"within_cos={ctrl['within_cos']:+.4f} eff_rank={ctrl['eff_rank']:7.2f} "
                     f"centroid_cos={ctrl['centroid_cos']:+.4f}", flush=True)
            collapse_out[axis][cls] = {"n": n, "cells": per_cell, "control_cells": ctrl_per_cell}

            # readability
            ce_tf = float(ce_e_sum[idx].sum() / max(ce_e_n[idx].sum(), 1))
            ce_smp = float(ce_s_sum[idx].sum() / max(ce_s_n[idx].sum(), 1))
            resid = {}
            for m in range(M):
                R = (Zs[idx, m] - Z[idx, m]).double()
                zc = Zc[idx, m].double()
                r_energy = float(R.pow(2).sum(-1).mean())
                z_energy = float(zc.pow(2).sum(-1).mean())
                resid[m] = r_energy / max(z_energy, 1e-12)
            readability_out[axis][cls] = {"n": n, "ce_tf": ce_tf, "ce_sampled": ce_smp,
                                          "resid_ratio_cells": resid,
                                          "mean_span_len": float(len_arr[idx].mean())
                                          if n else float("nan")}
            print(f"  [{axis}] {cls:15s} ce_tf={ce_tf:.4f} ce_sampled={ce_smp:.4f} "
                 f"resid_ratio={resid} mean_len={readability_out[axis][cls]['mean_span_len']:.2f}",
                 flush=True)

    out["collapse_test"] = collapse_out
    out["readability"] = readability_out
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
