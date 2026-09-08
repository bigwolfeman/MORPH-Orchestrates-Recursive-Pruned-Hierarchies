"""Depth sweep on Olympiad-AI held-out math (arc E15): token CE, ANSWER-region CE and
accuracy, and per-stage-band curves, at forced loop depth, paired over DOCUMENTS.

Rows come from the shards' ``eval_holdout.jsonl`` files (``input_ids`` + ``stage``), which
never entered a training shard (docs/olympiad-interop.md). Every checkpoint of every arm
scores the SAME documents (seeded interleave of the files, ``--docs`` of them), and the
per-document sums travel with the JSON, so any two arms or checkpoints pair offline.

Two model families, one script:

* a TUL arm (``tul.activate_at`` not ``never``): the packed slot layout, the forward of
  ``core_depth_sweep`` (``tul_forward_ablated`` at ``tul.slot_mean_depth = d``);
* the plain control (``notul``): plain ``seq_len + 1`` chunks of the same stream, the
  forward at ``model.cfg.mean_depth = d`` (``token_depth_sweep``'s lever).

The token → document map is recovered from the packed rows themselves and CHECKED: the
token positions of the packed rows, read in order, must equal the stream they were packed
from (the packer consumes tokens in order and peeks one label; ``pack_tul_batch``).

The answer region is the Olympiad ``<|A|> … <|/A|>`` block: a scored position belongs to
it when the token it PREDICTS lies after ``<|A|>`` up to and including ``<|/A|>``.

Usage:
  python lab/divergence/olympiad_sweep.py \
      --ckpt oly-mask=tul_oly_mask=/path/step_5000.pt \
      --holdout $OLYMPIAD_REPO/data/morph_shards/olympiad_stage11_13_v1/eval_holdout.jsonl \
      --holdout $OLYMPIAD_REPO/data/morph_shards/olympiad_math_2_10_v3/eval_holdout.jsonl \
      --docs 600 --depths 1,2,3,6,9,12,16 --batch 8 --out .../sweep.json
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

# Olympiad structure tokens (src/olympiad_data/core/tokens.py in Olympiad-AI; the 17
# specials sit at 49152+ on top of starcoder2's 49152). Only the answer block is read here.
ANSWER_OPEN = 49154
ANSWER_CLOSE = 49155
EOS = 0

MUX_METRICS = {"mux_local": ("mux_local", "mux_n_supervised")}


def load_docs(paths: list[str], n_docs: int, seed: int) -> list[dict]:
    """``n_docs`` held-out documents, interleaved across the files by a seeded shuffle."""
    docs: list[dict] = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                d = json.loads(line)
                ids = [int(t) for t in d["input_ids"]]
                if not ids:
                    continue
                if ids[-1] != EOS:
                    ids.append(EOS)
                stage = str(d.get("stage", "0"))
                docs.append({"ids": ids, "stage": stage, "band": int(stage.split(".")[0]),
                             "file": p})
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(docs))[:n_docs]
    return [docs[i] for i in order]


def build_stream(docs: list[dict]) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Token stream + per-token (doc id, band, in-answer flag)."""
    toks: list[int] = []
    doc_id: list[int] = []
    band: list[int] = []
    in_ans: list[bool] = []
    for i, d in enumerate(docs):
        inside = False
        for t in d["ids"]:
            flag = False
            if inside:
                flag = True                      # tokens after <|A|>, incl. <|/A|>
                if t == ANSWER_CLOSE:
                    inside = False
            if t == ANSWER_OPEN:
                inside = True
            toks.append(t)
            doc_id.append(i)
            band.append(d["band"])
            in_ans.append(flag)
    return toks, np.asarray(doc_id), np.asarray(band), np.asarray(in_ans)


@torch.no_grad()
def forward_maps(model, inp, labels, layout, device, want_mux: bool):
    """Per-position CE ``[B, L]`` and correctness ``[B, L]`` (argmax == label), plus the
    forecast stats from the labelled forward on a MUX arm."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        if layout is None:
            res = model(inp.to(device), labels=None)
        else:
            res = model.tul_forward_ablated(inp.to(device), None, layout, plan_mode="normal")
    logits = res["logits"].float()
    B, L, V = logits.shape
    lab = labels.to(device).clone()
    lab[lab < 0] = 0
    ce = F.cross_entropy(logits.reshape(B * L, V), lab.reshape(B * L),
                         reduction="none").reshape(B, L)
    correct = (logits.argmax(-1) == lab)
    stats: dict[str, float] = {}
    if want_mux and layout is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            res_l = model.tul_forward_ablated(inp.to(device), labels.to(device), layout,
                                              plan_mode="normal")
        for m, (vk, ck) in MUX_METRICS.items():
            if vk in res_l:
                stats[vk] = float(res_l[vk])
                stats[ck] = float(res_l.get(ck, 1.0))
    return ce.cpu(), correct.cpu(), stats


def pack_rows(stream: list[int], tul_rt, cfg, batch: int, plain: bool):
    """All batches once, with the stream index of every scored position.

    Returns ``[(inp, labels, layout_or_None, pos_index)]`` where ``pos_index`` is ``[B, L]``
    int64, the stream index of the INPUT token at each position (−1 at slot/pad positions).
    """
    from morph.model.tul_layout import pack_tul_batch

    out = []
    if plain:
        L = int(cfg.data.seq_len)
        c = 0
        while c + batch * L + 1 <= len(stream):
            ins, labs, idx = [], [], []
            for _ in range(batch):
                seg = stream[c:c + L + 1]
                ins.append(seg[:-1])
                labs.append(seg[1:])
                idx.append(list(range(c, c + L)))
                c += L
            out.append((torch.tensor(ins), torch.tensor(labs), None, torch.tensor(idx)))
        return out
    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    rule = tul_rt.data_cfg.rule
    buf = list(stream)
    cursor = 0
    need = batch * (spec.l_total + 1)
    while len(buf) >= need:
        before = len(buf)
        inp, labels, layout = pack_tul_batch(buf, rule, spec, batch)
        used = before - len(buf)
        tokpos = ~layout.slot_mask
        idx = torch.full(inp.shape, -1, dtype=torch.long)
        k = cursor
        for b in range(inp.shape[0]):
            ps = tokpos[b].nonzero().flatten().tolist()
            for p in ps:
                idx[b, p] = k
                k += 1
        if k - cursor != used:
            raise RuntimeError(f"packer consumed {used} tokens but the rows hold {k - cursor} "
                               "token positions; the stream map would be wrong")
        got = inp[tokpos].tolist()
        exp = stream[cursor:cursor + used]
        if got != exp:
            raise RuntimeError("packed token positions do not reproduce the stream in order")
        out.append((inp, labels, layout, idx))
        cursor += used
    return out


def _sums_by_doc(values: np.ndarray, units: np.ndarray, n_units: int) -> np.ndarray:
    return np.bincount(units, weights=values, minlength=n_units).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH[=OVR,..]")
    ap.add_argument("--holdout", action="append", required=True)
    ap.add_argument("--docs", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depths", default="1,2,3,6,9,12,16")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device
    depths = [int(x) for x in a.depths.split(",")]

    from morph.training.tul_setup import build_tul_runtime

    docs = load_docs(a.holdout, a.docs, a.seed)
    stream, doc_of, band_of, ans_of = build_stream(docs)
    n_docs = len(docs)
    bands = sorted(set(int(b) for b in band_of))
    print(f"holdout: {n_docs} docs, {len(stream)} tokens, bands {bands}, "
          f"answer tokens {int(ans_of.sum())}", flush=True)

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        parts = triple.split("=", 3)
        label, config, path = parts[0], parts[1], parts[2]
        ovr = parts[3].split(",") if len(parts) == 4 and parts[3] else []
        cfg = build_cfg(config, ["model.use_kernels=false", *ovr])
        tul_rt = build_tul_runtime(cfg)
        plain = tul_rt is None
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, None if plain else tul_rt.model_cfg)
        model.eval()
        batches = pack_rows(stream, tul_rt, cfg, a.batch, plain)
        # scored positions: token position with a valid label; label = stream[idx + 1]
        scored = []
        for inp, labels, layout, idx in batches:
            ok = (idx >= 0) & (labels >= 0)
            lab_idx = torch.where(ok, idx + 1, torch.zeros_like(idx))
            lab_idx = lab_idx.clamp(max=len(stream) - 1)
            ok &= lab_idx < len(stream)
            unit = torch.from_numpy(doc_of)[lab_idx]
            bnd = torch.from_numpy(band_of)[lab_idx]
            ans = torch.from_numpy(ans_of)[lab_idx] & ok
            first = torch.zeros_like(ok)
            if layout is not None:
                dump = tul_rt.data_cfg.spec_for(cfg.data.seq_len).max_slots
                for b in range(inp.shape[0]):
                    seen: set[int] = set()
                    for p in ok[b].nonzero().flatten().tolist():
                        bag = int(layout.bag_id[b, p])
                        if bag not in seen:
                            seen.add(bag)
                            if 0 < bag < dump:
                                first[b, p] = True
            scored.append((ok, unit, bnd, ans, first))
        # the layouts served the CPU-side masks above; the forward wants them on the device
        batches = [(inp, labels, (lay.to(device) if lay is not None else None), idx)
                   for inp, labels, lay, idx in batches]
        n_scored = sum(int(s[0].sum()) for s in scored)
        n_ans = sum(int(s[3].sum()) for s in scored)
        print(f"{label}: {len(batches)} batches of {a.batch}, {n_scored} scored positions, "
              f"{n_ans} in the answer region", flush=True)
        has_mux = (not plain) and float(model.cfg.tul.mux_beta) > 0.0
        if plain:
            orig = (int(model.cfg.mean_depth),)
        else:
            orig = (int(model.cfg.tul.slot_mean_depth), int(model.cfg.tul.slot_max_depth))
        arm = {"step": step, "docs": n_docs, "batch": a.batch, "plain": plain,
               "train_eval_depth": orig[0] or int(cfg.model.mean_depth), "seed": a.seed,
               "holdout": a.holdout, "depths": {}, "bands": bands}
        # per-depth, per-doc sums
        tok_sum: dict[int, np.ndarray] = {}
        ans_sum: dict[int, np.ndarray] = {}
        acc_sum: dict[int, np.ndarray] = {}
        tok_cnt = ans_cnt = None
        mux_sum: dict[int, np.ndarray] = {}
        mux_cnt = None
        try:
            for d in depths:
                if plain:
                    model.cfg.mean_depth = d
                else:
                    model.cfg.tul.slot_mean_depth = d
                    model.cfg.tul.slot_max_depth = max(d, orig[1] or int(cfg.model.max_depth))
                ts, tc = np.zeros(n_docs), np.zeros(n_docs)
                as_, ac, cs = np.zeros(n_docs), np.zeros(n_docs), np.zeros(n_docs)
                bs_ce = {b: [0.0, 0.0] for b in bands}
                bs_acc = {b: 0.0 for b in bands}
                fst = fst_n = 0.0
                ms, mc = [], []
                for (inp, labels, layout, idx), (ok, unit, bnd, ans, first) in zip(batches, scored):
                    ce, correct, stats = forward_maps(model, inp, labels, layout, device, has_mux)
                    okf = ok.numpy().reshape(-1)
                    cef = ce.numpy().reshape(-1)[okf]
                    cof = correct.numpy().reshape(-1)[okf].astype(np.float64)
                    u = unit.numpy().reshape(-1)[okf]
                    af = ans.numpy().reshape(-1)[okf]
                    bf = bnd.numpy().reshape(-1)[okf]
                    ts += _sums_by_doc(cef, u, n_docs)
                    tc += _sums_by_doc(np.ones_like(cef), u, n_docs)
                    as_ += _sums_by_doc(cef * af, u, n_docs)
                    ac += _sums_by_doc(af.astype(np.float64), u, n_docs)
                    cs += _sums_by_doc(cof * af, u, n_docs)
                    for b in bands:
                        sel = af & (bf == b)
                        bs_ce[b][0] += float(cef[sel].sum())
                        bs_ce[b][1] += float(sel.sum())
                        bs_acc[b] += float(cof[sel].sum())
                    fst += float(ce[first].sum())
                    fst_n += float(first.sum())
                    if has_mux and "mux_local" in stats:
                        ms.append(stats["mux_local"] * stats["mux_n_supervised"])
                        mc.append(stats["mux_n_supervised"])
                tok_sum[d], ans_sum[d], acc_sum[d] = ts, as_, cs
                if tok_cnt is None:
                    tok_cnt, ans_cnt = tc, ac
                entry = {"ce_tokens": float(ts.sum() / tc.sum()),
                         "ce_answer": float(as_.sum() / ac.sum()),
                         "acc_answer": float(cs.sum() / ac.sum()),
                         "n_tokens": float(tc.sum()), "n_answer": float(ac.sum()),
                         "band_ce_answer": {str(b): (bs_ce[b][0] / bs_ce[b][1] if bs_ce[b][1] else None) for b in bands},
                         "band_acc_answer": {str(b): (bs_acc[b] / bs_ce[b][1] if bs_ce[b][1] else None) for b in bands},
                         "band_n_answer": {str(b): bs_ce[b][1] for b in bands}}
                if not plain:
                    entry["ce_span_first"] = float(fst / fst_n) if fst_n else None
                if ms:
                    mux_sum[d] = np.asarray(ms)
                    if mux_cnt is None:
                        mux_cnt = np.asarray(mc)
                    entry["mux_local"] = float(np.sum(ms) / np.sum(mc))
                arm["depths"][d] = entry
                print(f"{label:10s} depth={d:<2d} ce_tokens={entry['ce_tokens']:.4f} "
                      f"ce_answer={entry['ce_answer']:.4f} acc_answer={entry['acc_answer']:.4f}"
                      + (f" span_first={entry['ce_span_first']:.4f}" if entry.get("ce_span_first") else "")
                      + (f" mux_local={entry['mux_local']:.4f}" if "mux_local" in entry else ""),
                      flush=True)
        finally:
            if plain:
                model.cfg.mean_depth = orig[0]
            else:
                model.cfg.tul.slot_mean_depth, model.cfg.tul.slot_max_depth = orig
        pairs = [(1, 6), (3, 6), (6, 12), (1, max(depths))]
        def cis(sums, cnt):
            o = {}
            for x, y in pairs:
                if x in sums and y in sums and x != y:
                    o[f"K{x}-K{y}"] = paired_bootstrap_ci(sums[x], sums[y], cnt)
            return o
        arm["ci_ce_tokens"] = cis(tok_sum, tok_cnt)
        arm["ci_ce_answer"] = cis(ans_sum, ans_cnt)
        arm["ci_acc_answer"] = cis(acc_sum, ans_cnt)
        if mux_sum:
            arm["ci_mux_local"] = cis(mux_sum, mux_cnt)
        for key in ("ci_ce_tokens", "ci_ce_answer", "ci_acc_answer", "ci_mux_local"):
            for k, v in arm.get(key, {}).items():
                print(f"{label:10s} {key[3:]} {k}: {v['point']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}] "
                      f"over {v['n_units']} units", flush=True)
        arm["doc_ce_sum"] = {str(d): tok_sum[d].tolist() for d in depths}
        arm["doc_n_tokens"] = tok_cnt.tolist()
        arm["doc_ans_ce_sum"] = {str(d): ans_sum[d].tolist() for d in depths}
        arm["doc_ans_correct"] = {str(d): acc_sum[d].tolist() for d in depths}
        arm["doc_n_answer"] = ans_cnt.tolist()
        arm["doc_band"] = [d["band"] for d in docs]
        arm["doc_stage"] = [d["stage"] for d in docs]
        results[label] = arm
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
