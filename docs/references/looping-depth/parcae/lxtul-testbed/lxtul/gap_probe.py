"""Where does the CE gap live? Per-token NLL of a plain run vs an LXTUL run, bucketed.

    python -m lxtul.gap_probe --plain /home/wolfe/parcae-runs/plain-5k \
        --arm /home/wolfe/parcae-runs/lxtul-5k --out gap.json

Both checkpoints score the same 480 held-out rows (`evaluate.eval_rows`) at depth 6; the plain
model reads the same tokens with slots removed, so token i of a row is the same (input, target)
pair in both. Buckets, each with count, plain NLL, arm NLL, gap and share of the total gap:

  pos_in_span   0 = the first token after a slot (its own span holds only itself) ... 16+
  span_index    which span of the row the token sits in (0 = the row's first span)
  copy          target id seen earlier in an EARLIER span ("cross_span"), only earlier in its
                own span ("own_span"), or not earlier in the row ("novel")
  bigram        (previous token, target) seen earlier in an earlier span ("far_bigram") or not

For the LXTUL arm it also reports, per bucket, the arm's NLL with every cell write zeroed
(`W_prefix` = 0, what the loop's write is worth there) and at depth 1 (where K1-K6 lands).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from lxtul.build import build
from lxtul.data import morph_cfg, tul_runtime
from lxtul.evaluate import eval_rows, plain_view


def load_run(run: Path):
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ck["config"])
    kind = str(cfg.arm.kind)
    ov = OmegaConf.to_container(cfg.arm.lxtul, resolve=True) if "lxtul" in cfg.arm else {}
    model, mcfg, _ = build(cfg, kind, overrides=ov, compile=False)
    sd = {k.replace("._orig_mod", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd, strict=True)
    return model.eval(), kind, cfg


def _tok_nll(logits, labels, mask_id):
    logits = logits.float()
    logits[..., mask_id] = float("-inf")
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten().clamp_min(0),
                           reduction="none").view(labels.shape)


@torch.no_grad()
def token_nll(model, kind, x, y, lay, slot_id, depth=6):
    """Per-token NLL in token order, one list per row (labels < 0 dropped)."""
    tok = ~lay.slot_mask
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind in ("plain", "gpt"):
            xp, yp = plain_view(x, y, lay, slot_id)
            kw = {} if kind == "gpt" else {"num_steps_pair": torch.tensor([depth, 0], device=x.device)}
            lg = model(xp, return_logits=True, **kw)["logits"]
            nll = _tok_nll(lg, yp, slot_id)
            return [nll[b][yp[b] >= 0].cpu() for b in range(x.shape[0])]
        h = model(x, None, lay, depth=depth)["hidden"]
        lg = h @ model.lm_head.weight.T * model.config.init.logit_scale
        lab = torch.where(lay.slot_mask, torch.full_like(y, -100), y)
        nll = _tok_nll(lg, lab, slot_id)
        return [nll[b][tok[b] & (lab[b] >= 0)].cpu() for b in range(x.shape[0])]


def row_features(x, y, lay):
    """Per scored token, in token order: pos_in_span, span_index, copy class, far bigram."""
    feats = []
    for b in range(x.shape[0]):
        sm, bg_id = lay.slot_mask[b].cpu(), lay.bag_id[b].cpu()
        tok = (~sm) & (y[b] >= 0)
        ids, tgt, bag = x[b][tok].tolist(), y[b][tok].tolist(), bg_id[tok].tolist()
        seen_span: dict[int, int] = {}        # token id -> earliest span it appeared in
        seen_bg: dict[tuple, int] = {}
        out, pos, prev_bag = [], 0, None
        for i, (t_in, t_out, s) in enumerate(zip(ids, tgt, bag)):
            pos = pos + 1 if s == prev_bag else 0
            prev_bag = s
            # the input token at i is visible when predicting t_out; record it first
            seen_span.setdefault(t_in, s)
            if i > 0:
                seen_bg.setdefault((ids[i - 1], t_in), s)
            first = seen_span.get(t_out)
            copy = "novel" if first is None else ("cross_span" if first < s else "own_span")
            bfirst = seen_bg.get((t_in, t_out))
            out.append((pos, s, copy, "far_bigram" if bfirst is not None and bfirst < s else "other"))
        feats.append(out)
    return feats


def _bucket_pos(p):
    return str(p) if p < 4 else ("4-7" if p < 8 else ("8-15" if p < 16 else "16+"))


def _bucket_span(s):
    return str(s) if s < 2 else ("2-3" if s < 4 else ("4-7" if s < 8 else ("8-15" if s < 16 else "16+")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plain", type=Path, required=True)
    ap.add_argument("--arm", type=Path, required=True)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    plain, pk, cfg = load_run(a.plain)
    arm, ak, _ = load_run(a.arm)
    plain.cuda(); arm.cuda()
    mcfg = morph_cfg(str(cfg.morph.config))
    rt = tul_runtime(mcfg)
    slot_id = int(rt.data_cfg.slot_id)
    rows = eval_rows(mcfg, rt, a.rows, 6)
    cols = ("plain", "arm", "arm_zero_write", "arm_d1")
    acc = {g: defaultdict(lambda: np.zeros(len(cols) + 1)) for g in ("pos_in_span", "span_index", "copy", "bigram", "all")}
    saved = [p.detach().clone() for p in arm.W_prefix]
    for x, y, lay in rows:
        x, y = x.cuda(), y.cuda()
        for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
            setattr(lay, f, getattr(lay, f).cuda())
        vals = [token_nll(plain, pk, x, y, lay, slot_id), token_nll(arm, ak, x, y, lay, slot_id)]
        with torch.no_grad():
            for p in arm.W_prefix:
                p.zero_()
        vals.append(token_nll(arm, ak, x, y, lay, slot_id))
        with torch.no_grad():
            for p, v in zip(arm.W_prefix, saved):
                p.copy_(v)
        vals.append(token_nll(arm, ak, x, y, lay, slot_id, depth=1))
        feats = row_features(x.cpu(), y.cpu(), lay)
        for b, fr in enumerate(feats):
            n = len(fr)
            if any(len(v[b]) != n for v in vals):
                raise ValueError(f"row {b}: token counts differ across models: not paired")
            m = np.stack([v[b].double().numpy() for v in vals], 1)        # [n, cols]
            for i, (p, s, c, bg) in enumerate(fr):
                row = np.concatenate([[1.0], m[i]])
                acc["pos_in_span"][_bucket_pos(p)] += row
                acc["span_index"][_bucket_span(s)] += row
                acc["copy"][c] += row
                acc["bigram"][bg] += row
                acc["all"]["all"] += row
    total_gap = acc["all"]["all"][2] - acc["all"]["all"][1]
    res = {}
    for g, d in acc.items():
        res[g] = {}
        for k, v in d.items():
            n = v[0]
            res[g][k] = {"n": int(n), "share_tokens": n / acc["all"]["all"][0],
                         **{c: v[i + 1] / n for i, c in enumerate(cols)},
                         "gap": (v[2] - v[1]) / n, "share_of_gap": (v[2] - v[1]) / total_gap,
                         "write_worth": (v[3] - v[2]) / n, "k1_minus_k6": (v[4] - v[2]) / n}
    a.out.write_text(json.dumps(res, indent=2))
    for g in ("pos_in_span", "span_index", "copy", "bigram"):
        print(f"\n== {g}")
        print(f"{'bucket':>11} {'tok%':>6} {'plain':>7} {'arm':>7} {'gap':>7} {'gap%':>6} {'write':>7} {'K1-K6':>7}")
        for k, r in sorted(res[g].items()):
            print(f"{k:>11} {100*r['share_tokens']:6.1f} {r['plain']:7.3f} {r['arm']:7.3f} {r['gap']:+7.3f} "
                  f"{100*r['share_of_gap']:6.1f} {r['write_worth']:+7.3f} {r['k1_minus_k6']:+7.4f}")
    r = res["all"]["all"]
    print(f"\nall: plain {r['plain']:.4f} arm {r['arm']:.4f} gap {r['gap']:+.4f} "
          f"write worth {r['write_worth']:+.4f} K1-K6 {r['k1_minus_k6']:+.4f}")


if __name__ == "__main__":
    main()
