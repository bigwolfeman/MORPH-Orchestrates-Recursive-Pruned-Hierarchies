"""Do the slot's readers pull its plan apart?

A TUL slot state `h_i` is graded two ways. Its OWN label — the first token of the next span
— carries 2.8 % of the span's loss weight. The other 97 % arrives through the coda's
attention, as a SUM of one gradient per token position that read the slot. `h_i` is a single
point in R^1024 with no way to represent "70 % this, 30 % that", so if the readers want
different things the sum is a compromise and every slot drifts toward the same vector.

That is the geometry behind Explorative Modeling (arXiv:2607.27372) transplanted from the
output distribution to the latent plan. See
`.agents/notes/proposed/architecture/2026-08-24-xm-applies-to-the-plan-not-the-head.md`.

    g_r        = d(CE at reader r) / d(h_i)
    conflict   = || sum_r g_r || / sum_r || g_r ||
    alignment  = conflict * sqrt(K)

`alignment` is the reported statistic, NOT `conflict`. K independent random directions in
high dimension give `conflict ~ 1/sqrt(K)`, so raw conflict falls with the reader count for
reasons that have nothing to do with the model. `alignment ~ 1` is the no-agreement
baseline, above 1 is agreement, below 1 is active disagreement.

Usage:
    PYTHONPATH=$PWD python lab/divergence/reader_conflict_probe.py \
        --ckpt-dir checkpoints/morph/onset-capture --out conflict.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lab.divergence._build import build_cfg, build_model            # noqa: E402
from morph.training.data import create_dataloader                   # noqa: E402
from morph.training.train import load_checkpoint                    # noqa: E402

__all__ = ["conflict_stats", "alignment"]


def alignment(g_sum_norm: float, sum_norms: float, k: int) -> float:
    """conflict * sqrt(K). 1.0 = K random directions; >1 agreement; <1 disagreement."""
    if sum_norms <= 0 or k < 1:
        return float("nan")
    return (g_sum_norm / sum_norms) * math.sqrt(k)


def conflict_stats(grads: list[torch.Tensor], g_direct: torch.Tensor | None) -> dict:
    """Summed-versus-separate geometry of the gradients arriving at one slot state."""
    k = len(grads)
    if k == 0:
        return {}
    stack = torch.stack([g.reshape(-1).float() for g in grads])
    norms = stack.norm(dim=1)
    tot = stack.sum(0)
    tot_n, sum_n = float(tot.norm()), float(norms.sum())
    out = {
        "k": k,
        "sum_norm": tot_n,
        "norm_sum": sum_n,
        "conflict": tot_n / sum_n if sum_n > 0 else float("nan"),
        "alignment": alignment(tot_n, sum_n, k),
        # mean pairwise cosine between readers, the same story without the sqrt(K)
        "mean_pair_cos": float(
            ((stack / (norms.unsqueeze(1) + 1e-30)) @ (stack / (norms.unsqueeze(1) + 1e-30)).T
             ).sum().sub(k) / max(k * (k - 1), 1)),
    }
    if g_direct is not None:
        gd = g_direct.reshape(-1).float()
        out["direct_norm"] = float(gd.norm())
        out["route_frac"] = float(gd.norm()) / (float(gd.norm()) + sum_n + 1e-30)
        out["cos_direct_readers"] = float(
            F.cosine_similarity(gd.unsqueeze(0), tot.unsqueeze(0)).item())
    return out


class _Capture:
    """Grab `h_slots` and the coda hidden state out of one forward, then restore.

    Neither is a `forward`, so a module hook cannot see them. Both are wrapped for the
    duration of the call and restored in `finally`; the model is left as it was found.
    """

    def __init__(self, model):
        root = getattr(model, "_orig_mod", model)
        self.root = root
        self.tul = root.tul
        self.got: dict = {}
        self._pp = self.tul.prefix_project
        self._gl = root._tul_group_losses

    def __enter__(self):
        def pp(h_slots, layout, l_total):
            self.got.setdefault("h", h_slots)
            return self._pp(h_slots, layout, l_total)

        def gl(x, labels, layout, want_groups=True):
            self.got.setdefault("xh", x)
            return self._gl(x, labels, layout, want_groups=want_groups)

        self.tul.prefix_project = pp
        self.root._tul_group_losses = gl
        return self

    def __exit__(self, *a):
        self.tul.prefix_project = self._pp
        self.root._tul_group_losses = self._gl
        return False


def _ce_at(xh, w_head, labels, b: int, pos: int):
    """CE at ONE position, from the hidden state and the tied head.

    Deliberately not the fused kernel over every position: a `[B, L, V]` logit tensor is
    1.4 GB here and this needs one row of it.
    """
    lab = int(labels[b, pos])
    if lab < 0:
        return None
    logit = xh[b, pos].float() @ w_head.float().T
    return F.cross_entropy(logit.unsqueeze(0), labels.new_tensor([lab]))


def measure(model, x, y, layout, seed: int, max_slots_probed: int,
            max_readers: int) -> dict:
    root = getattr(model, "_orig_mod", model)
    pk = root.cfg.tul.prefix_k
    w_head = root.embed.lm_weight()

    with _Capture(model) as cap:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model(x, labels=y, slot_layout=layout)
    h, xh = cap.got.get("h"), cap.got.get("xh")
    if h is None or xh is None:
        raise RuntimeError("prefix_project or _tul_group_losses was never called; "
                           "this arm does not run the slot path")

    tok = (~layout.slot_mask)
    rows = []
    B, S = layout.slot_index.shape
    # deterministic spread over the batch and the slot axis, so a rung is not scored on
    # whichever slots happen to come first
    picks = [(b, i) for b in range(B) for i in range(S)]
    picks = [p for p in picks if bool(layout.slot_valid[p[0], p[1]])]
    if len(picks) > max_slots_probed:
        step = len(picks) / max_slots_probed
        picks = [picks[int(j * step)] for j in range(max_slots_probed)]

    for b, i in picks:
        if i + 1 >= S or not bool(layout.slot_valid[b, i + 1]):
            continue
        rd = torch.nonzero((layout.bag_id[b] == (i + 1)) & tok[b]).reshape(-1)
        if rd.numel() < 3:
            continue
        if rd.numel() > max_readers:
            sel = torch.linspace(0, rd.numel() - 1, max_readers).long()
            rd = rd[sel]
        grads = []
        for r in rd.tolist():
            ce = _ce_at(xh, w_head, y, b, r)
            if ce is None:
                continue
            g = torch.autograd.grad(ce, h, retain_graph=True, allow_unused=True)[0]
            if g is not None:
                grads.append(g[b, i].detach())
        emit = int(layout.slot_index[b, i]) + pk - 1
        ce_a = _ce_at(xh, w_head, y, b, emit)
        g_a = None
        if ce_a is not None:
            ga = torch.autograd.grad(ce_a, h, retain_graph=True, allow_unused=True)[0]
            g_a = ga[b, i].detach() if ga is not None else None
        st = conflict_stats(grads, g_a)
        if st:
            st["b"], st["slot"] = b, i
            rows.append(st)
    return {"per_slot": rows}


def _mean(rows, f):
    v = [r[f] for r in rows if f in r and r[f] == r[f]]
    return sum(v) / len(v) if v else float("nan")


def _step_of(p): 
    m = re.search(r"(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="tul_a2")
    ap.add_argument("--overrides", default="training.batch_size=4,model.use_kernels=false")
    ap.add_argument("--extra", default="", help="SEMICOLON-separated hydra overrides")
    ap.add_argument("--ckpt-dir", default="")
    ap.add_argument("--ckpt", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slots", type=int, default=12)
    ap.add_argument("--readers", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    paths = list(a.ckpt)
    if a.ckpt_dir:
        paths += sorted(glob.glob(os.path.join(a.ckpt_dir, "*.pt")), key=_step_of)
    if not paths:
        ap.error("give --ckpt or --ckpt-dir")

    ov = [o for o in a.overrides.split(",") if o.strip()]
    ov += [o for o in a.extra.split(";") if o.strip()]
    cfg = build_cfg(a.config_name, ov)
    model, tul_rt = build_model(cfg, device="cuda")
    loader = iter(create_dataloader(cfg.data.tokenizer, cfg.data.dataset,
                                    int(cfg.data.seq_len), int(cfg.training.batch_size),
                                    split="validation", skip_samples=50_000,
                                    tul=tul_rt.val_data_cfg if tul_rt else None))
    x, y, layout = next(loader)
    x, y, layout = x.cuda(), y.cuda(), layout.to("cuda")
    model.train()

    scaler = torch.amp.GradScaler("cuda", enabled=False)
    results = []
    print(f"{'ckpt':<24}{'step':>6}{'slots':>7}{'K':>5}{'align':>9}{'conflict':>10}"
          f"{'paircos':>9}{'routefrac':>11}{'cosAR':>8}")
    for p in paths:
        load_checkpoint(p, model, scaler, torch.device("cuda"))
        r = measure(model, x, y, layout, a.seed, a.slots, a.readers)
        rows = r["per_slot"]
        rec = {"path": os.path.basename(p), "step": _step_of(p), "n_slots": len(rows),
               **{f: _mean(rows, f) for f in ("k", "alignment", "conflict",
                                              "mean_pair_cos", "route_frac",
                                              "cos_direct_readers")},
               "per_slot": rows}
        results.append(rec)
        print(f"{rec['path']:<24}{rec['step']:>6}{rec['n_slots']:>7}{rec['k']:>5.1f}"
              f"{rec['alignment']:>9.3f}{rec['conflict']:>10.4f}"
              f"{rec['mean_pair_cos']:>9.4f}{rec['route_frac']:>11.4f}"
              f"{rec['cos_direct_readers']:>8.3f}")
        sys.stdout.flush()

    if a.out:
        json.dump({"config": a.config_name, "overrides": ov, "rows": results},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
