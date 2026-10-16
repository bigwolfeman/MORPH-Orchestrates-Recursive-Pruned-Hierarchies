"""Depth sweep on fixed held-out rows, paired across arms.

Rows: the first `n_rows` TUL rows of MORPH's validation stream (OWT split "validation",
first 50k documents skipped), packed by MORPH's packer. A plain arm scores the SAME
tokens with the SAME targets: each row's token positions only, slots removed, padded at
the end with label -100. So every arm's per-row sums are over identical (token, target)
pairs, and arm-vs-arm differences are paired per row.

Output per depth K: per-row NLL sums and token counts (`sums[K]`, `count`), so K1-K6 and
gaps use `lab/divergence/_stats.paired_bootstrap_ci` exactly as MORPH's sweeps do.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lxtul.data import make_loader


def eval_rows(mcfg, tul_rt, n_rows: int, batch_size: int, doc_stats: dict | None = None,
              eval_cells: int = 0):
    """-> list of (x, y, layout) CPU batches: the first n_rows validation TUL rows.

    `doc_stats` (optional dict) receives the documents the rows came from: `docs` documents
    after the skip, i.e. validation documents [EVAL_SKIP_DOCS, EVAL_SKIP_DOCS + docs).
    `eval_cells` > 0 re-lays the rows to that many cells per slot (LXTULConfig.cell_count_eval);
    0 keeps MORPH's width."""
    if n_rows % batch_size:
        raise ValueError("n_rows must be a multiple of batch_size")
    it = make_loader(mcfg, "validation", batch_size, tul_rt=tul_rt, prefetch=0, doc_stats=doc_stats,
                     eval_cells=eval_cells)
    return [next(it) for _ in range(n_rows // batch_size)]


def plain_view(x, y, lay, pad_id: int):
    """Token positions only, left-aligned, padded at the end (label -100)."""
    tok = ~lay.slot_mask
    B, L = x.shape
    xp = torch.full_like(x, pad_id)
    yp = torch.full_like(y, -100)
    for b in range(B):
        n = int(tok[b].sum())
        xp[b, :n] = x[b][tok[b]]
        yp[b, :n] = y[b][tok[b]]
    return xp, yp


def _row_nll(logits, labels, mask_id: int):
    logits = logits.float()
    logits[..., mask_id] = float("-inf")
    nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten().clamp_min(0), reduction="none")
    ok = (labels >= 0).flatten()
    return (nll * ok).view(labels.shape).sum(1).double(), (labels >= 0).sum(1).double()


@torch.no_grad()
def depth_sweep(model, arm: str, rows, depths, slot_id: int, device="cuda") -> dict:
    was = model.training
    model.eval()
    sums = {k: [] for k in depths}
    counts = []
    spans = {"n_src": [], "n_bound": []}                # spiral arms: the span level's real size per row
    inner = getattr(model, "inner", model)
    # a flat model, and a spiral with no sampled-depth level, has no depth: score it once and reuse for every depth
    once = arm == "gpt" or (arm == "spiral" and not inner.has_sampled_depth)
    for x, y, lay in rows:
        x, y = x.to(device), y.to(device)
        if arm in ("plain", "gpt", "spiral"):
            xp, yp = plain_view(x, y, lay, slot_id)
            labels = yp
            if arm == "spiral":
                slay = inner.span_layout(xp).to(device)
                spans["n_src"].append(slay.n_src.cpu())
                spans["n_bound"].append(slay.n_bound.cpu())
        else:
            for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
                setattr(lay, f, getattr(lay, f).to(device))
            labels = torch.where(lay.slot_mask, torch.full_like(y, -100), y)
        cnt = None
        flat = None                                   # a flat model has no depth: score it once
        for k in depths:
            if once and flat is not None:
                sums[k].append(flat[0])
                cnt = flat[1]
                continue
            ptr = arm in ("plain", "gpt", "spiral") and hasattr(model, "pointer")     # + pointer head (plain_pointer.py)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if arm == "spiral":                   # depth sets the sampled-depth level's passes; None: none
                    kw = {"layout": slay, "depth": k if inner.has_sampled_depth else None}
                    if ptr:
                        nll = model.token_nll(xp, yp, **kw)
                    else:
                        logits = model(xp, return_logits=True, **kw)["logits"]
                elif arm == "gpt" and ptr:
                    nll = model.token_nll(xp, yp)
                elif arm == "gpt":                    # no loop: every depth is the same model
                    logits = model(xp, return_logits=True)["logits"]
                elif arm == "plain" and ptr:
                    nll = model.token_nll(xp, yp, num_steps_pair=torch.tensor([k, 0], device=device))
                elif arm == "plain":
                    logits = model(xp, return_logits=True,
                                   num_steps_pair=torch.tensor([k, 0], device=device))["logits"]
                else:
                    h = model(x, None, lay, depth=k)["hidden"]
                    nll = model.token_nll(h, x, y, lay.slot_mask, lay)
            if ptr or arm == "lxtul":
                s, c = nll.sum(1).double(), (labels >= 0).sum(1).double()
            else:
                s, c = _row_nll(logits, labels, slot_id)
            sums[k].append(s.cpu())
            cnt = c.cpu()
            if once:
                flat = (s.cpu(), cnt)
        counts.append(cnt)
    model.train(was)
    out = {"depths": list(depths), "count": torch.cat(counts).tolist(),
           "sums": {str(k): torch.cat(v).tolist() for k, v in sums.items()}}
    if arm == "spiral":
        out["spans"] = {k: torch.cat(v).tolist() for k, v in spans.items()}
    return out


def _stats():
    """MORPH's lab/divergence/_stats.py, loaded by path (lab/ is not a package)."""
    import importlib.util
    from lxtul.data import MORPH_ROOT
    spec = importlib.util.spec_from_file_location("_morph_stats", MORPH_ROOT / "lab/divergence/_stats.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def summarize(sweep: dict, ref: int = 6) -> dict:
    """CE per depth, and K - ref CI per depth (paired over rows)."""
    import numpy as np
    paired_bootstrap_ci = _stats().paired_bootstrap_ci
    cnt = np.asarray(sweep["count"])
    out = {"ce": {k: float(np.sum(v) / cnt.sum()) for k, v in sweep["sums"].items()}}
    r = np.asarray(sweep["sums"][str(ref)])
    out["minus_ref"] = {k: paired_bootstrap_ci(np.asarray(v), r, cnt)
                        for k, v in sweep["sums"].items() if k != str(ref)}
    return out
