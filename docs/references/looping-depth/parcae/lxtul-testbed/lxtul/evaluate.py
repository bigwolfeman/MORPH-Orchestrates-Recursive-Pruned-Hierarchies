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


def eval_rows(mcfg, tul_rt, n_rows: int, batch_size: int):
    """-> list of (x, y, layout) CPU batches: the first n_rows validation TUL rows."""
    if n_rows % batch_size:
        raise ValueError("n_rows must be a multiple of batch_size")
    it = make_loader(mcfg, "validation", batch_size, tul_rt=tul_rt, prefetch=0)
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
    for x, y, lay in rows:
        x, y = x.to(device), y.to(device)
        if arm in ("plain", "gpt"):
            xp, yp = plain_view(x, y, lay, slot_id)
            labels = yp
        else:
            for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"):
                setattr(lay, f, getattr(lay, f).to(device))
            labels = torch.where(lay.slot_mask, torch.full_like(y, -100), y)
        cnt = None
        for k in depths:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if arm == "gpt":                      # no loop: every depth is the same model
                    logits = model(xp, return_logits=True)["logits"]
                elif arm == "plain" and hasattr(model, "pointer"):   # plain + pointer head
                    nll = model.token_nll(xp, yp, num_steps_pair=torch.tensor([k, 0], device=device))
                elif arm == "plain":
                    logits = model(xp, return_logits=True,
                                   num_steps_pair=torch.tensor([k, 0], device=device))["logits"]
                else:
                    h = model(x, None, lay, depth=k)["hidden"]
                    nll = model.token_nll(h, x, y, lay.slot_mask, lay)
            if arm == "plain" and hasattr(model, "pointer"):
                s, c = nll.sum(1).double(), (labels >= 0).sum(1).double()
            elif arm in ("plain", "gpt"):
                s, c = _row_nll(logits, labels, slot_id)
            else:
                s, c = nll.sum(1).double(), (labels >= 0).sum(1).double()
            sums[k].append(s.cpu())
            cnt = c.cpu()
        counts.append(cnt)
    model.train(was)
    return {"depths": list(depths), "count": torch.cat(counts).tolist(),
            "sums": {str(k): torch.cat(v).tolist() for k, v in sums.items()}}


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
