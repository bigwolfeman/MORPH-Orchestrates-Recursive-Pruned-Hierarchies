"""Sequential importance sampling bound for an XM span-plan run (eval only, expensive).

    python -m lxtul.xm_sis_eval /home/wolfe/parcae-runs/RUN --depth 6 --out sis.json

The model is a latent-index model:  p(x) = sum over paths of prod_j (1/K) p(span_j | x_<, k_<=j)  with a fixed uniform prior.
The training CE is one path's term. SIS gets the whole sum's estimate span by span. Along the drawn path, span j + 1's K
candidates are scored EXACTLY (the coda run once per candidate index, everything else fixed), the span adds

    -log mean_k p(span_{j+1} | path_<j, k)

to the row's NLL, and k is drawn from the posterior  p(k | span) ~ p(span | path_<j, k)  (seeded). The product of the per-span
means is an unbiased estimate of p(x), so its log is an NLL bound in expectation (Jensen). Positions of span 0 (no plan)
add their plain NLL. At K = 1 the sum is the CE.

Why only the coda reruns: nothing in the trunk (prelude, levels, plan loop, candidates) depends on the index path, and a
span's tokens depend only on the plans <= its own (the coda is causal and plan j reaches positions >= p_last(j)), so the K
candidate forwards of step j fix every later plan at any filler. Cost: spans x K coda forwards per batch, about 40 PFLOP for
480 rows at K = 8 (the design's estimate; not measured).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import Tensor

from lxtul.xm import _rng_fork, plan_depth_sweep


def _span_nll(model, x: Tensor, yp: Tensor) -> Tensor:
    """[B, L] per-position NLL (0 where label < 0) under the run's head from the final-norm output x."""
    if not hasattr(model, "pointer"):
        raise ValueError("xm_sis_eval scores through the plain+pointer head: wrap the model in PlainPointer")
    return model._score(x, model.inner.logits_from(x), yp)


@torch.no_grad()
def candidate_logps(model, trunk, yp: Tensor, path: Tensor, j: int) -> Tensor:
    """[B, K] log p(span_{j+1} | path_<j, k) for every candidate k at plan j: the summed log-probability of the positions
    whose label is in span j + 1, with plan j set to k and plans < j taken from `path`."""
    inner = model.inner
    K = trunk.cand.shape[2]
    sel = trunk.layout.plan_recv == j
    out = []
    for k in range(K):
        pk = path.clone()
        pk[:, j] = k
        out.append(-(_span_nll(model, inner.plan_coda(trunk, pk), yp) * sel).sum(1))
    return torch.stack(out, 1).float()


@torch.no_grad()
def sis_batch(model, xp: Tensor, yp: Tensor, slay, depth: int, seed: int) -> Tensor:
    """-> [B] double: the SIS NLL of each row (summed over its scored tokens), one particle."""
    inner = model.inner
    K, G = inner.plan_cfg.k, inner.spiral.max_spans
    gen = torch.Generator(device=xp.device).manual_seed(int(seed))
    with _rng_fork(xp.device):
        torch.manual_seed(int(seed))                              # the loop's state init, as plan_depth_sweep seeds it
        trunk = inner.plan_trunk(xp, depth, slay)
    recv = trunk.layout.plan_recv
    path = torch.zeros(xp.shape[0], G, dtype=torch.long, device=xp.device)
    total = (_span_nll(model, inner.plan_coda(trunk, path), yp) * (recv < 0)).sum(1).double()   # span 0: no plan reaches it
    for j in range(int(recv.max()) + 1):
        if not bool(((recv == j) & (yp >= 0)).any()):
            continue
        lp = candidate_logps(model, trunk, yp, path, j)
        total -= (torch.logsumexp(lp, 1) - math.log(K)).double()
        path[:, j] = torch.multinomial(torch.softmax(lp, 1), 1, generator=gen).squeeze(1)
    return total


@torch.no_grad()
def sis_run(model, rows, depth: int, slot_id: int, seed: int = 20261016, device="cuda") -> dict:
    """SIS over `rows` (the trainer's eval rows), plus B, U and CE|k* at the same depth from `plan_depth_sweep`."""
    from lxtul.evaluate import plain_view
    inner = model.inner
    was = model.training
    model.eval()
    sis, count = [], []
    for bi, (x, y, lay) in enumerate(rows):
        x, y = x.to(device), y.to(device)
        xp, yp = plain_view(x, y, lay, slot_id)
        slay = inner.span_layout(xp).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            sis.append(sis_batch(model, xp, yp, slay, depth, seed * 1_000_003 + bi).cpu())
        count.append((yp >= 0).sum(1).double().cpu())
    sw = plan_depth_sweep(model, rows, [depth], slot_id, device, seed=seed)
    model.train(was)
    cnt, sis = torch.cat(count), torch.cat(sis)
    if not torch.equal(cnt, torch.tensor(sw["count"], dtype=torch.double)):
        raise RuntimeError("SIS scored a different token set than the sweep")
    tok = float(cnt.sum())
    d = str(depth)
    ce_k = sum(sw["sums"][d]) / tok
    return {"depth": depth, "k": inner.plan_cfg.k, "tokens": tok, "seed": seed,
            "sis": float(sis.sum()) / tok, "bound_B": sum(sw["sums_bound"][d]) / tok, "uniform_U": sum(sw["sums_uniform"][d]) / tok,
            "ce_given_argmax": ce_k, "plan_gap": sum(sw["sums_uniform"][d]) / tok - ce_k,
            "sis_rows": sis.tolist(), "count": cnt.tolist()}


def main() -> None:
    from lxtul.data import morph_cfg
    from lxtul.evaluate import eval_rows
    from lxtul.gap_probe import load_run
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    model, kind, cfg, rt = load_run(a.run)
    if kind != "spiral" or getattr(getattr(model, "inner", model), "plan_cfg", None) is None:
        raise ValueError(f"{a.run.name}: not a spiral run with spiral.plan")
    rows = eval_rows(morph_cfg(str(cfg.morph.config)), rt, a.rows, a.batch)
    res = sis_run(model.cuda(), rows, a.depth, int(rt.data_cfg.slot_id), a.seed)
    res["run"] = a.run.name
    a.out.write_text(json.dumps(res))
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_rows") and k != "count"}), flush=True)


if __name__ == "__main__":
    main()
