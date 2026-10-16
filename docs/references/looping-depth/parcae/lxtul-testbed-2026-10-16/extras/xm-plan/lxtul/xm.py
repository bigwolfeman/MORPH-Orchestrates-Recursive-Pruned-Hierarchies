"""XM span plan (phase 1): K plan heads on the span loop, a parameter-free selection, the winner added to the coda input.

Design: /home/wolfe/morph-scratch/abq/drawing-board/xm-span-plan.md. Read the module docstring of `lxtul/spiral.py`
(`spiral.plan`) for where the plan sits in the forward. This file holds the pieces that do not touch the model:

  PlanConfig       the `spiral.plan` block
  plan_tokens      tokens per (row, plan index)
  bag_scores       score_k(j) = sum over the tokens plan j conditions of log softmax(head(c_{j,k}) / tau)[token]
  sinkhorn_log     log-domain balancing of index use over a batch (training only)
  choose_winners   argmax of the (balanced) score
  charged_spans    spans that pay ln K on a scored set
  uniform_path     a seeded uniform index path (what generation sees)
  win_fraction     share of spans each index wins
  plan_depth_sweep per depth: CE given the argmax winners, the bound B, the uniform-path CE U

The likelihood. p(x) = sum over index paths of prod_j (1/K) p(span_j | x_<, k_<=j), a fixed uniform prior. Any path, even one
chosen with the data, is ONE term of that sum, so  -log p(x) <= CE(x | path) + ln K * (spans that carry a plan)  for every
selection rule. That is the bound B. The score only picks the path; nothing it computes enters the coda.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# ---------------------------------------------------------------- configuration
@dataclass(frozen=True)
class PlanConfig:
    k: int = 8                       # plan heads = indices; k == 1 is the control (no selection, no score)
    score: str = "bag_tied"          # bag_tied: the tied head reads each candidate (the only phase-1 score)
    tau: float = 1.0                 # temperature of the score's softmax
    balance: str = "sinkhorn"        # "none" | "sinkhorn": training only, eval is always argmax
    sinkhorn_iters: int = 3
    head_noise: float = 0.02         # std of the noise added to head k's copy of C, relative to C's std (k > 1 only)
    receive: str = "label_span"      # plan j reaches the positions whose LABEL is in span j + 1

    def __post_init__(self):
        if self.k < 1:
            raise ValueError(f"plan.k must be >= 1, got {self.k}")
        if self.score != "bag_tied":
            raise ValueError(f"unknown plan.score {self.score!r} (bag_tied)")
        if not self.tau > 0:
            raise ValueError(f"plan.tau must be > 0, got {self.tau}")
        if self.balance not in ("none", "sinkhorn"):
            raise ValueError(f"unknown plan.balance {self.balance!r} (none | sinkhorn)")
        if self.sinkhorn_iters < 1:
            raise ValueError(f"plan.sinkhorn_iters must be >= 1, got {self.sinkhorn_iters}")
        if self.head_noise < 0:
            raise ValueError(f"plan.head_noise must be >= 0, got {self.head_noise}")
        if self.receive != "label_span":
            raise ValueError(f"unknown plan.receive {self.receive!r} (label_span)")


# ---------------------------------------------------------------- score and selection
def plan_tokens(recv: Tensor, G: int) -> Tensor:
    """[B, L] plan index of each TOKEN as a label (-1 none) -> [B, G] float count of the tokens each plan conditions."""
    B = recv.shape[0]
    cnt = torch.zeros(B, G + 1, device=recv.device)
    cnt.scatter_add_(1, torch.where(recv >= 0, recv, torch.full_like(recv, G)), torch.ones_like(recv, dtype=torch.float))
    return cnt[:, :G]


@torch.no_grad()
def bag_scores(logits_fn, cand: Tensor, ids: Tensor, recv: Tensor, tau: float) -> Tensor:
    """score[b, j, k] = sum over tokens t with recv[b, t] == j of log softmax(logits_fn(cand[b, j, k]) / tau)[ids[b, t]].

    cand [B, G, K, d] (the candidates), ids [B, L] the input token ids, recv [B, L] the plan index each token is scored
    against (-1: none). `logits_fn` maps [n, d] to fp32 logits [n, V] (the tied head with Parcae's scale and softcap).
    No gradient: the selection is a function of detached values, so nothing flows through it. One row at a time keeps
    the [G * K, V] log-softmax at G * K * V * 4 bytes. -> scores [B, G, K] fp32."""
    B, G, K, d = cand.shape
    out = torch.zeros(B, G, K, device=cand.device)
    for b in range(B):
        lp = torch.log_softmax(logits_fn(cand[b].detach().reshape(G * K, d)) / tau, dim=-1).view(G, K, -1)
        j = recv[b].clamp_min(0)
        picked = lp[j, :, ids[b]]                                    # [L, K]: advanced indices around a slice go first
        picked = picked * (recv[b] >= 0)[:, None]
        out[b].index_add_(0, j, picked)
    return out


def sinkhorn_log(logw: Tensor, valid: Tensor, iters: int) -> Tensor:
    """Balance index use over the batch. logw [B, G, K] log-weights, valid [B, G] bool (the spans that carry a plan).
    `iters` rounds of (rows to sum 1, then columns to sum n / K, n = valid spans); the last step is a column step, so the
    column sums of exp(result) over valid spans are n / K to float precision. Invalid spans come back as -inf. No host sync."""
    B, G, K = logw.shape
    m = valid[..., None]
    n = valid.sum().clamp_min(1).to(logw.dtype)                  # no host sync; with no valid span the result is all -inf
    x = logw.float().masked_fill(~m, 0.0)
    for _ in range(iters):
        x = x - torch.logsumexp(x, dim=-1, keepdim=True)
        col = torch.logsumexp(x.masked_fill(~m, float("-inf")).reshape(B * G, K), dim=0)
        x = x - col + (n / K).log()
    return x.masked_fill(~m, float("-inf"))


def choose_winners(scores: Tensor, valid: Tensor, balance: str, iters: int, training: bool) -> Tensor:
    """Argmax per (row, span) of the score, after Sinkhorn balancing in training with `balance: sinkhorn`. Eval never
    balances: the winner is the argmax of the raw score. -> [B, G] long."""
    if training and balance == "sinkhorn":
        scores = sinkhorn_log(scores, valid, iters)
    return scores.argmax(-1)


def win_fraction(winners: Tensor, valid: Tensor, K: int) -> Tensor:
    """[K] share of the valid spans each index wins (device tensor, no sync)."""
    hot = torch.nn.functional.one_hot(winners, K).float() * valid[..., None]
    return hot.sum((0, 1)) / valid.sum().clamp_min(1)


def charged_spans(plan_recv: Tensor, labels: Tensor, G: int) -> Tensor:
    """[B] number of plans that condition at least one SCORED position: the spans that pay ln K. plan_recv [B, L] is the
    plan each POSITION receives (-1 none), labels [B, L] with -100 where unscored. The open tail counts (it carries the
    last pooled span's plan); span 0 does not (no plan reaches it)."""
    B = plan_recv.shape[0]
    hit = torch.zeros(B, G + 1, device=plan_recv.device)
    idx = torch.where((plan_recv >= 0) & (labels >= 0), plan_recv, torch.full_like(plan_recv, G))
    hit.scatter_(1, idx, 1.0)
    return hit[:, :G].sum(1)


def uniform_path(B: int, G: int, K: int, seed: int, device) -> Tensor:
    """A seeded uniform index path [B, G] (what generation draws)."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randint(0, K, (B, G), generator=g).to(device)


# ---------------------------------------------------------------- the sweep
def _rng_fork(device):
    cuda = torch.device(device).type == "cuda"
    return torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else [])


@torch.no_grad()
def plan_depth_sweep(model, rows, depths, slot_id: int, device="cuda", uniform: bool = True, seed: int = 20261016) -> dict:
    """Per-row sums per depth d on the trainer's scored set (`evaluate.plain_view`, the first n_rows validation rows).

      sums[d]          CE given the ARGMAX winners (eval selection, no Sinkhorn), summed over the row's scored tokens
      sums_bound[d]    B(d) = sums[d] + ln K * charged   (a valid NLL bound for the selected path)
      sums_uniform[d]  U(d): CE given a seeded uniform random path, the same path at every depth (what generation sees)
      plan             {"k", "ln_k", "charged": per row, "win_frac": {d: [K]} when the model exposes its winners}

    The loop's state init is random: every forward of a batch is re-seeded the same way, so the argmax run and the
    uniform run at a depth differ in the index path alone, and a depth differs from another in its pass count alone.
    Plan gap(d) = U(d) - CE|k*(d). The scored count is recounted from the packed row and must match."""
    import torch.nn.functional as F
    from lxtul.evaluate import plain_view

    inner = getattr(model, "inner", model)
    pc = inner.spiral.plan
    if pc is None:
        raise ValueError("plan_depth_sweep needs a model with spiral.plan")
    K, G = pc.k, inner.spiral.max_spans
    ptr = hasattr(model, "pointer")
    was = model.training
    model.eval()
    sums = {d: [] for d in depths}
    usums = {d: [] for d in depths}
    wins = {d: torch.zeros(K, dtype=torch.float64) for d in depths}
    count, ref, charged = [], [], []
    spans = {"n_src": [], "n_bound": []}
    for bi, (x, y, lay) in enumerate(rows):
        x, y = x.to(device), y.to(device)
        xp, yp = plain_view(x, y, lay, slot_id)
        slay = inner.span_layout(xp).to(device)
        spans["n_src"].append(slay.n_src.cpu())
        spans["n_bound"].append(slay.n_bound.cpu())
        count.append((yp >= 0).sum(1).double().cpu())
        ref.append(((~lay.slot_mask.to(device)) & (y >= 0)).sum(1).double().cpu())
        charged.append(charged_spans(slay.plan_recv, yp, G).double().cpu())
        path = uniform_path(x.shape[0], G, K, seed * 1_000_003 + bi, device)
        for d in depths:
            for tag, idx in (("arg", None), ("uni", path)):
                if tag == "uni" and not uniform:
                    continue
                if tag == "uni" and K == 1:                       # one index: the uniform path IS the argmax path
                    usums[d].append(sums[d][-1])
                    continue
                with _rng_fork(device):
                    torch.manual_seed(seed * 1_000_003 + bi)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
                        kw = {"layout": slay, "depth": d, "plan_index": idx}
                        if ptr:
                            nll = model.token_nll(xp, yp, **kw)
                            s = nll.sum(1).double()
                        else:
                            lg = model(xp, return_logits=True, **kw)["logits"].float()
                            lg[..., slot_id] = float("-inf")
                            nl = F.cross_entropy(lg.flatten(0, 1), yp.flatten().clamp_min(0), reduction="none")
                            s = (nl.view(yp.shape) * (yp >= 0)).sum(1).double()
                if tag == "arg":
                    sums[d].append(s.cpu())
                    lp = getattr(model, "last_plan", None)
                    if lp is not None:
                        wins[d] += (win_fraction(lp["winners"], lp["valid"], K) * lp["valid"].sum()).double().cpu()
                else:
                    usums[d].append(s.cpu())
    model.train(was)
    cat = torch.cat
    cnt, rf = cat(count), cat(ref)
    if not torch.equal(cnt, rf):
        raise RuntimeError("scored token count differs from the independent recount of the scored set")
    ch = cat(charged)
    out = {"depths": list(depths), "count": cnt.tolist(),
           "sums": {str(d): cat(sums[d]).tolist() for d in depths},
           "sums_bound": {str(d): (cat(sums[d]) + math.log(K) * ch).tolist() for d in depths},
           "plan": {"k": K, "ln_k": math.log(K), "charged": ch.tolist(),
                    "win_frac": {str(d): (wins[d] / wins[d].sum().clamp_min(1)).tolist() for d in depths}},
           "spans": {k: cat(v).tolist() for k, v in spans.items()}}
    if uniform:
        out["sums_uniform"] = {str(d): cat(usums[d]).tolist() for d in depths}
    return out
