"""Instruments for the toy slot loop.

Ported in spirit from the real probes:
  - lab/divergence/slot_gradient_probe.py  (per-pass cotangent, per-pass weight gradient)
  - lab/divergence/core_depth_sweep.py     (the K-curve)
  - lab/divergence/slot_z_optimize.py      (z = entry vs z = exit, participation rank)

The per-pass weight gradient uses a TAP, not module hooks: every pass calls the core
through `torch.func.functional_call` with its own graph node per parameter, so the
gradient banked at pass t is exactly that pass's contribution. The mandatory self-check is
`sum_t dW_t == leaf.grad` on every core parameter, measured in the SAME backward.
"""

from __future__ import annotations

import math

import torch

from tasks import make_batch


# --------------------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------------------


@torch.no_grad()
def eval_ce(model, batches, force_depth: int = 0, z_override: str = "") -> dict[str, float]:
    """Total token CE, value-position CE, value accuracy, MUX CE, at a forced depth."""
    model.eval()
    tot = torch.zeros(4)
    n_tok = n_val = 0
    mux_sum = 0.0
    correct = 0
    for b in batches:
        out = model(b, force_depth=force_depth, z_override=z_override)
        per = model.token_ce(out["xh"], b["labels"], reduce=False)
        valid = b["labels"] != -100
        tot[0] += per[valid].sum().item()
        n_tok += int(valid.sum())
        vp = b["value_pos"]
        pv = per[:, vp]
        tot[1] += pv.sum().item()
        n_val += pv.numel()
        logits = model.head(out["xh"][:, model.tok_pos])[:, vp]
        correct += (logits.argmax(-1) == b["labels"][:, vp]).sum().item()
        mux_sum += model.mux_ce(out["z"], b["mux_next"]).item() * b["tokens"].shape[0]
        n_rows = b["tokens"].shape[0]
    n_rows_tot = sum(b["tokens"].shape[0] for b in batches)
    return {
        "token_ce": tot[0].item() / n_tok,
        "value_ce": tot[1].item() / n_val,
        "value_acc": correct / n_val,
        "mux_ce": mux_sum / n_rows_tot,
    }


def k_curve(model, batches, depths=(1, 2, 3, 6, 8, 12)) -> dict[str, dict[str, float]]:
    return {str(d): eval_ce(model, batches, force_depth=d) for d in depths}


@torch.no_grad()
def write_contribution(model, batches, depth: int = 6) -> dict[str, float]:
    """CE with z = exit, z = entry (0 passes) and z = 0. The real probe's key reading."""
    out = {}
    for name, ov in (("exit", ""), ("entry", "entry"), ("zero", "zero")):
        r = eval_ce(model, batches, force_depth=depth, z_override=ov)
        out[f"{name}_token_ce"] = r["token_ce"]
        out[f"{name}_value_ce"] = r["value_ce"]
        out[f"{name}_value_acc"] = r["value_acc"]
    return out


@torch.no_grad()
def participation_rank(model, batches, depth: int = 6) -> dict[str, float]:
    """Participation ratio of the singular value spectrum of the slot states."""
    model.eval()
    zs, hs = [], []
    for b in batches:
        o = model(b, force_depth=depth)
        zs.append(o["z"].reshape(-1, o["z"].shape[-1]).float())
        hs.append(o["h0"].reshape(-1, o["h0"].shape[-1]).float())

    def pr(m):
        m = m - m.mean(0, keepdim=True)
        s = torch.linalg.svdvals(m)
        e = s.pow(2)
        return (e.sum().pow(2) / e.pow(2).sum()).item()

    Z = torch.cat(zs)
    H = torch.cat(hs)
    return {
        "z_rank": pr(Z),
        "entry_rank": pr(H),
        "z_norm": Z.norm(dim=-1).mean().item(),
        "n_slots": Z.shape[0],
    }


# --------------------------------------------------------------------------------------
# per-pass gradient probe
# --------------------------------------------------------------------------------------


def _flat(d: dict[str, torch.Tensor], names) -> torch.Tensor:
    return torch.cat([d[n].reshape(-1) for n in names])


def gradient_probe(model, batches, depth: int = 6, source: str = "total") -> dict:
    """Per-pass cotangent at the loop state and per-pass share of the shared core weights.

    source: "total" (the trained loss), "token_ce" or "mux".
    """
    model.train()
    core_names = [n for n, _ in model.core.named_parameters()]
    shared = [n for n in core_names if not n.startswith("lora_")]

    T = depth
    cot = [0.0] * T
    dW_sum = None
    dW_each = [None] * T
    n = 0
    selfcheck = 0.0

    for b in batches:
        model.zero_grad(set_to_none=True)
        model._tap = True
        model._cot = True
        model.tap_params = []
        model.cot_hooks = []
        out = model(b, force_depth=depth)
        if source == "total":
            loss = out["loss"]
        elif source == "token_ce":
            loss = model.token_ce(out["xh"], b["labels"])
        elif source == "mux":
            loss = model.mux_ce(out["z"], b["mux_next"])
        else:
            raise ValueError(source)
        loss.backward()
        model._tap = False
        model._cot = False

        assert len(model.tap_params) == T, (len(model.tap_params), T)
        for t in range(T):
            g = model.cot_hooks[t].grad
            cot[t] += 0.0 if g is None else g.norm().item()
            gt = {k: (v.grad if v.grad is not None else torch.zeros_like(v)) for k, v in model.tap_params[t].items()}
            f = _flat(gt, shared)
            dW_each[t] = f.clone() if dW_each[t] is None else dW_each[t] + f
            dW_sum = f.clone() if dW_sum is None else dW_sum + f
        # self-check: the taps must sum to the leaf gradient of the SAME backward
        for name, p in model.core.named_parameters():
            if p.grad is None:
                continue
            s = sum(model.tap_params[t][name].grad for t in range(T) if model.tap_params[t][name].grad is not None)
            rel = (s - p.grad).norm().item() / (p.grad.norm().item() + 1e-12)
            selfcheck = max(selfcheck, rel)
        n += 1
        model.tap_params = []
        model.cot_hooks = []
    model.zero_grad(set_to_none=True)

    total = torch.stack(dW_each).sum(0)
    cos = [torch.nn.functional.cosine_similarity(dW_each[t], total, dim=0).item() for t in range(T)]
    norms = [dW_each[t].norm().item() for t in range(T)]
    cot = [c / n for c in cot]
    s_cot = sum(cot) + 1e-12
    s_norm = sum(norms) + 1e-12
    # pairwise cosine between passes
    pair = []
    for i in range(T):
        for j in range(i + 1, T):
            pair.append(torch.nn.functional.cosine_similarity(dW_each[i], dW_each[j], dim=0).item())
    return {
        "source": source,
        "depth": depth,
        "cotangent": cot,
        "cotangent_share": [c / s_cot for c in cot],
        "dW_norm": norms,
        "dW_share": [x / s_norm for x in norms],
        "cos_to_total": cos,
        "pairwise_cos_mean": sum(pair) / len(pair) if pair else float("nan"),
        "cancellation": total.norm().item() / s_norm,
        "orthogonal_reference": 1.0 / math.sqrt(T),
        "selfcheck_max_rel_err": selfcheck,
        "n_batches": n,
        "n_shared_params": int(sum(p.numel() for nm, p in model.core.named_parameters() if nm in shared)),
    }


def make_eval_batches(task, cfg, n_rows, batch, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(n_rows // batch):
        out.append(
            make_batch(task, batch, cfg.layout.n_spans, cfg.layout.span_len, generator=g, device=device)
        )
    return out
