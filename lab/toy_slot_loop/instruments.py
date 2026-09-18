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
    per_pos = None
    per_pos_acc = None
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
        hit = (logits.argmax(-1) == b["labels"][:, vp]).float()
        correct += hit.sum().item()
        cols = pv.sum(0).cpu()
        acc_cols = hit.sum(0).cpu()
        per_pos = cols if per_pos is None else per_pos + cols
        per_pos_acc = acc_cols if per_pos_acc is None else per_pos_acc + acc_cols
        mux_sum += model.mux_ce(out["z"], b["mux_next"]).item() * b["tokens"].shape[0]
        n_rows = b["tokens"].shape[0]
    n_rows_tot = sum(b["tokens"].shape[0] for b in batches)
    return {
        "token_ce": tot[0].item() / n_tok,
        "value_ce": tot[1].item() / n_val,
        "value_acc": correct / n_val,
        "mux_ce": mux_sum / n_rows_tot,
        "value_ce_by_span": (per_pos / (n_val / per_pos.numel())).tolist(),
        "value_acc_by_span": (per_pos_acc / (n_val / per_pos_acc.numel())).tolist(),
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


# --------------------------------------------------------------------------------------
# `eliminate` instruments (2026-09-18): is the alive set carried, and as what?
# --------------------------------------------------------------------------------------
#
# Every one of these reads the slot state after t = 0..T passes, where t = 0 is the ENTRY
# state (zero passes). Three slot roles matter inside an instance whose candidate span is
# span s:
#   role "cand"   slot s+0, the slot that sees the three candidates  -> alive set of 3
#   role "elim1"  slot s+1, the slot that sees the first elimination -> alive set of 2
#   role "answer" slot s+2, the slot the answer is read from         -> alive set of 1
# Under strict geometry a slot reaches spans s+r-t .. s+r after t passes, so WHICH facts a
# state could possibly hold is known exactly, and every instrument is scored against that
# window rather than against the whole instance.

ROLE_NAMES = ("cand", "elim1", "answer")


def _reachable(role: int, t: int) -> dict[str, bool]:
    """What slot s+role can hold after t passes: its window is spans s+role-t .. s+role."""
    lo = role - t
    return {
        "cands": lo <= 0 <= role,  # the candidate span is s+0
        "e1": lo <= 1 <= role,  # the first elimination is at span s+1
        "e2": lo <= 2 <= role,  # the second elimination is at span s+2
    }


@torch.no_grad()
def _slot_states(model, b, depth: int) -> list[torch.Tensor]:
    """[h_0, h_1, ... h_T] for every slot, at a forced depth. h_0 is the entry state."""
    x = model._front(b["tokens"])
    e = x[:, model.slot_pos]
    depths = torch.full(
        (b["tokens"].shape[0], model.cfg.layout.n_spans),
        depth,
        dtype=torch.long,
        device=b["tokens"].device,
    )
    z, h0, states, _ = model.loop(e, depths)
    return [h0] + states


def _inst_masks(b, k: int):
    """[B, 3] one-hot style masks over the instance's three candidate columns."""
    cands = b["candidates"][:, k]  # [B, 3] candidate symbols in span order
    surv = b["survivor"][:, k].unsqueeze(-1)
    e1 = b["elim"][:, k, 0].unsqueeze(-1)
    e2 = b["elim"][:, k, 1].unsqueeze(-1)
    return cands, (cands == surv), (cands == e1), (cands == e2)


@torch.no_grad()
def candidate_mass(model, batches, depth: int = 6) -> dict:
    """Mass the TIED answer head puts on each candidate, per slot role and per pass.

    The softmax is restricted to the three answer ids of the instance's own candidates, so
    it reads how the state splits its belief BETWEEN candidates. A superposition of the
    alive set reads entropy ln(alive) and zero mass on anything the state could already
    know is dead. A committed point reads entropy near 0 whatever the alive set is, which
    is why the entropy and the top mass are reported next to the three masses: the mean
    mass on the survivor alone cannot tell a 3-way superposition from a uniformly random
    commitment (both give 1/3).
    """
    from tasks import ELIM_VALUE_BASE

    model.eval()
    acc = {(r, t): torch.zeros(5) for r in range(len(ROLE_NAMES)) for t in range(depth + 1)}
    n = 0
    for b in batches:
        states = _slot_states(model, b, depth)
        for k, base in enumerate(b["inst_base_span"].tolist()):
            cands, is_s, is_e1, is_e2 = _inst_masks(b, k)
            ids = ELIM_VALUE_BASE + cands  # [B, 3] answer ids
            for role in range(len(ROLE_NAMES)):
                for t in range(depth + 1):
                    reach = _reachable(role, t)
                    lg = model.mux_head(states[t][:, base + role]).gather(1, ids).float()
                    p = lg.softmax(-1)
                    known = (is_e1 & reach["e1"]) | (is_e2 & reach["e2"])
                    alive_other = (is_e1 | is_e2) & ~known
                    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
                    acc[(role, t)] += torch.tensor(
                        [
                            (p * is_s).sum(-1).mean().item(),
                            (p * known).sum(-1).mean().item(),
                            (p * alive_other).sum(-1).mean().item(),
                            ent.mean().item(),
                            p.max(-1).values.mean().item(),
                        ]
                    )
            n += 1
    out = {r: [] for r in ROLE_NAMES}
    for role, name in enumerate(ROLE_NAMES):
        for t in range(depth + 1):
            a = (acc[(role, t)] / n).tolist()
            reach = _reachable(role, t)
            out[name].append(
                {
                    "pass": t,
                    "survivor": a[0],
                    "eliminated_reachable": a[1],
                    "alive_non_survivor": a[2],
                    "entropy": a[3],
                    "top_mass": a[4],
                    "n_alive_reachable": (3 - reach["e1"] - reach["e2"]) if reach["cands"] else None,
                }
            )
    return out


@torch.no_grad()
def _probe_features(model, batches, depth: int, role: int):
    """(X, Y, in_set) per pass: the slot state, the alive target, the candidate mask."""
    from tasks import N_CAND

    X = [[] for _ in range(depth + 1)]
    Y = [[] for _ in range(depth + 1)]
    S = [[] for _ in range(depth + 1)]
    for b in batches:
        states = _slot_states(model, b, depth)
        for t in range(depth + 1):
            reach = _reachable(role, t)
            for k, base in enumerate(b["inst_base_span"].tolist()):
                cands, is_s, is_e1, is_e2 = _inst_masks(b, k)
                B = cands.shape[0]
                in_set = torch.zeros(B, N_CAND, dtype=torch.bool, device=cands.device)
                in_set.scatter_(1, cands, True)
                dead = torch.zeros_like(in_set)
                if reach["e1"]:
                    dead.scatter_(1, b["elim"][:, k, 0:1], True)
                if reach["e2"]:
                    dead.scatter_(1, b["elim"][:, k, 1:2], True)
                X[t].append(states[t][:, base + role].float())
                Y[t].append((in_set & ~dead).float())
                S[t].append(torch.stack([in_set, dead], dim=-1))
    return (
        [torch.cat(x) for x in X],
        [torch.cat(y) for y in Y],
        [torch.cat(s) for s in S],
    )


def membership_probe(
    model, train_batches, eval_batches, depth: int = 6, steps: int = 400, lr: float = 3e-2
) -> dict:
    """Linear probe: is candidate symbol c still alive in this state's reachable window?

    Trained post hoc on the FROZEN model's slot states, one probe per (role, pass), and
    scored on held-out rows. It answers a different question from `candidate_mass`: the
    mass reads whether the set is carried as a superposition of candidate EMBEDDINGS
    through the tied answer head, the probe reads whether the set is linearly decodable in
    ANY encoding. A high probe accuracy under a low mass entropy is the "carried, but not
    as a superposition" outcome.

    Two scores per cell, both balanced so a base rate cannot buy them:
      acc_balanced    over all 6 candidate symbols, mean of the accuracy on alive symbols
                      and the accuracy on the rest. It is the set-membership reading.
      acc_within_set  restricted to the instance's own three candidates: alive against
                      already-eliminated. None when no elimination is inside the window,
                      because then nothing in the set is dead.
    Where the candidate span is NOT in the window (pass t < role) the state cannot know the
    set at all, so those rows are the instrument's own null: they must read about 0.5.
    """
    dev = next(model.parameters()).device
    out = {}
    for role, name in enumerate(ROLE_NAMES):
        Xtr, Ytr, _ = _probe_features(model, train_batches, depth, role)
        Xte, Yte, Ste = _probe_features(model, eval_batches, depth, role)
        rows = []
        for t in range(depth + 1):
            xtr = Xtr[t]
            mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
            lin = torch.nn.Linear(xtr.shape[1], Ytr[t].shape[1]).to(dev)
            opt = torch.optim.Adam(lin.parameters(), lr=lr)
            xn = (xtr - mu) / sd
            for _ in range(steps):
                opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xn), Ytr[t])
                loss.backward()
                opt.step()
            with torch.no_grad():
                pred = lin((Xte[t] - mu) / sd) > 0
                y = Yte[t] > 0
                pos = (pred == y)[y].float().mean().item()
                neg = (pred == y)[~y].float().mean().item()
                in_set, dead = Ste[t][..., 0], Ste[t][..., 1]
                alive_in = in_set & ~dead
                dead_in = in_set & dead
                if dead_in.any():
                    a = (pred == y)[alive_in].float().mean().item()
                    d = (pred == y)[dead_in].float().mean().item()
                    within = 0.5 * (a + d)
                else:
                    within = None
                rows.append(
                    {
                        "pass": t,
                        "acc_balanced": 0.5 * (pos + neg),
                        "acc_alive": pos,
                        "acc_other": neg,
                        "acc_within_set": within,
                        "train_loss": loss.item(),
                        "n_rows": int(Xte[t].shape[0]),
                    }
                )
        out[name] = rows
    return out


@torch.no_grad()
def twin_divergence(model, twin_pairs, depth: int = 6, tol: float = 1e-3) -> dict:
    """Cosine between the slot states of two rows that differ ONLY in span s+1.

    A reachability test with teeth: the state of slot s+role may not move until span s+1
    enters its window, which happens at pass role-1. Role `cand` never reaches it, so its
    cosine must be 1.0 at EVERY pass; role `answer` must read 1.0 at pass 0 and drop at
    pass 1. `drop_pass` is the first pass whose mean cosine falls below 1 - tol.
    """
    model.eval()
    acc = {(r, t): 0.0 for r in range(len(ROLE_NAMES)) for t in range(depth + 1)}
    n = 0
    for ba, bb in twin_pairs:
        sa = _slot_states(model, ba, depth)
        sb = _slot_states(model, bb, depth)
        for base in ba["inst_base_span"].tolist():
            for role in range(len(ROLE_NAMES)):
                for t in range(depth + 1):
                    c = torch.nn.functional.cosine_similarity(
                        sa[t][:, base + role].float(), sb[t][:, base + role].float(), dim=-1
                    )
                    acc[(role, t)] += c.mean().item()
            n += 1
    out = {}
    for role, name in enumerate(ROLE_NAMES):
        cos = [acc[(role, t)] / n for t in range(depth + 1)]
        drop = next((t for t, c in enumerate(cos) if c < 1.0 - tol), None)
        out[name] = {
            "cosine": cos,
            "drop_pass": drop,
            "expected_drop_pass": None if role == 0 else role - 1,
        }
    ce = []
    for i, (ba, bb) in enumerate(twin_pairs):
        ce.append(
            {
                "a_value_ce": eval_ce(model, [ba], force_depth=depth)["value_ce"],
                "b_value_ce": eval_ce(model, [bb], force_depth=depth)["value_ce"],
            }
        )
    out["value_ce"] = {
        "a": sum(c["a_value_ce"] for c in ce) / len(ce),
        "b": sum(c["b_value_ce"] for c in ce) / len(ce),
    }
    return out


def make_twin_pairs(cfg, n_rows, batch, seed, device):
    from tasks import make_twin_batch

    g = torch.Generator(device=device).manual_seed(seed)
    return [
        tuple(
            make_twin_batch(batch, cfg.layout.n_spans, cfg.layout.span_len, generator=g, device=device)
        )
        for _ in range(max(1, n_rows // batch))
    ]


def make_eval_batches(task, cfg, n_rows, batch, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for _ in range(n_rows // batch):
        out.append(
            make_batch(task, batch, cfg.layout.n_spans, cfg.layout.span_len, generator=g, device=device)
        )
    return out
