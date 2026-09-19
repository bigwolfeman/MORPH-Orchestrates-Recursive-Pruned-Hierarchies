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
# elimination-task instruments: is the alive set carried, and as what?
# --------------------------------------------------------------------------------------
#
# Every one of these reads the slot state after t = 0..T passes, where t = 0 is the ENTRY
# state (zero passes), at the slot roles the task's `ElimSpec` names. Under strict geometry
# slot `r` after `t` passes holds spans r-t .. r, so WHICH facts a state could possibly
# hold is known exactly, and every instrument is scored against that window rather than
# against the whole instance.
#
# 2026-09-18 iteration 2: these were `eliminate`-only and are now spec-driven, so
# `eliminate` and `eliminate6` share one implementation. The membership probe gained two
# more facts (`in_set` and `dead` next to `alive`), which turns it into a hop-distance
# instrument; the `alive` probe is still built first with the same shape, so its numbers
# are unchanged. `adapted_reader` is new.


def _reachable(spec, role: int, t: int) -> dict:
    """What slot `role` can hold after t passes: its window is spans role-t .. role."""
    lo = role - t
    return {
        "cand_spans": [lo <= s <= role for s in spec.cand_spans],
        "all_cands": all(lo <= s <= role for s in spec.cand_spans),
        "elims": [lo <= s <= role for s in spec.elim_spans],
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
    """cands [B, n_alive] and the boolean masks over those columns."""
    cands = b["candidates"][:, k]
    surv = b["survivor"][:, k].unsqueeze(-1)
    is_surv = cands == surv
    is_elim = [cands == b["elim"][:, k, j].unsqueeze(-1) for j in range(b["elim"].shape[2])]
    return cands, is_surv, is_elim


@torch.no_grad()
def candidate_mass(model, batches, spec, depth: int = 6) -> dict:
    """Mass the TIED answer head puts on each candidate, per slot role and per pass.

    The softmax is restricted to the answer ids of the instance's own candidates, so it
    reads how the state splits its belief BETWEEN candidates. A superposition of the alive
    set reads entropy ln(alive) and zero mass on anything the state could already know is
    dead. A committed point reads entropy near 0 whatever the alive set is, which is why
    the entropy and the top mass are reported next to the three masses: the mean mass on
    the survivor alone cannot tell an n-way superposition from a uniformly random
    commitment (both give 1/n).
    """
    model.eval()
    roles = spec.roles
    acc = {(r, t): torch.zeros(5) for r in range(len(roles)) for t in range(depth + 1)}
    n = 0
    for b in batches:
        states = _slot_states(model, b, depth)
        for k, base in enumerate(b["inst_base_span"].tolist()):
            cands, is_s, is_e = _inst_masks(b, k)
            ids = spec.value_base + cands
            for ri, (_, role) in enumerate(roles):
                for t in range(depth + 1):
                    reach = _reachable(spec, role, t)
                    lg = model.mux_head(states[t][:, base + role]).gather(1, ids).float()
                    p = lg.softmax(-1)
                    known = torch.zeros_like(is_s)
                    for j, ok in enumerate(reach["elims"]):
                        if ok:
                            known = known | is_e[j]
                    alive_other = (~is_s) & (~known)
                    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
                    acc[(ri, t)] += torch.tensor(
                        [
                            (p * is_s).sum(-1).mean().item(),
                            (p * known).sum(-1).mean().item(),
                            (p * alive_other).sum(-1).mean().item(),
                            ent.mean().item(),
                            p.max(-1).values.mean().item(),
                        ]
                    )
            n += 1
    out = {}
    for ri, (name, role) in enumerate(roles):
        rows = []
        for t in range(depth + 1):
            a = (acc[(ri, t)] / n).tolist()
            reach = _reachable(spec, role, t)
            rows.append(
                {
                    "pass": t,
                    "slot": role,
                    "survivor": a[0],
                    "eliminated_reachable": a[1],
                    "alive_non_survivor": a[2],
                    "entropy": a[3],
                    "top_mass": a[4],
                    "n_alive_reachable": (spec.n_alive - sum(reach["elims"]))
                    if reach["all_cands"]
                    else None,
                }
            )
        out[name] = rows
    return out


@torch.no_grad()
def _probe_features(model, batches, spec, depth: int, role: int):
    """Per pass: the slot state and the three per-symbol facts, as float targets.

    alive  = the symbol is one of the instance's candidates and no reachable elimination
             has killed it. The fact the task is about.
    in_set = the symbol is one of the instance's candidates. Carried by the candidate
             spans, so its hop distance from the slot is fixed by the layout.
    dead   = the symbol is a candidate killed by an elimination inside the window. Its hop
             distance is one per elimination span, which is what makes the probe read as a
             function of hop distance.
    Where a fact's source span is NOT in the window the state cannot know it, so those
    rows are the instrument's own null and must read about 0.5.
    """
    X = [[] for _ in range(depth + 1)]
    Y = {k: [[] for _ in range(depth + 1)] for k in ("alive", "in_set", "dead")}
    S = [[] for _ in range(depth + 1)]
    nsym = spec.n_symbols
    for b in batches:
        states = _slot_states(model, b, depth)
        for t in range(depth + 1):
            reach = _reachable(spec, role, t)
            for k, base in enumerate(b["inst_base_span"].tolist()):
                cands, _, _ = _inst_masks(b, k)
                B = cands.shape[0]
                in_set = torch.zeros(B, nsym, dtype=torch.bool, device=cands.device)
                in_set.scatter_(1, cands, True)
                dead = torch.zeros_like(in_set)
                for j, ok in enumerate(reach["elims"]):
                    if ok:
                        dead.scatter_(1, b["elim"][:, k, j : j + 1], True)
                X[t].append(states[t][:, base + role].float())
                Y["alive"][t].append((in_set & ~dead).float())
                Y["in_set"][t].append(in_set.float())
                Y["dead"][t].append(dead.float())
                S[t].append(torch.stack([in_set, dead], dim=-1))
    return (
        [torch.cat(x) for x in X],
        {k: [torch.cat(y) for y in v] for k, v in Y.items()},
        [torch.cat(s) for s in S],
    )


def _fit_logistic(xn, y, steps, lr, dev, seed: int = 0):
    # a fixed seed per fit: the probe is then reproducible and independent of how many
    # other probes were fitted before it, which the 2026-09-18 version was not
    g = torch.Generator(device="cpu").manual_seed(seed)
    lin = torch.nn.Linear(xn.shape[1], y.shape[1])
    with torch.no_grad():
        bound = 1.0 / (xn.shape[1] ** 0.5)
        lin.weight.copy_(torch.empty_like(lin.weight).uniform_(-bound, bound, generator=g))
        lin.bias.copy_(torch.empty_like(lin.bias).uniform_(-bound, bound, generator=g))
    lin = lin.to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    loss = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xn), y)
        loss.backward()
        opt.step()
    return lin, float(loss.item())


def _new_linear(d_in, d_out, seed, dev):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lin = torch.nn.Linear(d_in, d_out)
    with torch.no_grad():
        bound = 1.0 / (d_in**0.5)
        lin.weight.copy_(torch.empty_like(lin.weight).uniform_(-bound, bound, generator=g))
        lin.bias.copy_(torch.empty_like(lin.bias).uniform_(-bound, bound, generator=g))
    return lin.to(dev)


def _fit_softmax(xn, y, n_out, steps, lr, dev, seed: int = 0, wd: float = 0.0):
    lin = _new_linear(xn.shape[1], n_out, seed, dev)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    loss = None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(lin(xn), y)
        loss.backward()
        opt.step()
    return lin, float(loss.item())


READER_WD = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _fit_reader(xn, y, n_out, steps, lr, dev, seed: int):
    """A ridge-selected linear reader: the weight decay is chosen on a held-out fifth of
    the TRAIN draw, then the reader is refitted on the whole draw at that decay.

    Measured 2026-09-19, which is why this exists: an unregularised fit of 96 features to
    22 classes on 1024 rows read a value CE of 2.504 on a RANDOM-INIT loop, against a
    chance of ln 8 = 2.0794. A reader that scores worse than chance on the held-out draw is
    measuring its own overfitting, not the state.
    """
    n = xn.shape[0]
    k = max(1, int(0.8 * n))
    g = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(xn.device)
    tr, va = perm[:k], perm[k:]
    best = (None, float("inf"))
    for i, wd in enumerate(READER_WD):
        lin, _ = _fit_softmax(xn[tr], y[tr], n_out, steps, lr, dev, seed=seed + i, wd=wd)
        with torch.no_grad():
            ce = torch.nn.functional.cross_entropy(lin(xn[va]), y[va]).item()
        if ce < best[1]:
            best = (wd, ce)
    lin, train_ce = _fit_softmax(xn, y, n_out, steps, lr, dev, seed=seed, wd=best[0])
    return lin, {"weight_decay": best[0], "val_ce": best[1], "train_ce": train_ce}


def _balanced(pred, y):
    """Balanced accuracy, or None when one class is empty (a fact with no positives, e.g.
    `dead` before any elimination is reachable). None, not NaN: NaN is not valid JSON and
    it silently poisons every comparison downstream."""
    hit = pred == y
    pos = hit[y].float().mean().item() if bool(y.any()) else None
    neg = hit[~y].float().mean().item() if bool((~y).any()) else None
    bal = None if (pos is None or neg is None) else 0.5 * (pos + neg)
    return bal, pos, neg


def membership_probe(
    model, train_batches, eval_batches, spec, depth: int = 6, steps: int = 400, lr: float = 3e-2
) -> dict:
    """Linear probes on the FROZEN model's slot states, scored on held-out rows.

    It answers a different question from `candidate_mass`: the mass reads whether the set
    is carried as a superposition of candidate EMBEDDINGS through the tied answer head, the
    probe reads whether the facts are linearly decodable in ANY encoding. A high probe
    accuracy under a low mass entropy is the "carried, but not as a superposition" outcome.

    Reported per (role, pass): balanced accuracy for `alive`, `in_set` and `dead`, plus
    `acc_within_set`, alive against already-eliminated inside the instance's own
    candidates. None when no elimination is inside the window.

    A random network preserves linearly decodable information, so a high accuracy here is
    NOT evidence that the loop uses a fact. The probe is a destruction test: a LOW accuracy
    is the evidence. The grid's random-init cells are its baseline.
    """
    dev = next(model.parameters()).device
    out = {}
    for name, role in spec.roles:
        Xtr, Ytr, _ = _probe_features(model, train_batches, spec, depth, role)
        Xte, Yte, Ste = _probe_features(model, eval_batches, spec, depth, role)
        rows = []
        for t in range(depth + 1):
            xtr = Xtr[t]
            mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
            xn = (xtr - mu) / sd
            xe = (Xte[t] - mu) / sd
            row = {"pass": t, "slot": role, "n_rows": int(Xte[t].shape[0])}
            # `alive` is fitted FIRST and with the same shape as the 2026-09-18 version, so
            # its numbers stay comparable with the cells filed that day
            for fi, fact in enumerate(("alive", "in_set", "dead")):
                lin, tl = _fit_logistic(
                    xn, Ytr[fact][t], steps, lr, dev, seed=1_000_000 + 1000 * role + 10 * t + fi
                )
                with torch.no_grad():
                    pred = lin(xe) > 0
                    y = Yte[fact][t] > 0
                    bal, pos, neg = _balanced(pred, y)
                row[f"acc_{fact}"] = bal
                row[f"train_loss_{fact}"] = tl
                row[f"n_pos_{fact}"] = int(y.sum().item())
                if fact == "alive":
                    row["acc_balanced"] = bal
                    row["acc_alive"] = pos
                    row["acc_other"] = neg
                    in_set, dead = Ste[t][..., 0], Ste[t][..., 1]
                    a_in, d_in = in_set & ~dead, in_set & dead
                    if d_in.any():
                        hit = pred == y
                        row["acc_within_set"] = 0.5 * (
                            hit[a_in].float().mean().item() + hit[d_in].float().mean().item()
                        )
                    else:
                        row["acc_within_set"] = None
            rows.append(row)
        out[name] = rows
    return out


@torch.no_grad()
def _reader_features(model, batches, spec, depth: int, role: int):
    X, Y = [], []
    for b in batches:
        states = _slot_states(model, b, depth)
        for k, base in enumerate(b["inst_base_span"].tolist()):
            X.append(states[depth][:, base + role].float())
            Y.append(spec.value_base + b["survivor"][:, k])
    return torch.cat(X), torch.cat(Y)


def adapted_reader(
    model,
    train_batches,
    eval_batches,
    spec,
    depths=(1, 2, 3, 4, 5, 6),
    steps: int = 400,
    lr: float = 3e-2,
) -> dict:
    """A linear reader FITTED to the exit state, against the coda's own reading.

    The coda reads z through a tied head it was trained with. If that head cannot recover
    the answer, two explanations remain: the state does not hold it, or the head cannot
    read it. This separates them. One linear map per (slot, forced depth), trained on one
    draw of rows with the model frozen and scored on a disjoint draw, reported as a value
    CE in nats so it sits next to the K-curve and the enumerated ceilings.
    """
    dev = next(model.parameters()).device
    out = {"by_depth": [], "by_slot": []}
    for d in depths:
        xtr, ytr = _reader_features(model, train_batches, spec, d, spec.answer_slot)
        xte, yte = _reader_features(model, eval_batches, spec, d, spec.answer_slot)
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
        lin, fit = _fit_reader(
            (xtr - mu) / sd, ytr, model.cfg.vocab, steps, lr, dev, seed=2_000_000 + 10 * d
        )
        with torch.no_grad():
            lo = lin((xte - mu) / sd)
            ce = torch.nn.functional.cross_entropy(lo, yte).item()
            acc = (lo.argmax(-1) == yte).float().mean().item()
        out["by_depth"].append(
            {"depth": d, "slot": spec.answer_slot, "reader_value_ce": ce, "reader_value_acc": acc,
             "n_rows": int(xte.shape[0]), "n_train_rows": int(xtr.shape[0]), **fit}
        )
    d = spec.probe_depth
    for name, role in spec.roles:
        xtr, ytr = _reader_features(model, train_batches, spec, d, role)
        xte, yte = _reader_features(model, eval_batches, spec, d, role)
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-5
        lin, fit = _fit_reader(
            (xtr - mu) / sd, ytr, model.cfg.vocab, steps, lr, dev, seed=3_000_000 + 10 * role
        )
        with torch.no_grad():
            lo = lin((xte - mu) / sd)
            out["by_slot"].append(
                {
                    "role": name,
                    "slot": role,
                    "depth": d,
                    "reader_value_ce": torch.nn.functional.cross_entropy(lo, yte).item(),
                    "reader_value_acc": (lo.argmax(-1) == yte).float().mean().item(),
                    **fit,
                }
            )
    return out


@torch.no_grad()
def twin_divergence(model, twin_pairs, spec, depth: int = 6, tol: float = 1e-3) -> dict:
    """Cosine between the slot states of two rows that differ ONLY at the FIRST elimination.

    A reachability test with teeth: the state of slot `role` may not move until that span
    enters its window, which happens at pass role - first_elim_span. A role at or before
    the changed span can never see it, so its cosine must be 1.0 at every pass -- but only
    WITHIN an instance: a row with several instances lets the core chain cross the
    boundary, so a role can reach an earlier instance's changed span at a larger pass.
    """
    model.eval()
    roles = spec.roles
    first_elim = spec.elim_spans[0]
    acc = {(r, t): 0.0 for r in range(len(roles)) for t in range(depth + 1)}
    n = 0
    for ba, bb in twin_pairs:
        sa = _slot_states(model, ba, depth)
        sb = _slot_states(model, bb, depth)
        for base in ba["inst_base_span"].tolist():
            for ri, (_, role) in enumerate(roles):
                for t in range(depth + 1):
                    c = torch.nn.functional.cosine_similarity(
                        sa[t][:, base + role].float(), sb[t][:, base + role].float(), dim=-1
                    )
                    acc[(ri, t)] += c.mean().item()
            n += 1
    out = {}
    for ri, (name, role) in enumerate(roles):
        cos = [acc[(ri, t)] / n for t in range(depth + 1)]
        drop = next((t for t, c in enumerate(cos) if c < 1.0 - tol), None)
        out[name] = {
            "slot": role,
            "cosine": cos,
            "drop_pass": drop,
            "expected_drop_pass": (role - first_elim) if role >= first_elim else None,
        }
    ce = [
        (
            eval_ce(model, [ba], force_depth=depth)["value_ce"],
            eval_ce(model, [bb], force_depth=depth)["value_ce"],
        )
        for ba, bb in twin_pairs
    ]
    out["value_ce"] = {
        "a": sum(c[0] for c in ce) / len(ce),
        "b": sum(c[1] for c in ce) / len(ce),
    }
    return out


def make_twin_pairs(cfg, n_rows, batch, seed, device, task="eliminate"):
    from tasks import make_twin_batch

    g = torch.Generator(device=device).manual_seed(seed)
    return [
        tuple(
            make_twin_batch(
                batch, cfg.layout.n_spans, cfg.layout.span_len, generator=g, device=device, task=task
            )
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
