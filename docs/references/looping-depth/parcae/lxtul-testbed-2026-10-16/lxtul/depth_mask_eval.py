"""Where does the loop's depth value sit: at span boundaries, or on every token?

  python -m lxtul.depth_mask_eval RUN_DIR --out X.json [--depth 6] [--draws 3]

RUN_DIR is a finished plain or plain_pointer run (final.pt, config.json, sweep.json). Eval only.
The probe gives loop depth to only some positions. The recurrent block runs `--depth` passes; after
pass 1 a position takes the new state only if `deep[b, t]` is set, otherwise it keeps the state it had
at the end of the previous pass (so a frozen position keeps its pass-1 state). Frozen positions still
serve as keys and values in later passes, computed from their frozen state. Prelude, state init,
injection, coda and the pointer head are the model's own code: the mask is applied by wrapping
`Parcae.update_recurrent_state` for the duration of one forward (`depth_mask`), so no model file changes.

A boundary position is the LAST token of a span (MORPH's `BoundaryRule`, the packer's rule, cut causally
from the input ids of each row). Conditions, each over all eval rows:

  all_1               nobody goes past pass 1 (depth 1)
  all_D               everybody gets D passes (the model's own depth D)
  boundary_D          only boundary positions get D passes
  random_D            as many positions per row as boundary_D, drawn uniformly from the non-boundary
                      positions; one draw per seed 0..draws-1, reported as mean and per draw
  all_but_boundary_D  every position except boundary positions

Readouts (mean token CE in nats, the model's own CE incl. the pointer mixture, label -100 excluded), on
position sets that are the SAME for every condition, so the columns are paired:
  ce_all            every scored position
  ce_shallow        non-boundary positions (the positions boundary_D leaves at depth 1)
  ce_next4          the first 4 tokens of the span after each boundary t, i.e. t+1..t+4, cut at the next
                    boundary (that boundary token, the span's last, is included when it falls in the 4)
  ce_boundary_own   the boundary positions' own labels

The state init is random (Parcae `state_init`); every condition of a batch is run under the same torch
seed, so the init noise is shared and conditions differ only by the mask. Rows are the rows the trainer's
sweep scored (`evaluate.eval_rows`, `evaluate.plain_view`). Gate: `all_1` and `all_D` ce_all must match the
run's own sweep.json at depth 1 and D within 2e-3 and score the same token count, else "gate": "FAIL".
"""
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from lxtul.data import morph_cfg
from lxtul.evaluate import eval_rows, plain_view
from lxtul.gap_probe import load_run
from lxtul.plain_pointer import PlainPointer

NEXT_K = 4
GATE_TOL = 2e-3
SETS = ("all", "shallow", "next4", "boundary_own")


def parcae_of(model):
    """The Parcae inside a plain or plain_pointer model. Anything without a recurrent block raises."""
    inner = model.inner if isinstance(model, PlainPointer) else model
    if not hasattr(inner, "update_recurrent_state"):
        raise ValueError(f"{type(inner).__name__} has no recurrent block: the depth mask needs a plain arm")
    return inner


@contextlib.contextmanager
def depth_mask(model, deep: torch.Tensor):
    """Inside the block, passes 2..D of the recurrence update only the positions with `deep[b, t]`.

    `deep` is a bool [B, L] on the model's device. Pass 1 (step 0) is untouched. Exact: a selection,
    no arithmetic. The wrapper is an instance attribute, removed on exit."""
    p = parcae_of(model)
    orig = p.update_recurrent_state

    def masked(x, input_embeds, freqs_cis, mask, step, total_steps, **kw):
        new = orig(x, input_embeds, freqs_cis, mask, step, total_steps, **kw)
        if int(step) == 0:
            return new
        if deep.shape != new.shape[:2]:
            raise ValueError(f"deep mask {tuple(deep.shape)} does not match the state {tuple(new.shape)}")
        return torch.where(deep[..., None], new, x)

    p.update_recurrent_state = masked
    try:
        yield
    finally:
        del p.update_recurrent_state


def boundary_mask(ids: torch.Tensor, rule) -> torch.Tensor:
    """[B, L] bool, True at the last token of each span. `rule.cut` is causal and resumable, so a
    position's flag depends only on the ids up to it."""
    out = torch.zeros(ids.shape, dtype=torch.bool)
    arr = ids.cpu().numpy()
    for b in range(arr.shape[0]):
        pos, _ = rule.cut(arr[b])
        out[b, torch.from_numpy(pos)] = True
    return out


def next_k_mask(bnd: torch.Tensor, k: int = NEXT_K) -> torch.Tensor:
    """Positions t+1..t+k after each boundary t, stopping at the next boundary (included)."""
    out = torch.zeros_like(bnd)
    L = bnd.shape[1]
    for b, t in bnd.nonzero().tolist():
        for pos in range(t + 1, min(t + k + 1, L)):
            out[b, pos] = True
            if bnd[b, pos]:
                break
    return out


def random_like(bnd: torch.Tensor, valid: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Per row, as many positions as `bnd` has, uniform over the valid non-boundary positions."""
    cand = valid & ~bnd
    score = torch.rand(bnd.shape, generator=gen)
    score[~cand] = -1.0
    n = torch.minimum(bnd.sum(1), cand.sum(1))
    rank = score.argsort(1, descending=True).argsort(1)
    return (rank < n[:, None]) & cand


def position_nll(model, xp, yp, depth: int, deep, slot_id: int, amp: bool, seed: int) -> torch.Tensor:
    """[B, L] NLL under the model's own scoring (0 where the label is -100) at `depth` passes, with
    `deep` (bool [B, L] or None) gating passes 2..depth."""
    torch.manual_seed(seed)                                     # the same state init for every condition
    pair = torch.tensor([depth, 0], device=xp.device)
    ctx = depth_mask(model, deep) if deep is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx, torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        if isinstance(model, PlainPointer):
            return model.token_nll(xp, yp, num_steps_pair=pair)
        logits = model(xp, return_logits=True, num_steps_pair=pair)["logits"].float()
        logits[..., slot_id] = float("-inf")
        nll = F.cross_entropy(logits.flatten(0, 1), yp.flatten().clamp_min(0), reduction="none")
        return torch.where(yp >= 0, nll.view(yp.shape), 0.0)


def condition_masks(bnd: torch.Tensor, depth: int) -> dict[str, tuple[int, torch.Tensor | None]]:
    """name -> (passes, deep mask) for the fixed conditions. `all_1` runs one pass and needs no mask
    (the all-False mask gives the same result, which the CPU test checks)."""
    return {"all_1": (1, None),
            f"all_{depth}": (depth, torch.ones_like(bnd)),
            f"boundary_{depth}": (depth, bnd),
            f"all_but_boundary_{depth}": (depth, ~bnd)}


def run_probe(model, rows, rule, slot_id: int, depth: int, draws: int, amp: bool, device: str, log=print) -> dict:
    gens = [torch.Generator().manual_seed(d) for d in range(draws)]
    acc: dict[str, dict[str, list[float]]] = {}                 # cond -> set -> [sum, count]
    nset = dict.fromkeys(SETS, 0)
    for i, (x, y, lay) in enumerate(rows):
        xp, yp = plain_view(x, y, lay, slot_id)
        valid = yp >= 0
        bnd = boundary_mask(xp, rule) & valid
        sets = {"all": valid, "shallow": valid & ~bnd, "next4": next_k_mask(bnd) & valid, "boundary_own": bnd}
        for s, m in sets.items():
            nset[s] += int(m.sum())
        conds = condition_masks(bnd, depth)
        for d in range(draws):
            conds[f"random_{depth}/draw{d}"] = (depth, random_like(bnd, valid, gens[d]))
        xd, yd = xp.to(device), yp.to(device)
        dsets = {s: m.to(device) for s, m in sets.items()}
        for name, (passes, deep) in conds.items():
            nll = position_nll(model, xd, yd, passes, None if deep is None else deep.to(device),
                               slot_id, amp, seed=1000 + i)
            a = acc.setdefault(name, {s: [0.0, 0] for s in SETS})
            for s, m in dsets.items():
                a[s][0] += float((nll.double() * m).sum())
        if (i + 1) % 10 == 0:
            log(f"batch {i + 1}/{len(rows)}", flush=True)
    ce = {c: {f"ce_{s}": acc[c][s][0] / nset[s] for s in SETS} for c in acc}
    rand = [f"random_{depth}/draw{d}" for d in range(draws)]
    ce[f"random_{depth}"] = {"mean": {k: float(np.mean([ce[r][k] for r in rand])) for k in ce[rand[0]]},
                             "draws": {r.split("/")[1]: ce.pop(r) for r in rand}}
    return {"n": {f"n_{s}": nset[s] for s in SETS}, "ce": ce}


def gate(res: dict, sweep: dict, depth: int) -> dict:
    cnt = float(np.sum(sweep["count"]))
    ref = {k: float(np.sum(sweep["sums"][str(k)]) / cnt) for k in (1, depth)}
    got = {1: res["ce"]["all_1"]["ce_all"], depth: res["ce"][f"all_{depth}"]["ce_all"]}
    ok = all(abs(got[k] - ref[k]) <= GATE_TOL for k in ref) and int(cnt) == res["n"]["n_all"]
    return {"gate": "PASS" if ok else "FAIL", "tol": GATE_TOL,
            "probe_ce": {str(k): v for k, v in got.items()}, "sweep_ce": {str(k): v for k, v in ref.items()},
            "probe_tokens": res["n"]["n_all"], "sweep_tokens": int(cnt)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--draws", type=int, default=3)
    a = ap.parse_args()
    if a.depth < 2:
        raise ValueError("--depth must be at least 2: the mask acts from pass 2")
    sweep = json.loads((a.run / "sweep.json").read_text())
    if str(a.depth) not in sweep["sums"]:
        raise ValueError(f"sweep.json has depths {sweep['depths']}, not {a.depth}: the gate needs it")
    model, kind, cfg, rt = load_run(a.run)
    if kind != "plain":
        raise ValueError(f"{a.run.name} is a {kind} run: the depth mask is for plain and plain_pointer runs")
    model = model.cuda().eval()
    slot_id = int(rt.data_cfg.slot_id)
    rule = rt.val_data_cfg.rule
    rows = eval_rows(morph_cfg(str(cfg.morph.config)), rt, int(cfg.eval.rows), int(cfg.eval.batch_size))
    res = run_probe(model, rows, rule, slot_id, a.depth, a.draws, amp=True, device="cuda")
    res.update(gate(res, sweep, a.depth))
    res.update(run=a.run.name, depth=a.depth, draws=a.draws, rows=int(cfg.eval.rows))
    a.out.write_text(json.dumps(res, indent=2))
    hdr = f"{'condition':<26}" + "".join(f"{'ce_' + s:>16}" for s in SETS)
    print(hdr)
    for c, v in res["ce"].items():
        v = v["mean"] if "mean" in v else v
        print(f"{c:<26}" + "".join(f"{v['ce_' + s]:16.4f}" for s in SETS))
    print("n", res["n"])
    print(f"GATE {res['gate']}: probe {res['probe_ce']} sweep {res['sweep_ce']} "
          f"tokens {res['probe_tokens']}/{res['sweep_tokens']}", flush=True)


if __name__ == "__main__":
    main()
