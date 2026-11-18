"""Hop accuracy: exact match at the labelled answer positions, by (family, hops) and by eval loop depth.

    python -m lxtul.hops.eval --run /home/wolfe/parcae-runs/<run> --depths 1,2,3,4,6,8,12,16

At every answer position the model's FULL output distribution is formed the way the NLL scores it (the LM
head, then the pointer mixture when the arm has one: `PointerHead.mixed_many`), and the answer counts as
right when its argmax is the label. Also read: the mean p(correct). The forward is the model's own, called as
`evaluate.depth_sweep` calls it, for all three kinds: lxtul (`model(x, None, lay, depth=k)`), plain
(`num_steps_pair=[k, 0]`) and gpt (no depth: scored once). The eval rows are `HopTask.eval_rows` (fixed seed
the training stream never uses), so every arm scores the same documents; TUL rows carry their hop labels
through the packer (`base.pack_row`), flat rows through `base.plain_row`.

Trainer hook (`HopEval`): at each `val_every`, a small subset at the arm's trained eval depth, logged as
`val/hop/<family>/acc_h<h>` and `val/hop/<family>/acc_mean` (the mean over that family's hop buckets); at the end
the full table over `task.eval.depths`, written to `hops.json` and logged to wandb.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from lxtul.hops.base import FAMILY_NAME, HopRow
from lxtul.hops.mix import HopTask


@dataclass
class HopBatch:
    """One eval batch. TUL kinds: `x`, `y` packed, `lay` the SlotLayout. Flat kinds: `lay` is None.
    `hops`, `family` are aligned to `y`."""
    x: torch.Tensor
    y: torch.Tensor
    lay: object
    hops: torch.Tensor
    family: torch.Tensor


def make_batches(task: HopTask, kind: str, rt, rows: list[HopRow], batch_size: int) -> list[HopBatch]:
    if len(rows) % batch_size:
        raise ValueError(f"{len(rows)} hop eval rows is not a multiple of the eval batch size {batch_size}")
    spec = None if kind in ("plain", "gpt") else task.spec(rt.data_cfg.prefix_k)
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i: i + batch_size]
        if spec is None:
            x, y, hops, fam = task.plain_batch(chunk)
            out.append(HopBatch(x, y, None, hops, fam))
        else:
            x, y, lay, hops, fam = task.tul_batch(chunk, spec)
            out.append(HopBatch(x, y, lay, hops, fam))
    return out


def _to(b: HopBatch, device) -> HopBatch:
    lay = b.lay
    if lay is not None:
        lay = dataclasses.replace(lay, slot_mask=lay.slot_mask.to(device), bag_id=lay.bag_id.to(device),
                                  slot_index=lay.slot_index.to(device), slot_valid=lay.slot_valid.to(device))
    return HopBatch(b.x.to(device), b.y.to(device), lay, b.hops.to(device), b.family.to(device))


@torch.no_grad()
def answer_scores(model, kind: str, b: HopBatch, depth: int, slot_id: int):
    """-> (pred [Q], p_correct [Q], label [Q], hops [Q], family [Q]) at the Q labelled answer positions of `b`
    (already on the model's device): `pred` the argmax of the model's full output distribution."""
    sel = (b.hops > 0).nonzero(as_tuple=True)
    if sel[0].numel() == 0:
        e = b.y.new_zeros(0)
        return e, b.y.new_zeros(0, dtype=torch.float32), e, e, e
    label = b.y[sel]
    if (label < 0).any():
        raise AssertionError("an answer position carries label -100: a hop label drifted off its target")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=b.x.is_cuda):      # the pointer head runs inside it too
        if kind == "lxtul":
            if model.tul.copy_cache:
                raise NotImplementedError("hop eval reads the pointer mixture, not the copy cache")
            h = model(b.x, None, b.lay, depth=depth)["hidden"]
            lg = (h[sel] @ model.lm_head.weight.T).float() * model.config.init.logit_scale
            is_slot, layout, mask_slot = b.lay.slot_mask, b.lay, True
            pointer = model.pointer if model.tul.pointer_heads > 0 else None
        else:
            kw = {} if kind == "gpt" else {"num_steps_pair": torch.tensor([depth, 0], device=b.x.device)}
            if hasattr(model, "pointer"):                    # PlainPointer: the head reads ln_f through a hook
                if hasattr(model, "step"):
                    model.inner.step = model.step
                logits = model.inner(b.x, return_logits=True, **kw)["logits"]
                h, model._h = model._h, None
                pointer = model.pointer if model.use_pointer else None
            else:
                logits, h, pointer = model(b.x, return_logits=True, **kw)["logits"], None, None
            lg = logits[sel].float()
            is_slot, layout = torch.zeros_like(b.y, dtype=torch.bool), None
            mask_slot = not hasattr(model, "pointer")           # as each NLL path does (PlainPointer does not)
        if mask_slot:
            lg[:, slot_id] = float("-inf")
        logp = torch.log_softmax(lg, -1)
        if pointer is not None:
            q_tok = (~is_slot).cumsum(1)[sel] - 1               # the query's index in token order
            logp = pointer.mixed_many(h, b.y, is_slot, logp, sel[0], q_tok, layout)
    p = logp.gather(1, label[:, None])[:, 0].exp()
    return logp.argmax(1), p, label, b.hops[sel].long(), b.family[sel].long()


@torch.no_grad()
def hop_accuracy(model, kind: str, batches: list[HopBatch], depths: list[int], slot_id: int, device="cuda") -> dict:
    """Exact match and mean p(correct) per (family, hops) and per eval depth.

    -> {"depths": [...], "buckets": {"<depth>": {"<family>/h<hops>": {"n", "acc", "p"}}},
        "family_mean": {"<depth>": {"<family>": macro mean of acc over the family's hop buckets}},
        "balanced": {"<depth>": `balanced_reading`}}.
    A flat model has no depth: it is scored once and the result repeated for every depth."""
    was = model.training
    model.eval()
    res = {"depths": [int(d) for d in depths], "buckets": {}, "family_mean": {}, "balanced": {}}
    flat = None
    for k in depths:
        if kind == "gpt" and flat is not None:
            res["buckets"][str(k)], res["family_mean"][str(k)], res["balanced"][str(k)] = flat
            continue
        acc: dict[tuple[int, int], list[float]] = {}
        for b in batches:
            pred, p, label, hops, fam = answer_scores(model, kind, _to(b, device), int(k), slot_id)
            ok = (pred == label).double().cpu().numpy()
            for f, h, o, pp in zip(fam.tolist(), hops.tolist(), ok, p.double().cpu().numpy()):
                a = acc.setdefault((f, h), [0.0, 0.0, 0.0])
                a[0] += 1
                a[1] += o
                a[2] += pp
        buckets = {f"{FAMILY_NAME[f]}/h{h}": {"n": int(a[0]), "acc": a[1] / a[0], "p": a[2] / a[0]}
                   for (f, h), a in sorted(acc.items())}
        fam_mean = {}
        for name in {k_.split("/")[0] for k_ in buckets}:
            fam_mean[name] = float(np.mean([v["acc"] for k_, v in buckets.items() if k_.split("/")[0] == name]))
        bal = balanced_reading(buckets)
        res["buckets"][str(k)], res["family_mean"][str(k)], res["balanced"][str(k)] = buckets, fam_mean, bal
        flat = (buckets, fam_mean, bal) if kind == "gpt" else None
    model.train(was)
    return res


def balanced_reading(buckets: dict) -> dict:
    """The Yes / No tasks read so that always answering Yes (or No) scores 0.5. `reach_balanced` = mean of (reach_yes
    accuracy averaged over its distance buckets, reach_no accuracy pooled over its answers); `reach_balanced/h<d>`
    pairs ONE distance bucket of reach_yes with the pooled reach_no; `edge_balanced` likewise for edge_yes / edge_no.
    A key is absent when a family has no answer."""
    def fam(name):
        return {k.split("/")[1]: v for k, v in buckets.items() if k.split("/")[0] == name}

    def pooled(b):
        n = sum(v["n"] for v in b.values())
        return sum(v["n"] * v["acc"] for v in b.values()) / n
    out = {}
    for task, yes, no in (("reach", "reach_yes", "reach_no"), ("edge", "edge_yes", "edge_no")):
        y, n = fam(yes), fam(no)
        if not y or not n:
            continue
        out[f"{task}_balanced"] = 0.5 * (float(np.mean([v["acc"] for v in y.values()])) + pooled(n))
        if task == "reach":
            out |= {f"reach_balanced/{h}": 0.5 * (v["acc"] + pooled(n)) for h, v in y.items()}
    return out


def val_metrics(res: dict, depth: int) -> dict:
    """The trainer's wandb keys for one depth of a `hop_accuracy` result."""
    m = {f"val/hop/{k.split('/')[0]}/acc_{k.split('/')[1]}": v["acc"] for k, v in res["buckets"][str(depth)].items()}
    m |= {f"val/hop/{f}/acc_mean": a for f, a in res["family_mean"][str(depth)].items()}
    m |= {f"val/hop/{k}": v for k, v in res["balanced"][str(depth)].items() if "/" not in k}
    return m


class HopEval:
    """Fixed hop eval rows of one task, packed for one arm kind: the `val` subset (first
    `val_rows_per_bucket` rows of every bucket) and the `full` set."""

    def __init__(self, task: HopTask, kind: str, rt, batch_size: int, slot_id: int, device="cuda"):
        self.task, self.kind, self.slot_id, self.device = task, kind, int(slot_id), device
        rows = task.eval_rows()
        self.full = make_batches(task, kind, rt, [r for _, r in rows], batch_size)
        self.val = make_batches(task, kind, rt, task.val_rows(rows), batch_size)
        self.n_rows = len(rows)

    def val_log(self, model, depth: int) -> dict:
        return val_metrics(hop_accuracy(model, self.kind, self.val, [depth], self.slot_id, self.device), depth)

    def final(self, model, depths: list[int] | None = None) -> dict:
        res = hop_accuracy(model, self.kind, self.full, list(depths or self.task.depths), self.slot_id, self.device)
        res["manifest"] = self.task.manifest | {"rows": self.n_rows, "rows_per_bucket": self.task.rows_per_bucket}
        return res


def write_final(res: dict, out: Path, run=None) -> None:
    """`hops.json` in the run dir; with a wandb run, the table and flat summary keys `hop/<family>/acc_h<h>_d<k>`."""
    (out / "hops.json").write_text(json.dumps(res, indent=2))
    if run is None:
        return
    import wandb
    table = wandb.Table(columns=["depth", "family", "hops", "n", "acc", "p"])
    for k, buckets in res["buckets"].items():
        for name, v in buckets.items():
            fam, h = name.split("/")
            table.add_data(int(k), fam, int(h[1:]), v["n"], v["acc"], v["p"])
            run.summary[f"hop/{fam}/acc_{h}_d{k}"] = v["acc"]
        for fam, a in res["family_mean"][k].items():
            run.summary[f"hop/{fam}/acc_mean_d{k}"] = a
        for name, a in res["balanced"][k].items():
            run.summary[f"hop/{name}_d{k}"] = a
    run.log({"hop/table": table})


def main() -> None:
    from lxtul.data import morph_cfg_of
    from lxtul.gap_probe import load_run
    from lxtul.hops.mix import build_task
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--depths", default="", help="comma list; default: task.eval.depths of the run's config")
    ap.add_argument("--out", type=Path, default=None, help="default: <run>/hops.json")
    a = ap.parse_args()
    model, kind, cfg, rt = load_run(a.run)
    task = build_task(cfg.get("task", {"kind": "none"}), morph_cfg_of(cfg), rt)
    if task is None:
        raise SystemExit(f"{a.run} was trained without a hop task (task.kind is none)")
    depths = [int(d) for d in a.depths.split(",")] if a.depths else None
    res = HopEval(task, kind, rt, int(cfg.eval.batch_size), int(rt.data_cfg.slot_id)).final(model, depths)
    out = a.out or a.run / "hops.json"
    out.write_text(json.dumps(res, indent=2))
    for k in res["depths"]:
        print(f"depth {k}: " + "  ".join(f"{f} {v:.3f}" for f, v in sorted(res["family_mean"][str(k)].items())))
    print(f"[hops] wrote {out}")


if __name__ == "__main__":
    main()
