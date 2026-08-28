"""TUL-FM P1 gate — the retrieval probe.

Doctrine: ``docs/tul-fm-probing.md`` §3. This probe REPLACED the blind decoder, which
refused twice (memorisation at 1.6k examples, underfitting at 41k;
``lab/experiments/failures/2026-08-28-plan-content.md``). It has no trained component, so
there is no fit/eval split to get wrong and nothing to overfit.

THE QUESTION. Generate the plan ``ẑ_i`` for every valid slot with the Euler ladder, then
ask whether ``ẑ_i`` ranks its OWN next-span target ``y_i`` first among the pooled targets
it competes with. Report top-1, top-5, MRR and the chance floor, at TWO candidate scopes:

* **batch-wide** — every valid target in the batch. Chance ``1/N``, N stated.
* **within-row** — only the targets of the SAME row. This deletes the document cue, which
  batch-wide retrieval would otherwise hand the planner for free (rows are different
  documents; "which document is this" is not the question P1 asks). It is the sharp
  number and the one the pre-registered gate is written against.

THE TWO CONTROLS, IN THE SAME INVOCATION, ON THE SAME BATCHES:

* ``untrained`` — a freshly initialised planner. ``out`` is zero-init, so ``F_θ ≡ 0`` and
  the ladder is a pure noise walk: this must sit at chance. It is the FLOOR. A probe on
  which the untrained planner scores above chance is measuring the target geometry, not
  the planner, and every number below it is void.
* ``shuffled_ctx`` — slot ``i`` conditioned on ANOTHER ROW's frozen states, with its own
  geometry and its own target. This must drop toward chance. It separates "the planner
  read this row's context" from "the planner learned the corpus-average next span".

THE COLLAPSE GUARD — TWO NUMBERS, NEVER ONE. ``effective_rank`` (participation ratio of
the CENTERED target covariance) says how many directions the targets vary along;
``mean_pairwise_cos`` says how far apart they actually are. Effective rank alone is
fooled by a tight cluster — 200 near-copies of one vector still vary along every axis —
and the cosine alone cannot tell a two-cluster space from a full one. The arc note lists
target gaming as the failure this probe cannot see by itself; these two are the guard.

Usage (standalone; writes JSON):

    PYTHONPATH=$PWD python -m lab.tulfm.retrieval_probe \\
        --planner checkpoints/tulfm/p1/step_4000.pt \\
        --out lab/experiments/results/2026-08-28-tulfm-p1/probe.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import Tensor

from lab.tulfm.fm_planner import (
    FMPlanner,
    FMPlannerConfig,
    SpanGeometry,
    build_schedule,
    effective_rank,
    generate_plans,
    mean_pairwise_cos,
    pool_targets,
    segment_rows,
)

__all__ = ["retrieval_scores", "row_index_of_valid", "probe_batch", "run_probe"]


# ── scoring ──────────────────────────────────────────────────────────────────

def retrieval_scores(zhat: Tensor, y: Tensor, valid: Tensor,
                     row_of: Tensor | None = None) -> dict:
    """Rank every valid slot's generated plan against the pooled targets it competes with.

    Args:
        zhat:   ``[B, S, d]`` generated plans.
        y:      ``[B, S, d]`` pooled unit-norm targets.
        valid:  ``[B, S]`` bool.
        row_of: ``[N]`` row index per valid slot. ``None`` → the candidate set is EVERY
                valid target in the batch. Given → the candidate set is restricted to the
                slot's OWN row (see below).

    WHY WITHIN-ROW MATTERS. Batch-wide retrieval can be won by topic alone: rows come
    from different documents, so "which document is this" is most of the signal and
    "which span of that document" is none of it. A planner that only learned the document
    would still post a large multiple of ``1/N``. Restricting the candidates to the same
    row deletes the document cue and leaves only the question P1 actually asks. Report
    both; the within-row number is the sharp one.

    Cosine, because the targets are unit-norm by construction and the plan's own scale is
    not part of the claim.
    """
    q = F.normalize(zhat[valid].float(), dim=-1)          # [N, d]
    k = F.normalize(y[valid].float(), dim=-1)             # [N, d]
    n = q.shape[0]
    if n < 2:
        raise ValueError(f"retrieval needs >= 2 valid slots, got {n}")

    sim = q @ k.T                                         # [N, N]
    gold = torch.arange(n, device=sim.device)
    gold_score = sim[gold, gold][:, None]
    if row_of is None:
        allowed = torch.ones_like(sim, dtype=torch.bool)
    else:
        allowed = row_of[:, None] == row_of[None, :]
    n_cand = allowed.sum(dim=1)
    keep = n_cand >= 2                                     # a 1-candidate row is vacuous
    if not bool(keep.any()):
        raise ValueError("no query has >= 2 candidates; the retrieval question is empty")

    # Rank 1 = best. Ties count against us (>= not >), so a degenerate all-equal
    # similarity matrix scores the worst rank, not the best.
    rank = ((sim >= gold_score) & allowed).sum(dim=1)[keep].float()
    n_cand_f = n_cand[keep].float()
    return {
        "n_candidates": float(n_cand_f.mean().item()),
        "n_queries": int(keep.sum().item()),
        "chance": float((1.0 / n_cand_f).mean().item()),
        "top1": float((rank == 1).float().mean().item()),
        "top5": float((rank <= 5).float().mean().item()),
        "mrr": float((1.0 / rank).mean().item()),
        "median_rank": float(rank.median().item()),
    }


def row_index_of_valid(valid: Tensor) -> Tensor:
    """``[N]`` row index for each valid slot, in the order ``y[valid]`` yields them."""
    B, S = valid.shape
    return torch.arange(B, device=valid.device)[:, None].expand(B, S)[valid]


def probe_batch(planner: FMPlanner, untrained: FMPlanner, h_ctx: Tensor,
                geom: SpanGeometry, schedule, n_steps: int,
                generator: torch.Generator | None = None) -> dict:
    """All three conditions plus the collapse guard, on ONE batch.

    The same ``generator`` state is NOT reused across conditions on purpose — each
    condition draws its own ladder noise. What IS held fixed is the batch, the geometry
    and the targets, which is what makes the three numbers comparable.
    """
    y = pool_targets(h_ctx, geom)
    out = {
        "slots": {
            "n_valid": int(geom.valid.sum().item()),
            "n_spans_total": int(geom.n_spans_total),
            "dropped_frac": float(geom.dropped_fraction),
            "n_dropped_budget": int(geom.n_dropped_budget),
        },
        "target_effective_rank": effective_rank(y, geom.valid),
        "target_mean_pairwise_cos": mean_pairwise_cos(y, geom.valid),
        "target_dim": int(y.shape[-1]),
        "target_norm_mean": float(y[geom.valid].norm(dim=-1).mean().item()),
    }

    rows = row_index_of_valid(geom.valid)

    if h_ctx.shape[0] < 2:
        raise ValueError("shuffled-context control needs batch >= 2")
    conds = {
        "trained": h_ctx,
        # Shuffled context: row b's slots read row (b+1 mod B)'s frozen states. The
        # geometry and the targets stay row b's, so the ONLY thing that changed is what
        # the planner was allowed to look at.
        "shuffled_ctx": h_ctx.roll(1, dims=0),
    }
    for name, h in conds.items():
        z = generate_plans(planner, h, geom, schedule, n_steps=n_steps,
                           generator=generator)
        out[name] = retrieval_scores(z, y, geom.valid)
        out[name + "_within_row"] = retrieval_scores(z, y, geom.valid, row_of=rows)

    z0 = generate_plans(untrained, h_ctx, geom, schedule, n_steps=n_steps,
                        generator=generator)
    out["untrained"] = retrieval_scores(z0, y, geom.valid)
    out["untrained_within_row"] = retrieval_scores(z0, y, geom.valid, row_of=rows)
    return out


def _mean_of(dicts: list[dict], path: tuple[str, ...]) -> float:
    vals = []
    for d in dicts:
        cur = d
        for p in path:
            cur = cur[p]
        vals.append(float(cur))
    return sum(vals) / len(vals)


# ── standalone entry point ───────────────────────────────────────────────────

def run_probe(planner_ckpt: str, n_batches: int, batch_size: int, seq_len: int,
              seed: int, device_str: str, out_path: str | None,
              backbone_ckpt: str | None = None) -> dict:
    """Load a planner checkpoint, rebuild its backbone and rule, and score the gate.

    The planner checkpoint carries the resolved P1 config and the resolved BACKBONE
    config, so the probe cannot silently score a planner against a different backbone,
    a different tokenizer or a different span rule than it was trained on. ``--backbone``
    overrides the checkpoint path only (never the architecture).
    """
    from omegaconf import OmegaConf

    from lab.tulfm.train_p1 import build_backbone, make_loader
    from morph.training.tul_setup import build_boundary_rule

    device = torch.device(device_str)
    torch.manual_seed(seed)

    blob = torch.load(planner_ckpt, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(blob["cfg"])
    bcfg = OmegaConf.create(blob["backbone_cfg"])
    if backbone_ckpt is not None:
        cfg.backbone.checkpoint = backbone_ckpt

    backbone = build_backbone(cfg, bcfg, device)
    rule, _lut, _eos, _subs = build_boundary_rule(bcfg)

    pcfg = FMPlannerConfig(**blob["planner_cfg"])
    planner = FMPlanner(pcfg).to(device)
    planner.load_state_dict(blob["planner"])
    planner.eval()

    torch.manual_seed(seed + 1)
    untrained = FMPlanner(pcfg).to(device).eval()

    schedule = build_schedule(float(cfg.sigma.p_mean), float(cfg.sigma.p_std),
                              float(cfg.sigma.sigma_data))
    loader = make_loader(bcfg, seq_len, batch_size,
                         skip_samples=int(cfg.data.val_skip_samples))

    gen = torch.Generator(device=device).manual_seed(seed)
    per_batch = []
    for _ in range(n_batches):
        ids = next(loader)[0].to(device)
        geom = segment_rows(ids, rule, pcfg.max_slots)
        with torch.no_grad():
            h = backbone.prelude_states(
                ids, apply_input_norm=bool(cfg.backbone.apply_input_norm)).float()
        per_batch.append(probe_batch(planner, untrained, h, geom, schedule,
                                     int(cfg.sigma.infer_steps), generator=gen))

    summary = {
        "planner_ckpt": planner_ckpt,
        "backbone_ckpt": cfg.backbone.checkpoint,
        "step": int(blob.get("step", -1)),
        "n_batches": n_batches,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "seed": seed,
        "planner_cfg": asdict(pcfg),
        "infer_steps": int(cfg.sigma.infer_steps),
        "target_effective_rank": _mean_of(per_batch, ("target_effective_rank",)),
        "target_mean_pairwise_cos": _mean_of(per_batch, ("target_mean_pairwise_cos",)),
        "chance": _mean_of(per_batch, ("trained", "chance")),
        "dropped_frac": _mean_of(per_batch, ("slots", "dropped_frac")),
    }
    for cond in ("trained", "untrained", "shuffled_ctx",
                 "trained_within_row", "untrained_within_row", "shuffled_ctx_within_row"):
        summary[cond] = {m: _mean_of(per_batch, (cond, m))
                         for m in ("top1", "top5", "mrr", "median_rank", "n_candidates",
                                   "chance")}
    summary["per_batch"] = per_batch

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[probe] wrote {out_path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="TUL-FM P1 retrieval probe")
    ap.add_argument("--planner", required=True, help="planner checkpoint (.pt)")
    ap.add_argument("--backbone", default=None, help="override the backbone ckpt path")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="JSON output path")
    a = ap.parse_args()

    s = run_probe(a.planner, a.batches, a.batch_size, a.seq_len, a.seed, a.device,
                  a.out, backbone_ckpt=a.backbone)
    print(f"\n  chance(batch-wide)={s['chance']:.4f}  "
          f"target_eff_rank={s['target_effective_rank']:.2f} / {s['planner_cfg']['d_ctx']}  "
          f"target_pairwise_cos={s['target_mean_pairwise_cos']:.4f}")
    print(f"  {'condition':<26} {'N':>7} {'chance':>8} {'top1':>8} {'top5':>8} "
          f"{'mrr':>8} {'med rank':>9}")
    for cond in ("trained", "untrained", "shuffled_ctx",
                 "trained_within_row", "untrained_within_row", "shuffled_ctx_within_row"):
        r = s[cond]
        print(f"  {cond:<26} {r['n_candidates']:>7.1f} {r['chance']:>8.4f} "
              f"{r['top1']:>8.4f} {r['top5']:>8.4f} {r['mrr']:>8.4f} "
              f"{r['median_rank']:>9.1f}")


if __name__ == "__main__":
    main()
