"""Train and instrument ONE grid cell of the toy slot-loop study.

    python run_cell.py --task compose --attach exit --seed 0 --out cell.json

Everything the cell used is written into the JSON under "config", so a run is
reproducible from its own artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instruments import (  # noqa: E402
    candidate_mass,
    eval_ce,
    gradient_probe,
    k_curve,
    make_eval_batches,
    make_twin_pairs,
    membership_probe,
    participation_rank,
    twin_divergence,
    write_contribution,
)
from model import Layout, ToyConfig, ToySlotLoop  # noqa: E402
from tasks import chance_ce, eliminate_ceilings, make_batch, vocab_for  # noqa: E402

EVAL_SEED = 20260910


def build_cfg(a) -> ToyConfig:
    return ToyConfig(
        vocab=vocab_for(a.task),
        d_model=a.d_model,
        n_heads=a.n_heads,
        d_ff=a.d_ff,
        n_prelude=a.n_prelude,
        n_coda=a.n_coda,
        layout=Layout(a.n_spans, a.span_len, a.prefix_k),
        mean_depth=a.mean_depth,
        max_depth=a.max_depth,
        fixed_depth=a.fixed_depth,
        entry=a.entry,
        noise_scale=a.noise_scale,
        inject_decay=a.inject_decay,
        attach=a.attach,
        mux_weight=a.mux_weight,
        progressive_p=a.progressive_p,
        bptt_last=a.bptt_last,
        fixed_point_lambda=a.fixed_point_lambda,
        pass_lora_rank=a.pass_lora_rank,
        coda_reads_z=not a.coda_blind,
        coda_token_input=a.coda_token_input,
        geometry=a.geometry,
    )


def lr_at(step, total, base, warmup):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="compose", choices=["compose", "summary", "eliminate"])
    ap.add_argument("--attach", default="exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--d_model", type=int, default=96)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--n_prelude", type=int, default=2)
    ap.add_argument("--n_coda", type=int, default=2)
    ap.add_argument("--n_spans", type=int, default=8)
    ap.add_argument("--span_len", type=int, default=3)
    ap.add_argument("--prefix_k", type=int, default=2)
    ap.add_argument("--mean_depth", type=float, default=6.0)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--fixed_depth", type=int, default=0)
    ap.add_argument("--entry", default="prelude")
    ap.add_argument("--noise_scale", type=float, default=0.5)
    ap.add_argument("--inject_decay", type=float, default=0.45)
    ap.add_argument("--mux_weight", type=float, default=1.0)
    ap.add_argument("--progressive_p", type=float, default=0.5)
    ap.add_argument("--bptt_last", type=int, default=0)
    ap.add_argument("--fixed_point_lambda", type=float, default=0.0)
    ap.add_argument("--pass_lora_rank", type=int, default=0)
    ap.add_argument("--coda_blind", action="store_true")
    ap.add_argument("--coda_token_input", default="embed", choices=["embed", "prelude"])
    ap.add_argument("--geometry", default="strict", choices=["strict", "permissive"])
    ap.add_argument("--eval_rows", type=int, default=2048)
    ap.add_argument("--eval_batch", type=int, default=256)
    ap.add_argument("--probe_batches", type=int, default=4)
    ap.add_argument("--probe_rows", type=int, default=1024)   # eliminate instruments
    ap.add_argument("--probe_steps", type=int, default=400)   # membership-probe fit steps
    ap.add_argument("--probe_batch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--label", default="cell")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    dev = torch.device(a.device)
    cfg = build_cfg(a)
    model = ToySlotLoop(cfg).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    gen = torch.Generator(device=dev).manual_seed(a.seed * 7919 + 13)

    # a small fixed probe draw, used to time the escape from the one-pass solution
    probe_g = torch.Generator(device=dev).manual_seed(EVAL_SEED + 2)
    probe = [make_batch(a.task, 256, cfg.layout.n_spans, cfg.layout.span_len, generator=probe_g, device=dev)]

    t0 = time.time()
    hist = []
    model.train()
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a.steps, a.lr, a.warmup)
        b = make_batch(a.task, a.batch, cfg.layout.n_spans, cfg.layout.span_len, generator=gen, device=dev)
        out = model(b, generator=gen)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            ev6 = eval_ce(model, probe, force_depth=6)
            model.train()
            hist.append(
                {
                    "step": step,
                    "loss": out["loss"].item(),
                    "token_ce": out["token_ce"].item(),
                    "mux": out["mux"].item(),
                    "grad_norm": float(gn),
                    "probe_value_acc": ev6["value_acc"],
                    "probe_value_ce": ev6["value_ce"],
                }
            )
    train_s = time.time() - t0

    ev = make_eval_batches(a.task, cfg, a.eval_rows, a.eval_batch, EVAL_SEED, dev)
    probe_ev = make_eval_batches(
        a.task, cfg, a.probe_batches * a.probe_batch, a.probe_batch, EVAL_SEED + 1, dev
    )

    kc = k_curve(model, ev, depths=(1, 2, 3, 6, 8, 12))
    wc = write_contribution(model, ev, depth=6)
    pr = participation_rank(model, ev, depth=6)
    probes = {s: gradient_probe(model, probe_ev, depth=6, source=s) for s in ("total", "token_ce", "mux")}

    elim: dict = {}
    if a.task == "eliminate":
        # the alive-set instruments: is the set carried, in what encoding, and when does
        # the state commit? Training rows for the membership probe are drawn from their own
        # seed and are disjoint from the held-out rows it is scored on.
        tr = make_eval_batches(a.task, cfg, a.probe_rows, a.eval_batch, EVAL_SEED + 3, dev)
        te = make_eval_batches(a.task, cfg, a.probe_rows, a.eval_batch, EVAL_SEED + 4, dev)
        twins = make_twin_pairs(cfg, a.probe_rows, a.eval_batch, EVAL_SEED + 5, dev)
        elim = {
            "ceilings": eliminate_ceilings(),
            "candidate_mass": candidate_mass(model, te, depth=6),
            "membership_probe": membership_probe(model, tr, te, depth=6, steps=a.probe_steps),
            "twin_divergence": twin_divergence(model, twins, depth=6),
        }

    esc = [h["step"] for h in hist if h["probe_value_acc"] > 0.9]
    res = {
        "label": a.label,
        "escaped": bool(esc),
        "escape_step": esc[0] if esc else None,
        "task": a.task,
        "seed": a.seed,
        "n_params": n_params,
        "train_seconds": train_s,
        "device": str(dev),
        "chance_ce": chance_ce(),
        "config": {**{k: v for k, v in asdict(cfg).items() if k != "layout"}, "layout": asdict(cfg.layout)},
        "args": vars(a),
        "history": hist,
        "final": eval_ce(model, ev, force_depth=6),
        "k_curve": kc,
        "k1_k6_value": kc["1"]["value_ce"] - kc["6"]["value_ce"],
        "k3_k6_value": kc["3"]["value_ce"] - kc["6"]["value_ce"],
        "k1_k6_mux": kc["1"]["mux_ce"] - kc["6"]["mux_ce"],
        "write_contribution": wc,
        "participation": pr,
        "gradient_probe": probes,
    }
    if elim:
        res["eliminate"] = elim
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(
        f"[{a.label}] task={a.task} seed={a.seed} "
        f"value_ce@6={res['final']['value_ce']:.4f} acc={res['final']['value_acc']:.3f} "
        f"K1-K6={res['k1_k6_value']:+.4f} cancel={probes['total']['cancellation']:.3f} "
        f"esc={res['escape_step']} "
        f"selfcheck={probes['total']['selfcheck_max_rel_err']:.2e} {train_s:.0f}s"
    )
    if elim:
        cm = elim["candidate_mass"]
        td = elim["twin_divergence"]
        print(
            f"[{a.label}] mass(cand,p0) surv/elim/other "
            f"{cm['cand'][0]['survivor']:.3f}/{cm['cand'][0]['eliminated_reachable']:.3f}/"
            f"{cm['cand'][0]['alive_non_survivor']:.3f} H={cm['cand'][0]['entropy']:.3f} | "
            f"mass(elim1,p1) {cm['elim1'][1]['survivor']:.3f}/{cm['elim1'][1]['eliminated_reachable']:.3f}/"
            f"{cm['elim1'][1]['alive_non_survivor']:.3f} H={cm['elim1'][1]['entropy']:.3f} | "
            f"probe(elim1,p1) within={elim['membership_probe']['elim1'][1]['acc_within_set']} "
            f"| twin drop cand/elim1/answer "
            f"{td['cand']['drop_pass']}/{td['elim1']['drop_pass']}/{td['answer']['drop_pass']}"
        )


if __name__ == "__main__":
    main()
