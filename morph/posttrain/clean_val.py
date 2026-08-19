"""Re-evaluate a checkpoint on a GENUINELY held-out slice, over a block-aware σ grid.

Why this exists. The in-training val loader is `ds.skip(50_000)` on the SAME stream train
reads from sample 0. At OpenWebText's measured 1299 tokens/doc and 14336 tokens/step, train
crosses document 50,000 at step ~4,531 — so every in-training val point after that is
contaminated, and a 20k run reads 220,722 documents. Default skip here is 250,000, past the
end of a 20k run.

Also fixes a second flaw in the in-training grid: with boundaries
[0.002, 0.07575, 1.19772, 80.0], the σ values (0.1, 0.3, 1.0, 3.0) put THREE probes inside
the core block and never touch the coda at all (σ < 0.076). The default grid below probes
every block, including each one's geometric centre — blocks score best at their centre and
worst at their edges, which is what the γ overlap exists to smooth.

Usage:
    python -m morph.posttrain.clean_val --ckpt <path> --config-name db_b3 [--skip 250000]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

# Block boundaries for the default b3 partition are [0.002, 0.07575, 1.19772, 80.0].
# Geometric centres: coda sqrt(0.002*0.07575)=0.0123, core sqrt(0.07575*1.19772)=0.301,
# prelude sqrt(1.19772*80)=9.79. The grid brackets each centre with an edge probe.
DEFAULT_SIGMAS = (0.012, 0.03, 0.1, 0.301, 1.0, 3.0, 9.79, 30.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="db_b3")
    ap.add_argument("--skip", type=int, default=250_000,
                    help="documents to skip. MUST exceed what the run itself consumed: a 20k "
                         "step run at batch 14 x seq 1024 reads ~220,722 docs.")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--sigmas", type=float, nargs="*", default=list(DEFAULT_SIGMAS))
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from hydra import compose, initialize_config_dir
    cfgdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "configs")
    with initialize_config_dir(version_base=None, config_dir=cfgdir):
        cfg = compose(config_name=a.config_name)
    if a.eager:
        cfg.model.use_kernels = False
        cfg.model.hc_use_kernel = False

    from morph.model.transformer import MORPHTransformer
    from morph.training.data import create_dataloader
    from morph.training.db_setup import build_db_runtime, build_db_step, db_loss
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config, load_checkpoint

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    db = build_db_runtime(cfg)
    if db is None:
        raise SystemExit("this tool evaluates the DiffusionBlocks objective; db is off")

    model = MORPHTransformer(build_morph_config(cfg, tul=None))
    model.build_db_modules(db.model_cfg)
    apply_quantization(model, cfg)          # renames tensors — must precede the load
    model = model.to(dev).eval()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step_loaded, _m, _ok = load_checkpoint(a.ckpt, model, scaler, torch.device(dev))

    seq = int(cfg.data.seq_len)
    batch = int(cfg.training.batch_size)
    docs_read = step_loaded * batch * seq / 1299.0
    if a.skip < docs_read:
        raise SystemExit(
            f"--skip {a.skip} is INSIDE the training stream: this checkpoint's run had "
            f"already read ~{docs_read:.0f} documents by step {step_loaded}. Raise --skip "
            f"above that or the result is contaminated, which is the whole point of this tool.")

    loader = iter(create_dataloader(str(cfg.data.tokenizer), str(cfg.data.dataset),
                                   seq, batch, split="validation", skip_samples=a.skip))
    print(f"ckpt step {step_loaded} | skip {a.skip} docs (run read ~{docs_read:.0f}) | "
          f"{a.batches} batches x {len(a.sigmas)} sigmas", flush=True)

    acc: dict[float, list[float]] = {s: [] for s in a.sigmas}
    with torch.no_grad():
        for _ in range(a.batches):
            try:
                b = next(loader)
            except StopIteration:
                break
            x, y = b[0].to(dev), b[1].to(dev)
            for sg in a.sigmas:
                st = build_db_step(db, model, y, fixed_sigma=sg)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    out = model(x, db_step=st, db_precond=db.precond)
                    _l, mt = db_loss(out, st, db.precond, model)
                acc[sg].append(mt["db/ce"])

    rep = {"clean_val/ckpt": a.ckpt, "clean_val/step": step_loaded,
           "clean_val/skip_docs": a.skip, "clean_val/batches": a.batches}
    print(f"\n{'sigma':>8} {'block':>6} {'CE':>8} {'ppl':>10}")
    for sg in a.sigmas:
        if not acc[sg]:
            continue
        ce = sum(acc[sg]) / len(acc[sg])
        blk = int(db.schedule.block_of_sigma(torch.tensor([sg]))[0])
        name = ("prelude", "core", "coda")[blk]
        print(f"{sg:8.3f} {name:>6s} {ce:8.4f} {math.exp(min(ce, 20)):10.2f}")
        rep[f"clean_val/ce_sigma{sg:g}"] = ce
        rep[f"clean_val/block_sigma{sg:g}"] = name
    ces = [sum(v) / len(v) for v in acc.values() if v]
    rep["clean_val/ce_mean"] = sum(ces) / max(len(ces), 1)
    # The highest sigma is the near-context-only probe: c_skip -> 0, so the model predicts
    # from clean context alone. That is the number closest to ordinary next-token CE and the
    # only one worth putting anywhere near a baseline.
    hi = max(a.sigmas)
    rep["clean_val/ce_context_only_proxy"] = rep.get(f"clean_val/ce_sigma{hi:g}")
    print(f"\nmean {rep['clean_val/ce_mean']:.4f} | context-only proxy (sigma={hi:g}) "
          f"{rep['clean_val/ce_context_only_proxy']:.4f}")

    out = a.out or (os.path.splitext(a.ckpt)[0] + ".cleanval.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
