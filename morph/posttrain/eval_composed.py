"""Evaluate the COMPOSED 3-block Euler chain — the inference procedure itself.

The gap this closes. Under `mode="b3"` every training and validation forward runs exactly
ONE section, so nothing measured so far says anything about the composed system: the actual
inference walks sigma_max -> sigma_min, invoking whichever block owns each sigma, starting
from PURE NOISE. That chain had never been run against real labels.

What is measured:
  * CE of the composed chain's final logits vs the true next tokens. If the chain works this
    is meaningfully below ln V; if it is ~ln V the method does not compose at all, and no
    amount of single-block val CE would have told us.
  * The same at several step counts (arm DB-13) — the denoiser conditions on sigma, not
    d-sigma, so step count is a free dial needing no retraining.
  * A SINGLE-SHOT reference: one forward at sigma_min given the true noised target. That is
    the number the val grid reports. The gap between it and the composed chain is the cost of
    actually having to walk from noise.
  * ||denoised|| per Euler step (risk R9): the softmax @ E bridge is a convex combination of
    embedding rows, so its norm collapses toward the table mean when the model is unsure.
    A trajectory decaying toward 0 is the model failing to commit, and it is invisible in CE.

Held-out by default (skip 250_000 docs, past what a 20k run reads).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config-name", default="db_b3")
    ap.add_argument("--skip", type=int, default=250_000)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--steps", type=int, nargs="*", default=[4, 8, 16, 32])
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

    from morph.inference.db_generate import SampleTrace, db_sample
    from morph.model.transformer import MORPHTransformer
    from morph.training.data import create_dataloader
    from morph.training.db_setup import build_db_runtime, build_db_step, db_loss
    from morph.training.quant_setup import apply_quantization
    from morph.training.train import build_morph_config, load_checkpoint

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    db = build_db_runtime(cfg)
    if db is None:
        raise SystemExit("db is off in this config")

    model = MORPHTransformer(build_morph_config(cfg, tul=None))
    model.build_db_modules(db.model_cfg)
    apply_quantization(model, cfg)
    model = model.to(dev).eval()
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    step_loaded, _m, _ok = load_checkpoint(a.ckpt, model, scaler, torch.device(dev))

    seq = int(cfg.data.seq_len)
    batch = int(cfg.training.batch_size)
    V = int(cfg.model.vocab_size)
    lnv = math.log(V)
    loader = iter(create_dataloader(str(cfg.data.tokenizer), str(cfg.data.dataset),
                                   seq, batch, split="validation", skip_samples=a.skip))
    print(f"ckpt step {step_loaded} | held-out skip {a.skip} | ln V = {lnv:.3f} | "
          f"logit_scale = {float(model.db_logit_scale.detach().exp()):.4f}", flush=True)

    def ce_of(logits, labels):
        return float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, V).float(), labels.reshape(-1)))

    rep = {"composed/ckpt": a.ckpt, "composed/step": step_loaded, "composed/lnV": lnv}
    acc_chain = {n: [] for n in a.steps}
    acc_single, acc_norm_first, acc_norm_last = [], [], []

    with torch.no_grad():
        for bi in range(a.batches):
            try:
                b = next(loader)
            except StopIteration:
                break
            x, y = b[0].to(dev), b[1].to(dev)

            # Single-shot reference: one forward at the LOWEST grid sigma with the TRUE
            # noised target. This is what the val grid reports.
            st = build_db_step(db, model, y, fixed_sigma=min(db.val_sigmas))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                o = model(x, db_step=st, db_precond=db.precond, db_want_logits=True)
            acc_single.append(ce_of(o["logits"], y))

            # Composed chain from PURE NOISE at several step counts.
            for n in a.steps:
                tr = SampleTrace()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    lg, _z = db_sample(model, x, db, n_steps=n,
                                       generator=torch.Generator(device=dev).manual_seed(bi),
                                       trace=tr)
                acc_chain[n].append(ce_of(lg, y))
                if n == max(a.steps) and tr.denoised_norm:
                    acc_norm_first.append(tr.denoised_norm[0])
                    acc_norm_last.append(tr.denoised_norm[-1])

    def mean(v):
        return sum(v) / max(len(v), 1)

    print(f"\n{'what':>28} {'CE':>8} {'ppl':>11} {'vs lnV':>8}")
    s = mean(acc_single)
    print(f"{'single-shot @ sigma_min':>28} {s:8.4f} {math.exp(min(s,20)):11.2f} "
          f"{lnv - s:+8.3f}")
    rep["composed/single_shot_ce"] = s
    for n in a.steps:
        c = mean(acc_chain[n])
        print(f"{f'composed chain, {n} steps':>28} {c:8.4f} {math.exp(min(c,20)):11.2f} "
              f"{lnv - c:+8.3f}")
        rep[f"composed/chain_ce_{n}steps"] = c

    nf, nl = mean(acc_norm_first), mean(acc_norm_last)
    print(f"\nR9 ||denoised||: first step {nf:.4f} -> last {nl:.4f} "
          f"(ratio {nl / max(nf, 1e-9):.3f})")
    rep["composed/denoised_norm_first"] = nf
    rep["composed/denoised_norm_last"] = nl

    best = min(mean(acc_chain[n]) for n in a.steps)
    print(f"\nVERDICT: best composed CE {best:.4f} vs ln V {lnv:.3f}")
    if best > lnv - 0.5:
        print("  THE CHAIN DOES NOT WORK. Starting from noise it is at ~chance, so the")
        print("  single-block val numbers describe a system that cannot actually generate.")
    elif best > s + 2.0:
        print("  The chain works but is FAR worse than single-shot: composing the blocks")
        print("  loses most of what each block knows in isolation.")
    else:
        print("  The chain composes: generating from noise is close to single-shot.")
    rep["composed/best_chain_ce"] = best
    rep["composed/works"] = bool(best <= lnv - 0.5)

    out = a.out or (os.path.splitext(a.ckpt)[0] + ".composed.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
