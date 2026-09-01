"""Gate-value probe: what did the GRT recurrence gate converge to?

Prereg: lab/experiments/planned/2026-08-30-tul-gate-value-probe.md. One eval pass
per checkpoint; a forward hook on ``tul_recur_gate`` collects g per iteration.
Reports mean/p10/p50/p90 over ACTIVE-slot elements and the per-iteration means
(does g drift open across t?). No training, no weight edits.
"""
from __future__ import annotations

import argparse
import json
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=CONFIG=PATH")
    ap.add_argument("--rows", type=int, default=48)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = a.device

    from morph.model.tul_layout import pack_tul_batch
    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    results: dict[str, dict] = {}
    for triple in a.ckpt:
        label, config, path = triple.split("=", 2)
        cfg = build_cfg(config, ["model.use_kernels=false"])
        tul_rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, path if path.startswith("/") else f"{ROOT}/{path}",
                                device, tul_rt.model_cfg if tul_rt else None)
        model.eval()
        if model.tul_recur_gate is None:
            raise RuntimeError(f"{label}: model has no recurrence gate")
        spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
        rule = tul_rt.data_cfg.rule
        loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                                   split="validation", skip_samples=0, bag_size=0, tul=None)
        buf: list[int] = []
        need = a.batch * (spec.l_total + 1)
        gs: list[torch.Tensor] = []          # one [B,S,n,d] per gate call, this batch
        per_iter: list[list[float]] = []     # per-batch list of per-iteration means

        def hook(mod, inp, out):
            gs.append(out.detach())

        handle = model.tul_recur_gate.register_forward_hook(hook)
        allg: list[torch.Tensor] = []
        try:
            rows_done = 0
            while rows_done < a.rows:
                while len(buf) < need:
                    buf.extend(next(loader)[0].reshape(-1).tolist())
                inp, labels, layout = pack_tul_batch(buf, rule, spec, a.batch)
                layout = layout.to(device)
                gs.clear()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                     enabled=device == "cuda"):
                    model.tul_forward_ablated(inp.to(device), None, layout,
                                              plan_mode="normal")
                valid = layout.slot_valid  # [B,S]
                it_means = []
                for g in gs:
                    vm = valid.view(*valid.shape, *([1] * (g.dim() - 2)))
                    sel = g[vm.expand_as(g)].float()
                    it_means.append(float(sel.mean()))
                    allg.append(sel.cpu())
                per_iter.append(it_means)
                rows_done += a.batch
        finally:
            handle.remove()
        flat = torch.cat(allg)
        # torch.quantile caps at ~16M elements; a 1M uniform subsample is exact
        # to well past the reported precision.
        sub = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]] \
            if flat.numel() > 1_000_000 else flat
        q = torch.quantile(sub, torch.tensor([0.10, 0.50, 0.90]))
        n_iters = max(len(x) for x in per_iter)
        iter_means = [float(torch.tensor([x[t] for x in per_iter if len(x) > t]).mean())
                      for t in range(n_iters)]
        results[label] = {
            "step": step, "rows": a.rows,
            "mean": float(flat.mean()), "p10": float(q[0]), "p50": float(q[1]),
            "p90": float(q[2]), "min": float(flat.min()), "max": float(flat.max()),
            "per_iteration_mean": iter_means, "n_elements": int(flat.numel()),
        }
        print(f"{label}: mean={flat.mean():.4f} p10={q[0]:.4f} p50={q[1]:.4f} "
              f"p90={q[2]:.4f} per-iter={['%.3f' % v for v in iter_means]}", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
