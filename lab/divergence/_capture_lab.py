"""Shared setup for the onset-capture lab probes (probe_state_diff, probe_grad_diff,
probe_alloc_diff): load a checkpoint the way the trainer would in deterministic mode,
pack one slot batch from the validation stream, and compare gradient snapshots."""
from __future__ import annotations

import os
import sys

import torch

from _build import ROOT, build_cfg

sys.path.insert(0, f"{ROOT}/scripts")
from tul_samples import load_ckpt  # noqa: E402

from morph.model.tul_layout import pack_tul_batch  # noqa: E402
from morph.training.data import create_dataloader  # noqa: E402
from morph.training.tul_setup import build_tul_runtime  # noqa: E402


def require_deterministic_env() -> None:
    if not os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise SystemExit("export CUBLAS_WORKSPACE_CONFIG=:4096:8 first")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def load_model_and_batch(config: str, ckpt: str, batch: int, device: str = "cuda"):
    """(model in train mode, x, y, layout, step) — eager kernels, compile off."""
    cfg = build_cfg(config, ["model.use_kernels=false", "training.compile=false"])
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit(f"{config} has no TUL block: there is no slot loop to measure")
    model, step = load_ckpt(cfg, ckpt, device, tul_rt.model_cfg if tul_rt else None)
    model.train()
    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    loader = create_dataloader(cfg.data.tokenizer, cfg.data.dataset, 2048, 8,
                               split="validation", skip_samples=0, bag_size=0, tul=None)
    buf: list[int] = []
    need = batch * (spec.l_total + 1)
    while len(buf) < need:
        buf.extend(next(loader)[0].reshape(-1).tolist())
    x, y, layout = pack_tul_batch(buf, tul_rt.data_cfg.rule, spec, batch)
    return model, x.to(device), y.to(device), layout.to(device), step


def fwd_bwd(model, x, y, layout, rng, *, rank: bool = True, cot: bool = True):
    """One trainer-shaped forward+backward from a pinned RNG state. Returns (loss, grads)."""
    root = getattr(model, "_orig_mod", model)
    root._probe_loop = True
    root._probe_rank = rank
    root._probe_cot = cot
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(rng[0]); torch.cuda.set_rng_state(rng[1])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, labels=y, bag_size=0, slot_layout=layout)
    out["loss"].backward()
    grads = {n: (None if p.grad is None else p.grad.detach().clone())
             for n, p in root.named_parameters()}
    return float(out["loss"].detach()), grads


def cmp_grads(ga: dict, gb: dict) -> tuple[float, str | None]:
    """(max |a-b| over every parameter gradient, first differing name)."""
    worst, first = 0.0, None
    for n in ga:
        a, b = ga[n], gb[n]
        if (a is None) != (b is None):
            return float("inf"), n
        if a is None:
            continue
        d = float((a.float() - b.float()).abs().max())
        if d > worst:
            worst = d
        if d > 0 and first is None:
            first = n
    return worst, first
