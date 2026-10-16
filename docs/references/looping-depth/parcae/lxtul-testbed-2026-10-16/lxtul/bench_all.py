"""One harness that times every testbed arm the same way (training step and inference forward).

    python -m lxtul.bench_all --arm minimal --batch 6 --mode train --out /home/wolfe/parcae-runs/bench4
    python -m lxtul.bench_all --arm minimal --batch 6 --mode infer --out /home/wolfe/parcae-runs/bench4

Always run under the GPU lock (gpu_todo.sh does). One process per (arm, batch, mode); the result goes to
`<out>/<arm>_b<batch>.json` under the key `train` or `infer` (the file is read, updated and written back, so
the two modes of one cell share a file). `summarize_bench4.py` turns the folder into the paper tables.

Timing (train): the testbed's own step (lxtul/train.py): zero_grad, bf16 autocast forward, backward, grad
clip 1.0, MuonAdamW, EMA twin update (lxtul arms), compiled the way training compiles (`build(..., compile=True)`:
per-block inductor, plus the chunked CE and flex kernels). CUDA synced around every step; median of
`--steps` (200) steps after `--warmup` (100). Peak GiB is from the timed window. The phase split (forward,
backward, clip, optimizer, EMA) is a second pass of synced steps. Batches are the 16 cached REAL batches
(batch 6), cycled; batch 12 is two cached batches stacked. tok/s counts real tokens, 1024 per row, not
packed positions.

Timing (infer): eval mode, no grad, bf16 autocast, the forward the K-sweep runs at depth 6 plus the per-token NLL
(lxtul/evaluate.py::depth_sweep): plain Parcae `num_steps_pair=[6, 0]`, LXTUL and minimal `depth=6` passes per
slot, the flat GPT one pass. tok/s counts 1024 real tokens per row.

FLOPs: FlopCounterMode on an EAGER copy of the model, in a subprocess (lxtul/flop_tally.py patches the
attention entry points, which must not touch the timed process). TFLOP/step is forward + backward (the
backward includes activation-checkpoint recompute); the optimizer (Muon Newton-Schulz, analytic) is separate.
Inference gives forward TFLOP per real token. A cell that does not fit records status "OOM" with the peak
it reached; it is never dropped.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = str(Path(__file__).resolve().parent / "configs")
BATCH_DIR = Path("/home/wolfe/morph-scratch/abq/parcae-speed")      # 16 cached real batches of 6 rows
NATIVE_DIR = Path("/home/wolfe/morph-scratch/abq/pz-min")           # the same, packed at k cells per slot
SEQ = 1024                                                          # real tokens per row
EVAL_DEPTH = 6                                                      # the K-sweep's reference depth

# name -> (hydra arm, extra hydra overrides, kind note). Every speed key an arm can have is in its arm yaml,
# except the plain loop's, which are `plain.*` keys.
ARMS: dict[str, tuple[str, list[str]]] = {
    "gpt8": ("gpt8", []),
    "gpt8_pointer": ("gpt8_pointer", []),
    "flat14": ("flat14", []),
    "flat14_pointer": ("flat14_pointer", []),
    "plain_pointer_rc": ("plain_pointer", ["plain.row_compact=true"]),
    "plain_pointer": ("plain_pointer", []),
    "plain_pointer_rc_ckpt": ("plain_pointer", ["plain.row_compact=true", "plain.gradient_checkpointing=true"]),
    "plain_pointer_ckpt": ("plain_pointer", ["plain.gradient_checkpointing=true"]),
    "lxtul_pointer_fast": ("lxtul_pointer_fast", []),
    "lxtul_pointer_fast_nockpt": ("lxtul_pointer_fast_nockpt", []),
    "lxtul_pointer": ("lxtul_pointer", []),
    "minimal": ("minimal", []),
    "minimal_noptr": ("minimal_noptr", []),
    # SpiralFormer-style multi-resolution recursion on the plain blocks (lxtul/spiral.py); plain rows,
    # span layouts cut on the host once per cached batch
    "spiral_published": ("spiral_published", []),
    "spiral_span": ("spiral_span", []),
    "spiral_span_pair": ("spiral_span_pair", []),
    "spiral_span_pair_mesh": ("spiral_span_pair_mesh", []),
    # Poisson-depth chunk loops (chunks of 4, of 2, and of 1 = plain Parcae's loop with the Anchor skip)
    "spiral_chunk4_loop": ("spiral_chunk4_loop", []),
    "spiral_chunk2_loop": ("spiral_chunk2_loop", []),
    "spiral_token_skip_loop": ("spiral_token_skip_loop", []),
    # Block-Parcae: the series hierarchy (coda local to its chunk, reads embeddings + the previous chunk's loop state)
    "spiral_block4_loop": ("spiral_block4_loop", []),
    "spiral_block2_loop": ("spiral_block2_loop", []),
    "spiral_block4_depth1": ("spiral_block4_depth1", []),
    # ... with the loop states of all earlier chunks as a memory the coda attends over (spiral.block_memory)
    "spiral_blockmem4_loop": ("spiral_blockmem4_loop", []),
    "spiral_blockmem2_loop": ("spiral_blockmem2_loop", []),
    # published spiral with one level looped at Parcae's Poisson depth
    "spiral_pub_poisson2": ("spiral_pub_poisson2", []),
    "spiral_pub_poisson4": ("spiral_pub_poisson4", []),
    # the published spiral's whole descent repeated as an outer cycle (spiral.cycle): sampled schedule with end credit,
    # the same with HRM deep supervision, and the fixed-schedule control (coarsest start, no skips)
    "spiral_cycle_end": ("spiral_cycle_end", []),
    "spiral_cycle_deep": ("spiral_cycle_deep", []),
    "spiral_cycle_fixedsched": ("spiral_cycle_fixedsched", []),
    "spiral_cycle_deep_fixedsched": ("spiral_cycle_deep_fixedsched", []),
    # plain Parcae + pointer with HRM deep supervision on its loop (plain.deep_supervision, lxtul/plain_deep.py)
    "plain_deep": ("plain_pointer", ["plain.deep_supervision=true"]),
}
CELL_FIELDS = ("slot_mask", "bag_id", "slot_index", "slot_valid")


# ------------------------------------------------------------------ build and data
def make(arm: str, batch: int, compile_: bool):
    """-> (cfg, kind, model, opt, batches (on GPU), slot_id)."""
    from lxtul.build import build, make_optimizer
    from lxtul.data import tul_runtime
    hydra_arm, extra = ARMS[arm]
    with initialize_config_dir(CONF_DIR, version_base=None):
        cfg = compose("train", overrides=[f"arm={hydra_arm}", "run.name=bench", "run.wandb_mode=disabled", *extra])
    cfg.optimizer.lr = cfg.optimizer.muon_lr = 0.008 * math.sqrt(batch / 256)
    kind = str(cfg.arm.kind)
    torch._dynamo.config.recompile_limit = 64          # build() raises both for the spiral arms
    torch._dynamo.config.accumulated_recompile_limit = 256
    overrides = OmegaConf.to_container(cfg.arm.lxtul, resolve=True) if "lxtul" in cfg.arm else {}
    torch.manual_seed(1)
    model, mcfg, rt = build(cfg, kind, overrides=overrides, compile=compile_)
    opt = make_optimizer(model, cfg)
    batches = load_batches(kind, model, rt, mcfg, batch)
    slot_id = int(tul_runtime(mcfg).data_cfg.slot_id)
    return cfg, kind, model, opt, batches, slot_id


def _native_lxtul(rt, mcfg, k: int):
    """16 batches of 6 packed at k positions per slot by the production loader (cached on disk)."""
    path = NATIVE_DIR / f"batches_lxtul_k{k}.pt"
    if path.exists():
        return torch.load(path, weights_only=False)
    from lxtul.data import make_loader
    it = make_loader(mcfg, "train", 6, tul_rt=rt, prefetch=0)
    bl = [next(it) for _ in range(16)]
    assert all(b[0].shape[1] == int(mcfg.data.seq_len) + k * rt.data_cfg.max_slots for b in bl), \
        "native rows must be seq_len + k * max_slots long"
    torch.save(bl, path)
    return bl


def _cat_layout(a, b):
    """Two SlotLayouts of equal width stacked along the row axis (tensor fields with a row axis are
    concatenated; scalars and the stats dict come from the first)."""
    rows = a.slot_mask.shape[0]
    kw = {}
    for f in dataclasses.fields(a):
        va, vb = getattr(a, f.name), getattr(b, f.name)
        kw[f.name] = torch.cat([va, vb], 0) if isinstance(va, torch.Tensor) and va.dim() >= 1 and va.shape[0] == rows else va
    return type(a)(**kw)


def load_batches(kind: str, model, rt, mcfg, batch: int):
    """The cached real batches on the GPU. batch 6 = as cached; batch 12 = two cached batches stacked."""
    if batch % 6:
        raise ValueError("batch must be a multiple of 6 (the cache holds batches of 6 rows)")
    if kind == "lxtul":
        raw = torch.load(BATCH_DIR / "batches_lxtul.pt", weights_only=False)
        k = int(model.tul.cells)
        if k != int(raw[0][2].prefix_k):
            raw = _native_lxtul(rt, mcfg, k)
    else:                                              # plain, gpt and spiral read plain rows
        raw = [(x, y, None) for x, y in torch.load(BATCH_DIR / "batches_plain.pt", weights_only=False)]
    n = batch // 6
    out = []
    for i in range(0, len(raw) - n + 1, n):
        grp = raw[i:i + n]
        x, y = torch.cat([g[0] for g in grp]), torch.cat([g[1] for g in grp])
        lay = None
        if kind == "lxtul":
            lay = grp[0][2]
            for g in grp[1:]:
                lay = _cat_layout(lay, g[2])
            for f in CELL_FIELDS:
                setattr(lay, f, getattr(lay, f).cuda())
        if kind == "spiral":                           # cut once on the host, like the trainer's loader path
            lay = getattr(model, 'inner', model).span_layout(x).to("cuda")
        out.append((x.cuda(), y.cuda(), lay))
    return out


# ------------------------------------------------------------------ steps
def fwd_train(model, kind, batch, i):
    x, y, lay = batch
    if kind in ("plain", "spiral"):
        model.step = i
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind == "spiral":
            return model(x, labels=y, layout=lay)
        return model(x, labels=y) if kind in ("plain", "gpt") else model(x, y, lay)


def train_step(model, opt, kind, batch, i):
    opt.zero_grad(set_to_none=True)
    out = fwd_train(model, kind, batch, i)
    if not out.get("backward_done"):                  # deep supervision backpropagates inside the forward, cycle by cycle
        out["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if kind == "lxtul":
        model.ema_update()
    return out["loss"].detach()


@torch.no_grad()
def infer_step(model, kind, batch, slot_id, pair):
    """The K-sweep's forward at depth 6 plus the per-token NLL; returns the summed NLL (a device scalar)."""
    from lxtul.evaluate import _row_nll
    x, y, lay = batch
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind == "lxtul":
            h = model(x, None, lay, depth=EVAL_DEPTH)["hidden"]
            return model.token_nll(h, x, y, lay.slot_mask, lay).sum()
        if kind == "spiral":                       # depth sets the sampled-depth level's passes; None: none
            kw = {"layout": lay, "depth": EVAL_DEPTH if getattr(model, 'inner', model).has_sampled_depth else None}
            if hasattr(model, "pointer"):
                return model.token_nll(x, y, **kw).sum()
            return _row_nll(model(x, return_logits=True, **kw)["logits"], y, slot_id)[0].sum()
        if hasattr(model, "pointer"):
            kw = {"num_steps_pair": pair} if kind == "plain" else {}
            return model.token_nll(x, y, **kw).sum()
        kw = {"num_steps_pair": pair} if kind == "plain" else {}
        logits = model(x, return_logits=True, **kw)["logits"]
        return _row_nll(logits, y, slot_id)[0].sum()


def _mem() -> dict:
    return {"peak_alloc_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}


def _dyn_graphs():
    return torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0)


# ------------------------------------------------------------------ timing
def time_train(model, opt, kind, batches, batch, warmup, steps):
    losses = []
    for i in range(warmup):
        losses.append(train_step(model, opt, kind, batches[i % len(batches)], i))
    torch.cuda.synchronize()
    graphs0 = _dyn_graphs()
    torch.cuda.reset_peak_memory_stats()
    walls = []
    for i in range(warmup, warmup + steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        losses.append(train_step(model, opt, kind, batches[i % len(batches)], i))
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t)
    res = {"median_ms": 1e3 * statistics.median(walls), "mean_ms": 1e3 * statistics.mean(walls),
           "p5_ms": 1e3 * statistics.quantiles(walls, n=20)[0], "p95_ms": 1e3 * statistics.quantiles(walls, n=20)[-1],
           "n_steps": steps, "warmup": warmup, **_mem(), "graphs_compiled_in_window": _dyn_graphs() - graphs0}
    res["tok_s"] = batch * SEQ / (res["median_ms"] / 1e3)
    lv = torch.stack(losses).float().cpu().tolist()
    res["loss_first"], res["loss_last"], res["finite"] = lv[0], lv[-1], all(math.isfinite(v) for v in lv)
    res["phases_ms"] = time_phases(model, opt, kind, batches)
    return res


def time_phases(model, opt, kind, batches, steps=100, skip=40):
    """Median synced ms per phase over `steps - skip` steps (a second pass, after the timed window)."""
    ph: dict[str, list[float]] = {k: [] for k in ("forward", "backward", "clip", "optimizer", "ema", "total")}
    for i in range(steps):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        out = fwd_train(model, kind, batches[i % len(batches)], i)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        if not out.get("backward_done"):              # deep supervision: the backward ran inside the forward
            out["loss"].backward()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.cuda.synchronize(); t3 = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(); t4 = time.perf_counter()
        if kind == "lxtul":
            model.ema_update()
        torch.cuda.synchronize(); t5 = time.perf_counter()
        if i >= skip:
            for k, a, b in (("forward", t0, t1), ("backward", t1, t2), ("clip", t2, t3), ("optimizer", t3, t4),
                            ("ema", t4, t5), ("total", t0, t5)):
                ph[k].append(1e3 * (b - a))
    return {k: statistics.median(v) for k, v in ph.items()}


def time_infer(model, kind, batches, batch, slot_id, warmup, steps):
    model.eval()
    pair = torch.tensor([EVAL_DEPTH, 0], device="cuda")
    for i in range(warmup):
        infer_step(model, kind, batches[i % len(batches)], slot_id, pair)
    torch.cuda.synchronize()
    graphs0 = _dyn_graphs()
    torch.cuda.reset_peak_memory_stats()
    walls, nll = [], []
    for i in range(warmup, warmup + steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        nll.append(infer_step(model, kind, batches[i % len(batches)], slot_id, pair))
        torch.cuda.synchronize()
        walls.append(time.perf_counter() - t)
    res = {"median_ms": 1e3 * statistics.median(walls), "mean_ms": 1e3 * statistics.mean(walls),
           "p5_ms": 1e3 * statistics.quantiles(walls, n=20)[0], "p95_ms": 1e3 * statistics.quantiles(walls, n=20)[-1],
           "n_steps": steps, "warmup": warmup, "eval_depth": EVAL_DEPTH, **_mem(),
           "graphs_compiled_in_window": _dyn_graphs() - graphs0}
    res["tok_s"] = batch * SEQ / (res["median_ms"] / 1e3)
    res["finite"] = bool(torch.isfinite(torch.stack(nll)).all())
    return res


# ------------------------------------------------------------------ FLOPs (own process, eager model)
def count_flops(arm: str, batch: int, mode: str, n: int) -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from lxtul.flop_tally import install_attn_tally, mm_total, muon_flops
    cfg, kind, model, opt, batches, slot_id = make(arm, batch, compile_=False)
    tally = install_attn_tally(model)
    # ModuleTracker leaks its backward module stack through Embedding modules (int inputs never fire the
    # backward hook), which mis-files the rest of the backward; bypass module hooks for them.
    torch.nn.Embedding.__call__ = lambda self, ids: torch.nn.functional.embedding(ids, self.weight, self.padding_idx)
    pair = torch.tensor([EVAL_DEPTH, 0], device="cuda")
    if mode == "infer":
        model.eval()
    else:                                             # warm (flex / chunk compiles), uncounted
        for i in range(4):
            train_step(model, opt, kind, batches[i % len(batches)], 100 + i)
    torch.cuda.synchronize()
    runs = []
    for i in range(n):
        tally.fwd.clear(); tally.bwd.clear(); tally.acc.clear(); tally.calls.clear()
        tally.phase = "fwd"
        fc = FlopCounterMode(model, display=False)
        tally.mm = lambda fc=fc: mm_total(fc)
        b = batches[i % len(batches)]
        if mode == "infer":
            with fc:
                infer_step(model, kind, b, slot_id, pair)
            runs.append({"fwd_mm": mm_total(fc), "attn_fwd": sum(tally.fwd.values())})
            continue
        opt.zero_grad(set_to_none=True)
        with fc:
            out = fwd_train(model, kind, b, i)
            fwd_mm = mm_total(fc)
            tally.phase = "bwd"
            if not out.get("backward_done"):          # deep supervision: its backward is counted with the forward
                out["loss"].backward()
        runs.append({"fwd_mm": fwd_mm, "bwd_mm": mm_total(fc) - fwd_mm,
                     "attn_fwd": sum(tally.fwd.values()),          # includes checkpoint recompute
                     "attn_bwd": sum(tally.bwd.values()),
                     "passes": float(model.metrics["passes"]) if kind == "lxtul" and "passes" in model.metrics else None})
        train_step(model, opt, kind, b, i)                          # keep the optimizer state moving
    if mode == "infer":
        tot = [r["fwd_mm"] + r["attn_fwd"] for r in runs]
        return {"tflop_per_batch": statistics.mean(tot) / 1e12, "tflop_per_token": statistics.mean(tot) / 1e12 / (batch * SEQ),
                "tflop_min": min(tot) / 1e12, "tflop_max": max(tot) / 1e12, "n": n}
    tot = [r["fwd_mm"] + r["bwd_mm"] + r["attn_fwd"] + r["attn_bwd"] for r in runs]
    return {"tflop_fwd_bwd": statistics.mean(tot) / 1e12, "tflop_min": min(tot) / 1e12, "tflop_max": max(tot) / 1e12,
            "tflop_fwd_mm": statistics.mean(r["fwd_mm"] for r in runs) / 1e12,
            "tflop_bwd_mm": statistics.mean(r["bwd_mm"] for r in runs) / 1e12,
            "tflop_attn": statistics.mean(r["attn_fwd"] + r["attn_bwd"] for r in runs) / 1e12,
            "tflop_optimizer": muon_flops(opt) / 1e12,
            "mean_passes": (statistics.mean(r["passes"] for r in runs) if runs[0]["passes"] is not None else None), "n": n}


def flops_in_subprocess(arm: str, batch: int, mode: str, n: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "flops.json"
        p = subprocess.run([sys.executable, "-m", "lxtul.bench_all", "--arm", arm, "--batch", str(batch),
                            "--mode", mode, "--flops-only", "--flop-steps", str(n), "--out", str(out)],
                           capture_output=True, text=True)
        if out.exists():
            return json.loads(out.read_text())
        tail = (p.stdout + p.stderr)[-1500:]
        status = "OOM" if "OutOfMemoryError" in tail else "ERROR"
        return {"status": status, "log_tail": tail}


# ------------------------------------------------------------------ driver
def _merge_write(path: Path, key: str, payload: dict, meta: dict) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur.update(meta)
    cur[key] = payload
    path.write_text(json.dumps(cur, indent=1, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--mode", required=True, choices=["train", "infer"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--flop-steps", type=int, default=8)
    ap.add_argument("--flops-only", action="store_true", help="internal: count FLOPs, write --out as the JSON")
    a = ap.parse_args()
    if a.flops_only:
        try:
            res = count_flops(a.arm, a.batch, a.mode, a.flop_steps)
        except torch.OutOfMemoryError:
            res = {"status": "OOM", **_mem()}
        a.out.write_text(json.dumps(res))
        return
    a.out.mkdir(parents=True, exist_ok=True)
    flops = flops_in_subprocess(a.arm, a.batch, a.mode, a.flop_steps)     # before this process touches the GPU
    cfg, kind, model, opt, batches, slot_id = None, None, None, None, None, None
    res: dict
    try:
        cfg, kind, model, opt, batches, slot_id = make(a.arm, a.batch, compile_=True)
        meta = {"arm": a.arm, "batch": a.batch, "kind": kind, "hydra_arm": ARMS[a.arm][0], "overrides": ARMS[a.arm][1],
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "power_limit_w": os.popen("nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits").read().strip(),
                "real_tokens_per_row": SEQ}
        if "lxtul" in cfg.arm:
            meta["lxtul_overrides"] = OmegaConf.to_container(cfg.arm.lxtul, resolve=True)
        if kind == "spiral":                           # the span loop's real size on the cached batches
            lays = [b[2] for b in batches]
            meta["spiral"] = OmegaConf.to_container(cfg.arm.spiral, resolve=True)
            meta["mean_spans_pooled_per_row"] = float(torch.stack([l.n_src.float().mean() for l in lays]).mean())
            meta["mean_spans_found_per_row"] = float(torch.stack([l.n_bound.float().mean() for l in lays]).mean())
        if a.mode == "train":
            res = time_train(model, opt, kind, batches, a.batch, a.warmup, a.steps)
        else:
            res = time_infer(model, kind, batches, a.batch, slot_id, a.warmup, a.steps)
        res["status"] = "ok"
    except torch.OutOfMemoryError:
        meta = {"arm": a.arm, "batch": a.batch}
        res = {"status": "OOM", **_mem()}
    res["flops"] = flops
    if a.mode == "train" and res["status"] == "ok" and "tflop_fwd_bwd" in flops:
        tf = flops["tflop_fwd_bwd"] + flops["tflop_optimizer"]
        res["achieved_tflops"] = tf / (res["median_ms"] / 1e3)
        res["achieved_tflops_fwd_bwd"] = flops["tflop_fwd_bwd"] / (res["median_ms"] / 1e3)
    if a.mode == "infer" and res["status"] == "ok" and "tflop_per_batch" in flops:
        res["achieved_tflops"] = flops["tflop_per_batch"] / (res["median_ms"] / 1e3)
    _merge_write(a.out / f"{a.arm}_b{a.batch}.json", a.mode, res, meta)
    print(json.dumps({"arm": a.arm, "batch": a.batch, "mode": a.mode,
                      **{k: v for k, v in res.items() if not isinstance(v, (dict, list))}}), flush=True)


if __name__ == "__main__":
    main()
