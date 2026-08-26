"""Cost benchmark for the K-candidate coda arm (2026-08-25-gradient-flow-soft-min-arm).

The planned arm runs the CODA region K=4 times per step (K candidates folded into the
batch dim) while the prelude and core run once. Before building it we need: what
fraction f of a training step is coda + loss, the estimated arm/baseline step-time
ratio at K=4, and whether it fits in 32 GB.

Method (per regime):
  Four fwd+bwd variants, timed with CUDA events, regions made identity via the SAME
  ablation contextmanager `region_shapley.py` uses (monkeypatched block.forward /
  `_apply_core_step`). Deltas give per-region fwd+bwd cost INCLUDING backward:

      full                      -> t_full
      core off                  -> t_core_off        core   = t_full - t_core_off
      core+prelude off          -> t_cp_off          prelude = t_core_off - t_cp_off
      core+prelude+coda off     -> t_cpc_off         coda   = t_cp_off - t_cpc_off
      loss microbench (head+CE fwd+bwd on a random carrier of the real shape)
                                -> t_loss            front  = t_cpc_off - t_loss

  f = (coda + loss) / t_full.  f_upper = t_cp_off / t_full (includes the front/embed
  work the arm would NOT replicate -> conservative).

Model: randomly initialized (NOT a checkpoint — pure cost benchmark; stated in the
README), built exactly as lab/divergence/_build.py builds it (quantization applied,
same config compose). Depth is PINNED: `_sample_slot_depths` is replaced with a
deterministic constant (--depth, default max_depth=8) for ALL variants. Rationale:
the loop is a masked update over the full compact slot sequence, so its cost is
total_iters = max over B*max_slots Poisson(6) draws clamped to 8 — which is 8 with
probability ~1 at 384 slots. Pinning 8 reproduces the realized training iteration
count deterministically.

Usage:
  python lab/divergence/bench_coda_k.py --regime eager   --batch 6
  python lab/divergence/bench_coda_k.py --regime kernels --batch 6
  python lab/divergence/bench_coda_k.py --regime kernels --batch 24 --variants full

Exit codes: 0 ok, 42 CUDA OOM (bisect the batch from the shell).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import statistics
import sys

import torch

# donated_buffer must be off before any compile (mirror morph/training/train.py).
import torch._functorch.config as _functorch_config
if hasattr(_functorch_config, "donated_buffer"):
    _functorch_config.donated_buffer = False

sys.path.insert(0, ".")
from lab.divergence._build import build_cfg, build_model    # noqa: E402
from morph.training.data import create_dataloader           # noqa: E402

REGION_SETS = {
    "full": (),
    "core_off": ("core",),
    "core_prelude_off": ("core", "prelude"),
    "core_prelude_coda_off": ("core", "prelude", "coda"),
}


@contextlib.contextmanager
def ablated(root, names):
    """Identity-ablate regions — verbatim the region_shapley.py mechanism."""
    undo = []
    for n in names:
        if n == "core":
            orig = root._apply_core_step
            root._apply_core_step = lambda h_in, *a, **kw: (h_in, None)
            undo.append(lambda o=orig: setattr(root, "_apply_core_step", o))
        else:
            blocks = getattr(root, n)
            origs = [b.forward for b in blocks]
            for b in blocks:
                b.forward = (lambda h, *a, **kw: h)
            undo.append(lambda bs=blocks, os=origs: [setattr(b, "forward", f)
                                                     for b, f in zip(bs, os)])
    try:
        yield
    finally:
        for f in undo:
            f()


def pin_depths(root, depth: int) -> None:
    """Deterministic loop depth for every variant (recorded in the output)."""
    def det_slot(layout, device):
        shape = layout.slot_index.shape
        d = torch.full(shape, depth, device=device, dtype=torch.long)
        return torch.where(layout.slot_valid, d, torch.ones_like(d))

    def det_seq(B, device):
        return torch.full((B,), depth, device=device, dtype=torch.long)

    root._sample_slot_depths = det_slot
    root._sample_depths = det_seq


def stats(times: list[float]) -> dict:
    return {
        "n": len(times),
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
        "all_ms": times,
    }


def time_fwd_bwd(model, batches, warmup: int, iters: int) -> list[float]:
    times = []
    for i in range(warmup + iters):
        x, y, layout = batches[i % len(batches)]
        model.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x, labels=y, slot_layout=layout)
        out["loss"].backward()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return times


def time_loss_only(root, batches, warmup: int, iters: int) -> list[float]:
    """Head + chunked CE fwd+bwd on a random carrier of the real coda-output shape.

    CE cost does not depend on carrier values, so a random carrier is exact for
    timing. Gradient flows to the carrier AND the tied head weight, as in training.
    """
    x0, y0, layout0 = batches[0]
    d = root.cfg.d_model
    times = []
    for i in range(warmup + iters):
        _, y, layout = batches[i % len(batches)]
        B, L = y.shape
        xh = torch.randn(B, L, d, device=y.device, dtype=torch.bfloat16,
                         requires_grad=True)
        root.zero_grad(set_to_none=True)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = root._tul_group_losses(xh, y, layout, want_groups=False)
        out["loss"].backward()
        end.record()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(end))
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=("eager", "kernels"), required=True,
                    help="eager: use_kernels=false, no compile. "
                         "kernels: use_kernels=true + torch.compile(mode='default') "
                         "on the MLPs, mirroring morph/training/train.py.")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--config", default="tul_a1")
    ap.add_argument("--depth", type=int, default=8,
                    help="pinned loop depth for ALL variants (default max_depth=8)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--n-batches", type=int, default=8,
                    help="distinct real batches cycled through the iterations")
    ap.add_argument("--variants", default="all",
                    help="'all' or comma list from: " + ",".join(REGION_SETS))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="merge results into this JSON file")
    ap.add_argument("--label", default=None, help="key in the output JSON")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    use_kernels = a.regime == "kernels"
    overrides = [f"training.batch_size={a.batch}",
                 f"model.use_kernels={'true' if use_kernels else 'false'}"]
    cfg = build_cfg(a.config, overrides)
    model, tul_rt = build_model(cfg, device="cuda")
    root = getattr(model, "_orig_mod", model)
    pin_depths(root, a.depth)
    model.train()

    compiled = False
    if use_kernels:
        # Mirror train.py: compile ONLY the MLP submodules, mode=default,
        # dynamic batch for the core group. reduce-overhead is banned (eval OOM).
        try:
            import torch._dynamo as _dynamo
            _dynamo.config.cache_size_limit = max(
                getattr(_dynamo.config, "cache_size_limit", 8), 64)
            for group in (model.prelude, model.core, model.coda):
                dyn = True if group is model.core else None
                for layer in group:
                    if hasattr(layer, "mlp"):
                        layer.mlp = torch.compile(layer.mlp, mode="default", dynamic=dyn)
            compiled = True
            print("  MLPs compiled (mode=default, core dynamic-batch)", flush=True)
        except Exception as e:                              # report, fall back, SAY SO
            print(f"  torch.compile FAILED ({type(e).__name__}: {e}) — "
                  f"falling back to kernels-on EAGER", flush=True)

    loader = iter(create_dataloader(
        cfg.data.tokenizer, cfg.data.dataset, int(cfg.data.seq_len), a.batch,
        split="validation", skip_samples=0,
        tul=tul_rt.val_data_cfg if tul_rt else None))
    batches = []
    for _ in range(a.n_batches):
        bx, by, bl = next(loader)
        batches.append((bx.cuda(), by.cuda(), bl.to("cuda")))
    L_total = batches[0][0].shape[1]
    print(f"  {len(batches)} fixed batches, shape [{a.batch}, {L_total}], "
          f"pinned depth {a.depth}", flush=True)

    names = (list(REGION_SETS) if a.variants == "all"
             else [v.strip() for v in a.variants.split(",")])

    result = {
        "regime": a.regime, "compiled": compiled, "config": a.config,
        "batch": a.batch, "seq_len": int(cfg.data.seq_len), "L_total": int(L_total),
        "pinned_depth": a.depth, "warmup": a.warmup, "iters": a.iters,
        "n_batches": a.n_batches, "seed": a.seed,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "variants": {},
    }

    try:
        for name in names:
            off = REGION_SETS[name]
            with ablated(root, off):
                # per-variant warmup happens inside time_fwd_bwd
                torch.cuda.reset_peak_memory_stats()
                times = time_fwd_bwd(model, batches, a.warmup, a.iters)
            s = stats(times)
            s["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
            s["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
            result["variants"][name] = s
            print(f"  {name:<24} mean {s['mean_ms']:8.2f} ms  "
                  f"median {s['median_ms']:8.2f}  std {s['std_ms']:6.2f}  "
                  f"peak {s['peak_alloc_gb']:.2f} GB", flush=True)

        if a.variants == "all":
            torch.cuda.reset_peak_memory_stats()
            times = time_loss_only(root, batches, a.warmup, a.iters)
            s = stats(times)
            s["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2**30
            result["variants"]["loss_only"] = s
            print(f"  {'loss_only':<24} mean {s['mean_ms']:8.2f} ms  "
                  f"median {s['median_ms']:8.2f}  std {s['std_ms']:6.2f}", flush=True)
    except torch.OutOfMemoryError as e:
        print(f"MORPH_BENCH_OOM batch={a.batch}: {e}", flush=True)
        sys.exit(42)

    # Derived region deltas when the full set ran.
    v = result["variants"]
    if all(k in v for k in REGION_SETS) and "loss_only" in v:
        t_full = v["full"]["mean_ms"]
        core = t_full - v["core_off"]["mean_ms"]
        prelude = v["core_off"]["mean_ms"] - v["core_prelude_off"]["mean_ms"]
        coda = v["core_prelude_off"]["mean_ms"] - v["core_prelude_coda_off"]["mean_ms"]
        loss = v["loss_only"]["mean_ms"]
        front = v["core_prelude_coda_off"]["mean_ms"] - loss
        f = (coda + loss) / t_full
        f_upper = v["core_prelude_off"]["mean_ms"] / t_full
        result["regions_ms"] = {"prelude": prelude, "core": core, "coda": coda,
                                "loss": loss, "front_embed_glue": front,
                                "full": t_full}
        result["f_coda_loss"] = f
        result["f_upper_incl_front"] = f_upper
        print(f"\n  regions (fwd+bwd, ms): prelude {prelude:.2f}  core {core:.2f}  "
              f"coda {coda:.2f}  loss {loss:.2f}  front/embed/glue {front:.2f}  "
              f"| full {t_full:.2f}")
        print(f"  f (coda+loss fraction) = {f:.4f}   "
              f"f_upper (incl front) = {f_upper:.4f}")

    if a.out:
        label = a.label or f"{a.regime}_b{a.batch}"
        data = {}
        if os.path.exists(a.out):
            with open(a.out) as fh:
                data = json.load(fh)
        data[label] = result
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"  wrote {a.out} [{label}]")


if __name__ == "__main__":
    main()
