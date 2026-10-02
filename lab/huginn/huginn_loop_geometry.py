"""Offline probe: does Huginn-0125's recurrent state rotate at a stable scale, or does it
grow along a shared direction? Frozen question and predictions:
lab/experiments/planned/2026-10-02-huginn-loop-geometry.md (read-only, do not edit).

Why: MORPH's latent-pulled slot-loop arms grow their cells ~4x along one shared direction
from pass 2 on (.agents/notes/proposed/architecture/2026-10-02-layernorm-common-mode-in-latent-targets.md).
Huginn-0125 is a loop that EARNS depth on web text (K3-K6 +0.566, 2026-09-04 filing). This
probe reads whether an earning loop's state grows that way too, or just rotates.

Reused from lab/huginn/huginn_depth_sweep.py (the pinned Sept-4 prereg): `load_huginn`
(model+tokenizer load), the row source (`morph.training.data.create_dataloader` over the
same 480-row OpenWebText arrow set, `skip_samples=0` so row 0 is the same row 0), and the
per-row-seeded-latent / fork_rng pattern (Huginn samples its initial latent even in eval).

Faithful capture: `core_block_forward` is a plain method called from `iterate_forward`'s
no-grad loop; it is intercepted on the model INSTANCE the same way
`lab/divergence/ln_common_mode_probe.py`'s `Capture` class intercepts `_front_tail`
(`model.__dict__["core_block_forward"] = wrapper`, which shadows the class method for
plain attribute lookup) so the recorded tensor is read, never altered, and the forward
returns bit-identical outputs to an unpatched call. CE at iteration k is read by feeding
that k's raw (pre-`ln_f`) state through `model.predict_from_latents`, which applies the
SAME `ln_f` -> coda -> `ln_f` -> `lm_head` chain `forward()` applies once at `num_steps=k`
(verified by `--faithfulness_only`, which must show max abs logit diff ~0 before any
reading is trusted).

Two full forward passes over the row set (hooked, `num_steps=max_k` each): pass 1 only
accumulates the per-iteration mean state (needed for both `v` and the per-iteration centring
mean `mu_k`) and the mean norm; pass 2 (now that `v`/`mu_k` are fixed) reads everything else
-- cosine to v, the centred along-v/perp-v RMS split, the k-1->k step, CE at every k, the
three interventions at `--keep_depths`, and the participation ratio of a fixed per-token
subsample. A third, separate, small forward (first row chunk only) proves faithfulness.

Usage (GPU; GPU work only under the gpu lock, see lab/huginn/README.md for the power cap):
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  flock /home/wolfe/morph-scratch/gpu.lock python lab/huginn/huginn_loop_geometry.py \\
    --device cuda --rows 64 --seq 1024 --batch 8 --max_k 32 \\
    --out lab/experiments/results/2026-10-02-huginn-loop-geometry/probe.json

CPU smoke (tiny slice, build/dev only -- see the bottom of the prereg's Method):
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 python lab/huginn/huginn_loop_geometry.py \\
    --device cpu --rows 1 --seq 64 --batch 1 --max_k 4 --keep_depths 4 --pr_subsample 16 \\
    --out /tmp/huginn_loop_geometry_smoke.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lab/huginn"))
sys.path.insert(0, str(ROOT / "lab/divergence"))
from huginn_depth_sweep import load_huginn  # noqa: E402
from _stats import paired_bootstrap_ci  # noqa: E402

DEFAULT_REVISION = "bb6621b65e90b6a4b9b29ef88dc83866d450470c"
DEFAULT_SNAPSHOT = (
    "/home/wolfe/.cache/huggingface/hub/models--tomg-group-umd--huginn-0125/"
    f"snapshots/{DEFAULT_REVISION}"
)
DEFAULT_DATASET = "/home/wolfe/.cache/huggingface/datasets/openwebtext/**/openwebtext-train-*.arrow"
DEFAULT_SEED = 20260905  # same seed as the pinned depth sweep (lab/huginn/configs/depth_sweep.yaml)


class _Obj:
    """Plain attribute bag, just enough for load_huginn's cfg.model.* / cfg.device reads."""


def build_model_cfg(snapshot: str, revision: str, device: str) -> _Obj:
    m = _Obj()
    m.snapshot, m.revision = snapshot, revision
    cfg = _Obj()
    cfg.model, cfg.device = m, device
    return cfg


def amp(device: str):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))


def rng_devices(device: str) -> list:
    return [torch.cuda.current_device()] if device.startswith("cuda") else []


class CoreCapture:
    """Patches ONE model instance's bound `core_block_forward` so every call also reports
    its (unmodified) output to `on_state(k, x)`, k = 1-indexed iteration count within this
    context. The patch never changes what the loop computes with: it calls the original
    bound method first and returns its result unchanged; `on_state` only gets to read it."""

    def __init__(self, model, on_state):
        self.model, self.on_state, self.k = model, on_state, 0

    def __enter__(self):
        self._orig = self.model.core_block_forward

        def _wrapped(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
            out_x, out_block_idx = self._orig(
                x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step)
            self.k += 1
            self.on_state(self.k, out_x)
            return out_x, out_block_idx

        self.model.__dict__["core_block_forward"] = _wrapped
        self.k = 0
        return self

    def __exit__(self, *exc):
        self.model.__dict__.pop("core_block_forward", None)
        return False


def load_rows(source: Path, dataset_glob: str, seq: int, rows: int, skip_samples: int):
    """The first `rows` rows of the row stream `huginn_depth_sweep.py` uses: row content is
    determined by cumulative position in the token buffer, not by the loader's batch size
    (`morph/training/data.py::create_dataloader` reshapes a contiguous chunk into
    `[batch, seq+1]`), so requesting `rows` in one shot gives the same rows 0..rows-1 the
    pinned 480-row sweep would give at the start of its own stream."""
    from morph.training.data import create_dataloader
    loader = create_dataloader(str(source), dataset_glob, seq, rows, split="train",
                               skip_samples=skip_samples)
    x, y = next(loader)[:2]
    if x.shape[0] != rows:
        raise RuntimeError(f"loader returned {x.shape[0]} rows, expected {rows}")
    return x, y


def chunks_of(total: int, batch: int) -> list[tuple[int, int]]:
    return [(r, min(r + batch, total)) for r in range(0, total, batch)]


@torch.inference_mode()
def faithfulness_check(model, x_chunk: torch.Tensor, seed: int, device: str, max_k: int) -> float:
    """logits from a plain `model(num_steps=max_k)` call vs. logits rebuilt from the hooked
    path's captured raw state at k=max_k through `predict_from_latents`. Returns the max abs
    diff in fp32 logits. Must be ~0: `forward()` returns `ln_f(raw)` from `iterate_forward`
    then runs coda -> `ln_f` -> `lm_head`; `predict_from_latents(raw)` applies `ln_f` itself
    first, then the SAME coda -> `ln_f` -> `lm_head` -- the identical formula, just entered
    from the raw (pre-ln_f) state instead of from inside `forward()`."""
    captured: dict = {}

    def on_state(k, state):
        if k == max_k:
            captured["state"] = state.detach().clone()

    with torch.random.fork_rng(devices=rng_devices(device)):
        torch.manual_seed(seed)
        with CoreCapture(model, on_state):
            with amp(device):
                ref = model(input_ids=x_chunk.to(device), num_steps=int(max_k), use_cache=False,
                           output_details={"return_logits": True, "return_latents": False,
                                           "return_head": False, "return_stats": False})
    if "state" not in captured:
        raise RuntimeError(f"hook never fired at k={max_k} during the faithfulness check")
    with amp(device):
        rebuilt = model.predict_from_latents(captured["state"])
    return float((ref.logits.float() - rebuilt.logits.float()).abs().max())


@torch.inference_mode()
def pass_moments(model, x_all: torch.Tensor, chunks, max_k: int, seed: int, device: str):
    """Pass 1: accumulate, per iteration k, the per-coordinate mean state `mu_k` (also gives
    the global `v`, since every k pools the same token count) and the mean per-token norm.
    Nothing is retained across iterations; this pass only needs running sums."""
    C = model.config.n_embd
    x_sum = {k: torch.zeros(C, dtype=torch.float64, device=device) for k in range(1, max_k + 1)}
    norm_sum = {k: torch.zeros((), dtype=torch.float64, device=device) for k in range(1, max_k + 1)}
    count = {k: 0 for k in range(1, max_k + 1)}
    for r0, r1 in chunks:
        xb = x_all[r0:r1].to(device)

        def on_state(k, state):
            flat = state.float().reshape(-1, C).double()
            x_sum[k] += flat.sum(0)
            norm_sum[k] += flat.norm(dim=-1).sum()
            count[k] += flat.shape[0]

        with torch.random.fork_rng(devices=rng_devices(device)):
            torch.manual_seed(seed + r0)
            with CoreCapture(model, on_state):
                with amp(device):
                    model(input_ids=xb, num_steps=int(max_k), use_cache=False,
                         output_details={"return_logits": False, "return_latents": False,
                                         "return_head": False, "return_stats": False})
    mu = {k: x_sum[k] / count[k] for k in x_sum}
    mean_norm = {k: float(norm_sum[k] / count[k]) for k in norm_sum}
    total_sum = sum(x_sum.values())
    total_count = sum(count.values())
    v = F.normalize(total_sum / total_count, dim=0)
    return mu, mean_norm, v


@torch.inference_mode()
def pass_main(model, x_all, y_all, chunks, max_k: int, mu: dict, v: torch.Tensor,
              device: str, keep_depths: list[int], pr_rows: np.ndarray, pr_toks: np.ndarray,
              rand_dir: torch.Tensor, seed: int):
    """Pass 2: everything that needs `v`/`mu_k` -- cosine to v, the centred along-v/perp-v
    RMS split, the k-1->k step (relative size + share along v), CE at every k (via
    `predict_from_latents` on the hooked raw state), the three interventions at
    `keep_depths`, and the fixed-subsample participation ratio."""
    C = model.config.n_embd
    total_rows, seq = x_all.shape
    ks = range(1, max_k + 1)
    cos_sum = {k: torch.zeros((), dtype=torch.float64, device=device) for k in ks}
    along_sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in ks}
    perp_sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in ks}
    cen_sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in ks}
    step_sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in range(2, max_k + 1)}
    base_sq = {k: torch.zeros((), dtype=torch.float64, device=device) for k in range(2, max_k + 1)}
    step_along_sq = {k: torch.zeros((), dtype=torch.float64, device=device)
                     for k in range(2, max_k + 1)}
    count = {k: 0 for k in ks}
    ce_row_sum = {k: np.zeros(total_rows, dtype=np.float64) for k in ks}
    ce_row_n = np.zeros(total_rows, dtype=np.float64)
    pr_buf: dict[int, list] = {k: [] for k in ks}
    keep = sorted(set(keep_depths))
    if 4 not in keep:
        raise ValueError("keep_depths must include 4 (the rescale intervention's reference)")
    interv_sum = {k: {"rescale": np.zeros(total_rows, dtype=np.float64),
                      "remove_v": np.zeros(total_rows, dtype=np.float64),
                      "remove_rand": np.zeros(total_rows, dtype=np.float64)} for k in keep}

    for r0, r1 in chunks:
        b = r1 - r0
        xb = x_all[r0:r1].to(device)
        yb = y_all[r0:r1].to(device)
        in_chunk = (pr_rows >= r0) & (pr_rows < r1)
        pr_local = (torch.as_tensor(pr_rows[in_chunk] - r0, device=device, dtype=torch.long),
                   torch.as_tensor(pr_toks[in_chunk], device=device, dtype=torch.long))
        prev = {"x": None}
        retained: dict[int, torch.Tensor] = {}
        norm4 = {"v": None}

        def ce_from(state) -> torch.Tensor:
            with amp(device):
                out = model.predict_from_latents(state)
            logits = out.logits.float()
            loss = F.cross_entropy(logits.flatten(0, 1), yb.flatten(), reduction="none")
            if not torch.isfinite(loss).all():
                raise FloatingPointError("non-finite CE in the loop-geometry probe")
            return loss.reshape(b, seq)

        def on_state(k, state):
            xf = state.float()
            flat = xf.reshape(-1, C).double()
            cos_sum[k] += F.normalize(flat, dim=-1).matmul(v.double()).sum()
            cen = flat - mu[k].double()
            a = cen @ v.double()
            along_sq[k] += a.square().sum()
            perp_sq[k] += (cen.square().sum(-1) - a.square()).sum()
            cen_sq[k] += cen.square().sum(-1).sum()
            count[k] += flat.shape[0]
            if prev["x"] is not None and k >= 2:
                step = xf.double().reshape(-1, C) - prev["x"]
                sa = step @ v.double()
                step_sq[k] += step.square().sum()
                base_sq[k] += prev["x"].square().sum()
                step_along_sq[k] += sa.square().sum()
            prev["x"] = xf.double().reshape(-1, C)
            ce = ce_from(state)
            ce_row_sum[k][r0:r1] += ce.double().sum(dim=1).cpu().numpy()
            if k == 1:
                ce_row_n[r0:r1] += seq
            if pr_local[0].numel():
                sub = xf[pr_local[0], pr_local[1]]
                pr_buf[k].append(F.normalize(sub.double(), dim=-1).cpu())
            if k == 4:
                norm4["v"] = xf.norm(dim=-1, keepdim=True).detach().clone()
            if k in keep:
                retained[k] = state.detach().clone()

        with torch.random.fork_rng(devices=rng_devices(device)):
            torch.manual_seed(seed + r0)
            with CoreCapture(model, on_state):
                with amp(device):
                    model(input_ids=xb, num_steps=int(max_k), use_cache=False,
                         output_details={"return_logits": False, "return_latents": False,
                                         "return_head": False, "return_stats": False})

        if norm4["v"] is None:
            raise RuntimeError("k=4 state was never captured (max_k < 4?)")
        for k in keep:
            state_k = retained[k].float()
            cur_norm = state_k.norm(dim=-1, keepdim=True)
            rescale = (state_k.double() * (norm4["v"].double() / cur_norm.double().clamp_min(1e-12))
                      ).to(retained[k].dtype)
            av = (state_k.double() @ v.double()).unsqueeze(-1)
            remove_v = (state_k.double() - av * v.double()).to(retained[k].dtype)
            ar = (state_k.double() @ rand_dir.double()).unsqueeze(-1)
            remove_rand = (state_k.double() - ar * rand_dir.double()).to(retained[k].dtype)
            for name, st in (("rescale", rescale), ("remove_v", remove_v),
                            ("remove_rand", remove_rand)):
                ce = ce_from(st)
                interv_sum[k][name][r0:r1] += ce.double().sum(dim=1).cpu().numpy()

    return {"cos_sum": cos_sum, "along_sq": along_sq, "perp_sq": perp_sq, "cen_sq": cen_sq,
            "step_sq": step_sq, "base_sq": base_sq, "step_along_sq": step_along_sq,
            "count": count, "ce_row_sum": ce_row_sum, "ce_row_n": ce_row_n, "pr_buf": pr_buf,
            "interv_sum": interv_sum}


def pr_from_buf(bufs: list[torch.Tensor]) -> float:
    """Participation ratio ``(tr cov)^2 / sum(cov^2)`` of the centred covariance -- same
    formula as ``pr_cov`` in lab/divergence/ln_common_mode_probe.py -- of states that are
    ALREADY per-token-normalised (the caller centres here; normalising happened at capture)."""
    z = torch.cat(bufs).double()
    zc = z - z.mean(0, keepdim=True)
    cov = zc.t() @ zc / max(zc.shape[0] - 1, 1)
    return float(torch.trace(cov) ** 2 / cov.square().sum())


def grade_predictions(readings: dict, interventions: dict) -> dict:
    """Mechanical check of the frozen G-1..G-6 predictions against this run's numbers. A
    single run with one seed; report, do not over-read (n=1)."""
    ks = sorted(int(k) for k in readings)
    g: dict = {}
    if 4 in readings and 32 in readings:
        ratio = readings[32]["mean_state_norm"] / readings[4]["mean_state_norm"]
        g["G1_norm_ratio_32_over_4"] = {"value": ratio, "holds": 0.0 < ratio <= 1.5}
    shares = [readings[k]["step_share_along_v"] for k in ks
             if k >= 8 and readings[k].get("step_share_along_v") is not None]
    if shares:
        g["G2_max_step_share_along_v_from_k8"] = {"value": max(shares), "holds": max(shares) < 0.30}
    coss = [readings[k]["cos_to_v"] for k in ks if k >= 4]
    if coss:
        g["G3_max_cos_to_v_from_k4"] = {"value": max(coss), "holds": max(coss) < 0.7}
    if 32 in interventions:
        d = interventions[32]["rescale"]["point"]
        g["G4_rescale_delta_ce_at_32"] = {"value": d, "holds": abs(d) < 0.01}
        d5 = interventions[32]["remove_v"]["point"]
        g["G5_remove_v_delta_ce_at_32"] = {"value": d5, "holds": d5 > 0.05}
    if 4 in readings and 16 in readings:
        pr4, pr16 = readings[4].get("pr"), readings[16].get("pr")
        if pr4 is not None and pr16 is not None:
            growth = (pr16 - pr4) / pr4
            g["G6_pr_growth_4_to_16"] = {"value": growth, "holds": growth > 0.10}
    return g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8, help="sub-batch of rows per forward call")
    ap.add_argument("--max_k", type=int, default=32)
    ap.add_argument("--keep_depths", default="4,8,16,32",
                    help="iterations where the interventions run; must include 4")
    ap.add_argument("--pr_subsample", type=int, default=2048,
                    help="fixed (row, token) subsample size for the participation ratio")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--skip_samples", type=int, default=0)
    ap.add_argument("--bootstrap_n", type=int, default=2000)
    ap.add_argument("--bootstrap_seed", type=int, default=0)
    ap.add_argument("--bootstrap_level", type=float, default=0.95)
    ap.add_argument("--cpu_threads", type=int, default=0, help="0 = leave torch's default")
    ap.add_argument("--faithfulness_only", action="store_true",
                    help="run only the faithfulness check (for the CPU smoke test) and exit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    if args.max_k < 4:
        raise ValueError("max_k must be >= 4 (the rescale intervention needs a k=4 state)")
    keep_depths = sorted({int(x) for x in args.keep_depths.split(",")})
    for k in keep_depths:
        if k > args.max_k:
            raise ValueError(f"keep_depths entry {k} exceeds max_k {args.max_k}")

    t0 = time.time()
    cfg = build_model_cfg(args.snapshot, args.revision, args.device)
    model, tok, model_cfg = load_huginn(cfg)
    print(f"[load] model ready ({time.time() - t0:.1f}s), n_embd={model.config.n_embd} "
          f"mean_recurrence={int(model_cfg.mean_recurrence)}", flush=True)

    source = Path(args.snapshot).expanduser()
    x_all, y_all = load_rows(source, args.dataset, args.seq, args.rows, args.skip_samples)
    print(f"[data] {args.rows} rows x {args.seq} tokens loaded ({time.time() - t0:.1f}s)",
          flush=True)
    chunks = chunks_of(args.rows, args.batch)

    diff = faithfulness_check(model, x_all[chunks[0][0]:chunks[0][1]],
                              args.seed + chunks[0][0], args.device, args.max_k)
    print(f"FAITHFULNESS max_abs_logit_diff={diff:.6g} (k={args.max_k}, "
          f"{chunks[0][1] - chunks[0][0]} rows) ({time.time() - t0:.1f}s)", flush=True)
    if diff > 0.05:
        raise RuntimeError(
            f"faithfulness check failed: max abs logit diff {diff} exceeds 0.05 -- the "
            "predict_from_latents replay does not match model(num_steps=k)")
    if args.faithfulness_only:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"faithfulness_max_abs_logit_diff": diff}, open(args.out, "w"), indent=1)
        print(f"wrote {args.out}")
        return

    mu, mean_norm, v = pass_moments(model, x_all, chunks, args.max_k, args.seed, args.device)
    print(f"[pass1] moments + v done ({time.time() - t0:.1f}s)", flush=True)

    rng = np.random.default_rng(args.seed + 777)
    total_tok = args.rows * args.seq
    n_sub = min(args.pr_subsample, total_tok)
    flat_idx = rng.choice(total_tok, size=n_sub, replace=False)
    pr_rows, pr_toks = flat_idx // args.seq, flat_idx % args.seq

    g_cpu = torch.Generator(device="cpu").manual_seed(args.seed + 999)
    rand_dir = F.normalize(torch.randn(model.config.n_embd, generator=g_cpu), dim=0).to(args.device)

    acc = pass_main(model, x_all, y_all, chunks, args.max_k, mu, v, args.device, keep_depths,
                    pr_rows, pr_toks, rand_dir, args.seed)
    print(f"[pass2] main readings done ({time.time() - t0:.1f}s)", flush=True)

    readings: dict = {}
    for k in range(1, args.max_k + 1):
        n = acc["count"][k]
        row = {
            "mean_state_norm": mean_norm[k],
            "cos_to_v": float(acc["cos_sum"][k] / n),
            "rms_along_v": float((acc["along_sq"][k] / n).sqrt()),
            "rms_perp_v": float((acc["perp_sq"][k] / n).sqrt()),
            "rms_centred": float((acc["cen_sq"][k] / n).sqrt()),
            "ce": float(acc["ce_row_sum"][k].sum() / acc["ce_row_n"].sum()),
        }
        if k >= 2:
            sn = float(acc["step_sq"][k])
            row["step_rel_rms"] = (float(acc["step_sq"][k] / n) ** 0.5
                                   / float(acc["base_sq"][k] / n) ** 0.5)
            row["step_share_along_v"] = (float(acc["step_along_sq"][k]) / sn
                                         if sn > 0 else None)
        if acc["pr_buf"][k]:
            row["pr"] = pr_from_buf(acc["pr_buf"][k])
        readings[k] = row

    interventions: dict = {}
    for k in keep_depths:
        base = acc["ce_row_sum"][k]
        cnt = acc["ce_row_n"]
        interventions[k] = {
            name: paired_bootstrap_ci(acc["interv_sum"][k][name], base, cnt,
                                     n_boot=args.bootstrap_n, seed=args.bootstrap_seed,
                                     level=args.bootstrap_level)
            for name in ("rescale", "remove_v", "remove_rand")
        }

    grading = grade_predictions(readings, interventions)

    print(f"\n{'k':>4} {'norm':>9} {'cos_v':>7} {'rms_v':>9} {'rms_perp':>9} {'step_rel':>9} "
         f"{'step_v':>7} {'pr':>8} {'ce':>8}")
    for k in sorted(readings):
        r = readings[k]
        print(f"{k:>4} {r['mean_state_norm']:>9.3f} {r['cos_to_v']:>7.3f} "
             f"{r['rms_along_v']:>9.3f} {r['rms_perp_v']:>9.3f} "
             f"{r.get('step_rel_rms', float('nan')):>9.4f} "
             f"{r.get('step_share_along_v', float('nan')):>7.3f} "
             f"{r.get('pr', float('nan')):>8.2f} {r['ce']:>8.4f}")
    print("\nInterventions (delta CE = intervention - unmodified, at that k; 95% paired CI):")
    for k in sorted(interventions):
        for name, ci in interventions[k].items():
            print(f"  k={k:>2} {name:>12}: {ci['point']:+.5f} "
                 f"[{ci['lo']:+.5f}, {ci['hi']:+.5f}] (n_units={ci['n_units']})")
    print("\nPrediction grading:")
    for name, g in grading.items():
        print(f"  {name}: value={g['value']:.5f} holds={g['holds']}")

    out = {
        "config": {k: v for k, v in vars(args).items()},
        "wall_s": round(time.time() - t0, 1),
        "faithfulness_max_abs_logit_diff": diff,
        "readings_by_k": readings,
        "interventions_by_k": interventions,
        "grading": grading,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {args.out} ({out['wall_s']}s)")


if __name__ == "__main__":
    main()
