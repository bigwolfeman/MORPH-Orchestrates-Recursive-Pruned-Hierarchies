"""Decode bench: LXTUL + pointer against plain MORPH, tok/s at batch 1 and N (tracker C1b).

Arms (each on real validation prompts read the trainer's way, greedy, ``--new`` tokens):

  lx_eager_today   generate_tul_batch WITHOUT the generation forward (the eager path as it
                   ran before C1e: every step also runs the span decoder and the fan's
                   diversity instruments), the gen-eval slot budget (max_slots 64 + new)
  lx_eager         generate_tul_batch with the generation forward (C1e), same budget
  lx_cached        generate_tul_cached, one row at a time (the decoder holds one row)
  lx_graphed       generate_tul_graphed, one row at a time, graphs captured once per run
  plain_eager      generate_plain_batch (recompute per step, the gen eval's plain arm)
  plain_cached     morph/inference/kv_cache.py prefill + decode_step, batched
  plain_graphed    morph/inference/engine.py StaticDecodeEngine (kv_cache prefill, then
                   CUDA-graph replay per token; bf16 LM-head GEMV). It MATERIALISES the
                   QAT weights in place, so it must run last in its group.

Timing: CUDA-synchronised wall over the whole call, prompt included (prefill is part of a
decode request); tok/s counts generated tokens. GPU-busy: the same arm for ``--prof_new``
tokens, run twice, once timed and once under torch.profiler: ``busy_frac`` = device kernel
time / wall of that short run (the profiler inflates only the host side), and
``gpu_busy_ms_per_tok`` = kernel time / generated tokens of that run (prompt included in
the time). ``busy_frac`` < 1 is the share of the wall the GPU sits idle (launch / host
bound).

``--check`` (LXTUL only, row 0): the generation forward against the old forward (tokens
identical, rows bit-equal) and the cached and graphed decoders against the eager one
(tokens, worst row difference as log-probs).

Run under the GPU lock, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_ROOT, os.path.join(_ROOT, "scripts"), os.path.join(_ROOT, "lab", "divergence")):
    if p not in sys.path:
        sys.path.insert(0, p)

import morph.inference.tul_generate as eager_mod  # noqa: E402
import morph.inference.tul_generate_cached as cached_mod  # noqa: E402
import morph.inference.tul_generate_graphed as graphed_mod  # noqa: E402
from gen_diversity_eval import read_val_rows  # noqa: E402
from morph.inference.kv_cache import MORPHKVCache, decode_step  # noqa: E402
from morph.inference.plain_generate import generate_plain_batch  # noqa: E402
from morph.inference.tul_generate import generate_tul_batch, generate_tul  # noqa: E402
from morph.inference.tul_generate_cached import generate_tul_cached  # noqa: E402
from morph.inference.tul_generate_graphed import TulGraphDecoder, generate_tul_graphed  # noqa: E402
from morph.training.tul_setup import build_tul_runtime  # noqa: E402
from tul_samples import load_cfg, load_ckpt  # noqa: E402


def _sync() -> None:
    torch.cuda.synchronize()


def _no_generate_mode():
    """Context: the eager generator without the generation forward (pre-C1e)."""
    class _Ctx:
        def __enter__(self):
            self.real = eager_mod._generation_kwargs
            eager_mod._generation_kwargs = lambda model: {}

        def __exit__(self, *exc):
            eager_mod._generation_kwargs = self.real
    return _Ctx()


@torch.no_grad()
def _plain_cached(model, prompts, new):
    ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
    cache = MORPHKVCache()
    cache.csa_pool_len = model.cfg.context_len
    logits = None
    for t in range(ids.shape[1]):
        logits = decode_step(model, ids[:, t], cache)
    out = [[] for _ in prompts]
    for step in range(new):
        nxt = logits.float().argmax(-1)
        for i, v in enumerate(nxt.tolist()):
            out[i].append(int(v))
        if step + 1 < new:
            logits = decode_step(model, nxt, cache)
    return out


@torch.no_grad()
def _plain_graphed(model, prompts, new):
    from morph.inference.engine import StaticDecodeEngine
    ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
    cache = MORPHKVCache()
    cache.csa_pool_len = model.cfg.context_len
    logits = None
    for t in range(ids.shape[1]):
        logits = decode_step(model, ids[:, t], cache)
    eng = StaticDecodeEngine(model, batch_size=ids.shape[0])
    eng.load_from_eager(cache)
    eng.capture()
    out = [[] for _ in prompts]
    for step in range(new):
        nxt = logits.float().argmax(-1)
        for i, v in enumerate(nxt.tolist()):
            out[i].append(int(v))
        if step + 1 < new:
            logits = eng.decode_step(nxt)
    return out


def run_arm(arm: str, model, prompts, new: int, ctx: dict) -> list[list[int]]:
    rule, spec_eval, spec_tight = ctx.get("rule"), ctx.get("spec_eval"), ctx.get("spec_tight")
    if arm == "lx_eager_today":
        with _no_generate_mode():
            out, _ = generate_tul_batch(model, prompts, rule, spec_eval, max_new_tokens=new,
                                        temperature=0.0, device="cuda", emit_source="token")
        return out
    if arm == "lx_eager":
        out, _ = generate_tul_batch(model, prompts, rule, spec_eval, max_new_tokens=new,
                                    temperature=0.0, device="cuda", emit_source="token")
        return out
    if arm == "lx_cached":
        return [generate_tul_cached(model, p, rule, spec_tight, max_new_tokens=new,
                                    temperature=0.0, device="cuda", emit_source="token")[0]
                for p in prompts]
    if arm == "lx_graphed":
        dec = ctx["graph_dec"]
        return [generate_tul_graphed(model, p, rule, spec_tight, max_new_tokens=new,
                                     temperature=0.0, device="cuda", emit_source="token",
                                     decoder=dec)[0] for p in prompts]
    if arm == "plain_eager":
        return generate_plain_batch(model, prompts, max_new_tokens=new, temperature=0.0,
                                    device="cuda")
    if arm == "plain_cached":
        return _plain_cached(model, prompts, new)
    if arm == "plain_graphed":
        return _plain_graphed(model, prompts, new)
    raise ValueError(arm)


def timed(arm, model, prompts, new, ctx):
    _sync()
    t0 = time.perf_counter()
    out = run_arm(arm, model, prompts, new, ctx)
    _sync()
    return out, time.perf_counter() - t0


def gpu_busy(arm, model, prompts, new, ctx) -> tuple[float, float]:
    """``(kernel seconds, wall seconds)`` of the same ``new``-token run: the wall unprofiled,
    the device kernel time under torch.profiler."""
    from torch.profiler import ProfilerActivity, profile
    _out, wall = timed(arm, model, prompts, new, ctx)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run_arm(arm, model, prompts, new, ctx)
        _sync()
    dev_us = sum(e.self_device_time_total for e in prof.key_averages()
                 if e.device_type == torch.autograd.DeviceType.CUDA)
    return dev_us * 1e-6, wall


def _rows_spy(module):
    rows: list[torch.Tensor] = []
    real = module.sample_next

    def _spy(logits, temperature, top_k, generator, top_p=0.0):
        rows.append(logits.detach().float().cpu().clone())
        return real(logits, temperature, top_k, generator, top_p)

    module.sample_next = _spy
    return rows, lambda: setattr(module, "sample_next", real)


def _worst(er, cr) -> float:
    w = 0.0
    for x, y in zip(er, cr):
        x, y = torch.log_softmax(x.double(), -1), torch.log_softmax(y.double(), -1)
        f = torch.isfinite(x) & torch.isfinite(y)
        w = max(w, float((x[f] - y[f]).abs().max()))
    return w


def _first_diff(a, b) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


@torch.no_grad()
def check(model, prompt, new, ctx) -> dict:
    rule, spec_eval, spec_tight = ctx["rule"], ctx["spec_eval"], ctx["spec_tight"]
    res = {}
    rows_old, undo = _rows_spy(eager_mod)
    with _no_generate_mode():
        a_old, _ = generate_tul(model, prompt, rule, spec_eval, max_new_tokens=new,
                                temperature=0.0, device="cuda", emit_source="token")
    undo()
    rows_gen, undo = _rows_spy(eager_mod)
    a_gen, _ = generate_tul(model, prompt, rule, spec_eval, max_new_tokens=new,
                            temperature=0.0, device="cuda", emit_source="token")
    undo()
    rows_rep, undo = _rows_spy(eager_mod)
    a_rep, _ = generate_tul(model, prompt, rule, spec_eval, max_new_tokens=new,
                            temperature=0.0, device="cuda", emit_source="token")
    undo()
    # The GPU forward is not bit-reproducible run to run (atomic scatter-adds: bag means,
    # the pointer's vocab scatter), so the mode is judged against the repeat's spread.
    res["generate_mode"] = {"tokens_equal": a_old == a_gen,
                            "rows_bit_equal": all(torch.equal(x, y)
                                                  for x, y in zip(rows_old, rows_gen)),
                            "worst_row_diff": _worst(rows_old, rows_gen)}
    res["eager_repeat"] = {"tokens_equal": a_rep == a_gen,
                           "rows_bit_equal": all(torch.equal(x, y)
                                                 for x, y in zip(rows_rep, rows_gen)),
                           "worst_row_diff": _worst(rows_rep, rows_gen)}
    for name, mod, fn in (("cached", cached_mod, lambda: generate_tul_cached(
            model, prompt, rule, spec_tight, max_new_tokens=new, temperature=0.0,
            device="cuda", emit_source="token")[0]),
                          ("graphed", graphed_mod, lambda: generate_tul_graphed(
            model, prompt, rule, spec_tight, max_new_tokens=new, temperature=0.0,
            device="cuda", emit_source="token", decoder=ctx["graph_dec"])[0])):
        rows, undo = _rows_spy(mod)
        b = fn()
        undo()
        n = _first_diff(a_gen, b)
        upto = len(b) if n < 0 else n + 1          # rows up to and incl. the first flip
        res[name] = {"tokens_equal": a_gen == b, "first_diff": n,
                     "worst_row_diff_until_first_diff": _worst(rows_gen[:upto], rows[:upto])}
    return res


@torch.no_grad()
def step_profile(model, ctx, reps: int = 50) -> dict:
    """Per-event cost of the graphed LXTUL decoder: replay time of the token graph and the
    slot graph (CUDA events, ``reps`` replays each), and the kernel count and device time of
    ONE eager execution of each step function (the kernels the graph holds)."""
    from torch.profiler import ProfilerActivity, profile
    dec = ctx["graph_dec"]
    dec.reset()
    # A live span of 8 tokens so the slot step pools a real span; the inputs stay fixed.
    for i in range(8):
        dec.feed_token(1000 + i, i)
    _sync()
    res = {}
    for name, g, fn in (("token", dec.g_tok, dec._token_step), ("slot", dec.g_slot,
                                                                    dec._slot_step)):
        t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay()
        t0.record()
        for _ in range(reps):
            g.replay()
        t1.record()
        _sync()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            _sync()
        ev = [e for e in prof.key_averages() if e.device_type == torch.autograd.DeviceType.CUDA]
        top = sorted(ev, key=lambda e: -e.self_device_time_total)[:8]
        res[name] = {"replay_ms": t0.elapsed_time(t1) / reps,
                     "kernels": int(sum(e.count for e in ev)),
                     "eager_kernel_ms": sum(e.self_device_time_total for e in ev) / 1e3,
                     "top": [(e.key[:60], e.count, round(e.self_device_time_total / 1e3, 3))
                             for e in top]}
    dec.reset()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lx_cfg", default="lxtul_pointer")
    ap.add_argument("--lx_ckpt", default="")
    ap.add_argument("--plain_cfg", default="notul_panel_norm_match")
    ap.add_argument("--plain_ckpt", default="")
    ap.add_argument("--arms", default="lx_eager_today,lx_eager,lx_cached,lx_graphed,"
                                      "plain_eager,plain_cached")
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--new", type=int, default=256)
    ap.add_argument("--prof_new", type=int, default=32)
    ap.add_argument("--prompt_len", type=int, default=128)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--step_profile", action="store_true")
    ap.add_argument("--no_tf32", action="store_true",
                    help="cuDNN TF32 off (it is ON by default and runs the CCA conv1d in fp32 "
                         "models at TF32 precision)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms (needs CUBLAS_WORKSPACE_CONFIG)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if a.no_tf32:
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    arms = a.arms.split(",")
    batches = [int(b) for b in a.batches.split(",")]
    nmax = max(batches)
    out_path = pathlib.Path(a.out)
    payload = json.loads(out_path.read_text()) if out_path.exists() else {}
    payload.setdefault("rows", [])

    def flush():
        out_path.write_text(json.dumps(payload, indent=1))

    lx_arms = [x for x in arms if x.startswith("lx_")]
    plain_arms = [x for x in arms if x.startswith("plain_")]
    prompts = None
    for group, cfg_name, ckpt in (("lx", a.lx_cfg, a.lx_ckpt),
                                  ("plain", a.plain_cfg, a.plain_ckpt)):
        g_arms = lx_arms if group == "lx" else plain_arms
        if not g_arms and not (group == "lx" and (a.check or a.step_profile)):
            continue
        cfg = load_cfg(cfg_name)
        rt = build_tul_runtime(cfg)
        model, step = load_ckpt(cfg, ckpt, torch.device("cuda"),
                                None if rt is None else rt.model_cfg)
        model.eval()
        if prompts is None:
            rows = read_val_rows(cfg.data.tokenizer, cfg.data.dataset, nmax, a.prompt_len,
                                 a.new)
            prompts = [r[:a.prompt_len] for r in rows]
        ctx: dict = {}
        if rt is not None:
            rule = rt.data_cfg.rule
            spec = rt.data_cfg.spec_for(a.prompt_len)
            # gen-eval budget (lab/divergence/gen_diversity_eval.py) and a tight one: a slot
            # needs >= min_span tokens, so (prompt + new) // min_span + 1 never overflows.
            ctx = {"rule": rule,
                   "spec_eval": dataclasses.replace(spec, max_slots=spec.max_slots + a.new),
                   "spec_tight": dataclasses.replace(
                       spec, max_slots=(a.prompt_len + a.new) // rule.min_span + 1)}
            if "lx_graphed" in g_arms or a.check or a.step_profile:
                _sync()
                t0 = time.perf_counter()
                ctx["graph_dec"] = TulGraphDecoder(model, ctx["spec_tight"], rule, "token")
                _sync()
                payload["graph_capture_s"] = time.perf_counter() - t0
                print(f"graph capture {payload['graph_capture_s']:.1f}s", flush=True)
            if a.step_profile:
                payload["step_profile"] = step_profile(model, ctx)
                print("STEP_PROFILE", json.dumps(payload["step_profile"]), flush=True)
                flush()
            if a.check:
                payload["check"] = check(model, prompts[0], a.new, ctx)
                print("CHECK", json.dumps(payload["check"]), flush=True)
                flush()
        for arm in g_arms:
            for B in batches:
                ps = prompts[:B]
                try:
                    run_arm(arm, model, ps[:1], 4, ctx)              # warm-up (compile, alloc)
                    out, wall = timed(arm, model, ps, a.new, ctx)
                    kern, wall_s = gpu_busy(arm, model, ps, a.prof_new, ctx)
                    peak = torch.cuda.max_memory_allocated() / 2**30
                    tok = B * a.new
                    row = {"arm": arm, "batch": B, "new": a.new, "wall_s": wall,
                           "tok_s": tok / wall, "wall_ms_per_tok": 1e3 * wall / tok,
                           "prof_new": a.prof_new,
                           "short_wall_ms_per_tok": 1e3 * wall_s / (B * a.prof_new),
                           "gpu_busy_ms_per_tok": 1e3 * kern / (B * a.prof_new),
                           "busy_frac": kern / wall_s, "peak_gib": peak,
                           "ckpt_step": step, "row0": out[0]}
                except torch.cuda.OutOfMemoryError as e:
                    row = {"arm": arm, "batch": B, "error": f"OOM: {str(e)[:200]}"}
                    torch.cuda.empty_cache()
                payload["rows"] = [r for r in payload["rows"]
                                   if not (r["arm"] == arm and r["batch"] == B)] + [row]
                print(json.dumps(row), flush=True)
                flush()
                torch.cuda.reset_peak_memory_stats()
        del model, ctx
        torch.cuda.empty_cache()
    flush()


if __name__ == "__main__":
    main()
