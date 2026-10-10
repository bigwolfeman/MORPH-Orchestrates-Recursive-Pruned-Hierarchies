"""Decode bench: LXTUL + pointer against plain MORPH, tok/s at batch 1 and N (tracker C1b).

Arms (each on real validation prompts read the trainer's way, greedy, ``--new`` tokens):

  lx_eager_today   generate_tul_batch WITHOUT the generation forward (the eager path as it
                   ran before C1e: every step also runs the span decoder and the fan's
                   diversity instruments), the gen-eval slot budget (max_slots 64 + new)
  lx_eager         generate_tul_batch with the generation forward (C1e), same budget
  lx_cached        generate_tul_cached, one row at a time (the decoder holds one row)
  lx_graphed       generate_tul_graphed, one row at a time, graphs captured once per run
  lx_fused         morph/inference/lx_fused_decode.py: every block on the plain engine's
                   kernels, the strict attention through lx_attn, real batching (one token
                   graph over all B rows, slot graphs per bucket of cutting rows); one
                   decoder per batch size built and captured once outside the timing;
                   fp32 weight copies (exact). ``--check_fused`` gates its tokens and rows
                   against lx_graphed and the eager generate_tul (prompts 0 and 1) and
                   every batch row against its B=1 run. ``--fused_attn`` /
                   ``--fused_kv_row`` pick the K1 / K2 kernels or their PyTorch stand-ins.
  lx_fused_bf16    the same with bf16 weight copies (the plain engine's default)
  plain_eager      generate_plain_batch (recompute per step, the gen eval's plain arm)
  plain_cached     morph/inference/kv_cache.py prefill + decode_step, batched
  plain_graphed    morph/inference/engine.py StaticDecodeEngine, one engine per batch size
                   built and captured once outside the timing (as lx_graphed's decoder);
                   each call reset()s it and feeds prompt and new tokens through CUDA-graph
                   replay (bf16 LM-head GEMV). It MATERIALISES the QAT weights in place, so
                   it must run last in its group. ``--check_plain`` gates it against
                   plain_cached (tokens, worst row difference as log-probs).

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
from morph.inference.lx_fused_decode import LxFusedDecoder, generate_lx_fused_batch  # noqa: E402
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


def _eager_prefill(model, ids):
    """kv_cache prefill of ``ids`` [B, P]; returns (last logits, cache)."""
    cache = MORPHKVCache()
    cache.csa_pool_len = model.cfg.context_len
    logits = None
    for t in range(ids.shape[1]):
        logits = decode_step(model, ids[:, t], cache)
    return logits, cache


def _greedy_tail(logits, step_fn, n_rows, new, rows):
    """Greedy loop shared by the kv_cache and engine arms: read the argmax of ``logits``,
    feed it to ``step_fn`` for the next logits. ``rows`` (a list or None) collects each
    step's logits on the host (the check only)."""
    out = [[] for _ in range(n_rows)]
    for step in range(new):
        if rows is not None:
            rows.append(logits.detach().float().cpu().clone())
        nxt = logits.float().argmax(-1)
        for i, v in enumerate(nxt.tolist()):
            out[i].append(int(v))
        if step + 1 < new:
            logits = step_fn(nxt)
    return out


@torch.no_grad()
def _plain_cached(model, prompts, new, rows=None):
    ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
    logits, cache = _eager_prefill(model, ids)
    return _greedy_tail(logits, lambda nxt: decode_step(model, nxt, cache), len(prompts),
                        new, rows)


@torch.no_grad()
def _plain_graphed(model, prompts, new, ctx, rows=None, convert=False, eng=None):
    """StaticDecodeEngine greedy decode on the engine built and captured ONCE for this
    batch size (``ctx["plain_engines"][B]``, see ``_plain_setup``), as the LXTUL graphed
    arm reuses its ``TulGraphDecoder``. Default: ``reset()`` and feed the WHOLE prompt
    through the graphs (no eager step at all). ``convert``: the engine's original entry,
    a kv_cache prefill of the prompt and ``load_from_eager`` (the check runs both)."""
    eng = ctx["plain_engines"][len(prompts)] if eng is None else eng
    ids = torch.tensor(prompts, dtype=torch.long, device="cuda")
    if convert:
        logits, cache = _eager_prefill(model, ids)
        eng.load_from_eager(cache)
    else:
        eng.reset()
        logits = None
        for t in range(ids.shape[1]):
            logits = eng.decode_step(ids[:, t])
    return _greedy_tail(logits, eng.decode_step, len(prompts), new, rows)


@torch.no_grad()
def _plain_setup(model, prompts, new, batches, check_plain, payload, ctx,
                 exact_weights=False) -> None:
    """Build and capture one StaticDecodeEngine per batch size (it MATERIALISES the QAT
    weights in place, bit-identical reads), timed into ``payload["plain_capture_s"]``.
    ``check_plain``: first record the kv_cache references (before materialising), then
    gate the engine against them (``payload["check_plain"]``). ``exact_weights``: the
    timed engines keep fp32 weight copies (the engine's parity mode)."""
    from morph.inference.engine import StaticDecodeEngine
    refs = _plain_refs(model, prompts, new, max(batches)) if check_plain else None
    _sync()
    t0 = time.perf_counter()
    ctx["plain_engines"] = {}
    for B in sorted(set(batches) | ({1, max(batches)} if check_plain else set())):
        eng = StaticDecodeEngine(model, batch_size=B, exact_weights=exact_weights)
        eng.capture()
        ctx["plain_engines"][B] = eng
    _sync()
    payload["plain_capture_s"] = time.perf_counter() - t0
    print(f"plain engine build+capture {payload['plain_capture_s']:.1f}s", flush=True)
    if refs is not None:
        payload["check_plain"] = _plain_check(model, prompts, new, refs, ctx)
        print("CHECK_PLAIN", json.dumps(payload["check_plain"]), flush=True)


def _plain_refs(model, prompts, new, nb) -> dict:
    """kv_cache greedy references: prompts 0 and 1 alone, prompt 0 again (the eager
    path's own run-to-run spread), and the first ``nb`` prompts as one batch."""
    refs = {}
    for name, ps in (("p0", prompts[:1]), ("p1", prompts[1:2]), ("p0_repeat", prompts[:1]),
                     (f"batch{nb}", prompts[:nb])):
        rows: list[torch.Tensor] = []
        refs[name] = (_plain_cached(model, ps, new, rows), rows)
    return refs


def _cmp_rows(ref_tok, ref_rows, tok, rows) -> dict:
    """Per batch row: tokens equal, first differing step, worst log-prob difference over
    the steps up to and including the first flip (after it the inputs differ)."""
    res = {"tokens_equal": ref_tok == tok, "rows": []}
    for b, (a, c) in enumerate(zip(ref_tok, tok)):
        n = _first_diff(a, c)
        upto = len(c) if n < 0 else n + 1
        row = {"first_diff": n, "worst_row_diff_until_first_diff": _worst(
            [r[b] for r in ref_rows[:upto]], [r[b] for r in rows[:upto]])}
        if n >= 0:
            # the reference's own top-1 minus top-2 log-prob at the flip: a flip inside the
            # engine's worst row difference is a near tie, one far outside it is a bug.
            top = torch.log_softmax(ref_rows[n][b].double(), -1).topk(2).values
            row["ref_top2_margin_at_flip"] = float(top[0] - top[1])
        res["rows"].append(row)
    res["worst_row_diff"] = max(r["worst_row_diff_until_first_diff"] for r in res["rows"])
    return res


def _plain_check(model, prompts, new, refs, ctx) -> dict:
    res = {}
    t0, r0 = refs["p0"]
    res["eager_repeat"] = _cmp_rows(t0, r0, *refs["p0_repeat"])
    for name, ps in (("p0", prompts[:1]), ("p1", prompts[1:2])):
        rt, rr = refs[name]
        for mode, conv in (("reset", False), ("convert", True)):
            rows: list[torch.Tensor] = []
            tok = _plain_graphed(model, ps, new, ctx, rows, convert=conv)
            res[f"{name}_{mode}"] = _cmp_rows(rt, rr, tok, rows)
    # parity mode (fp32 weight copies): what is left is the engine's own arithmetic.
    from morph.inference.engine import StaticDecodeEngine
    ex = StaticDecodeEngine(model, batch_size=1, exact_weights=True)
    ex.capture()
    for name, ps in (("p0", prompts[:1]), ("p1", prompts[1:2])):
        rows = []
        tok = _plain_graphed(model, ps, new, ctx, rows, eng=ex)
        res[f"{name}_reset_exact_weights"] = _cmp_rows(*refs[name], tok, rows)
    del ex
    bname = next(k for k in refs if k.startswith("batch"))
    nb = int(bname[5:])
    rt, rr = refs[bname]
    for mode, conv in (("reset", False), ("convert", True)):
        rows = []
        tok = _plain_graphed(model, prompts[:nb], new, ctx, rows, convert=conv)
        res[f"{bname}_{mode}"] = _cmp_rows(rt, rr, tok, rows)
    return res


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
    if arm in ("lx_fused", "lx_fused_bf16"):
        dec = ctx["fused_decs"][(arm, len(prompts))]
        return generate_lx_fused_batch(model, prompts, rule, spec_tight, max_new_tokens=new,
                                       emit_source="token", decoder=dec)[0]
    if arm == "plain_eager":
        return generate_plain_batch(model, prompts, max_new_tokens=new, temperature=0.0,
                                    device="cuda")
    if arm == "plain_cached":
        return _plain_cached(model, prompts, new)
    if arm == "plain_graphed":
        return _plain_graphed(model, prompts, new, ctx)
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


def _fused_setup(model, prompts, new, batches, arms, ctx, payload, attn, kv_row) -> None:
    """One LxFusedDecoder per (arm, batch size), built and captured outside the timing."""
    _sync()
    t0 = time.perf_counter()
    ctx["fused_decs"] = {}
    steps = max(len(p) for p in prompts) + new
    for arm in arms:
        for B in sorted(set(batches) | {1}):
            ctx["fused_decs"][(arm, B)] = LxFusedDecoder(
                model, ctx["spec_tight"], ctx["rule"], B, "token",
                exact_weights=(arm == "lx_fused"), max_tokens=steps, attn=attn,
                kv_row=kv_row)
    _sync()
    payload["fused_capture_s"] = time.perf_counter() - t0
    payload["fused_attn"], payload["fused_kv_row"] = attn, kv_row
    print(f"fused decoders build+capture {payload['fused_capture_s']:.1f}s", flush=True)


@torch.no_grad()
def check_fused(model, prompts, new, ctx, arm, nbs) -> dict:
    """lx_fused against lx_graphed and the eager generate_tul on prompts 0 and 1 (tokens,
    first differing step, worst row difference up to and including it), and, for every
    batch size in ``nbs`` above 1, every row of the batch against its own B=1 run
    (tokens, worst row difference)."""
    rule, spec_eval, spec_tight = ctx["rule"], ctx["spec_eval"], ctx["spec_tight"]
    res = {}
    single = {}
    for pi in range(min(2, len(prompts))):
        p = prompts[pi]
        out, _b, rows = generate_lx_fused_batch(model, [p], rule, spec_tight, new,
                                                emit_source="token", return_logits=True,
                                                decoder=ctx["fused_decs"][(arm, 1)])
        f_tok, f_rows = out[0], [r.float().cpu() for r in rows[0]]
        single[pi] = (f_tok, f_rows)
        gr, undo = _rows_spy(graphed_mod)
        g_tok = generate_tul_graphed(model, p, rule, spec_tight, max_new_tokens=new,
                                     temperature=0.0, device="cuda", emit_source="token",
                                     decoder=ctx["graph_dec"])[0]
        undo()
        er, undo = _rows_spy(eager_mod)
        e_tok, _ = generate_tul(model, p, rule, spec_eval, max_new_tokens=new,
                                temperature=0.0, device="cuda", emit_source="token")
        undo()
        for name, rt, rr in (("graphed", g_tok, gr), ("eager", e_tok, er)):
            n = _first_diff(rt, f_tok)
            upto = len(f_tok) if n < 0 else n + 1
            row = {"tokens_equal": rt == f_tok, "first_diff": n,
                   "worst_row_diff_until_first_diff": _worst(rr[:upto], f_rows[:upto])}
            if n >= 0:
                top = torch.log_softmax(rr[n].double(), -1).topk(2).values
                row["ref_top2_margin_at_flip"] = float(top[0] - top[1])
            res[f"p{pi}_vs_{name}"] = row
    for nb in sorted(x for x in set(nbs) if x > 1):
        ps = prompts[:nb]
        out, _b, rows = generate_lx_fused_batch(model, ps, rule, spec_tight, new,
                                                emit_source="token", return_logits=True,
                                                decoder=ctx["fused_decs"][(arm, nb)])
        per = []
        for b, p in enumerate(ps):
            if b in single:
                s_tok, s_rows = single[b]
            else:
                o1, _b1, r1 = generate_lx_fused_batch(
                    model, [p], rule, spec_tight, new, emit_source="token",
                    return_logits=True, decoder=ctx["fused_decs"][(arm, 1)])
                s_tok, s_rows = o1[0], [r.float().cpu() for r in r1[0]]
                single[b] = (s_tok, s_rows)
            n = _first_diff(s_tok, out[b])
            upto = len(out[b]) if n < 0 else n + 1
            per.append({"tokens_equal": s_tok == out[b], "first_diff": n,
                        "worst_row_diff_until_first_diff": _worst(
                            s_rows[:upto], [r.float().cpu() for r in rows[b][:upto]])})
        res[f"batch{nb}_vs_b1"] = per
    return res


@torch.no_grad()
def fused_step_profile(model, ctx, arm, reps: int = 50) -> dict:
    """Per-event cost of the fused decoder at each batch size it was built for: replay
    time of the token graph and of every slot bucket graph (CUDA events), and the kernel
    count and device time of ONE eager execution of each step function."""
    from torch.profiler import ProfilerActivity, profile
    res = {}
    for (a, B), dec in ctx["fused_decs"].items():
        if a != arm:
            continue
        dec.reset()
        dec._fill_warmup_inputs()
        events = [("token", dec.g_tok, dec._token_step)] + [
            (f"slot{nb}", dec.g_slot[nb], (lambda nb=nb: dec._slot_step(nb)))
            for nb in dec.buckets]
        per = {}
        for name, g, fn in events:
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
            ev = [e for e in prof.key_averages()
                  if e.device_type == torch.autograd.DeviceType.CUDA]
            top = sorted(ev, key=lambda e: -e.self_device_time_total)[:10]
            per[name] = {"replay_ms": t0.elapsed_time(t1) / reps,
                         "kernels": int(sum(e.count for e in ev)),
                         "eager_kernel_ms": sum(e.self_device_time_total for e in ev) / 1e3,
                         "top": [(e.key[:60], e.count, round(e.self_device_time_total / 1e3, 3))
                                 for e in top]}
        res[f"B{B}"] = per
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
    ap.add_argument("--check_plain", action="store_true",
                    help="gate plain_graphed against the kv_cache greedy path (prompts 0 "
                         "and 1 alone, the max batch; reset and convert entries); needs "
                         "the plain_graphed arm")
    ap.add_argument("--plain_exact_weights", action="store_true",
                    help="plain_graphed engines keep fp32 weight copies "
                         "(StaticDecodeEngine(exact_weights=True))")
    ap.add_argument("--step_profile", action="store_true")
    ap.add_argument("--check_fused", action="store_true",
                    help="gate lx_fused against lx_graphed / eager (prompts 0, 1) and the "
                         "max batch's rows against their B=1 runs; needs an lx_fused arm")
    ap.add_argument("--fused_attn", default="kernel", choices=("kernel", "reference"))
    ap.add_argument("--fused_kv_row", default="kernel", choices=("kernel", "reference"))
    ap.add_argument("--no_tf32", action="store_true",
                    help="cuDNN TF32 off (it is ON by default and runs the CCA conv1d in fp32 "
                         "models at TF32 precision)")
    ap.add_argument("--ref_fp32_conv", action="store_true",
                    help="the references' loop conv without the fused Triton q||k conv "
                         "(fused_cca_conv runs tl.dot at TF32); with --no_tf32 the "
                         "references are fp32 throughout, like lx_fused")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms (needs CUBLAS_WORKSPACE_CONFIG)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
    if a.ref_fp32_conv:
        import morph.model.attention as attn_mod
        attn_mod.set_fused_attn_proj(qkconv=False)
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
            if "lx_graphed" in g_arms or a.check or a.step_profile or a.check_fused:
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
            fused_arms = [x for x in g_arms if x.startswith("lx_fused")]
            if fused_arms:
                _fused_setup(model, prompts, a.new, batches, fused_arms, ctx, payload,
                             a.fused_attn, a.fused_kv_row)
                flush()
                if a.check_fused:
                    payload["check_fused"] = check_fused(model, prompts, a.new, ctx,
                                                         fused_arms[0], batches)
                    print("CHECK_FUSED", json.dumps(payload["check_fused"]), flush=True)
                    flush()
                if a.step_profile:
                    payload["fused_step_profile"] = fused_step_profile(model, ctx,
                                                                       fused_arms[0])
                    print("FUSED_STEP_PROFILE", json.dumps(payload["fused_step_profile"]),
                          flush=True)
                    flush()
            elif a.check_fused:
                raise SystemExit("--check_fused needs an lx_fused arm")
        if a.check_plain and group == "plain" and "plain_graphed" not in g_arms:
            raise SystemExit("--check_plain needs the plain_graphed arm")
        for arm in g_arms:
            if arm == "plain_graphed" and "plain_engines" not in ctx:
                _plain_setup(model, prompts, a.new, batches, a.check_plain, payload, ctx,
                             exact_weights=a.plain_exact_weights)
                payload["plain_exact_weights"] = a.plain_exact_weights
                flush()
            for B in batches:
                ps = prompts[:B]
                try:
                    run_arm(arm, model, ps[:1], 4, ctx)              # warm-up (compile, alloc)
                    out, wall = timed(arm, model, ps, a.new, ctx)
                    fdec = ctx.get("fused_decs", {}).get((arm, B))
                    events = None if fdec is None else {      # the timed run's, before
                        "token_steps": fdec.token_steps,       # gpu_busy reruns it
                        "slot_steps_per_bucket": dict(fdec.slot_events)}
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
                    if events is not None:
                        row["fused_events"] = events
                except torch.cuda.OutOfMemoryError as e:
                    row = {"arm": arm, "batch": B, "error": f"OOM: {str(e)[:200]}"}
                    torch.cuda.empty_cache()
                payload["rows"] = [r for r in payload["rows"]
                                   if not (r["arm"] == arm and r["batch"] == B)] + [row]
                print(json.dumps(row), flush=True)
                flush()
                torch.cuda.reset_peak_memory_stats()
        for fdec in ctx.get("fused_decs", {}).values():   # release graphs + QAT cache
            fdec.__exit__(None, None, None)
        del model, ctx
        torch.cuda.empty_cache()
    flush()


if __name__ == "__main__":
    main()
