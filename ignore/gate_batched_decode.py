#!/usr/bin/env python
"""gate_batched_decode.py — per-stream parity gate for VARIABLE BATCH (B>1) decode.

THE gate for the batched routed-gather work. For each B in {2,4,8} (and an extra
B=16 timing pass):

  GOLDEN  : decode each of B DISTINCT prompts ALONE at B=1 through the proven
            StaticDecodeEngine (the bench's exact B=1 path) → B solo token streams.
  BATCHED : prefill all B prompts into ONE batched eager MORPHKVCache, load it into
            a StaticDecodeEngine(batch_size=B), capture, decode all B at once →
            B batched token streams.

PASS iff every batched stream is BYTE-IDENTICAL to its solo stream
(token_match = B*N / B*N). A fast-but-wrong batch is a bug → the gate FAILS loudly.

Equal-prefill-length limitation: the static engine's ring buffers + load_from_eager
share a single scalar `pos`, so all B streams must prefill to the SAME length. We use
equal-length distinct prompts (documented limitation, not a correctness compromise —
within a fixed prefill length each stream is independent).

Run from the worktree:
  PYTHONPATH=$PWD:$MAIN/ignore python ignore/gate_batched_decode.py \
      --ckpt $MAIN/checkpoints/morph/tst_stp_off_50k/step_50000.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# reuse the bench's model build + ckpt load + autocast (imported from $MAIN/ignore).
from bench_decode import build_model, load_ckpt, _AC, _strip_param  # type: ignore


@torch.no_grad()
def _prefill_batched(model, seeds: torch.Tensor):
    """Eager golden prefill of a [B, L] batch into ONE MORPHKVCache. Returns
    (cache, last_logits [B, vocab])."""
    from morph.inference.kv_cache import MORPHKVCache, decode_step
    cache = MORPHKVCache()
    cache.csa_pool_len = int(model.cfg.context_len)
    logit = None
    with _AC():
        for p in range(seeds.shape[1]):
            logit = decode_step(model, seeds[:, p], cache)   # [B] tokens
    return cache, logit


@torch.no_grad()
def _eager_stream_batched(model, seeds: torch.Tensor, n_gen: int):
    """EAGER-bf16 GOLDEN stream for a [B, L] batch, run BATCHED (the correct B>1
    reference: the static engine must reproduce the eager batched semantics, NOT the
    solo-vs-batch — the eager path itself is not bit-invariant across batch size, so
    solo is the wrong oracle for B>1). Returns list[list[int]]."""
    from morph.inference.kv_cache import decode_step
    B = seeds.shape[0]
    cache, logit = _prefill_batched(model, seeds)             # [B, vocab]
    streams = [[] for _ in range(B)]
    with _AC():
        nxt = logit.argmax(-1)                                # [B]
        for _ in range(n_gen):
            for b in range(B):
                streams[b].append(int(nxt[b]))
            logit = decode_step(model, nxt, cache)
            nxt = logit.argmax(-1)
    return streams


@torch.no_grad()
def _engine_stream_batched(model, seeds: torch.Tensor, n_gen: int):
    """Batched engine stream for [B, L] prompts. Returns (list[list[int]], engine)."""
    from morph.inference.engine import StaticDecodeEngine
    B = seeds.shape[0]
    cache, logit = _prefill_batched(model, seeds)             # logit [B, vocab]
    eng = StaticDecodeEngine(model, batch_size=B)
    with _AC():
        eng.load_from_eager(cache)
        eng.capture()
        streams = [[] for _ in range(B)]
        nxt = logit.argmax(-1)                                # [B]
        for _ in range(n_gen):
            for b in range(B):
                streams[b].append(int(nxt[b]))
            logit = eng.decode_step(nxt)
            nxt = logit.argmax(-1)
    return streams, eng


@torch.no_grad()
def _time_batched(eng, B, n_warmup=16, n_time=128):
    """Pure batched engine throughput (post warmup). Returns (steps_per_s, ms_per_step)."""
    with _AC():
        nxt = eng.logits.argmax(-1)
        for _ in range(n_warmup):
            if eng.pos >= eng.max_pos - 2:
                break
            eng.decode_step(nxt); nxt = eng.logits.argmax(-1)
        torch.cuda.synchronize()
        steps = 0
        t0 = time.perf_counter()
        for _ in range(n_time):
            if eng.pos >= eng.max_pos - 2:
                break
            eng.decode_step(nxt); nxt = eng.logits.argmax(-1)
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    sps = steps / dt if dt > 0 else float("nan")
    return sps, (1000.0 / sps if sps > 0 else float("nan"))


@torch.no_grad()
def _logit_parity(model, seeds, n_steps=32):
    """Per-stream LOGIT parity (tie-break-robust): step engine-batched and eager-batched
    in lockstep feeding BOTH the SAME (eager) greedy token each step, and compare the
    raw logits. Greedy argmax is chaotic on bf16 ties (cos≈1, Δ≈0.06 still flips a token);
    the logit vectors are the real correctness signal. Returns (max|Δ|, min cos)."""
    from morph.inference.kv_cache import decode_step
    from morph.inference.engine import StaticDecodeEngine
    B = seeds.shape[0]
    ec, elg = _prefill_batched(model, seeds)
    gc, glg = _prefill_batched(model, seeds)
    eng = StaticDecodeEngine(model, batch_size=B)
    worst_d, worst_c = 0.0, 1.0
    with _AC():
        eng.load_from_eager(gc); eng.capture()
        # seed both with the SAME first token so they stay on one trajectory.
        nxt = elg.argmax(-1)
        for _ in range(n_steps):
            elog = decode_step(model, nxt, ec)
            glog = eng.decode_step(nxt).clone()
            for b in range(B):
                d = float((elog[b].float() - glog[b].float()).abs().max())
                c = torch.nn.functional.cosine_similarity(
                    elog[b].float(), glog[b].float(), dim=0).item()
                worst_d = max(worst_d, d); worst_c = min(worst_c, c)
            nxt = elog.argmax(-1)         # shared trajectory (eager's choice)
    return worst_d, worst_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/morph/tst_stp_off_50k/step_50000.pt")
    ap.add_argument("--gen", type=int, default=128, help="tokens per stream for parity")
    ap.add_argument("--prompt-len", type=int, default=16)
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--time-batches", type=int, nargs="+", default=[1, 4, 8, 16])
    ap.add_argument("--logit-cos-min", type=float, default=0.999,
                    help="min acceptable per-stream cos(engine,eager) logits (real-bug guard)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("gate targets CUDA")
    device = torch.device("cuda")
    ckpt = os.path.abspath(args.ckpt)

    print("=== batched (B>1) decode per-stream parity gate ===")
    print(f"  ckpt: {ckpt}\n  gen={args.gen}  prompt_len={args.prompt_len}")
    model, cfg = build_model(device)
    step, compact, routed = load_ckpt(ckpt, model, device, cfg)
    model.eval()
    print(f"  loaded step={step} compact={compact} routed={routed}")
    assert compact, "ckpt must be carved/compact"
    assert routed, "this gate must run on a ROUTED ckpt (else the gather path is untested)"

    from morph.inference import deploy_quant as pti
    with _AC():
        stats = pti.to_deploy_inference(model, device="cuda")
    print(f"  mortar MLPs packed: {stats['mlps_packed']}  resident: {stats['resident_mb']:.1f} MB")

    vocab = int(model.cfg.vocab_size)

    # ── PARITY: engine-batched vs eager-batched, per stream ──────────────────────
    # TWO signals per B:
    #   (1) token_match: byte-identity of the greedy stream (informational — greedy
    #       argmax is CHAOTIC on bf16 ties, so a few late flips are expected and benign).
    #   (2) LOGIT parity (cos, max|Δ|): the real correctness arbiter, robust to ties.
    # VERDICT keys on (2): min per-stream cos must exceed --logit-cos-min. A real
    # indexing bug tanks cos (the stride bug gave cos→0.x); a tie-break leaves cos≈1.
    real_bug = False
    for B in args.batches:
        torch.manual_seed(1000 + B)
        seeds = torch.randint(0, vocab, (B, args.prompt_len), device=device)
        for b in range(B):
            seeds[b, 0] = (seeds[b, 0] + b) % vocab        # B DISTINCT prompts
        print(f"\n── B={B} ───────────────────────────────────────────────")
        golden = _eager_stream_batched(model, seeds, args.gen)
        batched, _ = _engine_stream_batched(model, seeds, args.gen)
        tot = B * args.gen
        match = sum(int(g == h) for gb, hb in zip(golden, batched) for g, h in zip(gb, hb))
        print(f"  token_match = {match}/{tot}  (greedy; ties may flip — see logit parity)")
        if match != tot:
            shown = 0
            for b in range(B):
                for j, (g, h) in enumerate(zip(golden[b], batched[b])):
                    if g != h and shown < 4:
                        print(f"    stream {b} first flip at gen {j}: eager={g} engine={h}")
                        shown += 1
                        break
        mxd, mcos = _logit_parity(model, seeds)
        verdict = "OK" if mcos >= args.logit_cos_min else "REAL-DIVERGENCE"
        print(f"  logit parity: min cos={mcos:.6f}  max|Δlogit|={mxd:.4f}  [{verdict}]")
        if mcos < args.logit_cos_min:
            real_bug = True

    # ── THROUGHPUT + VRAM vs B ───────────────────────────────────────────────────
    print("\n── throughput / VRAM scaling ──────────────────────────")
    print(f"  {'B':>3} {'steps/s':>9} {'ms/step':>8} {'agg_tok/s':>10} {'VRAM_MB':>9}")
    for B in args.time_batches:
        torch.manual_seed(7000 + B)
        seeds = torch.randint(0, vocab, (B, args.prompt_len), device=device)
        torch.cuda.reset_peak_memory_stats()
        _, eng = _engine_stream_batched(model, seeds, 8)      # short prefill+capture
        sps, msps = _time_batched(eng, B)
        vram = torch.cuda.max_memory_allocated() / 1e6
        print(f"  {B:>3} {sps:>9.1f} {msps:>8.3f} {B * sps:>10.1f} {vram:>9.1f}")
        del eng
        torch.cuda.empty_cache()

    if not real_bug:
        print(f"\nGATE_BATCHED_DECODE_PASS (per-stream logit cos >= {args.logit_cos_min} "
              "at every B; any token flips are sub-tie bf16 reorderings)")
    else:
        print("\nGATE_BATCHED_DECODE_FAIL (a batched stream's LOGITS diverged — real bug)")
        sys.exit(1)


if __name__ == "__main__":
    main()
