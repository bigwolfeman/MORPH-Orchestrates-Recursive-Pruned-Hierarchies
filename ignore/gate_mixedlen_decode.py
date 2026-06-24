#!/usr/bin/env python
"""gate_mixedlen_decode.py — per-stream parity gate for MIXED-LENGTH (B>1) decode.

THE gate for the mixed-length work: a batched StaticDecodeEngine whose B streams were
prefilled to DIFFERENT lengths must reproduce each stream's own logits.

Why the oracle is SOLO eager (not eager-BATCHED):
  The eager decoder (morph/inference/kv_cache.py — the proven golden) drives the WHOLE
  batch from a single scalar `cache.pos`. It therefore CANNOT represent a mixed-length
  batch in one cache: every element would share one position. The only place each stream
  sees its TRUE absolute position in the eager path is a SOLO (B=1) cache prefilled to
  that stream's own length. So the mixed-length ground truth is, per stream b:

     solo eager cache @ P_b  ──decode_step──▶  stream b's logits at p = P_b, P_b+1, …

  The engine, run BATCHED in mixed mode (per-stream pos_dev, collapsed single graph),
  must match each solo stream's logits. This is strictly MORE faithful than batched-eager
  (which is undefined for mixed lengths) — each stream's reference is its own independent
  golden run.

Tie-robustness (same lesson as the equal-length gate):
  bf16 greedy argmax is chaotic on near-ties, so token identity is informational only.
  The arbiter is the LOGIT vector: per-stream cos(engine, solo-eager) must exceed
  --logit-cos-min over >=N decode steps. We pin BOTH engine and solo-eager onto the SAME
  per-stream token trajectory (the solo-eager's greedy choices) so a divergence is a real
  indexing/position bug, not a sub-tie reordering.

Run from the worktree:
  PYTHONPATH=$PWD:$MAIN/ignore python ignore/gate_mixedlen_decode.py \
      --ckpt $MAIN/checkpoints/morph/tst_stp_off_50k/step_50000.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from bench_decode import build_model, load_ckpt, _AC  # type: ignore


@torch.no_grad()
def _solo_prefill(model, seed_row: torch.Tensor):
    """Prefill ONE stream (B=1) to its own length through the proven eager golden.
    seed_row [L] int. Returns (cache, last_logits [1, vocab])."""
    from morph.inference.kv_cache import MORPHKVCache, decode_step
    cache = MORPHKVCache()
    cache.csa_pool_len = int(model.cfg.context_len)
    logit = None
    with _AC():
        for p in range(seed_row.shape[0]):
            logit = decode_step(model, seed_row[p:p + 1], cache)   # [1, vocab]
    return cache, logit


@torch.no_grad()
def _solo_eager_trajectory(model, seed_row: torch.Tensor, n_gen: int):
    """Solo eager GOLDEN for one stream: returns (tokens [n_gen] greedy, logits list
    [n_gen][vocab]). The decode is self-feeding greedy (the stream's own trajectory)."""
    from morph.inference.kv_cache import decode_step
    cache, logit = _solo_prefill(model, seed_row)
    toks, logs = [], []
    with _AC():
        nxt = logit.argmax(-1)                       # [1]
        for _ in range(n_gen):
            toks.append(int(nxt[0]))
            logs.append(logit[0].float().clone())
            logit = decode_step(model, nxt, cache)
            nxt = logit.argmax(-1)
    return toks, logs


@torch.no_grad()
def _engine_mixed_trajectory(model, seeds: list[torch.Tensor], per_stream_tokens,
                             n_gen: int):
    """Mixed-length engine: prefill EACH stream solo (eager) to its own length, load all
    into ONE StaticDecodeEngine(batch_size=B) via load_from_eager_mixed, capture the
    single collapsed graph, and decode all B at once. Each step we FEED each stream its
    OWN fixed token (per_stream_tokens[b][step]) so the comparison is on a shared
    trajectory. Returns logits [n_gen] each [B, vocab]."""
    from morph.inference.engine import StaticDecodeEngine
    B = len(seeds)
    caches = []
    first_logits = []
    for b in range(B):
        c, lg = _solo_prefill(model, seeds[b])
        caches.append(c)
        first_logits.append(lg)                      # [1, vocab]
    eng = StaticDecodeEngine(model, batch_size=B)
    out_logs = []
    with _AC():
        eng.load_from_eager_mixed(caches)
        eng.capture()
        for step in range(n_gen):
            # feed each stream its own pinned token for THIS step
            tok = torch.tensor([per_stream_tokens[b][step] for b in range(B)],
                               device=eng.logits.device, dtype=torch.long)
            log = eng.decode_step(tok).clone()       # [B, vocab]
            out_logs.append(log)
    return out_logs, eng


@torch.no_grad()
def _time_mixed(eng, n_warmup=16, n_time=128):
    with _AC():
        nxt = eng.logits.argmax(-1)
        for _ in range(n_warmup):
            if max(eng.pos_host) >= eng.max_pos - 2:
                break
            eng.decode_step(nxt); nxt = eng.logits.argmax(-1)
        torch.cuda.synchronize()
        steps = 0
        t0 = time.perf_counter()
        for _ in range(n_time):
            if max(eng.pos_host) >= eng.max_pos - 2:
                break
            eng.decode_step(nxt); nxt = eng.logits.argmax(-1)
            steps += 1
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    sps = steps / dt if dt > 0 else float("nan")
    return sps, (1000.0 / sps if sps > 0 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/morph/tst_stp_off_50k/step_50000.pt")
    ap.add_argument("--gen", type=int, default=48, help="decode steps per stream for parity")
    ap.add_argument("--lengths", type=int, nargs="+", default=[9, 13, 30, 64],
                    help="DISTINCT per-stream prefill lengths (>= 2*csa_m=8). >=3 streams.")
    ap.add_argument("--logit-cos-min", type=float, default=0.999)
    ap.add_argument("--time-lengths", type=int, nargs="+",
                    default=[8, 16, 40, 96, 5, 12, 30, 64],
                    help="prefill lengths for the throughput pass (len = batch size)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("gate targets CUDA")
    device = torch.device("cuda")
    ckpt = os.path.abspath(args.ckpt)
    lengths = args.lengths
    assert len(lengths) >= 3, "mixed-length gate needs >= 3 streams"
    assert len(set(lengths)) >= 3, "lengths must be DISTINCT (mixed-length)"
    assert args.gen >= 32, "gate needs >= 32 decode steps"

    print("=== MIXED-LENGTH (B>1) decode per-stream parity gate ===")
    print(f"  ckpt: {ckpt}")
    print(f"  prefill lengths (per stream): {lengths}   decode steps: {args.gen}")
    model, cfg = build_model(device)
    step, compact, routed = load_ckpt(ckpt, model, device, cfg)
    model.eval()
    print(f"  loaded step={step} compact={compact} routed={routed}")
    assert compact, "ckpt must be carved/compact"
    assert routed, "this gate must run on a ROUTED ckpt (else the gather path is untested)"
    csa_m = int(model.cfg.csa_compress_ratio)
    for L in lengths:
        assert L >= 2 * csa_m, f"prefill length {L} < 2*csa_m={2 * csa_m} (need first CSA emit)"

    from morph.inference import deploy_quant as pti
    with _AC():
        stats = pti.to_deploy_inference(model, device="cuda")
    print(f"  mortar MLPs packed: {stats['mlps_packed']}  resident: {stats['resident_mb']:.1f} MB")

    vocab = int(model.cfg.vocab_size)
    B = len(lengths)

    # ── distinct random prompts, one per stream at its own length ────────────────
    torch.manual_seed(31337)
    seeds = []
    for b, L in enumerate(lengths):
        row = torch.randint(0, vocab, (L,), device=device)
        row[0] = (row[0] + b) % vocab               # de-correlate stream starts
        seeds.append(row)

    # ── SOLO-EAGER golden per stream (own trajectory) ────────────────────────────
    # golden_logs[b][k] = stream b's logits BEFORE consuming gen-token k:
    #   k=0 → prefill's last logit (predicts pos P_b); k≥1 → after feeding tokens
    #   [t_0..t_{k-1}] (predicts pos P_b+k). golden_toks[b][k] = argmax(golden_logs[b][k]).
    # We generate gen+1 golden logits so the engine's logits[i] (produced by FEEDING
    # token t_i = golden_toks[b][i], i.e. predicting pos P_b+i+1) align with
    # golden_logs[b][i+1]. This +1 offset is intrinsic to "feed-then-compare-output".
    golden_toks, golden_logs = [], []
    for b in range(B):
        tk, lg = _solo_eager_trajectory(model, seeds[b], args.gen + 1)
        golden_toks.append(tk)
        golden_logs.append(lg)

    # ── ENGINE mixed, each stream pinned to its solo-eager token trajectory ───────
    eng_logs, _ = _engine_mixed_trajectory(model, seeds, golden_toks, args.gen)

    # ── per-stream LOGIT parity ──────────────────────────────────────────────────
    print("\n── per-stream logit parity (engine mixed vs solo-eager) ──")
    real_bug = False
    overall_min_cos = 1.0
    for b in range(B):
        worst_c, worst_d, tmatch = 1.0, 0.0, 0
        for step_i in range(args.gen):
            ev = golden_logs[b][step_i + 1].float()    # +1: feed-then-compare alignment
            gv = eng_logs[step_i][b].float()
            c = torch.nn.functional.cosine_similarity(ev, gv, dim=0).item()
            d = float((ev - gv).abs().max())
            worst_c = min(worst_c, c); worst_d = max(worst_d, d)
            if int(ev.argmax()) == int(gv.argmax()):
                tmatch += 1
        overall_min_cos = min(overall_min_cos, worst_c)
        verdict = "OK" if worst_c >= args.logit_cos_min else "REAL-DIVERGENCE"
        print(f"  stream {b} (P={lengths[b]:>3}): min cos={worst_c:.6f}  "
              f"max|Δ|={worst_d:.4f}  argmax_match={tmatch}/{args.gen}  [{verdict}]")
        if worst_c < args.logit_cos_min:
            real_bug = True

    # ── throughput (mixed-length aggregate tok/s) ────────────────────────────────
    print("\n── mixed-length throughput ─────────────────────────────")
    tl = [L for L in args.time_lengths if L >= 2 * csa_m]
    Bt = len(tl)
    print(f"  batch B={Bt}  lengths={tl}")
    from morph.inference.engine import StaticDecodeEngine
    torch.manual_seed(7)
    tseeds = [torch.randint(0, vocab, (L,), device=device) for L in tl]
    tcaches = []
    with _AC():
        for b in range(Bt):
            c, _ = _solo_prefill(model, tseeds[b])
            tcaches.append(c)
        teng = StaticDecodeEngine(model, batch_size=Bt)
        teng.load_from_eager_mixed(tcaches)
        teng.capture()
    torch.cuda.reset_peak_memory_stats()
    sps, msps = _time_mixed(teng, n_warmup=16, n_time=128)
    vram = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {'B':>3} {'steps/s':>9} {'ms/step':>8} {'agg_tok/s':>10} {'VRAM_MB':>9}")
    print(f"  {Bt:>3} {sps:>9.1f} {msps:>8.3f} {Bt * sps:>10.1f} {vram:>9.1f}")

    if not real_bug:
        print(f"\nGATE_MIXEDLEN_DECODE_PASS (per-stream logit cos >= {args.logit_cos_min} "
              f"at {B} distinct prefill lengths {lengths}, {args.gen} decode steps; "
              "each stream matches its own solo-eager golden)")
    else:
        print("\nGATE_MIXEDLEN_DECODE_FAIL (a mixed-length stream's LOGITS diverged from "
              "its solo-eager golden — real position/indexing bug)")
        sys.exit(1)


if __name__ == "__main__":
    main()
