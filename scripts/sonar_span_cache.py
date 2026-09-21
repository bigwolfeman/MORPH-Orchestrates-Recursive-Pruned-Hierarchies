#!/usr/bin/env python
"""Build the SONAR span-embedding cache a ``tul.code_target_source: sonar`` arm reads.

Rung P3 of LXTUL-P: the LCTUL flow thinker's target stops being E's verbatim
reconstruction code of the next span and becomes SONAR's 1024-d sentence embedding of that
span's TEXT. The cache is keyed by the span's TOKEN IDS, so the trainer's lookup and this
script's write cannot drift through a tokenizer round trip
(:func:`morph.model.sonar_cache.span_token_hash`).

TWO STAGES, TWO VENVS, ON PURPOSE. SONAR pulls ``fairseq2`` and its own ``torch``, and the
MORPH training venv must not be touched (breaking its torch breaks every run). So:

  stage `spans`  — the MORPH venv, CPU only. Iterates the trainer's own rows for a config
                   (``morph.training.data.create_dataloader`` with the run's
                   ``tul_rt.data_cfg``: the same deterministic, unshuffled stream the
                   trainer serves), cuts each row into the PACKER's spans, hashes the token
                   ids, detokenises, dedupes, and writes `spans.keys.npy` + `spans.jsonl`.
  stage `encode` — a SONAR venv on a GPU host. Reads those two files, encodes with
                   ``text_sonar_basic_encoder``, and writes `index.npy` (uint64, sorted) +
                   `emb.f16.npy` ([N, 1024] float16) + `meta.json`.

Only stage `spans` has to run on the machine that owns the training corpus: the dataset is
a local arrow glob, so the stream is a property of THAT disk. Stage `encode` needs only the
span texts, which travel as one JSONL file.

WHICH SPANS. For slot ``s`` the flow target is span ``s+1`` — the span
``TULCodeEncoder`` pools today. This script caches exactly the keys
:func:`morph.model.sonar_cache.next_span_hashes` produces over the same rows, so a MISS at
train time means the stream, the packer or the config moved, never that the script skipped
a span.

    # 1. spans (MORPH venv, no GPU)
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python scripts/sonar_span_cache.py spans \
        --config tul_code_cfg_tlow --steps 2000 --split train \
        --out /home/wolfe/sonar-cache/tul_code_cfg_tlow

    # 2. encode (SONAR venv, GPU host)
    PYTHONPATH=. python scripts/sonar_span_cache.py encode \
        --out /mnt/bigdata/sonar-cache/tul_code_cfg_tlow --batch 256

`--split val` uses the trainer's validation loader (``split="validation"``,
``skip_samples=50_000``, ``tul_rt.val_data_cfg``) and `--steps` is then the run's
``training.n_eval_batches``. Train and val spans belong in the SAME cache directory: pass
the same `--out` and stage `spans` merges (it reads any existing `spans.keys.npy`).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np


def _load_sonar_cache():
    """``morph.model.sonar_cache`` by path — the SONAR venv has no MORPH install."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "morph", "model", "sonar_cache.py")
    spec = importlib.util.spec_from_file_location("_morph_sonar_cache", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


# ── stage: spans ────────────────────────────────────────────────────────────────

def stage_spans(args) -> int:
    from hydra import compose, initialize_config_dir
    from transformers import AutoTokenizer

    from morph.training.data import create_dataloader
    from morph.training.tul_setup import build_tul_runtime

    sc = _load_sonar_cache()
    cfg_dir = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "morph", "configs"))
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name=args.config, overrides=list(args.override or []))
    tul_rt = build_tul_runtime(cfg)
    if tul_rt is None:
        raise SystemExit(f"{args.config} has no tul: block — nothing to cut into spans.")

    seq_len = int(cfg.data.seq_len)
    batch = int(args.batch or cfg.training.batch_size)
    if args.split == "train":
        steps = int(args.steps)
        loader = create_dataloader(str(cfg.data.tokenizer), str(cfg.data.dataset), seq_len,
                                   batch, split="train", bag_size=0,
                                   tul=tul_rt.data_cfg)
    else:
        steps = int(args.steps or getattr(cfg.training, "n_eval_batches", 20))
        loader = create_dataloader(str(cfg.data.tokenizer), str(cfg.data.dataset), seq_len,
                                   batch, split="validation", skip_samples=50_000,
                                   tul=tul_rt.val_data_cfg)
    it = iter(loader)
    tok = AutoTokenizer.from_pretrained(str(cfg.data.tokenizer))

    keys: list[np.ndarray] = []
    texts: list[str] = []
    n_seen = 0
    t0 = time.perf_counter()
    for step in range(steps):
        try:
            inp, _labels, layout = next(it)
        except StopIteration:
            print(f"[spans] stream ended at step {step}", flush=True)
            break
        ids = inp.numpy()
        bag = layout.bag_id.numpy()
        sm = layout.slot_mask.numpy()
        S = int(layout.slot_index.shape[1])
        h, n_tok, span_ids = sc.span_hashes(ids, bag, sm, S)
        tgt_h, tgt_has = sc.next_span_hashes(h, n_tok)
        valid = layout.slot_valid.numpy()
        want = tgt_has & valid
        want[:, :S - 1] &= valid[:, 1:S]
        for b in range(ids.shape[0]):
            for s in np.flatnonzero(want[b]):
                keys.append(np.uint64(tgt_h[b, s]))
                texts.append(tok.decode(span_ids[b][int(s) + 1].tolist()))
        n_seen += int(want.sum())
        if (step + 1) % max(1, steps // 20) == 0 or step + 1 == steps:
            el = time.perf_counter() - t0
            print(f"[spans] step {step + 1}/{steps}  spans {n_seen}  "
                  f"unique-so-far n/a  {el:.1f}s  ({(step + 1) / max(el, 1e-9):.2f} step/s)",
                  flush=True)

    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    k_new = np.asarray(keys, dtype=np.uint64)
    kp, tp = os.path.join(out, "spans.keys.npy"), os.path.join(out, "spans.jsonl")
    if os.path.isfile(kp) and os.path.isfile(tp) and not args.fresh:
        k_old = np.load(kp)
        with open(tp) as f:
            t_old = [json.loads(line) for line in f]
        if k_old.shape[0] != len(t_old):
            raise RuntimeError(f"{out}: {kp} has {k_old.shape[0]} keys but {tp} has "
                               f"{len(t_old)} lines")
        k_all = np.concatenate([k_old, k_new])
        t_all = t_old + texts
    else:
        k_all, t_all = k_new, texts
    # Dedupe: first occurrence wins (a key IS the token sequence, so duplicates are the
    # same span and the encoder is deterministic; first-wins makes the write reproducible).
    order = np.argsort(k_all, kind="stable")
    keep_sorted = np.ones(order.shape[0], dtype=bool)
    if order.shape[0] > 1:
        keep_sorted[1:] = np.diff(k_all[order]) > 0
    keep = np.sort(order[keep_sorted])
    k_u = k_all[keep]
    np.save(kp, k_u)
    with open(tp, "w") as f:
        for i in keep:
            f.write(json.dumps(t_all[int(i)]) + "\n")
    el = time.perf_counter() - t0
    dup = 1.0 - (len(k_u) / max(len(k_all), 1))
    per_1k = n_seen / max(steps, 1) * 1000.0
    print(f"[spans] DONE  rows {steps}x{batch}  spans written this pass {n_seen} "
          f"({per_1k:.0f} per 1,000 steps)  cache now {len(k_u)} unique of {len(k_all)} "
          f"(dedupe {dup * 100:.1f}%)  wall {el:.1f}s "
          f"({el / max(steps, 1) * 1000.0:.1f}s per 1,000 steps)", flush=True)
    print(f"[spans] projected emb.f16 bytes at {len(k_u)} rows: "
          f"{len(k_u) * 1024 * 2 / 2**20:.1f} MiB", flush=True)
    return 0


# ── stage: encode ───────────────────────────────────────────────────────────────

def stage_encode(args) -> int:
    import torch
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    sc = _load_sonar_cache()
    out = os.path.expanduser(args.out)
    kp, tp = os.path.join(out, "spans.keys.npy"), os.path.join(out, "spans.jsonl")
    keys = np.load(kp)
    with open(tp) as f:
        texts = [json.loads(line) for line in f]
    if keys.shape[0] != len(texts):
        raise RuntimeError(f"{kp} has {keys.shape[0]} keys but {tp} has {len(texts)} lines")
    print(f"[encode] {len(texts)} spans from {out}", flush=True)

    dev = torch.device(args.device)
    dt = torch.float16 if dev.type == "cuda" else torch.float32
    pipe = TextToEmbeddingModelPipeline(encoder=sc.SONAR_ENCODER,
                                        tokenizer=sc.SONAR_ENCODER, device=dev, dtype=dt)
    emb = np.zeros((len(texts), sc.SONAR_DIM), dtype=np.float32)
    chunk = int(args.chunk)
    t0 = time.perf_counter()
    for a in range(0, len(texts), chunk):
        z = min(a + chunk, len(texts))
        # A span can be empty after decode only if the packer gave it no token, which
        # `spans` already filters; SONAR still refuses an empty string, so guard loudly.
        seg = [t if t.strip() else " " for t in texts[a:z]]
        e = pipe.predict(seg, source_lang=sc.SONAR_LANG, batch_size=int(args.batch))
        emb[a:z] = e.to(torch.float32).cpu().numpy()
        el = time.perf_counter() - t0
        print(f"[encode] {z}/{len(texts)}  {z / max(el, 1e-9):.1f} sent/s  {el:.1f}s",
              flush=True)
    meta = sc.write_sonar_cache(out, keys, emb, meta={
        "source": args.out, "device": str(dev), "dtype": str(dt),
        "sent_per_s": len(texts) / max(time.perf_counter() - t0, 1e-9),
    })
    marker = os.path.join(out, "DONE")
    with open(marker, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    sz = sum(os.path.getsize(os.path.join(out, n)) for n in os.listdir(out)
             if os.path.isfile(os.path.join(out, n)))
    print(f"[encode] DONE  {meta['n_rows']} rows  {sz / 2**20:.1f} MiB total in {out}  "
          f"marker {marker}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("spans", help="cut the trainer's rows into spans (MORPH venv, CPU)")
    s.add_argument("--config", required=True, help="Hydra config name, e.g. tul_code_cfg_tlow")
    s.add_argument("--override", nargs="*", default=None, help="Hydra overrides")
    s.add_argument("--steps", type=int, default=2000)
    s.add_argument("--batch", type=int, default=0, help="0 = the config's training.batch_size")
    s.add_argument("--split", choices=("train", "val"), default="train")
    s.add_argument("--out", required=True)
    s.add_argument("--fresh", action="store_true", help="ignore an existing spans.* pair")
    s.set_defaults(fn=stage_spans)

    e = sub.add_parser("encode", help="encode the spans with SONAR (SONAR venv, GPU host)")
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=256)
    e.add_argument("--chunk", type=int, default=16384, help="spans per progress report")
    e.add_argument("--device", default="cuda")
    e.set_defaults(fn=stage_encode)

    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
