#!/usr/bin/env python
"""Build a VALIDATION shard from Olympiad-AI ``eval_holdout.jsonl`` files (arc E16).

An Olympiad shard ships its held-out documents as a jsonl next to the shard directory
(``stage``, ``question``, ``full_text``, ``answer``, ``input_ids``); those docs are never in
the shard (docs/olympiad-interop.md). This script writes them in the shard format the
curriculum loader reads (``tokens.u16.bin`` / ``doc_offsets.i64.npy`` / ``doc_lens.i32.npy``
/ ``meta.json``) with ``meta.split = "eval_holdout"``, which the loader accepts ONLY as a
``holdout=True`` (validation) source and refuses in any training blend. Docs keep file order;
the loader's seeded draw and ``rewind()`` make every eval score the same docs.

Usage:
  python scripts/olympiad_holdout_shard.py \
      --holdout $OLYMPIAD_REPO/data/morph_shards/olympiad_math_2_10_v3/eval_holdout.jsonl \
      --holdout $OLYMPIAD_REPO/data/morph_shards/olympiad_stage11_13_v1/eval_holdout.jsonl \
      --out $OLYMPIAD_REPO/data/olympiad_bands/holdout
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morph.training.curriculum_data import HOLDOUT_SPLIT  # noqa: E402


def _sibling_shard_meta(holdout_path: str) -> dict:
    """The parent shard's meta.json (tokenizer, vocab), if the holdout sits beside one."""
    for m in sorted(glob.glob(os.path.join(os.path.dirname(holdout_path), "*", "meta.json"))):
        try:
            return json.load(open(m))
        except (OSError, ValueError):
            continue
    return {}


def build_holdout_shard(holdouts: list[str], out: str, name: str = "holdout",
                        eos_id: int = 0) -> dict:
    toks: list[np.ndarray] = []
    lens: list[int] = []
    stages: list[str] = []
    files: list[dict] = []
    parent: dict = {}
    for path in holdouts:
        n0 = len(lens)
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                ids = np.asarray(d["input_ids"], dtype=np.int64)
                if ids.size == 0:
                    raise SystemExit(f"{path}: empty input_ids in doc {len(lens)}")
                if int(ids[-1]) != eos_id:
                    ids = np.append(ids, eos_id)          # every shard doc ends in EOS
                toks.append(ids)
                lens.append(int(ids.size))
                stages.append(str(d.get("stage", "")))
        files.append({"path": os.path.abspath(path), "n_docs": len(lens) - n0})
        if not parent:
            parent = _sibling_shard_meta(path)
    if not lens:
        raise SystemExit("no documents read")
    flat = np.concatenate(toks)
    if int(flat.max()) >= 65536 or int(flat.min()) < 0:
        raise SystemExit(f"token id out of uint16 range: [{flat.min()}, {flat.max()}]")
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    os.makedirs(out, exist_ok=True)
    flat.astype(np.uint16).tofile(os.path.join(out, "tokens.u16.bin"))
    np.save(os.path.join(out, "doc_offsets.i64.npy"), off)
    np.save(os.path.join(out, "doc_lens.i32.npy"), np.asarray(lens, dtype=np.int32))
    meta = {
        "eos_id": int(eos_id),
        "role": parent.get("role", "reasoning_midtrain"),
        "split": HOLDOUT_SPLIT,
        "source_name": name,
        "n_docs": int(len(lens)),
        "n_tokens": int(flat.size),
        "max_len": int(max(lens)),
        "stages": sorted(set(stages), key=lambda s: [int(x) if x.isdigit() else x
                                                     for x in s.split(".")]),
        "per_doc_stage": stages,
        "holdout_files": files,
        "tokenizer_name": parent.get("tokenizer_name"),
        "full_vocab_size": parent.get("full_vocab_size"),
        "max_token_id": int(flat.max()),
        "paths": ["olympiad_curriculum:eval_holdout"],
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "built_by": "scripts/olympiad_holdout_shard.py",
    }
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="append", required=True, help="eval_holdout.jsonl")
    ap.add_argument("--out", required=True, help="shard directory to write")
    ap.add_argument("--name", default="holdout")
    ap.add_argument("--eos-id", type=int, default=0)
    a = ap.parse_args()
    m = build_holdout_shard(a.holdout, a.out, a.name, a.eos_id)
    print(f"{m['source_name']}: {m['n_docs']:,} docs, {m['n_tokens']:,} tokens, max_len "
          f"{m['max_len']}, stages {m['stages'][0]}..{m['stages'][-1]} ({len(m['stages'])}) "
          f"-> {a.out}", flush=True)


if __name__ == "__main__":
    main()
