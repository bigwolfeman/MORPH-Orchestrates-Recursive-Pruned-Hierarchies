#!/usr/bin/env python
"""Per-band VIEWS of the Olympiad-AI shards for a staged data curriculum (arc E15).

An Olympiad shard (docs/olympiad-interop.md) is written in stage order: ``meta.stages`` lists
the substages and ``meta.per_stage[s].shard_docs`` how many consecutive documents each one
holds, so a band of stages is a contiguous document range. A view is a shard directory
that shares the parent's ``tokens.u16.bin`` (a symlink) and carries only the band's slice of
``doc_offsets`` / ``doc_lens`` (offsets stay absolute into the shared file) plus a
``meta.json`` of its own. MORPH's curriculum loader reads a view like any shard, so a
``curriculum.stages[i].blend`` over the views is the curriculum.

Usage:
  python scripts/olympiad_band_views.py \
      --shard $OLYMPIAD_REPO/data/morph_shards/olympiad_math_2_10_v3/olympiad_math \
      --shard $OLYMPIAD_REPO/data/morph_shards/olympiad_stage11_13_v1/olympiad_stage11_13 \
      --band b2_3=2-3 --band b4_5=4-5 --band b6_7=6-7 --band b8_10=8-10 --band b11_13=11-13 \
      --out $OLYMPIAD_REPO/data/olympiad_bands
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import numpy as np


def load_shard(sdir: str) -> dict:
    meta = json.load(open(os.path.join(sdir, "meta.json")))
    off = np.load(os.path.join(sdir, "doc_offsets.i64.npy"))
    lens = np.load(os.path.join(sdir, "doc_lens.i32.npy"))
    stages = [str(s) for s in meta["stages"]]
    counts = [int(meta["per_stage"][s]["shard_docs"]) for s in stages]
    if sum(counts) != len(lens):
        raise SystemExit(f"{sdir}: per_stage shard_docs sum {sum(counts)} != n_docs {len(lens)}")
    starts = np.concatenate([[0], np.cumsum(counts)])
    return {"dir": sdir, "meta": meta, "off": off, "lens": lens, "stages": stages,
            "starts": starts}


def band_of(stage: str) -> int:
    return int(stage.split(".")[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", action="append", required=True, help="parent shard dir")
    ap.add_argument("--band", action="append", required=True, help="NAME=LO-HI (stage bands)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shards = [load_shard(s) for s in a.shard]
    os.makedirs(a.out, exist_ok=True)
    for spec in a.band:
        name, rng = spec.split("=")
        lo, hi = (int(x) for x in rng.split("-"))
        pieces = []
        for sh in shards:
            idx = [i for i, s in enumerate(sh["stages"]) if lo <= band_of(s) <= hi]
            if not idx:
                continue
            if idx != list(range(idx[0], idx[-1] + 1)):
                raise SystemExit(f"{name}: stages {lo}-{hi} are not contiguous in {sh['dir']}")
            pieces.append((sh, int(sh["starts"][idx[0]]), int(sh["starts"][idx[-1] + 1]),
                           [sh["stages"][i] for i in idx]))
        if len(pieces) != 1:
            raise SystemExit(f"{name}: a view needs exactly one parent shard, found "
                             f"{[p[0]['dir'] for p in pieces]}")
        sh, d0, d1, stages = pieces[0]
        off = sh["off"][d0:d1 + 1]
        lens = sh["lens"][d0:d1]
        if not np.array_equal(np.diff(off), lens):
            raise SystemExit(f"{name}: doc_lens do not match doc_offsets on [{d0}, {d1})")
        vdir = os.path.join(a.out, name)
        os.makedirs(vdir, exist_ok=True)
        link = os.path.join(vdir, "tokens.u16.bin")
        target = os.path.abspath(os.path.join(sh["dir"], "tokens.u16.bin"))
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(target, link)
        np.save(os.path.join(vdir, "doc_offsets.i64.npy"), off.astype(np.int64))
        np.save(os.path.join(vdir, "doc_lens.i32.npy"), lens.astype(np.int32))
        pm = sh["meta"]
        meta = {
            "eos_id": int(pm["eos_id"]),
            "role": pm.get("role", "reasoning_midtrain"),
            "source_name": name,
            "view_of": target,
            "parent_source_name": pm.get("source_name"),
            "doc_range": [d0, d1],
            "stages": stages,
            "band": [lo, hi],
            "n_docs": int(d1 - d0),
            "n_tokens": int(off[-1] - off[0]),
            "tokenizer_name": pm.get("tokenizer_name"),
            "full_vocab_size": pm.get("full_vocab_size"),
            "max_token_id": pm.get("max_token_id"),
            "paths": pm.get("paths"),
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/olympiad_band_views.py",
        }
        json.dump(meta, open(os.path.join(vdir, "meta.json"), "w"), indent=1)
        print(f"{name:8s} stages {stages[0]}..{stages[-1]} ({len(stages)}) docs [{d0}, {d1}) "
              f"= {d1 - d0:,}  tokens {meta['n_tokens'] / 1e6:.1f}M  -> {vdir}", flush=True)


if __name__ == "__main__":
    main()
