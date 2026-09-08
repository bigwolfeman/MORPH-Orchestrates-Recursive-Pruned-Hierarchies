#!/usr/bin/env python
"""Build MORPH pretok shards from HuggingFace ``sapientinc/sudoku-extreme`` (arc E17).

A board serializes as fixed-length token ids, one row per line so the TUL boundary
rule (newline ends a span) gives one slot per grid row: the puzzle's 9 rows (blank
cell = digit-0 token), ``<|A|>``, the solution's 9 rows, ``<|/A|>``, EOS. Every board
is EXACTLY the same token count (the digits 0-9 are single starcoder2 tokens and the
serializer never re-tokenizes text, so there is no BPE merge to make it vary), which
is what lets ``morph/configs/sudoku_data.yaml`` pick a ``seq_len`` that never splits a
board across two packed rows (see ``tests/test_sudoku_shards.py``).

``<|A|>`` / ``<|/A|>`` reuse Olympiad-AI's ids (49154 / 49155, docs/olympiad-interop.md)
so ``lab/divergence/olympiad_sweep.py``'s hardcoded ``ANSWER_OPEN`` / ``ANSWER_CLOSE``
read a sudoku held-out jsonl unchanged.

Augmentation (HRM's ``dataset/build_sudoku_dataset.py`` recipe): a random 1-9 digit
relabeling, an independent random permutation of the 3 row-bands (and of the 3 rows
inside each band), the same for column-stacks, and a coin-flip transpose. Each step is
individually validity-preserving (bijective digit relabeling keeps every row/col/box
an "all different" set; band/stack shuffling keeps every box made of one band's 3 rows
x one stack's 3 columns; transpose maps rows<->columns and boxes to boxes) --
``is_valid_solution`` / ``puzzle_consistent`` check this on samples in
``tests/test_sudoku_shards.py``.

Usage:
  python scripts/sudoku_shards.py \
      --out-dir /home/wolfe/morph-scratch/data/sudoku \
      --train-puzzles 1000 --num-aug 1000 --holdout-docs 3000 --seed 0

Output tree (docs/olympiad-interop.md shard format, split=eval_holdout on holdout):
  <out-dir>/train/{tokens.u16.bin,doc_offsets.i64.npy,doc_lens.i32.npy,meta.json}
  <out-dir>/holdout/{tokens.u16.bin,doc_offsets.i64.npy,doc_lens.i32.npy,meta.json,eval_holdout.jsonl}
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from morph.training.curriculum_data import HOLDOUT_SPLIT  # noqa: E402

__all__ = [
    "TOKENIZER_NAME",
    "ANSWER_OPEN",
    "ANSWER_CLOSE",
    "TokenIds",
    "resolve_token_ids",
    "serialize_board",
    "deserialize",
    "is_valid_solution",
    "puzzle_consistent",
    "augment_board",
    "bucket_of_rating",
    "stage_of",
    "stratified_sample",
    "ShardWriter",
]

TOKENIZER_NAME = "bigcode/starcoder2-7b"
# Olympiad-AI structure tokens (docs/olympiad-interop.md; the 17 specials sit at 49152+
# on top of starcoder2's 49152-id vocab). Reused verbatim so olympiad_sweep.py's
# hardcoded ANSWER_OPEN/ANSWER_CLOSE also score a sudoku holdout.
ANSWER_OPEN = 49154
ANSWER_CLOSE = 49155

DEFAULT_RATING_EDGES = [0, 1, 11, 23, 38]  # lower edges; last bucket is [38, inf)
BLANK_CHARS = (".", "0")


class TokenIds:
    """Resolved-once token ids the serializer needs (digits 0-9, newline, EOS)."""

    def __init__(self, tokenizer_name: str = TOKENIZER_NAME):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.digit: dict[int, int] = {}
        for d in range(10):
            ids = tok.encode(str(d), add_special_tokens=False)
            if len(ids) != 1:
                raise RuntimeError(f"digit {d} is not a single token under {tokenizer_name}: {ids}")
            self.digit[d] = int(ids[0])
        nl = tok.encode("\n", add_special_tokens=False)
        if len(nl) != 1:
            raise RuntimeError(f"newline is not a single token under {tokenizer_name}: {nl}")
        self.newline = int(nl[0])
        self.eos = int(tok.eos_token_id if tok.eos_token_id is not None else tok.bos_token_id)
        if self.eos is None:
            raise RuntimeError(f"{tokenizer_name} has no eos/bos to use as a doc separator")
        self.tokenizer_name = tokenizer_name
        self.full_vocab_size = 49169  # base.yaml: StarCoder2 49152 + 17 Olympiad specials


_IDS: TokenIds | None = None


def resolve_token_ids(tokenizer_name: str = TOKENIZER_NAME) -> TokenIds:
    global _IDS
    if _IDS is None or _IDS.tokenizer_name != tokenizer_name:
        _IDS = TokenIds(tokenizer_name)
    return _IDS


# ── Board <-> token ids ───────────────────────────────────────────────────────────


def _digit_of(ch: str) -> int:
    return 0 if ch in BLANK_CHARS else int(ch)


def serialize_board(q: str, a: str, ids: TokenIds | None = None) -> list[int]:
    """81-char puzzle/answer strings ('.' or '0' = blank) -> a fixed-length token list.

    Layout: 9 puzzle rows (9 digit tokens + newline each) + ``<|A|>`` + 9 solution rows
    (same shape) + ``<|/A|>`` + EOS. Length is CONSTANT for every board (each digit is
    always exactly one token id — no text re-tokenization, so no BPE merge varies it).
    """
    if len(q) != 81 or len(a) != 81:
        raise ValueError(f"expected 81-char boards, got len(q)={len(q)} len(a)={len(a)}")
    ids = ids or resolve_token_ids()
    out: list[int] = []
    for grid in (q, a):
        for r in range(9):
            row = grid[r * 9 : (r + 1) * 9]
            for ch in row:
                out.append(ids.digit[_digit_of(ch)])
            out.append(ids.newline)
        out.append(ANSWER_OPEN if grid is q else ANSWER_CLOSE)
    out.append(ids.eos)
    return out


def deserialize(tok_ids: list[int], ids: TokenIds | None = None) -> tuple[str, str]:
    """Inverse of :func:`serialize_board`: token ids -> (q, a) 81-char strings."""
    ids = ids or resolve_token_ids()
    seq = list(tok_ids)
    if seq and seq[-1] == ids.eos:
        seq = seq[:-1]
    if ANSWER_OPEN not in seq or ANSWER_CLOSE not in seq:
        raise ValueError("missing <|A|>/<|/A|> markers")
    ai = seq.index(ANSWER_OPEN)
    ci = seq.index(ANSWER_CLOSE)
    if ci < ai:
        raise ValueError("<|/A|> precedes <|A|>")
    digit_of_id = {v: k for k, v in ids.digit.items()}

    def grid_str(region: list[int]) -> str:
        chars = []
        for t in region:
            if t == ids.newline:
                continue
            if t not in digit_of_id:
                raise ValueError(f"unexpected token {t} inside a grid region")
            d = digit_of_id[t]
            chars.append("." if d == 0 else str(d))
        if len(chars) != 81:
            raise ValueError(f"grid region has {len(chars)} cells, expected 81")
        return "".join(chars)

    return grid_str(seq[:ai]), grid_str(seq[ai + 1 : ci])


# ── Validity ────────────────────────────────────────────────────────────────────


def _grid(s: str) -> np.ndarray:
    return np.asarray([_digit_of(c) for c in s], dtype=np.int8).reshape(9, 9)


def _grid_str(g: np.ndarray) -> str:
    flat = g.reshape(-1)
    return "".join("." if int(v) == 0 else str(int(v)) for v in flat)


def is_valid_solution(a: str) -> bool:
    """True iff ``a`` is a complete, correct 9x9 sudoku solution (no blanks)."""
    if len(a) != 81:
        return False
    if any(c in BLANK_CHARS for c in a):
        return False
    try:
        g = _grid(a)
    except ValueError:
        return False
    full = set(range(1, 10))
    for i in range(9):
        if set(g[i, :].tolist()) != full or set(g[:, i].tolist()) != full:
            return False
    for br in range(3):
        for bc in range(3):
            box = g[br * 3 : br * 3 + 3, bc * 3 : bc * 3 + 3]
            if set(box.reshape(-1).tolist()) != full:
                return False
    return True


def puzzle_consistent(q: str, a: str) -> bool:
    """True iff ``a`` is a valid solution and every filled cell of ``q`` matches it."""
    if len(q) != 81 or len(a) != 81:
        return False
    if not is_valid_solution(a):
        return False
    for cq, ca in zip(q, a):
        if cq not in BLANK_CHARS and cq != ca:
            return False
    return True


# ── Augmentation (HRM recipe) ────────────────────────────────────────────────────


def _band_perm(rng: np.random.Generator) -> np.ndarray:
    """A row (or column) permutation that shuffles the 3 bands and, within each
    band, its 3 rows -- the sudoku symmetry that keeps every 3x3 box a box."""
    bands = rng.permutation(3)
    rows = []
    for b in bands:
        within = rng.permutation(3)
        rows.extend((int(b) * 3 + within).tolist())
    return np.asarray(rows, dtype=np.int64)


def augment_board(q: str, a: str, rng: np.random.Generator) -> tuple[str, str]:
    """One validity-preserving augmentation: digit relabel, band/stack shuffle,
    optional transpose. Blanks stay blank; the augmented (q, a) is a consistent
    puzzle/solution pair whenever the input is (checked in tests)."""
    qg, ag = _grid(q), _grid(a)
    digit_map = np.zeros(10, dtype=np.int8)
    digit_map[1:10] = rng.permutation(np.arange(1, 10, dtype=np.int8))
    qg = digit_map[qg]
    ag = digit_map[ag]
    rperm = _band_perm(rng)
    cperm = _band_perm(rng)
    qg = qg[rperm][:, cperm]
    ag = ag[rperm][:, cperm]
    if rng.random() < 0.5:
        qg = qg.T
        ag = ag.T
    return _grid_str(qg), _grid_str(ag)


# ── Rating buckets ────────────────────────────────────────────────────────────────


def bucket_of_rating(rating: int, edges: list[int]) -> int:
    """Index of the bucket ``rating`` falls in, ``edges`` ascending lower bounds
    (the last bucket is unbounded above)."""
    b = 0
    for i, e in enumerate(edges):
        if rating >= e:
            b = i
        else:
            break
    return b


def stage_of(rating: int, edges: list[int]) -> str:
    """``"<bucket_index>.<rating>"`` -- ``int(stage.split('.')[0])`` recovers the
    bucket, matching the Olympiad convention ``lab/divergence/olympiad_sweep.py`` reads."""
    return f"{bucket_of_rating(rating, edges)}.{rating}"


# ── Streaming shard writer with in-shard dedup ───────────────────────────────────


class ShardWriter:
    """Appends serialized boards straight to ``tokens.u16.bin`` (no giant in-memory
    token array) and tracks doc offsets/lens plus a hash set for exact-sequence dedup."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._fh = open(os.path.join(out_dir, "tokens.u16.bin"), "wb")
        self.offsets: list[int] = [0]
        self.lens: list[int] = []
        self.stages: list[str] = []
        self._hashes: set[bytes] = set()
        self.n_dup = 0

    def _hash(self, arr: np.ndarray) -> bytes:
        return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()

    def contains(self, ids: list[int]) -> bool:
        arr = np.asarray(ids, dtype=np.uint16)
        return self._hash(arr) in self._hashes

    def add(self, ids: list[int], stage: str, dedup: bool = True) -> bool:
        """Returns False (and does not write) iff ``dedup`` and the exact token
        sequence already appeared in this shard."""
        arr = np.asarray(ids, dtype=np.uint16)
        if dedup:
            h = self._hash(arr)
            if h in self._hashes:
                self.n_dup += 1
                return False
            self._hashes.add(h)
        arr.tofile(self._fh)
        self.lens.append(int(arr.size))
        self.offsets.append(self.offsets[-1] + int(arr.size))
        self.stages.append(stage)
        return True

    def close(self, meta_extra: dict) -> dict:
        self._fh.close()
        off = np.asarray(self.offsets, dtype=np.int64)
        lens = np.asarray(self.lens, dtype=np.int32)
        np.save(os.path.join(self.out_dir, "doc_offsets.i64.npy"), off)
        np.save(os.path.join(self.out_dir, "doc_lens.i32.npy"), lens)
        n_tokens = int(off[-1]) if len(off) else 0
        max_id = 0
        if n_tokens:
            flat = np.fromfile(os.path.join(self.out_dir, "tokens.u16.bin"), dtype=np.uint16)
            max_id = int(flat.max())
        meta = {
            "n_docs": int(len(lens)),
            "n_tokens": n_tokens,
            "max_len": int(lens.max()) if len(lens) else 0,
            "per_doc_stage": self.stages,
            "stages": sorted(
                set(self.stages), key=lambda s: [int(x) if x.isdigit() else x for x in s.split(".")]
            ),
            "n_dup_dropped": int(self.n_dup),
            "max_token_id": max_id,
            "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "built_by": "scripts/sudoku_shards.py",
        }
        meta.update(meta_extra)
        json.dump(meta, open(os.path.join(self.out_dir, "meta.json"), "w"), indent=1)
        return meta


# ── CSV reading / stratified sampling ────────────────────────────────────────────


def _read_rows(csv_path: str):
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header != ["source", "question", "answer", "rating"]:
            raise ValueError(f"{csv_path}: unexpected header {header}")
        for row in r:
            source, q, a, rating = row
            yield source, q, a, int(rating)


def rating_histogram(csv_path: str, edges: list[int]) -> tuple[int, dict[int, int], int, int, int]:
    n = 0
    counts: dict[int, int] = {b: 0 for b in range(len(edges))}
    lo = None
    hi = None
    ratings_sum = 0
    for _source, _q, _a, rating in _read_rows(csv_path):
        n += 1
        counts[bucket_of_rating(rating, edges)] += 1
        lo = rating if lo is None else min(lo, rating)
        hi = rating if hi is None else max(hi, rating)
        ratings_sum += rating
    return n, counts, lo or 0, hi or 0, (ratings_sum // n if n else 0)


def stratified_sample(
    csv_path: str, n_total: int, edges: list[int], seed: int
) -> tuple[list[tuple[str, str, str, int]], list[int]]:
    """Reservoir-samples ``n_total`` rows from ``csv_path``, split as evenly as
    possible across the rating buckets. Returns (rows, per_bucket_target)."""
    n_buckets = len(edges)
    per_bucket = [n_total // n_buckets] * n_buckets
    for i in range(n_total % n_buckets):
        per_bucket[i] += 1
    reservoirs: list[list[tuple[str, str, str, int]]] = [[] for _ in range(n_buckets)]
    seen = [0] * n_buckets
    rngs = [np.random.default_rng(seed * 1_000_003 + b) for b in range(n_buckets)]
    for row in _read_rows(csv_path):
        _source, _q, _a, rating = row
        b = bucket_of_rating(rating, edges)
        k = per_bucket[b]
        if k == 0:
            continue
        seen[b] += 1
        if len(reservoirs[b]) < k:
            reservoirs[b].append(row)
        else:
            j = int(rngs[b].integers(0, seen[b]))
            if j < k:
                reservoirs[b][j] = row
    rows: list[tuple[str, str, str, int]] = []
    for b in range(n_buckets):
        rows.extend(reservoirs[b])
    return rows, per_bucket


# ── Build ─────────────────────────────────────────────────────────────────────


def build_train_shard(
    rows: list[tuple[str, str, str, int]],
    out_dir: str,
    num_aug: int,
    edges: list[int],
    seed: int,
    ids: TokenIds,
) -> dict:
    w = ShardWriter(out_dir)
    for pi, (source, q, a, rating) in enumerate(rows):
        stage = stage_of(rating, edges)
        rng = np.random.default_rng(seed * 7_919 + pi)
        for k in range(num_aug):
            qa, aa = augment_board(q, a, rng)
            board_ids = serialize_board(qa, aa, ids)
            w.add(board_ids, stage)
    meta = w.close(
        {
            "eos_id": ids.eos,
            "role": "reasoning_midtrain",
            "source_name": "sudoku_train",
            "paths": ["sudoku_extreme:train"],
            "num_aug": int(num_aug),
            "num_puzzles": len(rows),
            "seed": int(seed),
            "tokenizer_name": ids.tokenizer_name,
            "full_vocab_size": ids.full_vocab_size,
            "rating_edges": edges,
        }
    )
    return meta


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="/home/wolfe/morph-scratch/data/sudoku")
    ap.add_argument("--train-puzzles", type=int, default=1000)
    ap.add_argument("--num-aug", type=int, default=200)
    ap.add_argument("--holdout-docs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rating-edges", default=",".join(str(e) for e in DEFAULT_RATING_EDGES))
    ap.add_argument("--tokenizer-name", default=TOKENIZER_NAME)
    a = ap.parse_args()

    edges = [int(x) for x in a.rating_edges.split(",")]
    if edges != sorted(edges) or edges[0] != 0:
        raise SystemExit(f"--rating-edges must start at 0 and be ascending, got {edges}")
    ids = resolve_token_ids(a.tokenizer_name)
    print(
        f"[sudoku] tokenizer={a.tokenizer_name} eos_id={ids.eos} newline_id={ids.newline} "
        f"digit_ids={ids.digit} answer_open={ANSWER_OPEN} answer_close={ANSWER_CLOSE}",
        flush=True,
    )

    print("[sudoku] downloading train.csv / test.csv from sapientinc/sudoku-extreme…", flush=True)
    from huggingface_hub import hf_hub_download

    train_csv = hf_hub_download("sapientinc/sudoku-extreme", "train.csv", repo_type="dataset")
    test_csv = hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset")

    t0 = time.time()
    n_train_rows, train_hist, train_lo, train_hi, train_med = rating_histogram(train_csv, edges)
    n_test_rows, test_hist, test_lo, test_hi, test_med = rating_histogram(test_csv, edges)
    print(
        f"[sudoku] train.csv: {n_train_rows:,} rows, rating min={train_lo} median~{train_med} "
        f"max={train_hi}, bucket counts (edges {edges}): {train_hist} ({time.time()-t0:.1f}s)",
        flush=True,
    )
    print(
        f"[sudoku] test.csv:  {n_test_rows:,} rows, rating min={test_lo} median~{test_med} "
        f"max={test_hi}, bucket counts (edges {edges}): {test_hist}",
        flush=True,
    )

    board_len = len(serialize_board("." * 81, "1" * 81, ids))
    print(f"[sudoku] one serialized board = {board_len} tokens", flush=True)

    t1 = time.time()
    train_rows, train_per_bucket = stratified_sample(train_csv, a.train_puzzles, edges, a.seed)
    print(
        f"[sudoku] sampled {len(train_rows)} train puzzles (target/bucket {train_per_bucket}, "
        f"{time.time()-t1:.1f}s)",
        flush=True,
    )

    train_out = os.path.join(a.out_dir, "train")
    t2 = time.time()
    train_meta = build_train_shard(train_rows, train_out, a.num_aug, edges, a.seed, ids)
    print(
        f"[sudoku] train shard: {train_meta['n_docs']:,} docs, {train_meta['n_tokens']:,} "
        f"tokens, {train_meta['n_dup_dropped']:,} exact-duplicate boards dropped -> {train_out} "
        f"({time.time()-t2:.1f}s)",
        flush=True,
    )

    # build_train_shard's ShardWriter is closed and out of scope; re-hash straight from
    # the on-disk shard rather than keep a live writer around across the two builds.
    train_hashes = _hashes_from_shard(train_out)

    holdout_rows, holdout_per_bucket = stratified_sample(
        test_csv, a.holdout_docs, edges, a.seed + 1
    )
    print(
        f"[sudoku] sampled {len(holdout_rows)} holdout boards (target/bucket "
        f"{holdout_per_bucket})",
        flush=True,
    )
    holdout_out = os.path.join(a.out_dir, "holdout")
    t3 = time.time()
    holdout_meta, n_leaked = build_holdout_shard(
        holdout_rows, holdout_out, edges, ids, train_hashes
    )
    print(
        f"[sudoku] holdout shard: {holdout_meta['n_docs']:,} docs, {holdout_meta['n_tokens']:,} "
        f"tokens -> {holdout_out} ({time.time()-t3:.1f}s)",
        flush=True,
    )
    print(
        f"[sudoku] leak check: {len(holdout_rows)} held-out boards checked against the train "
        f"shard's {len(train_hashes):,} unique token sequences, {n_leaked} found in train "
        f"(must be 0)",
        flush=True,
    )
    if n_leaked:
        raise SystemExit(f"[sudoku] {n_leaked} held-out boards leaked into the train shard")
    print("[sudoku] done.", flush=True)


def _hashes_from_shard(shard_dir: str) -> set[bytes]:
    off = np.load(os.path.join(shard_dir, "doc_offsets.i64.npy"))
    toks = np.fromfile(os.path.join(shard_dir, "tokens.u16.bin"), dtype=np.uint16)
    out: set[bytes] = set()
    for i in range(len(off) - 1):
        out.add(hashlib.blake2b(toks[off[i] : off[i + 1]].tobytes(), digest_size=16).digest())
    return out


def build_holdout_shard(
    rows: list[tuple[str, str, str, int]],
    out_dir: str,
    edges: list[int],
    ids: TokenIds,
    train_hashes: set[bytes],
) -> tuple[dict, int]:
    w = ShardWriter(out_dir)
    jsonl_path_dir = out_dir
    os.makedirs(jsonl_path_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "eval_holdout.jsonl")
    n_leaked = 0
    with open(jsonl_path, "w") as jf:
        for source, q, a, rating in rows:
            stage = stage_of(rating, edges)
            board_ids = serialize_board(q, a, ids)
            h = hashlib.blake2b(
                np.asarray(board_ids, dtype=np.uint16).tobytes(), digest_size=16
            ).digest()
            if h in train_hashes:
                n_leaked += 1
            w.add(board_ids, stage, dedup=True)
            jf.write(
                json.dumps(
                    {"input_ids": board_ids, "stage": stage, "rating": rating, "source": source}
                )
                + "\n"
            )
    meta = w.close(
        {
            "eos_id": ids.eos,
            "role": "reasoning_midtrain",
            "split": HOLDOUT_SPLIT,
            "source_name": "sudoku_holdout",
            "paths": ["sudoku_extreme:test"],
            "seed": None,
            "tokenizer_name": ids.tokenizer_name,
            "full_vocab_size": ids.full_vocab_size,
            "rating_edges": edges,
        }
    )
    return meta, n_leaked


if __name__ == "__main__":
    main()
