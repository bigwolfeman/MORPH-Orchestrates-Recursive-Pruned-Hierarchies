"""MORPH curriculum pretraining — pre-tokenize sources to token shards + length index.

Produces, per source, under <out>/<name>/:
  tokens.u16.bin       raw uint16 token ids, all docs concatenated (each doc ends with EOS sep)
  doc_offsets.i64.npy  [n_docs+1] start-token index of each doc (offsets[-1] == n_tokens)
  doc_lens.i32.npy     [n_docs]   token length of each doc (INCLUDES the trailing EOS)
  meta.json            {name, kind, paths, text_field, tokenizer, n_docs, n_tokens, eos_id, ...}

Why uint16: StarCoder2 vocab = 49152 < 65536, so ids fit in 16 bits (half the disk of int32).
Why per-doc offsets/lens: the length-bucketed curriculum buckets on doc_lens and packs WHOLE
docs (no cross-doc bleed) — both are O(index) once this exists, and we stop re-tokenizing
the corpus every epoch.

PERFORMANCE (measured on a 9950X3D, 16c/32t — see ignore/sat_probe.py, ignore/stream_probe.py):
  * BATCH encode — HF fast tokenizers fan a batch across cores via Rust/Rayon (GIL released).
    One-doc-at-a-time leaves cores idle; batching is a 10-30x win.
  * VECTORIZED writer — each batch is flattened in C (np.fromiter(chain(...))) and written with
    ONE fout.write (not per-doc), and the index is numpy chunks (no boxed-int lists).
  * pyarrow-DIRECT arrow reads (not datasets streaming) — ~3.8x faster row decode, no per-row
    dict / buffering. This + file-sharded mp is the local fast path.
  * FILE-SHARDED multiprocessing (--num-proc) for the arrow bulk source — each worker owns a
    DISJOINT file subset (no duplicated streaming), writes its own part, merged by one cumsum.
    Sweet spot np=8 ≈ 14 M tok/s @ ~19GB RSS (np=16 only +6% for 2x RAM); full OWT ~12 min.
  * CONCURRENT OVERLAP — local CPU sources (arrow/jsonl) run in the FOREGROUND while remote
    network-bound streams (hf/hf_stream: Dolma3/Nemotron) run as BACKGROUND processes. The
    download hides under the local CPU work AND fills the cores owt's mp leaves idle. Remote
    streams are ~3.5 M tok/s each (network-bound); 1 stream/source is optimal (parallel HTTP
    trips HF's 429 rate limiter). Each remote source is capped to a token budget (--remote-max-tokens).

ROLE-SPLIT (no-theater): only DOMAIN text enters pretraining. The synthesis-REASONING jsonls
(commentary / *_reasoning / cross_tradition / reimagined) are SFT/RL gold seed and are refused
here by an explicit denylist. Dolma3's first ~170 alphabetical shards are common_crawl-adult_content-*;
those files are EXCLUDED from the stream (no wasted download) and shard order is shuffled.

Usage:
  PYTHONPATH=$PWD python scripts/pretokenize.py --out data/pretok --limit 50        # gate slice (seq)
  PYTHONPATH=$PWD python scripts/pretokenize.py --out data/pretok --only owt --num-proc 8
  PYTHONPATH=$PWD python scripts/pretokenize.py --out data/pretok --num-proc 8       # FULL: local∥remote
  PYTHONPATH=$PWD python scripts/pretokenize.py --bench --only owt --limit 20000     # throughput probe
"""
from __future__ import annotations
import argparse, glob, itertools, json, os, random, re, sys, time
from typing import Iterator
import numpy as np

# Batched encode_batch is already multi-threaded in Rust; the fork-warning only matters when WE
# fork (the --num-proc path passes a clean env to children). Default-on for the in-proc batch path.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

TOKENIZER = "bigcode/starcoder2-7b"
DATASETS = os.path.expanduser("~/projects/datasets")
HF_CACHE = os.path.expanduser("~/.cache/huggingface/datasets")

# Default per-remote-source token budget ("not the whole thing"). CLI --remote-max-tokens overrides.
REMOTE_MAX_TOKENS = 5_000_000_000

# (name, kind, spec, text_field). kind ∈ {arrow (local), jsonl (local), hf (hub stream),
# hf_stream (hub stream w/ config/file-filter/shuffle + token budget)}.
SOURCES = [
    # ── LOCAL (CPU-bound, disk) ────────────────────────────────────────────────
    # SPECIFIC train-shard glob (matches base.yaml): excludes cache-*.arrow index files
    # whose schema is {indices: uint64}, not text — a broad **/*.arrow sweeps those in and casts fail.
    # PRETOK_OWT_GLOB overrides for testing / pointing at a relocated (e.g. HF-downloaded) cache.
    ("owt",   "arrow", os.environ.get("PRETOK_OWT_GLOB",
                                      f"{HF_CACHE}/openwebtext/**/openwebtext-train-*.arrow"), "text"),
    ("code",  "hf",    "code_search_net",                               "whole_func_string"),
    ("dharma","jsonl", [f"{DATASETS}/dharma/output/84000_pretrain.jsonl",
                        f"{DATASETS}/dharma/output/public_domain_pretrain.jsonl",
                        f"{DATASETS}/dharma/output/youtube_pretrain.jsonl",
                        f"{DATASETS}/dharma/output/flatland_pretrain.jsonl"], "text"),
    ("books", "jsonl", [f"{DATASETS}/reddit/book_pretrain.jsonl"],       "text"),
    # ── REMOTE (network-bound streams — overlap these with the local CPU work) ──
    # Web bulk. Dolma 3: ungated. First ~170 shards are alphabetically common_crawl-adult_content-*;
    # EXCLUDE those (no wasted download) + shuffle shard order for a category-balanced subset.
    ("dolma", "hf_stream", {"repo": "allenai/dolma3_mix-5.5T-1125", "exclude": "adult_content",
                            "jsonl_zst": True,  # robust manual stream — datasets cast-fails on the
                                                # mix's heterogeneous per-subset metadata schemas
                            "shuffle_files": True, "seed": 0, "max_tokens": REMOTE_MAX_TOKENS}, "text"),
    # SYNTHETIC / REASONING — replaces the lost (gated) Nemotron-CC-v2 role. Mix chosen by reading
    # samples (Ai-notes .../PRETOK_RUNBOOK.md). All ungated; blend weights live in the curriculum cfg.
    #   nemotron_qa: grounded synthetic QA (the diverse-QA anchor). Single `text`. Source-grouped +
    #                topic-clustered → MUST .shuffle or a token-cap slice is monodisperse.
    ("nemotron_qa", "hf_stream", {"repo": "fineinstructions-pretraining/nemotron_qa_1T",
                                  "shuffle": True, "buffer_size": 50000, "seed": 0,
                                  "max_tokens": REMOTE_MAX_TOKENS}, "text"),
    #   reasoning: procedurally-generated, verified-correct CoT/reasoning. prompt+answer (NO single
    #              text field) → concatenated. Well-mixed in stream order.
    ("reasoning", "hf_stream", {"repo": "reasoning-core/procedural-pretraining-pile",
                                "config": "default", "max_tokens": REMOTE_MAX_TOKENS}, ["prompt", "answer"]),
    #   math: proof-pile-2/auto-math-text/dm_math/cosmopedia/amps/math_pile. Single `text`. Files are
    #         pretty-printed JSON ARRAYS (datasets' json loader int32-overflows on 7.5M-char docs) →
    #         json_array reader (bulk download + local ijson). shuffle_files mixes the subsets.
    ("math", "hf_stream", {"repo": "aslawliet/math-pretraining-corpus", "json_array": True,
                           "shuffle_files": True, "seed": 0, "max_tokens": REMOTE_MAX_TOKENS}, "text"),
]
SOURCE_MAP = {name: (kind, spec, field) for name, kind, spec, field in SOURCES}
LOCAL_KINDS = ("arrow", "jsonl")   # CPU/disk-bound → foreground; hf/hf_stream → background streams

# Paths that must NEVER be pretrained on (SFT/RL reasoning gold). Hard guard.
DENY = re.compile(r"(commentary|reasoning|cross_tradition|reimagined|_qa)", re.I)

BATCH = 4096   # docs per encode_batch call — big enough to keep all Rayon threads busy.


def _arrow_files(spec) -> list[str]:
    """Resolve an arrow glob to the sorted data-shard file list, with the role-split guard."""
    files = sorted(glob.glob(os.path.expanduser(spec), recursive=True))
    if not files:
        raise RuntimeError(f"no arrow shards match {spec!r}")
    if any(DENY.search(os.path.basename(f)) for f in files):
        raise RuntimeError(f"ROLE-SPLIT VIOLATION in arrow glob {spec!r}")
    return files


def _extract(sample, field):
    """Pull training text from a row. `field` is a str (single column) or a list of columns to
    concatenate with blank lines (e.g. reasoning-core's prompt+answer, which has no single text)."""
    if isinstance(field, (list, tuple)):
        parts = [str(sample.get(f, "")).strip() for f in field]
        return "\n\n".join(p for p in parts if p)
    return sample.get(field, sample.get("text", sample.get("content", "")))


def _arrow_batches(path):
    """Yield record batches from an Arrow IPC file (HF writes the STREAM format; fall back to
    the random-access FILE format). Direct pyarrow is ~3.8x faster than datasets streaming and
    avoids its per-row dict construction + buffering (the per-worker RAM/throughput bottleneck)."""
    import pyarrow as pa
    mm = pa.memory_map(path, "r")
    try:
        reader = pa.ipc.open_stream(mm)
        yield from reader
    except pa.lib.ArrowInvalid:
        mm = pa.memory_map(path, "r")
        reader = pa.ipc.open_file(mm)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def _iter_texts(kind: str, spec, field: str, limit: int | None,
                files: list[str] | None = None) -> Iterator[str]:
    """Yield non-empty texts. For arrow, `files` (when given) overrides the glob so a worker reads
    only its assigned shard files — NO duplicated streaming across workers (file-level sharding)."""
    n_yield = 0
    if kind == "jsonl":
        paths = spec if isinstance(spec, list) else [spec]
        for p in paths:
            if DENY.search(os.path.basename(p)):
                raise RuntimeError(f"ROLE-SPLIT VIOLATION: {p} is reasoning gold, not pretrain bulk.")
            if not os.path.exists(p):
                print(f"  [warn] missing jsonl: {p}", flush=True); continue
            with open(p) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    t = json.loads(line).get(field, "")
                    if t and t.strip():
                        yield t
                        n_yield += 1
                        if limit and n_yield >= limit:
                            return
    elif kind == "arrow":
        # Direct pyarrow column read — no datasets per-row dict / buffering (the per-worker
        # throughput + RAM bottleneck). Batch column → to_pylist is C-level.
        shard_files = files if files is not None else _arrow_files(spec)
        for path in shard_files:
            for batch in _arrow_batches(path):
                for t in batch.column(field).to_pylist():
                    if t and t.strip():
                        yield t
                        n_yield += 1
                        if limit and n_yield >= limit:
                            return
    elif kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(spec, split="train", streaming=True)
        for sample in ds:
            t = _extract(sample, field)
            if t and t.strip():
                yield t
                n_yield += 1
                if limit and n_yield >= limit:
                    return
    elif kind == "hf_stream":
        # Remote hub stream. spec = {repo, [config], [exclude], [shuffle_files], [shuffle],
        # [buffer_size], [seed]}. Token budget enforced by the caller (_write_part max_tokens).
        #   exclude       — drop data files whose path contains this substring (e.g. adult_content)
        #   shuffle_files — randomize shard ORDER (category balance when files==categories)
        #   shuffle       — IterableDataset.shuffle (ROW reservoir + shard reshuffle): REQUIRED for
        #                   source-grouped / front-loaded corpora (nemotron_qa, math) so a token-cap
        #                   slice isn't monodisperse. buffer_size rows held (~2KB each).
        repo = spec["repo"]; seed = spec.get("seed", 0)
        if spec.get("json_array"):
            # Pretty-printed JSON ARRAY files (e.g. math corpus): datasets' json loader overflows
            # pyarrow's int32 block_size on multi-MB docs, and fsspec streaming is latency-bound
            # (~450 items/s). Bulk-DOWNLOAD each file (line speed) then C-parse LOCALLY with ijson
            # (~200k items/s) — 456x faster, constant RAM, no block_size limit.
            import ijson
            from huggingface_hub import HfApi, hf_hub_download
            files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                     if f.endswith(".json")]
            if not files:
                raise RuntimeError(f"{repo}: no .json files found")
            if spec.get("shuffle_files"):
                random.Random(seed).shuffle(files)   # mix subsets (amps/proof_pile/etc are files)
            for rel in files:
                local = hf_hub_download(repo, rel, repo_type="dataset")   # bulk; HF-cached (resume)
                with open(local, "rb") as fh:
                    for obj in ijson.items(fh, "item"):
                        t = _extract(obj, field)
                        if t and t.strip():
                            yield t
                            n_yield += 1
                            if limit and n_yield >= limit:
                                return
            return
        if spec.get("jsonl_zst"):
            # Heterogeneous .jsonl.zst shards (e.g. Dolma3 mix): different subsets carry different
            # metadata columns (warcinfo / sa_remove_ranges / ...), so datasets' streaming schema
            # unification throws CastError when it hits a shard whose columns don't match the first.
            # Bypass it entirely: stream each shard over HTTP, zstd-decompress, json.loads per line,
            # take ONLY `text` — immune to per-shard column drift. No full-file persistence (constant
            # disk). exclude/shuffle_files honored; the caller's max_tokens budget stops the stream.
            import io, zstandard
            from huggingface_hub import HfApi, HfFileSystem
            allf = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                    if f.endswith(".jsonl.zst")]
            if spec.get("exclude"):
                allf = [f for f in allf if spec["exclude"] not in f]
            if not allf:
                raise RuntimeError(f"{repo}: no .jsonl.zst data files (after exclude={spec.get('exclude')!r})")
            if spec.get("shuffle_files"):
                random.Random(seed).shuffle(allf)     # mix CC subsets (one subset per dir/file group)
            fs = HfFileSystem()
            dctx = zstandard.ZstdDecompressor()
            for rel in allf:
                with fs.open(f"datasets/{repo}/{rel}", "rb") as raw, \
                        dctx.stream_reader(raw) as reader:
                    for line in io.TextIOWrapper(reader, encoding="utf-8"):
                        line = line.strip()
                        if not line:
                            continue
                        t = _extract(json.loads(line), field)
                        if t and t.strip():
                            yield t
                            n_yield += 1
                            if limit and n_yield >= limit:
                                return
            return
        from datasets import load_dataset
        if spec.get("exclude"):
            from huggingface_hub import HfApi
            allf = [f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                    if f.endswith((".jsonl.zst", ".jsonl.gz", ".jsonl", ".json", ".parquet"))]
            dataf = [f for f in allf if spec["exclude"] not in f]
            if not dataf:
                raise RuntimeError(f"{repo}: exclude={spec['exclude']!r} removed ALL data files")
            if spec.get("shuffle_files"):
                random.Random(seed).shuffle(dataf)
            ds = load_dataset(repo, data_files=dataf, split="train", streaming=True)
        elif spec.get("config"):
            ds = load_dataset(repo, spec["config"], split="train", streaming=True)
        else:
            ds = load_dataset(repo, split="train", streaming=True)
        if spec.get("shuffle"):
            ds = ds.shuffle(seed=seed, buffer_size=spec.get("buffer_size", 50000))
        for sample in ds:
            t = _extract(sample, field)
            if t and t.strip():
                yield t
                n_yield += 1
                if limit and n_yield >= limit:
                    return
    else:
        raise ValueError(f"unknown kind {kind!r}")


def _batched(it: Iterator[str], n: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def _write_part(texts_iter, tokenizer, eos_id, fout, lens_chunks, counters, label, t0,
                limit, max_tokens=None):
    """Batch-encode `texts_iter`, append uint16 tokens to `fout`, collect per-doc lengths.

    Hot-path is VECTORIZED: each batch is flattened in C (`np.fromiter(chain(...))`) and written
    with ONE `fout.write` — NOT a per-doc asarray+write loop (that serial Python starved the cores
    while Rayon idled). `lens_chunks` accumulates one int32 array per batch (no boxed-int lists);
    offsets are derived once by the caller via cumsum. counters = {'total':int,'docs':int}.
    Stops at `limit` docs and/or `max_tokens` tokens (whichever first) — the latter caps the
    streamed remote subset to a token budget."""
    encode = tokenizer  # transformers fast tokenizer __call__ with a list → Rust batch path
    for batch in _batched(texts_iter, BATCH):
        enc = encode(batch, truncation=False, add_special_tokens=False)["input_ids"]
        for ids in enc:
            ids.append(eos_id)                       # in-place: each doc now ends with EOS
        # lengths (incl. EOS) and a single C-level flatten of the whole batch → one write.
        lens_b = np.fromiter((len(x) for x in enc), dtype=np.int32, count=len(enc))
        n_b = int(lens_b.sum())
        flat = np.fromiter(itertools.chain.from_iterable(enc), dtype=np.uint16, count=n_b)
        fout.write(flat.tobytes())                   # safe: len(tokenizer) < 65536 asserted at startup
        lens_chunks.append(lens_b)
        counters["total"] += n_b
        counters["docs"] += len(enc)
        d, tot = counters["docs"], counters["total"]
        dt = time.perf_counter() - t0
        if max_tokens and tot >= max_tokens:
            pct = f" {tot/max_tokens*100:.0f}% of budget"
        elif limit and d < limit and dt > 0:
            pct = f" eta {((limit - d) / (d / dt)):.0f}s"
        elif max_tokens and dt > 0:
            pct = f" eta {((max_tokens - tot) / (tot / dt)):.0f}s"
        else:
            pct = ""
        print(f"  [{label}] {d:,} docs, {tot:,} tokens, {tot/dt/1e6:.2f} M tok/s{pct}", flush=True)
        if max_tokens and tot >= max_tokens:         # token budget reached → stop the stream
            break


def _worker(args):
    """Subprocess entry: tokenize this worker's assigned arrow FILES → its own part files.
    No duplicated streaming — each worker reads a disjoint file subset."""
    name, field, sdir, part_idx, files, rayon_threads = args
    # Cap THIS worker's Rust thread pool so N workers × rayon_threads ≈ core count (no
    # oversubscription). Must be set BEFORE transformers/tokenizers import.
    os.environ["RAYON_NUM_THREADS"] = str(rayon_threads)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.bos_token_id
    bin_path = os.path.join(sdir, f"tokens.part{part_idx:02d}.bin")
    lens_chunks: list[np.ndarray] = []
    counters = {"total": 0, "docs": 0}
    t0 = time.perf_counter()
    with open(bin_path, "wb") as fout:
        _write_part(_iter_texts("arrow", None, field, None, files=files),
                    tok, eos_id, fout, lens_chunks, counters, f"{name}/p{part_idx:02d}", t0, None)
    lens_arr = np.concatenate(lens_chunks) if lens_chunks else np.zeros(0, dtype=np.int32)
    np.save(os.path.join(sdir, f"lens.part{part_idx:02d}.npy"), lens_arr)
    return part_idx, bin_path, counters["docs"], counters["total"]


def _load_tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _source_job(name, kind, spec, field, out_dir, limit, num_proc, max_tokens, rayon):
    """Spawned background entry for one (usually remote/streaming) source. Caps its Rust threads,
    loads its own tokenizer, writes its own <out>/<name>/ shard independently of other sources."""
    os.environ["RAYON_NUM_THREADS"] = str(rayon)
    try:
        pretokenize_source(name, kind, spec, field, out_dir, None, None, limit, num_proc, max_tokens)
    except Exception as e:          # fail LOUD with which source + why (e.g. gated/lapsed access)
        print(f"  [{name}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
        raise


def pretokenize_source(name, kind, spec, field, out_dir, tokenizer, eos_id, limit, num_proc,
                       max_tokens=None):
    if tokenizer is None:                            # spawned background job loads its own
        tokenizer = _load_tok()
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.bos_token_id
    sdir = os.path.join(out_dir, name)
    os.makedirs(sdir, exist_ok=True)
    bin_path = os.path.join(sdir, "tokens.u16.bin")

    # Parallel = arrow source, num_proc>1, FULL run (no limit). File-level sharding only; jsonl/hf
    # (small) and gate slices (--limit) go single-proc batched. (round-robin so big/small shards mix.)
    files = _arrow_files(spec) if kind == "arrow" else None
    parallel = num_proc > 1 and kind == "arrow" and not limit and files and len(files) >= num_proc
    if not parallel:
        if num_proc > 1 and kind == "arrow" and limit:
            print(f"  [{name}] --limit set → single-proc (parallel is for the full run)", flush=True)
        lens_chunks: list[np.ndarray] = []
        counters = {"total": 0, "docs": 0}
        t0 = time.perf_counter()
        with open(bin_path, "wb") as fout:
            _write_part(_iter_texts(kind, spec, field, limit), tokenizer, eos_id,
                        fout, lens_chunks, counters, name, t0, limit, max_tokens)
        lens_arr = np.concatenate(lens_chunks) if lens_chunks else np.zeros(0, dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(lens_arr, dtype=np.int64)]).tolist()
        lens = lens_arr.tolist()
        n_docs, n_tokens = int(lens_arr.size), int(offsets[-1])
    else:
        import multiprocessing as mp
        rayon = max(1, (os.cpu_count() or num_proc) // num_proc)   # bound per-worker Rust threads
        buckets = [files[i::num_proc] for i in range(num_proc)]    # round-robin file assignment
        jobs = [(name, field, sdir, i, buckets[i], rayon) for i in range(num_proc)]
        print(f"  [{name}] {num_proc} workers, {len(files)} files (file-sharded, "
              f"rayon={rayon}/worker) ...", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_proc) as pool:
            parts = sorted(pool.map(_worker, jobs))     # [(part_idx, bin_path, docs, tokens), ...]
        # Merge: concatenate part blobs in part order, gather part lens, then derive global
        # offsets in one cumsum at the end (the blob order == lens order == part order).
        lens_all: list[np.ndarray] = []
        with open(bin_path, "wb") as fout:
            for pidx, part_bin, _d, _t in parts:
                with open(part_bin, "rb") as pf:
                    while True:
                        chunk = pf.read(1 << 24)        # 16 MiB streaming copy
                        if not chunk:
                            break
                        fout.write(chunk)
                lens_all.append(np.load(os.path.join(sdir, f"lens.part{pidx:02d}.npy")))
                os.remove(part_bin)
                os.remove(os.path.join(sdir, f"lens.part{pidx:02d}.npy"))
        lens_arr = (np.concatenate(lens_all) if lens_all else np.zeros(0, dtype=np.int32))
        offsets = np.concatenate([[0], np.cumsum(lens_arr, dtype=np.int64)]).tolist()
        lens = lens_arr.tolist()
        n_docs, n_tokens = int(lens_arr.size), int(offsets[-1])

    np.save(os.path.join(sdir, "doc_offsets.i64.npy"), np.asarray(offsets, dtype=np.int64))
    np.save(os.path.join(sdir, "doc_lens.i32.npy"), np.asarray(lens, dtype=np.int32))
    assert int(np.asarray(offsets[-1])) == n_tokens, "offset/token mismatch — merge bug"
    meta = {"name": name, "kind": kind, "paths": spec, "text_field": field,
            "tokenizer": TOKENIZER, "n_docs": n_docs, "n_tokens": n_tokens,
            "eos_id": eos_id, "dtype": "uint16", "batch": BATCH, "num_proc": num_proc,
            "max_tokens": max_tokens}
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [{name}] DONE: {n_docs:,} docs, {n_tokens:,} tokens → {sdir}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/pretok")
    ap.add_argument("--only", default=None, help="comma-list of source names")
    ap.add_argument("--limit", type=int, default=None, help="docs per source (gate slice)")
    ap.add_argument("--num-proc", type=int, default=1,
                    help="worker processes for the arrow bulk source, file-sharded "
                         "(jsonl/hf + --limit slices always in-proc)")
    ap.add_argument("--bench", action="store_true",
                    help="throughput probe: tokenize --limit docs of --only, report M tok/s, no shards kept")
    ap.add_argument("--sequential", action="store_true",
                    help="disable the concurrent overlap (process sources one at a time)")
    ap.add_argument("--remote-max-tokens", type=float, default=None,
                    help="override the per-remote-source token budget (default 5e9; e.g. 5e7 for a test)")
    ap.add_argument("--bg-rayon", type=int, default=4,
                    help="Rust threads per background (streaming) source — network-bound, keep small")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.bos_token_id
    assert eos_id is not None, "tokenizer has no eos/bos to use as a doc separator"
    # One-time correctness guard: EVERY producible id (incl. added specials) must fit uint16.
    assert len(tok) < 65536, f"tokenizer has {len(tok)} ids — overflows uint16; bump dtype to int32"
    assert tok.is_fast, "need a FAST tokenizer for the Rust batch path; got a slow/python tokenizer"
    print(f"[pretok] tokenizer={TOKENIZER} (fast, {len(tok)} ids) eos_id={eos_id} "
          f"out={args.out} limit={args.limit} num_proc={args.num_proc} batch={BATCH}")

    if args.bench:
        name = (args.only or "owt").split(",")[0]
        kind, spec, field = SOURCE_MAP[name]
        n = args.limit or 20000
        counters = {"total": 0, "docs": 0}
        lens_chunks: list[np.ndarray] = []
        t0 = time.perf_counter()
        with open(os.devnull, "wb") as devnull:
            _write_part(_iter_texts(kind, spec, field, n), tok, eos_id,
                        devnull, lens_chunks, counters, f"bench:{name}", t0, n)
        dt = time.perf_counter() - t0
        print(f"[bench] {name}: {counters['docs']:,} docs, {counters['total']:,} tokens in "
              f"{dt:.1f}s = {counters['total']/dt/1e6:.2f} M tok/s "
              f"(mean {counters['total']/max(1,counters['docs']):.0f} tok/doc)")
        return

    want = set(args.only.split(",")) if args.only else None
    os.makedirs(args.out, exist_ok=True)
    selected = [s for s in SOURCES if not (want and s[0] not in want)]

    def _mt(spec):   # per-source token budget: remote sources only, CLI-overridable
        base = spec.get("max_tokens") if isinstance(spec, dict) else None
        return args.remote_max_tokens if (base and args.remote_max_tokens) else base

    # CONCURRENT OVERLAP: local CPU-bound sources (arrow/jsonl) run in the FOREGROUND while remote
    # network-bound streams (hf/hf_stream) run as BACKGROUND processes → network hides under local
    # CPU and fills the cores owt's mp leaves idle. Disabled for gate slices / single source.
    fg = [s for s in selected if s[1] in LOCAL_KINDS]
    bg = [s for s in selected if s[1] not in LOCAL_KINDS]
    concurrent = (not args.sequential) and (not args.limit) and bg and (len(selected) > 1)

    procs = []
    if concurrent:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        for name, kind, spec, field in bg:
            print(f"[pretok] launching background stream: {name} ({kind}) "
                  f"max_tokens={_mt(spec)}", flush=True)
            p = ctx.Process(target=_source_job, args=(name, kind, spec, field, args.out,
                                                       args.limit, 1, _mt(spec), args.bg_rayon))
            p.start(); procs.append((name, p))
    else:
        fg = selected          # sequential: everything in the foreground, in order

    summary = []
    for name, kind, spec, field in fg:
        print(f"[pretok] source={name} kind={kind} (foreground)")
        summary.append(pretokenize_source(name, kind, spec, field, args.out, tok,
                                           eos_id, args.limit, args.num_proc, _mt(spec)))
    fails = []
    for name, p in procs:
        p.join()
        # Judge success by the WRITTEN SHARD, not the exit code: meta.json is written LAST, so its
        # presence + a blob whose size matches n_tokens*2 == a complete shard. A nonzero exitcode
        # with a complete shard is the datasets/HF streaming C-extension teardown SIGABRT (fires at
        # interpreter shutdown AFTER all rows yielded) — NOT a real failure.
        sdir = os.path.join(args.out, name)
        mp_path = os.path.join(sdir, "meta.json")
        m = json.load(open(mp_path)) if os.path.exists(mp_path) else None
        complete = (m is not None and
                    os.path.getsize(os.path.join(sdir, "tokens.u16.bin")) == m["n_tokens"] * 2)
        if complete:
            summary.append(m)
            if p.exitcode != 0:
                print(f"  [{name}] shard complete ({m['n_tokens']:,} tok); ignoring teardown "
                      f"exit {p.exitcode} (datasets stream-shutdown artifact)", flush=True)
        else:
            fails.append((name, p.exitcode))

    print("\n[pretok] summary:")
    for m in sorted(summary, key=lambda x: x["name"]):
        print(f"  {m['name']:10s} {m['n_docs']:>11,} docs  {m['n_tokens']:>14,} tokens")
    if fails:
        # e.g. Nemotron gated-access lapsed → re-request. Loud, non-zero exit, but local shards kept.
        print(f"\n[pretok] ⚠ FAILED sources: {fails} (other shards written OK)")
        sys.exit(1)


if __name__ == "__main__":
    main()
