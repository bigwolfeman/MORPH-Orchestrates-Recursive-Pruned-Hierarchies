"""The training stream never yields the eval documents (lxtul/data.py).

Validation reads OWT documents [50_000, 50_000 + n) of the unshuffled shards; training reads
the same shards from document 0. These tests prove, by document id, that with
`exclude_docs` the training stream skips that range and is unchanged before it.

  1. `test_stream_never_yields_dropped_docs`: the PRODUCTION path (`make_loader` with
     `exclude_docs`, real StarCoder2 tokenizer, with and without the Prefetcher thread) on a
     small synthetic arrow set. Every document carries its id as text; the batches are decoded
     and the ids read back. A control without `exclude_docs` must show the range, so the
     test fails if the filter is broken.
  2. `test_real_shards_drop_eval_range`: `drop_docs` on the REAL OWT shards at the REAL eval
     range (read from `eval_rows` doc_stats), by position over the first 50k + range + 300
     documents: docs before 50_000 identical, the next 300 documents are original docs
     [hi, hi + 300). Walking the training stream through `make_loader` to doc 50k would
     tokenize 66M tokens, so the real-shard test stops at the document layer; test 1 covers
     everything above it.

CPU only. Offline: needs the cached StarCoder2 tokenizer and OWT arrow shards.

    HF_HUB_OFFLINE=1 PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_data_hygiene.py
"""
from __future__ import annotations

import glob
import os
import re
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import datasets
import pytest

from lxtul.data import drop_docs, eval_doc_range, make_loader, morph_cfg, tul_runtime

TOKENIZER = "bigcode/starcoder2-7b"
N_DOCS, LO, HI, SEQ, B = 120, 50, 70, 32, 2


@pytest.fixture(scope="module")
def tiny_cfg(tmp_path_factory):
    d = tmp_path_factory.mktemp("owt")
    text = [f"docmark{i:04d} " + "alpha beta gamma delta " * 12 for i in range(N_DOCS)]
    datasets.Dataset.from_dict({"text": text}).save_to_disk(str(d))
    shard = sorted(glob.glob(str(d / "*.arrow")))
    assert shard, "save_to_disk wrote no .arrow shard"
    return SimpleNamespace(data=SimpleNamespace(tokenizer=TOKENIZER, dataset=str(d / "*.arrow"),
                                                seq_len=SEQ))


def _doc_ids_in_stream(cfg, exclude, prefetch, until_docs: int) -> list[int]:
    """Decode the training stream and return the document ids in order of appearance."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    stats: dict = {}
    it = make_loader(cfg, "train", B, tul_rt=None, prefetch=prefetch, doc_stats=stats,
                     exclude_docs=exclude)
    ids: list[int] = []
    while stats.get("docs", 0) < until_docs:
        x, y = next(it)
        for r in range(B):                      # row r of x plus y's last id = the chunk slice
            ids.extend(x[r].tolist() + y[r, -1:].tolist())
    if hasattr(it, "close"):
        it.close()
    found = [int(m) for m in re.findall(r"docmark(\d{4})", tok.decode(ids))]
    return list(dict.fromkeys(found))


@pytest.mark.parametrize("prefetch", [0, 2])
def test_stream_never_yields_dropped_docs(tiny_cfg, prefetch):
    until = N_DOCS - 10                         # stay inside epoch 1
    control = _doc_ids_in_stream(tiny_cfg, None, prefetch, until)
    assert set(range(LO, HI)) <= set(control), "control stream must contain the range"
    got = _doc_ids_in_stream(tiny_cfg, (LO, HI), prefetch, until)
    assert not set(got) & set(range(LO, HI)), f"dropped docs leaked: {sorted(set(got) & set(range(LO, HI)))}"
    assert got[:LO] == list(range(LO)), "documents before the range must be untouched"
    tail = got[LO:]
    assert tail and tail[0] == HI and tail == list(range(HI, HI + len(tail))), \
        "documents after the range must follow it in order"


def test_filter_is_scoped_to_the_stream_thread(tiny_cfg):
    """While a stream is live on thread A, load_dataset on thread B is not filtered, and the
    library function is restored when the stream closes."""
    import threading
    from lxtul.data import _stream_without_docs
    kw = dict(tokenizer_name=TOKENIZER, dataset_name=tiny_cfg.data.dataset, seq_len=SEQ,
              batch_size=B, split="train", skip_samples=0, tul=None, doc_stats=None)
    real = datasets.load_dataset
    files = sorted(glob.glob(tiny_cfg.data.dataset))
    count = lambda: sum(1 for _ in datasets.load_dataset(
        "arrow", data_files=files, split="train", streaming=True))
    live, done, out = threading.Event(), threading.Event(), {}

    def producer():
        gen = _stream_without_docs(LO, HI, **kw)
        next(gen)
        out["patched"] = datasets.load_dataset is not real
        out["own_thread"] = count()
        live.set()
        done.wait()
        gen.close()

    t = threading.Thread(target=producer)
    t.start()
    assert live.wait(120)
    out["other_thread"] = count()
    done.set()
    t.join()
    assert out == {"patched": True, "own_thread": N_DOCS - (HI - LO), "other_thread": N_DOCS}
    assert datasets.load_dataset is real


def test_real_shards_drop_eval_range():
    from lxtul.evaluate import eval_rows
    m = morph_cfg("lxtul")
    stats: dict = {}
    eval_rows(m, tul_runtime(m), 480, 6, doc_stats=stats)
    lo, hi = eval_doc_range(stats)
    assert lo == 50_000 and hi > lo + 300, (lo, hi)
    files = sorted(glob.glob(os.path.expanduser(str(m.data.dataset)), recursive=True))
    assert files, "OWT arrow shards not found"

    def head(ds, n):
        out = []
        for s in ds:
            out.append(hash(s["text"]))
            if len(out) == n:
                break
        return out
    base = datasets.load_dataset("arrow", data_files=files, split="train", streaming=True)
    n = hi + 300
    orig = head(base, n)
    kept = head(drop_docs(base, lo, hi), lo + 300)
    assert kept[:lo] == orig[:lo], "documents before the eval range changed"
    assert kept[lo:] == orig[hi:hi + 300], "documents after the eval range are not [hi, hi+300)"
    assert orig[lo:hi] != orig[hi:hi + len(orig[lo:hi])]       # the dropped slice is not a repeat
