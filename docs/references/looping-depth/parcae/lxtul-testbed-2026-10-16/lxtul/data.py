"""MORPH's OWT stream, tokenizer and TUL packer, served to the Parcae testbed unchanged.

The testbed reads the SAME rows as MORPH: the same StarCoder2 tokenizer, the same
unshuffled OWT shard order, the same boundary rule and the same fixed-shape packer
(`morph.model.tul_layout.pack_tul_batch`). So a CE here is comparable to a MORPH
ledger CE, and a K-sweep here can use MORPH's 480 rows.

`morph_cfg()` composes a MORPH config from MORPH's own config dir (MORPH_ROOT).
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import torch

MORPH_ROOT = Path(os.environ.get(
    "MORPH_ROOT",
    "/mnt/BigAssDrive/00projects/00DeepNet/00-MORPH-Orchestrates-Recursive-Pruned-Hierarchies"))
if str(MORPH_ROOT) not in sys.path:
    sys.path.append(str(MORPH_ROOT))


def morph_cfg(name: str, overrides: list[str] | tuple[str, ...] = ()):
    """Compose a MORPH config from MORPH's own config dir, in a private Hydra context.

    The caller's Hydra context (a running @hydra.main) is saved and put back, so this
    works inside the testbed's entry points and outside them."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    gh = GlobalHydra.instance()
    saved = gh.hydra if gh.is_initialized() else None
    gh.clear()
    try:
        with initialize_config_dir(str(MORPH_ROOT / "morph" / "configs"), version_base=None):
            return compose(config_name=name, overrides=list(overrides))
    finally:
        if saved is not None:
            GlobalHydra.instance().initialize(saved)


def tul_runtime(mcfg):
    """MORPH's resolved TUL runtime: rule, layout spec, slot id. None for a plain cfg."""
    from morph.training.tul_setup import build_tul_runtime
    return build_tul_runtime(mcfg, cache_dir=str(MORPH_ROOT / "ignore" / "tul_cache"))


def layout_for_cells(rt, cells: int):
    """`rt` (MORPH's resolved TUL runtime) with its packed row laid out at `cells` prefix positions per
    slot: L_total = seq_len + cells * max_slots, so a one-cell model sees 1088 positions, not 1280.

    A STATIC layout, the same for every batch (the per-batch Poisson draw is `cell_law` / `relay_cells`,
    which re-lay these rows to fewer cells). MORPH's own `tul_runtime` cannot be asked for it: its
    config validation ties `prefix_k` to the fan keys (`fan_k` 4, joint selection, epivol) and refuses
    the fan-off combination. The loader reads only `data_cfg.prefix_k`, and `val_data_cfg` derives from
    `data_cfg`, so one replace covers train and validation. `cells` equal to the runtime's own returns
    `rt` itself."""
    import dataclasses
    if rt is None:
        return None
    if cells < 1:
        raise ValueError(f"cells must be >= 1, got {cells}")
    if cells == rt.data_cfg.prefix_k:
        return rt
    return dataclasses.replace(rt, data_cfg=dataclasses.replace(rt.data_cfg, prefix_k=int(cells)))


def packed_length(mcfg, rt) -> int:
    """Positions of one packed TUL row for `rt`'s layout: seq_len + prefix_k * max_slots."""
    return int(rt.data_cfg.spec_for(int(mcfg.data.seq_len)).l_total)


# Validation reads the OWT shards in order after this many documents (MORPH's val loader).
EVAL_SKIP_DOCS = 50_000


def eval_doc_range(doc_stats: dict) -> tuple[int, int]:
    """Document ids [lo, hi) of the OWT stream that the eval rows were packed from, given the
    `doc_stats` that `evaluate.eval_rows` filled. The last document may be read only in part."""
    return EVAL_SKIP_DOCS, EVAL_SKIP_DOCS + int(doc_stats["docs"])


def drop_docs(ds, lo: int, hi: int):
    """A streaming dataset with documents [lo, hi) removed, order kept. Documents before `lo`
    stream exactly as before, so a run that stops short of `lo` sees the unchanged stream."""
    if not 0 <= lo <= hi:
        raise ValueError(f"need 0 <= lo <= hi, got [{lo}, {hi})")
    if lo == hi:
        return ds
    from datasets import concatenate_datasets
    return concatenate_datasets([ds.take(lo), ds.skip(hi)])


def _stream_without_docs(lo: int, hi: int, **kw):
    """`create_dataloader(**kw)` whose OWT stream has documents [lo, hi) removed.

    MORPH's loader reads documents through `datasets.load_dataset` (once per pass over the
    data) and has no document filter. This generator wraps that one call, for the life of the
    stream and only on the thread that advances it (the Prefetcher's producer thread, or the
    caller's when prefetch is 0), so a loader built elsewhere is not touched. The
    packing, the TUL rng streams and the batch counter are MORPH's, unchanged.
    """
    import datasets
    from morph.training.data import create_dataloader
    owner = threading.get_ident()
    real, live = datasets.load_dataset, [True]

    def load(*a, **k):
        ds = real(*a, **k)
        return drop_docs(ds, lo, hi) if live[0] and threading.get_ident() == owner else ds

    datasets.load_dataset = load
    try:
        yield from create_dataloader(**kw)
    finally:
        live[0] = False
        if datasets.load_dataset is load:
            datasets.load_dataset = real


def cell_law(tul, seed: int):
    """MORPH's `CellCountLaw` for an LXTULConfig (`cell_count` != "fixed"), else None. P(M = m) is the
    Poisson law clamped to [cell_count_min, cell_count_max or cells], padded to the `cells` role table.
    Draws are counter-based on the loader's batch index and `seed`, so a resumed loader replays them."""
    if tul.cell_count == "fixed":
        return None
    from morph.model.tul_layout import CellCountLaw, poisson_cell_probs
    probs = poisson_cell_probs(float(tul.cell_count_mean), int(tul.cell_count_min),
                               int(tul.cell_count_max) or int(tul.cells))
    return CellCountLaw(probs + (0.0,) * (int(tul.cells) - len(probs)), subset="prefix", seed=int(seed))


def relay_cells(it, tul_rt, seq_len: int, law=None, eval_cells: int = 0):
    """Re-lay every batch of `it` (MORPH TUL batches at the reference width) to M cells per slot:
    M drawn from `law` for batch i (counter-based, CPU), or the fixed `eval_cells`. The same tokens,
    labels, boundaries and spans; only the cell positions after each boundary change (MORPH's
    `relayout_batch`, as its own loader does under `tul.cell_count`)."""
    from morph.model.tul_layout import relayout_batch
    spec, span_cap = tul_rt.data_cfg.spec_for(seq_len), tul_rt.data_cfg.rule.span_cap
    for i, (x, y, lay) in enumerate(it):
        m = law.draw(i)[0] if law is not None else eval_cells
        yield relayout_batch(x, y, lay, m, spec, span_cap) if m != lay.prefix_k else (x, y, lay)


def make_loader(mcfg, split: str, batch_size: int, tul_rt=None, prefetch: int = 4,
                doc_stats: dict | None = None, exclude_docs: tuple[int, int] | None = None,
                cell_law=None, eval_cells: int = 0):
    """Infinite iterator of MORPH batches: (x, y) plain, (x, y, SlotLayout) with TUL.

    `split="validation"` skips the first `EVAL_SKIP_DOCS` documents, as MORPH's val loader does.
    `doc_stats`: filled with the documents read and their tokens (bookkeeping, stream unchanged).
    `exclude_docs=(lo, hi)`: train only; the stream never yields documents [lo, hi). Training
    passes the eval document range here so a long run cannot train on its held-out rows.
    `cell_law` (train) / `eval_cells` (validation): re-lay the rows to a drawn / fixed cell count
    (LXTULConfig.cell_count); both default to the reference width, the batches MORPH serves.
    The re-lay wraps the (possibly document-filtered) stream, so both keys work together.
    """
    from morph.training.data import create_dataloader
    from morph.training.data_placement import Prefetcher
    val = split == "validation"
    if val and exclude_docs is not None:
        raise ValueError("exclude_docs applies to the training stream only")
    tul = None if tul_rt is None else (tul_rt.val_data_cfg if val else tul_rt.data_cfg)
    kw = dict(tokenizer_name=str(mcfg.data.tokenizer), dataset_name=str(mcfg.data.dataset),
              seq_len=int(mcfg.data.seq_len), batch_size=batch_size, split=split,
              skip_samples=EVAL_SKIP_DOCS if val else 0, tul=tul, doc_stats=doc_stats)
    it = iter(create_dataloader(**kw) if exclude_docs is None
              else _stream_without_docs(*exclude_docs, **kw))
    if cell_law is not None or eval_cells:
        if tul_rt is None:
            raise ValueError("a cell count needs the TUL layout (tul_rt)")
        it = relay_cells(it, tul_rt, int(mcfg.data.seq_len), law=None if val else cell_law,
                         eval_cells=eval_cells if val else 0)
    if prefetch > 0:
        it = Prefetcher(it, depth=prefetch, name=f"owt-{split}",
                        pin=torch.cuda.is_available())      # pinning needs CUDA; CPU tests have none
    return it
