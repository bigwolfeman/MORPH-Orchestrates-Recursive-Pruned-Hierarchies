"""MORPH data pipeline — OpenWebText + StarCoder2 tokenizer, streaming token buffer.

Usage:
    from morph.training.data import create_dataloader
    loader = create_dataloader("bigcode/starcoder2-7b", "Skylion007/openwebtext",
                               seq_len=4096, batch_size=4, split="train")
    x, y = next(loader)   # x: [B, T], y: [B, T] (labels = x shifted by 1)
"""

from __future__ import annotations

import torch
from typing import Generator, Tuple

from morph.model.tul_layout import TulDataConfig, pack_tul_batch

__all__ = ["create_dataloader"]


def create_dataloader(
    tokenizer_name: str,
    dataset_name: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    skip_samples: int = 0,
    bag_size: int = 0,
    tul: TulDataConfig | None = None,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Infinite generator of (input_ids, labels) pairs.

    Streams the dataset, accumulates tokens in a buffer, and yields
    contiguous (seq_len+1)-token chunks sliced into inputs + labels.

    Token-Superposition Training (TST, ``bag_size = s > 0``)
    -------------------------------------------------------
    Serves the SUPERPOSITION-phase batch: ``s·(seq_len+1)`` raw tokens per sample →
      input_ids  = raw[:, : s·seq_len]                  → [B, s·seq_len]
      label_bags = raw[:, s : s·(seq_len+1)] reshaped   → [B, seq_len, s]
    so s-token position ``p`` (covering raw ``[p·s, p·s+s-1]``) predicts the next bag
    ``[(p+1)·s, (p+2)·s-1]`` (net shift = s → 1-shift NTP labels then s-1 more, the
    paper's causality rule). The MODEL averages the s input-token embeddings into one
    s-token, so it processes ``seq_len`` positions — equal-FLOPs/VRAM to the baseline,
    ``s×`` more raw tokens ingested per step. ``bag_size == 0`` → standard NTP (below).

    TUL span layout (``tul`` set — docs/tul-spec.md §4 [W]: this arrow path is the
    one the TUL arms run)
    ---------------------------------------------------------------------------
    Yields a 3-TUPLE ``(input_ids, labels, slot_layout)``, each row a fixed-shape
    ``L_total = seq_len + prefix_k · max_slots`` mix of token positions and inserted
    slot positions. The token count therefore VARIES per row (spec §3.1, invariant 5)
    while the shape does not. Segmentation and packing live in
    ``morph.model.tul_layout`` — the same functions the generator uses (invariant 1).
    Mutually exclusive with ``bag_size`` (TUL runs with ``bag_size 0``, invariant 6).

    StarCoder2 does not use BOS by default; we add it at document boundaries
    so segment boundaries are explicit to the model.

    Args:
        tokenizer_name: HuggingFace tokenizer id (e.g. "bigcode/starcoder2-7b").
        dataset_name:   HuggingFace dataset id (e.g. "Skylion007/openwebtext").
        seq_len:        Number of tokens per sequence (T dimension).
        batch_size:     Number of sequences per batch (B dimension).
        split:          Dataset split ("train" / "validation").
        skip_samples:   Number of documents to skip at the start of the stream.
                        Used to offset the validation loader from train.

    Yields:
        (input_ids, labels): each [batch_size, seq_len], dtype=torch.long.
        labels[t] = input_ids[t+1] (causal LM objective).

    Raises:
        ImportError: if `datasets` or `transformers` are not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets not installed — run: uv pip install datasets"
        ) from exc

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers not installed — run: uv pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # StarCoder2 has eos_token but no bos_token; use eos as document separator.
    sep_token_id: int | None = None
    if tokenizer.eos_token_id is not None:
        sep_token_id = tokenizer.eos_token_id
    elif tokenizer.bos_token_id is not None:
        sep_token_id = tokenizer.bos_token_id

    import os
    import glob as _glob

    dataset_kwargs: dict = {"split": split, "streaming": True}
    # Validation split is not present in openwebtext — fall back gracefully.
    if split == "validation" and "openwebtext" in dataset_name.lower():
        dataset_kwargs["split"] = "train"

    # datasets>=4.x dropped script datasets (trust_remote_code gone), so 'Skylion007/openwebtext'
    # no longer loads. Two supported sources:
    #   (a) data.dataset = a glob/path of local *.arrow shards (e.g. the cached OWT) → codeless,
    #       OFFLINE, and IDENTICAL to the #186 data. This is the preferred path.
    #   (b) data.dataset = a codeless Parquet HF mirror (loads with no trust_remote_code).
    _expanded = sorted(_glob.glob(os.path.expanduser(dataset_name), recursive=True))
    is_local_arrow = bool(_expanded) and all(f.endswith(".arrow") for f in _expanded)
    if is_local_arrow:
        print(f"[data] local arrow source: {len(_expanded)} shards "
              f"(first={os.path.basename(_expanded[0])})")

    buf: list[int] = []
    # TST superposition phase needs s·(seq_len+1) raw tokens per sample; standard
    # NTP needs (seq_len+1). bag_size==0 → identical to the pre-TST behaviour.
    tokens_per_sample = (bag_size * (seq_len + 1)) if bag_size > 0 else (seq_len + 1)
    chunk_len = batch_size * tokens_per_sample
    tul_spec = None
    if tul is not None:
        if bag_size > 0:
            raise ValueError("tul and bag_size are mutually exclusive (spec invariant 6)")
        tul_spec = tul.spec_for(seq_len)
        # A TUL row spends at most L_total tokens and peeks one more for the last label;
        # the peek is left in the buffer to open the next row.
        chunk_len = batch_size * (tul_spec.l_total + 1)
        print(f"[data] TUL layout: seq_len={seq_len} prefix_k={tul_spec.prefix_k} "
              f"max_slots={tul_spec.max_slots} L_total={tul_spec.l_total} "
              f"slot_id={tul_spec.slot_id}", flush=True)

    while True:
        if is_local_arrow:
            ds = load_dataset("arrow", data_files=_expanded,
                              split=dataset_kwargs["split"], streaming=True)
        else:
            try:
                ds = load_dataset(dataset_name, **dataset_kwargs)
            except Exception as exc:  # informative, not the cryptic trust_remote_code error
                raise RuntimeError(
                    f"load_dataset({dataset_name!r}) failed. Under datasets>=4 only codeless "
                    f"(Parquet/Arrow) datasets load — script datasets like "
                    f"'Skylion007/openwebtext' are unsupported. Point cfg.data.dataset at a "
                    f"Parquet OWT mirror or a local *.arrow glob, or pin datasets<4. Orig: {exc}"
                ) from exc
        if skip_samples > 0:
            ds = ds.skip(skip_samples)
        for sample in ds:
            text = sample.get("text", sample.get("content", ""))
            ids = tokenizer(
                text,
                truncation=False,
                add_special_tokens=False,
            )["input_ids"]
            if sep_token_id is not None:
                ids = ids + [sep_token_id]
            buf.extend(ids)

            while len(buf) >= chunk_len:
                if tul_spec is not None:
                    # Consumes only what the rows actually use (token count varies per
                    # row) and leaves the remainder — including the label peek — in buf.
                    yield pack_tul_batch(buf, tul.rule, tul_spec, batch_size)
                    continue
                chunk = buf[:chunk_len]
                buf = buf[chunk_len:]
                if bag_size > 0:
                    s = bag_size
                    t = torch.tensor(chunk, dtype=torch.long).reshape(
                        batch_size, s * (seq_len + 1)
                    )
                    input_ids = t[:, : s * seq_len]                              # [B, s·L]
                    label_bags = t[:, s : s * (seq_len + 1)].reshape(
                        batch_size, seq_len, s
                    )                                                            # [B, L, s]
                    yield input_ids, label_bags
                else:
                    t = torch.tensor(chunk, dtype=torch.long).reshape(
                        batch_size, seq_len + 1
                    )
                    yield t[:, :seq_len], t[:, 1 : seq_len + 1]
