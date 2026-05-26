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

__all__ = ["create_dataloader"]


def create_dataloader(
    tokenizer_name: str,
    dataset_name: str,
    seq_len: int,
    batch_size: int,
    split: str = "train",
    skip_samples: int = 0,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Infinite generator of (input_ids, labels) pairs.

    Streams the dataset, accumulates tokens in a buffer, and yields
    contiguous (seq_len+1)-token chunks sliced into inputs + labels.

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

    dataset_kwargs: dict = {"split": split, "streaming": True}
    # Validation split is not present in openwebtext — fall back gracefully.
    if split == "validation" and "openwebtext" in dataset_name.lower():
        dataset_kwargs["split"] = "train"

    buf: list[int] = []
    chunk_len = batch_size * (seq_len + 1)

    while True:
        ds = load_dataset(dataset_name, **dataset_kwargs, trust_remote_code=True)
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
                chunk = buf[:chunk_len]
                buf = buf[chunk_len:]
                t = torch.tensor(chunk, dtype=torch.long).reshape(
                    batch_size, seq_len + 1
                )
                yield t[:, :seq_len], t[:, 1 : seq_len + 1]
