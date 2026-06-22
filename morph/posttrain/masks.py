"""Token-label masking utilities for Morpheus SFT rows."""

from __future__ import annotations

from typing import Any

import torch

from morph.posttrain.validation import Span

IGNORE_INDEX = -100


def char_train_mask(text: str, target_spans: list[dict[str, Any] | tuple[int, int]]) -> list[bool]:
    """Return a per-character trainable mask from target spans."""
    mask = [False] * len(text)
    for raw in target_spans:
        span = Span.from_obj(raw)
        for i in range(span.start, span.end):
            mask[i] = True
    return mask


def build_loss_labels(
    text: str,
    tokenizer: Any,
    target_spans: list[dict[str, Any] | tuple[int, int]],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize `text` and mask labels outside target spans.

    The tokenizer must support `return_offsets_mapping=True`, which is available
    for Hugging Face fast tokenizers. Labels are equal to input ids for trainable
    tokens and `ignore_index` for context/runtime tokens.
    """
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors=None,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise TypeError("tokenizer must return offset_mapping for loss-mask construction")

    c_mask = char_train_mask(text, target_spans)
    labels = []
    for token_id, (start, end) in zip(input_ids, offsets):
        if end <= start:
            labels.append(ignore_index)
            continue
        trainable = any(c_mask[start:end])
        labels.append(int(token_id) if trainable else ignore_index)

    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
