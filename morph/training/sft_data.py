"""Dolly-15k instruction-tuning data for MORPH SFT (response-masked, packed).

MORPH has NO attention_mask input — the pretraining pipeline packs contiguous
token streams separated by EOS. SFT mirrors that: each example is tokenized as
`prompt + response + EOS`, the PROMPT tokens are masked to -100 (loss only on the
response + EOS, so the model learns to answer AND to stop), and examples are
concatenated into a stream sliced into fixed seq_len chunks. Cross-example bleed
under causal+windowed attention is the standard "packed SFT" regime.

Both SFT arms (base_off, base_on) consume IDENTICAL data: same Dolly subset (seeded
shuffle), same per-epoch packing order (epoch-seeded), so the only variable is which
pretrained checkpoint is being fine-tuned.

A second, NORMAL-SFT data path lives alongside the packed one: `sft_batches_padded`
places exactly ONE example per row and right-pads each batch to its own max length
(see that function's docstring). It exists to test whether packing (cross-example
bleed) is what collapses free generation after SFT.

Usage:
    from morph.training.sft_data import build_dolly_examples, sft_batches
    exs = build_dolly_examples(tokenizer, n_examples=512, seed=1234, max_len=1024)
    # packed (default):
    for step, (ids, labels) in enumerate(sft_batches(exs, seq_len=1024,
                                                      batch_size=4, epochs=8, seed=1234)):
        ...
    # padded (one example per row, right-padded; for causal attn no mask is needed):
    for ids, labels, seq_lens in sft_batches_padded(exs, batch_size=8, epochs=3,
                                                     seed=1234, pad_id=pad_id):
        ...
"""
from __future__ import annotations

import random
from typing import Generator

import torch

IGNORE_INDEX = -100

# Alpaca-style instruction template (plain text; starcoder2 tokenizer has no chat tokens).
_PROMPT_CTX = (
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
)
_PROMPT_NOCTX = "### Instruction:\n{instruction}\n\n### Response:\n"


def _format_prompt(instruction: str, context: str) -> str:
    if context and context.strip():
        return _PROMPT_CTX.format(instruction=instruction.strip(), context=context.strip())
    return _PROMPT_NOCTX.format(instruction=instruction.strip())


def build_dolly_examples(
    tokenizer,
    n_examples: int,
    seed: int,
    max_len: int,
    dataset_name: str = "databricks/databricks-dolly-15k",
) -> list[tuple[list[int], list[int]]]:
    """Load + tokenize + response-mask a seeded Dolly subset.

    Returns a list of (input_ids, labels) per example (Python int lists). Labels equal
    input_ids on the response (+EOS) tokens and IGNORE_INDEX on the prompt tokens.
    Examples whose prompt+response exceeds `max_len` are dropped (so no example is
    silently truncated mid-response, which would teach the model to never stop).
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id — cannot teach the model to stop")

    examples: list[tuple[list[int], list[int]]] = []
    dropped = 0
    for i in idx:
        row = ds[i]
        prompt = _format_prompt(row.get("instruction", ""), row.get("context", ""))
        response = (row.get("response", "") or "").strip()
        if not response:
            continue
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        # Leading space so the BPE merges the response naturally after the template newline.
        c_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [eos]
        if len(p_ids) + len(c_ids) > max_len:
            dropped += 1
            continue
        ids = p_ids + c_ids
        # NEXT-TOKEN (PRE-SHIFTED) labels — the model/CE do NOT shift internally; they
        # expect targets aligned like the pretraining loader (`data.py`: labels[t] =
        # input_ids[t+1]). Supervise only positions whose TARGET is a response/eos token,
        # i.e. position t iff its next token ids[t+1] lies in the response region
        # (t+1 >= len(p_ids)). This makes the last PROMPT token learn to emit the first
        # response token, the pre-eos token learn to STOP, and the final position carry no
        # target. (The earlier `labels = [-100]*len(p_ids) + c_ids` was UNSHIFTED — it
        # trained the model to copy the CURRENT token → ppl→1.0 + degenerate single-token
        # looping at generation. Verified bug, fixed 2026-06-23.)
        n, p = len(ids), len(p_ids)
        labels = [IGNORE_INDEX] * n
        for t in range(n - 1):
            if t + 1 >= p:
                labels[t] = ids[t + 1]
        examples.append((ids, labels))
        if len(examples) >= n_examples:
            break

    if not examples:
        raise RuntimeError("no Dolly examples survived tokenization/length filter")
    n_resp = sum(sum(1 for t in lbl if t != IGNORE_INDEX) for _, lbl in examples)
    n_tot = sum(len(ids) for ids, _ in examples)
    print(f"[sft-data] {len(examples)} examples (dropped {dropped} > max_len={max_len}); "
          f"{n_tot} tokens total, {n_resp} trainable response tokens "
          f"({100*n_resp/n_tot:.1f}% of stream)", flush=True)
    return examples


# ──────────────────────────────────────────────────────────────────────────────
# Multiple-choice (ARC) SFT data — same SHIFTED-label contract as build_dolly_examples.
#
# Each ARC question is formatted as an instruction→answer SFT example. The Response is
# the FULL TEXT of the correct choice (not just the letter) + EOS: this gives the LM a
# richer training signal and matches the length-normalized LL eval (ignore/mcq_eval.py),
# which scores the LL of each choice's TEXT continuation. Labels are pre-shifted exactly
# like build_dolly_examples (response+eos supervised, prompt + final position masked).
# Uses the ARC TRAIN split only — the eval harness uses the disjoint TEST split (no leak).
# ──────────────────────────────────────────────────────────────────────────────

# Lettered-choice instruction template (Alpaca "### Instruction:/### Response:" style).
_PROMPT_MCQ = (
    "### Instruction:\n{question}\n\nChoices:\n{choices}\n\n### Response:\n"
)

# Map ARC's variable label keys to canonical letters. ARC uses either "A".."E" or
# "1".."5" as the answerKey / per-choice label; we normalize the answerKey to a letter
# and present choices as "A. <text>", "B. <text>", … by POSITION (so the displayed
# letters are always contiguous A,B,C,… regardless of the raw label keys).
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F", "7": "G"}


def _mcq_label_to_letter(label: str) -> str:
    """Normalize an ARC label key ('1'/'2'/… or 'A'/'B'/…) to an uppercase letter."""
    s = str(label).strip()
    return _NUM_TO_LETTER.get(s, s.upper())


def _format_mcq_prompt(question: str, choice_texts: list[str]) -> str:
    """Build the lettered-choice prompt. Choices are lettered by POSITION (A,B,C,…)."""
    letters = [chr(ord("A") + i) for i in range(len(choice_texts))]
    lines = "\n".join(f"{ltr}. {txt.strip()}" for ltr, txt in zip(letters, choice_texts))
    return _PROMPT_MCQ.format(question=question.strip(), choices=lines)


def build_mcq_examples(
    tokenizer,
    n_examples: int,
    seed: int,
    max_len: int,
    split: str = "train",
    dataset_configs: tuple[str, ...] = ("ARC-Easy", "ARC-Challenge"),
) -> list[tuple[list[int], list[int]]]:
    """Load + tokenize + response-mask a seeded ARC multiple-choice subset.

    Combines `allenai/ai2_arc` configs (ARC-Easy + ARC-Challenge) on the given `split`
    (TRAIN only for SFT — the eval uses the disjoint TEST split). Each question becomes:

        ### Instruction:
        {question}

        Choices:
        A. {choice0}
        B. {choice1}
        ...

        ### Response:
        {FULL TEXT of correct choice}<eos>

    Labels are NEXT-TOKEN (pre-shifted), identical construction to build_dolly_examples:
    position t is supervised iff its TARGET ids[t+1] lies in the response region
    (t+1 >= len(prompt)). The prompt is masked, the final position carries no target.
    Returns list[(input_ids, labels)] (same type as build_dolly_examples). Examples whose
    prompt+response exceed `max_len`, or whose correct choice can't be resolved, are dropped.
    """
    from datasets import load_dataset

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id — cannot teach the model to stop")

    # Combine the requested ARC configs into one pool, then seeded-shuffle the pool.
    pool: list[dict] = []
    for cfg_name in dataset_configs:
        ds = load_dataset("allenai/ai2_arc", cfg_name, split=split)
        pool.extend(list(ds))
    random.Random(seed).shuffle(pool)

    examples: list[tuple[list[int], list[int]]] = []
    dropped_len = 0
    dropped_bad = 0
    for row in pool:
        question = row.get("question", "")
        choices = row.get("choices", {}) or {}
        choice_texts = choices.get("text", []) or []
        choice_labels = choices.get("label", []) or []
        answer = row.get("answerKey", "")
        if not question or len(choice_texts) < 2 or not answer:
            dropped_bad += 1
            continue

        # Resolve the correct choice's INDEX by matching the normalized answerKey letter
        # against the normalized per-choice label keys. Robust to '1/2/3/4' vs 'A/B/C/D'.
        ans_letter = _mcq_label_to_letter(answer)
        norm_labels = [_mcq_label_to_letter(l) for l in choice_labels]
        if ans_letter in norm_labels:
            correct_idx = norm_labels.index(ans_letter)
        elif ans_letter in [chr(ord("A") + i) for i in range(len(choice_texts))]:
            # answerKey is a positional letter even though label keys differ.
            correct_idx = ord(ans_letter) - ord("A")
        else:
            dropped_bad += 1
            continue
        if correct_idx >= len(choice_texts):
            dropped_bad += 1
            continue

        prompt = _format_mcq_prompt(question, choice_texts)
        response = (choice_texts[correct_idx] or "").strip()
        if not response:
            dropped_bad += 1
            continue

        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tokenizer(response, add_special_tokens=False)["input_ids"] + [eos]
        if len(p_ids) + len(c_ids) > max_len:
            dropped_len += 1
            continue
        ids = p_ids + c_ids
        n, p = len(ids), len(p_ids)
        labels = [IGNORE_INDEX] * n
        for t in range(n - 1):
            if t + 1 >= p:
                labels[t] = ids[t + 1]
        examples.append((ids, labels))
        if len(examples) >= n_examples:
            break

    if not examples:
        raise RuntimeError("no ARC MCQ examples survived tokenization/length filter")
    n_resp = sum(sum(1 for t in lbl if t != IGNORE_INDEX) for _, lbl in examples)
    n_tot = sum(len(ids) for ids, _ in examples)
    print(f"[mcq-data] {len(examples)} ARC examples (split={split}, configs={list(dataset_configs)}; "
          f"dropped {dropped_len} > max_len={max_len}, {dropped_bad} unparseable); "
          f"{n_tot} tokens total, {n_resp} trainable response tokens "
          f"({100*n_resp/n_tot:.1f}% of stream)", flush=True)
    return examples


def count_steps(examples, seq_len: int, batch_size: int, epochs: int) -> int:
    """Total optimizer steps the generator below will yield (for the LR schedule)."""
    tok_per_epoch = sum(len(ids) for ids, _ in examples)
    nseq = tok_per_epoch // seq_len
    return (nseq // batch_size) * epochs


def sft_batches(
    examples,
    seq_len: int,
    batch_size: int,
    epochs: int,
    seed: int,
) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    """Yield (input_ids[B,T], labels[B,T]) packed batches over `epochs` passes.

    Each epoch re-shuffles example ORDER (epoch-seeded → deterministic + identical across
    arms), concatenates into one token stream, slices into seq_len chunks, and batches.
    Trailing tokens that don't fill a full (batch_size × seq_len) block are dropped.
    """
    for epoch in range(epochs):
        rng = random.Random(seed * 100003 + epoch)
        order = list(range(len(examples)))
        rng.shuffle(order)

        stream_ids: list[int] = []
        stream_lbl: list[int] = []
        for i in order:
            ids, lbl = examples[i]
            stream_ids.extend(ids)
            stream_lbl.extend(lbl)

        nseq = len(stream_ids) // seq_len
        nbatch = nseq // batch_size
        if nbatch == 0:
            raise RuntimeError(
                f"epoch {epoch}: only {nseq} seqs < batch_size {batch_size} — "
                f"lower seq_len/batch_size or raise n_examples"
            )
        ids_t = torch.tensor(stream_ids[: nseq * seq_len], dtype=torch.long).view(nseq, seq_len)
        lbl_t = torch.tensor(stream_lbl[: nseq * seq_len], dtype=torch.long).view(nseq, seq_len)
        for b in range(nbatch):
            sl = slice(b * batch_size, (b + 1) * batch_size)
            yield ids_t[sl], lbl_t[sl]


# ══════════════════════════════════════════════════════════════════════════════
# Padded / batched ("normal SFT") data path — ONE example per row, right-padded.
#
# The packed loader above bleeds examples together under causal+windowed attention
# (the standard packed-SFT regime). The PADDED loader places exactly ONE example per
# row and right-pads each batch to its own max length (dynamic padding). For CAUSAL
# attention with RIGHT-padding no attention_mask is required: real tokens sit at the
# FRONT, so they never attend FORWARD to pad positions; and the pad positions are
# excluded from the loss because their labels are IGNORE_INDEX (-100). The model also
# receives `seq_lens` (true non-pad length per row) so any length-aware sub-component
# restricts itself to the real tokens.
# ══════════════════════════════════════════════════════════════════════════════


def count_steps_padded(examples, batch_size: int, epochs: int) -> int:
    """Total optimizer steps the padded generator yields (for the LR schedule).

    One example per row, `batch_size` rows per step, trailing partial batch dropped
    (drop_last). Consistent with `sft_batches_padded` below.
    """
    per_epoch = len(examples) // batch_size
    return per_epoch * epochs


def sft_batches_padded(
    examples,
    batch_size: int,
    epochs: int,
    seed: int,
    pad_id: int,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """Yield (ids[B,Lmax], labels[B,Lmax], seq_lens[B]) — one example per row, right-padded.

    Normal-SFT policy (NOT packed): each row holds exactly one `(input_ids, labels)`
    example as produced by `build_dolly_examples` (labels already -100 on the prompt,
    real token ids on response+EOS). No re-packing, no cross-example concatenation.

    Per epoch: a seeded shuffle of example ORDER using the SAME convention as the packed
    loader (`random.Random(seed * 100003 + epoch)`) → deterministic and identical across
    arms. Examples are grouped into batches of `batch_size`; each batch is RIGHT-padded to
    the max example length IN THAT BATCH (dynamic padding):
        - input_ids  pad value = `pad_id`
        - labels     pad value = IGNORE_INDEX (-100)  → pads excluded from cross-entropy
    `seq_lens[r]` is the true non-pad token count of row r (long tensor). The trailing
    partial batch is dropped (drop_last) so step counts match `count_steps_padded`.

    pad_id: resolved by the caller as tokenizer.pad_token_id (else eos_token_id).
    """
    n = len(examples)
    for epoch in range(epochs):
        rng = random.Random(seed * 100003 + epoch)
        order = list(range(n))
        rng.shuffle(order)

        nbatch = n // batch_size  # drop_last
        if nbatch == 0:
            raise RuntimeError(
                f"epoch {epoch}: only {n} examples < batch_size {batch_size} — "
                f"lower batch_size or raise n_examples"
            )
        for b in range(nbatch):
            rows = order[b * batch_size : (b + 1) * batch_size]
            batch_ids = [examples[i][0] for i in rows]
            batch_lbl = [examples[i][1] for i in rows]
            lens = [len(x) for x in batch_ids]
            lmax = max(lens)

            ids_t = torch.full((batch_size, lmax), pad_id, dtype=torch.long)
            lbl_t = torch.full((batch_size, lmax), IGNORE_INDEX, dtype=torch.long)
            for r, (ids, lbl, ln) in enumerate(zip(batch_ids, batch_lbl, lens)):
                ids_t[r, :ln] = torch.tensor(ids, dtype=torch.long)
                lbl_t[r, :ln] = torch.tensor(lbl, dtype=torch.long)
            seq_lens = torch.tensor(lens, dtype=torch.long)
            yield ids_t, lbl_t, seq_lens
