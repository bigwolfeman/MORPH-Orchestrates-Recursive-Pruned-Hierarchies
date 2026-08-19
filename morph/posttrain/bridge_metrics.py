"""Bridge metrics — the ONLY legal cross-family comparison for DiffusionBlocks arms.

Plan item A7. Contract: ``docs/diffusionblocks-experiment-sheet.md`` §1.3.

Why this exists. A DB arm's CE is conditioned on a σ-noised target, so it is a
reconstruction number, not a likelihood (paper App. E.4: *"computing traditional perplexity
is non-trivial"*). Putting it next to A0's ``val/ppl_tokens`` would be comparing two
different quantities. The bridge is generation-side and applies identically to both
families:

* **gen-PPL(teacher)** — generate continuations, score them with a frozen teacher. Lower is
  more fluent. The paper's own protocol (App. E.4) uses Llama-2-7B and GPT2-XL.
* **rep4@512** — fraction of repeated 4-grams. Catches the degenerate-repetition failure
  that gen-PPL alone rewards.
* **distinct-2 / self-BLEU-free diversity** — a cheap stand-in for MAUVE, which needs a
  large reference sample and an embedding model; MAUVE proper is left for later and is NOT
  claimed here.

Run POST-HOC in a separate process, never inside the training job — the teacher wants its
own VRAM and the training loop must not pay for it.

**Honest scope.** gen-PPL under a teacher is a fluency proxy, not a likelihood, and it is
gameable by low-entropy text (which is exactly why rep4 is reported beside it). Two arms
are comparable here only if they generated with the SAME decoding settings and the SAME
prompt set; the runner enforces that by taking both from one config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

__all__ = ["BridgeConfig", "rep_ngram_fraction", "distinct_n", "teacher_gen_ppl",
           "bridge_report"]


@dataclass
class BridgeConfig:
    """Decoding + scoring settings. Identical across arms or the comparison is void."""

    teacher: str = "openai-community/gpt2-xl"   # the paper's second teacher
    n_prompts: int = 256
    prompt_tokens: int = 32
    gen_tokens: int = 50                        # App. E.4 uses 50
    top_p: float = 0.95                         # baseline decoding (App. E.4)
    db_steps: int = 8                           # Euler steps for a DB arm (their AR used 4)
    seed: int = 0
    batch: int = 16
    rep_n: int = 4
    rep_window: int = 512
    extra: dict = field(default_factory=dict)


def rep_ngram_fraction(ids: torch.Tensor, n: int = 4, window: int = 512) -> float:
    """Fraction of n-grams in each row that are REPEATS, averaged over rows.

    0.0 = every n-gram unique. High values mean degenerate looping, the failure mode a
    fluency score alone rewards.
    """
    if ids.dim() != 2:
        raise ValueError(f"expected [B, L], got {tuple(ids.shape)}")
    out = []
    for row in ids[:, :window].tolist():
        if len(row) < n:
            continue
        grams = [tuple(row[i:i + n]) for i in range(len(row) - n + 1)]
        if not grams:
            continue
        out.append(1.0 - len(set(grams)) / len(grams))
    return sum(out) / max(len(out), 1)


def distinct_n(ids: torch.Tensor, n: int = 2) -> float:
    """Distinct n-grams over total n-grams, pooled across rows. Higher = more diverse."""
    seen, total = set(), 0
    for row in ids.tolist():
        for i in range(len(row) - n + 1):
            seen.add(tuple(row[i:i + n]))
            total += 1
    return len(seen) / max(total, 1)


@torch.no_grad()
def teacher_gen_ppl(text_ids: torch.Tensor, teacher, tok_teacher, tokenizer_src,
                    device="cuda", batch: int = 8) -> float:
    """Perplexity of generated text under a frozen teacher.

    The generated ids come from OUR tokenizer, so they are decoded to text and re-encoded
    with the teacher's. That round-trip is lossy and it is the reason this is a *proxy*:
    two arms are comparable to each other, but the absolute value is not a likelihood.
    """
    texts = [tokenizer_src.decode(r, skip_special_tokens=True) for r in text_ids.tolist()]
    texts = [t for t in texts if t.strip()]
    if not texts:
        return float("nan")
    total_nll, total_tok = 0.0, 0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok_teacher(chunk, return_tensors="pt", padding=True, truncation=True,
                          max_length=256).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        out = teacher(**enc, labels=labels)
        n_tok = int((labels != -100).sum())
        # HF returns the MEAN nll over valid tokens; re-weight to pool correctly.
        total_nll += float(out.loss) * n_tok
        total_tok += n_tok
    return math.exp(min(total_nll / max(total_tok, 1), 20.0))


def bridge_report(gen_ids: torch.Tensor, cfg: BridgeConfig, tokenizer_src,
                  teacher=None, tok_teacher=None, device="cuda") -> dict:
    """Assemble the bridge row for ONE arm.

    ``teacher=None`` skips gen-PPL and returns the tokenizer-free metrics only, so the
    harness is usable on a machine without the teacher downloaded — with the omission
    explicit rather than silently reported as 0.
    """
    rep = {
        f"bridge/rep{cfg.rep_n}@{cfg.rep_window}": rep_ngram_fraction(
            gen_ids, cfg.rep_n, cfg.rep_window),
        "bridge/distinct2": distinct_n(gen_ids, 2),
        "bridge/distinct3": distinct_n(gen_ids, 3),
        "bridge/n_sequences": int(gen_ids.shape[0]),
        "bridge/gen_tokens": int(gen_ids.shape[1]),
        "bridge/decode_top_p": cfg.top_p,
        "bridge/db_steps": cfg.db_steps,
        "bridge/teacher": cfg.teacher if teacher is not None else "SKIPPED",
    }
    if teacher is not None:
        rep["bridge/gen_ppl_teacher"] = teacher_gen_ppl(
            gen_ids, teacher, tok_teacher, tokenizer_src, device=device, batch=cfg.batch)
    return rep
