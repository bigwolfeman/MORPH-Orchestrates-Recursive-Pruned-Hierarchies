"""Punctuation-based step-boundary mask for pretraining-STP.

arXiv:2604.18464 uses reasoning-step boundaries during SFT; this module derives
equivalent boundaries from punctuation in general pretraining text (bag_size=0 / TST off),
so the forecastability mechanism applies without <|step|> tokens.

Design notes
------------
BPE wrinkle: punctuation characters almost never tokenize as standalone codepoints in
StarCoder2 / GPT-style BPEs.  E.g. "—" (U+2014) typically merges with adjacent whitespace
or a preceding letter.  To capture these:

  1. For each target string, encode it in several SYNTHETIC CONTEXTS (e.g. " — ", "a—b",
     "end. Next") and collect every token-id whose decoded form CONTAINS the target char.
  2. Build an id→label dict from the union — so a merged token like " —" or ".\n" is
     captured even though it spans the target and something else.

Because we only care about POSITION (not exact token), false-positives (tokens that happen
to decode to a string containing a period, say, because of an adjacent contraction) are
rare and harmless: the dedup pass eliminates adjacent boundaries anyway.

⚠️⚠️⚠️  KNOWN-INADEQUATE BOUNDARY DETECTION — TESTING-ONLY HEURISTIC  ⚠️⚠️⚠️
=============================================================================
This token-id-membership scheme is a CRUDE STAND-IN for sentence segmentation. Fine for a
clean-prose smoke/ablation (WikiText/OWT, mostly "X. Word"), but WRONG in general and MUST be
replaced with a real text-level sentence segmenter before any result is trusted or shipped.
The robust problem is genuinely hard — it must handle ALL terminators AND ALL their
COMBINATIONS, including (non-exhaustive):
  - closing-punctuation CLUSTERS that BPE-fuse the terminator into one token (so the period id
    is never even seen):  ."  .'  .)  .”  ?"  !)  ."'  ...)   → boundary MISSED entirely.
  - non-terminal periods: abbreviations (Dr. Mr. U.S. etc. i.e. e.g. Inc. vs.),
    decimals/versions ($1.50  3.14  v2.0), URLs/emails/code (example.com self.x a.b()).
    → FALSE-POSITIVE mid-sentence split.
  - ellipses "..." (one boundary? none? three?), multi-terminators "?!", "!?".
  - terminator followed by NEWLINE or MULTI-SPACE → the "+1 = next-clause start" target used by
    LatentForecast (model/prediction.py) lands on a '\n'/' ' token, NOT the next word.
The dedup pass does NOT fix these — it only merges ADJACENT detected boundaries.
PROPER FIX (deferred): run a real sentence segmenter (spaCy/pySBD/punkt) on the DECODED text →
char spans → map back to token indices → store an explicit boundary + next-content-start INDEX
MAP at data-prep (zero runtime guessing). See Ai-notes/06-25-2026 boundary note + CLAUDE.md.
Until then: clean single-space prose only; treat boundary metrics as NOISY.

Public API
----------
resolve_punct_token_ids(tokenizer, include_comma, include_newline) -> dict[int, str]
punctuation_boundary_mask(input_ids, punct_ids, min_gap) -> torch.BoolTensor [B, T]
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor


# ── 1. Resolve punctuation token ids ─────────────────────────────────────────

def resolve_punct_token_ids(
    tokenizer,
    include_comma: bool = False,
    include_newline: bool = True,
) -> Dict[int, str]:
    """Return {token_id: label} for punctuation boundary tokens.

    Targets and their labels
    ------------------------
    Always included:
        "."   sentence terminator
        "!"   exclamation
        "?"   question
        ";"   clause separator
        "—"   em dash (U+2014)  ← CRITICAL
        "–"   en dash (U+2013)
        "--"  ASCII em-dash stand-in
    include_newline=True (default):
        "\n"  line break
    include_comma=True:
        ","   comma (off by default — too frequent, noisy boundaries)

    BPE strategy
    ------------
    For each target string t, encode t embedded in several synthetic contexts.
    Collect all token ids whose DECODED string contains t (or a canonical
    alias — e.g. "--" is also caught by the em/en-dash id set for robustness).
    The id→label map stores the MOST SPECIFIC label (earliest match wins if a
    token decodes to multiple targets; order of `targets` list = priority).

    Prints a diagnostic line for the em-dash set so the caller can verify it.
    """
    targets: list[tuple[str, str]] = [
        ("—", "em_dash"),           # U+2014, highest priority
        ("–", "en_dash"),           # U+2013
        ("--", "double_hyphen"),    # ASCII fallback (merges → often same ids as em/en dash)
        (".", "period"),
        ("!", "exclamation"),
        ("?", "question"),
        (";", "semicolon"),
    ]
    if include_newline:
        targets.append(("\n", "newline"))
    if include_comma:
        targets.append((",", "comma"))

    # Synthetic contexts — each char encoded in varied left/right contexts so BPE sees
    # different merge opportunities and reveals all the ids that carry the target.
    def _contexts(ch: str) -> list[str]:
        stripped = ch.strip()
        return [
            f"a{ch}b",
            f" {ch} ",
            f"end{ch} Next",
            f"word{ch}word",
            f"{ch}",
            f" {ch}",
            f"{ch} ",
            # extra for dashes
            f"long{ch}dash",
            f"sentence{ch}\n",
        ]

    id_to_label: dict[int, str] = {}

    for target_str, label in targets:
        found_ids: set[int] = set()
        for ctx in _contexts(target_str):
            try:
                enc = tokenizer(ctx, add_special_tokens=False)["input_ids"]
            except Exception:
                continue
            for tid in enc:
                try:
                    decoded = tokenizer.decode([tid])
                except Exception:
                    continue
                # A token "captures" the target if its decoded form contains the target
                # character (or any of its aliases for the --/em-dash case).
                if target_str in decoded:
                    found_ids.add(tid)
                # Extra: "--" tokens also capture em/en dash equivalents
                if target_str == "--" and ("—" in decoded or "–" in decoded):
                    found_ids.add(tid)

        # Register ids not already claimed by a higher-priority target
        for tid in found_ids:
            if tid not in id_to_label:
                id_to_label[tid] = label

    # ── Diagnostic for em dash (CRITICAL — must not be empty) ────────────────
    em_ids = {tid: lbl for tid, lbl in id_to_label.items() if lbl == "em_dash"}
    en_ids = {tid: lbl for tid, lbl in id_to_label.items() if lbl == "en_dash"}
    dh_ids = {tid: lbl for tid, lbl in id_to_label.items() if lbl == "double_hyphen"}
    print(f"[punct_boundary] em_dash  ids ({len(em_ids)}): "
          + ", ".join(f"{tid}→{repr(tokenizer.decode([tid]))}" for tid in sorted(em_ids)))
    print(f"[punct_boundary] en_dash  ids ({len(en_ids)}): "
          + ", ".join(f"{tid}→{repr(tokenizer.decode([tid]))}" for tid in sorted(en_ids)))
    print(f"[punct_boundary] dbl_hyph ids ({len(dh_ids)}): "
          + ", ".join(f"{tid}→{repr(tokenizer.decode([tid]))}" for tid in sorted(dh_ids)))
    print(f"[punct_boundary] full map ({len(id_to_label)} ids): "
          + ", ".join(f"{tid}({lbl})" for tid, lbl in sorted(id_to_label.items())))

    return id_to_label


# ── 2. Compute the boundary mask ─────────────────────────────────────────────

def punctuation_boundary_mask(
    input_ids: Tensor,
    punct_ids: Dict[int, str] | Sequence[int],
    min_gap: int = 2,
) -> Tensor:
    """Return a [B, T] bool mask with True at punctuation boundary positions.

    Deduplication
    -------------
    Adjacent boundary tokens within `min_gap` tokens of each other form a
    *cluster*.  Within each cluster only the LAST position is kept.  This
    prevents two boundaries separated by just one or two tokens (e.g. a period
    immediately followed by a newline) from producing a near-zero latent
    increment that corrupts the STP cosine signal.

    Parameters
    ----------
    input_ids : LongTensor [B, T]
    punct_ids : dict {id: label} or iterable of ids
    min_gap   : merge adjacent marks ≤ min_gap apart; keep the LAST one.
                min_gap=2 means a mark at t and t+1 (or t+2) → keep t+1 (t+2).

    Returns
    -------
    BoolTensor [B, T]  (same device as input_ids, no GPU ops required)
    """
    if isinstance(punct_ids, dict):
        id_set = set(punct_ids.keys())
    else:
        id_set = set(int(i) for i in punct_ids)

    B, T = input_ids.shape
    device = input_ids.device

    # Build the raw hit mask: True wherever token id is in id_set.
    # Use a lookup tensor for vectorised membership.  Vocab may be large
    # (49 152 for StarCoder2) but this is a 1-D integer lookup, trivially fast.
    max_id = input_ids.max().item() if T > 0 else 0
    vocab_bound = int(max_id) + 1
    # Build id_set clamped to present ids only (avoid large alloc)
    present_ids = [i for i in id_set if i < vocab_bound]

    if not present_ids:
        return torch.zeros(B, T, dtype=torch.bool, device=device)

    lookup = torch.zeros(vocab_bound, dtype=torch.bool, device=device)
    lookup[torch.tensor(present_ids, dtype=torch.long, device=device)] = True

    # Clamp input ids to lookup size (any id ≥ vocab_bound is not punctuation)
    clamped = input_ids.clamp(0, vocab_bound - 1)
    raw_mask = lookup[clamped]  # [B, T]

    if min_gap <= 0:
        return raw_mask

    # ── Dedup: within each cluster of marked positions separated by ≤ min_gap,
    # keep ONLY the last one. ──────────────────────────────────────────────────
    # Algorithm (vectorised, CPU friendly):
    #   A position p is the LAST in its cluster if:
    #     raw_mask[p] == True
    #   AND for all q in [p+1, p+min_gap]: raw_mask[q] == False
    #   (i.e. no subsequent mark within the gap window).
    #
    # Equivalently: raw_mask[p] AND NOT any(raw_mask[p+1 .. p+min_gap]).
    # Implement with a running max (OR) in a small forward window.

    # Pad T axis on the right by min_gap zeros so the window is always valid
    # shape: [B, T + min_gap]
    padded = torch.cat(
        [raw_mask, torch.zeros(B, min_gap, dtype=torch.bool, device=device)],
        dim=1,
    )  # [B, T + min_gap]

    # For each position t, compute OR over [t+1, t+min_gap]
    # i.e. "does ANY of the next min_gap positions also have a mark?"
    any_next = torch.zeros(B, T, dtype=torch.bool, device=device)
    for offset in range(1, min_gap + 1):
        any_next |= padded[:, offset : offset + T]

    # Keep raw_mask position only if no subsequent mark within the gap
    deduped = raw_mask & ~any_next  # [B, T]
    return deduped
