"""Step-delimited reasoning data pipeline for MORPH Semantic Step Prediction SFT.

Implements arXiv:2604.18464 (Semantic Step Prediction): we SFT on reasoning traces and
at inference time read hidden states at REASONING-STEP boundaries (the last token of each
completed step), whose representations are expected to predict the NEXT step as a latent.

The shared interface contract with the model side is:
    (input_ids: list[int], labels: list[int], boundary_positions: list[int])

where boundary_positions are absolute indices into input_ids, each pointing to the LAST
token of a completed reasoning step.  Every boundary_position must satisfy
labels[pos] != -100 (i.e. inside the supervised response region).

LABEL CONTRACT: identical to build_dolly_examples (sft_data.py):
    labels[t] = ids[t+1]   if t+1 >= len(prompt_ids)   [NEXT-TOKEN, pre-shifted]
    labels[t] = IGNORE_INDEX (-100) otherwise
    labels[-1] = IGNORE_INDEX always (final position has no next token to supervise)

STEP SPLITTING:
    GSM8K: The answer field is already newline-separated calculation steps with the final
           line "#### <answer>".  We split on '\\n' and each non-empty line is a step.

    MATH (EleutherAI/hendrycks_math, all 7 configs combined):
        Hierarchical splitter — tries in order:
            1. '\\n\\n' paragraph splits (≥3 steps)  → richest semantic boundaries
            2. '\\n' line splits (≥3 steps)           → sentence-per-line LaTeX style
            3. '. ' / '! ' / '? ' sentence splits     → last resort for single-paragraph sols
        In practice: ~10% para, ~30% newline, ~60% sentence (measured on 200-row sample;
        76% of rows reach ≥3 steps before the <3-boundary drop filter).

    OLYMPIAD (GeneratorRegistry, all 225 generators):
        Steps come directly from Problem.steps (native ReasoningStep tuples — no text
        splitting needed).  Each ReasoningStep.content is one step.  Examples with <3
        steps are dropped (same as GSM8K/MATH).  {{SLOT}} markers in question_template
        are stripped (we don't spin — we want plain text for the prompt).

Usage:
    from morph.training.reasoning_data import (
        build_gsm8k_examples, build_math_examples, build_olympiad_examples,
        reasoning_batches_padded, count_reasoning_steps,
    )
    examples = build_gsm8k_examples(tokenizer, n_examples=512, seed=42, max_len=1024)
    oly_examples = build_olympiad_examples(tokenizer, n_examples=512, seed=42, max_len=1024)
    for ids, labels, bm, slens in reasoning_batches_padded(examples, batch_size=8,
                                                            epochs=3, seed=42,
                                                            pad_id=tok.eos_token_id):
        ...
"""
from __future__ import annotations

import re
import random
from typing import Generator

import torch

IGNORE_INDEX = -100

# ── Prompt template (mirrors sft_data.py Alpaca style; no chat tokens for starcoder2) ──
_PROMPT_NOCTX = "### Instruction:\n{instruction}\n\n### Response:\n"


def _format_prompt(question: str) -> str:
    return _PROMPT_NOCTX.format(instruction=question.strip())


# ── GSM8K step splitting ─────────────────────────────────────────────────────────────────

def _split_gsm8k_steps(answer: str) -> list[str]:
    """Split a GSM8K answer into step strings.

    The answer field is newline-separated calc steps with a final '#### <num>' line.
    We preserve all non-empty lines (including the #### line as the final step).
    """
    return [line.strip() for line in answer.split("\n") if line.strip()]


# ── MATH step splitting ──────────────────────────────────────────────────────────────────

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_math_steps(solution: str) -> list[str]:
    """Hierarchical step splitter for MATH/competition solutions.

    Tries in order:
      1. '\\n\\n' paragraph boundaries → captures distinct proof paragraphs
      2. '\\n' line boundaries         → captures sentence-per-line LaTeX
      3. '. '/'! '/'? ' sentence splits → last resort for single-paragraph solutions

    Returns the first split that yields ≥3 non-empty chunks.  If none does, returns
    whatever the deepest split gives (will be dropped later by the ≥3 boundary filter).
    """
    for sep in ("\n\n", "\n"):
        steps = [s.strip() for s in solution.split(sep) if s.strip()]
        if len(steps) >= 3:
            return steps
    # sentence split
    steps = [s.strip() for s in _SENT_SPLIT_RE.split(solution) if s.strip()]
    return steps


# ── Core tokenization helper ─────────────────────────────────────────────────────────────

def _build_example(
    tokenizer,
    prompt: str,
    steps: list[str],
    max_len: int,
    eos: int,
) -> tuple[list[int], list[int], list[int]] | None:
    """Tokenize prompt + steps → (input_ids, labels, boundary_positions) or None if invalid.

    Steps are tokenized sequentially with a leading '\\n' continuation prefix so that BPE
    merges at step boundaries mirror what the model would see in free generation.  The
    boundary_position for each step is the absolute index of its LAST token in input_ids.

    Returns None if:
      - total length > max_len
      - fewer than 3 boundaries
      - any boundary_position fails the labels[pos] != -100 check
    """
    p_ids: list[int] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    p_len = len(p_ids)

    # Tokenize each step with a leading newline so step concatenation preserves the
    # original whitespace contract (the prompt ends with "### Response:\n", so the
    # first step continues from there; subsequent steps are separated by newlines).
    step_token_seqs: list[list[int]] = []
    for i, step in enumerate(steps):
        prefix = "\n" if i > 0 else ""
        tok_out = tokenizer(prefix + step, add_special_tokens=False)["input_ids"]
        step_token_seqs.append(tok_out)

    # Build full ids: prompt + all step tokens + EOS
    response_ids: list[int] = []
    for seq in step_token_seqs:
        response_ids.extend(seq)
    response_ids.append(eos)

    ids = p_ids + response_ids
    total_len = len(ids)
    if total_len > max_len:
        return None

    # Boundary = absolute index of last token of each step (before EOS)
    boundary_positions: list[int] = []
    cursor = p_len  # absolute index into ids where response starts
    for seq in step_token_seqs:
        cursor += len(seq)
        # last token of this step is at cursor-1
        boundary_positions.append(cursor - 1)

    # Pre-shifted labels (identical convention to build_dolly_examples):
    #   labels[t] = ids[t+1]  iff t+1 >= p_len (i.e. target is in the response region)
    #   labels[t] = IGNORE_INDEX otherwise
    #   labels[-1] = IGNORE_INDEX always (no next token at the end)
    labels = [IGNORE_INDEX] * total_len
    for t in range(total_len - 1):
        if t + 1 >= p_len:
            labels[t] = ids[t + 1]

    # Validate: every boundary must be inside the supervised region
    if len(boundary_positions) < 3:
        return None
    for bp in boundary_positions:
        if bp < 0 or bp >= total_len:
            return None
        if labels[bp] == IGNORE_INDEX:
            # boundary is either in the prompt region or at the very last position
            return None

    return ids, labels, boundary_positions


# ── Public builders ──────────────────────────────────────────────────────────────────────

def build_gsm8k_examples(
    tokenizer,
    n_examples: int,
    seed: int,
    max_len: int,
    split: str = "train",
) -> list[tuple[list[int], list[int], list[int]]]:
    """Load + tokenize + step-boundary-annotate a seeded GSM8K subset.

    Dataset: openai/gsm8k, config 'main'.  Each row: question (problem) + answer
    (newline-separated calculation steps ending with '#### <final>').

    Returns list of (input_ids, labels, boundary_positions):
      input_ids  — flat int list, prompt + step tokens + EOS
      labels     — IGNORE_INDEX on prompt, NEXT-TOKEN pre-shifted on response+EOS
                   (labels[t]=ids[t+1] for t+1>=len(prompt)), final pos -100
      boundary_positions — absolute indices of the LAST token of each step,
                           all within the supervised region (labels[pos]!=-100)
    Dropped: examples > max_len, examples with <3 step boundaries.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")

    examples: list[tuple[list[int], list[int], list[int]]] = []
    dropped_len = 0
    dropped_steps = 0

    for i in idx:
        row = ds[i]
        question = row.get("question", "").strip()
        answer = row.get("answer", "").strip()
        if not question or not answer:
            continue

        steps = _split_gsm8k_steps(answer)
        if len(steps) < 3:
            dropped_steps += 1
            continue

        prompt = _format_prompt(question)
        result = _build_example(tokenizer, prompt, steps, max_len, eos)
        if result is None:
            # Could be too long OR <3 boundaries post-tokenization
            # Distinguish by re-checking length
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            resp_check = tokenizer("\n".join(steps), add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(resp_check) + 1 > max_len:
                dropped_len += 1
            else:
                dropped_steps += 1
            continue

        examples.append(result)
        if len(examples) >= n_examples:
            break

    if not examples:
        raise RuntimeError("no GSM8K examples survived tokenization/length/step filter")

    n_resp = sum(sum(1 for t in lbl if t != IGNORE_INDEX) for _, lbl, _ in examples)
    n_tot = sum(len(ids) for ids, _, _ in examples)
    n_bounds = sum(len(bp) for _, _, bp in examples)
    print(
        f"[gsm8k-data] {len(examples)} examples "
        f"(dropped {dropped_len} > max_len={max_len}, {dropped_steps} <3 steps); "
        f"{n_tot} tokens total, {n_resp} trainable response tokens "
        f"({100 * n_resp / n_tot:.1f}%), {n_bounds} total boundaries "
        f"(avg {n_bounds / len(examples):.1f}/example)",
        flush=True,
    )
    return examples


_MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Canonical dataset IDs to try in order
_MATH_HF_IDS = [
    ("EleutherAI/hendrycks_math", _MATH_CONFIGS),   # multi-config; fields: problem, solution
    ("hendrycks/competition_math", None),             # single-config fallback; same fields
    ("lighteval/MATH", None),                         # single-config fallback
]


def _load_math_pool(split: str) -> list[dict]:
    """Load all MATH rows from whichever HuggingFace dataset is available.

    Tries EleutherAI/hendrycks_math (all 7 configs combined) first, then single-config
    fallbacks.  Returns a flat list of dicts with 'problem' and 'solution' keys.
    Raises RuntimeError if all sources fail.
    """
    from datasets import load_dataset

    last_err: Exception | None = None
    for hf_id, configs in _MATH_HF_IDS:
        try:
            if configs is not None:
                rows: list[dict] = []
                for cfg in configs:
                    ds = load_dataset(hf_id, cfg, split=split)
                    rows.extend(list(ds))
                print(f"[math-data] loaded {len(rows)} rows from {hf_id} ({len(configs)} configs)", flush=True)
                return rows
            else:
                ds = load_dataset(hf_id, split=split)
                rows = list(ds)
                print(f"[math-data] loaded {len(rows)} rows from {hf_id}", flush=True)
                return rows
        except Exception as e:
            last_err = e
            print(f"[math-data] {hf_id} unavailable: {e}", flush=True)
            continue

    raise RuntimeError(
        f"Could not load any MATH dataset (tried: {[h for h,_ in _MATH_HF_IDS]}). "
        f"Last error: {last_err}"
    )


def build_math_examples(
    tokenizer,
    n_examples: int,
    seed: int,
    max_len: int,
    split: str = "train",
) -> list[tuple[list[int], list[int], list[int]]]:
    """Load + tokenize + step-boundary-annotate a seeded MATH subset.

    Dataset: EleutherAI/hendrycks_math (all 7 configs) → fallback hendrycks/competition_math
    → fallback lighteval/MATH.  Rows: problem, solution (LaTeX multi-step, \\boxed{} answer).

    Step splitting: hierarchical (\\n\\n → \\n → sentence; see _split_math_steps).
    Same return type and label contract as build_gsm8k_examples.
    """
    pool = _load_math_pool(split)
    random.Random(seed).shuffle(pool)

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")

    examples: list[tuple[list[int], list[int], list[int]]] = []
    dropped_len = 0
    dropped_steps = 0

    for row in pool:
        problem = row.get("problem", "").strip()
        solution = row.get("solution", "").strip()
        if not problem or not solution:
            continue

        steps = _split_math_steps(solution)
        if len(steps) < 3:
            dropped_steps += 1
            continue

        prompt = _format_prompt(problem)
        result = _build_example(tokenizer, prompt, steps, max_len, eos)
        if result is None:
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            resp_check = tokenizer("\n".join(steps), add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(resp_check) + 1 > max_len:
                dropped_len += 1
            else:
                dropped_steps += 1
            continue

        examples.append(result)
        if len(examples) >= n_examples:
            break

    if not examples:
        raise RuntimeError("no MATH examples survived tokenization/length/step filter")

    n_resp = sum(sum(1 for t in lbl if t != IGNORE_INDEX) for _, lbl, _ in examples)
    n_tot = sum(len(ids) for ids, _, _ in examples)
    n_bounds = sum(len(bp) for _, _, bp in examples)
    print(
        f"[math-data] {len(examples)} examples "
        f"(dropped {dropped_len} > max_len={max_len}, {dropped_steps} <3 steps); "
        f"{n_tot} tokens total, {n_resp} trainable response tokens "
        f"({100 * n_resp / n_tot:.1f}%), {n_bounds} total boundaries "
        f"(avg {n_bounds / len(examples):.1f}/example)",
        flush=True,
    )
    return examples


# ── Padded batch generator ────────────────────────────────────────────────────────────────

def count_reasoning_steps(
    examples: list[tuple[list[int], list[int], list[int]]],
    batch_size: int,
    epochs: int,
) -> int:
    """Total optimizer steps reasoning_batches_padded will yield.

    One example per row, batch_size rows per step, trailing partial batch dropped
    (drop_last). Consistent with count_steps_padded in sft_data.py.
    """
    per_epoch = len(examples) // batch_size
    return per_epoch * epochs


def reasoning_batches_padded(
    examples: list[tuple[list[int], list[int], list[int]]],
    batch_size: int,
    epochs: int,
    seed: int,
    pad_id: int,
) -> Generator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """Yield padded batches over `epochs` passes of the reasoning examples.

    Returns:
        ids[B, Lmax]           long  — input token ids, right-padded with pad_id
        labels[B, Lmax]        long  — pre-shifted next-token targets, -100 on prompt+pad
        seq_lens[B]            long  — true non-pad length per row
        boundary_mask[B, Lmax] bool  — True at each boundary_position, False elsewhere/pad

    Mirrors sft_batches_padded conventions EXACTLY:
      - One example per row (no packing / cross-example concatenation)
      - Right-padded to max example length in each batch (dynamic padding)
      - Per-epoch seeded shuffle: random.Random(seed * 100003 + epoch)
      - drop_last: trailing partial batch dropped
      - ids pad = pad_id;  labels pad = IGNORE_INDEX (-100)

    pad_id: pass tokenizer.pad_token_id if set, else tokenizer.eos_token_id.
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
            batch_bp = [examples[i][2] for i in rows]
            lens = [len(x) for x in batch_ids]
            lmax = max(lens)

            ids_t = torch.full((batch_size, lmax), pad_id, dtype=torch.long)
            lbl_t = torch.full((batch_size, lmax), IGNORE_INDEX, dtype=torch.long)
            bm_t = torch.zeros((batch_size, lmax), dtype=torch.bool)

            for r, (ids, lbl, bp, ln) in enumerate(
                zip(batch_ids, batch_lbl, batch_bp, lens)
            ):
                ids_t[r, :ln] = torch.tensor(ids, dtype=torch.long)
                lbl_t[r, :ln] = torch.tensor(lbl, dtype=torch.long)
                for pos in bp:
                    bm_t[r, pos] = True

            seq_lens = torch.tensor(lens, dtype=torch.long)
            yield ids_t, lbl_t, seq_lens, bm_t


# ── Olympiad slot-stripping ──────────────────────────────────────────────────────────────

# Matches {{ANYTHING}} markers left in question_template after generation (no spinning)
_SLOT_RE = re.compile(r"\{\{[^}]+\}\}")


def _strip_slots(template: str) -> str:
    """Remove {{SLOT}} markers from a question_template, collapse extra whitespace."""
    text = _SLOT_RE.sub("", template)
    # Collapse multiple spaces that may have been created by stripping a leading slot
    text = re.sub(r"  +", " ", text).strip()
    return text


# ── Olympiad builder ─────────────────────────────────────────────────────────────────────

_OLYMPIAD_SRC = "/mnt/BigAssDrive/00projects/00DeepNet/111TitanMAC-Standalone/Olympiad-AI/src"


def build_olympiad_examples(
    tokenizer,
    n_examples: int,
    seed: int,
    max_len: int,
    difficulty_range: tuple[float, float] = (0.2, 0.8),
    domains: list[str] | None = None,
) -> list[tuple[list[int], list[int], list[int]]]:
    """Build tokenized step-boundary examples from the Olympiad generator suite.

    Uses GeneratorRegistry (225 generators across arithmetic, algebra, fractions,
    geometry, number_theory, combinatorics) to produce a MIXED set — randomly
    picks a generator on each draw (across all domains unless `domains` restricts)
    and samples difficulty uniformly from `difficulty_range`.

    This is cleaner than MATH's heuristic text splitting because Problem.steps is
    the native structured reasoning trace — no parsing needed.

    Args:
        tokenizer:       HF tokenizer with eos_token_id set.
        n_examples:      Target number of examples to return.
        seed:            Controls both generator selection order and difficulty sampling.
        max_len:         Drop examples whose total token length exceeds this.
        difficulty_range: (lo, hi) uniform range for difficulty passed to generate().
        domains:         Optional list of domain strings to restrict to
                         (e.g. ['algebra', 'geometry']).  None = all domains.
                         Domain strings match Domain enum values (lowercase).

    Returns:
        list of (input_ids, labels, boundary_positions) — same contract as
        build_gsm8k_examples / build_math_examples:
          input_ids         — prompt + step tokens + EOS
          labels            — IGNORE_INDEX on prompt, pre-shifted next-token on response
          boundary_positions — absolute indices of the LAST token of each step,
                               all with labels[pos] != -100.
        Dropped: examples > max_len, examples with < 3 step boundaries.
    """
    import sys as _sys
    if _OLYMPIAD_SRC not in _sys.path:
        _sys.path.insert(0, _OLYMPIAD_SRC)

    from olympiad_data.generators.registry import GeneratorRegistry
    import olympiad_data.generators  # noqa: F401 — triggers auto-registration of all 225

    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("tokenizer has no eos_token_id")

    # Build the candidate pool of generator names, optionally filtered by domain
    all_names = GeneratorRegistry.list_all()
    if domains is not None:
        domain_set = set(d.lower() for d in domains)
        filtered: list[str] = []
        for name in all_names:
            try:
                gen_tmp = GeneratorRegistry.get(name)
                if str(gen_tmp.domain).lower() in domain_set:
                    filtered.append(name)
            except Exception:
                pass
        candidate_names = filtered
    else:
        candidate_names = list(all_names)

    if not candidate_names:
        raise ValueError(
            f"No generators available for domains={domains}. "
            f"Available domains: {[str(d) for d in GeneratorRegistry.list_domains()]}"
        )

    rng = random.Random(seed)
    diff_lo, diff_hi = difficulty_range

    examples: list[tuple[list[int], list[int], list[int]]] = []
    dropped_len = 0
    dropped_steps = 0
    domain_counts: dict[str, int] = {}
    # Track attempts to avoid infinite loops on degenerate configs
    max_attempts = max(n_examples * 20, 2000)
    attempts = 0

    while len(examples) < n_examples and attempts < max_attempts:
        attempts += 1

        # Pick a random generator and difficulty
        gen_name = rng.choice(candidate_names)
        difficulty = rng.uniform(diff_lo, diff_hi)

        try:
            gen = GeneratorRegistry.get(gen_name)
            problem = gen.generate(difficulty=difficulty)
        except Exception:
            # Generator failed for this difficulty — skip silently
            continue

        # Extract question text: strip {{SLOT}} markers from template
        question_template = getattr(problem, "question_template", None) or getattr(
            problem, "question", None
        )
        if not question_template:
            continue
        question = _strip_slots(str(question_template))
        if not question:
            continue

        # Extract steps — native ReasoningStep tuples, no text parsing needed
        raw_steps = getattr(problem, "steps", ())
        steps: list[str] = [
            s.content.strip() for s in raw_steps if s.content and s.content.strip()
        ]

        # Append the final answer as the last step (mirrors the "#### answer" in GSM8K)
        answer = str(getattr(problem, "answer", "")).strip()
        if answer:
            steps.append(answer)

        if len(steps) < 3:
            dropped_steps += 1
            continue

        prompt = _format_prompt(question)
        result = _build_example(tokenizer, prompt, steps, max_len, eos)
        if result is None:
            # Distinguish length drop from step drop (for stats)
            p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            resp_check = tokenizer("\n".join(steps), add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(resp_check) + 1 > max_len:
                dropped_len += 1
            else:
                dropped_steps += 1
            continue

        examples.append(result)
        domain_str = str(getattr(problem, "domain", "unknown"))
        domain_counts[domain_str] = domain_counts.get(domain_str, 0) + 1

    if not examples:
        raise RuntimeError(
            f"No Olympiad examples survived tokenization/length/step filter "
            f"(tried {attempts} attempts; dropped_len={dropped_len}, "
            f"dropped_steps={dropped_steps})"
        )

    n_resp = sum(sum(1 for t in lbl if t != IGNORE_INDEX) for _, lbl, _ in examples)
    n_tot = sum(len(ids) for ids, _, _ in examples)
    n_bounds = sum(len(bp) for _, _, bp in examples)
    domain_summary = ", ".join(f"{d}:{c}" for d, c in sorted(domain_counts.items()))
    print(
        f"[olympiad-data] {len(examples)} examples "
        f"(dropped {dropped_len} > max_len={max_len}, {dropped_steps} <3 steps, "
        f"{attempts} attempts); "
        f"{n_tot} tokens total, {n_resp} trainable response tokens "
        f"({100 * n_resp / n_tot:.1f}%), {n_bounds} total boundaries "
        f"(avg {n_bounds / len(examples):.1f}/example); "
        f"domains: {domain_summary}",
        flush=True,
    )
    return examples
