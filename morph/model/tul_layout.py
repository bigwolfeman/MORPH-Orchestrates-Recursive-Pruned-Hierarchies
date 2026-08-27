"""TUL span layout — the ONE boundary rule, the row packer, and the slot layout.

Spec: ``docs/tul-spec.md`` §3.1 (sequence layout + boundary rule), §4 (data),
§6 (generation parity), §9 invariants 1/3/5.

This module is the single source of truth for "where does a span end and where
does its slot sit". :class:`BoundaryRule` is used **verbatim** by the training
loader (``morph/training/curriculum_data.py``, ``morph/training/data.py``) and by
the eager generator (``morph/inference/tul_generate.py``) — invariant 1 requires
one implementation, and :func:`BoundaryRule.cut` is a resumable state machine so
the generator can feed it one token at a time and get bit-identical boundaries to
the loader's whole-row call (``tests/test_tul_layout.py::test_incremental_parity``).

Provenance of the rule (spec §3.1, sources in §11): deterministic punctuation
boundaries with a collapse/merge/cap rule are within noise of learned ones and
beat fixed stride — BLT §4, Dynamic Token Pooling Table 2 (whitespace 1.133 ≈
unigram 1.134 > Gumbel 1.136 > entropy 1.138, fixed SF2 1.149), H-Net Table 1.
The fixed-stride control (``fixed_stride``) is arm A5, the SpaceByte Table 1 /
Hierarchical AT Table 1 control that separates alignment from depth.

RESOLVED SPEC AMBIGUITY — run collapse is causal (see ignore/Ai-notes/08-16-2026/
tul-impl/IMPL.md, "run collapse"). §3.1 rule 2 places the boundary after the LAST
token of a run of boundary tokens, which cannot be decided without looking at the
NEXT token; §6's generator loop and §9 invariant 1 both require a causal rule that
is identical at training and generation. The boundary therefore lands after the
FIRST token of a run, and ``min_span`` absorbs the rest of the run into the
following span — a ``.`` + ``\\n`` pair still yields exactly ONE boundary, which is
what rule 2 was for. See :meth:`BoundaryRule.cut`.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "BOUNDARY_SUFFIX_CHARS",
    "BOUNDARY_SUBSTRINGS",
    "BoundaryRule",
    "SlotLayout",
    "TulDataConfig",
    "TulGateSpec",
    "TulLayoutSpec",
    "boundary_lut_from_strings",
    "boundary_lut_from_tokenizer",
    "pack_tul_batch",
    "pack_tul_row",
    "slot_layout_from_ids",
]

# spec §3.1 rule 1 / §8 `tul.boundary_chars`: a token ends a span when its decoded
# string (trailing whitespace stripped) ENDS IN one of these …
BOUNDARY_SUFFIX_CHARS: str = ".;!?"
# … or CONTAINS one of these. No comma [W].
BOUNDARY_SUBSTRINGS: tuple[str, ...] = ("\n", "—", "–", "--")


# ── Boundary id resolution (spec §3.1 rule 1) ────────────────────────────────

def boundary_lut_from_strings(
    vocab_strings: list[str],
    eos_id: int,
    suffix_chars: str = BOUNDARY_SUFFIX_CHARS,
    substrings: tuple[str, ...] = BOUNDARY_SUBSTRINGS,
) -> np.ndarray:
    """``[V]`` bool lookup table: True where the token id ends a span.

    The rule is pure id-membership so it is causal (spec §3.1 rule 1 accepts the
    documented mis-cut of abbreviations/decimals for v1 — that is the price of a
    rule usable at generation, and §10 lists it as a known risk measured by the
    boundary stats).
    """
    lut = np.zeros(len(vocab_strings), dtype=bool)
    for i, s in enumerate(vocab_strings):
        if not s:
            continue
        if any(sub in s for sub in substrings):
            lut[i] = True
            continue
        stripped = s.rstrip()
        if stripped and stripped[-1] in suffix_chars:
            lut[i] = True
    if not 0 <= eos_id < len(lut):
        raise ValueError(f"eos_id {eos_id} outside vocab of {len(lut)}")
    lut[eos_id] = True          # spec §3.1: EOS ∈ B
    return lut


def boundary_lut_from_tokenizer(
    tokenizer_name: str,
    vocab_size: int,
    eos_id: int,
    cache_dir: str | None = None,
    suffix_chars: str = BOUNDARY_SUFFIX_CHARS,
    substrings: tuple[str, ...] = BOUNDARY_SUBSTRINGS,
) -> np.ndarray:
    """Resolve B from a HuggingFace tokenizer once, with an on-disk cache.

    Spec §3.1: "Resolved once from the tokenizer at build". Decoding 49k single-id
    sequences costs a few seconds, so the result is cached under ``cache_dir``
    keyed by (tokenizer, vocab_size, rule) — a stale cache would silently change
    the segmentation of a run, so the key covers every input to the rule.
    """
    key = f"{tokenizer_name}|{vocab_size}|{eos_id}|{suffix_chars}|{'.'.join(substrings)}"
    # hashlib, NOT hash(): PYTHONHASHSEED is randomised per process, so hash() gives a
    # different digest every run — the cache would never hit and would leak one .npy per
    # process, while still LOOKING like a cache (reviewer, 2026-08-16).
    digest = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"tul_boundary_{digest}.npy")
        if os.path.exists(path):
            lut = np.load(path)
            if lut.shape == (vocab_size,) and lut.dtype == bool:
                return lut
            raise RuntimeError(f"boundary cache {path} has wrong shape/dtype {lut.shape}/{lut.dtype}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    # The model vocabulary can be WIDER than the tokenizer (base.yaml: StarCoder2 49152
    # + 17 Olympiad structure tokens). Those extra ids have no decoded string; they are
    # structural, never span-final, so they stay out of B — but the LUT must still be
    # indexable by any model token id.
    n_tok = min(int(len(tok)), vocab_size)
    strings = tok.batch_decode([[i] for i in range(n_tok)]) + [""] * (vocab_size - n_tok)
    lut = boundary_lut_from_strings(strings, eos_id, suffix_chars, substrings)
    if path:
        np.save(path, lut)
    return lut


# ── The rule (spec §3.1 rules 2-4, §6) ───────────────────────────────────────

@dataclass(frozen=True)
class BoundaryRule:
    """Causal span cutter. ONE implementation for the loader and the generator.

    Args:
        is_boundary: ``[V]`` bool LUT from :func:`boundary_lut_from_strings`.
        min_span:    spec §3.1 rule 3 — a boundary below this span length is
                     suppressed and the short piece merges into the same span.
        span_cap:    spec §3.1 rule 4 [W] — a boundary is FORCED at this length.
        eos_id:      EOS ends a span unconditionally (see :meth:`cut`).
        fixed_stride: arm A5 (spec §3.5/§7.1). ``>0`` replaces the punctuation
                     rule with "a slot every N tokens"; EOS still cuts so spans
                     never straddle a document, which is what makes A5 a control
                     for alignment rather than for document mixing.
    """

    is_boundary: np.ndarray
    min_span: int = 4
    span_cap: int = 32
    eos_id: int = 0
    fixed_stride: int = 0

    def __post_init__(self) -> None:
        if self.min_span < 1:
            raise ValueError(f"min_span must be ≥ 1, got {self.min_span}")
        if self.span_cap < self.min_span:
            raise ValueError(
                f"span_cap ({self.span_cap}) < min_span ({self.min_span}): the cap would "
                f"force boundaries the min-span rule suppresses"
            )
        if self.fixed_stride < 0:
            raise ValueError(f"fixed_stride must be ≥ 0, got {self.fixed_stride}")
        if self.is_boundary.dtype != np.bool_:
            raise TypeError(f"is_boundary must be bool, got {self.is_boundary.dtype}")

    # -- the state machine -------------------------------------------------
    def cut(self, ids: np.ndarray, span_len: int = 0) -> tuple[np.ndarray, int]:
        """Boundary positions in ``ids``, and the span length left open at the end.

        ``span_len`` is the number of tokens already accumulated in the open span
        BEFORE this call — 0 at the start of a training row, threaded by the
        generator across single-token calls. The returned ``span_len_out`` feeds
        the next call, which is what makes the whole-row and one-token-at-a-time
        paths bit-identical (invariant 1, parity-tested).

        Rule, per token i (span_len counts the current token, spec §6):

            boundary  ⟺  ids[i] is EOS
                      ∨  (ids[i] ∈ B ∧ span_len ≥ min_span)
                      ∨  span_len == span_cap

        EOS is exempt from ``min_span``: §3.1 states "EOS is a boundary" without
        qualification, and letting a short tail document merge across the EOS
        would put two documents in one slot's bag-mean, which §3.1's "nothing
        crosses a document boundary" forbids.

        Returns:
            ``(positions [n_slots] int64, span_len_out)``. A slot is inserted
            AFTER ``ids[p]`` for each p in ``positions``.
        """
        n = int(ids.shape[0])
        if n == 0:
            return np.empty(0, dtype=np.int64), span_len

        eos_idx = np.flatnonzero(ids == self.eos_id)
        if self.fixed_stride > 0:
            cand_idx = np.empty(0, dtype=np.int64)      # A5: no punctuation rule
            cap = self.fixed_stride
            min_span = self.fixed_stride                # unreachable (no candidates)
        else:
            cand_idx = np.flatnonzero(self.is_boundary[ids])
            cap = self.span_cap
            min_span = self.min_span

        out: list[int] = []
        # base is the index of the previous boundary; span_len(i) == i - base.
        base = -span_len - 1
        while True:
            # first punctuation candidate whose span_len ≥ min_span
            j = np.searchsorted(cand_idx, base + min_span, side="left")
            nxt = int(cand_idx[j]) if j < cand_idx.shape[0] else n
            # first EOS strictly after the previous boundary (span_len ≥ 1)
            je = np.searchsorted(eos_idx, base + 1, side="left")
            nxt_eos = int(eos_idx[je]) if je < eos_idx.shape[0] else n
            pos = min(nxt, nxt_eos, base + cap)
            if pos >= n:
                break
            out.append(pos)
            base = pos
        return np.asarray(out, dtype=np.int64), (n - 1) - base


# ── The gate augmentation (docs/tul-gate-spec.md §3) ─────────────────────────

@dataclass(frozen=True)
class TulGateSpec:
    """Loader-side settings of the span-length gate (docs/tul-gate-spec.md §3).

    ``truncate_p`` is the ONLY augmentation. Spec §3 listed two (start jitter and end
    truncation); they are the two halves of ONE edit, so this implements one knob —
    see :func:`insert_truncations`. AMENDMENT recorded in the spec.

    Args:
        k_max:      ``gate_k_max`` — the length label is ``span_len / k_max``, so
                    ``span_cap`` must not exceed it (checked in :func:`insert_truncations`'s
                    caller, :func:`pack_tul_row`); otherwise the label saturates silently.
        truncate_p: per-span probability of inserting an extra RNG boundary. 0 = off,
                    and off is bit-identical to the pre-gate packer (no RNG is drawn).
    """

    k_max: int = 32
    truncate_p: float = 0.0

    def __post_init__(self) -> None:
        if self.k_max < 1:
            raise ValueError(f"gate_k_max must be ≥ 1, got {self.k_max}")
        if not 0.0 <= self.truncate_p <= 1.0:
            raise ValueError(f"gate_truncate_p must be in [0,1], got {self.truncate_p}")


def insert_truncations(
    bpos: np.ndarray, rule: BoundaryRule, gate: "TulGateSpec | None", rng
) -> tuple[np.ndarray, np.ndarray]:
    """Split spans at random interior points → ``(boundaries, is_rng)``.

    docs/tul-gate-spec.md §3.2: a span cut SHORT of its unit's boundary ends without
    punctuation, and the next span then starts mid-unit. That second span is exactly
    §3.1's "start jitter" — one insertion produces both — so this is the single
    augmentation, and §3.1's separate ``jitter_p`` is not built. (A row-phase offset
    would jitter ONE span per row; at ``truncate_p`` 0.15 over the ~55 spans of a
    1024-token row this jitters ~8, from the same knob.)

    The insertion is CONSISTENT with the rule, not a violation of it: the cut lands in
    ``[s + min_span − 1, e − min_span]``, and ``BoundaryRule.cut`` never places a
    candidate strictly inside that window (its first eligible candidate at or after
    ``s + min_span − 1`` IS ``e``, whether ``e`` came from punctuation, EOS or the cap).
    So restarting the state machine at the inserted cut still yields ``e`` next —
    which is what lets the generator force a cut at ``k`` tokens and stay in sync
    (``tests/test_tul_gate.py::test_truncation_is_consistent_with_the_rule``).

    Returns:
        ``(bpos_aug, is_rng)`` — sorted boundary positions, and a bool of the same
        shape that is True exactly at the inserted (RNG-chosen) ones. Those slots'
        length label is OUR noise, so §6 masks them out of the length term.
    """
    n = int(bpos.shape[0])
    if gate is None or gate.truncate_p <= 0.0 or n == 0:
        return bpos, np.zeros(n, dtype=bool)
    ms = rule.min_span
    starts = np.concatenate([[0], bpos[:-1] + 1])
    lens = bpos - starts + 1
    room = lens - 2 * ms                       # ≥ 0 ⇒ both halves clear min_span
    sel = (room >= 0) & (rng.random(n) < gate.truncate_p)
    n_sel = int(sel.sum())
    if n_sel == 0:
        return bpos, np.zeros(n, dtype=bool)
    off = np.floor(rng.random(n_sel) * (room[sel] + 1)).astype(np.int64)
    cuts = starts[sel] + ms - 1 + off
    aug = np.concatenate([bpos, cuts])
    flags = np.concatenate([np.zeros(n, dtype=bool), np.ones(n_sel, dtype=bool)])
    order = np.argsort(aug, kind="stable")
    return aug[order], flags[order]


# ── Row packing (spec §3.1 "Fixed shapes", §4) ───────────────────────────────

@dataclass(frozen=True)
class TulLayoutSpec:
    """Fixed-shape budget of one TUL row (spec §3.1). ``L_total`` is constant per stage."""

    seq_len: int
    prefix_k: int = 2
    max_slots: int = 0          # 0 → seq_len // 8 (spec §8 default)
    slot_id: int = 4            # "<fim_pad>" for starcoder2-7b; resolved at build

    def __post_init__(self) -> None:
        if self.prefix_k < 1:
            raise ValueError(f"prefix_k must be ≥ 1, got {self.prefix_k}")
        if self.max_slots == 0:
            object.__setattr__(self, "max_slots", self.seq_len // 8)
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be ≥ 1, got {self.max_slots}")

    @property
    def l_total(self) -> int:
        """``L_total = seq_len_tokens + prefix_k · max_slots`` (spec §3.1, invariant 5)."""
        return self.seq_len + self.prefix_k * self.max_slots


def pack_tul_row(
    ids: np.ndarray,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    gate: TulGateSpec | None = None,
    rng=None,
) -> tuple[dict[str, np.ndarray], int, dict[str, float]]:
    """Pack one fixed-shape TUL row out of a token buffer.

    Consumes tokens from ``ids`` until ``tokens + prefix_k · slots == L_total``.
    A boundary token is placed only when its ``prefix_k`` slot positions fit too,
    so no boundary inside the row is ever dropped (spec §3.1). When the next unit
    does not fit — or the row has spent its ``max_slots`` budget — the row ends
    and the ≤ ``prefix_k`` leftover positions become TAIL PADS (input ``slot_id``,
    label −100, marked in ``slot_mask``, absent from ``slot_index``). Tail pads sit
    at the END of the row, so causal attention for every real position is
    unchanged; they are the price of a fixed shape and are logged as
    ``tul/pad_frac``.

    Args:
        ids:  ``[n]`` int64 token buffer. Must hold at least one token MORE than
              the row consumes — the last row token's label is the next token.
        rule: the shared boundary rule.
        spec: the fixed-shape budget.
        gate: docs/tul-gate-spec.md §3. ``None`` (or ``truncate_p`` 0) draws NO random
              number and produces byte-identical output to the pre-gate packer, which
              is what keeps the reference arm reproducible.
        rng:  ``np.random.Generator``; required when ``gate.truncate_p > 0``.

    Returns:
        ``(arrays, n_consumed, stats)`` where ``arrays`` holds ``input_ids``,
        ``labels``, ``slot_mask``, ``bag_id`` (each ``[L_total]``) plus
        ``slot_index`` / ``slot_valid`` / ``span_len`` / ``len_supervised``
        (each ``[max_slots]``), and ``n_consumed`` is how many tokens of ``ids``
        the row used (the label peek is NOT consumed).

    ``span_len`` is the token count of each slot's span, clamped to ``gate.k_max``;
    ``len_supervised`` is False at pad slots and at slots whose span was ended by the
    truncation RNG rather than by the data (docs/tul-gate-spec.md §6 masks those out
    of the length term — grading a head on our own noise is label noise, not signal).
    """
    L, K, S = spec.l_total, spec.prefix_k, spec.max_slots
    n_buf = int(ids.shape[0])
    if n_buf < 2:
        raise ValueError(f"need ≥2 tokens to pack a row (label peek), got {n_buf}")
    # Spec §3.1: slot_id "never occurs in the shards … absence asserted at data prep".
    # A real slot_id in the corpus would be indistinguishable from an inserted slot, so
    # this is a hard failure, not a warning — silent corruption of the layout would be
    # unrecoverable and invisible in the loss.
    if bool(np.any(ids == spec.slot_id)):
        raise ValueError(
            f"slot_id {spec.slot_id} occurs in the token stream — it must be a special "
            f"token absent from the corpus (spec §3.1). Pick another tul.slot_token."
        )

    # Boundaries over the whole buffer; the row is a prefix of it, and the rule is
    # causal, so a prefix's boundaries are exactly this array truncated.
    bpos, _ = rule.cut(ids, span_len=0)
    k_max = gate.k_max if gate is not None else rule.span_cap
    if gate is not None and rule.span_cap > k_max:
        # The length label is span_len / k_max, so a span longer than k_max cannot be
        # expressed. Saturating silently would train the head on a wrong target for the
        # longest spans — exactly the class of silent label corruption the packer's
        # slot_id check already refuses (docs/tul-gate-spec.md §3.3).
        raise ValueError(
            f"tul.span_cap={rule.span_cap} > tul.gate_k_max={k_max}: the length label "
            f"would saturate. Raise gate_k_max or lower span_cap.")
    # The LABEL is clamped to span_cap, never to k_max: k_max is only the denominator, and
    # it carries deliberate headroom above span_cap so no target sits on the sigmoid's
    # asymptote. A label above span_cap would also index a budget row that has no
    # training example. The one length that is not a real span — the row's open tail — is
    # what this clamp actually bites on.
    lab_max = rule.span_cap
    bpos, b_is_rng = insert_truncations(bpos, rule, gate, rng)

    # cost(i) = positions used after placing tokens 0..i = (i+1) + K·(#boundaries ≤ i)
    is_b = np.zeros(n_buf, dtype=np.int64)
    if bpos.size:
        is_b[bpos] = 1
    n_b_upto = np.cumsum(is_b)
    cost = np.arange(1, n_buf + 1, dtype=np.int64) + K * n_b_upto
    n_tok = int(np.searchsorted(cost, L, side="right"))
    # spec §3.1: "If a row would exceed max_slots the packer ends the row early."
    if bpos.size > S:
        n_tok = min(n_tok, int(bpos[S]))
    if n_tok >= n_buf:
        raise ValueError(
            f"token buffer of {n_buf} exhausted while packing L_total={L}; caller must "
            f"supply more tokens (need one spare for the label peek)"
        )
    if n_tok < 1:
        raise ValueError(f"L_total={L} too small to hold one token")

    row_b = bpos[bpos < n_tok]
    n_slots = int(row_b.shape[0])
    used = n_tok + K * n_slots
    n_pad = L - used
    assert 0 <= n_pad, f"packer overflow: used {used} > L_total {L}"

    # ── position assignment ───────────────────────────────────────────────
    # Token i lands at row position i + K·(#boundaries strictly before i).
    n_b_before = n_b_upto[:n_tok] - is_b[:n_tok]
    tok_pos = np.arange(n_tok, dtype=np.int64) + K * n_b_before
    # Slot s (closing at token row_b[s]) owns positions tok_pos[row_b[s]]+1 … +K.
    slot_first = tok_pos[row_b] + 1 if n_slots else np.empty(0, dtype=np.int64)

    input_ids = np.full(L, spec.slot_id, dtype=np.int64)
    labels = np.full(L, -100, dtype=np.int64)
    slot_mask = np.ones(L, dtype=bool)
    bag_id = np.full(L, S, dtype=np.int64)            # S = the "no span" dump bin

    input_ids[tok_pos] = ids[:n_tok]
    labels[tok_pos] = ids[1:n_tok + 1]                # next-token, incl. the peek
    slot_mask[tok_pos] = False

    # span membership: token i belongs to the span closed by the first boundary ≥ i.
    # n_b_before is exactly that slot index while a slot still follows; tokens after
    # the last boundary keep the dump bin.
    if n_slots:
        last_b = int(row_b[-1])
        bag_id[tok_pos[:last_b + 1]] = n_b_before[:last_b + 1]
        # A slot position carries its OWN index, so one tensor drives both the bag-mean
        # (summed over token positions only — `slot_mask` selects them) and the gather that
        # writes each slot's input into its prefix_k positions. Tail pads keep the dump bin.
        for k in range(K):
            bag_id[slot_first + k] = np.arange(n_slots, dtype=np.int64)
        # emitting position = the LAST of the slot's prefix_k positions (spec §3.1):
        # the earlier ones carry the plan with NO label, the last predicts t_1(i+1).
        emit_pos = slot_first + (K - 1)
        labels[emit_pos] = ids[row_b + 1]

    slot_index = np.zeros(S, dtype=np.int64)          # invalid → 0 (masked; pads sit last)
    slot_valid = np.zeros(S, dtype=bool)
    if n_slots:
        slot_index[:n_slots] = slot_first
        slot_valid[:n_slots] = True

    # ── gate labels (docs/tul-gate-spec.md §3.3) ──────────────────────────
    # THE LABEL IS THE **NEXT** SPAN'S LENGTH, not the slot's own.
    # Slot i is built from span i's tokens and sits AFTER them, so causal attention lets
    # it condition only what comes after: span i+1. The budget the coda needs is therefore
    # the length of the span it is ABOUT TO DECODE, and a gate graded on span i's length —
    # a quantity slot i already contains, and one generation cannot use — would be reading
    # out the past while generation asks it to predict the future. (Spec §1's "the data's
    # own span length" was ambiguous about which span; AMENDED, see §3.3.)
    #
    # Invariant: 0 / False at every pad slot. A pad slot's slot_index is 0, so a missing
    # validity mask would silently train the gate on row 0's first span.
    span_len_arr = np.zeros(S, dtype=np.int64)
    len_sup = np.zeros(S, dtype=bool)
    if n_slots and gate is not None:
        _lens = np.diff(np.concatenate([[-1], row_b]))      # _lens[i] = len(span i)
        _next = np.empty(n_slots, dtype=np.int64)
        _next[:n_slots - 1] = _lens[1:n_slots]
        # The last slot's next span is the row's OPEN tail: real tokens the coda does
        # decode, so they are conditioned on their true count, but a count our row
        # boundary chose — so it is not graded (same rule as an RNG truncation).
        _next[n_slots - 1] = max(1, (n_tok - 1) - int(row_b[-1]))
        span_len_arr[:n_slots] = np.clip(_next, 1, lab_max)
        len_sup[:n_slots - 1] = ~b_is_rng[1:n_slots]
        len_sup[n_slots - 1] = False

    # ── boundary statistics (spec §4) ─────────────────────────────────────
    if n_slots:
        span_lens = np.diff(np.concatenate([[-1], row_b])).astype(np.float64)
        stats = {
            "spans": float(n_slots),
            "mean_span": float(span_lens.mean()),
            "one_tok_frac": float((span_lens == 1).mean()),
            "cap_frac": float((span_lens == rule.span_cap).mean()),
        }
    else:
        stats = {"spans": 0.0, "mean_span": 0.0, "one_tok_frac": 0.0, "cap_frac": 0.0}
    stats["tokens"] = float(n_tok)
    stats["pad_frac"] = float(n_pad) / float(L)
    arrays = {
        "input_ids": input_ids,
        "labels": labels,
        "slot_mask": slot_mask,
        "bag_id": bag_id,
        "slot_index": slot_index,
        "slot_valid": slot_valid,
    }
    if gate is not None:
        # Only when the gate is configured: the reference arm's layout tensors, its
        # wandb stat keys and its host→device copies stay exactly what they were.
        arrays["span_len"] = span_len_arr
        arrays["len_supervised"] = len_sup
        stats["trunc_frac"] = float(b_is_rng[:n_slots].mean()) if n_slots else 0.0
        stats["len_sup_frac"] = float(len_sup[:n_slots].mean()) if n_slots else 0.0
        stats["gate_span_mean"] = (float(span_len_arr[:n_slots].mean()) if n_slots else 0.0)
    return arrays, n_tok, stats


# ── The per-forward layout object (spec §8: `slot_layout` is a forward arg) ──

@dataclass
class SlotLayout:
    """Per-batch slot bookkeeping handed to ``MORPHTransformer.forward``.

    ``slot_layout=None`` selects the plain MORPH path, bit-identically (spec §8:
    "no runtime flags in the forward"; this is a data argument like ``bag_size``).

    Fields (all batched, ``B`` rows):
        slot_mask:  ``[B, L]`` bool — True at EVERY slot position, tail pads included.
                    Token-state dropout (§3.4) and the plan-nats gather (§7.2) key off it.
        bag_id:     ``[B, L]`` int64 — for a TOKEN position, the slot index whose span it
                    belongs to; for a SLOT position, its own slot index; ``max_slots``
                    (a dump bin) for tail pads and for tokens past the last slot. One
                    tensor drives both halves of the bag-mean slot input (§3.2): the sum
                    is taken over token positions (``~slot_mask``) and read back at slot
                    positions.
        slot_index: ``[B, max_slots]`` int64 — row position of each slot's FIRST
                    position; the core's gather map (§3.3). Invalid entries are 0.
        slot_valid: ``[B, max_slots]`` bool — real slots. Pads are last (§3.3), so
                    causal attention over the compact slot sequence is unchanged.
        prefix_k:   coda positions per slot (§3.1).
        span_len:   ``[B, max_slots]`` int64 — tokens this slot's span covers, 1…k_max;
                    0 at pad slots (docs/tul-gate-spec.md §3.3). None ⇒ no gate.
        len_supervised: ``[B, max_slots]`` bool — True when ``span_len`` is the DATA's
                    answer; False at pad slots and at RNG-truncated ones (gate §6).
    """

    slot_mask: Tensor
    bag_id: Tensor
    slot_index: Tensor
    slot_valid: Tensor
    prefix_k: int
    stats: dict[str, float] | None = None    # per-batch boundary statistics (spec §4)
    span_len: Tensor | None = None           # gate §3.3
    len_supervised: Tensor | None = None     # gate §3.3

    def __post_init__(self) -> None:
        B, L = self.slot_mask.shape
        if self.bag_id.shape != (B, L):
            raise ValueError(f"bag_id {tuple(self.bag_id.shape)} != slot_mask {(B, L)}")
        if self.slot_index.shape != self.slot_valid.shape:
            raise ValueError("slot_index and slot_valid must share a shape")
        if self.slot_index.shape[0] != B:
            raise ValueError("slot_index batch dim must match slot_mask")
        for _n in ("span_len", "len_supervised"):
            _t = getattr(self, _n)
            if _t is not None and _t.shape != self.slot_index.shape:
                raise ValueError(
                    f"{_n} {tuple(_t.shape)} != slot_index {tuple(self.slot_index.shape)}")
        if (self.span_len is None) != (self.len_supervised is None):
            raise ValueError("span_len and len_supervised must both be set or both None")

    @property
    def max_slots(self) -> int:
        return int(self.slot_index.shape[1])

    @property
    def l_total(self) -> int:
        return int(self.slot_mask.shape[1])

    def to(self, device) -> "SlotLayout":
        _mv = lambda t: None if t is None else t.to(device, non_blocking=True)
        return SlotLayout(
            slot_mask=self.slot_mask.to(device, non_blocking=True),
            bag_id=self.bag_id.to(device, non_blocking=True),
            slot_index=self.slot_index.to(device, non_blocking=True),
            slot_valid=self.slot_valid.to(device, non_blocking=True),
            prefix_k=self.prefix_k,
            stats=self.stats,
            span_len=_mv(self.span_len),
            len_supervised=_mv(self.len_supervised),
        )

    @staticmethod
    def from_rows(rows: list[dict[str, np.ndarray]], prefix_k: int,
                  stats: list[dict[str, float]] | None = None) -> "SlotLayout":
        """Stack per-row packer output into one batched layout."""
        agg = None
        if stats:
            agg = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        _st = lambda k: (torch.from_numpy(np.stack([r[k] for r in rows]))
                         if k in rows[0] else None)
        return SlotLayout(
            slot_mask=torch.from_numpy(np.stack([r["slot_mask"] for r in rows])),
            bag_id=torch.from_numpy(np.stack([r["bag_id"] for r in rows])),
            slot_index=torch.from_numpy(np.stack([r["slot_index"] for r in rows])),
            slot_valid=torch.from_numpy(np.stack([r["slot_valid"] for r in rows])),
            prefix_k=prefix_k,
            stats=agg,
            span_len=_st("span_len"),
            len_supervised=_st("len_supervised"),
        )


# ── TG restriction masks (docs/tul-tg-spec.md §1, §4) ────────────────────────
#
# The ONE builder both the window branch (via `_window_fallback`'s `extra_mask`) and
# the tests share for the causal "same-span-or-slot" allow relation. The compressed
# branch's own mask (`causal AND slot_mask[j]`, spec §3) is a strict subset of this
# one and is built directly from `layout.slot_mask` at the call site — it needs no
# `bag_id` term, so it is not this function's job.


def tg_allow_mask(layout: "SlotLayout", soft_prev_span: bool = False) -> Tensor:
    """``[B, 1, L, L]`` bool: TG1's within-span-or-slot allow relation (spec §1).

        allow(i, j) = (j <= i)                             # causal
                      AND ( bag_id[i] == bag_id[j]          # same span (tokens+own slot)
                            OR slot_mask[j] )                # or j is any slot position

    ``soft_prev_span=True`` (TG3, spec §6) adds one more disjunct:
    ``bag_id[i] == bag_id[j] + 1`` — spans are numbered in row order (the packer's
    ``n_b_before`` increments by exactly one per boundary, ``pack_tul_row``), so the
    previous span's id is always the current one minus one.

    CONSERVATIVE READING (not spelled out in the spec, documented here): the extra
    term is gated on the QUERY not itself being in the tail dump bin
    (``bag_id[i] < max_slots``). Tail-pad / post-last-slot positions all share the
    dump bin id ``max_slots`` (spec §1 note), so ``bag_id[i] - 1 == max_slots - 1``
    would otherwise open every dump-bin tail position onto the LAST real span — a
    grant "soft mode" does not intend, since the dump bin is not a span at all. No
    such gating is needed on the KEY side: a dump-bin key never satisfies
    ``bag_id[j] + 1 == bag_id[i]`` for any real span id, because dump-bin positions
    all carry the same id ``max_slots``, one past every real span.
    """
    bag_id = layout.bag_id                                    # [B, L] int64
    slot_mask = layout.slot_mask                               # [B, L] bool
    device = bag_id.device
    L = bag_id.shape[1]
    row = torch.arange(L, device=device).unsqueeze(1)
    col = torch.arange(L, device=device).unsqueeze(0)
    causal = (col <= row)                                       # [L, L], j <= i

    bag_i = bag_id.unsqueeze(2)                                 # [B, L, 1]
    bag_j = bag_id.unsqueeze(1)                                 # [B, 1, L]
    allow = (bag_i == bag_j) | slot_mask.unsqueeze(1)            # [B, L, L]
    if soft_prev_span:
        i_not_dump = bag_i < layout.max_slots                    # [B, L, 1]
        allow = allow | ((bag_i == bag_j + 1) & i_not_dump)
    allow = allow & causal.unsqueeze(0)
    return allow.unsqueeze(1)                                   # [B, 1, L, L]


def tg_reset_mask(layout: "SlotLayout") -> Tensor:
    """``[B, L]`` bool: GLA segment-reset positions (spec §4).

        reset[i] = (i == 0) OR (bag_id[i] != bag_id[i-1])

    True at every segment start — a segment is a span's tokens plus its own slot
    (the dump bin counts as one trailing segment too, since it is one constant id).
    """
    bag_id = layout.bag_id
    reset = torch.zeros_like(bag_id, dtype=torch.bool)
    reset[:, 0] = True
    reset[:, 1:] = bag_id[:, 1:] != bag_id[:, :-1]
    return reset


@dataclass
class TulDataConfig:
    """What a loader needs to emit TUL batches (spec §4, §8 `tul:` keys).

    Held by ``morph/training/data.py`` and ``morph/training/curriculum_data.py`` alike —
    one layout function, two token sources (spec §4 [W]: the arrow OpenWebText path in
    ``data.py`` is the one the arms run).
    """

    rule: BoundaryRule
    prefix_k: int = 2
    slot_id: int = 4
    max_slots: int = 0                # 0 → seq_len // 8 (spec §8)
    # docs/tul-gate-spec.md §3. `gate=None` ⇒ no span_len/len_supervised arrays and NO
    # random draw, i.e. the reference arm's loader is untouched. The VAL loader always
    # gets truncate_p 0 (train.py builds it with `for_val=True`): a val CE that depends
    # on our augmentation RNG is not comparable to the reference arm's.
    gate: TulGateSpec | None = None
    seed: int = 0                     # seeds the truncation RNG; logged in the manifest

    def spec_for(self, seq_len: int) -> TulLayoutSpec:
        """The fixed-shape budget for a stage of ``seq_len`` tokens."""
        return TulLayoutSpec(seq_len=seq_len, prefix_k=self.prefix_k,
                             max_slots=self.max_slots, slot_id=self.slot_id)


def pack_tul_batch(buf: list[int], rule: BoundaryRule, spec: TulLayoutSpec,
                   batch_size: int, gate: TulGateSpec | None = None,
                   rng=None) -> tuple[Tensor, Tensor, SlotLayout]:
    """Consume ``batch_size`` rows from ``buf`` (MUTATED in place) → one TUL batch.

    The single batching entry point for every loader. ``buf`` must hold at least
    ``batch_size · (L_total + 1)`` tokens: a row consumes at most ``L_total`` tokens and
    peeks one more for the last label, and the peeked token is deliberately left in the
    buffer to start the next row.

    Each row restarts the boundary state machine at ``span_len = 0``. A row is an
    independent training context, so its first span must be cut the way the generator
    would cut a fresh context — that is what makes the parity test meaningful.
    """
    need = batch_size * (spec.l_total + 1)
    if len(buf) < need:
        raise ValueError(f"buffer holds {len(buf)} tokens, need {need} for {batch_size} rows")
    rows, ins, labs, stats = [], [], [], []
    for _ in range(batch_size):
        cur = np.asarray(buf[:spec.l_total + 1], dtype=np.int64)
        arrays, n_used, st = pack_tul_row(cur, rule, spec, gate=gate, rng=rng)
        del buf[:n_used]
        rows.append(arrays)
        ins.append(arrays["input_ids"])
        labs.append(arrays["labels"])
        stats.append(st)
    return (
        torch.from_numpy(np.stack(ins)),
        torch.from_numpy(np.stack(labs)),
        SlotLayout.from_rows(rows, spec.prefix_k, stats),
    )


def slot_layout_from_ids(
    ids: np.ndarray,
    rule: BoundaryRule,
    spec: TulLayoutSpec,
    gate: TulGateSpec | None = None,
    rng=None,
) -> tuple[Tensor, Tensor, SlotLayout, list[dict[str, float]]]:
    """Pack a ``[B, n]`` token buffer into ``(input_ids, labels, layout, stats)``.

    The batch entry point used by the loaders, the generator's parity test and the
    unit tests. Every row consumes tokens independently, so rows may hold different
    token counts — that is the design (spec §3.1: "token count varies per row";
    ``tokens_per_batch`` is logged, BLT §4.3).
    """
    rows, ins, labs, stats = [], [], [], []
    for b in range(ids.shape[0]):
        arrays, _n, st = pack_tul_row(ids[b], rule, spec, gate=gate, rng=rng)
        rows.append(arrays)
        ins.append(arrays["input_ids"])
        labs.append(arrays["labels"])
        stats.append(st)
    return (
        torch.from_numpy(np.stack(ins)),
        torch.from_numpy(np.stack(labs)),
        SlotLayout.from_rows(rows, spec.prefix_k),
        stats,
    )
