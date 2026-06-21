"""MORPH curriculum pretraining dataloader.

Serves the length-bucketed curriculum over pre-tokenized shards (see scripts/pretokenize.py):
  - **weighted multi-source blend** (token-level weights honored via per-doc weighted draw),
  - **length buckets** — each doc is assigned to the stage whose native seq_len first fits it;
    a stage serves only its bucket's docs, so short docs train at short seq_len and genuinely
    long docs populate the long-context stages (no short docs padded up to look long),
  - **carry-split packing** — whole docs are packed EOS-separated to exactly fill seq_len with
    ZERO padding; at most one doc is split at a bin boundary and its remainder carried to the
    next bin (intra-document, contiguous — not cross-*doc* bleed). The carry is flushed on a
    stage switch so a short-bucket remainder never contaminates a long-context sequence.

Stage is switched live by the CurriculumScheduler via ``set_stage(k)``; the ``batches`` generator
reads the current stage/seq_len each iteration, so one infinite generator spans the whole run.

TST (``bag_size > 0``) mirrors morph/training/data.py: serves ``s·(seq_len+1)`` raw tokens →
input [B, s·L] + label_bags [B, L, s]. ``bag_size == 0`` → standard NTP (input/label, [B, L]).
"""
from __future__ import annotations
import json, os
from typing import Generator, Tuple
import numpy as np
import torch

from morph.training.source_roles import (
    DEFAULT_ALLOWED_PRETRAIN_ROLES,
    validate_source_for_pretraining,
)

__all__ = ["MultiSourceCurriculumLoader"]


class _Source:
    def __init__(self, name: str, sdir: str, weight: float, allowed_roles):
        self.name = name
        self.weight = float(weight)
        self.meta = json.load(open(os.path.join(sdir, "meta.json")))
        self.role = validate_source_for_pretraining(
            name,
            explicit_role=self.meta.get("role"),
            allowed_roles=allowed_roles,
            paths=self.meta.get("paths"),
        )
        self.offsets = np.load(os.path.join(sdir, "doc_offsets.i64.npy"))   # [n_docs+1]
        self.lens = np.load(os.path.join(sdir, "doc_lens.i32.npy"))          # [n_docs]
        # memmap the token blob — never pull 8B tokens into RAM.
        self.tokens = np.memmap(os.path.join(sdir, "tokens.u16.bin"),
                                dtype=np.uint16, mode="r")
        self.eos_id = int(self.meta["eos_id"])
        # filled by the loader once boundaries are known: stage_idx per doc
        self.stage_of_doc: np.ndarray | None = None
        # per-stage shuffled doc-index queues + cursor
        self._stage_docs: dict[int, np.ndarray] = {}
        self._cursor: dict[int, int] = {}

    def doc_tokens(self, i: int) -> np.ndarray:
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return np.asarray(self.tokens[a:b], dtype=np.int64)   # includes trailing EOS

    def assign_stages(self, boundaries: list[int]):
        # stage = first bucket whose seq_len >= doc_len; docs longer than the top
        # boundary fall in the top stage (they'll be carry-split across sequences).
        b = np.asarray(boundaries)
        idx = np.searchsorted(b, self.lens, side="left")     # lens<=b[idx]
        idx = np.clip(idx, 0, len(boundaries) - 1)
        self.stage_of_doc = idx.astype(np.int64)

    def build_stage_queue(self, stage: int, rng: np.random.Generator):
        docs = np.nonzero(self.stage_of_doc == stage)[0]
        rng.shuffle(docs)
        self._stage_docs[stage] = docs
        self._cursor[stage] = 0

    def has_stage(self, stage: int) -> bool:
        return self._stage_docs.get(stage) is not None and len(self._stage_docs[stage]) > 0

    def next_doc(self, stage: int, rng: np.random.Generator) -> np.ndarray:
        q = self._stage_docs[stage]
        c = self._cursor[stage]
        if c >= len(q):                       # epoch over this source's bucket → reshuffle
            rng.shuffle(q)
            c = 0
        self._cursor[stage] = c + 1
        return self.doc_tokens(int(q[c]))


class MultiSourceCurriculumLoader:
    def __init__(self, pretok_dir: str, weights: dict, stage_boundaries: list[int],
                 seed: int = 0, allowed_roles=DEFAULT_ALLOWED_PRETRAIN_ROLES):
        """weights: {source_name: weight} (need not sum to 1). stage_boundaries: ascending
        seq_lens, e.g. [4096, 8192, 16384] → 3 stages."""
        self.boundaries = [int(x) for x in stage_boundaries]
        self.n_stages = len(self.boundaries)
        self.rng = np.random.default_rng(seed)
        self.sources: list[_Source] = []
        for name, w in weights.items():
            if w <= 0:
                continue
            sdir = os.path.join(pretok_dir, name)
            if not os.path.isdir(sdir):
                raise FileNotFoundError(f"pretok shard missing for source {name!r}: {sdir} "
                                        f"(run scripts/pretokenize.py)")
            s = _Source(name, sdir, w, allowed_roles)
            s.assign_stages(self.boundaries)
            self.sources.append(s)
        if not self.sources:
            raise ValueError("no sources with positive weight")
        self.cur_stage = -1
        self.cur_seq_len = self.boundaries[0]
        self._carry: list[int] = []
        self._tok_count = {s.name: 0 for s in self.sources}    # realized per-source tokens
        self.set_stage(0)

    # ── stage control (driven by CurriculumScheduler) ──────────────────────
    def set_stage(self, k: int):
        if not (0 <= k < self.n_stages):
            raise IndexError(f"stage {k} out of range [0,{self.n_stages})")
        self.cur_stage = k
        self.cur_seq_len = self.boundaries[k]
        self._carry = []                                       # flush: no cross-stage bleed
        active = []
        for s in self.sources:
            s.build_stage_queue(k, self.rng)
            if s.has_stage(k):
                active.append(s)
        if not active:
            raise RuntimeError(f"stage {k} (seq_len {self.cur_seq_len}) has NO docs in any "
                               f"source — check bucket boundaries vs data lengths.")
        self._active = active
        # Token-proportioned draw: configured weights are TOKEN fractions (PRETRAINING.md),
        # so per-DOC draw prob must be ∝ weight / mean_doc_len → expected token fraction == weight.
        w = np.array([s.weight for s in active], dtype=np.float64)
        mean_len = np.array([float(s.lens[s.stage_of_doc == k].mean()) for s in active])
        p = w / mean_len
        self._probs = p / p.sum()
        active_names = [f"{s.name}:{s.role}" for s in active]
        print(f"[curriculum] stage {k}: seq_len={self.cur_seq_len}, "
              f"active={active_names}, mean_len={mean_len.round(0).tolist()}, "
              f"token-target={(w / w.sum()).round(3).tolist()}, "
              f"draw-probs={self._probs.round(3).tolist()}", flush=True)

    def realized_token_fractions(self) -> dict:
        tot = sum(self._tok_count.values()) or 1
        return {n: c / tot for n, c in self._tok_count.items()}

    # ── packing ────────────────────────────────────────────────────────────
    def _fill(self, n_tokens: int) -> list[int]:
        """Carry-split pack: return exactly n_tokens, drawing whole docs (source chosen by
        weight) and carrying any remainder to the next call. Each doc already ends in EOS."""
        buf = self._carry
        while len(buf) < n_tokens:
            src = self._active[int(self.rng.choice(len(self._active), p=self._probs))]
            toks = src.next_doc(self.cur_stage, self.rng)
            self._tok_count[src.name] += len(toks)
            buf.extend(toks.tolist())
        out = buf[:n_tokens]
        self._carry = buf[n_tokens:]
        return out

    def batches(self, batch_size: int, bag_size: int = 0
                ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Infinite. Reads cur_seq_len LIVE so a stage switch takes effect on the next batch."""
        while True:
            L = self.cur_seq_len
            if bag_size > 0:
                s = bag_size
                per = s * (L + 1)
                chunk = self._fill(batch_size * per)
                t = torch.tensor(chunk, dtype=torch.long).reshape(batch_size, s * (L + 1))
                input_ids = t[:, : s * L]
                label_bags = t[:, s : s * (L + 1)].reshape(batch_size, L, s)
                yield input_ids, label_bags
            else:
                chunk = self._fill(batch_size * (L + 1))
                t = torch.tensor(chunk, dtype=torch.long).reshape(batch_size, L + 1)
                yield t[:, :L], t[:, 1 : L + 1]
