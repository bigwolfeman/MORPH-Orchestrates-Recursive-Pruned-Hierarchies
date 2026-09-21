"""The SONAR span-embedding cache — rung P3 of LXTUL-P (``tul.code_target_source``).

WHAT IT IS. The LCTUL flow thinker's target today is E's VERBATIM reconstruction code of
the next span, and the context explains 5–15 % of it (`lab/theory/lctul_euler_depth/`,
``explained_of_residual_ratio``). Rung P3 swaps that target for a frozen, paraphrase-
invariant SENTENCE embedding of the same span's TEXT: SONAR's 1024-d
``text_sonar_basic_encoder`` (Meta's ``sonar-space`` package). LCM's number for why that
is worth a run: in SONAR space the true next sentence is retrievable among in-batch
alternatives 75–80 % of the time (their Table 3 CA column), so a denoiser on it has a
context-dependence to learn. The design note is
``.agents/notes/proposed/architecture/2026-09-21-lxtul-particles-what-gives-a-pass-a-job.md``
(rung P3); the LCM reading is
``docs/references/tul-latent-emission/lcm/2026-09-21-lcm-reading.md``.

WHY A CACHE AND NOT AN ENCODER IN THE FORWARD. SONAR is a 700 M-parameter seq2seq encoder
and MORPH's 5090 has about 1 GB of slack on a TUL step (root CLAUDE.md). LCM made the same
call for the same reason and measured it: reading precomputed embeddings runs at over
20k/s/GPU against 300–400/s to encode on the fly (their Appendix A). The trainer therefore
reads a memmap.

THE KEY IS THE TOKEN IDS, NOT THE TEXT. Two spans that detokenise to the same string can
come from different token sequences, and the packer works in token space. The cache key is
``span_token_hash`` — BLAKE2b-64 over the span's int32 little-endian token ids — so the
trainer's lookup cannot drift from the script's write through a tokenizer round trip. At
6 M spans the 64-bit birthday collision probability is about 1e-6; a collision would hand
one span another's embedding, silently. That is the one accepted risk of the 8-byte key,
named here so it is not discovered later.

A MISS RAISES. A silent zero row would be a target of exactly 0 for that slot, which reads
as a perfectly learnable direction and would poison the arm invisibly (the
`counters-that-always-read-zero` failure). :meth:`SonarSpanCache.lookup` returns −1 and the
caller raises with the span's first token ids in the message.

THE EXPANSION IS FROZEN. SONAR is 1024-d; the thinker's target is ``[B, S, M, C]`` with
``M = prefix_k`` cells of width ``C = d_model``. :func:`frozen_code_expansion` is a fixed
random semi-orthogonal ``[M·C, 1024]`` matrix drawn from a seeded private generator and
held as a NON-PERSISTENT buffer. Frozen, not trained, for the reason measured on
2026-09-17 (`frozen-encoder-on-live-front-is-not-a-fixed-target`,
``tul.code_target_ref``): a target the model can move is not a target — E's codes
collapsed to own-cosine 0.99 within 3000 steps when only E's INPUT was allowed to drift.
A TRAINED 1024→M·C map would hand the model that same freedom by construction, and the
flow loss would fall by shrinking the target rather than by predicting it. Orthogonal so
the map is an isometry up to the RMS norm that follows: every SONAR direction survives
with its own length, no direction is amplified, and cosine geometry in SONAR space is
cosine geometry in cell space. Non-persistent because it is a pure function of the seed:
the checkpoint layout of a ``sonar`` arm is byte-identical to its ``"e"`` partner's.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "SONAR_DIM", "SONAR_ENCODER", "SONAR_LANG",
    "span_token_hash", "row_span_slices", "span_hashes", "next_span_hashes",
    "frozen_code_expansion", "SonarSpanCache", "write_sonar_cache",
]

SONAR_DIM = 1024
SONAR_ENCODER = "text_sonar_basic_encoder"
SONAR_LANG = "eng_Latn"

_INDEX_FILE = "index.npy"
_EMB_FILE = "emb.f16.npy"
_META_FILE = "meta.json"


def span_token_hash(ids: Iterable[int]) -> int:
    """Stable 64-bit key of one span's ORDERED token ids.

    BLAKE2b with an 8-byte digest over the ids as little-endian int32, read back as an
    unsigned 64-bit integer. Stable across processes, machines and Python versions
    (``hash()`` is not — it is salted per process). The cache script and the trainer call
    THIS function, so there is one key rule.
    """
    a = np.ascontiguousarray(np.asarray(ids, dtype="<i4"))
    d = hashlib.blake2b(a.tobytes(), digest_size=8).digest()
    return int(np.frombuffer(d, dtype="<u8")[0])


def row_span_slices(bag_id_row: np.ndarray, slot_mask_row: np.ndarray,
                    max_slots: int) -> list[tuple[int, int]]:
    """``[(start, stop)]`` per bag id ``0 .. max_slots-1`` into the row's TOKEN positions.

    The packer lays tokens out in stream order and gives token ``i`` the index of the span
    it closes (``pack_tul_row``: ``bag_id[tok_pos] = n_b_before``, tail tokens keep the
    ``max_slots`` dump bin), so the bag ids of a row's token positions are NON-DECREASING
    and each span is one contiguous run. That is asserted here rather than assumed: a
    packer change that broke it would otherwise silently mis-key every span in the cache.
    An empty span (a bag with no token — the pad slots, and the dump bin) gets
    ``(0, 0)``.
    """
    tok = np.flatnonzero(~slot_mask_row)
    bags = bag_id_row[tok]
    if bags.size and not bool(np.all(np.diff(bags) >= 0)):
        raise RuntimeError(
            "bag_id over a row's token positions is not non-decreasing; the packer's "
            "span layout changed and the SONAR cache key rule no longer cuts spans.")
    lo = np.searchsorted(bags, np.arange(max_slots, dtype=bags.dtype), side="left")
    hi = np.searchsorted(bags, np.arange(max_slots, dtype=bags.dtype), side="right")
    return [(int(tok[lo[j]]) if hi[j] > lo[j] else 0,
             int(tok[hi[j] - 1]) + 1 if hi[j] > lo[j] else 0) for j in range(max_slots)]


def span_hashes(input_ids: np.ndarray, bag_id: np.ndarray, slot_mask: np.ndarray,
                max_slots: int) -> tuple[np.ndarray, np.ndarray, list[list[np.ndarray]]]:
    """Per-BAG keys over a packed batch.

    Args:
        input_ids: ``[B, L]`` int64 — the packed row (slot positions hold ``slot_id``).
        bag_id:    ``[B, L]`` int64 — the layout's bag ids.
        slot_mask: ``[B, L]`` bool  — True at slot positions.
        max_slots: the layout's slot budget ``S``.

    Returns ``(hashes [B, S] uint64, n_tok [B, S] int64, ids [B][S] arrays)``. Bag ``j``
    is the span that slot ``j`` closes; ``n_tok[b, j] == 0`` means that bag holds no token
    (a pad slot) and its hash is 0. ``ids`` carries the token ids themselves so a caller
    that has to detokenise does not cut the row twice.
    """
    B = int(input_ids.shape[0])
    h = np.zeros((B, max_slots), dtype=np.uint64)
    n = np.zeros((B, max_slots), dtype=np.int64)
    out: list[list[np.ndarray]] = []
    for b in range(B):
        sl = row_span_slices(bag_id[b], slot_mask[b], max_slots)
        row: list[np.ndarray] = []
        for j, (a, z) in enumerate(sl):
            if z > a:
                seg = input_ids[b, a:z]
                if bool(slot_mask[b, a:z].any()):
                    raise RuntimeError(
                        f"span {j} of row {b} spans a slot position; the packer's span "
                        "layout changed and the SONAR cache key rule no longer cuts spans.")
                h[b, j] = span_token_hash(seg)
                n[b, j] = int(z - a)
                row.append(np.asarray(seg, dtype=np.int64))
            else:
                row.append(np.zeros(0, dtype=np.int64))
        out.append(row)
    return h, n, out


def next_span_hashes(h: np.ndarray, n_tok: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift the per-bag keys to the per-SLOT TARGET keys.

    Slot ``s`` is followed by span ``s+1`` — the span E pools today
    (``TULCodeEncoder.forward``: ``bag_id == s + 1``) and the one
    ``tul_code.code_target_valid`` calls slot ``s``'s code. The last slot of the budget has
    no next span. Returns ``(hashes [B, S] uint64, has [B, S] bool)``.
    """
    B, S = h.shape
    out = np.zeros((B, S), dtype=np.uint64)
    has = np.zeros((B, S), dtype=bool)
    out[:, :S - 1] = h[:, 1:S]
    has[:, :S - 1] = n_tok[:, 1:S] > 0
    return out, has


def frozen_code_expansion(d_in: int, d_out: int, seed: int) -> Tensor:
    """``[d_out, d_in]`` fp32 with ``Qᵀ Q = I_{d_in}`` — the frozen SONAR→cells lift.

    Drawn from a PRIVATE generator so a ``sonar`` arm's trainable weights are byte-identical
    to its ``"e"`` partner's, and sign-fixed against R's diagonal so the QR is deterministic
    across LAPACK builds. See the module docstring for why it is frozen and orthogonal.
    """
    if d_out < d_in:
        raise ValueError(
            f"frozen_code_expansion needs d_out >= d_in for an orthogonal lift, got "
            f"d_out={d_out} < d_in={d_in}: with M = prefix_k cells of width d_model, "
            f"prefix_k * d_model must be at least {d_in} (SONAR's width).")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    q, r = torch.linalg.qr(a, mode="reduced")
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return q.to(torch.float32).contiguous()


class SonarSpanCache:
    """A read-only span→SONAR-embedding cache on disk.

    Layout of ``path``: ``index.npy`` (``[N]`` uint64, SORTED, unique), ``emb.f16.npy``
    (``[N, 1024]`` float16, row ``i`` belongs to key ``index[i]``) and ``meta.json``. The
    embeddings are memmapped: the trainer touches only the rows a batch asks for, so the
    resident cost is the index (8 bytes a span) and not the table (2 KiB a span).
    """

    def __init__(self, path: str, mmap: bool = True):
        self.path = os.path.expanduser(str(path))
        idx_p = os.path.join(self.path, _INDEX_FILE)
        emb_p = os.path.join(self.path, _EMB_FILE)
        if not os.path.isdir(self.path):
            raise FileNotFoundError(
                f"tul.code_sonar_cache={path!r} is not a directory. Build it with "
                f"`scripts/sonar_span_cache.py`.")
        for p in (idx_p, emb_p):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"{p} is missing — {self.path} is not a SONAR span cache. Build it "
                    f"with `scripts/sonar_span_cache.py`.")
        self.index: np.ndarray = np.load(idx_p)
        self.emb: np.ndarray = np.load(emb_p, mmap_mode="r" if mmap else None)
        if self.index.dtype != np.uint64 or self.index.ndim != 1:
            raise ValueError(f"{idx_p}: expected a 1-D uint64 index, got "
                             f"{self.index.dtype} {self.index.shape}")
        if self.emb.ndim != 2 or self.emb.shape[0] != self.index.shape[0]:
            raise ValueError(f"{emb_p} {self.emb.shape} does not match the index "
                             f"{self.index.shape}")
        if self.emb.shape[1] != SONAR_DIM:
            raise ValueError(f"{emb_p} is {self.emb.shape[1]}-d; SONAR is {SONAR_DIM}-d")
        if self.index.size > 1 and not bool(np.all(np.diff(self.index) > 0)):
            raise ValueError(f"{idx_p} is not sorted and unique; rebuild the cache.")
        self.meta: dict = {}
        meta_p = os.path.join(self.path, _META_FILE)
        if os.path.isfile(meta_p):
            with open(meta_p) as f:
                self.meta = json.load(f)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.index.shape[0])

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """``[n]`` uint64 keys -> ``[n]`` int64 row indices; ``-1`` where the key is absent."""
        k = np.asarray(keys, dtype=np.uint64).reshape(-1)
        pos = np.searchsorted(self.index, k, side="left")
        pos_c = np.clip(pos, 0, max(self.index.shape[0] - 1, 0))
        hit = (self.index.shape[0] > 0) & (pos < self.index.shape[0]) & \
              (self.index[pos_c] == k)
        return np.where(hit, pos.astype(np.int64), np.int64(-1))

    def take(self, rows: np.ndarray) -> np.ndarray:
        """``[n]`` row indices -> ``[n, 1024]`` float32, copied out of the memmap."""
        r = np.asarray(rows, dtype=np.int64).reshape(-1)
        if r.size == 0:
            return np.zeros((0, SONAR_DIM), dtype=np.float32)
        return np.asarray(self.emb[r], dtype=np.float32)


def write_sonar_cache(path: str, keys: np.ndarray, emb: np.ndarray,
                      meta: dict | None = None) -> dict:
    """Write ``index.npy`` / ``emb.f16.npy`` / ``meta.json``, sorted and deduped.

    ``keys`` ``[n]`` uint64 and ``emb`` ``[n, 1024]``. Duplicate keys keep their FIRST
    embedding (a key is a token sequence, so the duplicates are the same span and the
    encoder is deterministic; keeping the first makes the write reproducible). Returns the
    meta dict that was written.
    """
    k = np.asarray(keys, dtype=np.uint64).reshape(-1)
    e = np.asarray(emb, dtype=np.float32)
    if e.ndim != 2 or e.shape[0] != k.shape[0]:
        raise ValueError(f"keys {k.shape} and emb {e.shape} disagree")
    if e.shape[1] != SONAR_DIM:
        raise ValueError(f"emb is {e.shape[1]}-d; SONAR is {SONAR_DIM}-d")
    order = np.argsort(k, kind="stable")
    k_s, e_s = k[order], e[order]
    keep = np.ones(k_s.shape[0], dtype=bool)
    if k_s.shape[0] > 1:
        keep[1:] = np.diff(k_s) > 0
    k_u, e_u = k_s[keep], e_s[keep]
    os.makedirs(os.path.expanduser(path), exist_ok=True)
    p = os.path.expanduser(path)
    np.save(os.path.join(p, _INDEX_FILE), k_u)
    np.save(os.path.join(p, _EMB_FILE), e_u.astype(np.float16))
    m = dict(meta or {})
    m.update({"n_rows": int(k_u.shape[0]), "n_written": int(k.shape[0]),
              "dim": SONAR_DIM, "encoder": SONAR_ENCODER, "lang": SONAR_LANG})
    with open(os.path.join(p, _META_FILE), "w") as f:
        json.dump(m, f, indent=2, sort_keys=True)
    return m
