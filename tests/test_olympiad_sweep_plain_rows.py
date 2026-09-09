"""The plain-control rows of ``lab/divergence/olympiad_sweep.py`` must be the trainer's cut.

``MultiSourceCurriculumLoader._fill`` cuts a stream into NON-overlapping rows of
``seq_len + 1`` tokens. The sweep's plain path once advanced by ``seq_len`` per row, which
shifted row k by k tokens. On a shard whose every document is exactly ``seq_len + 1``
tokens (Sudoku, E17) a model trained on that cut has only ever seen a document at offset
0, and the shifted rows read 2.46 nats where the trainer read 0.55 (2026-09-08). The test
pins the cut on such a shard: every row starts on a document start, every document is
scored once, and the last, partial batch is kept.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, "lab/divergence")
from olympiad_sweep import pack_rows  # noqa: E402

L = 7           # seq_len: every document below is L + 1 = 8 tokens
N_DOCS = 11     # 11 rows of 8 tokens at batch 4: 2 full batches + a partial one of 3


def _stream() -> list[int]:
    # document d = [100 + d, 1, 2, 3, 4, 5, 6, 0]: a unique first token then a fixed body
    return [t for d in range(N_DOCS) for t in [100 + d, 1, 2, 3, 4, 5, 6, 0]]


def _cfg():
    return SimpleNamespace(data=SimpleNamespace(seq_len=L))


def test_every_row_starts_on_a_document_start_and_every_document_is_scored():
    stream = _stream()
    batches = pack_rows(stream, None, _cfg(), batch=4, plain=True)
    rows = torch.cat([b[0] for b in batches])
    labels = torch.cat([b[1] for b in batches])
    idx = torch.cat([b[3] for b in batches])
    assert rows.shape == (N_DOCS, L), rows.shape
    assert [len(b[0]) for b in batches] == [4, 4, 3]  # the partial last batch is kept
    for k in range(N_DOCS):
        assert int(rows[k, 0]) == 100 + k, f"row {k} does not start on document {k}"
        assert rows[k].tolist() == stream[k * (L + 1): k * (L + 1) + L]
        assert labels[k].tolist() == stream[k * (L + 1) + 1: (k + 1) * (L + 1)]
        assert idx[k].tolist() == list(range(k * (L + 1), k * (L + 1) + L))
    # the label of every row's last position is the document's EOS; no label crosses a
    # document boundary and no token is scored twice
    assert (labels[:, -1] == 0).all()
    assert torch.equal(labels[:, :-1], rows[:, 1:])


def test_a_stride_of_seq_len_would_shift_row_k_by_k_tokens():
    # the control for the assertion above: the old cut lands row 1 on document 0's EOS
    stream = _stream()
    shifted = stream[L: 2 * L]
    assert shifted[0] == 0 and shifted[1] == 101, "the old stride begins row 1 one token early"
