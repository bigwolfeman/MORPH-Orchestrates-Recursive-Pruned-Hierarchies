"""Row packing shared by the depth sweeps: the same cut for every arm, with a stream map.

Two model families, one packer:

* a TUL arm: ``pack_tul_batch`` over the token stream (the trainer's packer), and the
  token -> stream-index map is recovered from the packed rows and CHECKED (the packer
  consumes tokens in order and peeks one label);
* the plain control: non-overlapping rows of ``seq_len + 1`` tokens, exactly the
  trainer's cut (``MultiSourceCurriculumLoader._fill``): row k holds
  ``stream[k(L+1) : (k+1)(L+1)]``. A stride of L (the bug fixed 2026-09-08, E17) shifts
  row k by k tokens; on a shard whose documents are all exactly L + 1 tokens (Sudoku)
  the trained model has only ever seen a document at offset 0 and read 2.46 nats
  against the trainer's 0.55. The last batch keeps its partial set of rows so every
  document is scored.

The stream index of every scored position travels with the rows, so two arms that pack
the SAME stream differently (a wider slot prefix, or the plain cut) pair at the TOKEN
level offline (`lab/experiments/results/2026-09-08-arc-e18/score_e18.py`).
"""
from __future__ import annotations

import torch


def pack_rows(stream: list[int], tul_rt, cfg, batch: int, plain: bool):
    """All batches once, with the stream index of every scored position.

    Returns ``[(inp, labels, layout_or_None, pos_index)]`` where ``pos_index`` is ``[B, L]``
    int64, the stream index of the INPUT token at each position (−1 at slot/pad positions).
    The label of a scored position is ``stream[pos_index + 1]``.
    """
    from morph.model.tul_layout import pack_tul_batch

    out = []
    if plain:
        L = int(cfg.data.seq_len)
        c = 0
        while c + L + 1 <= len(stream):
            ins, labs, idx = [], [], []
            while len(ins) < batch and c + L + 1 <= len(stream):
                seg = stream[c:c + L + 1]
                ins.append(seg[:-1])
                labs.append(seg[1:])
                idx.append(list(range(c, c + L)))
                c += L + 1
            out.append((torch.tensor(ins), torch.tensor(labs), None, torch.tensor(idx)))
        return out
    spec = tul_rt.data_cfg.spec_for(cfg.data.seq_len)
    rule = tul_rt.data_cfg.rule
    buf = list(stream)
    cursor = 0
    need = batch * (spec.l_total + 1)
    while len(buf) >= need:
        before = len(buf)
        inp, labels, layout = pack_tul_batch(buf, rule, spec, batch)
        used = before - len(buf)
        tokpos = ~layout.slot_mask
        idx = torch.full(inp.shape, -1, dtype=torch.long)
        k = cursor
        for b in range(inp.shape[0]):
            ps = tokpos[b].nonzero().flatten().tolist()
            for p in ps:
                idx[b, p] = k
                k += 1
        if k - cursor != used:
            raise RuntimeError(f"packer consumed {used} tokens but the rows hold {k - cursor} "
                               "token positions; the stream map would be wrong")
        got = inp[tokpos].tolist()
        exp = stream[cursor:cursor + used]
        if got != exp:
            raise RuntimeError("packed token positions do not reproduce the stream in order")
        out.append((inp, labels, layout, idx))
        cursor += used
    return out


def stream_from_loader(loader, n_tokens: int) -> list[int]:
    """At least ``n_tokens`` tokens of the loader's stream, in order, from its start."""
    stream: list[int] = []
    while len(stream) < n_tokens:
        stream.extend(next(loader)[0].reshape(-1).tolist())
    return stream
