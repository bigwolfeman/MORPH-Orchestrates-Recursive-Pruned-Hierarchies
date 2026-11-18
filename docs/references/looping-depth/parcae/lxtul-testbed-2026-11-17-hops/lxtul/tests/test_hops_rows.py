"""Hop rows: the row builder, the packer glue and the training mix (lxtul/hops/base.py, mix.py, data.py).

  1. a row is whole documents under BOTH the token cap and the span cap (the first one that does not fit is
     dropped), and a first document that does not fit raises;
  2. a packed row is MORPH's packer on the same tokens, then the packer's tail pad: label -100 on every pad
     position and on the row's last token, no slot token in a token position, no span beyond the documents';
  3. hop labels stay aligned through the packer (TUL) and through the flat row: the reference readers, run on
     the tokens recovered from the PACKED row, find every answer exactly where the labels say;
  4. the mix: exact share when B * share is whole, a seeded draw otherwise; hop row r depends on (seed, r)
     only; the same hop documents reach a TUL and a flat stream; `kind: none` builds no task; with a task the
     non-hop rows of a loader are the rows the loader without a task serves.

    PYTHONPATH=.:$MORPH_ROOT HF_HUB_OFFLINE=1 python -m pytest -q lxtul/tests/test_hops_rows.py
"""
from __future__ import annotations

import glob

import datasets
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from lxtul.data import layout_for_cells, make_loader
from lxtul.hops import chain as C
from lxtul.hops import graph as G
from lxtul.hops.base import FAMILIES, build_row, n_spans, pack_row, plain_row
from lxtul.hops.mix import build_task, mix_hop_rows
from lxtul.tests.hops_support import EOS, SLOT, compose_cfg, real, real_task, tiny_rule, tiny_task

from morph.model.tul_layout import TulLayoutSpec, pack_tul_batch, pack_tul_row


# ----------------------------------------------------------------------------------- 1. the row builder
def _docs_of(task, src, r):
    """The documents `train_row` draws, regenerated, to check the stopping rule."""
    rng = np.random.default_rng([task.seed, 1, src, r])
    while True:
        yield task.sources[src].train_document(rng)


@pytest.mark.parametrize("seq_len,max_slots,binds", [(128, 64, "tokens"), (128, 10, "spans"), (40, 64, "tokens")])
def test_a_row_is_whole_documents_under_both_caps(seq_len, max_slots, binds):
    task = tiny_task(seq_len, max_slots)
    rule = tiny_rule()
    bound = set()
    for r in range(30):
        row = task.train_row(0, r)
        n, s = len(row.tokens), n_spans(rule, row.tokens)
        assert n <= seq_len and s <= max_slots and row.tokens[-1] == EOS
        got, n_tok, n_sp, docs = 0, 0, 0, _docs_of(task, 0, r)
        for d in docs:                                           # the first document that does not fit ends the row
            if n_tok + len(d.tokens) > seq_len or n_sp + n_spans(rule, d.tokens) > max_slots:
                bound.add("tokens" if n_tok + len(d.tokens) > seq_len else "spans")
                break
            n_tok, n_sp, got = n_tok + len(d.tokens), n_sp + n_spans(rule, d.tokens), got + 1
        assert n_tok == n and n_sp == s and (row.tokens == EOS).sum() == got
    assert bound == {binds}                                      # the named cap ended every row


def test_a_first_document_that_does_not_fit_raises():
    with pytest.raises(ValueError, match="does not fit"):
        build_row(tiny_task(128, 16).sources[0].train_document, np.random.default_rng(0), 20, 16, tiny_rule())


# ------------------------------------------------------------------------------------- 2. the packed row
def _tok_pos(arrays):
    return np.flatnonzero(~arrays["slot_mask"])


@pytest.mark.parametrize("cells", [1, 4])
def test_packed_row_is_the_packer_plus_tail_pad(cells):
    task, (_, env, rt, _, _) = real_task(), real()
    spec = task.spec(cells)
    assert spec.max_slots == 256 and spec.l_total == 1024 + cells * 256
    for src in (0, 1):
        for r in range(4):
            row = task.train_row(src, r)
            a, N, ns = pack_row(row, task.rule, spec), len(row.tokens), n_spans(task.rule, row.tokens)
            used = N + cells * ns
            assert all(a[k].shape[0] == spec.l_total for k in ("input_ids", "labels", "slot_mask", "bag_id", "hops", "family"))
            assert a["slot_index"].shape == (256,) and int(a["slot_valid"].sum()) == ns and a["slot_valid"][:ns].all()
            tp = _tok_pos(a)
            assert len(tp) == N and (a["input_ids"][tp] == row.tokens).all()
            assert not (a["input_ids"][tp] == SLOT).any() and (a["input_ids"][a["slot_mask"]] == SLOT).all()
            pad = np.arange(spec.l_total) >= used
            assert (a["labels"][pad] == -100).all() and a["slot_mask"][pad].all() and (a["input_ids"][pad] == SLOT).all()
            assert (a["bag_id"][pad] == spec.max_slots).all() and not a["hops"][pad].any() and not a["family"][pad].any()
            assert a["labels"][tp[-1]] == -100 and a["labels"][a["slot_index"][ns - 1] + cells - 1] == -100
            assert not a["hops"][a["slot_mask"]].any(), "a hop label sits on a slot position"
            # the same layout as the production packer on the same tokens (+ filler it never reaches)
            filler = np.full(spec.l_total + 8, row.tokens[0], dtype=np.int64)
            ref, _, _ = pack_tul_row(np.concatenate([row.tokens, filler]), task.rule, spec)
            for k in ("input_ids", "slot_mask", "bag_id"):
                assert (ref[k][:used] == a[k][:used]).all(), k
            keep = np.ones(used, dtype=bool)
            keep[[tp[-1], a["slot_index"][ns - 1] + cells - 1]] = False
            assert (ref["labels"][:used][keep] == a["labels"][:used][keep]).all()
            assert (ref["slot_index"][:ns] == a["slot_index"][:ns]).all()


def test_flat_row_shifts_and_pads_with_ignore():
    task = real_task()
    for r in range(4):
        row = task.train_row(1, r)
        p, N = plain_row(row, task.seq_len, task.pad_id), len(row.tokens)
        assert p["x"].shape == p["y"].shape == (task.seq_len,)
        assert (p["x"][:N] == row.tokens).all() and (p["x"][N:] == task.pad_id).all()
        assert (p["y"][: N - 1] == row.tokens[1:]).all() and (p["y"][N - 1:] == -100).all()
        assert not p["hops"][N - 1:].any()


# ---------------------------------------------------------------------------- 3. labels stay aligned
def _recover(a, spec):
    """The row's tokens and the label position that predicts each, read off a packed row."""
    tp = _tok_pos(a)
    n = int((a["input_ids"][tp] != SLOT).sum())
    return a["input_ids"][tp][:n], tp[:n]


def test_hop_labels_survive_the_packer_chain_and_graph():
    task = real_task()
    spec = task.spec(1)
    chain, graph = task.sources
    for src, solve in ((0, lambda t: C.solve(chain.v, t[:-1])), (1, lambda t: G.solve(graph.v, t))):
        for r in range(6):
            row = task.train_row(src, r)
            a = pack_row(row, task.rule, spec)
            toks, pos = _recover(a, spec)
            assert (toks == row.tokens).all()
            found = set()
            for d0, d1 in _documents(toks, EOS_ID := real()[1].eos):
                for rel, value, *rest in solve(toks[d0:d1]):
                    at = d0 + rel                                  # the answer token; its label sits one before
                    p = pos[at - 1]
                    assert a["labels"][p] == value == toks[at] and a["hops"][p] > 0
                    assert a["hops"][p] == rest[0] and a["family"][p] == (FAMILIES["chain"] if src == 0 else rest[1])
                    found.add(int(p))
            assert found == set(np.flatnonzero(a["hops"]).tolist()) and len(found) == row.n_answers > 0
            p = plain_row(row, task.seq_len, EOS_ID)               # the flat row: label position = token - 1
            assert set(np.flatnonzero(p["hops"]).tolist()) == {int(np.flatnonzero(row.hops)[i]) - 1 for i in range(row.n_answers)}
            for q in np.flatnonzero(p["hops"]):
                assert p["y"][q] == row.tokens[q + 1] and p["hops"][q] == row.hops[q + 1]


def _documents(toks, eos):
    """[(start, end)] of the documents of a row (each ends with EOS, end exclusive)."""
    ends = np.flatnonzero(toks == eos) + 1
    return list(zip(np.concatenate([[0], ends[:-1]]), ends))


# ----------------------------------------------------------------------------------------- 4. the mix
def test_assignment_is_exact_when_the_share_is_whole_and_a_seeded_draw_otherwise():
    exact = tiny_task(128, 16, share=0.25)
    assert all(sum(s >= 0 for s in exact.assign(i, 4)) == 1 for i in range(50))
    assert {tuple(exact.assign(i, 4)).index(0) for i in range(50)} == {0, 1, 2, 3}      # positions are shuffled
    draw = tiny_task(128, 16, share=0.25)
    hits = np.array([[s >= 0 for s in draw.assign(i, 6)] for i in range(4000)])
    assert abs(hits.mean() - 0.25) < 0.02 and hits.sum(1).std() > 0.5                   # not fixed per batch
    assert draw.assign(7, 6) == draw.assign(7, 6) and any(draw.assign(i, 6) != draw.assign(0, 6) for i in range(1, 20))
    real_t = real_task()
    assert real_t.assign(3, 6) == real_t.assign(3, 6)


def test_a_hop_row_depends_on_seed_and_row_index_only():
    a, b = tiny_task(128, 16, seed=1), tiny_task(128, 16, seed=1)
    assert (a.train_row(0, 5).tokens == b.train_row(0, 5).tokens).all()
    assert not np.array_equal(a.train_row(0, 5).tokens, a.train_row(0, 6).tokens)
    assert not np.array_equal(a.train_row(0, 5).tokens, tiny_task(128, 16, seed=2).train_row(0, 5).tokens)
    with pytest.raises(ValueError, match="must differ"):
        tiny_task(128, 16, seed=1).__class__(**{**a.__dict__, "eval_seed": 1})


def _owt_like(spec, steps, B, seed):
    rng = np.random.default_rng(seed)
    rule = real_task().rule
    out = []
    for _ in range(steps):
        buf = rng.integers(1000, 40000, size=B * (spec.l_total + 1)).tolist()
        out.append(pack_tul_batch(buf, rule, spec, B))
    return out


def test_the_same_hop_documents_reach_a_slot_stream_and_a_flat_stream():
    task, B, steps = real_task(), 6, 5
    spec = task.spec(1)                                          # hop layout; the stream below is stock (64 slots)
    base = _owt_like(layout_for_cells(real()[2], 1).data_cfg.spec_for(task.seq_len), steps, B, 0)
    flat = [(torch.randint(1000, 40000, (B, task.seq_len)), torch.randint(1000, 40000, (B, task.seq_len)))
            for _ in range(steps)]
    tul = list(mix_hop_rows(iter(base), task, spec))
    fl = list(mix_hop_rows(iter(flat), task, None))
    n_hop = 0
    for i, ((x, y, lay), (fx, fy)) in enumerate(zip(tul, fl)):
        for j, s in enumerate(task.assign(i, B)):
            if s < 0:                                           # an ordinary row is the stream's row, untouched
                L0 = base[i][0].shape[1]                        # an ordinary row: the stock row, widened (tail pads)
                assert (x[j, :L0] == base[i][0][j]).all() and (y[j, :L0] == base[i][1][j]).all()
                assert (lay.slot_mask[j, :L0] == base[i][2].slot_mask[j]).all() and (fx[j] == flat[i][0][j]).all()
                continue
            n_hop += 1
            want = task.train_row(s, i * B + j).tokens
            toks = x[j][~lay.slot_mask[j]].numpy()
            assert (toks[: len(want)] == want).all()
            assert (fx[j][: len(want)].numpy() == want).all() and int((fy[j] >= 0).sum()) == len(want) - 1
        valid = lay.slot_valid
        assert not (valid[:, 1:] & ~valid[:, :-1]).any(), "a pad slot sits before a real one"
        assert lay.slot_mask.shape == x.shape == y.shape == (B, spec.l_total) and lay.prefix_k == 1
    assert n_hop >= 3


def test_kind_none_builds_no_task_and_a_task_leaves_the_other_rows_alone(tmp_path):
    _, _, rt, mcfg, cfg = real()
    assert build_task({"kind": "none"}, mcfg, rt) is None and build_task(compose_cfg("minimal").task, mcfg, rt) is None
    d = tmp_path / "owt"
    text = [" ".join(["alpha beta gamma delta epsilon"] * 400) + f" doc{i}" for i in range(40)]
    datasets.Dataset.from_dict({"text": text}).save_to_disk(str(d))
    assert glob.glob(str(d / "*.arrow"))
    ocfg = SimpleNamespace(data=SimpleNamespace(tokenizer=str(mcfg.data.tokenizer), dataset=str(d / "*.arrow"),
                                                seq_len=int(mcfg.data.seq_len)))
    task, B = real_task(), 6
    for tul_rt in (layout_for_cells(rt, 1), None):
        plain = make_loader(ocfg, "train", B, tul_rt=tul_rt, prefetch=0)
        again = make_loader(ocfg, "train", B, tul_rt=tul_rt, prefetch=0, hop_task=None)
        mixed = make_loader(ocfg, "train", B, tul_rt=tul_rt, prefetch=0, hop_task=task)
        changed = 0
        for i in range(4):
            p, a, m = next(plain), next(again), next(mixed)
            assert all(torch.equal(u, v) for u, v in zip(p[:2], a[:2]))          # hop_task=None: the stream as before
            if tul_rt is not None:
                assert all(torch.equal(getattr(p[2], f), getattr(a[2], f)) for f in ("slot_mask", "bag_id", "slot_index", "slot_valid"))
            for j, s in enumerate(task.assign(i, B)):
                L0 = p[0].shape[1]                                              # slot arm: widened rows, same tokens
                same = torch.equal(p[0][j], m[0][j][:L0]) and torch.equal(p[1][j], m[1][j][:L0])
                assert same == (s < 0), (i, j, s)
                changed += s >= 0
        assert changed >= 2
    with pytest.raises(ValueError, match="training stream only"):
        make_loader(ocfg, "validation", B, tul_rt=None, prefetch=0, hop_task=task)


# ------------------------------------------------------------------------- eval rows are pinned
EVAL_HASH = {"chain": ("126b3c01c2ee442a4b47c37e3f80000ce14ee9a80760823d357cd57a4313e99a", 120),
             "graph": ("48e8f9e630d570a6e0c69e669f432bd458ff3c71d9aa167e0f69cc6bd6e58645", 120)}


def test_eval_rows_are_byte_identical_to_the_pinned_design():
    """sha256 of the eval tokens (int64), hops and family (uint8), row by row, default config. Computed before the
    2026-11-17 training-document change (dense questions, size mix); the eval must not move with it."""
    import hashlib
    h, n = {"chain": hashlib.sha256(), "graph": hashlib.sha256()}, {"chain": 0, "graph": 0}
    for name, r in real_task().eval_rows():
        src = name.split("/")[0]
        for a in (r.tokens.astype("int64"), r.hops.astype("uint8"), r.family.astype("uint8")):
            h[src].update(a.tobytes())
        n[src] += 1
    assert {k: (h[k].hexdigest(), n[k]) for k in h} == EVAL_HASH
