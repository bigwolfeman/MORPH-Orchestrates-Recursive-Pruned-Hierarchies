"""Hop accuracy on the strict slot loop (`kind: lxtul`), CPU, the tiny `minimal` model of test_minimal.py.

  1. hop rows packed for the slot arm run through the model's own forward, and the labels the packer carried are
     still on the answer tokens (the reference reader run on the packed row finds them);
  2. p(correct) at every answer position is exp(-token_nll) of the same forward (the NLL's pointer mixture), and
     the prediction is the argmax of that mixture (no other id scores higher through the NLL rule);
  3. eval depths past the trained `max_depth` run, and the depth reaches the loop (every depth differs);
  4. a hop row replaces a row of a packed batch and the model's loss on it is finite.

    PYTHONPATH=.:$MORPH_ROOT python -m pytest -q lxtul/tests/test_hops_eval_tul.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lxtul.hops import chain as C
from lxtul.hops.base import FAMILIES
from lxtul.hops.eval import HopEval, answer_scores, hop_accuracy, make_batches
from lxtul.hops.mix import mix_hop_rows
from lxtul.tests.hops_support import EOS, SLOT, V, tiny_chain_vocab, tiny_rt, tiny_task
from lxtul.tests.test_compact import S, SEQ, cpu_reference_kernels, _to  # noqa: F401  (the fixture patches flex -> SDPA, fp64)
from lxtul.tests.test_minimal import _model

assert V == 512 and SLOT == 4


def _setup(B=2):
    task = tiny_task(SEQ, S, rows_per_bucket=4)
    rt = tiny_rt(SEQ, 1, S)
    rows = [r for _, r in task.eval_rows()]
    return task, rt, rows, make_batches(task, "lxtul", rt, rows, B)


def test_packed_labels_are_on_the_answer_tokens():
    task, rt, rows, batches = _setup()
    v = tiny_chain_vocab()
    for b, chunk in zip(batches, [rows[i: i + 2] for i in range(0, len(rows), 2)]):
        for j in range(b.x.shape[0]):
            tok = b.x[j][~b.lay.slot_mask[j]].numpy()
            tok = tok[tok != SLOT]
            pos = np.flatnonzero(~b.lay.slot_mask[j].numpy())
            got = 0
            ends = np.flatnonzero(tok == EOS) + 1
            for d0, d1 in zip(np.concatenate([[0], ends[:-1]]), ends):
                for at, value, h in C.solve(v, tok[d0: d1 - 1]):
                    p = pos[d0 + at - 1]
                    assert b.y[j, p] == value and b.hops[j, p] == h and b.family[j, p] == FAMILIES["chain"]
                    got += 1
            assert got == int((b.hops[j] > 0).sum()) == chunk[j].n_answers > 0


def _scores(m, b, depth):
    with torch.no_grad():
        return answer_scores(m, "lxtul", b, depth, SLOT)


def test_p_correct_is_the_nll_mixture_and_the_prediction_is_its_argmax():
    _, _, _, batches = _setup()
    m = _model(1).eval()
    with torch.no_grad():
        m.pointer.gate_norm_head.bias.copy_(torch.tensor([0.0, 1.5, 1.0, 0.5, 0.0]))
        m.pointer.q.weight.normal_(0, 0.6)
        m.pointer.k.weight.normal_(0, 0.6)
    b = batches[0]
    sel = (b.hops > 0).nonzero(as_tuple=True)
    pred, p, label, hops, fam = _scores(m, b, 3)
    assert len(label) == len(sel[0]) >= 2
    with torch.no_grad():
        h = m(b.x, None, b.lay, depth=3)["hidden"]
        nll = m.token_nll(h, b.x, b.y, b.lay.slot_mask, b.lay)
        assert torch.allclose(p, torch.exp(-nll[sel]), rtol=1e-8), (p, torch.exp(-nll[sel]))
        scored = []
        for alt in [pred, label] + [(label * 7 + k) % V for k in range(1, 10)]:
            y2 = b.y.clone()
            y2[sel] = alt
            scored.append(torch.exp(-m.token_nll(h, b.x, y2, b.lay.slot_mask, b.lay)[sel]))
    assert all((scored[0] >= t - 1e-9).all() for t in scored), "an id scores above the prediction"
    assert not torch.equal(pred, label) or (scored[0] == scored[1]).all()


def test_depths_past_max_depth_run_and_reach_the_loop():
    task, rt, rows, batches = _setup()
    m = _model(1).eval()
    assert m.tul.max_depth == 8
    res = hop_accuracy(m, "lxtul", batches, [1, 2, 8, 12, 16], SLOT, device="cpu")
    ps = {d: tuple(round(v["p"], 12) for v in res["buckets"][d].values()) for d in res["buckets"]}
    assert len(set(ps.values())) == 5, ps                               # 12 and 16 are not clamped to 8
    assert all(v["n"] > 0 and 0.0 <= v["acc"] <= 1.0 for d in res["buckets"].values() for v in d.values())


def test_hop_eval_wrapper_splits_val_and_full_rows():
    task, rt, rows, _ = _setup()
    ev = HopEval(task, "lxtul", rt, 2, SLOT, device="cpu")
    assert len(ev.full) * 2 == len(rows) == 8 and len(ev.val) * 2 == 4        # 2 buckets x 2 of 4 rows
    m = _model(1).eval()
    keys = ev.val_log(m, 2)
    assert set(keys) == {"val/hop/chain/acc_h1", "val/hop/chain/acc_h2", "val/hop/chain/acc_mean"}
    fin = ev.final(m, [1, 4])
    assert fin["depths"] == [1, 4] and fin["manifest"]["rows"] == 8


def test_a_mixed_batch_trains_one_finite_step():
    from lxtul.tests.test_minimal import _row1
    task, rt = tiny_task(SEQ, S, share=0.5), tiny_rt(SEQ, 1, S)
    (x, y, lay), spec = _row1(1, B=4)
    assert spec.l_total == SEQ + S
    mixed = next(mix_hop_rows(iter([(x, y, lay)]), task, spec))
    assert any(s >= 0 for s in task.assign(0, 4))
    m = _model(1).train()
    xx, yy, ll = _to(mixed, "cpu")
    o = m(xx, yy, ll)
    o["loss"].backward()
    assert torch.isfinite(o["loss"]) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
