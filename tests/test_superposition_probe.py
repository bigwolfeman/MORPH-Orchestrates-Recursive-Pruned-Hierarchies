"""lab/divergence/superposition_probe.py — the superposition / committed / blur instrument.

Contracts, one test (or one parametrised family) each:

  * PLANTED WORTH: a stub model whose cell read is exact (the right cell or not) and whose
    offset-0 rank of t0 is planted per span gives back the planted per-bucket worth on
    offsets 1..n EXACTLY, a zero worth on offset 0, the planted bucket of every span, and
    the verdict the planted ratio implies (superposition / committed / blur).
  * THE SHUFFLES are real permutations: ``xrow`` never keeps a slot in its own row and
    reads only valid source slots; ``row`` is a bijection on each row's valid slots with
    no fixed point; pads map to themselves; a fan slot's K cells move together.
  * THE SEAM IS REACHED on the real model: under the tiny strict geometry a shuffled
    forward changes CE where a token reads a cell and leaves span-0 tokens (which read no
    cell) bit-exact; the probe reports its calls and raises if a shuffle never landed.
  * COUNTS: bucket span counts sum to the scored spans, and the scored spans are exactly
    the (row, bag) pairs with 1 <= bag < max_slots and a valid offset-0 label.
  * GRAM: a gram arm is read through the PRIOR at the fixed seed; two runs at one seed are
    bit-identical, another seed moves the own CE.
  * FAN-ALL: the probe runs on a tiny fan_mix="all" model and the seam moves [B, S, K, ...].

CPU, fp32, the tests/test_tul_strict_geometry.py and tests/test_tul_fan.py fixtures.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

from test_tul_strict_geometry import _model, _pack

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LAB = os.path.join(_ROOT, "lab", "divergence")
if _LAB not in sys.path:
    sys.path.insert(0, _LAB)

import superposition_probe as sp  # noqa: E402


def _idx(layout) -> torch.Tensor:
    """Stream index per token position: row b's i-th token is b * 1000 + i (-1 at slots)."""
    tok = ~layout.slot_mask
    idx = torch.full(tok.shape, -1, dtype=torch.long)
    for b in range(tok.shape[0]):
        ps = tok[b].nonzero().flatten()
        idx[b, ps] = b * 1000 + torch.arange(ps.numel())
    return idx


def _batches(B: int = 3, seed: int = 0):
    _ids, inp, lab, layout = _pack(B=B, seed=seed)
    return [(inp, lab, layout, _idx(layout))]


def _expected_spans(layout, labels) -> set[tuple[int, int]]:
    """(row, bag) pairs with 1 <= bag < S whose first token position carries a label."""
    S = layout.slot_valid.shape[1]
    out = set()
    for b in range(layout.bag_id.shape[0]):
        seen: set[int] = set()
        for p in range(layout.bag_id.shape[1]):
            if bool(layout.slot_mask[b, p]):
                continue
            s = int(layout.bag_id[b, p])
            if s in seen:
                continue
            seen.add(s)
            if 1 <= s < S and int(labels[b, p]) >= 0:
                out.add((b, s))
    return out


# ── a stub whose worth is known per bucket ─────────────────────────────────────────────

V = 16


def _ce_of(a: float) -> float:
    """CE of the label when its logit is ``a`` and the other V - 1 logits are 0."""
    return math.log(V - 1 + math.exp(a)) - a


class _Planted:
    """A model stub with the real seam. Each slot's cell is its own id ``b * S + s + 1``;
    the forward runs it through ``self._tul_plan_ablate`` exactly as the real forward does.
    A token of span s at offset >= 1 gets label logit ``hi[k]`` if the cell it reads (slot
    s - 1) is its own and ``lo[k]`` otherwise, ``k = bucket_of(s)``. At offset 0 the label's
    rank is planted by ``bucket_of(s)`` and does NOT depend on the cell (worth 0)."""

    def __init__(self, hi, lo, bucket_of=lambda s: s % 3, ctx_bucket_of=None):
        self.hi, self.lo, self.bucket_of = hi, lo, bucket_of
        # the offset-0 rank under a FOREIGN cell (default: the same as the own cell's)
        self.ctx_bucket_of = ctx_bucket_of or bucket_of
        self.calls: list[dict] = []

    def eval(self):
        return self

    def _tul_plan_ablate(self, h, layout, mode):
        if mode == "normal":
            return h
        if mode == "zero":
            return torch.zeros_like(h)
        raise AssertionError(f"the stub's own random shuffle must never run (mode {mode!r})")

    def tul_forward_ablated(self, inp, labels, layout, plan_mode="normal", **kw):
        self.calls.append({"plan_mode": plan_mode, **kw})
        B, L = inp.shape
        S = layout.slot_valid.shape[1]
        own = (torch.arange(B).view(B, 1) * S + torch.arange(S).view(1, S) + 1).float()
        cells = self._tul_plan_ablate(own.unsqueeze(-1), layout, plan_mode).squeeze(-1)
        bag, off, _ = sp.span_offsets(layout, self._labels)
        lg = torch.zeros(B, L, V)
        for b in range(B):
            for p in range(L):
                s, o, y = int(bag[b, p]), int(off[b, p]), int(self._labels[b, p])
                if o < 0 or y < 0 or not (1 <= s < S):
                    continue
                right = bool(cells[b, s - 1] == own[b, s - 1])
                k = self.bucket_of(s)
                if o == 0:
                    k = k if right else self.ctx_bucket_of(s)
                    if k == 0:
                        lg[b, p, y] = 8.0
                    elif k == 1:
                        lg[b, p, y], lg[b, p, (y + 1) % V] = 6.0, 8.0
                    else:
                        lg[b, p, (y + 1) % V], lg[b, p, (y + 2) % V] = 8.0, 7.0
                    continue
                lg[b, p, y] = self.hi[k] if right else self.lo[k]
        return {"logits": lg}


def _run_planted(hi, lo, B=3, ctx_bucket_of=None, **kw):
    batches = _batches(B=B)
    inp, lab, layout, idx = batches[0]
    lab = lab % V                           # the stub's vocabulary; -100 stays -100
    lab[batches[0][1] < 0] = -100
    m = _Planted(hi, lo, ctx_bucket_of=ctx_bucket_of)
    m._labels = lab
    res, arr = sp.superposition_probe(m, [(inp, lab, layout, idx)], "cpu", seed=0,
                                      n_boot=200, **kw)
    return m, res, arr, (inp, lab, layout, idx)


@pytest.mark.parametrize("hi,lo,verdict", [
    ((3.0, 3.0, 3.0), (0.0, 1.0, 2.5), "superposition"),   # ratio ~0.59
    ((3.0, 3.0, 3.0), (0.0, 2.8, 2.5), "committed"),       # ratio ~0.08
    ((3.0, 3.0, 3.0), (0.0, 3.5, 2.5), "committed"),       # top2 worth < 0
    ((3.0, 3.0, 3.0), (3.0, 3.0, 3.0), "blur"),            # every worth exactly 0
])
def test_planted_worth_per_bucket_and_the_verdict(hi, lo, verdict):
    m, res, arr, (inp, lab, layout, idx) = _run_planted(hi, lo)
    want = [_ce_of(lo[k]) - _ce_of(hi[k]) for k in range(3)]
    # the zero control reads a zero cell, never the own id: the same planted worth
    for c in ("xrow", "row", "zero"):
        for k, b in enumerate(sp.BUCKETS):
            d = res["worth"][c]["pos1plus"][b]
            assert d["n"] > 0, (c, b)
            assert d["mean"] == pytest.approx(want[k], abs=1e-5), (c, b)
            # offset 0 does not read the cell in the stub: worth exactly 0
            assert res["worth"][c]["pos0"][b]["mean"] == pytest.approx(0.0, abs=1e-6)
    assert res["decision"]["verdict"] == verdict
    if verdict != "blur":
        assert res["ratio_top2_top1"]["xrow"]["point"] == pytest.approx(want[1] / want[0],
                                                                        rel=1e-4)
    # every span landed in its planted bucket
    bag, off, scored = sp.span_offsets(layout, lab)
    first = scored & (off == 0)
    planted = [m.bucket_of(int(bag[b, p])) for b, p in zip(*torch.nonzero(first, as_tuple=True))]
    for k, b in enumerate(sp.BUCKETS):
        assert res["buckets"][b]["n_spans"] == planted.count(k)


def test_planted_counts_sum_to_the_scored_spans():
    _m, res, arr, (inp, lab, layout, idx) = _run_planted((3.0,) * 3, (0.0, 1.0, 2.0))
    want = _expected_spans(layout, lab)
    assert res["n_spans"] == len(want) > 6
    assert sum(res["buckets"][b]["n_spans"] for b in sp.BUCKETS) == res["n_spans"]
    # offset 0 appears exactly once per scored span; offsets are a contiguous 0..n-1 run
    assert int((arr["offset"] == 0).sum()) == res["n_spans"]
    for key in np.unique(arr["span_key"]):
        offs = np.sort(arr["offset"][arr["span_key"] == key])
        assert offs[0] == 0 and np.array_equal(offs, np.arange(offs[0], offs[0] + offs.size))
    # the pos1+ tokens are exactly the scored tokens at offset >= 1
    n1 = sum(res["worth"]["xrow"]["pos1plus"][b]["n"] for b in sp.BUCKETS)
    assert n1 == int((arr["offset"] >= 1).sum()) == res["worth"]["xrow"]["pos1plus"]["all"]["n"]


def test_planted_forwards_use_the_shuffle_seam_not_the_models_own_random_one():
    m, res, *_ = _run_planted((3.0,) * 3, (0.0, 1.0, 2.0))
    modes = [c["plan_mode"] for c in m.calls]
    # own, xrow, xrow2, row (all through the "shuffle" seam), zero
    assert modes == ["normal", "shuffle", "shuffle", "shuffle", "zero"]
    assert all(v > 0 for v in res["shuffle_calls"].values())
    assert res["shuffle_fixed_slots"] == {"xrow": 0, "xrow2": 0, "row": 0}
    assert res["gram_prior_seed"] is None and all("gram_mode" not in c for c in m.calls)


def test_a_shuffle_that_never_reaches_the_seam_raises():
    class _Deaf(_Planted):
        def tul_forward_ablated(self, inp, labels, layout, plan_mode="normal", **kw):
            B, L = inp.shape
            return {"logits": torch.zeros(B, L, V)}      # never calls _tul_plan_ablate
    b = _batches()
    inp, lab, layout, idx = b[0]
    lab = lab % V
    lab[b[0][1] < 0] = -100
    with pytest.raises(RuntimeError, match="never reached"):
        sp.superposition_probe(_Deaf((1,) * 3, (0,) * 3), [(inp, lab, layout, idx)], "cpu",
                               n_boot=10)


# ── the shuffles ───────────────────────────────────────────────────────────────────────

def _valid(B=4, S=7, n=(5, 3, 1, 6)):
    v = torch.zeros(B, S, dtype=torch.bool)
    for b, k in enumerate(n):
        v[b, :k] = True
    return v


@pytest.mark.parametrize("seed", range(5))
def test_xrow_is_a_row_derangement_onto_valid_slots(seed):
    v = _valid()
    sr, ss, nf = sp.make_shuffle(v, "xrow", np.random.default_rng(seed))
    assert nf == 0
    for b in range(v.shape[0]):
        for s in range(v.shape[1]):
            if v[b, s]:
                assert int(sr[b, s]) != b
                assert bool(v[int(sr[b, s]), int(ss[b, s])])
            else:
                assert (int(sr[b, s]), int(ss[b, s])) == (b, s)
    # all slots of a row come from ONE source row
    for b in range(v.shape[0]):
        assert len(set(sr[b][v[b]].tolist())) == 1


@pytest.mark.parametrize("seed", range(5))
def test_row_is_a_fixed_point_free_bijection_on_each_rows_valid_slots(seed):
    v = _valid()
    sr, ss, nf = sp.make_shuffle(v, "row", np.random.default_rng(seed))
    assert nf == 1                                     # the one-slot row cannot move
    for b in range(v.shape[0]):
        k = int(v[b].sum())
        assert torch.equal(sr[b], torch.full_like(sr[b], b))
        src = ss[b, :k].tolist()
        assert sorted(src) == list(range(k))
        if k >= 2:
            assert all(src[i] != i for i in range(k))
        assert ss[b, k:].tolist() == list(range(k, v.shape[1]))


@pytest.mark.parametrize("seed", range(5))
def test_xrow2_is_cross_row_and_differs_from_xrow_wherever_it_can(seed):
    v = _valid(B=2, n=(5, 3))                          # 2 rows: ONE row derangement exists
    rng = np.random.default_rng(seed)
    x = sp.make_shuffle(v, "xrow", rng)
    y = sp.make_shuffle(v, "xrow2", rng)
    for b in range(2):
        for s in range(int(v[b].sum())):
            assert int(y[0][b, s]) != b and bool(v[int(y[0][b, s]), int(y[1][b, s])])
            assert (int(x[0][b, s]), int(x[1][b, s])) != (int(y[0][b, s]), int(y[1][b, s]))


def test_planted_probe_reports_xrow2_distinct_from_xrow():
    _m, res, *_ = _run_planted((3.0,) * 3, (0.0, 1.0, 2.0), B=2)
    assert res["xrow2_same_as_xrow_slots"] == 0
    assert res["n_blocks"] < sp.MIN_BLOCKS and "not meaningful" in res["ci_warning"]


def test_xrow_refuses_one_row_and_a_fan_slot_moves_whole():
    with pytest.raises(ValueError, match=">= 2 rows"):
        sp.make_shuffle(_valid(B=1, n=(4,)), "xrow", np.random.default_rng(0))
    v = _valid()
    sr, ss, _ = sp.make_shuffle(v, "xrow", np.random.default_rng(3))
    B, S, K, C = v.shape[0], v.shape[1], 4, 5
    h = torch.randn(B, S, K, C)
    out = sp.apply_shuffle(h, sr, ss)
    for b in range(B):
        for s in range(S):
            assert torch.equal(out[b, s], h[int(sr[b, s]), int(ss[b, s])])


def test_the_decision_rule_thresholds():
    w = lambda m1, m2, lo1=None: {  # noqa: E731
        "top1": {"mean": m1, "lo": m1 - 0.01 if lo1 is None else lo1, "hi": m1 + 0.01, "n": 9},
        "top2": {"mean": m2, "lo": m2 - 0.01, "hi": m2 + 0.01, "n": 9},
        "other": {"mean": 0.0, "lo": -0.001, "hi": 0.001, "n": 9}}
    r = lambda p, lo, hi: {"point": p, "lo": lo, "hi": hi}  # noqa: E731
    assert sp.classify(w(0.2, 0.1), r(0.5, 0.45, 0.55), 0.01)["verdict"] == "superposition"
    assert sp.classify(w(0.2, 0.1), r(0.5, 0.5, 0.6), 0.01) == {"verdict": "superposition",
                                                                "ci_supported": True}
    assert sp.classify(w(0.2, 0.04), r(0.2, 0.1, 0.24), 0.01) == {"verdict": "committed",
                                                                  "ci_supported": True}
    assert sp.classify(w(0.2, -0.05), r(-0.25, -0.3, -0.2), 0.01)["verdict"] == "committed"
    assert sp.classify(w(0.2, 0.07), r(0.35, 0.3, 0.4), 0.01)["verdict"] == "intermediate"
    assert sp.classify(w(0.2, 0.1, lo1=-0.01), r(0.5, 0.1, 3.0), 0.01)["verdict"] == \
        "inconclusive"
    tiny = {b: {"mean": 0.0, "lo": -0.005, "hi": 0.005, "n": 9} for b in sp.BUCKETS}
    assert sp.classify(tiny, r(0.5, -9, 9), 0.01)["verdict"] == "blur"


def test_clusters_of_identical_distributions_carry_no_positive_gain():
    lp = torch.log_softmax(torch.randn(6, V), -1)
    n, pos = sp._clusters(lp, lp.clone(), 16, seed=0)
    assert (n >= 1).all() and (pos == 0).all()
    # a reference that puts no mass on the drawn tokens makes every cluster positive
    ref = torch.full_like(lp, -50.0)
    n2, pos2 = sp._clusters(lp, ref, 16, seed=0)
    assert np.array_equal(n, n2) and np.array_equal(pos2, n2)


# ── the real model ─────────────────────────────────────────────────────────────────────

def test_the_real_strict_forward_reaches_the_seam_and_span0_reads_no_cell():
    torch.manual_seed(0)
    m = _model(tg_geometry="strict")
    (inp, lab, layout, idx), = _batches()
    bag, off, scored = sp.span_offsets(layout, lab)
    fb, fp = torch.nonzero(scored & (off == 0), as_tuple=True)
    ce_own, _ = sp.forward_ce(m, inp, layout, lab, "cpu", "normal", {}, (fb, fp))
    sr, ss, _ = sp.make_shuffle(layout.slot_valid, "xrow", np.random.default_rng(0))
    with sp.controlled_shuffle(m, sr, ss) as calls:
        ce_x, _ = sp.forward_ce(m, inp, layout, lab, "cpu", "shuffle", {}, (fb, fp))
    assert calls["calls"] >= 1
    assert "_tul_plan_ablate" not in m.__dict__          # the override is gone after
    tok = (~layout.slot_mask) & (lab >= 0)
    span0 = tok & (bag == 0)
    later = tok & (bag >= 1) & (bag < layout.slot_valid.shape[1])
    assert span0.any() and later.any()
    assert torch.equal(ce_own[span0], ce_x[span0])      # strict: span 0 reads no cell
    assert (ce_own[later] - ce_x[later]).abs().max() > 1e-4


def test_the_real_strict_probe_counts_and_runs():
    torch.manual_seed(0)
    m = _model(tg_geometry="strict")
    batches = _batches()
    res, arr = sp.superposition_probe(m, batches, "cpu", seed=0, n_boot=200)
    _, lab, layout, _ = batches[0]
    assert res["n_spans"] == len(_expected_spans(layout, lab))
    assert sum(res["buckets"][b]["n_spans"] for b in sp.BUCKETS) == res["n_spans"]
    assert all(v >= 1 for v in res["shuffle_calls"].values())
    assert res["shuffle_fixed_slots"]["xrow"] == 0
    assert res["decision"]["verdict"] in ("superposition", "committed", "intermediate",
                                          "blur", "inconclusive")
    assert np.isfinite(arr["ce_own"]).all() and np.isfinite(arr["ce_xrow"]).all()


def test_a_gram_arm_reads_the_prior_at_the_fixed_seed_deterministically():
    from test_tul_gram import _gram
    m = _gram()
    seen: list[dict] = []
    real = m.tul_forward_ablated

    def spy(*a, **kw):
        seen.append({k: kw.get(k) for k in ("gram_mode", "gram_sample_seed", "plan_mode")})
        return real(*a, **kw)

    m.tul_forward_ablated = spy
    batches = _batches()
    r1, a1 = sp.superposition_probe(m, batches, "cpu", seed=0, n_boot=50)
    r2, a2 = sp.superposition_probe(m, batches, "cpu", seed=0, n_boot=50)
    r3, a3 = sp.superposition_probe(m, batches, "cpu", seed=5, n_boot=50)
    assert all(s["gram_mode"] == "prior" for s in seen)
    assert {s["gram_sample_seed"] for s in seen} == {0, 5}
    assert r1["gram_prior_seed"] == 0 and r3["gram_prior_seed"] == 5
    for k in a1:
        assert np.array_equal(a1[k], a2[k]), k
    assert not np.array_equal(a1["ce_own"], a3["ce_own"])     # another seed, other noise


def test_a_fan_all_model_moves_all_k_cells_of_a_slot():
    from test_tul_fan import _batch as _fan_batch, _model as _fan_model
    torch.manual_seed(0)
    m = _fan_model(fan_k=4, slot_cells=4, prefix_k=4, fan_mix="all", fan_select_eps=0.0)
    _ids, inp, lab, layout = _fan_batch(prefix_k=4)
    assert inp.shape[0] >= 2
    shapes: list[tuple] = []
    real = sp.controlled_shuffle

    import contextlib

    @contextlib.contextmanager
    def _record(model, sr, ss):
        with real(model, sr, ss) as calls:
            yield calls
        shapes.extend(calls["shapes"])

    sp_cs, sp.controlled_shuffle = sp.controlled_shuffle, _record
    try:
        res, arr = sp.superposition_probe(m, [(inp, lab, layout, _idx(layout))], "cpu",
                                          seed=0, n_boot=50)
    finally:
        sp.controlled_shuffle = sp_cs
    S = layout.slot_valid.shape[1]
    assert shapes and all(s[:3] == (inp.shape[0], S, 4) for s in shapes), shapes
    assert res["n_spans"] == len(_expected_spans(layout, lab))
    assert np.isfinite(arr["ce_xrow"]).all()


def test_the_context_bucket_ranks_t0_under_the_xrow_cell_not_the_own_cell():
    """The own cell ranks span s's t0 in bucket ``s % 3``; a foreign cell ranks it in
    ``(s + 1) % 3``. So context bucket j holds exactly the own-bucket ``(j - 1) % 3``
    spans, and its worth is that own bucket's planted worth; the ratio and verdict follow."""
    hi, lo = (3.0, 3.0, 3.0), (0.0, 1.0, 2.5)
    m, res, arr, _ = _run_planted(hi, lo, ctx_bucket_of=lambda s: (s + 1) % 3)
    want = [_ce_of(lo[k]) - _ce_of(hi[k]) for k in range(3)]
    for j, b in enumerate(sp.BUCKETS):
        d = res["worth"]["xrow"]["pos1plus_ctx"][b]
        assert d["n"] > 0, b
        assert d["mean"] == pytest.approx(want[(j - 1) % 3], abs=1e-5), b
        # the own bucket is unchanged by the foreign ranking
        assert res["worth"]["xrow"]["pos1plus"][b]["mean"] == pytest.approx(want[j], abs=1e-5)
        assert res["buckets_ctx"][b]["own_top1_share"] == pytest.approx(
            1.0 if (j - 1) % 3 == 0 else 0.0)
    r = res["ratio_top2_top1_ctx"]["xrow"]["point"]
    assert r == pytest.approx(want[0] / want[2], rel=1e-4)     # ctx top2 = own top1
    assert res["decision"]["verdict"] == "superposition"
    assert np.array_equal(arr["bucket_ctx"], (arr["bucket"] + 1) % 3)
