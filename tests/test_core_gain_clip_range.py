"""The core-gain governor must apply to exactly the loop iterations it is told to.

Why the range exists: measured on the divergent TUL control, the realized per-iteration
gain is 1.422 at t=0 and 1.08-1.13 at t=1..7
(docs/experiments/results/2026-08-23-tul-onset-ordering.md), so a typical cap of 1.5 can
only ever bind on the FIRST iteration. Every previous run applied the cap to every
iteration and read the result as "the governor cures it". These tests make the range a
contract so the mediation arms in
docs/experiments/planned/2026-08-23-tul-iteration0-mediation.md mean what they say.

CPU only, tiny config, no tokenizer — mirrors tests/test_tul_forward.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=3, max_depth=4, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _model(tul, seed=1234, **kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **kw))


def _batch(B=2, n=90, seed=0):
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    rule = BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)
    spec = TulLayoutSpec(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), rule, spec)


# ── the range predicate itself ───────────────────────────────────────────────

@pytest.mark.parametrize(
    "lo,hi,expect",
    [
        (0, -1, [True, True, True, True]),     # the default: every iteration
        (0, 0, [True, False, False, False]),   # arm M0: the first iteration only
        (1, -1, [False, True, True, True]),    # arm M1: everything but the first
        (1, 2, [False, True, True, False]),    # a closed interval, both ends inclusive
        (9, -1, [False, False, False, False]),  # a range past the loop clips nothing
    ],
)
def test_clip_range_selects_the_named_iterations(lo, hi, expect):
    m = _model(None, core_gain_clip=1.5, core_gain_clip_iter_lo=lo, core_gain_clip_iter_hi=hi)
    assert [m._clip_applies(t) for t in range(4)] == expect


# ── behaviour: the range must actually change the forward ────────────────────

def _loss(lo, hi, tau=1.0):
    """Forward one TUL batch under a given clip range. tau is deliberately tight (1.0)
    so the cap BINDS at every iteration it is allowed to touch — a cap that never binds
    would make every arm agree and the test would pass on broken wiring."""
    tul = TULConfig(prefix_k=2, slot_id=4)
    m = _model(tul, core_gain_clip=tau, core_gain_clip_iter_lo=lo, core_gain_clip_iter_hi=hi)
    m.eval()  # deterministic depth, no dropout
    x, y, layout, _ = _batch()
    torch.manual_seed(7)
    return m(x, labels=y, slot_layout=layout)["loss"].detach()


def test_clipping_only_the_first_iteration_differs_from_clipping_the_rest():
    """M0 and M1 must not be the same model. If the range were ignored, both would equal
    the clip-everything arm and the mediation experiment would compare nothing."""
    first_only = _loss(0, 0)
    rest_only = _loss(1, -1)
    everything = _loss(0, -1)
    assert not torch.equal(first_only, rest_only), "the clip range did not reach the loop"
    assert not torch.equal(first_only, everything)
    assert not torch.equal(rest_only, everything)


def test_a_range_past_the_loop_is_the_same_as_no_clip():
    """lo beyond max_depth clips nothing, so it must reproduce tau=0 exactly."""
    off = _loss(0, -1, tau=0.0)
    past_end = _loss(99, -1, tau=1.0)
    assert torch.equal(off, past_end)


def test_the_default_range_is_the_un_ranged_behaviour():
    """(0, -1) must be every iteration — i.e. adding these keys changed nothing for
    every config that does not set them."""
    m = _model(None, core_gain_clip=1.5)
    assert m.cfg.core_gain_clip_iter_lo == 0
    assert m.cfg.core_gain_clip_iter_hi == -1
    assert all(m._clip_applies(t) for t in range(8))


def test_clip_off_ignores_the_range_entirely():
    """tau=0 is OFF and must stay bit-identical to baseline whatever the range says."""
    a = _loss(0, 0, tau=0.0)
    b = _loss(1, -1, tau=0.0)
    assert torch.equal(a, b)
