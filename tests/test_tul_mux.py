"""MUX local head + configurable §5 weights (arm v1a).

Pre-registration: lab/experiments/planned/2026-08-25-mux-head-arm-v1a.md.
Contracts protected here, chosen so each test FAILS if the shipped code breaks:
the geometric target weights are normalised per span and decay at exactly rho;
span 0 and unterminated trailing text supervise nothing; ``mux_beta=0`` is
bit-identical to the default build; the composite loss decomposes as
CE + mux_weighted; ``emit_weight=0`` really removes the emit position's gradient
(corrupting its labels changes nothing); and the mux gradient reaches the slot
path. CPU only, tiny config, no tokenizer (the test_tul_forward pattern).
"""

from __future__ import annotations

import numpy as np
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, mux_span_targets
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=128, context_len=128,
        n_prelude=1, n_core=2, n_coda=1, mean_depth=2, max_depth=3, bptt_depth=2,
        channel_dims=(32, 20, 12), compression=2, csa_compress_ratio=4,
        hca_compress_ratio=8, top_k=8, window_size=16,
        retention=False, bigram_hash_vocab=V, use_kernels=False, hc_use_kernel=False,
        dropout=0.0,
    )
    base.update(kw)
    return MORPHConfig(**base)


def _rule() -> BoundaryRule:
    lut = np.zeros(V, dtype=bool)
    lut[[DOT, 11]] = True
    lut[0] = True
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=8, eos_id=0)


def _spec(**kw) -> TulLayoutSpec:
    base = dict(seq_len=32, prefix_k=2, max_slots=5, slot_id=4)
    base.update(kw)
    return TulLayoutSpec(**base)


def _batch(spec, B=2, n=90, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == spec.slot_id] = 5
    ids[:, ::6] = DOT
    return slot_layout_from_ids(ids.astype(np.int64), _rule(), spec)


def _model(tul: TULConfig | None, seed=1234, **cfg_kw) -> MORPHTransformer:
    torch.manual_seed(seed)
    return MORPHTransformer(_tiny(tul=tul, **cfg_kw))


RHO = 0.9


# ── mux_span_targets: the target construction alone ──────────────────────────

def test_mux_targets_geometric_and_normalised_per_span():
    x, y, layout, _ = _batch(_spec())
    pos_valid, alpha, tgt_slot, sup = mux_span_targets(x, layout, RHO)
    B, L = x.shape
    S = layout.slot_index.shape[1]
    checked_spans = 0
    for b in range(B):
        for s in range(S):
            ps = [p for p in range(L)
                  if bool(pos_valid[b, p]) and int(tgt_slot[b, p]) == s]
            assert bool(sup[b, s]) == (len(ps) > 0)
            if not ps:
                continue
            checked_spans += 1
            # contiguous span, weights sum to 1, exact geometric ratio
            assert ps == list(range(ps[0], ps[0] + len(ps)))
            a = alpha[b, ps]
            assert torch.allclose(a.sum(), torch.tensor(1.0), atol=1e-5)
            for j in range(1, len(ps)):
                assert torch.allclose(a[j] / a[j - 1], torch.tensor(RHO), atol=1e-5)
            # the span really is span s+1: it starts right after slot s's positions
            start = int(layout.slot_index[b, s]) + layout.prefix_k
            assert ps[0] == start
    assert checked_spans > 0, "batch produced no supervised span — fixture broken"


def test_mux_targets_span0_and_trailing_supervise_nothing():
    x, y, layout, _ = _batch(_spec())
    pos_valid, alpha, tgt_slot, sup = mux_span_targets(x, layout, RHO)
    B, L = x.shape
    S = layout.slot_index.shape[1]
    # span 0 (bag_id == 0) has no preceding slot: never a valid position
    assert not bool((pos_valid & (layout.bag_id == 0)).any())
    # tokens in the dump bin (past the last slot / pads) never supervise
    assert not bool((pos_valid & (layout.bag_id >= S)).any())
    # a slot whose next span is unterminated is unsupervised: for every supervised
    # slot s, slot s+1 (terminating span s+1) must be valid
    for b in range(B):
        for s in range(S):
            if bool(sup[b, s]):
                assert bool(layout.slot_valid[b, s]) and bool(layout.slot_valid[b, s + 1] if s + 1 < S else False)


# ── the head in the forward ──────────────────────────────────────────────────

def _loss(m, x, y, layout, seed=7):
    torch.manual_seed(seed)   # pins the Poisson depth draw
    m.eval()                  # dropout off; graph still built
    return m(x, labels=y, slot_layout=layout)


def test_mux_beta_zero_is_bit_identical_to_default():
    x, y, layout, _ = _batch(_spec())
    m0 = _model(TULConfig(slot_id=4))
    m1 = _model(TULConfig(slot_id=4, mux_beta=0.0))
    o0, o1 = _loss(m0, x, y, layout), _loss(m1, x, y, layout)
    assert torch.equal(o0["loss"], o1["loss"])
    assert "mux_local" not in o1


def test_mux_loss_decomposes_as_ce_plus_weighted_term():
    x, y, layout, _ = _batch(_spec())
    m0 = _model(TULConfig(slot_id=4))
    m1 = _model(TULConfig(slot_id=4, mux_beta=0.7))
    o0, o1 = _loss(m0, x, y, layout), _loss(m1, x, y, layout)
    assert "mux_local" in o1 and torch.isfinite(o1["mux_local"])
    assert float(o1["mux_local"]) > 0.0
    assert torch.allclose(o1["mux_weighted"], 0.7 * o1["mux_local"], atol=1e-6)
    assert torch.allclose(o1["loss"] - o1["mux_weighted"], o0["loss"], atol=1e-5)


def test_emit_weight_zero_removes_the_emit_gradient():
    """Corrupting every emit-position label must not move the loss when
    emit_weight=0 — and MUST move it at the 0.5 default (the control that the
    corruption itself is real)."""
    x, y, layout, _ = _batch(_spec())
    base = layout.slot_index + layout.prefix_k - 1          # emit positions [B, S]
    y_bad = y.clone()
    for b in range(y.shape[0]):
        for s in range(layout.slot_index.shape[1]):
            if bool(layout.slot_valid[b, s]):
                p = int(base[b, s])
                if y_bad[b, p] != -100:
                    y_bad[b, p] = (int(y_bad[b, p]) + 7) % V or 5
    m_off = _model(TULConfig(slot_id=4, emit_weight=0.0, plast_weight=1.0))
    m_def = _model(TULConfig(slot_id=4))
    assert torch.equal(_loss(m_off, x, y, layout)["loss"],
                       _loss(m_off, x, y_bad, layout)["loss"])
    assert not torch.equal(_loss(m_def, x, y, layout)["loss"],
                           _loss(m_def, x, y_bad, layout)["loss"])


def test_mux_gradient_reaches_the_slot_path():
    """Same init, same RNG: the ONLY difference between the models is mux_beta,
    so any E_slot grad difference is the mux head's gradient reaching the slot
    path. The beta=0 pair is the control that the RNG pinning is real."""
    x, y, layout, _ = _batch(_spec())

    def grad(beta, seed=1234):
        m = _model(TULConfig(slot_id=4, mux_beta=beta), seed=seed)
        out = _loss(m, x, y, layout)
        out["loss"].backward()
        return m.tul.E_slot.grad.clone()

    g0a, g0b, g1 = grad(0.0), grad(0.0), grad(1.0)
    assert torch.equal(g0a, g0b), "RNG pinning broken — the comparison below is void"
    assert not torch.allclose(g0a, g1), "mux gradient never reached E_slot"


# ── mux_detach_head: the auxiliary must not train the tied embedding table ────

def _mux_only_embed_grad(detach: bool, x, y, layout, seed=1234):
    """Grad on the euclidean embedding table from the MUX loss ALONE.

    Runs the real front/core/head methods (not a stand-in) so the test breaks if
    the shipped path changes. The CE is deliberately NOT included: it touches
    every vocabulary row through the fused head, which would mask the very
    difference under test.
    """
    m = _model(TULConfig(slot_id=4, mux_beta=1.0, mux_detach_head=detach), seed=seed)
    m.eval()
    torch.manual_seed(7)
    xx, x0, bg = m._tul_front(x, layout)
    _xn, h_slots, _d, _g = m._tul_core(xx, x0, bg, layout)
    m._tul_mux_loss(h_slots, x, layout).backward()
    return m.embed.hybrid.euc_embed.weight.grad.clone()


def test_detached_head_leaves_absent_vocab_rows_untouched():
    """`lm_weight()` is WEIGHT-TIED to the input embeddings, so an undetached head
    trains the embedding table on the auxiliary target — the v1a divergence.

    The readout term puts gradient on EVERY vocabulary row (softmax is dense); the
    legitimate path — mux → z → core → slot input `mean(embed(span))` — can only
    reach rows whose token is actually in the batch. Rows absent from the batch are
    therefore an exact discriminator, not a magnitude heuristic.
    """
    x, y, layout, _ = _batch(_spec())
    present = set(int(v) for v in x.unique())
    absent = [v for v in range(5, V) if v not in present and v != 4]
    assert len(absent) >= 5, "fixture has no absent vocabulary rows to test with"

    g_det = _mux_only_embed_grad(True, x, y, layout)
    g_raw = _mux_only_embed_grad(False, x, y, layout)
    # the control: the undetached head DOES reach rows the batch never used
    assert g_raw[absent].abs().max() > 0, "undetached head did not reach absent rows — test void"
    # the contract: the detached head does not
    assert torch.equal(g_det[absent], torch.zeros_like(g_det[absent]))
    # and it still trains the plan: present rows keep gradient through the slot input
    assert g_det.abs().max() > 0, "detached head produced NO embedding gradient at all"


def test_detach_default_is_on():
    assert TULConfig(slot_id=4, mux_beta=1.0).mux_detach_head is True
