"""`tul.row_contrast_lambda` — the WITHIN-ROW CONTRASTIVE objective (lever C2, 2026-09-13).

THE DEFECT. On the strict ruler a row's 64 written slot states sit at effective rank
5.7598 in 1024 dimensions with mean pairwise cosine 0.7104. Every lever so far attacked
the LOOP. This one attacks the JOB: for each valid slot `i` whose next span exists, span
`i+1`'s pooled prelude states must pick slot `i`'s exit state out of the row's other valid
slots. Indistinguishable states cannot do it, and `tul/row_contrast_acc` says by how much.

WHAT THIS FILE HAS TO PROVE:

1. OFF IS NOTHING. `row_contrast_lambda: 0` builds no head, draws no RNG, adds no key and
   leaves the forward bit-identical.
2. THE TERM IS REAL AND IT IS IN THE LOSS. Positive, reported, and `row_contrast_weighted`
   is exactly `lambda * row_contrast` and is what the loss grew by.
3. THE CHANCE VALUE IS EXACT. With every slot state identical the term is `log n_valid`
   and the accuracy is `1/n_valid` — the absolute reference the reported number rests on.
4. THE NEGATIVES ARE THE ROW'S OTHER VALID SLOTS AND NOTHING ELSE. A pad is never a key;
   the LAST valid slot is never an anchor (its "next span" is the dump bin, an exact zero);
   and perturbing row 1 leaves row 0's term bit-identical.
5. THE TARGETS ARE DETACHED. The prelude receives NO gradient through the pooled targets.
6. THE ANCHOR PATH IS LIVE. The gradient reaches the core.
7. NO NaN, including on a row that carries no anchor at all.
8. THE REFUSALS: the paid loop, an FM planner, `n_core: 0`, a non-positive temperature.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-rank-levers-center-and-contrast.md
Note: .agents/notes/proposed/architecture/2026-09-13-rank-levers-center-and-contrast.md
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig, TULRowContrast, next_span_pool
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids

V = 64
DOT = 10
LAM = 0.1


def _tiny(**kw) -> MORPHConfig:
    base = dict(
        d_model=64, n_heads=2, n_kv_heads=2, vocab_size=V, max_seq_len=256, context_len=256,
        n_prelude=2, n_core=2, n_coda=2, mean_depth=3, max_depth=4, bptt_depth=4,
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
    return BoundaryRule(is_boundary=lut, min_span=4, span_cap=32, eos_id=0)


def _ids(B: int = 2, n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[:, ::8] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=8,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(M: int = 1, seed: int = 0):
    """`max_slots` 10 with a short token budget, so the row carries PAD slots."""
    spec = TulLayoutSpec(seq_len=64, prefix_k=max(M, 2), max_slots=10, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(_ids(seed=seed), _rule(), spec)
    return inp, lab, layout


def _model(M: int = 1, lam: float = 0.0, seed: int = 99, **tul_kw):
    kw = dict(prefix_k=max(M, 2), row_contrast_lambda=lam)
    if M > 1:
        kw.update(slot_cells=M, slot_cell_init="distinct")
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


# ── 1. OFF IS NOTHING ────────────────────────────────────────────────────────

def test_off_builds_no_head():
    m = _model(1, lam=0.0)
    assert m.cfg.tul.row_contrast_lambda == 0.0
    assert m.tul_contrast is None
    assert not any("tul_contrast" in k for k in m.state_dict())


def test_on_adds_exactly_one_tensor_and_moves_no_base_weight():
    """`W_contrast` draws from a PRIVATE generator with the global stream snapshotted and
    restored, so a contrast arm's base weights are byte-identical to its ruler's."""
    a, b = _model(1, lam=0.0), _model(1, lam=LAM)
    ka, kb = set(a.state_dict()), set(b.state_dict())
    assert kb - ka == {"tul_contrast.W_contrast.weight"}
    assert ka - kb == set()
    for k in sorted(ka):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k]), (
            f"{k} differs — building TULRowContrast drew from the GLOBAL RNG stream")


def test_the_head_does_not_depend_on_the_ambient_rng():
    a, b = _model(1, lam=LAM, seed=1), _model(1, lam=LAM, seed=2)
    assert torch.equal(a.state_dict()["tul_contrast.W_contrast.weight"],
                       b.state_dict()["tul_contrast.W_contrast.weight"])
    assert not torch.equal(a.state_dict()["embed.hybrid.euc_embed.weight"],
                           b.state_dict()["embed.hybrid.euc_embed.weight"]), (
        "the ambient seed changed nothing — the fixture proves nothing")


def test_off_reports_no_key_and_the_forward_is_unchanged():
    inp, lab, layout = _batch()
    a, b = _model(1, lam=0.0), _model(1, lam=LAM)
    torch.manual_seed(3)
    oa = a(inp, labels=lab, slot_layout=layout)
    assert "row_contrast" not in oa and "row_contrast_weighted" not in oa
    torch.manual_seed(3)
    ob = b(inp, labels=lab, slot_layout=layout)
    # The MODEL's own forward is untouched: the term is an addition, not a change of
    # forward. `spandec` reads the same seam the term reads, so it is the sharp witness.
    assert float(oa["spandec"]) == float(ob["spandec"])


# ── 2. THE TERM IS REAL AND IT IS IN THE LOSS ────────────────────────────────

def test_the_term_is_positive_and_weighted_into_the_loss():
    inp, lab, layout = _batch()
    a, b = _model(1, lam=0.0), _model(1, lam=LAM)
    torch.manual_seed(3)
    la = float(a(inp, labels=lab, slot_layout=layout)["loss"])
    torch.manual_seed(3)
    ob = b(inp, labels=lab, slot_layout=layout)
    rc, rw = float(ob["row_contrast"]), float(ob["row_contrast_weighted"])
    assert rc > 0.0
    assert abs(rw - LAM * rc) < 1e-6, "row_contrast_weighted is not lambda * row_contrast"
    assert abs(float(ob["loss"]) - (la + rw)) < 1e-5, (
        "the loss did not grow by exactly the weighted term — the auxiliary is either "
        "missing from the loss or it changed the model's own CE")


def test_the_readouts_are_reported():
    inp, lab, layout = _batch()
    out = _model(1, lam=LAM)(inp, labels=lab, slot_layout=layout)
    for k in ("row_contrast", "row_contrast_weighted", "row_contrast_acc",
              "row_contrast_n_rows", "row_contrast_n_anchors"):
        assert k in out and out[k] is not None, f"{k} is not reported"
    n_rows, n_anch = float(out["row_contrast_n_rows"]), float(out["row_contrast_n_anchors"])
    assert n_rows >= 1 and n_anch >= 2 * n_rows
    assert 0.0 <= float(out["row_contrast_acc"]) <= 1.0


# ── 3. THE CHANCE VALUE IS EXACT ─────────────────────────────────────────────

@pytest.mark.parametrize("n", [3, 6, 9])
def test_identical_states_give_exactly_log_n_and_accuracy_one_over_n(n):
    """The absolute reference. With every slot state identical the span -> z softmax is
    uniform over the row's valid slots, whatever the SPAN pools look like."""
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    S = n + 2
    z = torch.randn(1, 1, 8).expand(1, S, 8).contiguous()     # every slot the same
    pool = torch.randn(1, S, 8)                               # spans stay distinct
    ok = torch.zeros(1, S, dtype=torch.bool)
    ok[0, :n] = True
    loss, acc, n_rows = con(z, pool, ok)
    assert abs(float(loss) - math.log(n)) < 1e-5, (
        f"term {float(loss):.6f} != log({n}) = {math.log(n):.6f} — the chance value the "
        f"reported number is read against does not hold")
    assert abs(float(acc) - 1.0 / n) < 1e-5
    assert float(n_rows) == 1.0


def test_a_perfectly_aligned_row_reads_near_zero():
    """The other end of the scale, so "log n" is not passing by accident."""
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    with torch.no_grad():
        con.W_contrast.weight.copy_(torch.eye(8))
    z = torch.randn(1, 5, 8)
    loss, acc, _n = con(z, z.clone(), torch.ones(1, 5, dtype=torch.bool))
    assert float(loss) < 0.05 and float(acc) == 1.0


def test_a_row_with_one_anchor_is_excluded_not_scored_as_zero():
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    ok = torch.zeros(2, 5, dtype=torch.bool)
    ok[0, :4] = True        # 4 anchors
    ok[1, :1] = True        # 1 anchor — would score an unearned exact 0
    z = torch.randn(2, 1, 8).expand(2, 5, 8).contiguous()
    loss, _acc, n_rows = con(z, torch.randn(2, 5, 8), ok)
    assert float(n_rows) == 1.0
    assert abs(float(loss) - math.log(4)) < 1e-5, (
        "the single-anchor row was averaged in and pulled the term below chance")


# ── 4. THE NEGATIVES ARE THE ROW'S OTHER VALID SLOTS AND NOTHING ELSE ────────

def test_a_pad_slot_is_never_a_key():
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    ok = torch.zeros(1, 6, dtype=torch.bool)
    ok[0, :4] = True
    z = torch.randn(1, 6, 8)
    pool = torch.randn(1, 6, 8)
    a, _acc, _n = con(z, pool, ok)
    z2 = z.clone()
    z2[0, 4:] = 1e3                      # garbage at the two pad slots
    b, _acc2, _n2 = con(z2, pool, ok)
    assert torch.equal(a.detach(), b.detach()), (
        "moving a PAD slot's state changed the term — pads are entering the softmax as "
        "keys")


def test_perturbing_one_row_leaves_the_other_rows_term_unchanged():
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    ok = torch.ones(2, 5, dtype=torch.bool)
    z = torch.randn(2, 5, 8)
    pool = torch.randn(2, 5, 8)

    def row0(zz):
        one = torch.zeros(2, 5, dtype=torch.bool)
        one[0] = True
        # score row 0 alone by masking row 1 out entirely
        l_, _a, _n = con(zz, pool, one)
        return float(l_)

    before = row0(z)
    z2 = z.clone()
    z2[1] = torch.randn(5, 8) * 5.0
    assert abs(row0(z2) - before) < 1e-6, "row 1's states reached row 0's softmax"
    # ...and the full-batch term DOES move, so the fixture is not vacuous.
    assert float(con(z, pool, ok)[0]) != float(con(z2, pool, ok)[0])


def test_next_span_pool_is_the_next_span_and_the_last_slot_is_not_an_anchor():
    inp, _lab, layout = _batch()
    tok = torch.arange(inp.shape[1], dtype=torch.float32)
    sig = tok.reshape(1, -1, 1).expand(inp.shape[0], -1, 1).contiguous()
    pool, ok = next_span_pool(sig, layout)
    S = layout.slot_valid.shape[1]
    assert pool.shape == (inp.shape[0], S, 1) and ok.shape == (inp.shape[0], S)
    for b in range(inp.shape[0]):
        nv = int(layout.slot_valid[b].sum())
        assert bool(ok[b, : nv - 1].all()), "a valid slot with a valid successor is not ok"
        assert not bool(ok[b, nv - 1:].any()), (
            "the LAST valid slot is an anchor — its next span is the dump bin, whose "
            "bag_mean row is defined to be exactly 0")
        # The pool really is the NEXT span: entry i is the mean position index of the
        # tokens whose bag is i+1, which is strictly greater than slot i's own span mean.
        own = pool[b, : max(nv - 2, 0)]
        nxt = pool[b, 1: max(nv - 1, 1)]
        assert bool((nxt > own).all()), "the pooled target is not the NEXT span"


def test_a_row_with_no_anchor_is_finite_and_contributes_nothing():
    """The all-masked softmax row is where the register's NaN was born. A finite fill, not
    -inf, is what keeps this finite in the BACKWARD too."""
    torch.manual_seed(0)
    con = TULRowContrast(8, tau=0.1)
    z = torch.randn(1, 4, 8, requires_grad=True)
    loss, acc, n_rows = con(z, torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.bool))
    assert float(loss) == 0.0 and float(acc) == 0.0 and float(n_rows) == 0.0
    loss.backward()
    assert z.grad is not None and bool(torch.isfinite(z.grad).all())


# ── 5. THE TARGETS ARE DETACHED ──────────────────────────────────────────────

def test_the_pooled_targets_carry_no_gradient_to_the_prelude():
    """The isolating form: `x` and `h_slots` are independent leaves, so ANY gradient on
    `x` can only have come through the pooled-target path."""
    m = _model(1, lam=LAM)
    inp, _lab, layout = _batch()
    B, L = inp.shape
    S = layout.slot_valid.shape[1]
    n, C = m._n_streams, m.cfg.d_model
    x = torch.randn(B, L, n, C, requires_grad=True)
    h = torch.randn(B, S, n, C, requires_grad=True)
    m._tul_row_contrast_loss(h, x, layout).backward()
    assert h.grad is not None and float(h.grad.abs().sum()) > 0.0, (
        "the ANCHOR path carries no gradient — the term trains nothing")
    assert x.grad is None or float(x.grad.abs().sum()) == 0.0, (
        f"the prelude received gradient {float(x.grad.abs().sum()):.3e} through the "
        f"pooled targets — they are NOT detached, and this term is reshaping the "
        f"representation it is supposed to be scored against")


def test_the_target_is_detached_BEFORE_the_readout():
    """WHERE the detach sits, not just that one exists. `_readout` is `lm_mixer` +
    `final_norm`, both TRAINED and both shared with the LM head, so detaching after it
    would let the target path reshape them (`mux_detach_head`'s subject: the tied table is
    the input embedding table, and arm v1a diverged at step 2800 over exactly this).

    The anchor's readout must be LIVE and the target's must not, so the probe records the
    `requires_grad` of each call's input, in order."""
    m = _model(1, lam=LAM)
    inp, _lab, layout = _batch()
    B, L = inp.shape
    S = layout.slot_valid.shape[1]
    n, C = m._n_streams, m.cfg.d_model
    x = torch.randn(B, L, n, C, requires_grad=True)
    h = torch.randn(B, S, n, C, requires_grad=True)
    seen: list[bool] = []
    orig = m._readout

    def spy(t):
        seen.append(bool(t.requires_grad))
        return orig(t)

    m._readout = spy
    try:
        m._tul_row_contrast_loss(h, x, layout)
    finally:
        m._readout = orig
    assert seen == [True, False], (
        f"readout inputs were {seen}, expected [anchor live, target detached] — a target "
        f"that reaches the readout trains `lm_mixer` and `final_norm` from the target "
        f"side")


# ── 6. THE ANCHOR PATH IS LIVE, END TO END ───────────────────────────────────

def test_the_term_alone_reaches_the_core():
    m = _model(1, lam=LAM)
    inp, _lab, layout = _batch()
    _fkw, _fr, _ckw, _cr = m._tul_tg_kwargs(layout)
    x, x0, bg = m._tul_front(inp, layout, attn_kwargs=_fkw, ret_reset_mask=_fr)
    _xn, h_slots, *_ = m._tul_core(x, x0, bg, layout, input_ids=inp)
    m._tul_row_contrast_loss(h_slots, x, layout).backward()
    core = [p for n_, p in m.named_parameters() if n_.startswith("core.")]
    assert core, "fixture has no core parameters"
    g = sum(float(p.grad.abs().sum()) for p in core if p.grad is not None)
    assert g > 0.0, "the contrastive term does not reach the looped core"


def test_no_parameter_reads_a_nan_gradient():
    for M in (1, 4):
        m = _model(M, lam=LAM)
        inp, lab, layout = _batch(M)
        m(inp, labels=lab, slot_layout=layout)["loss"].backward()
        bad = [n_ for n_, p in m.named_parameters()
               if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
        assert bad == [], f"M={M}: non-finite gradient at {bad}"


def test_it_reads_the_cells_mean_at_m4():
    """At M > 1 the term takes the same state every other reader at that seam takes. It
    must therefore RUN, report, and move with the register's cells."""
    inp, lab, layout = _batch(4)
    out = _model(4, lam=LAM)(inp, labels=lab, slot_layout=layout)
    assert float(out["row_contrast"]) > 0.0
    assert float(out["row_contrast_n_anchors"]) > 0.0


# ── 7. THE REFUSALS ──────────────────────────────────────────────────────────

def test_row_contrast_refuses_the_paid_loop():
    with pytest.raises(ValueError, match="row_contrast_lambda"):
        TULConfig(prefix_k=2, slot_id=4, row_contrast_lambda=LAM,
                  tokens_through_core=True)


def test_row_contrast_refuses_a_negative_lambda_and_a_nonpositive_tau():
    with pytest.raises(ValueError, match="row_contrast_lambda"):
        TULConfig(prefix_k=2, slot_id=4, row_contrast_lambda=-1.0)
    with pytest.raises(ValueError, match="row_contrast_tau"):
        TULConfig(prefix_k=2, slot_id=4, row_contrast_lambda=LAM, row_contrast_tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        TULRowContrast(8, tau=0.0)


def test_row_contrast_refuses_a_zero_core():
    with pytest.raises(ValueError, match="row_contrast_lambda"):
        MORPHTransformer(_tiny(n_core=0, tul=_tul(row_contrast_lambda=LAM)))


def test_row_contrast_refuses_an_fm_planner():
    from morph.model.tul_fm import FMArmConfig

    tc = TULConfig(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                   mux_beta=0.0, row_contrast_lambda=LAM)
    with pytest.raises(NotImplementedError, match="row_contrast_lambda"):
        MORPHTransformer(_tiny(n_core=0, tul=tc,
                               fm=FMArmConfig(d_p=16, n_layers=1, n_heads=2, d_ff=32,
                                              cond_dim=16, max_slots=10, l_total=84)))


def test_row_contrast_composes_with_center_exit():
    """The two levers are independent and must run together: C1 is upstream of this seam,
    so the term scores the CENTERED state."""
    inp, lab, layout = _batch()
    a = _model(1, lam=LAM, center_exit=False)
    b = _model(1, lam=LAM, center_exit=True)
    torch.manual_seed(3)
    ra = float(a(inp, labels=lab, slot_layout=layout)["row_contrast"])
    torch.manual_seed(3)
    rb = float(b(inp, labels=lab, slot_layout=layout)["row_contrast"])
    assert ra != rb, "the contrastive term did not see the centering"
