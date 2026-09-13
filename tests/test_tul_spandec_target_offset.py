"""`tul.spandec_target_offset` — WHICH span the span decoder decodes.

THE DEFECT. Every span-decoder arm in the arc has graded z on span s+1. The 2026-08-14
JEPA screen said the informative target sits TWO TO THREE spans downstream, and the
measured cross-span budget (lab/experiments/failures/2026-09-11-arc-span-budget.md) is a
0.958-nat SPIKE at a span's first position on top of a FLAT 0.315 at every offset eight or
more tokens in. A next-span target is dominated by the spike.

THIS IS NOT `spandec_horizon`, and that distinction is the thing this file pins. Horizon
WIDENS: H = 3 decodes spans s+1, s+2 AND s+3 as one concatenated run, so the next span is
still in the target (arm `slot-spandec-strict-h3`, 2026-09-12). Offset MOVES: k = 3 at
H = 1 decodes span s+3 ALONE and the next span leaves the objective. Only the first has
ever run.

WHAT THIS FILE HAS TO PROVE:

1. OFFSET 1 IS NOTHING. The default is bit-identical to the tree before the key, measured
   (the shared pin in `tests/test_tul_spandec_reads_cells.py`), and setting the key to its
   default explicitly changes no tensor and no number.
2. THE TARGETS ARE THE RIGHT SPAN'S TOKENS. Checked against an INDEPENDENT oracle that
   walks `layout.bag_id` and collects each span's token positions in order — not against
   `span_slots` at another shift, which would only pin the function to itself.
3. THE LAST k-1 SLOTS ARE MASKED. They have no span s+k in the row, so every one of their
   labels is `ignore_index`, exactly as a pad slot's is.
4. THE SUPERVISED TOKEN COUNT MATCHES the oracle's, exactly.
5. HORIZON AND OFFSET COMPOSE. Offset k at horizon H is spans s+k .. s+k+H-1, block by
   block, and H = 1 / k = 1 is `next_span_slots` tensor for tensor.
6. THE REFUSALS. k < 1, k at or past the row's slot budget, k != 1 with `spandec: false`,
   and k != 1 with each mechanism that defines "the next span" for itself.

CPU only, fp32, `use_kernels=False`, tiny config.

Record: lab/experiments/planned/2026-09-13-arc-register-reader-and-downstream-target.md
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morph.model.transformer import MORPHConfig, MORPHTransformer
from morph.model.tul import TULConfig
from morph.model.tul_layout import BoundaryRule, TulLayoutSpec, slot_layout_from_ids
from morph.model.tul_spandec import horizon_span_slots, next_span_slots, span_slots

V = 64
DOT = 10
J = 8


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


def _ids(B: int = 2, n: int = 160, seed: int = 0) -> np.ndarray:
    """RAGGED spans, and that is load-bearing.

    An every-8th-token boundary makes every span the same length, and then a target that
    lands on the WRONG slot is silently overwritten by the next span at the same offsets:
    sabotage R2-b (`k >= shift` weakened to `k >= 1`, which sends span 1's tokens to slot
    0 at every offset) MISSED on a uniform fixture and is caught on this one. Span lengths
    here run 5, 7, 11, 6, 9, ... with `min_span` 4 respected.
    """
    rng = np.random.default_rng(seed)
    ids = rng.integers(5, V, size=(B, n))
    ids[ids == 4] = 5
    ids[ids == DOT] = 5
    ids[ids == 11] = 5
    lens = [5, 7, 11, 6, 9, 5, 13, 6, 8, 5, 10, 7, 6, 12, 5, 9]
    for b in range(B):
        p = (b * 3) % 4                      # rows do not share a boundary pattern
        i = 0
        for step in lens * 4:
            i += step + (p % 3)
            if i >= n:
                break
            ids[b, i] = DOT
    return ids.astype(np.int64)


def _tul(**kw) -> TULConfig:
    base = dict(prefix_k=2, slot_id=4, emit_weight=0.0, token_state_dropout=0.0,
                mux_beta=0.0, spandec=True, spandec_layers=1, spandec_max_tokens=J,
                tg_restrict=True, tg_restrict_scope="all", tg_geometry="strict")
    base.update(kw)
    return TULConfig(**base)


def _batch(seed: int = 0, max_slots: int = 10):
    spec = TulLayoutSpec(seq_len=64, prefix_k=2, max_slots=max_slots, slot_id=4)
    inp, lab, layout, _ = slot_layout_from_ids(_ids(seed=seed), _rule(), spec)
    return inp, lab, layout


def _model(offset: int = 1, seed: int = 99, **tul_kw):
    kw = dict(prefix_k=2)
    if offset != 1:
        kw["spandec_target_offset"] = offset
    kw.update(tul_kw)
    torch.manual_seed(seed)
    m = MORPHTransformer(_tiny(tul=_tul(**kw)))
    with torch.no_grad():
        m.embed.bigram.lambdas.fill_(0.5)
    return m.train().float()


# ── the INDEPENDENT oracle ───────────────────────────────────────────────────

def _oracle(input_ids, layout, j_max: int, shift: int):
    """``(ids, valid)`` built by walking the layout in Python, per row, per span.

    It never calls `span_slots`. It collects each span's TOKEN positions in row order off
    `bag_id` and `slot_mask`, then hands span ``b``'s first ``j_max`` tokens to slot
    ``b - shift`` whenever both slots exist. That is the whole rule, written the other way
    round from the tensor implementation.
    """
    B, L = input_ids.shape
    S = layout.slot_index.shape[1]
    ids = torch.zeros(B, S, j_max, dtype=torch.long)
    ok = torch.zeros(B, S, j_max, dtype=torch.bool)
    for b in range(B):
        spans: dict[int, list[int]] = {}
        for p in range(L):
            if bool(layout.slot_mask[b, p]):
                continue
            k = int(layout.bag_id[b, p])
            if k >= S:                                 # the dump bin (unterminated tail)
                continue
            spans.setdefault(k, []).append(int(input_ids[b, p]))
        for k, toks in spans.items():
            tgt = k - shift
            if tgt < 0:
                continue
            if not (bool(layout.slot_valid[b, k]) and bool(layout.slot_valid[b, tgt])):
                continue
            for j, t in enumerate(toks[:j_max]):
                ids[b, tgt, j] = t
                ok[b, tgt, j] = True
    return ids, ok


def test_the_oracle_agrees_with_the_shipped_next_span_target():
    """The oracle is only worth something if it reproduces the target that already ships."""
    x, _y, layout = _batch()
    a_ids, a_ok = next_span_slots(x, layout, J)
    b_ids, b_ok = _oracle(x, layout, J, shift=1)
    assert torch.equal(a_ok, b_ok)
    assert torch.equal(a_ids * a_ok, b_ids * b_ok)


# ── 1. OFFSET 1 IS NOTHING ───────────────────────────────────────────────────

def test_offset_one_is_the_default_and_the_shipped_target():
    x, _y, layout = _batch()
    a = horizon_span_slots(x, layout, J, 1, start=1)
    b = next_span_slots(x, layout, J)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert TULConfig(prefix_k=2, slot_id=4).spandec_target_offset == 1


def test_setting_the_key_to_its_default_changes_nothing():
    a = _model(1)
    torch.manual_seed(99)
    b = MORPHTransformer(_tiny(tul=_tul(prefix_k=2, spandec_target_offset=1)))
    with torch.no_grad():
        b.embed.bigram.lambdas.fill_(0.5)
    b = b.train().float()
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    x, y, layout = _batch()
    torch.manual_seed(7)
    oa = a(x, labels=y, slot_layout=layout)
    torch.manual_seed(7)
    ob = b(x, labels=y, slot_layout=layout)
    assert repr(float(oa["loss"])) == repr(float(ob["loss"]))
    assert repr(float(oa["spandec_ce"])) == repr(float(ob["spandec_ce"]))


# ── 2/3/4. THE TARGETS, THE MASK, THE COUNT ──────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_targets_are_exactly_span_s_plus_k(k):
    x, _y, layout = _batch()
    got_ids, got_ok = span_slots(x, layout, J, shift=k)
    want_ids, want_ok = _oracle(x, layout, J, shift=k)
    assert torch.equal(got_ok, want_ok), f"validity disagrees at offset {k}"
    assert torch.equal(got_ids * got_ok, want_ids * want_ok), (
        f"the decoded token ids at offset {k} are not span s+{k}'s")


@pytest.mark.parametrize("k", [2, 3, 4])
def test_offset_k_is_not_offset_one(k):
    """A target that equalled the next span's would make every other test vacuous."""
    x, _y, layout = _batch()
    a_ids, a_ok = span_slots(x, layout, J, shift=1)
    b_ids, b_ok = span_slots(x, layout, J, shift=k)
    assert not torch.equal(a_ids * a_ok, b_ids * b_ok)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_the_last_k_minus_one_slots_have_no_target(k):
    x, _y, layout = _batch()
    _ids, ok = span_slots(x, layout, J, shift=k)
    B, S = layout.slot_valid.shape
    for b in range(B):
        n_valid = int(layout.slot_valid[b].sum())
        assert n_valid > k, "fixture too short to test the tail"
        # The last k-1 REAL slots have no span s+k in the row; nor does any pad slot.
        for s in range(n_valid - (k - 1), S):
            assert not bool(ok[b, s].any()), (
                f"slot {s} of row {b} is supervised at offset {k} but span {s + k} is not "
                f"in the row ({n_valid} real slots)")


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_the_supervised_token_count_matches_the_oracle(k):
    x, _y, layout = _batch()
    _gi, ok = span_slots(x, layout, J, shift=k)
    _wi, want = _oracle(x, layout, J, shift=k)
    assert int(ok.sum()) == int(want.sum())
    assert int(ok.sum()) > 0, "offset {k} supervises nothing on this fixture"


def test_the_supervised_count_falls_as_the_offset_grows():
    """Each extra span of offset removes one slot per row from the supervised set."""
    x, _y, layout = _batch()
    counts = [int(span_slots(x, layout, J, shift=k)[1].any(dim=2).sum()) for k in (1, 2, 3)]
    B = layout.slot_valid.shape[0]
    assert counts[1] == counts[0] - B, counts
    assert counts[2] == counts[1] - B, counts


# ── 5. HORIZON AND OFFSET COMPOSE ────────────────────────────────────────────

def test_horizon_two_at_offset_two_is_spans_s2_and_s3():
    x, _y, layout = _batch()
    cat_ids, cat_ok = horizon_span_slots(x, layout, J, horizon=2, start=2)
    assert cat_ids.shape[2] == 2 * J
    for i, shift in enumerate((2, 3)):
        sid, sok = span_slots(x, layout, J, shift=shift)
        sl = slice(i * J, (i + 1) * J)
        assert torch.equal(cat_ok[:, :, sl], sok)
        assert torch.equal(cat_ids[:, :, sl], sid)


def test_horizon_three_at_offset_one_is_unchanged_by_the_new_parameter():
    """`slot-spandec-strict-h3`'s target, built through the new signature."""
    x, _y, layout = _batch()
    a = horizon_span_slots(x, layout, J, 3)
    b = horizon_span_slots(x, layout, J, 3, start=1)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


# ── the model forward ────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [2, 3])
def test_the_forward_runs_at_offset_k_and_reports_it(k):
    m = _model(k)
    assert m.tul_spandec.target_offset == k
    x, y, layout = _batch()
    torch.manual_seed(7)
    out = m(x, labels=y, slot_layout=layout)
    assert float(out["spandec_target_offset"]) == float(k)
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all()
               for p in m.tul_spandec.parameters() if p.grad is not None)


def test_the_offset_changes_the_decoder_term_and_the_token_count():
    x, y, layout = _batch()
    outs = {}
    for k in (1, 2):
        m = _model(k)
        torch.manual_seed(7)
        outs[k] = m(x, labels=y, slot_layout=layout)
    assert float(outs[1]["spandec_ce"]) != float(outs[2]["spandec_ce"])
    assert float(outs[2]["spandec_n_tokens"]) < float(outs[1]["spandec_n_tokens"])


def test_the_token_path_is_unchanged_by_the_offset():
    """The offset moves the AUXILIARY target only. Nothing about the embeddings, the
    prelude, the loop, the coda or the head depends on it, so two models that differ by
    the offset alone produce BIT-IDENTICAL logits on the same forward.

    The logits, not `loss - spandec_weighted`: the total is a float32 sum and subtracting
    the auxiliary term back out rounds differently when the term's magnitude differs
    (measured 5.313729286 vs 5.313729763 on this fixture). That is float arithmetic, not
    the token path, and a test that could not tell the two apart would be worthless."""
    x, _y, layout = _batch()
    a, b = _model(1), _model(2)
    torch.manual_seed(7)
    with torch.no_grad():
        la = a(x, slot_layout=layout)["logits"]
    torch.manual_seed(7)
    with torch.no_grad():
        lb = b(x, slot_layout=layout)["logits"]
    assert torch.equal(la, lb)


# ── 6. THE REFUSALS ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [0, -1, -3])
def test_offset_below_one_raises(k):
    with pytest.raises(ValueError, match="spandec_target_offset must be >= 1"):
        _tul(spandec_target_offset=k)


@pytest.mark.parametrize("k", [0, -2])
def test_span_slots_refuses_a_negative_shift(k):
    x, _y, layout = _batch()
    if k < 0:
        with pytest.raises(ValueError, match="shift must be >= 0"):
            span_slots(x, layout, J, shift=k)
    else:
        span_slots(x, layout, J, shift=k)             # 0 IS the own span, and is legal


def test_offset_at_or_past_max_slots_raises():
    """k >= max_slots leaves every slot in every row unsupervised."""
    x, _y, layout = _batch(max_slots=6)
    assert layout.slot_index.shape[1] == 6
    span_slots(x, layout, J, shift=5)                 # the last legal one
    with pytest.raises(ValueError, match="slot budget"):
        span_slots(x, layout, J, shift=6)
    with pytest.raises(ValueError, match="slot budget"):
        span_slots(x, layout, J, shift=9)


def test_offset_without_the_decoder_raises():
    with pytest.raises(ValueError, match="spandec=false"):
        TULConfig(prefix_k=2, slot_id=4, spandec=False, spandec_target_offset=2)


@pytest.mark.parametrize("other", ["spandec_per_pass", "oracle_z", "coda_span_heads"])
def test_offset_with_a_mechanism_that_owns_the_next_span_raises(other):
    kw = {other: 4 if other == "coda_span_heads" else True}
    with pytest.raises(NotImplementedError, match=other):
        _tul(spandec_target_offset=2, **kw)


def test_those_mechanisms_still_build_at_offset_one():
    """The refusal above must be about the OFFSET, not about the mechanism."""
    _tul(coda_span_heads=4)
    _tul(oracle_z=True)


# ── the Hydra seam: `tul.spandec_target_offset` must REACH the TULConfig ─────
#
# `build_tul_runtime` resolves the tokenizer before it builds anything, and this suite
# must not depend on a network. The two fakes below replace exactly that step — the
# tokenizer object and the boundary rule — and leave every line this file is about
# untouched. Without them nothing tests the `tc.get(...)` wiring or the max_slots refusal,
# and sabotages R2-g and R2-h (the refusal disabled; the Hydra key read as a constant 1)
# both MISSED.

class _FakeTok:
    eos_token_id = 0

    def convert_tokens_to_ids(self, t):
        return {"<fim_pad>": 4}[t]


def _fake_setup(monkeypatch, max_slots: int = 10):
    import transformers

    from morph.training import tul_setup

    monkeypatch.setattr(transformers, "AutoTokenizer",
                        type("A", (), {"from_pretrained": staticmethod(
                            lambda *a, **k: _FakeTok())}))
    lut = _rule().is_boundary
    monkeypatch.setattr(tul_setup, "build_boundary_rule",
                        lambda cfg, cache_dir=None: (_rule(), lut, 0, ()))
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "model": {"vocab_size": V, "n_heads": 2, "mean_depth": 3, "max_depth": 4},
        "data": {"tokenizer": "fake/tok", "seq_len": 64},
        "tul": {"activate_at": 0.0, "prefix_k": 2, "max_slots": max_slots,
                "spandec": True, "spandec_max_tokens": J},
    })


def test_the_hydra_key_reaches_the_config_and_the_manifest(monkeypatch):
    from morph.training.tul_setup import build_tul_runtime
    cfg = _fake_setup(monkeypatch)
    cfg.tul.spandec_target_offset = 3
    rt = build_tul_runtime(cfg)
    assert rt.model_cfg.spandec_target_offset == 3
    assert rt.manifest["spandec_target_offset"] == 3
    assert rt.manifest["spandec_reads_cells"] is False


def test_the_default_still_reaches_the_config_as_one(monkeypatch):
    from morph.training.tul_setup import build_tul_runtime
    rt = build_tul_runtime(_fake_setup(monkeypatch))
    assert rt.model_cfg.spandec_target_offset == 1


def test_an_offset_at_or_past_the_derived_slot_budget_is_refused_at_config_time(monkeypatch):
    """The run must die at startup, not five minutes into a queue slot."""
    from morph.training.tul_setup import build_tul_runtime
    cfg = _fake_setup(monkeypatch, max_slots=8)
    cfg.tul.spandec_target_offset = 7
    build_tul_runtime(cfg)                       # the last legal one
    cfg.tul.spandec_target_offset = 8
    with pytest.raises(ValueError, match="slot budget"):
        build_tul_runtime(cfg)
    cfg.tul.spandec_target_offset = 40
    with pytest.raises(ValueError, match="slot budget"):
        build_tul_runtime(cfg)
